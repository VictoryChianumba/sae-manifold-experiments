"""
Unsupervised legibility, take three: **intrinsic-dimension-adaptive parsimony**.

`iso_parsimony.py` showed that a label-free participation-ratio (PR) parsimony
prior recovers legibility — but only on genuinely low-D manifolds.  A *single
global* parsimony weight has one target (drive PR -> 1, i.e. one active coord dim)
that it applies to every chart, so it over-collapses the multi-D manifolds:
years 0.16->0.70 and temperature 0.79->0.89 (good), but geography 0.41->0.01 and
colors ->0.07 (destroyed).  You cannot force a sphere onto one straight axis.

The fix proposed in that file's verdict: let **each chart learn how many coordinate
dimensions it needs**.  Instead of a global PR target, we *price* each coordinate
dimension at a fixed cost and let **reconstruction** decide how many to keep:

  * per-chart, per-dim learned **gate**  g_{m,j} in (0,1) (sigmoid of a parameter).
    The coordinate fed to chart m's decoder is the gated coordinate g_m * z_m, so a
    closed gate (g->0) removes that dimension entirely.
  * a fixed **per-dimension cost** (L1 on the gates), so a chart keeps a dim *only
    if* reconstruction pays for it.  Mild Matryoshka ordering (cost weight j+1) packs
    variance into earlier dims, so dim 0 becomes "the" axis.

The key idea that makes this *adaptive*: a **nonlinear** chart needs coordinate dims
equal to the manifold's **intrinsic** dimension, not its embedding dimension — the
MLP supplies the curvature.  So reconstruction should pay to keep ~1 dim for the
years-helix / temperature-line and ~2 for the geography sphere / colors loop.  The
active-dim count then self-adapts per chart, which a global knob cannot do.

Anti-gaming (the reason PR was scale-invariant): an L1 on raw coordinate std can be
gamed by shrinking the coordinate and growing the decoder.  We instead normalize
each coordinate dim by a running-std buffer **before** gating, so scale is removed
and only the gate g controls a dim's contribution — the L1 on gates cannot be dodged
by rescaling.

We grid over (lam_iso, lam_gate) so the marginal effect of adaptive parsimony is
visible (lam_iso=0 row = adaptive parsimony alone; lam_gate=0 row = isometry alone).
Held-out label R^2 (cyclic-aware) is a pure generalization test — no labels in
training.  We report each manifold's dominant chart's effective active-dim count
(sum of its gates) as the diagnostic that replaces PR.

Run (CPU, from repo root):
  SAE_DEVICE=cpu SAE_D_MODEL=576 uv run python prototype/adaptive_parsimony.py \
      --seeds 0 1 2 --lam-isos 0 1 --lam-gates 0 2 4 8
  # ablation isolating the coordinate-normalization effect from the gate penalty:
  SAE_DEVICE=cpu SAE_D_MODEL=576 uv run python prototype/adaptive_parsimony.py \
      --seeds 0 1 2 --lam-isos 0 --lam-gates 0 --no-coord-norm

What we found (3 seeds; honest, qualified-negative):
  * The explicit goal is met *narrowly*: adaptive per-dim gates do NOT over-collapse
    the multi-D manifolds the way the global PR knob did (geography stays ~0.30-0.45
    vs PR's 0.08; colors ~0.14-0.23 vs 0.07).  But the gate penalty is nearly inert
    as a legibility lever — gates close roughly *uniformly* (~2.67 -> ~1.3 active dims
    for every manifold, no clean intrinsic-dim differentiation), held-out legibility
    is flat in gate strength, and VE erodes mildly.  It avoids harm rather than
    delivering the targeted differential dim allocation.
  * The one genuinely new lever is incidental and fully isolated by --no-coord-norm:
    per-dim coordinate normalization lifts years legibility 0.28 -> 0.64 but TRADES
    reconstruction for it (years VE 0.79 -> 0.45) — a Pareto move on the
    fidelity<->legibility frontier, not a free win, and it helps the curved manifold
    (years) while slightly hurting the near-linear one (age).
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import CACHE_DIR
from factored_sae import FactoredSAE
from fair_comparison import (load_split, factored_eval, label_score, _ms,
                             _pca_basis, _subspace_rep)
from legible_coord import _mixture_arrays

RESULTS_DIR = CACHE_DIR / "adaptive_parsimony"
DEFAULT_MANIFOLDS = ["years", "age", "temperature", "colors", "geography"]
CYCLIC = {"colors"}


class GatedFactoredSAE(FactoredSAE):
    """FactoredSAE whose per-chart coordinate passes through learned per-dim gates.

    Each coordinate dim is first normalized by a running-std buffer (so scale is
    removed and the gate cost cannot be gamed by rescaling), then multiplied by a
    learned gate g_{m,j} in (0,1).  An L1 cost on the gates prices each active
    dimension; reconstruction keeps a dim open only if it pays for it -> the active
    coord-dim count self-adapts per chart to the manifold's intrinsic dimension.
    """

    def __init__(self, d_in, n_charts=12, coord_dim=3, gate_init=2.0,
                 coord_norm=True, hard_routing=False):
        super().__init__(d_in, n_charts, coord_dim, linear_charts=False,
                         hard_routing=hard_routing)
        # gate_init=2.0 -> sigmoid ~0.88: charts start near-full and *prune* down.
        self.coord_norm = coord_norm
        self.gate_logits = nn.Parameter(
            torch.full((n_charts, coord_dim), float(gate_init)))
        self.register_buffer("coord_std", torch.ones(n_charts, coord_dim))

    def gates(self):
        return torch.sigmoid(self.gate_logits)            # [M, cd] in (0,1)

    def forward(self, x, temp=1.0):
        # Same straight-through one-hot routing as FactoredSAE.forward, but the
        # decoder chain runs on gated coords so we duplicate rather than call
        # super().forward (the gating sits between coord_enc and charts[m]).
        a_soft = F.softmax(self.router(x) / temp, dim=-1)             # [B, M]
        if self.hard_routing:
            a_hard = F.one_hot(a_soft.argmax(-1), self.n_charts).to(a_soft.dtype)
            a = a_hard - a_soft.detach() + a_soft
        else:
            a = a_soft
        coords = self.coord_enc(x).view(-1, self.n_charts, self.coord_dim)
        if self.coord_norm:
            if self.training:
                with torch.no_grad():                      # batchnorm-style EMA
                    self.coord_std.mul_(0.9).add_(0.1 * coords.std(0))
            std = self.coord_std.clamp_min(1e-6)           # [M, cd]
        else:
            std = 1.0
        gcoords = self.gates() * coords / std              # broadcast [B,M,cd]
        recons = torch.stack([self.charts[m](gcoords[:, m])
                              for m in range(self.n_charts)], dim=1)
        recon = (a.unsqueeze(-1) * recons).sum(1) + self.bias
        return recon, a, gcoords                           # report GATED coords


def train_factored_adaptive(Xtr_mix, coord_dim, lam_iso, lam_gate, seed,
                            n_charts=12, epochs=120, batch=512, lr=2e-3,
                            lam_sparse=0.02, lam_balance=0.3, fd_eps=1e-2,
                            iso_target=4.0, coord_norm=True):
    """Gated nonlinear factored SAE + label-free isometry + adaptive parsimony."""
    torch.manual_seed(seed); np.random.seed(seed)
    from data import DEVICE
    mean = Xtr_mix.mean(0, keepdims=True)
    std = float(Xtr_mix.std()) + 1e-6
    Xn = torch.from_numpy((Xtr_mix - mean) / std).to(DEVICE)
    N, d_in = Xn.shape
    # hard_routing=True so the model trains as a true atlas (single-chart
    # recon per point) and factored_eval's dominant-chart restriction matches.
    model = GatedFactoredSAE(d_in, n_charts, coord_dim, coord_norm=coord_norm,
                             hard_routing=True).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    eps = 1e-9
    # Mild Matryoshka: later dims cost more, so variance packs into dim 0 first.
    dim_cost = torch.arange(1, coord_dim + 1, dtype=torch.float32, device=DEVICE)

    model.train()
    for ep in range(epochs):
        perm = torch.randperm(N, device=DEVICE)
        for i in range(0, N, batch):
            xb = Xn[perm[i:i + batch]]
            recon, a, gcoords = model(xb)         # a is one-hot (hard routing)
            mse = ((recon - xb) ** 2).sum(-1).mean()
            # Entropy regularizers need soft routing for the gradient.
            a_soft = F.softmax(model.router(xb), dim=-1)
            ent_point = -(a_soft * (a_soft + eps).log()).sum(-1).mean()
            usage_soft = a_soft.mean(0)
            ent_usage = -(usage_soft * (usage_soft + eps).log()).sum()
            loss = mse + lam_sparse * ent_point - lam_balance * ent_usage

            # Iso/parsimony weighting on the hard assignment (atlas semantics).
            w = a.detach()                                   # [B, M] one-hot
            wsum = w.sum(0) + 1e-6                            # [M]
            usage = a.mean(0)                                 # [M] for pars

            if lam_iso > 0:
                # Isometry: fixed-target directional speed of each chart decoder,
                # measured in the *gated* coordinate it actually uses.
                u = torch.randn(coord_dim, device=DEVICE); u = u / (u.norm() + 1e-8)
                g = model.gates() * u / model.coord_std.clamp_min(1e-6)  # [M,cd]
                r0 = torch.stack([model.charts[m](gcoords[:, m])
                                  for m in range(n_charts)], dim=1)
                rp = torch.stack([model.charts[m](gcoords[:, m] + fd_eps * g[m])
                                  for m in range(n_charts)], dim=1)
                speed = ((rp - r0) / fd_eps).norm(dim=-1)    # [B, M]
                iso = (w * (speed - iso_target) ** 2).sum() / wsum.sum()
                loss = loss + lam_iso * iso

            if lam_gate > 0:
                # Adaptive parsimony: usage-weighted, dim-ordered L1 on the gates.
                g = model.gates()                            # [M, cd]
                cost = (g * dim_cost).sum(-1)                # [M]
                pars = (usage.detach() * cost).sum() / (usage.sum() + 1e-6)
                loss = loss + lam_gate * pars

            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    model.cpu()
    return model, (mean.astype(np.float32), std)


def run(manifolds, seeds, lam_isos, lam_gates, coord_dim, headline_N,
        coord_norm=True):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    configs = [(li, lg) for li in lam_isos for lg in lam_gates]
    ve, r2lin, r2knn, gate_act, pca_r2 = {}, {}, {}, {}, {}

    def add(store, key, val):
        store.setdefault(key, []).append(val)

    for seed in seeds:
        print(f"\n===== seed {seed} =====")
        per, _ = load_split(manifolds, seed)
        Xtr_mix, _, _, names = _mixture_arrays(per)

        for name in names:
            d = per[name]
            B = _pca_basis(d["Xtr"], d["mean_m"], headline_N)
            Ztr, Zte = _subspace_rep(d["Xtr"], d["Xte"], d["mean_m"], B)
            add(pca_r2, name, label_score(Ztr, d["ytr"], Zte, d["yte"], name)[0])

        for (li, lg) in configs:
            print(f"  training: lam_iso={li}, lam_gate={lg}...")
            model, norm = train_factored_adaptive(Xtr_mix, coord_dim, li, lg, seed,
                                                  coord_norm=coord_norm)
            gates = model.gates().detach().numpy()           # [M, cd]
            for name in names:
                d = per[name]
                fve, Ztr, Zte, dom = factored_eval(
                    model, norm, d["Xtr"], d["Xte"], d["mean_m"])
                lin, knn = label_score(Ztr, d["ytr"], Zte, d["yte"], name)
                add(ve, (li, lg, name), fve)
                add(r2lin, (li, lg, name), lin)
                add(r2knn, (li, lg, name), knn)
                add(gate_act, (li, lg, name), float(gates[dom].sum()))

    _report(manifolds, configs, ve, r2lin, gate_act, pca_r2, coord_dim, headline_N, seeds)
    _save(configs, ve, r2lin, r2knn, gate_act, pca_r2, seeds, coord_dim, headline_N)
    _plot(manifolds, configs, ve, r2lin, pca_r2, coord_dim)


def _report(manifolds, configs, ve, r2lin, gate_act, pca_r2, cd, N, seeds):
    names = [m for m in manifolds if (configs[0][0], configs[0][1], m) in ve]
    legible = [m for m in names if m not in CYCLIC]

    print(f"\n{'='*88}\nADAPTIVE PARSIMONY (gated, nonlinear, coord_dim={cd}, N={N}, "
          f"{len(seeds)} seeds; NO labels in training)\n{'='*88}")
    print("PCA reference label R²:  " +
          "  ".join(f"{m}={_ms(pca_r2[m])[0]:.2f}" for m in names))

    def block(title, store, fmt="{:.2f}", extra=None):
        print(f"\n{title}:")
        print(f"{'iso':>5}{'gate':>6}" + "".join(f"{m[:9]:>11}" for m in names)
              + (f"{extra:>9}" if extra else ""))
        for (li, lg) in configs:
            vals = [_ms(store[(li, lg, m)])[0] for m in names]
            row = f"{li:>5g}{lg:>6g}" + "".join(f"{fmt.format(v):>11}" for v in vals)
            if extra == "mean*":
                lm = np.mean([_ms(store[(li, lg, m)])[0] for m in legible])
                row += f"{lm:>9.2f}"
            elif extra == "mean":
                row += f"{np.mean(vals):>9.2f}"
            print(row)

    block(f"Held-out VE@{N}", ve, extra="mean")
    block("Held-out label R² (colors cyclic)", r2lin, extra="mean*")
    block("Active coord dims (Σ gates of dominant chart, lower=fewer)", gate_act)
    print("  (mean* excludes cyclic; iso=0 row = adaptive parsimony alone, "
          "gate=0 row = isometry alone)")


def _save(configs, ve, r2lin, r2knn, gate_act, pca_r2, seeds, cd, N):
    key = lambda k: f"{k[0]}|{k[1]}|{k[2]}"
    out = dict(seeds=list(seeds), coord_dim=cd, headline_N=N,
               configs=[list(c) for c in configs],
               ve={key(k): v for k, v in ve.items()},
               r2lin={key(k): v for k, v in r2lin.items()},
               r2knn={key(k): v for k, v in r2knn.items()},
               gate_act={key(k): v for k, v in gate_act.items()},
               pca_r2=pca_r2)
    path = RESULTS_DIR / "results.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved raw results -> {path}")


def _plot(manifolds, configs, ve, r2lin, pca_r2, cd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [m for m in manifolds if (configs[0][0], configs[0][1], m) in ve]
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.6),
                             squeeze=False)
    isos = sorted(set(c[0] for c in configs))
    colors = {iso: c for iso, c in zip(isos, ["tab:gray", "tab:purple",
                                              "tab:red", "tab:green"])}
    for ax, m in zip(axes[0], names):
        for iso in isos:
            cfgs = [c for c in configs if c[0] == iso]
            xs = [_ms(r2lin[(c[0], c[1], m)])[0] for c in cfgs]
            ys = [_ms(ve[(c[0], c[1], m)])[0] for c in cfgs]
            ax.plot(xs, ys, "-o", color=colors[iso], label=f"iso={iso:g}", zorder=3)
            for c, x, y in zip(cfgs, xs, ys):
                ax.annotate(f"g{c[1]:g}", (x, y), fontsize=6,
                            xytext=(3, 3), textcoords="offset points")
        pr = _ms(pca_r2[m])[0]
        ax.axvline(pr, color="k", ls="--", lw=1, label=f"PCA={pr:.2f}")
        cyc = " (cyclic)" if m in CYCLIC else ""
        ax.set_title(m + cyc); ax.set_xlabel("held-out label R²")
        ax.grid(alpha=0.3); ax.legend(fontsize=6, loc="lower left")
    axes[0][0].set_ylabel(f"held-out VE@{cd}")
    fig.suptitle("Adaptive parsimony (label-free, gated): fidelity vs legibility "
                 "(g = lam_gate)")
    fig.tight_layout()
    path = RESULTS_DIR / "pareto.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"Saved plot -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifolds", nargs="+", default=DEFAULT_MANIFOLDS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--lam-isos", nargs="+", type=float, default=[0, 1])
    ap.add_argument("--lam-gates", nargs="+", type=float, default=[0, 2, 8])
    ap.add_argument("--coord-dim", type=int, default=3)
    ap.add_argument("--headline-n", type=int, default=3)
    ap.add_argument("--no-coord-norm", dest="coord_norm", action="store_false",
                    help="ablation: disable per-dim coordinate normalization")
    a = ap.parse_args()
    run(a.manifolds, a.seeds, a.lam_isos, a.lam_gates, a.coord_dim, a.headline_n,
        coord_norm=a.coord_norm)


if __name__ == "__main__":
    main()
