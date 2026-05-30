"""
Unsupervised legibility, take two: **isometry + coordinate parsimony**.

`iso_coord.py` showed isometry alone fails — arc-length is necessary but not
sufficient, because a 3-D isometric coordinate can still *wind* through coordinate
space (linear R² needs the label to be ~linear in the coords).  The fix proposed in
that file's verdict: also force the coordinate to use **as few dimensions as the
manifold needs**, so the legible axis is low-dimensional and straight.

This script adds a **parsimony** prior to the isometry one, both still label-free:

  * isometry (from iso_coord): pin each chart decoder's directional speed
    ``||J_m u||`` to a fixed target -> constant-speed -> arc-length coordinate.
  * parsimony (new): minimize the **participation ratio** of the coordinate's
    per-dimension variance, ``PR = (Σ v_i)^2 / Σ v_i^2 ∈ [1, cd]`` — a
    *scale-invariant* measure of how many coordinate dims are active.  Driving PR
    toward 1 collapses unused dims; reconstruction keeps the needed ones alive.
    Scale-invariance matters: an L1-on-std penalty can be gamed by shrinking the
    coordinate and growing the decoder, PR cannot.

We grid over (lam_iso, lam_pars) so the marginal effect of parsimony is visible
(lam_iso=0 row = parsimony alone; lam_pars=0 row = isometry alone).  Held-out label
R² (cyclic-aware) is a pure generalization test — no labels in training.  We also
report the held-out coordinate's effective active-dim count (PR) as a diagnostic.

Run (CPU, from repo root):
  SAE_DEVICE=cpu SAE_D_MODEL=576 uv run python prototype/iso_parsimony.py \
      --seeds 0 1 2 --lam-isos 0 1 --lam-pars 0 2 8
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from data import CACHE_DIR
from factored_sae import FactoredSAE
from fair_comparison import (load_split, factored_eval, label_score, _ms,
                             _pca_basis, _subspace_rep)
from legible_coord import _mixture_arrays

RESULTS_DIR = CACHE_DIR / "iso_parsimony"
DEFAULT_MANIFOLDS = ["years", "age", "temperature", "colors", "geography"]
CYCLIC = {"colors"}


def _participation_ratio(v, axis=-1, eps=1e-8):
    """(Σ v_i)^2 / Σ v_i^2 — effective number of active dims, in [1, cd]."""
    return v.sum(axis) ** 2 / ((v ** 2).sum(axis) + eps)


def train_factored_iso_pars(Xtr_mix, coord_dim, lam_iso, lam_pars, seed,
                            n_charts=12, epochs=120, batch=512, lr=2e-3,
                            lam_sparse=0.02, lam_balance=0.3, fd_eps=1e-2,
                            iso_target=4.0):
    """Nonlinear factored SAE + label-free isometry + coordinate-parsimony priors."""
    torch.manual_seed(seed); np.random.seed(seed)
    from data import DEVICE
    mean = Xtr_mix.mean(0, keepdims=True)
    std = float(Xtr_mix.std()) + 1e-6
    Xn = torch.from_numpy((Xtr_mix - mean) / std).to(DEVICE)
    N, d_in = Xn.shape
    # hard_routing=True: model is trained as a true atlas so factored_eval's
    # dominant-chart restriction matches the training regime.  See the
    # correction block in [[goodfire-sae-manifold-repro]] memory.
    model = FactoredSAE(d_in, n_charts, coord_dim, linear_charts=False,
                        hard_routing=True).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    eps = 1e-9

    for ep in range(epochs):
        perm = torch.randperm(N, device=DEVICE)
        for i in range(0, N, batch):
            xb = Xn[perm[i:i + batch]]
            recon, a, coords = model(xb)          # a is one-hot (hard routing)
            mse = ((recon - xb) ** 2).sum(-1).mean()
            # Entropy regularizers need the *soft* routing — one-hot has 0
            # entropy and no gradient signal for the router params.
            a_soft = F.softmax(model.router(xb), dim=-1)
            ent_point = -(a_soft * (a_soft + eps).log()).sum(-1).mean()
            usage_soft = a_soft.mean(0)
            ent_usage = -(usage_soft * (usage_soft + eps).log()).sum()
            loss = mse + lam_sparse * ent_point - lam_balance * ent_usage

            # Iso/parsimony weighting stays on the *hard* assignment: each
            # point contributes to exactly the chart it was routed to (which
            # matches the atlas semantics; soft weights would smear the
            # per-chart variance/speed across charts the point isn't on).
            w = a.detach()                                   # [B, M] one-hot
            wsum = w.sum(0) + 1e-6                            # [M]
            usage = a.mean(0)                                 # [M], hard-mean

            if lam_iso > 0:
                # Isometry: fixed-target directional speed of each chart decoder.
                u = torch.randn(coord_dim, device=DEVICE); u = u / (u.norm() + 1e-8)
                r0 = torch.stack([model.charts[m](coords[:, m])
                                  for m in range(n_charts)], dim=1)
                rp = torch.stack([model.charts[m](coords[:, m] + fd_eps * u)
                                  for m in range(n_charts)], dim=1)
                speed = ((rp - r0) / fd_eps).norm(dim=-1)    # [B, M]
                iso = (w * (speed - iso_target) ** 2).sum() / wsum.sum()
                loss = loss + lam_iso * iso

            if lam_pars > 0:
                # Parsimony: minimize participation ratio of per-dim coord variance,
                # per chart, weighted by chart usage (scale-invariant).
                mean_c = (w.unsqueeze(-1) * coords).sum(0) / wsum.unsqueeze(-1)  # [M,cd]
                var_c = (w.unsqueeze(-1) * (coords - mean_c) ** 2).sum(0) \
                    / wsum.unsqueeze(-1)                                          # [M,cd]
                pr = _participation_ratio(var_c, axis=-1)                         # [M]
                pars = (usage.detach() * pr).sum() / (usage.sum() + 1e-6)
                loss = loss + lam_pars * pars

            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    model.cpu()
    return model, (mean.astype(np.float32), std)


def run(manifolds, seeds, lam_isos, lam_pars_list, coord_dim, headline_N):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    configs = [(li, lp) for li in lam_isos for lp in lam_pars_list]
    ve, r2lin, r2knn, pr_act, pca_r2 = {}, {}, {}, {}, {}

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

        for (li, lp) in configs:
            print(f"  training: lam_iso={li}, lam_pars={lp}...")
            model, norm = train_factored_iso_pars(Xtr_mix, coord_dim, li, lp, seed)
            for name in names:
                d = per[name]
                fve, Ztr, Zte, _ = factored_eval(
                    model, norm, d["Xtr"], d["Xte"], d["mean_m"])
                lin, knn = label_score(Ztr, d["ytr"], Zte, d["yte"], name)
                pr = float(_participation_ratio(np.asarray(Zte).var(0)))
                add(ve, (li, lp, name), fve)
                add(r2lin, (li, lp, name), lin)
                add(r2knn, (li, lp, name), knn)
                add(pr_act, (li, lp, name), pr)

    _report(manifolds, configs, ve, r2lin, pr_act, pca_r2, coord_dim, headline_N, seeds)
    _save(configs, ve, r2lin, r2knn, pr_act, pca_r2, seeds, coord_dim, headline_N)
    _plot(manifolds, configs, ve, r2lin, pca_r2, coord_dim)


def _report(manifolds, configs, ve, r2lin, pr_act, pca_r2, cd, N, seeds):
    names = [m for m in manifolds if (configs[0][0], configs[0][1], m) in ve]
    legible = [m for m in names if m not in CYCLIC]

    print(f"\n{'='*88}\nISOMETRY + PARSIMONY  (nonlinear, coord_dim={cd}, N={N}, "
          f"{len(seeds)} seeds; NO labels in training)\n{'='*88}")
    print("PCA reference label R²:  " +
          "  ".join(f"{m}={_ms(pca_r2[m])[0]:.2f}" for m in names))

    def block(title, store, fmt="{:.2f}", extra=None):
        print(f"\n{title}:")
        print(f"{'iso':>5}{'pars':>6}" + "".join(f"{m[:9]:>11}" for m in names)
              + (f"{extra:>9}" if extra else ""))
        for (li, lp) in configs:
            vals = [_ms(store[(li, lp, m)])[0] for m in names]
            row = f"{li:>5g}{lp:>6g}" + "".join(f"{fmt.format(v):>11}" for v in vals)
            if extra == "mean*":
                lm = np.mean([_ms(store[(li, lp, m)])[0] for m in legible])
                row += f"{lm:>9.2f}"
            elif extra == "mean":
                row += f"{np.mean(vals):>9.2f}"
            print(row)

    block(f"Held-out VE@{N}", ve, extra="mean")
    block("Held-out label R² (colors cyclic)", r2lin, extra="mean*")
    block("Effective active coord dims (PR of held-out coords, lower=fewer)", pr_act)
    print("  (mean* excludes cyclic; iso=0 row = parsimony alone, "
          "pars=0 row = isometry alone)")


def _save(configs, ve, r2lin, r2knn, pr_act, pca_r2, seeds, cd, N):
    key = lambda k: f"{k[0]}|{k[1]}|{k[2]}"
    out = dict(seeds=list(seeds), coord_dim=cd, headline_N=N,
               configs=[list(c) for c in configs],
               ve={key(k): v for k, v in ve.items()},
               r2lin={key(k): v for k, v in r2lin.items()},
               r2knn={key(k): v for k, v in r2knn.items()},
               pr_act={key(k): v for k, v in pr_act.items()},
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
                ax.annotate(f"p{c[1]:g}", (x, y), fontsize=6,
                            xytext=(3, 3), textcoords="offset points")
        pr = _ms(pca_r2[m])[0]
        ax.axvline(pr, color="k", ls="--", lw=1, label=f"PCA={pr:.2f}")
        cyc = " (cyclic)" if m in CYCLIC else ""
        ax.set_title(m + cyc); ax.set_xlabel("held-out label R²")
        ax.grid(alpha=0.3); ax.legend(fontsize=6, loc="lower left")
    axes[0][0].set_ylabel(f"held-out VE@{cd}")
    fig.suptitle("Isometry + parsimony (label-free): fidelity vs legibility "
                 "(p = lam_pars)")
    fig.tight_layout()
    path = RESULTS_DIR / "pareto.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"Saved plot -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifolds", nargs="+", default=DEFAULT_MANIFOLDS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--lam-isos", nargs="+", type=float, default=[0, 1])
    ap.add_argument("--lam-pars", nargs="+", type=float, default=[0, 2, 8])
    ap.add_argument("--coord-dim", type=int, default=3)
    ap.add_argument("--headline-n", type=int, default=3)
    a = ap.parse_args()
    run(a.manifolds, a.seeds, a.lam_isos, a.lam_pars, a.coord_dim, a.headline_n)


if __name__ == "__main__":
    main()
