"""
Unsupervised legibility: can an **isometry** prior make the chart coordinate
legible *without* using any labels in training?

`legible_coord.py` earned a legible coordinate by aligning it to the label (weak
supervision).  This script drops that crutch: it adds a label-free
**isometry / arc-length** penalty to the factored-SAE training and asks whether
the coordinate becomes legible on its own.

The idea (cf. Gropp et al., "Isometric Autoencoders", 2020): make each chart's
decoder `g_m: R^cd -> R^d` a near-isometry — equal steps in the coordinate map to
equal-length steps along the manifold in activation space.  Then the coordinate is
an **arc-length** parameterization, and for manifolds whose prompts are sampled
~uniformly in the underlying factor (age 1..99, year 1800.., temp -30..) arc length
is ~affine in the label, so a plain linear probe decodes it.  Crucially, *no label
is used* — the held-out label R^2 (cyclic-aware via `label_score`) is pure
generalization test, not a fit target.

Penalty (per chart, router-weighted, scale-free): along a shared random unit
direction `u` in coordinate space, the decoder's directional speed `||J_m u||`
should be *constant across points* — we penalize its within-chart variance.  `u`
is resampled each step, so over training this enforces `J_m^T J_m ~ c_m^2 I`
(isometry up to a per-chart scale).  Speed is estimated by finite difference.

Run (CPU, from repo root):
  SAE_DEVICE=cpu SAE_D_MODEL=576 uv run python prototype/iso_coord.py \
      --seeds 0 1 2 --lams 0 3 10 30 100
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from data import CACHE_DIR
from factored_sae import FactoredSAE
from fair_comparison import (load_split, factored_eval, label_score, _ms,
                             _pca_basis, _subspace_rep, CYCLIC_LABEL)
from legible_coord import _mixture_arrays

RESULTS_DIR = CACHE_DIR / "iso_coord"
DEFAULT_MANIFOLDS = ["years", "age", "temperature", "colors", "geography"]
CYCLIC = {"colors"}


def train_factored_iso(Xtr_mix, coord_dim, lam_iso, seed, n_charts=12,
                       epochs=120, batch=512, lr=2e-3, lam_sparse=0.02,
                       lam_balance=0.3, fd_eps=1e-2, iso_target=4.0):
    """Nonlinear factored SAE + label-free isometry penalty on chart decoders.

    The isometry term makes each chart's decoder constant-speed (arc-length
    coordinate); it uses no labels.  `lam_iso=0` is the unsupervised baseline.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    mean = Xtr_mix.mean(0, keepdims=True)
    std = float(Xtr_mix.std()) + 1e-6
    Xn = torch.from_numpy((Xtr_mix - mean) / std)
    N, d_in = Xn.shape
    model = FactoredSAE(d_in, n_charts, coord_dim, linear_charts=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    eps = 1e-9

    for ep in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, batch):
            xb = Xn[perm[i:i + batch]]
            recon, a, coords = model(xb)          # coords [B, M, cd], a [B, M]
            mse = ((recon - xb) ** 2).sum(-1).mean()
            ent_point = -(a * (a + eps).log()).sum(-1).mean()
            usage = a.mean(0)
            ent_usage = -(usage * (usage + eps).log()).sum()
            loss = mse + lam_sparse * ent_point - lam_balance * ent_usage

            if lam_iso > 0:
                # Directional speed ||J_m u|| of each chart decoder along a shared
                # random unit u, by finite difference; pin it to a FIXED unit
                # target (Gropp et al.): ||J u|| = iso_target for every point and
                # direction => isometry.  A fixed (not free) target is what stops
                # the trivial collapse z->const that a variance-only penalty allows.
                u = torch.randn(coord_dim); u = u / (u.norm() + 1e-8)
                r0 = torch.stack([model.charts[m](coords[:, m])
                                  for m in range(n_charts)], dim=1)
                rp = torch.stack([model.charts[m](coords[:, m] + fd_eps * u)
                                  for m in range(n_charts)], dim=1)
                speed = ((rp - r0) / fd_eps).norm(dim=-1)        # [B, M]
                w = a.detach()
                iso = (w * (speed - iso_target) ** 2).sum() / (w.sum() + 1e-6)
                loss = loss + lam_iso * iso

            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model, (mean.astype(np.float32), std)


def run(manifolds, seeds, lams, coord_dim, headline_N):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ve, r2lin, r2knn, pca_r2 = {}, {}, {}, {}

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

        for lam in lams:
            print(f"  training nonlinear factored, lam_iso={lam}...")
            model, norm = train_factored_iso(Xtr_mix, coord_dim, lam, seed)
            for name in names:
                d = per[name]
                fve, Ztr, Zte, _ = factored_eval(
                    model, norm, d["Xtr"], d["Xte"], d["mean_m"])
                lin, knn = label_score(Ztr, d["ytr"], Zte, d["yte"], name)
                add(ve, (lam, name), fve)
                add(r2lin, (lam, name), lin)
                add(r2knn, (lam, name), knn)

    _report(manifolds, lams, ve, r2lin, r2knn, pca_r2, coord_dim, headline_N, seeds)
    _save(lams, ve, r2lin, r2knn, pca_r2, seeds, coord_dim, headline_N)
    _plot(manifolds, lams, ve, r2lin, pca_r2, coord_dim)


def _report(manifolds, lams, ve, r2lin, r2knn, pca_r2, cd, N, seeds):
    names = [m for m in manifolds if (lams[0], m) in ve]
    legible = [m for m in names if m not in CYCLIC]

    print(f"\n{'='*84}\nUNSUPERVISED ISOMETRY SWEEP  (nonlinear, coord_dim={cd}, "
          f"N={N}, {len(seeds)} seeds; NO labels in training)\n{'='*84}")
    print("PCA reference label R²:  " +
          "  ".join(f"{m}={_ms(pca_r2[m])[0]:.2f}" for m in names))

    print(f"\nHeld-out VE@{N} by lam_iso:")
    print(f"{'lam':>6}" + "".join(f"{m[:9]:>11}" for m in names) + f"{'mean':>9}")
    for lam in lams:
        means = [_ms(ve[(lam, m)])[0] for m in names]
        print(f"{lam:>6}" + "".join(f"{v:>11.2f}" for v in means)
              + f"{np.mean(means):>9.2f}")

    print(f"\nHeld-out label R² (linear; colors cyclic) by lam_iso  "
          f"— label-free training:")
    print(f"{'lam':>6}" + "".join(f"{m[:9]:>11}" for m in names) + f"{'mean*':>9}")
    for lam in lams:
        row = [_ms(r2lin[(lam, m)])[0] for m in names]
        leg_mean = np.mean([_ms(r2lin[(lam, m)])[0] for m in legible])
        print(f"{lam:>6}" + "".join(f"{v:>11.2f}" for v in row)
              + f"{leg_mean:>9.2f}")
    print(f"  (mean* = mean over non-cyclic labeled manifolds, excludes {sorted(CYCLIC)})")


def _save(lams, ve, r2lin, r2knn, pca_r2, seeds, cd, N):
    out = dict(seeds=list(seeds), coord_dim=cd, headline_N=N, lams=list(lams),
               ve={f"{k[0]}|{k[1]}": v for k, v in ve.items()},
               r2lin={f"{k[0]}|{k[1]}": v for k, v in r2lin.items()},
               r2knn={f"{k[0]}|{k[1]}": v for k, v in r2knn.items()},
               pca_r2=pca_r2)
    path = RESULTS_DIR / "results.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved raw results -> {path}")


def _plot(manifolds, lams, ve, r2lin, pca_r2, cd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [m for m in manifolds if (lams[0], m) in ve]
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.6),
                             squeeze=False)
    for ax, m in zip(axes[0], names):
        xs = [_ms(r2lin[(lam, m)])[0] for lam in lams]
        ys = [_ms(ve[(lam, m)])[0] for lam in lams]
        ax.plot(xs, ys, "-o", color="tab:purple", zorder=3)
        for lam, x, y in zip(lams, xs, ys):
            ax.annotate(f"{lam:g}", (x, y), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        pr = _ms(pca_r2[m])[0]
        ax.axvline(pr, color="gray", ls="--", lw=1, label=f"PCA R²={pr:.2f}")
        cyc = " (cyclic)" if m in CYCLIC else ""
        ax.set_title(m + cyc); ax.set_xlabel("held-out label R²")
        ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower left")
    axes[0][0].set_ylabel(f"held-out VE@{cd}")
    fig.suptitle("Unsupervised isometry: fidelity vs legibility as iso weight "
                 "increases (labels = lam_iso; NO labels in training)")
    fig.tight_layout()
    path = RESULTS_DIR / "pareto.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"Saved plot -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifolds", nargs="+", default=DEFAULT_MANIFOLDS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--lams", nargs="+", type=float, default=[0, 3, 10, 30, 100])
    ap.add_argument("--coord-dim", type=int, default=3)
    ap.add_argument("--headline-n", type=int, default=3)
    a = ap.parse_args()
    run(a.manifolds, a.seeds, a.lams, a.coord_dim, a.headline_n)


if __name__ == "__main__":
    main()
