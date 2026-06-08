"""
Can a chart coordinate be BOTH reconstructive and legible?  (Follow-up to the
fair-comparison verdict in ./README.md: the nonlinear charts beat the PCA linear
ceiling on held-out reconstruction, but their coordinate decoded the manifold's
label *worse* than PCA — a Pareto win on geometry fidelity but not on
interpretability.)

This script adds a weak **label-alignment** term to the factored-SAE training: a
per-manifold probe on the dominant chart's coordinate, trained on TRAIN labels
only.  The target is *concept-shaped* — a standardized scalar for non-cyclic
labels, but ``(cos θ, sin θ)`` of the angle for cyclic ones (colours/hue), so a
wrap-around coordinate can be oriented onto the concept's loop, not just a line.
Sweeping its weight `lam_label` traces the **fidelity <-> legibility Pareto curve**:

  * y-axis: held-out subspace-capture VE@N (geometry fidelity)
  * x-axis: held-out label R^2 from the N-dim coordinate, FRESH linear probe
            (same protocol as fair_comparison.label_r2 — alignment only shapes
            the coordinate; the reported R^2 still tests generalization)

`lam_label=0` reproduces the earlier unsupervised failure.  The question: can we
push R^2 up to the PCA level while keeping VE above the PCA ceiling?  If yes, the
interpretability half of the bar is met; if pushing R^2 up tanks VE below PCA,
there is a real tradeoff.

Run (CPU, from repo root):
  SAE_DEVICE=cpu SAE_D_MODEL=576 uv run python prototype/legible_coord.py \
      --seeds 0 1 2 --lams 0 0.3 1 3 10
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from data import load_manifold_data, CACHE_DIR
from factored_sae import FactoredSAE, PRIMARY_LABEL
from fair_comparison import (load_split, factored_eval, label_score, _ms,
                             _pca_basis, _subspace_rep, CYCLIC_LABEL)

RESULTS_DIR = CACHE_DIR / "legible_coord"
DEFAULT_MANIFOLDS = ["years", "age", "temperature", "colors", "geography"]
# Cyclic-label manifolds are scored cyclically (cos/sin of the angle, via
# fair_comparison.label_score) AND, when lam_label>0, aligned cyclically (the
# per-manifold probe targets cos/sin — see train_factored_legible).  They are
# reported separately and kept out of mean* only so that average stays comparable
# to earlier (non-cyclic) runs; the cyclic column shows their trajectory.
CYCLIC = {"colors"}  # hue wraps


def _mixture_arrays(per):
    """Rebuild the mixture train set with per-point manifold id + label, in the
    exact concatenation order load_split uses (dict insertion = manifold order)."""
    Xs, mids, ys = [], [], []
    names = list(per)
    for mi, name in enumerate(names):
        d = per[name]
        Xs.append(d["Xtr"]); ys.append(d["ytr"])
        mids.append(np.full(len(d["Xtr"]), mi, dtype=np.int64))
    return (np.concatenate(Xs).astype(np.float32), np.concatenate(mids),
            np.concatenate(ys).astype(np.float32), names)


def train_factored_legible(Xtr_mix, mid_mix, y_mix, names, coord_dim,
                           lam_label, seed, n_charts=12, epochs=120, batch=512,
                           lr=2e-3, lam_sparse=0.02, lam_balance=0.3):
    """Nonlinear factored SAE + weak per-manifold label alignment on the
    dominant chart's coordinate (train labels only).

    The alignment target is *concept-shaped*: non-cyclic manifolds are aligned to
    the standardized scalar label (a line), but cyclic manifolds (CYCLIC_LABEL,
    e.g. colours/hue) are aligned to ``(cos θ, sin θ)`` of the angle — so the
    coordinate can be oriented onto the concept's *loop* instead of a line, which
    a linear probe on the raw value could never do.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    n_manifolds = len(names)
    mean = Xtr_mix.mean(0, keepdims=True)
    std = float(Xtr_mix.std()) + 1e-6
    # Train on DEVICE (GPU when available), then move model back to CPU before
    # returning so factored_eval keeps feeding CPU tensors.
    from data import DEVICE
    Xn = torch.from_numpy((Xtr_mix - mean) / std).to(DEVICE)
    mid = torch.from_numpy(mid_mix).to(DEVICE)
    N, d_in = Xn.shape

    # Per-point 2-component alignment target + active-component mask:
    #   non-cyclic: target = (standardized label, --),  mask = (1, 0)
    #   cyclic:     target = (cos θ, sin θ),             mask = (1, 1)
    y = y_mix.astype(np.float32)
    target = np.zeros((N, 2), np.float32)
    tmask = np.zeros((N, 2), bool)
    for m, name in enumerate(names):
        sel = (mid_mix == m) & ~np.isnan(y)
        if sel.sum() < 2:
            continue
        period = CYCLIC_LABEL.get(name)
        if period is not None:
            theta = 2 * np.pi * y[sel] / period
            target[sel, 0], target[sel, 1] = np.cos(theta), np.sin(theta)
            tmask[sel] = True
        elif y[sel].std() > 0:
            target[sel, 0] = (y[sel] - y[sel].mean()) / (y[sel].std() + 1e-6)
            tmask[sel, 0] = True
    target = torch.from_numpy(target).to(DEVICE)
    tmask = torch.from_numpy(tmask).to(DEVICE)

    # hard_routing=True: model is trained as a true atlas (single-chart recon
    # per point) so that factored_eval's dominant-chart restriction matches the
    # training regime.  See `fair_comparison.train_factored` for the bug this
    # closes (was: soft mixture training + dominant-chart eval -> catastrophic).
    model = FactoredSAE(d_in, n_charts, coord_dim, linear_charts=False,
                        hard_routing=True).to(DEVICE)
    # Per-manifold probe: coord (R^cd) -> R^2; non-cyclic uses only component 0.
    probe_w = torch.nn.Parameter(torch.zeros(n_manifolds, coord_dim, 2, device=DEVICE))
    probe_b = torch.nn.Parameter(torch.zeros(n_manifolds, 2, device=DEVICE))
    opt = torch.optim.Adam(list(model.parameters()) + [probe_w, probe_b], lr=lr)
    eps = 1e-9

    for ep in range(epochs):
        perm = torch.randperm(N, device=DEVICE)
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            xb, mb, tb, mkb = Xn[idx], mid[idx], target[idx], tmask[idx]
            recon, a, coords = model(xb)          # a is one-hot (hard routing)
            mse = ((recon - xb) ** 2).sum(-1).mean()
            # Entropy regularizers need the *soft* routing (one-hot has 0 entropy
            # and no gradient signal).
            a_soft = F.softmax(model.router(xb), dim=-1)
            ent_point = -(a_soft * (a_soft + eps).log()).sum(-1).mean()
            usage = a_soft.mean(0)
            ent_usage = -(usage * (usage + eps).log()).sum()
            loss = mse + lam_sparse * ent_point - lam_balance * ent_usage

            if lam_label > 0 and mkb.any():
                # Align each labeled point's argmax-chart coordinate (detached
                # selection) with its manifold's concept-shaped target.
                ch = a.argmax(-1).detach()                       # [B]
                csel = coords[torch.arange(len(xb)), ch]         # [B, cd]
                pred = torch.einsum("bc,bck->bk", csel, probe_w[mb]) + probe_b[mb]
                sq = (pred - tb) ** 2 * mkb                       # [B, 2]
                loss = loss + lam_label * sq.sum() / mkb.sum().clamp(min=1)

            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    model.cpu()
    return model, (mean.astype(np.float32), std)


def run(manifolds, seeds, lams, coord_dim, headline_N):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # store[(lam, manifold, metric)] -> list over seeds
    ve, r2lin, r2knn = {}, {}, {}

    def add(store, key, val):
        store.setdefault(key, []).append(val)

    pca_r2 = {}  # manifold -> list of PCA linear R^2 (reference ceiling for legibility)
    for seed in seeds:
        print(f"\n===== seed {seed} =====")
        per, _ = load_split(manifolds, seed)
        Xtr_mix, mid_mix, y_mix, names = _mixture_arrays(per)

        # PCA reference legibility (held-out linear R^2 from top-N PCs).
        for name in names:
            d = per[name]
            B = _pca_basis(d["Xtr"], d["mean_m"], headline_N)
            Ztr, Zte = _subspace_rep(d["Xtr"], d["Xte"], d["mean_m"], B)
            add(pca_r2, name, label_score(Ztr, d["ytr"], Zte, d["yte"], name)[0])

        for lam in lams:
            print(f"  training nonlinear factored, lam_label={lam}...")
            model, norm = train_factored_legible(
                Xtr_mix, mid_mix, y_mix, names, coord_dim, lam, seed)
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


def run_names(manifolds, ve):
    return [m for m in manifolds if any((l, m) in ve for l in [0])] or manifolds


def _report(manifolds, lams, ve, r2lin, r2knn, pca_r2, cd, N, seeds):
    names = [m for m in manifolds if (lams[0], m) in ve]
    legible = [m for m in names if m not in CYCLIC]

    print(f"\n{'='*84}\nLEGIBILITY SWEEP  (nonlinear, coord_dim={cd}, N={N}, "
          f"{len(seeds)} seeds)\n{'='*84}")
    print("PCA reference linear R²:  " +
          "  ".join(f"{m}={_ms(pca_r2[m])[0]:.2f}" for m in names))

    print(f"\nHeld-out VE@{N} by lam_label:")
    print(f"{'lam':>6}" + "".join(f"{m[:9]:>11}" for m in names) + f"{'mean':>9}")
    for lam in lams:
        means = [_ms(ve[(lam, m)])[0] for m in names]
        print(f"{lam:>6}" + "".join(f"{v:>11.2f}" for v in means)
              + f"{np.mean(means):>9.2f}")

    print(f"\nHeld-out label R² (linear) by lam_label:")
    print(f"{'lam':>6}" + "".join(f"{m[:9]:>11}" for m in names)
          + f"{'mean*':>9}")
    for lam in lams:
        row = [_ms(r2lin[(lam, m)])[0] for m in names]
        leg_mean = np.mean([_ms(r2lin[(lam, m)])[0] for m in legible])
        print(f"{lam:>6}" + "".join(f"{v:>11.2f}" for v in row)
              + f"{leg_mean:>9.2f}")
    print("  (colors = cyclic cos/sin R², now cyclically aligned too; "
          f"mean* = mean over non-cyclic labeled manifolds, excludes {sorted(CYCLIC)})")


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
    ncol = len(names)
    fig, axes = plt.subplots(1, ncol, figsize=(3.6 * ncol, 3.6), squeeze=False)
    for ax, m in zip(axes[0], names):
        xs = [_ms(r2lin[(lam, m)])[0] for lam in lams]
        ys = [_ms(ve[(lam, m)])[0] for lam in lams]
        ax.plot(xs, ys, "-o", color="tab:red", zorder=3)
        for lam, x, y in zip(lams, xs, ys):
            ax.annotate(f"{lam:g}", (x, y), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        pr = _ms(pca_r2[m])[0]
        ax.axvline(pr, color="gray", ls="--", lw=1, label=f"PCA R²={pr:.2f}")
        cyc = " (cyclic)" if m in CYCLIC else ""
        ax.set_title(m + cyc); ax.set_xlabel("held-out label R²")
        ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower left")
    axes[0][0].set_ylabel(f"held-out VE@{cd}")
    fig.suptitle("Fidelity ↔ legibility trade-off as label-alignment weight increases "
                 "(points connected in lam_label order — not a Pareto frontier)")
    fig.tight_layout()
    path = RESULTS_DIR / "pareto.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"Saved plot -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifolds", nargs="+", default=DEFAULT_MANIFOLDS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--lams", nargs="+", type=float, default=[0, 0.3, 1, 3, 10])
    ap.add_argument("--coord-dim", type=int, default=3)
    ap.add_argument("--headline-n", type=int, default=3)
    a = ap.parse_args()
    run(a.manifolds, a.seeds, a.lams, a.coord_dim, a.headline_n)


if __name__ == "__main__":
    main()
