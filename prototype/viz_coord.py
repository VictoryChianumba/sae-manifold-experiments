"""
Visualize what "legible coordinate" means — the qualitative companion to the
held-out label-R² numbers in legible_coord.py.

A number (R² 0.16 -> 0.89) can hide a lot. These figures make the legibility
claim falsifiable at a glance, comparing the *unsupervised* chart coordinate
(lam_label=0) with the *weakly-aligned* one (lam_label=1):

  coord_vs_label.png  — for non-cyclic manifolds, held-out probe-predicted label
                        vs true label.  Legible => tight diagonal; scrambled =>
                        a cloud.  R² annotated (this IS the regression the metric
                        reports, drawn out).
  colors_circle.png   — colours (cyclic hue): held-out points in the predicted
                        (cos, sin) plane, coloured by true hue.  Legible => a
                        clean rainbow loop.

Run (CPU, from repo root):
  SAE_DEVICE=cpu SAE_D_MODEL=576 uv run python prototype/viz_coord.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.linear_model import LinearRegression

from data import CACHE_DIR
from fair_comparison import load_split, factored_eval, CYCLIC_LABEL
from legible_coord import train_factored_legible, _mixture_arrays

RESULTS_DIR = CACHE_DIR / "viz"
NONCYCLIC = ["years", "temperature", "geography"]
LAMS = [0.0, 1.0]                       # non-cyclic: unsupervised vs weak align (sweet spot)
COLOR_LAMS = [0.0, 10.0]                # colours need strong cyclic align to move at all
MODEL_LAMS = sorted(set(LAMS + COLOR_LAMS))
LABEL_NAME = {"years": "year", "temperature": "°F", "geography": "latitude",
              "age": "age", "colors": "hue"}


def _coords(model, norm, d):
    """Dominant-chart train/test coords + VE for one manifold."""
    fve, Ztr, Zte, dom = factored_eval(model, norm, d["Xtr"], d["Xte"], d["mean_m"])
    return Ztr, Zte, fve


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifolds = NONCYCLIC + ["colors"]
    per, _ = load_split(manifolds, seed=0)
    Xtr_mix, mid_mix, y_mix, names = _mixture_arrays(per)

    print("Training factored models (seed 0)...")
    models = {lam: train_factored_legible(Xtr_mix, mid_mix, y_mix, names,
                                          coord_dim=3, lam_label=lam, seed=0)
              for lam in MODEL_LAMS}

    # ── Figure 1: predicted vs true label, non-cyclic manifolds ──────────────
    fig, axes = plt.subplots(len(NONCYCLIC), len(LAMS),
                             figsize=(3.4 * len(LAMS), 3.2 * len(NONCYCLIC)),
                             squeeze=False)
    for r, name in enumerate(NONCYCLIC):
        d = per[name]
        ok_tr = ~np.isnan(d["ytr"]); ok_te = ~np.isnan(d["yte"])
        for c, lam in enumerate(LAMS):
            model, norm = models[lam]
            Ztr, Zte, _ = _coords(model, norm, d)
            reg = LinearRegression().fit(Ztr[ok_tr], d["ytr"][ok_tr])
            yhat = reg.predict(Zte[ok_te]); ytrue = d["yte"][ok_te]
            r2 = reg.score(Zte[ok_te], ytrue)
            ax = axes[r][c]
            ax.scatter(ytrue, yhat, s=14, alpha=0.6, color="tab:red")
            lo = min(ytrue.min(), yhat.min()); hi = max(ytrue.max(), yhat.max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
            ax.set_title(f"{name} — {'unsupervised' if lam == 0 else 'aligned (λ=1)'}"
                         f"\nheld-out R²={r2:.2f}", fontsize=10)
            ax.set_xlabel(f"true {LABEL_NAME[name]}")
            if c == 0:
                ax.set_ylabel("predicted (from coordinate)")
            ax.grid(alpha=0.3)
    fig.suptitle("What a legible coordinate looks like: predicted vs true label\n"
                 "(left = unsupervised curved coord, right = weak label alignment)",
                 fontsize=12)
    fig.tight_layout()
    p1 = RESULTS_DIR / "coord_vs_label.png"
    fig.savefig(p1, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {p1}")

    # ── Figure 2: colours, predicted (cos,sin) plane coloured by hue ─────────
    d = per["colors"]
    period = CYCLIC_LABEL["colors"]
    ok_tr = ~np.isnan(d["ytr"]); ok_te = ~np.isnan(d["yte"])
    Ttr = np.column_stack([np.cos(2 * np.pi * d["ytr"][ok_tr] / period),
                           np.sin(2 * np.pi * d["ytr"][ok_tr] / period)])
    fig, axes = plt.subplots(1, len(COLOR_LAMS), figsize=(4.2 * len(COLOR_LAMS), 4.0),
                             squeeze=False)
    for c, lam in enumerate(COLOR_LAMS):
        model, norm = models[lam]
        Ztr, Zte, _ = _coords(model, norm, d)
        reg = LinearRegression().fit(Ztr[ok_tr], Ttr)
        pred = reg.predict(Zte[ok_te]); r2 = reg.score(Zte[ok_te],
            np.column_stack([np.cos(2 * np.pi * d["yte"][ok_te] / period),
                             np.sin(2 * np.pi * d["yte"][ok_te] / period)]))
        ax = axes[0][c]
        sc = ax.scatter(pred[:, 0], pred[:, 1], c=d["yte"][ok_te], cmap="hsv",
                        s=20, alpha=0.8, vmin=0, vmax=1)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_title(f"colors — {'unsupervised' if lam == 0 else f'cyclic-aligned (λ={lam:g})'}"
                     f"\nheld-out cyclic R²={r2:.2f}", fontsize=10)
        ax.set_xlabel("predicted cos(2π·hue)"); ax.set_ylabel("predicted sin(2π·hue)")
    fig.colorbar(sc, ax=axes[0].tolist(), label="true hue", shrink=0.8)
    fig.suptitle("Colours is the hard case: even strong cyclic alignment forms only a "
                 "partial hue loop\n(hue is entangled with lightness/saturation)",
                 fontsize=11)
    p2 = RESULTS_DIR / "colors_circle.png"
    fig.savefig(p2, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {p2}")


if __name__ == "__main__":
    main()
