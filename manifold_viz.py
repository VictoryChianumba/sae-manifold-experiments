"""
3D manifold visualizations — the picture from the Goodfire article/paper.

Recipe (same as the article's ORDINALS / COMPOSERS / SCOTUS / CHEMICAL-BONDS
figures and the helix at the top of the post):

  1. take a concept manifold's cached activations,
  2. PCA them to 3 dimensions,
  3. 3D scatter, colored by the ground-truth label (the rainbow gradient),
  4. cast a gray shadow on a floor plane as a depth cue,
  5. optionally overlay the SAE decoder directions that the *statistical*
     greedy selects for this manifold (the straight chords across the curve).

Usage:
  SAE_DEVICE=cpu SAE_D_MODEL=576 uv run python manifold_viz.py \
      --manifold years colors geography days \
      --sae cache/sae_4608_k32.pt --k 32 --overlay-features 6
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA

from data import load_manifold_data, CACHE_DIR, DEVICE

OUT = CACHE_DIR / "manifold_viz"

# Default label to color each manifold by (continuous unless noted).
COLOR_KEY = {
    "years": "year", "age": "age", "temperature": "fahrenheit",
    "days": "time_idx", "colors": "hue", "geography": "latitude",
    "formality": "formality", "sent_length": "n_tokens",
}


def _labels_array(labels, key):
    """Return (values, is_numeric) for a label key across all samples."""
    vals = [l.get(key) for l in labels]
    try:
        return np.array([float(v) for v in vals], dtype=float), True
    except (TypeError, ValueError):
        uniq = sorted(set(map(str, vals)))
        idx = {u: i for i, u in enumerate(uniq)}
        return np.array([idx[str(v)] for v in vals]), False


# A concept's *defining* structure can sit in low-variance PCs, so a top-3-by-
# variance projection hides it. Empirically (Llama-3.1-8B L16): the years helix's
# decade cycle lives in PC4/PC5 while PC1 holds only the linear century trend, so
# the top-3 plot shows a smear, not a helix; colours' hue circle, by contrast, is
# in the top PCs at 8B but buried in PC8 at 135M. We therefore pick which 3 PCs to
# display by how well they track the concept (cyclic-aware), not by raw variance.
# These (label_key, period) pairs name cyclic components used ONLY for PC
# selection — they never alter the data or the colouring.
CYCLIC_COMPONENTS = {
    "colors": [("hue", 1.0)],                       # hue wraps with period 1
    "years":  [("year", 100.0), ("year", 10.0)],    # century + decade cycles -> helix
    "days":   [("day_idx", 7.0)],                   # day-of-week wraps with period 7
}


def _abscorr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if d < 1e-12 else abs(float(a @ b / d))


def _concept_targets(labels, color_key, manifold):
    """Label-derived target vectors used to score how 'concept-bearing' a PC is:
    the standardised primary label plus cos/sin of any configured cyclic component."""
    targets = []
    if color_key:
        vals, numeric = _labels_array(labels, color_key)
        if numeric and np.std(vals) > 0:
            targets.append((vals - vals.mean()) / (vals.std() + 1e-9))
    for key, period in CYCLIC_COMPONENTS.get(manifold, []):
        v, numeric = _labels_array(labels, key)
        if numeric:
            th = 2 * np.pi * v / period
            targets.append(np.cos(th)); targets.append(np.sin(th))
    return targets


def _select_concept_pcs(P_all, targets, k=3):
    """Pick the k PCs (from the candidate pool) most correlated with any concept
    target. Falls back to the top-k by variance when no numeric target exists.
    Returns indices into ``P_all``'s columns, ordered best-first."""
    n_pc = P_all.shape[1]
    if not targets or n_pc <= k:
        return list(range(min(k, n_pc)))
    score = [max(_abscorr(P_all[:, i], t) for t in targets) for i in range(n_pc)]
    return sorted(range(n_pc), key=lambda i: score[i], reverse=True)[:k]


def viz_manifold(manifold, sae=None, overlay_features=0, k=None,
                 color_key=None, point_size=14):
    data = load_manifold_data(manifold)
    if data is None:
        print(f"  {manifold}: not cached, skipping")
        return
    X = data["activations"].float().numpy()
    labels = data["labels"]

    key = color_key or COLOR_KEY.get(manifold)
    if key is None or key not in labels[0]:
        key = next((k_ for k_ in labels[0]
                    if isinstance(labels[0][k_], (int, float))), None)
    cvals, numeric = _labels_array(labels, key) if key else (None, True)

    # PCA to a candidate pool, then pick the 3 PCs that best track the concept
    # (cyclic-aware) rather than the top-3 by variance — the concept can live in
    # low-variance PCs (years' decade cycle is PC4/PC5 at 8B). See
    # _select_concept_pcs; with no numeric label it falls back to top-3.
    Xc = X - X.mean(0)
    n_pool = min(10, *Xc.shape)
    pca = PCA(n_components=n_pool).fit(Xc)
    P_all = pca.transform(Xc)
    targets = _concept_targets(labels, key, manifold)
    sel = _select_concept_pcs(P_all, targets, k=3)
    P = P_all[:, sel]
    var = pca.explained_variance_ratio_[sel] * 100
    pc_label = "/".join(f"PC{i + 1}" for i in sel)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    zfloor = P[:, 2].min() - 0.15 * np.ptp(P[:, 2])
    # Floor shadow (depth cue).
    ax.scatter(P[:, 0], P[:, 1], np.full(len(P), zfloor),
               s=point_size * 0.7, c="0.6", alpha=0.10, edgecolors="none")
    # The manifold itself, colored by label.
    cmap = "twilight" if (manifold == "colors") else "plasma"
    sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=point_size, c=cvals,
                    cmap=cmap if numeric else "tab10", alpha=0.9,
                    edgecolors="none")

    # Overlay the SAE decoder directions the statistical greedy picks.
    if sae is not None and overlay_features > 0:
        from saes import encode_sae, get_decoder
        from subspace_capture import find_support_greedy_codes
        codes = encode_sae(sae, data["activations"])
        sel_feats, _, _ = find_support_greedy_codes(
            X, sae, codes, max_k=overlay_features, var_threshold=1.0)
        dec = get_decoder(sae)
        scale = 0.9 * np.abs(P).max()
        for fid in sel_feats[:overlay_features]:
            # Project the decoder direction onto the 3 *selected* concept PCs and
            # draw it as a single ray from the centroid (not a symmetric chord,
            # which read as meaningless clutter in earlier versions).
            d3 = pca.components_[sel] @ dec[fid]
            d3 = d3 / (np.linalg.norm(d3) + 1e-8) * scale
            ax.plot([0, d3[0]], [0, d3[1]], [0, d3[2]],
                    color="0.4", lw=1.0, alpha=0.5)

    if key and cvals is not None:
        cb = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.02)
        cb.set_label(key.replace("_", " "))
    ax.set_title(f"{manifold.capitalize()} manifold — concept PCs {pc_label}\n"
                 f"(variance {var[0]:.0f}/{var[1]:.0f}/{var[2]:.0f}%)",
                 fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor((1, 1, 1, 0))
        pane.pane.set_edgecolor((0.9, 0.9, 0.9, 1))
    ax.view_init(elev=18, azim=-60)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{manifold}_manifold3d.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {manifold}: saved {out}  (colored by '{key}', "
          f"concept PCs {pc_label}, {var.sum():.0f}% var)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifold", nargs="+",
                    default=["years", "age", "temperature", "days"])
    ap.add_argument("--sae", default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--d-in", type=int, default=None)
    ap.add_argument("--overlay-features", type=int, default=0)
    ap.add_argument("--color-key", default=None)
    a = ap.parse_args()
    sae = None
    if a.sae:
        from saes import load_sae
        sae = load_sae(a.sae, device=DEVICE,
                       **({"d_in": a.d_in} if a.d_in else {}),
                       **({"k": a.k} if a.k else {}))
    for m in a.manifold:
        viz_manifold(m, sae=sae, overlay_features=a.overlay_features, k=a.k,
                     color_key=a.color_key)


if __name__ == "__main__":
    main()
