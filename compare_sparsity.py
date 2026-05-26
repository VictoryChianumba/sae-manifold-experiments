"""
Compare manifold-reconstruction (subspace-capture) curves across SAEs.

Overlays the *statistical* reconstruction curve (greedy over the SAE's actual
codes — the honest "does the SAE capture the manifold" measure) for several
SAE checkpoints on one axis per manifold, plus the PCA optimal-linear baseline.

This is the local analogue of the paper's sparsity sweep: it shows how manifold
capture changes as the SAE's L0 (k) changes.

Usage:
  SAE_DEVICE=cpu SAE_D_MODEL=576 uv run python compare_sparsity.py \
      --saes cache/sae_4608_k16.pt cache/sae_4608_k32.pt cache/sae_4608_k64.pt \
      --ks 16 32 64 --manifold years age temperature
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from data import load_manifold_data, CACHE_DIR
from saes import load_sae, encode_sae
from subspace_capture import find_support_greedy_codes

OUT = CACHE_DIR / "subspace_capture"


def curve_at(curve, ks):
    """Variance-explained sampled at feature counts ``ks`` (clamped)."""
    return [curve[min(k, len(curve)) - 1] if len(curve) else 0.0 for k in ks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saes", nargs="+", required=True)
    ap.add_argument("--ks", nargs="+", type=int, required=True)
    ap.add_argument("--manifold", nargs="+",
                    default=["years", "age", "temperature", "days"])
    ap.add_argument("--max-k", type=int, default=64)
    a = ap.parse_args()
    assert len(a.saes) == len(a.ks), "need one --ks per --saes"

    saes = [(k, load_sae(p, device="cpu", k=k)) for p, k in zip(a.saes, a.ks)]
    eval_ks = sorted(set([1, 2, 4, 8, 16, 32, a.max_k]))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(saes)))

    for manifold in a.manifold:
        data = load_manifold_data(manifold)
        if data is None:
            print(f"  {manifold}: not cached, skipping")
            continue
        X = data["activations"].float().numpy()
        Xc = X - X.mean(0)
        vpca = np.cumsum(PCA(n_components=min(a.max_k, *Xc.shape))
                         .fit(Xc).explained_variance_ratio_)

        fig, ax = plt.subplots(figsize=(7, 4.2))
        ks_eval = [k for k in eval_ks if k <= len(vpca)]
        ax.plot(ks_eval, curve_at(vpca, ks_eval), "o--", color="gray",
                alpha=0.7, label="PCA (optimal linear)")
        for (k, sae), c in zip(saes, colors):
            codes = encode_sae(sae, data["activations"])
            _, vstat, _ = find_support_greedy_codes(
                X, sae, codes, max_k=a.max_k, var_threshold=1.0)
            ks_e = [kk for kk in eval_ks if kk <= max(len(vstat), 1)]
            ax.plot(ks_e, curve_at(vstat, ks_e), "o-", color=c, alpha=0.9,
                    label=f"SAE codes (k={k})")
        ax.set_xlabel("Number of SAE features")
        ax.set_ylabel("Variance explained")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{manifold.capitalize()} — manifold capture vs. SAE sparsity")
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        out = OUT / f"{manifold}_sparsity_sweep.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  {manifold}: saved {out}")


if __name__ == "__main__":
    main()
