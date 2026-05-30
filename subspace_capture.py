"""
Subspace capture evaluation for SAE features on manifold datasets.

This implements the two reconstruction-curve experiments and the
feature-tuning visualization from "Do Sparse Autoencoders Capture Concept
Manifolds?":

  - **Geometric reconstruction** (``find_support_greedy``): greedy subspace
    pursuit over decoder directions. At each step, pick the decoder atom that
    best reduces the residual of the centered manifold activations.
  - **Statistical reconstruction** (``find_support_greedy_codes``): greedy
    selection using the actual SAE codes — at each step, pick the feature
    whose centered contribution ``(z_i - <z_i>) d_i`` most reduces the
    residual.

Compared against PCA (optimal linear) and random-baseline curves, these two
greedy methods produce Fig. 4 of the paper. The tuning-curve plot reproduces
Fig. 5 (years_recon_tuning).

Usage:
  uv run subspace_capture.py plot   --sae /path/to/sae.pt --k 64
  uv run subspace_capture.py tuning --sae /path/to/sae.pt --k 64 --manifold years
"""
import argparse
from pathlib import Path

import torch
import numpy as np
from sklearn.decomposition import PCA

from data import (
    D_MODEL, DEVICE, CACHE_DIR,
    get_all_manifold_names, load_manifold_data,
)
from saes import load_sae, encode_sae, get_decoder

RESULTS_DIR = CACHE_DIR / "subspace_capture"


# ── Elbow detection (used by the greedy curves) ──────────────────────────────

def _detect_elbow(curve, min_k=1):
    """Maximum-distance-to-chord elbow detector on a monotone curve.

    Returns the index of the elbow point; used by the greedy methods below
    to suggest a cutoff when neither a variance threshold nor ``max_k`` is hit.
    """
    n = len(curve)
    if n <= min_k + 1:
        return n - 1
    x = np.arange(n, dtype=float)
    y = np.asarray(curve, dtype=float)
    p0 = np.array([x[0], y[0]])
    p1 = np.array([x[-1], y[-1]])
    line_vec = p1 - p0
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-10:
        return n - 1
    line_unit = line_vec / line_len
    dists = np.abs(np.cross(line_unit, p0 - np.column_stack([x, y])))
    dists[:min_k] = -1
    return int(np.argmax(dists))


# ── Geometric reconstruction (decoder directions) ────────────────────────────

def find_support_greedy(activations, decoder, max_k=100, var_threshold=0.95,
                        device=None):
    """Greedy subspace pursuit over decoder directions.

    At each step adds the decoder atom whose direction captures the most
    remaining variance of the centered manifold activations. Returns the
    selected feature indices, the cumulative variance-explained curve, and
    the suggested elbow ``k``.

    The hot loop (projections, scores, SVD) runs on ``device`` (default
    "cpu" for back-compat); pass ``device=DEVICE`` to put it on cuda at the
    8B/32k-feature scale. Algorithm is unchanged; inputs/outputs match the
    original numpy signature.
    """
    dev = torch.device(device) if device is not None else torch.device("cpu")
    X = torch.as_tensor(activations, dtype=torch.float32, device=dev)
    X = X - X.mean(0)
    total_ss = (X ** 2).sum().item()
    if total_ss < 1e-10:
        return np.array([], dtype=int), np.array([]), 0

    # candidates = np.arange(n_atoms) in the original, so local==global.
    D_cand = torch.as_tensor(decoder, dtype=torch.float32, device=dev)
    d_norms_sq = (D_cand ** 2).sum(1)
    alive = d_norms_sq > 1e-10

    selected = []
    var_curve = []
    residual = X.clone()
    neg_inf = float("-inf")

    for _ in range(max_k):
        projections = residual @ D_cand.T
        scores = (projections ** 2).sum(0) / d_norms_sq.clamp_min(1e-10)
        scores = torch.where(alive, scores, torch.full_like(scores, neg_inf))
        for i in selected:
            scores[i] = neg_inf
        best = int(torch.argmax(scores).item())
        if scores[best].item() <= 0:
            break
        selected.append(best)
        D_sel = D_cand[selected]
        _, s, Vt = torch.linalg.svd(D_sel, full_matrices=False)
        basis = Vt[s > 1e-8]
        residual = X - (X @ basis.T) @ basis
        explained = 1.0 - (residual ** 2).sum().item() / total_ss
        var_curve.append(float(explained))
        if explained >= var_threshold:
            break

    var_curve = np.array(var_curve)
    elbow_k = (_detect_elbow(var_curve, min_k=1) + 1
               if len(var_curve) > 2 else len(var_curve))
    return np.array(selected), var_curve, elbow_k


# ── Statistical reconstruction (actual SAE codes) ────────────────────────────

def find_support_greedy_codes(activations, sae, codes,
                              max_k=100, var_threshold=0.95, device=None):
    """Greedy selection by manifold variance explained from actual SAE codes.

    For each feature ``i``, its centered contribution to the reconstruction is
    ``(z_i - <z_i>) d_i``. At each step adds the feature whose contribution
    most reduces the residual.

    The hot loop runs on ``device`` (default "cpu" for back-compat); pass
    ``device=DEVICE`` for cuda at the 8B/32k-feature scale.
    """
    dev = torch.device(device) if device is not None else torch.device("cpu")
    X = torch.as_tensor(activations, dtype=torch.float32, device=dev)
    X_c = X - X.mean(0)
    total_ss = (X_c ** 2).sum().item()
    if total_ss < 1e-10:
        return np.array([], dtype=int), np.array([]), 0

    Z = torch.as_tensor(codes, dtype=torch.float32, device=dev)
    decoder = get_decoder(sae)  # CPU numpy by convention

    # Candidate features = those that ever fire positively on the manifold.
    candidates_t = torch.nonzero((Z > 0).any(0), as_tuple=False).flatten()
    if candidates_t.numel() == 0:
        return np.array([], dtype=int), np.array([]), 0
    candidates = candidates_t.cpu().numpy()  # for the global-index return

    Z_cand = Z[:, candidates_t]
    Z_c = Z_cand - Z_cand.mean(0)
    D_cand = torch.as_tensor(decoder[candidates], dtype=torch.float32, device=dev)
    n_cand = candidates_t.numel()
    z_c_sq = (Z_c ** 2).sum(0)
    d_sq = (D_cand ** 2).sum(1)
    contrib_ss = z_c_sq * d_sq

    selected, selected_local, var_curve = [], [], []
    residual = X_c.clone()
    alive = torch.ones(n_cand, dtype=torch.bool, device=dev)
    neg_inf = float("-inf")

    for _ in range(max_k):
        cross = (residual @ D_cand.T) * Z_c
        scores = 2 * cross.sum(0) - contrib_ss
        scores = torch.where(alive, scores, torch.full_like(scores, neg_inf))
        best_local = int(torch.argmax(scores).item())
        if scores[best_local].item() <= 0:
            break
        selected_local.append(best_local)
        selected.append(int(candidates[best_local]))
        alive[best_local] = False
        S_local = torch.tensor(selected_local, dtype=torch.long, device=dev)
        recon = Z_c[:, S_local] @ D_cand[S_local]
        residual = X_c - recon
        explained = 1.0 - (residual ** 2).sum().item() / total_ss
        var_curve.append(float(explained))
        if explained >= var_threshold:
            break

    var_curve = np.array(var_curve)
    elbow_k = (_detect_elbow(var_curve, min_k=1) + 1
               if len(var_curve) > 2 else len(var_curve))
    return np.array(selected), var_curve, elbow_k


# ── Plotting: variance-explained curves ──────────────────────────────────────

def plot_greedy_variance_curves(sae_path, manifolds=None, max_k=64,
                                out_dir=None, **sae_kwargs):
    """Plot ``# features`` vs ``variance explained`` for both greedy methods
    alongside PCA and random baselines (one PDF per manifold).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    sae = load_sae(sae_path, device=DEVICE, **sae_kwargs)
    decoder = get_decoder(sae)

    out_dir = Path(out_dir) if out_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    names = manifolds or get_all_manifold_names()
    for manifold in names:
        data = load_manifold_data(manifold)
        if data is None:
            print(f"  {manifold}: no cached data, skipping")
            continue
        acts = data['activations']
        acts_np = (acts.float().numpy()
                   if isinstance(acts, torch.Tensor) else acts)
        codes = encode_sae(sae, acts)

        X_c = acts_np - acts_np.mean(0)
        pca = PCA(n_components=min(max_k, *X_c.shape)).fit(X_c)
        vc_pca = np.cumsum(pca.explained_variance_ratio_)

        n_active = int((codes > 0).any(0).sum())
        k_max = min(max_k, n_active)

        print(f"  {manifold}: geometric reconstruction (decoder directions)...")
        _, vc_dir, _ = find_support_greedy(
            acts_np, decoder, max_k=max_k, var_threshold=1.0, device=DEVICE)

        print(f"  {manifold}: statistical reconstruction (SAE codes)...")
        _, vc_codes, _ = find_support_greedy_codes(
            acts_np, sae, codes, max_k=max_k, var_threshold=1.0, device=DEVICE)

        d = X_c.shape[1]
        n_seeds = 5
        n_atoms = decoder.shape[0]

        print(f"  {manifold}: random orthogonal baseline...")
        vc_orth_list = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            Q, _ = np.linalg.qr(rng.standard_normal((d, d)).astype(np.float32))
            _, vc_r, _ = find_support_greedy(
                acts_np, Q, max_k=max_k, var_threshold=1.0, device=DEVICE)
            vc_orth_list.append(vc_r[:k_max])

        print(f"  {manifold}: random overcomplete baseline...")
        vc_over_list = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed + 100)
            R = rng.standard_normal((n_atoms, d)).astype(np.float32)
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            _, vc_r, _ = find_support_greedy(
                acts_np, R, max_k=max_k, var_threshold=1.0, device=DEVICE)
            vc_over_list.append(vc_r[:k_max])

        def _avg_curves(curves):
            if not curves:
                return np.array([])
            ml = max(len(v) for v in curves)
            padded = np.array([
                np.pad(v, (0, ml - len(v)),
                       constant_values=v[-1] if len(v) > 0 else 0)
                for v in curves
            ])
            return padded.mean(0)

        vc_rand_orth = _avg_curves(vc_orth_list)
        vc_rand_over = _avg_curves(vc_over_list)

        eval_ks = sorted(set(
            [1, 2] + [2**i for i in range(2, 10) if 2**i <= k_max]
            + [k_max]
        ))

        def _subsample(curve, ks=eval_ks):
            if len(curve) == 0:
                return np.array([]), np.array([])
            valid = [k for k in ks if k <= len(curve)]
            return np.array(valid), np.array([curve[k - 1] for k in valid])

        sns.set_style('whitegrid', {'grid.color': '#dedede'})
        fig, ax = plt.subplots(figsize=(8, 4))

        ks_pca, vs_pca = _subsample(vc_pca[:k_max])
        if len(ks_pca) > 0:
            ax.plot(ks_pca, vs_pca, linewidth=1.5, color='gray', alpha=0.6,
                    linestyle='--', label='PCA (optimal)',
                    marker='o', markersize=3)
        if len(vc_rand_orth) > 0:
            ks_ro, vs_ro = _subsample(vc_rand_orth)
            ax.plot(ks_ro, vs_ro, linewidth=1.5, color='orange', alpha=0.6,
                    linestyle=':', label='Random orthogonal',
                    marker='o', markersize=3)
        if len(vc_rand_over) > 0:
            ks_rov, vs_rov = _subsample(vc_rand_over)
            ax.plot(ks_rov, vs_rov, linewidth=1.5, color='green', alpha=0.6,
                    linestyle=':', label='Random overcomplete',
                    marker='o', markersize=3)
        if len(vc_dir) > 0:
            ks_d, vs_d = _subsample(vc_dir[:k_max])
            ax.plot(ks_d, vs_d, linewidth=1.5, color='blue', alpha=0.8,
                    label='Geometric (decoder directions)',
                    marker='o', markersize=4)
        if len(vc_codes) > 0:
            vc = vc_codes[:k_max]
            if len(vc) < k_max:
                vc = np.concatenate([vc, np.full(k_max - len(vc), vc[-1])])
            ks_c, vs_c = _subsample(vc)
            ax.plot(ks_c, vs_c, linewidth=1.5, color='magenta', alpha=0.8,
                    label='Statistical (code reconstruction)',
                    marker='o', markersize=4)

        ax.set_xlabel('Number of SAE features')
        ax.set_ylabel('Variance explained')
        ax.set_title(f"{manifold.capitalize()} — manifold reconstruction")
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1),
                  borderaxespad=0, frameon=False)
        sns.despine()
        fig.tight_layout()
        out_path = out_dir / f"{manifold}_greedy_ve.pdf"
        fig.savefig(str(out_path), bbox_inches='tight')
        plt.close(fig)
        print(f"  {manifold}: saved to {out_path}")


# ── Plotting: tuning curves ──────────────────────────────────────────────────

# Manifold -> list of continuous label keys to plot on the x-axis.
_TUNING_LABELS = {
    'age':          ['age'],
    'temperature':  ['fahrenheit'],
    'colors':       ['hue', 'lightness', 'saturation'],
    'geography':    ['latitude', 'longitude'],
    'formality':    ['formality'],
    'sent_length':  ['n_tokens'],
    'years':        ['year'],
}


def plot_tuning_curves(sae_path, manifolds=None, n_features=10, sigma=3,
                       out_dir=None, label_range=None, normalize=True,
                       **sae_kwargs):
    """Plot feature tuning curves — ground-truth label (x) vs activation (y).

    For each manifold with continuous labels, finds the top-``n_features``
    features via the statistical-reconstruction greedy and plots each
    feature's binned activation as a smoothed line over sorted ground-truth
    values. Reproduces Fig. 5 (years_recon_tuning) of the paper.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as mpl_cm
    import seaborn as sns
    from scipy.ndimage import gaussian_filter1d

    sae = load_sae(sae_path, device=DEVICE, **sae_kwargs)

    out_dir = Path(out_dir) if out_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    names = [m for m in (manifolds or list(_TUNING_LABELS.keys()))
             if m in _TUNING_LABELS]
    if not names:
        print("No manifolds with continuous labels to plot.")
        return

    plasma = mpl_cm.get_cmap('magma')

    for manifold in names:
        data = load_manifold_data(manifold)
        if data is None:
            print(f"  {manifold}: no cached data, skipping")
            continue

        acts = data['activations']
        mlabels = data['labels']
        acts_np = (acts.float().numpy()
                   if isinstance(acts, torch.Tensor) else acts)
        codes = encode_sae(sae, acts)

        label_keys = [k for k in _TUNING_LABELS[manifold]
                      if k in mlabels[0]]
        if not label_keys:
            print(f"  {manifold}: no matching label keys, skipping")
            continue

        if label_range is not None:
            primary_all = np.array([l[label_keys[0]] for l in mlabels],
                                   dtype=float)
            mask = (primary_all >= label_range[0]) & (primary_all <= label_range[1])
            idx = np.where(mask)[0]
            if len(idx) == 0:
                print(f"  {manifold}: no samples in range {label_range}")
                continue
            acts_np = acts_np[idx]
            codes = codes[idx]
            mlabels = [mlabels[i] for i in idx]

        print(f"  {manifold}: finding top {n_features} features...")
        sel_codes, _, _ = find_support_greedy_codes(
            acts_np, sae, codes, max_k=n_features, var_threshold=1.0,
            device=DEVICE)
        top_feats = sel_codes[:n_features]
        n_top = len(top_feats)

        primary_vals = np.array([l[label_keys[0]] for l in mlabels], dtype=float)
        primary_order = np.argsort(primary_vals)
        primary_sorted = primary_vals[primary_order]
        _ux, _inv = np.unique(primary_sorted, return_inverse=True)
        peak_x = []
        for fi in top_feats:
            y = codes[primary_order, fi].astype(float)
            y_bin = np.array([y[_inv == bi].mean() for bi in range(len(_ux))])
            if sigma > 0:
                y_bin = gaussian_filter1d(y_bin, sigma=sigma)
            peak_x.append(_ux[np.argmax(y_bin)])
        top_feats = top_feats[np.argsort(peak_x)]

        sns.set_style('whitegrid', {'grid.color': '#dedede'})
        n_labels = len(label_keys)
        fig, axes = plt.subplots(1, n_labels, figsize=(6 * n_labels, 4),
                                 squeeze=False)

        for li, lkey in enumerate(label_keys):
            ax = axes[0, li]
            vals = np.array([l[lkey] for l in mlabels], dtype=float)
            order = np.argsort(vals)
            x_sorted = vals[order]
            unique_x, inverse = np.unique(x_sorted, return_inverse=True)

            for fi_idx, fi in enumerate(top_feats):
                y_sorted = codes[order, fi].astype(float)
                y_binned = np.array([y_sorted[inverse == bi].mean()
                                     for bi in range(len(unique_x))])
                y_smooth = (gaussian_filter1d(y_binned, sigma=sigma)
                            if sigma > 0 else y_binned)
                if normalize:
                    y_min, y_max = y_smooth.min(), y_smooth.max()
                    if y_max - y_min > 1e-10:
                        y_smooth = (y_smooth - y_min) / (y_max - y_min)
                    else:
                        y_smooth = np.zeros_like(y_smooth)
                color = plasma(fi_idx / max(n_top - 1, 1))
                ax.plot(unique_x, y_smooth, linewidth=1.2, color=color,
                        alpha=0.85, label=f'feat {int(fi)}')
                ax.fill_between(unique_x, 0, y_smooth, color=color, alpha=0.15)

            ax.set_xlabel(lkey.replace('_', ' ').capitalize())
            ax.set_ylabel('Feature activation (normalized)' if normalize
                          else 'Feature activation')
            if n_labels > 1:
                ax.set_title(lkey.replace('_', ' ').capitalize())

        fig.suptitle(f'{manifold.capitalize()} — feature tuning curves (top {n_top})',
                     fontsize=13)
        handles, lbls = axes[0, -1].get_legend_handles_labels()
        fig.legend(handles, lbls, loc='center left',
                   bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=8)
        sns.despine()
        fig.tight_layout()
        out_path = out_dir / f"{manifold}_tuning_curves.pdf"
        fig.savefig(str(out_path), bbox_inches='tight')
        plt.close(fig)
        print(f"  {manifold}: saved to {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def _add_sae_args(parser):
    parser.add_argument('--sae', type=str, required=True,
                        help='Path to SAE checkpoint')
    parser.add_argument('--d-in', type=int, default=D_MODEL)
    parser.add_argument('--d-sae', type=int, default=None,
                        help='SAE width; inferred from checkpoint if absent')
    parser.add_argument('--expansion-factor', type=int, default=None,
                        help='Alternative to --d-sae (d_sae = d_in * factor)')
    parser.add_argument('--k', type=int, default=None,
                        help='BatchTopK sparsity')


def _sae_kwargs(a):
    return dict(d_in=a.d_in, d_sae=a.d_sae,
                expansion_factor=a.expansion_factor, k=a.k)


def main():
    p = argparse.ArgumentParser(
        description="Subspace capture evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest='command')

    plot_p = sub.add_parser('plot',
                            help='VE curves: geometric vs. statistical greedy + PCA/random baselines')
    _add_sae_args(plot_p)
    plot_p.add_argument('--manifold', nargs='*', default=None)
    plot_p.add_argument('--max-k', type=int, default=64)
    plot_p.add_argument('--out-dir', type=str, default=None)

    tc_p = sub.add_parser('tuning', help='Feature tuning curves (label vs activation)')
    _add_sae_args(tc_p)
    tc_p.add_argument('--manifold', nargs='*', default=None)
    tc_p.add_argument('--n-features', type=int, default=10)
    tc_p.add_argument('--sigma', type=float, default=3,
                       help='Gaussian smoothing sigma (samples)')
    tc_p.add_argument('--out-dir', type=str, default=None)
    tc_p.add_argument('--label-range', type=float, nargs=2, default=None,
                       metavar=('MIN', 'MAX'))
    tc_p.add_argument('--no-normalize', action='store_true')

    a = p.parse_args()
    if a.command == 'plot':
        plot_greedy_variance_curves(
            a.sae, manifolds=a.manifold, max_k=a.max_k,
            out_dir=a.out_dir, **_sae_kwargs(a))
    elif a.command == 'tuning':
        plot_tuning_curves(
            a.sae, manifolds=a.manifold,
            n_features=a.n_features, sigma=a.sigma,
            out_dir=a.out_dir,
            label_range=tuple(a.label_range) if a.label_range else None,
            normalize=not a.no_normalize,
            **_sae_kwargs(a))
    else:
        p.print_help()


if __name__ == '__main__':
    main()
