"""
Fair-comparison experiment: does an atlas-of-curved-charts recover concept-
manifold geometry better than a linear dictionary, at matched dimensionality,
held out?  (See ../REPRODUCTION.md and ./NEXT_SESSION.md, "ITEM ONE".)

This is the controlled follow-up to the suggestive-but-confounded ablation in
factored_sae.py.  Everything here is on a held-out test split, scored in raw
activation units against a single per-manifold denominator, so every method is
directly comparable.

Primary metric — held-out subspace-capture VE(N) (the paper's own yardstick):
  for each manifold, fraction of its TEST activation variance explained by an
  N-dimensional per-point code.  Methods:
    * PCA-N                    optimal linear ceiling (top-N PCs from train).
    * SAE geometric greedy     N decoder atoms greedily selected on the train
                               residual; VE on test (a linear subspace, so <= PCA-N).
    * SAE statistical codes     the SAE's actual centered-code reconstruction
                               (the paper's "dilution" curve), select-on-train.
    * Factored linear,  cd=N    dominant-chart reconstruction from N coords.
    * Factored nonlinear, cd=N  same, but curved charts.
  Only a *curved* decoder can exceed PCA-N.  factored_nl - PCA-N isolates
  curvature; factored_nl - factored_lin controls for clustering/params.

Secondary metric — held-out label decodability: predict the manifold's primary
  label from the N-dim representation, same regressor for every condition
  (linear + kNN), fit on train, R^2 on test.

Standard SAE baselines: the existing C4-trained SAE (realistic off-the-shelf
  lens) AND one retrained on the mixture train split (isolates architecture
  from training data).

Run (CPU; from repo root):
  SAE_DEVICE=cpu SAE_D_MODEL=576 uv run python prototype/fair_comparison.py \
      --seeds 0 1 2 --coord-dims 2 3
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import repo modules

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from data import load_manifold_data, CACHE_DIR
from saes import BatchTopKSAE, load_sae, get_decoder
from factored_sae import FactoredSAE, PRIMARY_LABEL

RESULTS_DIR = CACHE_DIR / "fair_comparison"
DEFAULT_MANIFOLDS = ["years", "age", "temperature", "colors", "geography"]
# `days` (56 pts, 2D grid) is too small for a 70/30 held-out split; excluded
# from the headline but can be re-added via --manifolds.


# ── Data: per-manifold train/test split + concatenated mixture train ──────────

def load_split(manifolds, seed, test_frac=0.3):
    """Per-manifold 70/30 split + the concatenated mixture train set.

    Returns (per, Xtr_mix) where per[name] = dict(Xtr, Xte, ytr, yte, mean_m).
    The mixture train set is what the SAE and factored models train on; each
    manifold is scored on its own held-out test points.
    """
    rng = np.random.default_rng(seed)
    per = {}
    mix = []
    for name in manifolds:
        d = load_manifold_data(name)
        if d is None:
            print(f"  skip {name} (not cached)")
            continue
        X = d["activations"].float().numpy().astype(np.float32)
        key = PRIMARY_LABEL.get(name)
        y = (np.array([float(l.get(key, np.nan)) for l in d["labels"]], dtype=np.float32)
             if key else np.full(len(X), np.nan, np.float32))
        n = len(X)
        idx = rng.permutation(n)
        n_te = max(5, int(round(n * test_frac)))
        te, tr = idx[:n_te], idx[n_te:]
        per[name] = dict(Xtr=X[tr], Xte=X[te], ytr=y[tr], yte=y[te],
                         mean_m=X[tr].mean(0, keepdims=True))
        mix.append(X[tr])
    Xtr_mix = np.concatenate(mix).astype(np.float32)
    return per, Xtr_mix


# ── VE helpers (raw units, per-manifold train-mean denominator) ───────────────

def _ve(Xte, Xhat, mean_m):
    """1 - ||Xte - Xhat||^2 / ||Xte - mean_m||^2  (variance explained on test)."""
    ss_res = float(((Xte - Xhat) ** 2).sum())
    ss_tot = float(((Xte - mean_m) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def pca_curve(Xtr, Xte, mean_m, maxN):
    """VE(N) for top-N PCA reconstruction (optimal linear ceiling)."""
    Xc_tr = Xtr - mean_m
    pca = PCA(n_components=min(maxN, *Xc_tr.shape)).fit(Xc_tr)
    V = pca.components_                       # [n_comp, d]
    Xc_te = Xte - mean_m
    out = []
    for N in range(1, maxN + 1):
        if N > V.shape[0]:
            out.append(out[-1]); continue
        B = V[:N]
        Xhat = mean_m + (Xc_te @ B.T) @ B
        out.append(_ve(Xte, Xhat, mean_m))
    return np.array(out)


def sae_geometric_curve(Xtr, Xte, mean_m, decoder, maxN):
    """Greedy subspace pursuit over decoder atoms: select on train, score test.

    At each step pick the decoder atom that most reduces the centered TRAIN
    residual, then measure VE on TEST through the orthonormal basis of all
    atoms selected so far.  A linear N-subspace, hence <= PCA-N.
    """
    Xc_tr = Xtr - mean_m
    Xc_te = Xte - mean_m
    d_norms_sq = (decoder ** 2).sum(1).clip(1e-10)
    alive = d_norms_sq > 1e-10
    selected, curve, residual = [], [], Xc_tr.copy()
    for _ in range(maxN):
        proj = residual @ decoder.T
        scores = (proj ** 2).sum(0) / d_norms_sq
        scores[~alive] = -np.inf
        for i in selected:
            scores[i] = -np.inf
        best = int(np.argmax(scores))
        if not np.isfinite(scores[best]) or scores[best] <= 0:
            curve.append(curve[-1] if curve else 0.0); continue
        selected.append(best)
        _, s, Vt = np.linalg.svd(decoder[selected], full_matrices=False)
        B = Vt[s > 1e-8]
        residual = Xc_tr - (Xc_tr @ B.T) @ B
        Xhat = mean_m + (Xc_te @ B.T) @ B
        curve.append(_ve(Xte, Xhat, mean_m))
    while len(curve) < maxN:
        curve.append(curve[-1] if curve else 0.0)
    return np.array(curve)


def sae_statistical_curve(sae, Xtr, Xte, mean_m, maxN):
    """The SAE's actual centered-code reconstruction (paper's "statistical").

    Select features greedily by train-residual reduction; reconstruct test from
    the selected features' centered contributions (z_i - <z_i>_train) d_i.
    Not a subspace, so this is what the SAE *really represents*, not a ceiling.
    """
    with torch.no_grad():
        Ztr = sae.encode(torch.from_numpy(Xtr)).cpu().numpy().astype(np.float32)
        Zte = sae.encode(torch.from_numpy(Xte)).cpu().numpy().astype(np.float32)
    decoder = get_decoder(sae)
    cand = np.where((Ztr > 0).any(0))[0]
    if len(cand) == 0:
        return np.zeros(maxN)
    zmean = Ztr[:, cand].mean(0)
    Ztr_c = Ztr[:, cand] - zmean
    Zte_c = Zte[:, cand] - zmean
    D = decoder[cand]
    Xc_tr = Xtr - mean_m
    res = Xc_tr.copy()
    sel, curve = [], []
    contrib_ss = (Ztr_c ** 2).sum(0) * (D ** 2).sum(1)
    alive = np.ones(len(cand), bool)
    for _ in range(maxN):
        cross = (res @ D.T) * Ztr_c
        scores = 2 * cross.sum(0) - contrib_ss
        scores[~alive] = -np.inf
        best = int(np.argmax(scores))
        if not np.isfinite(scores[best]) or scores[best] <= 0:
            curve.append(curve[-1] if curve else 0.0); continue
        sel.append(best); alive[best] = False
        S = np.array(sel)
        res = Xc_tr - Ztr_c[:, S] @ D[S]
        Xhat = mean_m + Zte_c[:, S] @ D[S]
        curve.append(_ve(Xte, Xhat, mean_m))
    while len(curve) < maxN:
        curve.append(curve[-1] if curve else 0.0)
    return np.array(curve)


# ── Standard SAE trained on the mixture train split ───────────────────────────

def _batch_topk(pre, k):
    B = pre.shape[0]
    n_keep = min(max(1, int(k * B)), pre.numel())
    thresh = torch.topk(pre.flatten(), n_keep, sorted=False).values.min()
    return torch.where(pre >= thresh, pre, torch.zeros_like(pre))


def train_standard_sae(Xtr_mix, seed, expansion_factor=8, k=32,
                       epochs=60, batch_size=512, lr=4e-4, dead_steps=100,
                       aux_coef=1 / 32):
    """Minimal BatchTopK SAE trainer (same recipe as train_sae.py) on the
    mixture train split, so the standard baseline sees the same data as the
    factored models — isolating architecture from training data."""
    torch.manual_seed(seed); np.random.seed(seed)
    X = torch.from_numpy(Xtr_mix).float()
    N, d_in = X.shape
    d_sae = d_in * expansion_factor
    data_mean = X.mean(0)
    sae = BatchTopKSAE(d_in=d_in, d_sae=d_sae, k=k, device="cpu")
    with torch.no_grad():
        sae.decoder.bias.copy_(data_mean)
        W = sae.decoder.weight
        W.div_(W.norm(dim=0, keepdim=True).clamp_min(1e-8))
        sae.encoder.weight.copy_(W.t())
    for p in sae.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    last_fired = torch.zeros(d_sae)
    for ep in range(epochs):
        perm = torch.randperm(N)
        for bi in range(0, N, batch_size):
            x = X[perm[bi:bi + batch_size]]
            pre = torch.relu(sae.encoder(x))
            z = _batch_topk(pre, k)
            recon = sae.decode(z)
            loss = ((recon - x) ** 2).sum(1).mean()
            dead = last_fired > dead_steps
            if aux_coef > 0 and dead.any():
                pre_dead = pre.clone(); pre_dead[:, ~dead] = 0.0
                k_aux = min(int(dead.sum()), 2 * k)
                if k_aux > 0:
                    top = pre_dead.topk(k_aux, dim=-1)
                    z_aux = torch.zeros_like(pre_dead)
                    z_aux.scatter_(-1, top.indices, top.values)
                    aux = sae.decoder(z_aux) - sae.decoder.bias
                    loss = loss + aux_coef * ((aux - (x - recon)) ** 2).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                W = sae.decoder.weight
                W.div_(W.norm(dim=0, keepdim=True).clamp_min(1e-8))
                active = (z > 0).any(0)
                last_fired += 1; last_fired[active] = 0
    sae.eval()
    for p in sae.parameters():
        p.requires_grad = False
    return sae


# ── Factored model trained on the mixture train split ─────────────────────────

def train_factored(Xtr_mix, coord_dim, linear, seed, n_charts=12, epochs=120,
                   batch=512, lr=2e-3, lam_sparse=0.02, lam_balance=0.3):
    torch.manual_seed(seed); np.random.seed(seed)
    mean = Xtr_mix.mean(0, keepdims=True)
    std = float(Xtr_mix.std()) + 1e-6
    Xn = torch.from_numpy((Xtr_mix - mean) / std)
    N, d_in = Xn.shape
    model = FactoredSAE(d_in, n_charts, coord_dim, linear_charts=linear)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    eps = 1e-9
    for ep in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, batch):
            xb = Xn[perm[i:i + batch]]
            recon, a, _ = model(xb)
            mse = ((recon - xb) ** 2).sum(-1).mean()
            ent_point = -(a * (a + eps).log()).sum(-1).mean()
            usage = a.mean(0)
            ent_usage = -(usage * (usage + eps).log()).sum()
            loss = mse + lam_sparse * ent_point - lam_balance * ent_usage
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model, (mean.astype(np.float32), std)


@torch.no_grad()
def factored_eval(model, norm, Xtr, Xte, mean_m):
    """VE on test via dominant-chart reconstruction + the dominant chart's
    coords (for label decoding).  Dominant chart = modal argmax-router chart
    over this manifold's TRAIN points, so train and test use the same chart."""
    mean, std = norm
    Xn_tr = torch.from_numpy((Xtr - mean) / std)
    Xn_te = torch.from_numpy((Xte - mean) / std)
    _, a_tr, _ = model(Xn_tr)
    dom = int(np.bincount(a_tr.argmax(-1).numpy(),
                          minlength=model.n_charts).argmax())
    recon_te, _, coords_te = model(Xn_te)
    _, _, coords_tr = model(Xn_tr)
    Xhat = recon_te.numpy() * std + mean
    ve = _ve(Xte, Xhat, mean_m)
    Ztr = coords_tr[:, dom, :].numpy()
    Zte = coords_te[:, dom, :].numpy()
    return ve, Ztr, Zte, dom


# ── Label decodability (secondary): same regressor for every condition ────────

def label_r2(Ztr, ytr, Zte, yte):
    """Held-out R^2 predicting the label from an N-dim representation.
    Returns (linear_r2, knn_r2); NaN-safe."""
    ok_tr = ~np.isnan(ytr); ok_te = ~np.isnan(yte)
    if ok_tr.sum() < 5 or ok_te.sum() < 3 or np.ptp(ytr[ok_tr]) == 0:
        return np.nan, np.nan
    Ztr, ytr = Ztr[ok_tr], ytr[ok_tr]
    Zte, yte = Zte[ok_te], yte[ok_te]
    lin = LinearRegression().fit(Ztr, ytr).score(Zte, yte)
    kn = KNeighborsRegressor(n_neighbors=min(10, len(Ztr))).fit(Ztr, ytr).score(Zte, yte)
    return float(lin), float(kn)


def _subspace_rep(Xtr, Xte, mean_m, basis):
    """Project onto an orthonormal basis -> N-dim scores for label decoding."""
    return (Xtr - mean_m) @ basis.T, (Xte - mean_m) @ basis.T


def _pca_basis(Xtr, mean_m, N):
    return PCA(n_components=N).fit(Xtr - mean_m).components_[:N]


def _sae_greedy_basis(Xtr, mean_m, decoder, N):
    """The N-dim orthonormal basis the geometric greedy selects (on train)."""
    Xc = Xtr - mean_m
    d_norms_sq = (decoder ** 2).sum(1).clip(1e-10)
    sel, residual = [], Xc.copy()
    for _ in range(N):
        scores = ((residual @ decoder.T) ** 2).sum(0) / d_norms_sq
        for i in sel:
            scores[i] = -np.inf
        sel.append(int(np.argmax(scores)))
        _, s, Vt = np.linalg.svd(decoder[sel], full_matrices=False)
        B = Vt[s > 1e-8]
        residual = Xc - (Xc @ B.T) @ B
    return B


# ── Driver ────────────────────────────────────────────────────────────────────

def run(manifolds, seeds, coord_dims, maxN, c4_sae_path, headline_N):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    c4_sae = load_sae(c4_sae_path, d_in=576, device="cpu")
    c4_dec = get_decoder(c4_sae)

    # results[(manifold, method, N)] -> list of VE over seeds
    ve = {}
    lab = {}   # (manifold, method, N) -> list of (lin, knn)

    def add(store, key, val):
        store.setdefault(key, []).append(val)

    for seed in seeds:
        print(f"\n===== seed {seed} =====")
        per, Xtr_mix = load_split(manifolds, seed)
        names = list(per)

        print("  training mixture standard SAE...")
        mix_sae = train_standard_sae(Xtr_mix, seed)
        mix_dec = get_decoder(mix_sae)

        factored = {}
        for cd in coord_dims:
            for linear in (True, False):
                tag = f"{'lin' if linear else 'nl'}{cd}"
                print(f"  training factored {tag}...")
                factored[(cd, linear)] = train_factored(Xtr_mix, cd, linear, seed)

        for name in names:
            d = per[name]
            Xtr, Xte, mean_m = d["Xtr"], d["Xte"], d["mean_m"]
            ytr, yte = d["ytr"], d["yte"]

            curves = {
                "pca": pca_curve(Xtr, Xte, mean_m, maxN),
                "sae_c4_geo": sae_geometric_curve(Xtr, Xte, mean_m, c4_dec, maxN),
                "sae_c4_stat": sae_statistical_curve(c4_sae, Xtr, Xte, mean_m, maxN),
                "sae_mix_geo": sae_geometric_curve(Xtr, Xte, mean_m, mix_dec, maxN),
                "sae_mix_stat": sae_statistical_curve(mix_sae, Xtr, Xte, mean_m, maxN),
            }
            for method, curve in curves.items():
                for N in range(1, maxN + 1):
                    add(ve, (name, method, N), float(curve[N - 1]))

            # Label decodability for the linear-subspace methods at headline N.
            for method, basis_fn in [
                ("pca", lambda: _pca_basis(Xtr, mean_m, headline_N)),
                ("sae_c4_geo", lambda: _sae_greedy_basis(Xtr, mean_m, c4_dec, headline_N)),
                ("sae_mix_geo", lambda: _sae_greedy_basis(Xtr, mean_m, mix_dec, headline_N)),
            ]:
                B = basis_fn()
                Ztr, Zte = _subspace_rep(Xtr, Xte, mean_m, B)
                add(lab, (name, method, headline_N), label_r2(Ztr, ytr, Zte, yte))

            # Factored conditions.
            for cd in coord_dims:
                for linear in (True, False):
                    model, norm = factored[(cd, linear)]
                    fve, Ztr, Zte, _ = factored_eval(model, norm, Xtr, Xte, mean_m)
                    method = f"factored_{'lin' if linear else 'nl'}"
                    add(ve, (name, method, cd), fve)
                    if cd == headline_N:
                        add(lab, (name, method, cd),
                            label_r2(Ztr, ytr, Zte, yte))

    _report(manifolds, ve, lab, coord_dims, headline_N, seeds)
    _save(ve, lab, seeds, coord_dims, headline_N)
    _plot(manifolds, ve, maxN, coord_dims, seeds)


# ── Reporting ─────────────────────────────────────────────────────────────────

def _ms(vals):
    a = np.array([v for v in vals if v is not None and np.isfinite(v)])
    return (np.nan, np.nan) if len(a) == 0 else (a.mean(), a.std())


def _report(manifolds, ve, lab, coord_dims, N, seeds):
    methods = ["pca", "sae_c4_geo", "sae_c4_stat", "sae_mix_geo", "sae_mix_stat",
               "factored_lin", "factored_nl"]
    label = {"pca": "PCA (linear ceiling)", "sae_c4_geo": "SAE-C4 geometric",
             "sae_c4_stat": "SAE-C4 statistical", "sae_mix_geo": "SAE-mix geometric",
             "sae_mix_stat": "SAE-mix statistical",
             "factored_lin": "Factored LINEAR", "factored_nl": "Factored NONLINEAR"}
    names = [m for m in manifolds if (m, "pca", N) in ve]

    print(f"\n{'='*78}\nHELD-OUT SUBSPACE-CAPTURE VE at N={N}  (mean±sd over {len(seeds)} seeds)\n{'='*78}")
    header = f"{'method':22}" + "".join(f"{n[:9]:>11}" for n in names)
    print(header)
    for m in methods:
        row = f"{label[m]:22}"
        for n in names:
            mean, sd = _ms(ve.get((n, m, N), []))
            row += f"{('%.2f±%.2f' % (mean, sd)) if np.isfinite(mean) else '   n/a':>11}"
        print(row)

    print(f"\nCURVATURE DELTA (factored_nl − factored_lin) and (factored_nl − PCA) at N={N}:")
    for n in names:
        nl, _ = _ms(ve.get((n, "factored_nl", N), []))
        ln, _ = _ms(ve.get((n, "factored_lin", N), []))
        pc, _ = _ms(ve.get((n, "pca", N), []))
        print(f"  {n:12} nl−lin={nl-ln:+.3f}   nl−PCA={nl-pc:+.3f}")

    print(f"\n{'='*78}\nSECONDARY: held-out label R² at N={N} (linear / kNN)\n{'='*78}")
    print(f"{'method':22}" + "".join(f"{n[:9]:>13}" for n in names))
    for m in ["pca", "sae_c4_geo", "sae_mix_geo", "factored_lin", "factored_nl"]:
        row = f"{label[m]:22}"
        for n in names:
            pairs = lab.get((n, m, N), [])
            lin = _ms([p[0] for p in pairs]); kn = _ms([p[1] for p in pairs])
            row += (f"{('%.2f/%.2f' % (lin[0], kn[0])) if np.isfinite(lin[0]) else 'n/a':>13}")
        print(row)


def _save(ve, lab, seeds, coord_dims, N):
    out = dict(seeds=list(seeds), coord_dims=list(coord_dims), headline_N=N,
               ve={f"{k[0]}|{k[1]}|{k[2]}": v for k, v in ve.items()},
               label_r2={f"{k[0]}|{k[1]}|{k[2]}": v for k, v in lab.items()})
    path = RESULTS_DIR / "results.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved raw results -> {path}")


def _plot(manifolds, ve, maxN, coord_dims, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [m for m in manifolds if (m, "pca", 1) in ve]
    ncol = len(names)
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4), squeeze=False)
    Ns = np.arange(1, maxN + 1)
    line_methods = [("pca", "gray", "--", "PCA (ceiling)"),
                    ("sae_c4_geo", "tab:blue", "-", "SAE-C4 geom"),
                    ("sae_mix_geo", "tab:cyan", "-", "SAE-mix geom"),
                    ("sae_mix_stat", "magenta", ":", "SAE-mix stat")]
    for ax, n in zip(axes[0], names):
        for m, c, ls, lbl in line_methods:
            ys = [_ms(ve.get((n, m, N), []))[0] for N in Ns]
            ax.plot(Ns, ys, color=c, ls=ls, label=lbl, lw=1.5, marker="o", ms=3)
        for linear, c, mk in [(True, "tab:orange", "s"), (False, "tab:red", "D")]:
            method = f"factored_{'lin' if linear else 'nl'}"
            xs = [cd for cd in coord_dims if (n, method, cd) in ve]
            ys = [_ms(ve[(n, method, cd)])[0] for cd in xs]
            ax.scatter(xs, ys, color=c, marker=mk, s=60, zorder=5,
                       label=f"Factored {'lin' if linear else 'NL'}")
        ax.set_title(n); ax.set_xlabel("N (dims)"); ax.set_ylim(-0.05, 1.02)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("Held-out variance explained")
    axes[0][-1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Held-out subspace capture: curved charts vs linear dictionary")
    fig.tight_layout()
    path = RESULTS_DIR / "ve_curves.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"Saved plot -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifolds", nargs="+", default=DEFAULT_MANIFOLDS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--coord-dims", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--max-n", type=int, default=8)
    ap.add_argument("--headline-n", type=int, default=3)
    ap.add_argument("--c4-sae", default=str(CACHE_DIR / "sae_4608_k32.pt"))
    a = ap.parse_args()
    run(a.manifolds, a.seeds, a.coord_dims, a.max_n, a.c4_sae, a.headline_n)


if __name__ == "__main__":
    main()
