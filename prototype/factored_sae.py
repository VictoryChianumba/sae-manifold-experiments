"""
Factored "which-manifold + where-on-it" SAE — a toy manifold-native architecture.

Motivation (see ../REPRODUCTION.md "The idea, in one analogy"): a standard SAE
represents a curved concept manifold with a dictionary of *straight* atoms, so it
shatters/dilutes the geometry. This prototype replaces the dictionary-of-rays with
an **atlas of charts**:

  - a soft **router**  r(x) -> distribution over M charts   ("which manifold")
  - per-chart **coordinate**  z_m(x) in R^coord_dim          ("where on it")
  - per-chart **nonlinear decoder**  g_m(z_m) -> activation   (a *curved* chart)

  reconstruction  x_hat = bias + sum_m  a_m * g_m(z_m)

Because each chart decoder is a small MLP, a *single* chart can bend along a curved
manifold that a linear SAE would need many atoms to (badly) tile. The router is the
learned, unsupervised analogue of the paper's post-hoc feature clustering — here it
is baked into the architecture. (Conceptually: router = MoE-style expert selection,
charts = local coordinate maps = an atlas.)

This is a research toy, not a scalable SAE: it trains on the small concept-manifold
activations we already cached, and is meant to *demonstrate the idea* and let us
measure (a) does the router discover the manifolds unsupervised, and (b) does each
chart's coordinate recover the manifold's geometry (track its ground-truth label).

Run:
  SAE_DEVICE=cpu SAE_D_MODEL=576 uv run python prototype/factored_sae.py
  SAE_DEVICE=cpu SAE_D_MODEL=576 uv run python prototype/factored_sae.py --linear-charts  # ablation
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import data.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression

from data import load_manifold_data, CACHE_DIR

# Manifold -> primary continuous label used ONLY to score the learned coordinate.
PRIMARY_LABEL = {
    "years": "year", "age": "age", "temperature": "fahrenheit",
    "days": "time_idx", "colors": "hue", "geography": "latitude",
}


# ── Data: concatenate cached concept manifolds into one labeled mixture ────────

def load_mixture(manifolds):
    Xs, mids, labels, names = [], [], [], []
    for mi, name in enumerate(manifolds):
        d = load_manifold_data(name)
        if d is None:
            print(f"  skip {name} (not cached)")
            continue
        X = d["activations"].float().numpy()
        key = PRIMARY_LABEL.get(name)
        y = np.array([float(l.get(key, np.nan)) for l in d["labels"]]) \
            if key else np.full(len(X), np.nan)
        Xs.append(X); mids.append(np.full(len(X), len(names)))
        labels.append(y); names.append(name)
        print(f"  + {name}: {len(X)} pts, label='{key}'")
    X = np.concatenate(Xs).astype(np.float32)
    mid = np.concatenate(mids).astype(int)
    y = np.concatenate(labels).astype(np.float32)
    return X, mid, y, names


# ── Model ──────────────────────────────────────────────────────────────────--

class FactoredSAE(nn.Module):
    def __init__(self, d_in, n_charts=12, coord_dim=3, hidden=64,
                 linear_charts=False, hard_routing=False):
        super().__init__()
        self.n_charts, self.coord_dim = n_charts, coord_dim
        self.hard_routing = hard_routing
        self.router = nn.Linear(d_in, n_charts)
        self.coord_enc = nn.Sequential(
            nn.Linear(d_in, 128), nn.GELU(),
            nn.Linear(128, n_charts * coord_dim))
        if linear_charts:
            self.charts = nn.ModuleList(
                [nn.Linear(coord_dim, d_in) for _ in range(n_charts)])
        else:
            self.charts = nn.ModuleList([
                nn.Sequential(nn.Linear(coord_dim, hidden), nn.GELU(),
                              nn.Linear(hidden, hidden), nn.GELU(),
                              nn.Linear(hidden, d_in))
                for _ in range(n_charts)])
        self.bias = nn.Parameter(torch.zeros(d_in))

    def forward(self, x, temp=1.0):
        """Returns (recon, a, coords).

        With ``hard_routing=False`` (default): ``a`` is the softmax over charts,
        reconstruction is a soft mixture ``sum_m a_m g_m(z_m) + bias``.

        With ``hard_routing=True``: ``a`` is one-hot in the forward pass
        (straight-through estimator: ``a_hard - a_soft.detach() + a_soft`` lets
        the soft gradient still update the router), so the reconstruction is
        exactly ``g_{argmax}(z_{argmax}) + bias``.  This makes the
        "dominant-chart-only" eval (see ``fair_comparison.factored_eval``) be
        the same function the model was trained on, instead of forcing it into
        an out-of-distribution regime it never saw.  Caller is responsible for
        re-deriving the soft routing (``F.softmax(self.router(x))``) if it
        needs gradients on entropy/balance terms — the one-hot ``a`` has
        entropy 0 and provides no gradient signal for them."""
        a_soft = F.softmax(self.router(x) / temp, dim=-1)        # [B, M]
        if self.hard_routing:
            a_hard = F.one_hot(a_soft.argmax(-1), self.n_charts).to(a_soft.dtype)
            a = a_hard - a_soft.detach() + a_soft
        else:
            a = a_soft
        coords = self.coord_enc(x).view(-1, self.n_charts, self.coord_dim)
        recons = torch.stack([self.charts[m](coords[:, m])
                              for m in range(self.n_charts)], dim=1)  # [B,M,d]
        recon = (a.unsqueeze(-1) * recons).sum(1) + self.bias
        return recon, a, coords


# ── Train ──────────────────────────────────────────────────────────────────--

def train(X, n_charts, coord_dim, linear_charts, epochs, batch, lr,
          lam_sparse, lam_balance, device):
    mean = X.mean(0, keepdims=True)
    std = X.std() + 1e-6
    Xn = torch.tensor((X - mean) / std, device=device)
    N, d_in = Xn.shape

    model = FactoredSAE(d_in, n_charts, coord_dim,
                        linear_charts=linear_charts).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    eps = 1e-9
    for ep in range(epochs):
        perm = torch.randperm(N, device=device)
        tot = 0.0
        for i in range(0, N, batch):
            xb = Xn[perm[i:i + batch]]
            recon, a, _ = model(xb)
            mse = ((recon - xb) ** 2).sum(-1).mean()
            # per-point peaky (low entropy) + across-batch balanced usage.
            ent_point = -(a * (a + eps).log()).sum(-1).mean()
            usage = a.mean(0)
            ent_usage = -(usage * (usage + eps).log()).sum()
            loss = mse + lam_sparse * ent_point - lam_balance * ent_usage
            opt.zero_grad(); loss.backward(); opt.step()
            tot += mse.item() * len(xb)
        if ep % max(1, epochs // 8) == 0 or ep == epochs - 1:
            ve = 1 - tot / N / (Xn.var(0).sum().item())
            print(f"  epoch {ep+1:>3}/{epochs}  recon_mse={tot/N:8.3f}  "
                  f"frac_var_explained={ve:.3f}")
    return model, Xn


# ── Evaluate: discovery (router) + geometry (coordinate recovery) ─────────────

@torch.no_grad()
def evaluate(model, Xn, mid, y, names):
    recon, a, coords = model(Xn)
    ve = 1 - ((recon - Xn) ** 2).sum(-1).mean().item() / Xn.var(0).sum().item()
    assign = a.argmax(-1).cpu().numpy()
    coords = coords.cpu().numpy()

    print(f"\n  overall variance explained: {ve:.3f}")
    print(f"\n  {'manifold':12} {'dom.chart':>9} {'purity':>7} "
          f"{'coord→label R²':>15}")
    for mi, name in enumerate(names):
        sel = mid == mi
        ch = assign[sel]
        dom = np.bincount(ch, minlength=model.n_charts).argmax()
        purity = (ch == dom).mean()
        # Coordinate recovery: predict the label from this chart's coordinate.
        yy = y[sel]
        ok = ~np.isnan(yy)
        r2 = np.nan
        if ok.sum() > 5 and not np.all(yy[ok] == yy[ok][0]):
            Z = coords[sel][:, dom, :][ok]
            r2 = LinearRegression().fit(Z, yy[ok]).score(Z, yy[ok])
        print(f"  {name:12} {dom:>9} {purity:>7.2f} "
              f"{('%.3f' % r2) if not np.isnan(r2) else '   n/a':>15}")
    # How many charts does each manifold spread across (lower = cleaner)?
    return ve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifolds", nargs="+",
                    default=["years", "age", "temperature", "days",
                             "colors", "geography"])
    ap.add_argument("--n-charts", type=int, default=12)
    ap.add_argument("--coord-dim", type=int, default=3)
    ap.add_argument("--linear-charts", action="store_true",
                    help="ablation: linear chart decoders (rays, like an SAE)")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lam-sparse", type=float, default=0.02)
    ap.add_argument("--lam-balance", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    print(f"Loading mixture ({'LINEAR charts' if a.linear_charts else 'NONLINEAR charts'}):")
    X, mid, y, names = load_mixture(a.manifolds)
    print(f"  total: {X.shape[0]} points, d_in={X.shape[1]}, "
          f"{len(names)} true manifolds, {a.n_charts} charts")

    model, Xn = train(X, a.n_charts, a.coord_dim, a.linear_charts, a.epochs,
                      a.batch, a.lr, a.lam_sparse, a.lam_balance, device="cpu")
    evaluate(model, Xn, mid, y, names)


if __name__ == "__main__":
    main()
