"""
Minimal BatchTopK SAE *training* loop.

The upstream repo ships an inference-only ``BatchTopKSAE`` (see ``saes.py``)
and assumes you bring your own trained checkpoint. This script is the missing
training half, kept deliberately small, so the whole pipeline can be run
end-to-end on modest hardware against a small model.

It trains on the background activations produced by ``background.py`` and
saves a checkpoint in the ``{"state_dict", "model_config"}`` layout that
``saes.load_sae`` understands.

Training uses BatchTopK (top ``k*batch`` activations across the whole batch);
inference in ``saes.py`` uses a per-sample top-k, which is the standard way
BatchTopK SAEs are evaluated (Bussmann, Leask & Nanda, 2024).

Usage:
  uv run train_sae.py --n-tokens 200000 --expansion-factor 8 --k 32 --epochs 40
  # -> writes cache/sae_{d_sae}_k{k}.pt
"""
import argparse
import math
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from data import D_MODEL, DEVICE, CACHE_DIR
from background import load_background
from saes import BatchTopKSAE


def batch_topk(pre, k):
    """Keep the top ``k * batch`` pre-activations across the whole batch.

    ``pre`` is ``[B, d_sae]`` post-ReLU. Returns a same-shape tensor with all
    but the top ``k*B`` entries zeroed (BatchTopK; Bussmann et al. 2024).
    """
    B = pre.shape[0]
    n_keep = max(1, int(k * B))
    flat = pre.flatten()
    n_keep = min(n_keep, flat.numel())
    thresh = torch.topk(flat, n_keep, sorted=False).values.min()
    return torch.where(pre >= thresh, pre, torch.zeros_like(pre))


@torch.no_grad()
def _unit_norm_decoder_(sae):
    """Renormalise decoder columns (one per feature) to unit L2 norm."""
    W = sae.decoder.weight  # [d_in, d_sae]
    norms = W.norm(dim=0, keepdim=True).clamp_min(1e-8)
    W.div_(norms)


def train(n_tokens, expansion_factor, k, epochs, batch_size, lr,
          aux_coef, dead_steps, out_path=None, max_train=400_000):
    acts_mmap = load_background(n_tokens)  # [N, d_in] float32 memmap
    N = min(acts_mmap.shape[0], max_train)
    d_in = acts_mmap.shape[1]
    d_sae = d_in * expansion_factor
    print(f"Training BatchTopK SAE: N={N} d_in={d_in} d_sae={d_sae} "
          f"k={k} on {DEVICE}")

    X = torch.from_numpy(np.ascontiguousarray(acts_mmap[:N])).float()
    # Centre inputs; the data mean is folded into the decoder bias so the model
    # only has to reconstruct deviations from it.
    data_mean = X.mean(0)
    X = X.to(DEVICE)
    data_mean = data_mean.to(DEVICE)

    sae = BatchTopKSAE(d_in=d_in, d_sae=d_sae, k=k, device=DEVICE)
    with torch.no_grad():
        sae.decoder.bias.copy_(data_mean)
        # Initialise encoder as the decoder transpose (a common, stable start).
        _unit_norm_decoder_(sae)
        sae.encoder.weight.copy_(sae.decoder.weight.t())
    for p in sae.parameters():
        p.requires_grad_(True)

    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    total_var = ((X - data_mean) ** 2).sum(1).mean().item()

    # Track steps since each feature last fired, to drive the auxiliary loss
    # that revives dead features.
    last_fired = torch.zeros(d_sae, device=DEVICE)
    step = 0
    n_batches = math.ceil(N / batch_size)

    for epoch in range(epochs):
        perm = torch.randperm(N, device=DEVICE)
        running, fired = 0.0, torch.zeros(d_sae, device=DEVICE)
        for bi in range(n_batches):
            idx = perm[bi * batch_size:(bi + 1) * batch_size]
            x = X[idx]
            pre = torch.relu(sae.encoder(x))
            z = batch_topk(pre, k)
            recon = sae.decode(z)
            mse = ((recon - x) ** 2).sum(1).mean()

            loss = mse
            # AuxK: let the currently-dead features reconstruct the residual,
            # which pulls them back to life (Gao et al. 2024).
            dead = last_fired > dead_steps
            if aux_coef > 0 and dead.any():
                resid = x - recon
                pre_dead = pre.clone()
                pre_dead[:, ~dead] = 0.0
                k_aux = min(int(dead.sum().item()), 2 * k)
                if k_aux > 0:
                    top = pre_dead.topk(k_aux, dim=-1)
                    z_aux = torch.zeros_like(pre_dead)
                    z_aux.scatter_(-1, top.indices, top.values)
                    aux_recon = sae.decoder(z_aux) - sae.decoder.bias
                    loss = loss + aux_coef * ((aux_recon - resid) ** 2).sum(1).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                _unit_norm_decoder_(sae)

            with torch.no_grad():
                active = (z > 0).any(0)
                last_fired += 1
                last_fired[active] = 0
                fired += (z > 0).float().sum(0)
            running += mse.item()
            step += 1

        ev = 1.0 - (running / n_batches) / max(total_var, 1e-8)
        n_dead = int((last_fired > dead_steps).sum().item())
        n_alive = int((fired > 0).sum().item())
        print(f"  epoch {epoch+1:>3}/{epochs}  recon_mse={running/n_batches:.3f}  "
              f"frac_var_explained={ev:.3f}  alive={n_alive}/{d_sae}  dead={n_dead}")

    sae.eval()
    out_path = Path(out_path) if out_path else CACHE_DIR / f"sae_{d_sae}_k{k}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": {kk: v.cpu() for kk, v in sae.state_dict().items()},
         "model_config": {"d_in": d_in, "d_sae": d_sae, "k": k}},
        out_path,
    )
    print(f"Saved SAE -> {out_path}")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-tokens", type=int, default=200_000,
                   help="Background cache size to load (matches background.py)")
    p.add_argument("--expansion-factor", type=int, default=8)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--aux-coef", type=float, default=1.0 / 32)
    p.add_argument("--dead-steps", type=int, default=200,
                   help="Steps without firing before a feature counts as dead")
    p.add_argument("--max-train", type=int, default=400_000)
    p.add_argument("--out", type=str, default=None)
    a = p.parse_args()
    train(a.n_tokens, a.expansion_factor, a.k, a.epochs, a.batch_size,
          a.lr, a.aux_coef, a.dead_steps, out_path=a.out, max_train=a.max_train)
