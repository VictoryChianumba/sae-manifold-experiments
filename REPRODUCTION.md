# Local reproduction — "Do SAEs Capture Concept Manifolds?" on an 8 GB M1

This documents a **scaled-down, fully local** reproduction of the Goodfire paper
[*Do Sparse Autoencoders Capture Concept Manifolds?*](https://arxiv.org/abs/2604.28119)
([blog](https://www.goodfire.ai/research/can-saes-capture-neural-geometry)),
run end-to-end on an Apple M1 / 8 GB laptop.

## The idea, in one analogy

Think of the model's internal representation of a concept (say *temperature*, or
*years*) as a **globe** — a smooth, *curved* surface. Each **SAE feature is a flat
local map of one county** (Somerset). It's locally accurate, but it's flat, and the
thing it's a map of is curved.

Two errors follow from reading features one at a time — the way SAEs are normally
used — and the paper is about both:

- **Over-generalization:** you look at the Somerset map and draw conclusions about
  all of Europe. (Reading one feature and treating it as the whole concept.)
- **False multiplicity:** you see the maps of Somerset, Devon, and Cornwall and
  conclude there are three unrelated places — when they're one country.
  (One curved concept gets shattered across many features that look distinct.)

The deeper point is about **curvature, not zoom level**: no single flat county map
ever reveals that the Earth is a *sphere*. You only recover the global curvature by
**stitching many overlapping local maps together** — which in differential geometry
is exactly an *atlas* of a manifold. That is what an SAE is: a pile of flat local
charts. The geometry is recoverable, but only by reassembling the charts.

The catch the paper actually solves: the model doesn't hand you county/country
labels. You have to infer *which counties belong to the same country* purely from
**how they behave together** — which features fire on the same inputs — like
reconstructing national borders from trade and commuting patterns with no map. The
**unsupervised feature clustering** step *is* that inference; the per-cluster
subspace is the recovered country/continent.

So the title answer is "it depends on the unit": as individual counties, no — SAEs
mislead you about the globe. As clustered, collectively-analyzed regions, the
geometry comes back. See `manifold_viz.py` for the literal version of this picture
(the curved manifold + the straight SAE chords cutting across it).

## Why a scaled-down version

The released [`goodfire-ai/sae-manifold`](https://github.com/goodfire-ai/sae-manifold)
repo ships **two evaluations** (`subspace_capture.py`, `unsupervised_clustering.py`)
but **not** the things they depend on:

1. **Llama-3.1-8B activations** — `data.py` runs the full 8B model via `nnsight`.
   An 8B model needs ~16 GB just for fp16 weights, so it does **not** fit in 8 GB RAM.
2. **A trained SAE checkpoint** — `saes.py` is inference-only; no weights or
   training code are included.

So this reproduction keeps the paper's *pipeline* (observed model activations →
trained SAE → subspace-capture + clustering metrics) but swaps the 8B model for a
small **Llama-architecture** model that fits locally.

### Substitutions vs. the paper

| | Paper | This reproduction |
|---|---|---|
| Model | Llama-3.1-8B (d=4096) | **SmolLM2-135M** (`LlamaForCausalLM`, d=576) |
| Layer | residual stream, layer 19 | layer 19 (valid: 135M has 30 layers) |
| SAE training data | 500M tokens of The Pile | **40k tokens** of C4 (streaming) |
| SAE | 5 architectures, exp 8/16, k 64/128/256 | **BatchTopK**, exp 8 (d_sae=4608), **k=32** |
| Manifolds | full set (incl. dataset-backed) | `age`, `temperature`, `days`, `years` (no downloads) |

SmolLM2-135M is also `LlamaForCausalLM`, so `data.py`'s `model.model.layers[LAYER]`
access path is **unchanged** — only the model name / dims differ.

## Changes made to the repo

- **`data.py`** — `MODEL_NAME`, `LAYER`, `D_MODEL`, `DEVICE` are now read from
  environment variables (`SAE_MODEL_NAME`, `SAE_LAYER`, `SAE_D_MODEL`, `SAE_DEVICE`),
  defaulting to the paper's Llama-3.1-8B values. `load_llm()` places the model on
  CPU/MPS explicitly instead of `device_map="auto"` (which assumes a CUDA planner).
  `background.py` and `subspace_capture.py` inherit these via their imports.
- **`train_sae.py`** (new) — the missing training half. Trains a BatchTopK SAE on
  the background activations and saves a checkpoint compatible with `saes.load_sae`
  (`{"state_dict", "model_config"}`). Train uses BatchTopK; inference uses per-sample
  top-k, matching the repo's convention.

## How to re-run

```bash
uv sync     # installs torch, nnsight, transformers, datasets, igraph, leidenalg, ...

# Small-model config (used by every step below)
export SAE_MODEL_NAME=HuggingFaceTB/SmolLM2-135M SAE_LAYER=19 SAE_D_MODEL=576

# 1. Observed manifold activations (last-token hidden states)
uv run python data.py --manifold age temperature days years

# 2. Background corpus for SAE training (~3 min on MPS)
uv run python background.py --n-tokens 40000

# 3. Train the SAE  (CPU is fastest/most reliable here — MPS topk is slow)
SAE_DEVICE=cpu uv run python train_sae.py --n-tokens 40000 \
    --expansion-factor 8 --k 32 --epochs 18 --batch-size 4096 --lr 7e-4
# -> cache/sae_4608_k32.pt   (final ~75% variance explained, 4522/4608 alive)

# 4. Subspace-capture curves (paper Fig. 4) + tuning curves (Fig. 5)
SAE_DEVICE=cpu uv run python subspace_capture.py plot \
    --sae cache/sae_4608_k32.pt --k 32 --d-in 576 \
    --manifold age temperature days years
SAE_DEVICE=cpu uv run python subspace_capture.py tuning \
    --sae cache/sae_4608_k32.pt --k 32 --d-in 576 --manifold years age temperature
# -> cache/subspace_capture/*.pdf

# 5. Unsupervised feature clustering
SAE_DEVICE=cpu uv run python unsupervised_clustering.py matrices \
    --sae cache/sae_4608_k32.pt --k 32 --d-in 576 --n-tokens 40000 \
    --matrices cosine coactivation correlation
SAE_DEVICE=cpu uv run python unsupervised_clustering.py cluster \
    --sae cache/sae_4608_k32.pt --matrix correlation --method leiden --percentile 99
```

## Result — the paper's core finding reproduces

Variance of the manifold activations explained, as a function of how many
directions you are allowed (PCA = optimal linear basis; geo-SAE = greedy over SAE
decoder directions; stat-SAE = greedy over the SAE's actual codes):

```
== years / helix (N=199) ==
   k=        4     8    16    32    64
   PCA      51%   77%   94%   98%  100%      <- compact linear subspace exists
   geo-SAE  12%   20%   30%   42%   57%
   stat-SAE  8%   11%   14%   17%   17%      <- SAE codes plateau at 17%

== age / line (N=99) ==
   k=        4     8    16    32    64
   PCA      84%   92%   96%   99%  100%
   geo-SAE   7%   12%   20%   31%   47%
   stat-SAE  3%    5%    6%    6%    6%
```

The manifold lives in a ~12–16 dimensional **linear** subspace (PCA hits 90% fast),
but the SAE spreads it across **many partially-shared features** — even 64 greedy
SAE features don't span it, and the *actual codes* (stat-SAE) plateau far below.
This is exactly the paper's **"dilution / shattering"** regime: SAE features tile
the curved manifold with localized atoms rather than capturing it with a small,
shared coordinate system.

Leiden clustering on the feature-correlation / co-activation graphs groups the
4,608 features into ~8–11 communities (`cache/clusters/*.json`), demonstrating the
unsupervised side of the pipeline.

## 3D manifold visualizations (`manifold_viz.py`, new)

Recreates the article's 3D manifold figures (ORDINALS / COMPOSERS / SCOTUS / the
helix). Method: PCA the manifold activations to 3D → scatter colored by the
ground-truth label → gray floor shadow as a depth cue → overlay the SAE decoder
directions the *statistical* greedy selects (the straight chords across the curve).

```bash
SAE_DEVICE=cpu SAE_D_MODEL=576 uv run python manifold_viz.py \
    --manifold years age temperature days colors geography \
    --sae cache/sae_4608_k32.pt --k 32 --d-in 576 --overlay-features 6
# -> cache/manifold_viz/*_manifold3d.png
```

The straight gray decoder chords visibly cut *across* the curved, label-colored
manifold rather than following it — the geometric picture of dilution/shattering.
(Because this is a 135M model, 3 PCs explain only ~42–81% of the variance, so the
shapes are noisier than Llama-8B's; the method and picture reproduce, the crispness
scales with model size.)

## Sparsity sweep (`compare_sparsity.py`, new)

Trains SAEs at k=16, 32, 64 (the paper's central knob) and overlays their
statistical-reconstruction curves per manifold against PCA.

```bash
# train the extra sparsities (k=32 already exists)
SAE_DEVICE=cpu uv run python train_sae.py --n-tokens 40000 --expansion-factor 8 --k 16 --epochs 18 --batch-size 4096 --lr 7e-4
SAE_DEVICE=cpu uv run python train_sae.py --n-tokens 40000 --expansion-factor 8 --k 64 --epochs 18 --batch-size 4096 --lr 7e-4
SAE_DEVICE=cpu SAE_D_MODEL=576 uv run python compare_sparsity.py \
    --saes cache/sae_4608_k16.pt cache/sae_4608_k32.pt cache/sae_4608_k64.pt \
    --ks 16 32 64 --manifold years age temperature colors
# -> cache/subspace_capture/*_sparsity_sweep.pdf
```

Finding: on the years manifold all three sparsities plateau at ~17–20% variance
explained — **changing k barely moves manifold capture**, all stuck far below PCA.
Dilution looks fairly intrinsic here, not just a sparsity artifact.

## More manifolds

Beyond the four no-download manifolds, this run also extracts the dataset-backed
**colors** (paraboloid, 1.8k pts, colored by hue) and **geography** (hierarchical
tree, 3k world cities, colored by latitude) — exercising the method on 2D and
hierarchical geometry, not just 1D lines. Their subspace-capture and tuning curves
are in `cache/subspace_capture/`.

## Project location

Moved from `~/Desktop/...` to **`~/Documents/goodfire`** on 2026-05-26: macOS TCC
(privacy) was denying access to the Desktop folder mid-session (`~/Desktop` is
privacy-protected; `~/Documents` is not). Rebuild the venv there with `uv sync`.

## Known limitation on macOS

The **Ising** similarity matrix uses a `fork`-based multiprocessing pool of
liblinear fits, which deadlocks/crash-loops on macOS (fork + threaded Accelerate
BLAS). Run it with single-threaded BLAS env vars
(`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`) and fewer
jobs, or compute it on Linux. The cosine / coactivation / correlation matrices and
their Leiden clustering run fine and already exercise the clustering pipeline.
</content>
