# SAE Manifold Evaluation

Code accompanying **"Do Sparse Autoencoders Capture Concept Manifolds?"**
([arxiv.org/abs/2604.28119](https://arxiv.org/abs/2604.28119))
— *Bhalla, Fel, Rager, Feucht, Haklay, Wurgaft, Boppana, Kowal, Shyam,
Lewis, McGrath, Merullo, Geiger, Lubana*.

This repository releases two of the evaluations from the paper:

1. **Subspace capture** (`subspace_capture.py`) — measures whether an SAE's
   features span a fixed low-dimensional subspace that contains a given
   concept manifold, or whether they *shatter* into many disjoint atoms
   across the manifold.
2. **Unsupervised feature clustering** (`unsupervised_clustering.py`) —
   groups SAE features by their pairwise relationships (decoder cosine,
   co-activation, correlation, mutual information, inverse Ising couplings),
   then runs Leiden / spectral community detection.

The implementation targets Llama-3.1-8B by default, but the model
and layer can be overridden in `data.py`. SAE encoding is done with a
minimal `BatchTopKSAE` class in `saes.py`; if you trained with a different
architecture (TopK, JumpReLU, Matryoshka, …) you can swap in your own class
— the rest of the code only depends on three methods (`encode`, `decode`,
decoder weight access).

## Quickstart

### 1. Extract manifold activations

Each "manifold" is a set of prompts whose ground-truth structure is known.
We push them through the LM once and cache the last-token hidden state.

```bash
uv run data.py                    # extract all manifolds
uv run data.py --manifold colors  # or just one
```

Cached activations live in `cache/{manifold}.pt`.

### 2. Run subspace-capture experiments

```bash
# Variance-explained curves: geometric vs. statistical greedy + PCA / random
# baselines (Fig. 4 of the paper). One PDF per manifold.
uv run subspace_capture.py plot --sae /path/to/sae.pt --k 64

# Feature tuning curves: label (x) vs. activation (y) for the top features
# from the statistical-reconstruction greedy (Fig. 5 / years_recon).
uv run subspace_capture.py tuning --sae /path/to/sae.pt --k 64 --manifold years
```

The `--k` flag is the BatchTopK sparsity. If your checkpoint stores `k` in
its model config you can omit it.

### 3. Run unsupervised clustering

```bash
# (a) Extract background activations from a streaming text dataset.
uv run background.py --n-tokens 500000

# (b) Compute all five similarity matrices (cosine, coactivation,
#     correlation, MI, Ising).
uv run unsupervised_clustering.py matrices \
    --sae /path/to/sae.pt --k 64 --n-tokens 500000

# (c) Cluster features. Ising is recommended (Sec. 5 of the paper).
uv run unsupervised_clustering.py cluster --sae /path/to/sae.pt --matrix ising --method leiden

# Optional: refit Ising with a tuned alpha or on a feature subset.
uv run unsupervised_clustering.py ising --sae /path/to/sae.pt --k 64 --alpha 0.005

# Optional: heatmap of a specific feature subset across all matrices.
uv run unsupervised_clustering.py heatmap --sae /path/to/sae.pt --features 100 200 300 400
```

## Manifolds shipped

| Manifold | Structure | Source |
|---|---|---|
| `colors` | paraboloid (hue, lightness, saturation) | `burkelibbey/colors` |
| `temperature` | line | synthetic |
| `age` | line | synthetic |
| `geography` | hierarchical tree (lat/long + continent) | `jamescalam/world-cities-geo` |
| `days` | 2D grid (7 days × 8 times of day) | synthetic |
| `years` | helix (1800–1999) | synthetic |
| `formality` | line | `osyvokon/pavlick-formality-scores` |
| `sent_length` | line | `wikitext` |

## SAE format

`saes.load_sae(path, ...)` accepts either a raw `state_dict` (as saved by
`torch.save(sae.state_dict(), path)`) or a dict with `state_dict` and
`model_config` keys. It expects an encoder linear, a decoder linear, and
an optional per-feature `threshold` buffer for JumpReLU-style inference.

If you trained with a different architecture, replace `BatchTopKSAE` with
your class — `subspace_capture.py` and `unsupervised_clustering.py` only
use `sae.encode`, `sae.decode`, `get_decoder(sae)`, and `get_decoder_bias(sae)`.

## Citation

```bibtex
@article{bhalla2026sae,
  title   = {Do Sparse Autoencoders Capture Concept Manifolds?},
  author  = {Bhalla, Usha and Fel, Thomas and Rager, Can and Feucht, Sheridan
             and Haklay, Tal and Wurgaft, Daniel and Boppana, Siddharth
             and Kowal, Matthew and Shyam, Vasudev and Lewis, Owen
             and McGrath, Thomas and Merullo, Jack and Geiger, Atticus
             and Lubana, Ekdeep Singh},
  journal = {arXiv preprint arXiv:2604.28119},
  year    = {2026},
}
```

## License

MIT. See `LICENSE`.
