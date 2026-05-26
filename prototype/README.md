# Prototype: a factored "which-manifold + where-on-it" SAE

A toy of a **manifold-native** alternative to the standard SAE, motivated by the
finding in `../REPRODUCTION.md` that a dictionary of straight atoms shatters/dilutes
curved concept manifolds.

## Idea

Replace the dictionary-of-rays with an **atlas of charts** (`factored_sae.py`):

- a soft **router** `r(x)` → distribution over `M` charts — *which manifold*
- per-chart **coordinate** `z_m(x) ∈ R^coord_dim` — *where on it*
- per-chart **nonlinear decoder** `g_m(z_m)` → activation — a *curved* chart

```
x_hat = bias + Σ_m  a_m · g_m(z_m)
```

The router is the learned, **unsupervised** analogue of the paper's post-hoc feature
clustering — here it is baked into the architecture. (Router = MoE-style expert
selection; charts = local coordinate maps = an atlas. The two ideas unify.)

## Run

```bash
SAE_DEVICE=cpu SAE_D_MODEL=576 uv run python prototype/factored_sae.py
# ablations:
SAE_DEVICE=cpu SAE_D_MODEL=576 uv run python prototype/factored_sae.py --linear-charts
SAE_DEVICE=cpu SAE_D_MODEL=576 uv run python prototype/factored_sae.py --linear-charts --coord-dim 1
```

Trains in ~1–2 min on CPU on the concept-manifold activations already in `cache/`.

## What it measures

- **Discovery** — does the router separate the manifolds unsupervised? (`purity`:
  fraction of a manifold's points landing on its dominant chart)
- **Geometry** — does each chart's coordinate *parameterize* the manifold?
  (`coord→label R²`: linear regression from the chart's coordinate to the manifold's
  ground-truth label, e.g. year/age/temperature)

## Result — the ablation ladder

Six manifolds (years, age, temperature, days, colors, geography), 12 charts,
SmolLM2-135M activations:

| chart type (per cluster) | overall recon VE | age R² | temp R² | years R² |
|---|---|---|---|---|
| **1 ray** (`--linear-charts --coord-dim 1`, SAE-like) | 0.978 | 0.08 | 0.01 | 0.04 |
| **3-D subspace** (`--linear-charts`) | 0.985 | 0.72 | 0.78 | 0.50 |
| **3-D curved chart** (default, nonlinear) | 0.996 | 0.95 | 0.86 | 0.42 |

Two takeaways, both echoing the paper:

1. **Reconstruction ≠ captured geometry.** The ray model reconstructs at 97.8%
   variance yet its coordinate recovers nothing (R²≈0) — the dilution/shattering
   failure reproduced inside this architecture.
2. **The lever is structural.** Subspace-per-cluster (baking in the clustering)
   takes coordinate-recovery from ~0 to ~0.7–0.8; **curvature** (nonlinear charts)
   adds the final lift on the genuinely curved manifolds.

## Caveats

- **Toy scale**: SmolLM2-135M activations; ~5.4k points; results are noisy
  (e.g. `years` splits across charts; nonlinear vs subspace R² ordering wobbles).
- **Cyclic-label artifact**: `colors` (hue) and `days` (time-of-day) wrap around, so
  *linear* regression on the raw label scores ≈0 even when the coordinate is good.
  A cyclic-aware score (predict sin/cos) would fix this — not yet implemented.
- Not a scalable SAE: no sparsity in the SAE sense, trained on tiny curated data.

## Possible next steps

- Cyclic-aware coordinate scoring; a viz of one chart's coordinate vs. its label.
- Group-sparse selection (top-k charts) instead of soft softmax.
- Train on background activations and see whether charts discover manifolds in the
  wild (not just in the curated mixture).
- Factor the code explicitly into discrete selector + continuous coordinate and
  compare to a matched standard SAE on the paper's subspace-capture metric.
