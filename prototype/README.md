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

## The fair comparison (`fair_comparison.py`)

The ablation ladder above is *suggestive but confounded* (coord-dim capacity,
in-sample R², no real-SAE baseline, one seed). `fair_comparison.py` is the
controlled redo — everything **held out**, scored in **raw activation units**
against a single per-manifold denominator, **3 seeds**:

- **Primary metric = the paper's own subspace-capture VE(N)**, made
  architecture-agnostic: for each manifold, the fraction of its *test* activation
  variance explained by an **N-dimensional per-point code**. PCA-N is the optimal
  linear ceiling; a linear dictionary can at best reach it; only a *curved* decoder
  can exceed it.
- **Curvature is the only free variable** between the two factored conditions
  (linear vs nonlinear charts at identical `coord_dim`, `n_charts`, training).
- **Two real SAE baselines**: the off-the-shelf C4-trained SAE, and a standard
  BatchTopK SAE *retrained on the same mixture train split* (isolates architecture
  from training data).

```bash
SAE_DEVICE=cpu SAE_D_MODEL=576 uv run python prototype/fair_comparison.py \
    --seeds 0 1 2 --coord-dims 2 3
# -> cache/fair_comparison/{results.json, ve_curves.png, run.log}
```

### Result — held-out VE at N=3 (mean ± sd over 3 seeds)

| method | years | age | temperature | colors | geography |
|---|---|---|---|---|---|
| PCA (linear ceiling) | 0.39 | **0.79** | **0.79** | 0.74 | 0.52 |
| SAE-C4 geometric | 0.09 | 0.06 | 0.09 | 0.09 | 0.04 |
| SAE-mix geometric | 0.10 | 0.27 | 0.50 | 0.53 | 0.20 |
| SAE-mix statistical | 0.05 | 0.03 | 0.38 | 0.37 | 0.09 |
| Factored **LINEAR** | 0.09 | −0.07 | 0.05 | 0.57 | 0.12 |
| Factored **NONLINEAR** | **0.84** | 0.29 | 0.68 | **0.80** | **0.76** |

Curvature delta (`nl − lin`) is large and positive everywhere: years +0.75,
geography +0.65, temperature +0.63, age +0.36, colors +0.23 — robust across
seeds and at `coord_dim = 2` as well.

**The honest verdict — a split decision, leaning positive:**

1. **Curvature is real and decisive at matched dimensionality.** The nonlinear
   charts beat the linear charts at the *same* `coord_dim` on every manifold, by
   margins many× the seed spread. This is the cleanest possible isolation of the
   paper's thesis: the failure is the *straightness* of the atoms, not the
   clustering or the parameter count (the linear-chart control has both).
2. **On genuinely curved manifolds, curvature beats even the linear ceiling.**
   Factored-NL exceeds **PCA-N** on years (+0.45), geography (+0.25), colors
   (+0.07) — a curved 3-coord code reconstructs held-out activations better than
   *any* linear 3-D subspace can. That is a non-trivial positive result, not just
   "matches the remedy the paper already names."
3. **On essentially-linear manifolds it loses, by design.** age (−0.50) and
   temperature (−0.11) are near-1-D lines; PCA already saturates and the nonlinear
   chart *underfits* them (tiny manifolds: age = 69 train points). Curvature helps
   exactly where there is curvature — and the held-out split correctly penalises it
   where there isn't.
4. **The interpretability leg does NOT come for free — report this loudly.** The
   secondary *held-out label R²* (same regressor, all conditions) shows the
   nonlinear coords decode the label **worse** than PCA or even the linear charts on
   years (NL 0.16 lin / 0.29 kNN vs PCA 0.75) and geography. Better reconstruction
   geometry ≠ a more legible coordinate: the curved chart spends its fidelity on
   bending through activation space, not on laying the label out linearly. So this
   is a Pareto move on *geometry fidelity* but **not** (yet) on
   *interpretability/parsimony* — the bar we set ourselves. kNN recovers some of the
   gap, so the label is present in the coords but nonlinearly embedded.

Bottom line: **curvature genuinely buys held-out geometric fidelity beyond the
linear ceiling on curved manifolds** — the central claim survives a fair test. But
the win is reconstruction-geometry, and the interpretability payoff the whole
premise rests on is *not* demonstrated and partly contradicted. Earning that is the
real next step, not a foregone conclusion.

## Making the coordinate legible (`legible_coord.py`)

The unmet bar above was *interpretability*: the nonlinear coordinate reconstructed
well but decoded the label poorly. `legible_coord.py` tests whether a coordinate
can be **both** by adding a weak **label-alignment** term — a per-manifold linear
probe on the dominant chart's coordinate, trained on TRAIN labels only
(per-manifold standardized) — and sweeping its weight `lam_label`. The reported
label R² still uses a *fresh* held-out linear probe (same protocol as
`fair_comparison`), so alignment only *shapes* the coordinate; R² still tests
generalization.

```bash
SAE_DEVICE=cpu SAE_D_MODEL=576 uv run python prototype/legible_coord.py \
    --seeds 0 1 2 --lams 0 0.3 1 3 10
# -> cache/legible_coord/{results.json, pareto.png, run.log}
```

### Result — legibility is nearly free (3 seeds, coord_dim = 3)

| lam_label | VE@3 (mean) | label R² linear, non-cyclic mean | years R² | geography R² |
|---|---|---|---|---|
| 0 (unsupervised) | 0.68 | 0.55 | 0.16 | 0.41 |
| 0.3 | 0.67 | 0.66 | 0.24 | 0.69 |
| **1** | **0.68** | **0.89** | **0.89** | **0.80** |
| 3 | 0.64 | 0.94 | 0.98 | 0.87 |
| 10 | 0.63 | 0.95 | 0.99 | 0.91 |

**Verdict — the interpretability bar is met (for curved manifolds).** At
`lam_label = 1`, held-out label R² jumps from 0.16→0.89 (years) and 0.41→0.80
(geography) **with essentially no reconstruction cost** (VE@3 mean 0.68→0.68; years
0.84→0.83). At that operating point the curved-manifold coordinate **beats PCA on
both axes simultaneously** — fidelity (years VE 0.83 vs PCA-3 0.39; geography 0.76
vs 0.52) *and* legibility (years R² 0.89 vs PCA 0.75; geography 0.80 vs 0.60). So
the earlier fidelity↔legibility tension was an artifact of the *unsupervised*
coordinate being free to bend arbitrarily; a weak orientation prior removes it.

Honest caveats (see `pareto.png`):
- **Push too hard and the tradeoff reappears.** At `lam = 10`, R² saturates (~0.95)
  but VE erodes (geography 0.76→0.69, colors 0.80→0.77). The clean operating point
  is the knee, `lam ≈ 1`; alignment and reconstruction only *compete* past it.
- **This uses weak label supervision.** The router/charts stay unsupervised; only
  the within-chart *axis* is oriented by the label. That fits a "probe a known
  concept" setting, not unsupervised discovery — be clear about which claim is made.
- **`age` stays weak** (VE ≈ 0.29, and it degrades at high `lam`): 69 train points
  is too few for the MLP, so alignment and reconstruction fight over scarce data.
- **`colors` is a genuine miss** — see the cyclic-scoring section below.

## Cyclic-aware scoring for colors (`fair_comparison.label_score`)

Colours are labelled by **hue**, which wraps (0.99 and 0.01 are both red), so
linear R² on the raw value unfairly scored every method ≈0 — penalising a
coordinate that correctly parameterises the colour *loop*. `label_score`
(`CYCLIC_LABEL = {"colors": 1.0}`) instead regresses the N-dim representation onto
`(cos θ, sin θ)` of the hue angle and reports multi-output R² (same linear + kNN
regressors as everything else); non-cyclic manifolds are unchanged. Both
experiments now use it.

Re-scored, colours rise off the floor but the story is **honest and unflattering to
the factored model**:

| method (held-out N=3) | colors raw-linear R² (old) | colors cyclic R² (new) |
|---|---|---|
| PCA | −0.01 | **0.30** |
| SAE-mix geometric | −0.01 | 0.29 |
| Factored NONLINEAR (unsup., lam=0) | −0.00 | 0.13 |

And across the legibility sweep the linear-alignment term barely moves it
(0.13 → 0.13 → 0.15 → 0.17 → 0.34 as `lam` 0→10) — it only nudges up at `lam = 10`,
exactly where VE erodes. So:

- Cyclic scoring confirms there *is* recoverable circular structure (PCA gets 0.30),
  and the earlier ≈0 was a scoring artifact — but it does **not** rescue the
  factored coordinate. Under the *linear* alignment probe the curved coordinate
  stays at ~0.13–0.17, **below PCA** — the one labelled manifold where the factored
  approach loses on legibility even when scored fairly.
- The reason is mechanical: a linear probe `w·z + b` cannot orient a coordinate onto
  a circle. Making colours legible needs a **cyclic alignment term** (align to
  `cos/sin`), not just cyclic scoring — that is the open item, not a quick fix.

## Possible next steps

- **Cyclic-aware *alignment*** (align the coordinate to `cos/sin`, not just score it
  that way) — the missing piece for colours; scoring alone left it below PCA.
- **Unsupervised legibility**: replace the label prior with an isometry/arc-length
  penalty and see if a *label-free* coordinate lands near the `lam ≈ 1` point.
- A **causal/steering** leg: move along a (now-legible) chart coordinate → smooth
  predicted-output change.
- Group-sparse (top-k charts) selection instead of soft softmax.
