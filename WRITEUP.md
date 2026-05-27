# Do SAEs capture concept manifolds — and can a curved SAE do better?

*A working log of a scaled-down reproduction and a research prototype. Honest
process notes for a future blog/article — what we did, why, what worked, what
didn't, and what's still open. This is a living document; we add to it as the work
continues.*

---

## 0. TL;DR

We reproduced the core finding of Goodfire's *Do Sparse Autoencoders Capture Concept
Manifolds?* on a small model, then built a **manifold-native "atlas" SAE** (a router
over charts + per-chart curved coordinate) and stress-tested whether it actually
recovers concept-manifold geometry better than a standard SAE — under a fair,
held-out, matched-dimensionality protocol.

The short version:

- **Reproduced** the "dilution/shattering" effect: a linear SAE dictionary spreads a
  curved concept manifold across many partially-shared atoms; PCA captures it in a
  handful of dimensions, the SAE's actual codes plateau far below.
- **Curved charts beat the linear ceiling on fidelity.** Under a fair held-out
  subspace-capture test at matched dimensionality, a nonlinear chart reconstructs
  curved manifolds (years-helix, geography, colours) *better than the optimal linear
  N-D subspace (PCA-N)*. On essentially-1-D manifolds (age, temperature) it loses, as
  it should.
- **Better reconstruction ≠ a legible coordinate.** The curved coordinate, left
  unsupervised, decoded the ground-truth label *worse* than PCA — a Pareto win on
  geometry but not on interpretability.
- **A weak, concept-shaped label prior fixes legibility almost for free** (held-out
  R² 0.16→0.89 on years at ~zero reconstruction cost), and a cyclic (cos/sin) variant
  does the same for the wrap-around colours manifold.
- **Label-free legibility is harder.** An isometry/arc-length prior alone *fails*;
  adding a coordinate-parsimony prior *recovers* it — but only on genuinely
  low-dimensional manifolds, and a single global strength over-collapses higher-D
  ones. Qualified positive; adaptivity is the open frontier.

Throughout, the honest caveat: this is a **toy** (135M model, ~5k points, a research
SAE, not a scalable one), and we changed the **lens (the SAE), not the model**.

---

## 1. Background and the question

**The paper.** Goodfire's *Do SAEs Capture Concept Manifolds?* (arXiv:2604.28119)
argues that sparse autoencoders, whose dictionaries are collections of *straight*
atoms, struggle to represent *curved* concept manifolds (years as a helix, colours
as a wheel, geography on a sphere). The SAE "shatters" or "dilutes" the manifold
across many localized features instead of capturing it with a compact coordinate
system. Their yardstick is **subspace capture**: how much of a concept manifold's
activation variance you can explain with N directions — PCA (optimal linear) vs the
SAE's decoder directions vs the SAE's actual codes.

**Our question.** If the failure is that the *atoms are straight*, would a
**manifold-native, curved** SAE do better — and can we show that *fairly*, rather
than fooling ourselves with a confounded comparison? And if it reconstructs better,
is the resulting representation actually more *interpretable* (does its coordinate
track the concept), which is the entire point?

---

## 2. Setup and constraints

- **Hardware:** Apple M1, 8 GB RAM. The paper's Llama-3.1-8B needs ~16 GB, so we
  **substituted SmolLM2-135M** (also `LlamaForCausalLM`, d=576, layer 19). Same
  `model.model.layers[L]` access path, so the pipeline is unchanged; we made
  `data.py` model-configurable via env vars.
- **Why this is fair-ish but limited:** a 135M model has weaker, noisier concept
  geometry than an 8B one. We treat absolute numbers as illustrative and lean on
  *relative* comparisons (same data, same metric, only the architecture differs).
- **Compute discipline:** everything runs on **CPU** (MPS top-k was pathologically
  slow); `SAE_DEVICE=cpu SAE_D_MODEL=576`. Background jobs via `nohup` + logfile.
- **Manifolds:** prompt-defined datasets with ground-truth labels — `years` (helix),
  `age`/`temperature` (lines), `colors` (hue/lightness/saturation, hue is cyclic),
  `geography` (lat/long on a sphere), plus `days` (too small, dropped from headline).
  We extract last-token activations once and cache them.

---

## 3. Part I — Reproducing the core finding

We trained a BatchTopK SAE (`train_sae.py`, the missing training half of the
inference-only upstream repo) on 40k C4 background tokens (d_sae=4608, k∈{16,32,64}),
then ran the paper's subspace-capture curves (`subspace_capture.py`).

**Result (years/helix, variance explained vs # directions):**

```
   k=        4     8    16    32    64
   PCA      51%   77%   94%   98%  100%     <- compact linear subspace exists
   geo-SAE  12%   20%   30%   42%   57%     <- greedy over SAE decoder directions
   stat-SAE  8%   11%   14%   17%   17%     <- the SAE's ACTUAL codes plateau at 17%
```

The manifold lives in a ~12–16-D linear subspace (PCA hits 94% by 16 dims), but the
SAE's actual codes plateau around **17%** even with 64 features — the **dilution /
shattering** regime, reproduced. A sparsity sweep (k=16/32/64) showed the plateau is
fairly intrinsic, not just a sparsity artifact. We also built 3-D manifold
visualizations (`manifold_viz.py`) showing SAE decoder chords cutting *across* the
curve rather than following it.

**Why this matters:** it confirms the premise on our small model, so any improvement
we claim later is against a real, reproduced failure — not a strawman.

---

## 4. Part II — The idea: an "atlas" SAE

**Analogy.** A standard SAE is like trying to describe every place on Earth with a
fixed dictionary of straight arrows from the planet's centre. A better scheme is an
**atlas**: first say *which* chart (country/region), then give *local coordinates*
within it. We bake that in (`factored_sae.py`, `FactoredSAE`):

- a soft **router** `r(x)` → distribution over `M` charts (*which manifold*),
- a per-chart **coordinate** `z_m(x) ∈ R^coord_dim` (*where on it*),
- a per-chart **nonlinear decoder** `g_m(z_m)` → activation (a *curved* chart),
- reconstruction `x̂ = bias + Σ_m a_m · g_m(z_m)`.

A single curved chart can bend along a manifold that a linear dictionary must tile
with many atoms. The router is the *learned, unsupervised* analogue of the paper's
post-hoc feature clustering, baked into the architecture (router = MoE-style expert
selection; charts = local coordinate maps = an atlas).

**Honest framing from the start:** this is a research toy, not a scalable SAE. It
trains on the cached concept-manifold mixture, and — importantly — it changes the
**SAE (the lens)**, not the model. The main lever (subspace-per-cluster) is the
paper's own remedy operationalized, so the novelty bar is "does curvature
*specifically* buy something, fairly measured."

---

## 5. Part III — The first signal, and why we distrusted it

An initial ablation ladder (single seed, in-sample) hinted at the effect:

| chart type | recon VE | age R² | temp R² | years R² |
|---|---|---|---|---|
| 1 ray (SAE-like) | 0.978 | 0.08 | 0.01 | 0.04 |
| 3-D linear subspace | 0.985 | 0.72 | 0.78 | 0.50 |
| 3-D curved (nonlinear) | 0.996 | 0.95 | 0.86 | 0.42 |

Suggestive — but we listed the confounds and refused to believe it:

1. **Capacity confound:** coord_dim 1 vs 3 (3 predictors trivially beat 1).
2. **In-sample R²:** fit and scored on the same points → inflates higher-capacity
   models.
3. **No real baseline:** the "1-ray" config is not a real standard SAE (it still has
   the router + curated data).
4. **One seed, ~5k points, 135M model;** `years` even split across charts.

This is the methodological heart of the project: *a hint is not a result.* So we
designed a fair test before claiming anything.

---

## 6. Part IV — The fair comparison (`fair_comparison.py`)

### 6.1 The design (the "five-point" bar we set ourselves)

To isolate **curvature** as the only variable and kill the confounds:

- **(a) A real standard-SAE baseline** — the off-the-shelf C4-trained BatchTopK SAE
  *and* a standard SAE retrained on the same mixture train split (isolates
  architecture from training data).
- **(b) Held-out** 70/30 split per manifold; everything scored on the test set.
- **(c) Fixed coordinate dimensionality** across the factored conditions, so
  *linear-vs-nonlinear charts* is the only difference.
- **(d) The paper's subspace-capture metric** as the architecture-agnostic common
  yardstick.
- **(e) Multiple seeds** (3), report mean ± spread.

**The common metric.** For each manifold, the fraction of its **held-out** activation
variance explained by an **N-dimensional per-point code**, in raw activation units
against a single per-manifold (train-mean) denominator, so every method is directly
comparable:

- **PCA-N** — optimal linear ceiling.
- **SAE geometric greedy** — N decoder atoms greedily selected on the *train*
  residual, scored on test (a linear subspace ⇒ ≤ PCA-N).
- **SAE statistical** — the SAE's actual centered-code reconstruction (the paper's
  "dilution" curve).
- **Factored linear / nonlinear, coord_dim = N** — dominant-chart reconstruction.

The leveler: every method answers "reconstruct held-out X from an N-dimensional
per-point code." A linear dictionary can at best reach PCA-N; **only a curved decoder
can exceed it.** So `factored_nl − PCA-N` isolates curvature, and
`factored_nl − factored_lin` controls for clustering/parameters.

### 6.2 Result — held-out VE at N=3 (mean over 3 seeds)

| method | years | age | temperature | colors | geography |
|---|---|---|---|---|---|
| PCA (linear ceiling) | 0.39 | **0.79** | **0.79** | 0.74 | 0.52 |
| SAE-C4 geometric | 0.09 | 0.06 | 0.09 | 0.09 | 0.04 |
| SAE-mix geometric | 0.10 | 0.27 | 0.50 | 0.53 | 0.20 |
| SAE-mix statistical | 0.05 | 0.03 | 0.38 | 0.37 | 0.09 |
| Factored **LINEAR** | 0.09 | −0.07 | 0.05 | 0.57 | 0.12 |
| Factored **NONLINEAR** | **0.84** | 0.29 | 0.68 | **0.80** | **0.76** |

### 6.3 Verdict — a split decision, leaning positive

1. **Curvature is decisive at matched dimensionality.** Nonlinear beats linear charts
   at the *same* coord_dim on every manifold (Δ = +0.23 to +0.75), many× the seed
   spread — the cleanest isolation of the paper's thesis: the failure is the
   *straightness* of the atoms, not clustering or parameter count.
2. **On curved manifolds, it beats even the linear ceiling.** Factored-NL exceeds
   PCA-3 on years (+0.45), geography (+0.25), colours (+0.07). A curved 3-coord code
   reconstructs held-out activations better than *any* linear 3-D subspace.
3. **On 1-D manifolds it loses, by design.** age (−0.50) and temperature (−0.11):
   PCA saturates, and the MLP *underfits* tiny data (age = 69 train points). The
   held-out split correctly penalises curvature where there is none.
4. **Better reconstruction ≠ a legible coordinate.** The secondary metric — held-out
   label R² from the N-D representation, same regressor for all conditions — showed
   the nonlinear coords decode the label *worse* than PCA or even the linear charts
   (years: NL 0.16 vs PCA 0.75). So this is a Pareto move on **geometry fidelity** but
   **not** on **interpretability/parsimony** — the bar we had set.

This is the pivot: fidelity was won, but the interpretability payoff (the whole point
of the paper) was *not demonstrated and partly contradicted*. Earning it became the
next thread.

---

## 7. Part V — Making the coordinate legible

### 7.1 Weak label alignment (`legible_coord.py`)

Hypothesis: the unsupervised coordinate is free to *bend arbitrarily* through
coordinate space; a weak orientation prior should straighten it without hurting
reconstruction. We add a per-manifold probe on the dominant chart's coordinate,
trained on **train labels only**, and sweep its weight `lam_label`. The reported
label R² uses a *fresh* held-out probe (alignment only shapes the coordinate; R²
still tests generalization).

**Result (3 seeds, coord_dim = 3):**

| lam_label | VE@3 (mean) | label R² (non-cyclic mean) | years | geography |
|---|---|---|---|---|
| 0 (unsup.) | 0.68 | 0.55 | 0.16 | 0.41 |
| **1** | **0.68** | **0.89** | **0.89** | **0.80** |
| 10 | 0.63 | 0.95 | 0.99 | 0.91 |

At `lam = 1`, held-out label R² jumps from 0.16→0.89 (years) with **essentially zero
reconstruction cost**, and the curved coordinate **beats PCA on both axes at once**.
The earlier tension was an artifact of the unsupervised coordinate bending freely; a
weak prior removes it. Push too hard (`lam = 10`) and a real tradeoff reappears (VE
erodes). **Caveat:** this uses *weak label supervision* — fine for "probe a known
concept," not for unsupervised discovery.

**What it looks like** (`viz_coord.py` → `cache/viz/coord_vs_label.png`): plotting
the held-out probe's *predicted* label against the *true* label makes the R² concrete
— the unsupervised coordinate is a scattered cloud (years R²≈0.04), the weakly-aligned
one snaps to a tight diagonal (years 0.86, temperature 1.00, geography 0.79). The
number wasn't hiding anything; the coordinate genuinely straightens.

### 7.2 Cyclic concepts: scoring then alignment

Colours are labelled by **hue**, which wraps (0.99 and 0.01 are both red).

- **Cyclic scoring** (`fair_comparison.label_score`): regress the representation onto
  `(cos θ, sin θ)` instead of raw hue. This revealed the earlier ≈0 was a *scoring
  artifact* — PCA actually recovers 0.30 of the hue circle — but the *linear*
  alignment probe still couldn't lift the factored coordinate (stuck ~0.13–0.17,
  below PCA). A linear probe `w·z + b` cannot orient a coordinate onto a circle.
- **Cyclic alignment** (`legible_coord`, concept-shaped target): make the alignment
  target `(cos θ, sin θ)` too (a 2-output probe). Now colours' cyclic R² climbs
  0.13→0.24→0.33→0.41→0.54 across `lam` 0→10, **crossing its PCA reference (0.30) at
  `lam ≈ 1`** while VE holds at 0.80. So the miss was *mechanical* — legibility needs
  a *concept-shaped* target, not just a fair score.

**Residue:** colours is still the hardest manifold in absolute terms (R² ~0.4–0.5 vs
~0.9 for years/geography) — hue is entangled with lightness/saturation over ~1.8k
noisy points. The figure `cache/viz/colors_circle.png` shows this honestly: even
strong cyclic alignment forms only a *partial* hue loop in the predicted (cos, sin)
plane, not the clean rainbow circle the clean manifolds would give.

---

## 8. Part VI — Unsupervised legibility (dropping the label crutch)

The Part-V wins all use labels. Can a coordinate be made legible **with no labels**?

### 8.1 Isometry alone (`iso_coord.py`) — a clean negative

Idea (cf. Gropp et al., *Isometric Autoencoders*, 2020): make each chart decoder
**constant-speed** (`‖J_m u‖` pinned to a fixed target along random directions, via
finite difference, router-weighted) so the coordinate becomes **arc-length**. For
manifolds sampled uniformly in the underlying factor, arc length ≈ affine in the
label, so it should decode linearly. No labels in training.

**Result (3 seeds): it does not work.** The non-cyclic mean R² never exits seed noise
(0.55 → 0.63 peak, then flat), the iso points never reach the PCA reference on *any*
manifold, and any weight large enough to reshape the coordinate **destroys
reconstruction** (VE 0.68 → 0.32 at lam=10, negative at 30).

*Why:* **arc-length ≠ linear-in-coordinate.** A 3-D isometric coordinate can still
*wind* through coordinate space; isometry is necessary but not sufficient for linear
legibility. It also fights reconstruction and says nothing about *which* direction is
the concept. (We also caught and fixed a subtlety mid-stream: a "constant speed"
variance-only penalty has a trivial `z→const` collapse minimizer; a *fixed* unit-ish
target avoids it.)

### 8.2 Isometry + coordinate parsimony (`iso_parsimony.py`) — a qualified positive

If the problem is spare dimensions to wind in, also **minimize the number of active
coordinate dims**. We add a **participation-ratio** penalty
`PR = (Σ vᵢ)² / Σ vᵢ²` on the coordinate's per-dim variance — a *scale-invariant*
active-dim count (an L1-on-std penalty can be gamed by shrinking the coord and growing
the decoder; PR cannot). Still label-free. We grid `(lam_iso, lam_pars)`.

**Result (3 seeds, held-out label R², no labels in training):**

| config (iso, pars) | years | temperature | geography | colors | active-dims (PR) |
|---|---|---|---|---|---|
| (0, 0) unsup. | 0.16 | 0.79 | 0.41 | 0.13 | ~2.3 |
| (1, 8) iso+parsimony | **0.70** | **0.89** | 0.08 | 0.07 | ~1.4 |
| PCA reference | 0.75 | 0.92 | 0.60 | 0.30 | — |

The active-dim count collapses (~2.3→~1) as designed, and the effect **splits cleanly
by intrinsic dimension**:

- **Works, label-free, on low-D manifolds:** years 0.16→0.70 and temperature
  0.79→0.89 — near their PCA references, and years approaches the *supervised* 0.89.
  *The first time the unsupervised route recovered real legibility.*
- **Backfires on multi-D manifolds:** a single global parsimony weight over-collapses
  geography (~2-3D sphere: 0.41→0.01) and colours (2-D cyclic: →0.07), destroying both
  their legibility and VE — you cannot force a sphere onto one straight axis.
- **Aggregate is flat** (gains and losses cancel) and VE drops as parsimony bites.

**Verdict:** parsimony *was* the missing ingredient the isometry post-mortem
predicted — but its strength must match the manifold's intrinsic dimension, which a
global knob can't know. **Intrinsic-dimension-adaptive parsimony is the open
frontier.**

---

## 9. What worked, what didn't (at a glance)

| Step | Outcome |
|---|---|
| Reproduce dilution/shattering | ✅ Reproduced (SAE codes plateau ~17% vs PCA 94% @16-D) |
| Curved charts vs linear, matched dim, held-out | ✅ Curvature wins on fidelity; beats PCA on curved manifolds |
| Curved charts on 1-D manifolds | ❌ Underfit (PCA already saturates; tiny data) |
| Unsupervised curved coordinate → legible? | ❌ Worse label-decoding than PCA |
| Weak label alignment → legible? | ✅ R² 0.16→0.89 at ~zero VE cost; beats PCA on both axes |
| Cyclic scoring (colours) | ✅ Honest scoring (artifact removed); ⚠️ scoring alone insufficient |
| Cyclic alignment (colours) | ✅ Crosses PCA at lam≈1; ⚠️ still hardest manifold |
| Unsupervised isometry alone | ❌ Negative; arc-length ≠ linear-in-coordinate |
| Isometry + parsimony | ◐ Qualified positive: works on low-D, over-collapses multi-D |

---

## 10. Limitations (read this before believing anything)

- **Toy scale.** SmolLM2-135M (not Llama-3.1-8B), ~5k points total, manifolds as small
  as 56–199 points. `age` (69 train points) is consistently unreliable; `days` was
  dropped entirely. Three seeds is few.
- **We changed the lens, not the model.** All of this re-represents fixed activations
  with different SAEs/decoders; we never intervened on the model. No causal claim about
  the model's computation.
- **The factored model is not a scalable SAE.** It has no SAE-style sparsity, trains on
  a tiny curated mixture, and its "router discovers manifolds" property was only
  checked loosely (purity), not rigorously held-out in the fair comparison.
- **"Matched dimensionality," not fully "matched sparsity/capacity."** We equalized the
  *representation dimension N* via subspace capture, but a factored chart's nonlinear
  decoder has many parameters; the linear-chart control addresses "is it just params?"
  but the comparison is not a like-for-like parameter budget.
- **Label supervision in Part V.** The legibility wins there are *semi-supervised* (the
  axis is oriented by the concept label). Honest framing: "probe a known concept," not
  unsupervised discovery.
- **Held-out label R² depends on the regressor/label.** We report linear + kNN and use
  cyclic scoring for hue, but other concepts could have their own pathologies.
- **Small-model geometry is weak/noisy**, so absolute numbers are illustrative; we lean
  on relative, within-protocol comparisons.

---

## 11. The five-point bar — status, and what's left

The fair-comparison "five-point" requirements are **all met**:

| # | Requirement | Status |
|---|---|---|
| (a) | Real standard-SAE baseline | ✅ C4 SAE **and** mixture-retrained SAE |
| (b) | Held-out train/test eval | ✅ 70/30 per manifold |
| (c) | Fixed coordinate dimensionality | ✅ linear vs nonlinear at matched coord_dim |
| (d) | Paper's subspace-capture metric | ✅ held-out VE(N), raw units |
| (e) | Multiple seeds | ✅ 3 seeds, mean ± spread |

The broader **"definition of improvement"** bar (a Pareto move on *geometry fidelity ×
interpretability/parsimony* at matched sparsity & capacity, held-out, ideally on a
real model, **bonus:** a causal/steering leg) is **partially met**:

- Geometry fidelity ✅; interpretability ✅ *with* weak supervision, ◐ unsupervised.
- "Ideally on a real model" ❌ — still SmolLM2-135M.
- **Bonus causal/steering leg ❌ — not done.** This is the most visible gap.

**Explicitly still on the list / things we have not yet done:**

1. **Causal / steering leg** (the listed bonus): move along a legible chart coordinate
   and show a smooth, predictable change in model output. We have a *now-legible*
   coordinate to try this on — it's the natural next experiment and would turn a
   "representation" claim into a "control" claim.
2. ~~**Coordinate-vs-label visualization.**~~ ✅ **Done** (`viz_coord.py`):
   `cache/viz/coord_vs_label.png` (predicted-vs-true label, unsupervised vs aligned)
   and `cache/viz/colors_circle.png` (the partial hue loop). The diagonal-tightening
   makes the legibility claim falsifiable at a glance.
3. **Intrinsic-dimension-adaptive parsimony** (per-chart learned dim, or a
   Matryoshka/nested coordinate with per-dim gates) — to fix the over-collapse of
   geography/colours while keeping the years/temperature win.
4. **A proper isometric-AE** (exact Jacobian + encoder pseudo-inverse term) instead of
   the finite-difference surrogate, to give isometry its fairest shot.
5. **Validate on a real model** — at least one larger model / more layers, to show the
   135M result isn't an artifact of weak small-model geometry.
6. **In-the-wild router.** Train the factored model on *background* activations (not the
   curated mixture) and test whether charts discover manifolds unsupervised — the claim
   "the router is the learned analogue of feature clustering" is currently only shown on
   curated data.
7. **Group-sparse (top-k charts)** selection instead of soft softmax, for a more
   SAE-like, genuinely sparse object.
8. **Statistical rigor:** more seeds, confidence intervals, and ideally larger manifolds
   (drop/augment `age`, `days`).

---

## 12. Reproducibility appendix

All commands run from the repo root on CPU with `SAE_DEVICE=cpu SAE_D_MODEL=576`
(prepend `SAE_MODEL_NAME=HuggingFaceTB/SmolLM2-135M SAE_LAYER=19` for anything that
loads the model).

| Stage | Command | Output |
|---|---|---|
| Extract manifolds | `uv run data.py` | `cache/<manifold>.pt` |
| Train SAE | `uv run train_sae.py --expansion-factor 8 --k 32` | `cache/sae_4608_k32.pt` |
| Reproduce subspace capture | `uv run subspace_capture.py plot --sae cache/sae_4608_k32.pt --k 32` | `cache/subspace_capture/` |
| Fair comparison | `uv run python prototype/fair_comparison.py --seeds 0 1 2 --coord-dims 2 3` | `cache/fair_comparison/` |
| Legible coordinate (supervised) | `uv run python prototype/legible_coord.py --seeds 0 1 2 --lams 0 0.3 1 3 10` | `cache/legible_coord/` |
| Isometry (unsupervised) | `uv run python prototype/iso_coord.py --seeds 0 1 2 --lams 0 1 3 10 30` | `cache/iso_coord/` |
| Isometry + parsimony | `uv run python prototype/iso_parsimony.py --seeds 0 1 2 --lam-isos 0 1 --lam-pars 0 2 8` | `cache/iso_parsimony/` |
| Legibility figures | `uv run python prototype/viz_coord.py` | `cache/viz/` |

**Key files:** `data.py` (manifolds + extraction), `train_sae.py` / `saes.py`
(standard BatchTopK SAE), `subspace_capture.py` (the paper's metric),
`prototype/factored_sae.py` (the atlas SAE), `prototype/fair_comparison.py` (the
fair test + shared eval helpers), `prototype/legible_coord.py`,
`prototype/iso_coord.py`, `prototype/iso_parsimony.py`. Detailed reproduction notes
in `REPRODUCTION.md`; prototype notes in `prototype/README.md`.

---

*Last updated: 2026-05-27. Open threads tracked in §11.*
