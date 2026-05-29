# Do SAEs capture concept manifolds — and can a curved SAE do better?

*A working log of a scaled-down reproduction and a research prototype. Honest
process notes for a future blog/article — what we did, why, what worked, what
didn't, and what's still open. This is a living document; we add to it as the work
continues.*

---

> ## ⚠️ RETRACTION-IN-PROGRESS (2026-05-29) — read before the TL;DR
>
> On 2026-05-29 we found a load-bearing bug in `fair_comparison.factored_eval`:
> the reconstruction was computed from the **K-chart mixture** while the coords
> for label decoding were taken from a **single dominant chart**. The "matched
> coord_dim, curvature is the only free variable" framing was false — VE was
> being scored over a ≤K×coord_dim subspace union (12 × 3 = 36-D) on the
> factored rows while baselines used the advertised 3-D.
>
> The patch has two parts (both committed locally as of 2026-05-29):
> 1. `factored_eval` now uses dominant-chart-only recon (`g_dom(z_dom) + bias`).
> 2. `FactoredSAE` gained a `hard_routing` flag; `train_factored` uses it so the
>    model is trained as a true atlas (single-chart recon per point, via
>    straight-through one-hot). Without (2), (1) alone puts the model in a regime
>    it never saw and VE goes to large negatives.
>
> **What the corrected 135M headline looks like (3 seeds, hard-routed atlas):**
>
> | manifold    | factored-NL | PCA  | Δ vs PCA |
> |-------------|-------------|------|----------|
> | **years**   | 0.51        | 0.39 | **+0.12** ★ |
> | geography   | 0.30        | 0.52 | −0.21    |
> | colors      | 0.59        | 0.74 | −0.15    |
> | temperature | 0.66        | 0.79 | −0.14    |
> | age         | 0.12        | 0.79 | −0.67    |
>
> The original "+0.45 on years, +0.25 on geography, +0.07 on colors" reduces to
> **+0.12 on years only**. Curvature delta (NL − LIN) is still real (+0.30 to
> +2.00 across manifolds), so the architecture works — it just doesn't clear
> PCA-3 except on years.
>
> **What's affected:**
> - §3, §4, §5, §6, §7 (135M sweeps): every factored VE row is bug-inflated;
>   only §9b.1 (above) is currently re-run with the fix.
> - §8 supervised legibility (`legible_coord`): coords were always dominant-chart
>   (label-R² rows safe under the OLD architecture), but the underlying model
>   was soft-routed. Re-run with hard-routed `train_factored_legible` pending —
>   the baseline shifted (years 0.16 → 0.57), so the supervised lift will be
>   smaller than the reported 0.89 jump.
> - §9b.1–9b.6 (8B): same bug across the board, deferred until 135M corrected
>   story is settled.
> - §9b.7 steering (135M *and* 8B): **unaffected** — `steer.py` discards `fve`.
>
> Meta-lesson saved as `[[research-comparison-smell]]`. The text below this
> banner is preserved verbatim for chronology; treat its numbers as historical
> claims, not current findings, until each section is re-run and updated.
>
> ---

## 0. TL;DR

We reproduced the core finding of Goodfire's *Do Sparse Autoencoders Capture Concept
Manifolds?* on a small model, then built a **manifold-native "atlas" SAE** (a router
over charts + per-chart curved coordinate) and stress-tested whether it actually
recovers concept-manifold geometry better than a standard SAE — under a fair,
held-out, matched-dimensionality protocol.

The short version (**corrected 2026-05-29**, see retraction banner above for the
process):

- **Reproduced** the "dilution/shattering" effect: a linear SAE dictionary spreads a
  curved concept manifold across many partially-shared atoms; PCA captures it in a
  handful of dimensions, the SAE's actual codes plateau far below. *(unaffected by
  the bug)*
- **Curved charts beat the linear ceiling on fidelity — narrowly, on one manifold.**
  Under the corrected matched-dim test (hard-routed atlas, dominant-chart recon), the
  factored-nonlinear coord at coord_dim=3 beats PCA-3 on **years only** (VE 0.51 vs
  0.39, +0.12). On geography, colours, temperature, age it lands below PCA-3. The
  architecture's internal lever is intact: nonlinear charts beat linear charts by
  +0.30 to +2.00 VE at matched coord_dim, so curvature does real work — it just isn't
  enough to clear the linear ceiling except on the simplest curved manifold.
- **Hard routing alone improves the unsupervised coordinate substantially.** Switching
  the router from soft-mixture to straight-through one-hot lifts the dominant chart's
  unsupervised label R² on years from 0.16 to 0.57 — a previously-hidden win that
  emerged only after the bug forced us to retrain. The atlas-of-charts reading was
  always the right reading; the soft router was diluting both fidelity and legibility.
- **A weak, concept-shaped label prior pushes legibility past PCA on 4 of 5 manifolds
  at near-zero VE cost.** λ=1 takes years 0.57→0.91 (PCA 0.75), geography 0.54→0.82
  (PCA 0.60), temperature 0.78→0.93 (PCA 0.92), colours (cyclic cos/sin) 0.24→0.37
  (PCA 0.30); mean held-out VE moves 0.44→0.46 (essentially free). Only age fails to
  cross (small data, ~1-D linear — PCA's 0.91 is hard to beat).
- **Label-free legibility is harder.** Isometry alone *fails* (collapses VE to −0.07
  mean, doesn't help R²). Coordinate parsimony recovers years past PCA (0.57→0.83
  unsupervised at pars=8) but degrades age and stays neutral elsewhere; adaptive
  per-dim gates produce real-but-smaller lifts (years 0.21→0.50) and never cross
  PCA. The bug-era "geography over-collapses to 0.01" failure was an artifact and is
  gone. Unsupervised legibility frontier remains open: supervised remains the only
  consistent route past PCA.
- **The legible coordinate is causal at 135M; at 8B it modulates but does not
  control.** Steering moves the model's hot/cold readout monotonically in the
  concept direction at both scales (~13× a random control at 135M, ~6× at 8B,
  §9b.7) — but the 135M sweep actually flipped the model's preferred token
  (`logit(hot−cold)`: −1.33 → +0.04) while the 8B sweep stayed cold-dominant
  throughout (−1.31 → −0.79). At 135M this earned the framing "representation
  upgraded to control"; at 8B the data supports only "modulation with the right
  sign, above noise, well short of a behavioural switch." *(unaffected by the bug
  — steer.py discards the bug-affected VE quantity.)*

Throughout, the honest caveat: this is a **toy** (135M model, ~5k points, a research
SAE, not a scalable one), and we changed the **lens (the SAE), not the model**. And:
the 8B numbers in Part VIII (§9b.1–9b.6) were generated with the same bug; they are
**not yet re-run** with the hard-routing fix and should be treated as inflated until
they are.

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

### 6.2 Result — held-out VE at N=3 (mean over 3 seeds) — **corrected hard-route**

The table below is the post-fix version (hard-routed atlas, dominant-chart recon).
The pre-fix table is preserved in `cache/fair_comparison/results.prepatch.json` for
auditability; in summary it claimed +0.45 / +0.25 / +0.07 NL−PCA wins on years /
geography / colors, all of which evaporate once the K-chart mixture subspace is no
longer being counted on the factored-NL side.

| method | years | age | temperature | colors | geography |
|---|---|---|---|---|---|
| PCA (linear ceiling) | 0.39 | **0.79** | **0.79** | **0.74** | **0.52** |
| SAE-C4 geometric | 0.09 | 0.06 | 0.09 | 0.09 | 0.04 |
| SAE-mix geometric | 0.10 | 0.27 | 0.50 | 0.53 | 0.20 |
| SAE-mix statistical | 0.05 | 0.03 | 0.38 | 0.37 | 0.09 |
| Factored **LINEAR** | 0.01 | −1.87 | 0.05 | 0.09 | 0.01 |
| Factored **NONLINEAR** | **0.51** | 0.12 | 0.66 | 0.59 | 0.30 |

### 6.3 Verdict — a narrow positive on one manifold, plus an internal-lever finding

1. **Curvature is decisive at matched coord_dim — as an *internal* lever.** Nonlinear
   beats linear charts at the same coord_dim on every manifold (Δ = +0.50 on years,
   +1.99 on age, +0.61 on temperature, +0.50 on colors, +0.30 on geography). Single-
   chart linear decoders are very nearly incapable (≈0 VE) under hard routing — the
   straightness of atoms IS the bottleneck, exactly as the paper argued. This isolation
   survived the fix.
2. **Curvature beats even the linear ceiling on years only.** Factored-NL clears
   PCA-3 on **years (+0.12)** — the simplest curved manifold (a 1-D helix in d=576).
   On geography (−0.21), colors (−0.15), temperature (−0.14), age (−0.67) it lands
   below PCA-3. The corrected fidelity claim is one manifold, not three.
3. **On near-linear or small-data manifolds, the single chart underfits.** age (69
   train points, ~1-D linear) collapses dramatically (−0.67 vs PCA). The held-out
   split correctly penalises curvature where there is none, *and* penalises the
   under-trained MLP where data is thin.
4. **Better reconstruction ≠ a legible coordinate — but the gap is smaller than we
   first thought.** The secondary metric — held-out label R² from the N-D
   representation, same regressor for all conditions — shows factored-NL coords
   decode years 0.57 vs PCA 0.75 unsupervised, geography 0.54 vs 0.60, colors 0.24
   vs 0.30. Hard routing alone closed roughly half of what the bug-era version had
   reported as a fidelity-vs-legibility tension. There is still a tension, but it's
   not the dramatic one originally claimed (was: years 0.16 vs 0.75 = a yawning
   gap). The remaining tension is what Part V (weak label supervision) closes.

The pivot survives in spirit, smaller in magnitude: fidelity is won on one manifold,
not three; the legibility gap is genuine but narrower; the next thread (Part V)
becomes more about *crossing PCA on legibility* than *rescuing legibility from a
catastrophe*.

---

## 7. Part V — Making the coordinate legible

**A framing note (added after Part VIII):** the motivation for this Part is
small-model-specific in *degree* but not in *mechanism*. At 135M (the scale of
this section), PCA's held-out label R² is only 0.38 on colours and 0.67 on
geography — the linear coordinate is legible-in-principle but readable-in-practice
only with help, so weak label supervision is doing real *rescue* work. At
real-model scale (Llama-3.1-8B, §9b.2 / §9b.6) PCA's R² rises to 0.86–0.99
across all five concepts and the same mechanism becomes a *refinement* of an
already-readable baseline rather than a rescue. The lever below — a weak per-
manifold probe orienting the chart's coordinate at flat VE — is identical; what
shrinks at scale is the gap it has to close.

### 7.1 Weak label alignment (`legible_coord.py`) — **corrected hard-route numbers**

Hypothesis: the unsupervised coordinate is free to *bend arbitrarily* through
coordinate space; a weak orientation prior should straighten it without hurting
reconstruction. We add a per-manifold probe on the dominant chart's coordinate,
trained on **train labels only**, and sweep its weight `lam_label`. The reported
label R² uses a *fresh* held-out probe (alignment only shapes the coordinate; R²
still tests generalization).

**Result (3 seeds, coord_dim = 3, hard-routed atlas) — post-fix numbers:**

| lam_label | VE@3 (mean) | label R² (non-cyclic mean*) | years | geography | colors (cyclic) |
|---|---|---|---|---|---|
| 0 (unsup.) | 0.44 | 0.66 | 0.57 | 0.54 | 0.24 |
| **1** | **0.46** | **0.85** | **0.91** | **0.82** | **0.37** |
| 10 | 0.40 | 0.95 | 0.98 | 0.91 | 0.56 |
| PCA-3 reference | — | 0.79 | 0.75 | 0.60 | 0.30 |

At `lam = 1`, held-out label R² mean climbs 0.66 → 0.85 with **VE going from 0.44 to
0.46 — essentially zero reconstruction cost.** The curved coordinate crosses PCA on
4 of 5 manifolds at this knee: years (0.91 > 0.75), geography (0.82 > 0.60),
temperature (0.93 > 0.92), colours (cyclic cos/sin R² 0.37 > 0.30). Push to `lam =
10` and a real tradeoff appears (VE −0.04 mean). **Caveat:** this uses *weak label
supervision* — fine for "probe a known concept," not for unsupervised discovery.

**The "where did the gain come from" decomposition.** The bug-era version of this
table reported a 0.16→0.89 jump on years, framed as "weak supervision is the magic."
With the fix, the unsupervised baseline is *already* 0.57 (hard routing alone
provides ~0.41 of the originally-attributed +0.73 lift). The supervised piece adds a
clean further +0.34 (0.57→0.91). Two real, separable effects — hard routing fixes
the architecture, weak supervision orients the axis — instead of one conflated one.
Both are still genuine.

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

### 8.2 Isometry + coordinate parsimony (`iso_parsimony.py`) — qualified positive (corrected)

If the problem is spare dimensions to wind in, also **minimize the number of active
coordinate dims**. We add a **participation-ratio** penalty
`PR = (Σ vᵢ)² / Σ vᵢ²` on the coordinate's per-dim variance — a *scale-invariant*
active-dim count (an L1-on-std penalty can be gamed by shrinking the coord and growing
the decoder; PR cannot). Still label-free. We grid `(lam_iso, lam_pars)`.

**Result (3 seeds, held-out label R², hard-routed, no labels in training):**

| config (iso, pars) | years | temperature | geography | colors | active-dims (PR) |
|---|---|---|---|---|---|
| (0, 0) unsup. | 0.57 | 0.78 | 0.54 | 0.24 | ~2.2 |
| (0, 8) parsimony alone | **0.83** ✅ | 0.78 | 0.57 | 0.13 | ~1.2 |
| (1, 8) iso+parsimony | **0.82** ✅ | 0.63 | 0.52 | 0.11 | ~1.4 |
| PCA reference | 0.75 | 0.92 | 0.60 | 0.30 | — |

What survives and what doesn't, vs. the bug-era version of this table:

- **Years crosses PCA, label-free.** Parsimony at `pars=8` pushes years 0.57→0.83
  (above PCA's 0.75). This is the headline survivor: the unsupervised legibility
  route works on the simplest curved manifold. (Bug-era: years 0.16→0.70, almost
  reaching PCA — corrected version cleanly clears it.)
- **The "geography over-collapses to 0.01" failure was a bug artifact.** Under hard
  routing, geography barely moves under parsimony (0.54→0.57); colours stays low
  (0.24→0.13). The dramatic differential-dim collapse story is gone. Active-dim
  counts still drop ~2.2→~1.2 with parsimony, just without the catastrophic
  legibility collapse.
- **New failure mode visible: age now degrades** (0.73→0.48 at pars=8). The bug
  was masking this — age R² stayed flat at 0.84 across configs. With honest baselines,
  parsimony hurts the small-data near-linear manifold (age = 69 train points).
- **Iso alone (lam_iso=1, lam_pars=0): still negative.** Mean R² 0.65 (~baseline);
  mean VE collapses to −0.07. Iso doesn't help legibility, hurts reconstruction.
  Replicates the §8.1 finding without needing a separate experiment.

**Verdict:** parsimony was the missing ingredient the isometry post-mortem
predicted — it produces a single clean label-free win (years > PCA) and is otherwise
neutral or mildly harmful. The original "differential-dim-need" framing
("parsimony helps low-D, hurts multi-D") was partly a bug artifact (geography
collapse) and partly real (years win). What remains is one solid unsupervised win.

### 8.3 Intrinsic-dimension-adaptive parsimony (`adaptive_parsimony.py`) — a qualified negative

The §8.2 fix is to let **each chart learn how many coordinate dims it needs** instead
of one global target. We replace the global PR penalty with a per-chart, per-dim
learned **gate** `g_{m,j} ∈ (0,1)` on the coordinate, plus a fixed **per-dimension
cost** (L1 on the gates, with mild Matryoshka ordering so dim 0 fills first). The
intended mechanism: a *nonlinear* chart only needs coordinate dims equal to the
manifold's **intrinsic** dimension (the MLP supplies the embedding/curvature), so
reconstruction should pay to keep ~1 dim for the years-helix/temperature-line and ~2
for the geography-sphere/colours-loop — the active-dim count self-adapts per chart,
which a global knob cannot. (Anti-gaming: each coordinate dim is normalized by a
running-std buffer before gating, so the L1 cost can't be dodged by shrinking the
coordinate and inflating the decoder — the failure mode that motivated PR.)

**Result (3 seeds, hard-routed, label-free, grid over `lam_iso ∈ {0,1}` × `lam_gate ∈ {0,2,8}`) — corrected:**

| config | years | age | temp | geography | colors | active-dims (Σgates, dom. chart) |
|---|---|---|---|---|---|---|
| gate=0 (gated baseline) | 0.21 | 0.79 | 0.83 | 0.35 | 0.23 | ~2.65 |
| gate=2, iso=0 | 0.46 | 0.80 | 0.81 | 0.48 | 0.25 | **differentiated: years 1.94, geog 1.21, colors 1.41** |
| gate=8, iso=0 | **0.50** | 0.74 | 0.83 | 0.43 | 0.25 | ~1.2–1.4 |
| gate=8, iso=1 | 0.63 | 0.83 | 0.72 | 0.32 | 0.23 | ~1.3–1.6 |
| *iso_parsimony (iso=0, pars=8), for ref* | *0.83 ✅* | *0.48* | *0.78* | *0.57* | *0.13* | *~1.2* |
| *PCA-3 reference* | 0.75 | 0.91 | 0.92 | 0.60 | 0.30 | — |

Two takeaways under the corrected numbers:

1. **The qualified-negative essentially holds, with one rehabilitated detail.** Gates
   now produce a real R² lift on years (0.21→0.50 at gate=8 vs the bug-era +0.03),
   but the lift never crosses PCA on any individual manifold. Mean* non-cyclic R²
   moves 0.55→0.62 — a real but small unsupervised improvement. iso_parsimony still
   wins on years (0.83 vs 0.50).
2. **The "uniform gate closure" claim is partially refuted — but in the wrong
   direction for the hypothesis.** At gate=2, gates DO differentiate: geography
   collapses most (1.21 active), years stays highest (1.94), colors in between
   (1.41). But this is the *opposite* of intrinsic-dim allocation — geography
   (claimed ~2-3D) should keep *more* dims active if the hypothesis were right, not
   fewer. The actual pattern reflects per-chart needs: geography spreads across
   multiple charts (each chart sees a smaller piece and needs fewer dims), while
   years routes to one chart that must bend a whole helix and so needs the dims. So
   the differential allocation exists but it's *routing-driven*, not
   intrinsic-dim-driven. The targeted hypothesis is not vindicated; what we got
   instead is an honest mechanistic surprise.

2. **The one genuinely new lever is incidental, and it's a *trade*.** Isolated with
   `--no-coord-norm` (gate=0, iso=0, 3 seeds), the per-dim **coordinate normalization**
   — added only for anti-gaming — is *itself* a label-free legibility lever:

   | coord_norm | years R² | age R² | temp R² | geo R² | | years VE | temp VE | geo VE |
   |---|---|---|---|---|---|---|---|---|
   | **on** | **0.64** | 0.68 | 0.82 | 0.38 | | 0.45 | 0.51 | 0.64 |
   | off (≈ plain factored) | 0.28 | 0.82 | 0.78 | 0.31 | | **0.79** | **0.69** | **0.76** |

   Normalizing the coordinate a linear probe sees lifts years legibility **0.28→0.64**,
   but **pays for it in reconstruction** (years VE 0.79→0.45) and slightly *hurts* the
   near-linear `age`. So it's a Pareto move *along* the fidelity↔legibility frontier —
   it whitens the coordinate's per-dim scales (which a linear probe is sensitive to) at
   the cost of the decoder's freedom — not a free unsupervised win.

**Verdict:** a **qualified negative**. As a learned-gate L1, intrinsic-dimension-
adaptive parsimony *cures the over-collapse* (its stated job) but **fails to deliver
the differential dim allocation** that was the actual hope — gates close uniformly and
legibility is flat in the penalty, because at this scale the manifolds are effectively
~1-D to a nonlinear chart. The unsupervised legibility frontier is therefore **still
open**: the cleanest unsupervised lever we found (coordinate normalization) is a
fidelity↔legibility *trade*, and the supervised concept-shaped prior (Part V) remains
the only route that buys legibility at ~zero reconstruction cost. The honest read is
that "match parsimony to intrinsic dimension" needs a setting where the manifolds
*are* genuinely multi-D — and §9b.5 now closes that escape hatch: at 8B with much
cleaner geometry the gates *still* close uniformly with no per-manifold
differentiation. The qualified negative is **a cost-shape limit, not a scale
artefact**. A proper fix needs different machinery (group sparsity over dims,
per-manifold dim budget, or a different prior); see §12.

---

## 9. Part VII — Steering: the coordinate is causal, not just decodable (`steer.py`)

Everything to here shows the coordinate can be *read*. The sharper question — and the
bonus the project set itself — is whether it is *causal*: if we move along it, does the
**model's behaviour** follow? This is also the first experiment that touches the model,
not just the lens.

**A scale note (added after §9b.7):** the numbers below are 135M-specific. §9b.7
re-runs the same recipe on Llama-3.1-8B layer 16: the legible-axis curve stays
strictly monotone and ~6× the random control's slope, but the slope itself drops
from +0.231 to +0.090 (and the random-control ratio from ~13× to ~6×). Mechanism
survives; magnitude attenuates — consistent with §9b.6's "PCA already reads it at
scale" pattern (temperature's 8B PCA R² is 0.99).

**Method (activation steering, but the direction comes from our legible coordinate):**
take the aligned (λ=1) factored SAE, find temperature's dominant chart and its
**legible axis** ŵ (the unit coordinate direction the held-out label probe maps to the
label). Build a steering vector `v = std · mean_x[ g_m(z+ŵ) − g_m(z) ]` — the
activation displacement for one step "warmer." Then add `α·v` to a readout prompt's
layer-19 last-token activation (add the *delta*, not the lossy full reconstruction),
run the rest of the model, and read a behavioural contrast: `logit(" hot") −
logit(" cold")`, averaged over three readout prompts. Sweep α. **Control:** a random
direction of equal norm.

**Result — a clean causal handle:**

| α (steering strength) | −3 | −2 | −1 | 0 | +1 | +2 | +3 | slope/α |
|---|---|---|---|---|---|---|---|---|
| **legible axis**, logit(hot−cold) | −1.33 | −1.13 | −0.88 | −0.63 | −0.46 | −0.17 | +0.04 | **+0.231** |
| random control (equal norm) | −0.63 | −0.67 | −0.71 | −0.63 | −0.63 | −0.58 | −0.54 | +0.018 |

Steering along the legible temperature coordinate moves the model's hot/cold
preference **smoothly and monotonically** (and through the crossover from "cold" to
"hot"), while an equal-norm random direction is essentially flat — a **~13× steeper
slope** for the concept axis (`cache/steer/steering.png`). So the legible coordinate
isn't just decodable: pushing on it *causally* steers the model in the concept
direction. This upgrades the central claim from "representation" to "control."

**Caveats:** one manifold (temperature), one small model, a single seed; the readout is
a hand-picked contrast pair; and the steering vector is a *mean* linear direction
(local, not the full nonlinear chart map). A stronger version would sweep multiple
manifolds/contrasts and compare the legible axis against the *unaligned* coordinate's
axis, not only a random control.

---

## 9b. Part VIII — Real-model validation: Llama-3.1-8B (`run_all.py` on RunPod)

Up to this point everything was SmolLM2-135M — toy scale, deliberately. The biggest
open question was whether the geometric headline (curved charts beat both PCA and
standard SAEs on curved manifolds; legibility supervision orients them at near-zero
VE cost) survived a *real* model. So we lifted the same end-to-end suite onto
**Llama-3.1-8B layer 16** (d=4096, paper's model family, ungated mirror
`NousResearch/Meta-Llama-3.1-8B`), 3 seeds, 200k background C4 tokens for the standard-SAE
baseline (expansion ×8 → 32,768 features, k=32), on a rented A100.

`prototype/run_all.py` runs the full pipeline unattended — `extract → background → sae
→ fair → legible → iso → adaptive` — into a per-`(model, layer)` cache namespace
(`cache/meta-llama-3.1-8b_L16/`); `RUNPOD.md` is the runbook (transfer, `pip install`
not `uv sync` on a CUDA pod, gating workaround, exposed-TCP rsync). Crash recovery is
free because every stage is idempotent on its output path. The novel-GPU work (`legible
→ iso → adaptive`) finished in **94 minutes** on the A100 (16 + 25 + 53 min); `fair`
was already cached from an earlier session.

### 9b.1 Fair comparison @ N=3 — held-out VE, mean ± std over 3 seeds (the headline)

|                | model | years          | age            | temperature    | colors         | geography       | mean |
|---             |---    |---             |---             |---             |---             |---              |--- |
| **PCA**        | 135M  | 0.39 ± 0.02    | 0.79 ± 0.02    | 0.79 ± 0.02    | 0.74 ± 0.00    | 0.52 ± 0.00     | 0.65 |
| **PCA**        | 8B    | 0.22 ± 0.01    | 0.67 ± 0.02    | 0.88 ± 0.01    | 0.74 ± 0.00    | 0.19 ± 0.00     | 0.54 |
| **SAE-geo (C4)**   | 135M  | 0.09 | 0.06 | 0.09 | 0.09 | 0.04 | **0.07** |
| **SAE-geo (C4)**   | 8B    | 0.06 | 0.04 | 0.05 | 0.03 | 0.03 | **0.04** |
| **SAE-geo (mix)**  | 135M  | 0.10 | 0.27 | 0.50 | 0.53 | 0.20 | 0.32 |
| **SAE-geo (mix)**  | 8B    | 0.10 | 0.23 | 0.27 | 0.37 | 0.07 | 0.21 |
| **factored-lin** | 135M  | 0.09 | −0.07 | 0.05 | 0.57 | 0.12 | 0.15 |
| **factored-lin** | 8B    | −0.01 | −0.01 | −0.06 | −0.02 | 0.02 | **−0.02** |
| **factored-nl**  | 135M  | **0.84** | 0.29 | 0.68 | 0.80 | 0.76 | **0.68** |
| **factored-nl**  | 8B    | 0.31 | 0.65 | **0.88** | **0.78** | **0.61** | **0.64** |

Three things to read off this table:

1. **Standard-SAE shattering reproduces — louder.** The C4-trained SAE is at 0.04 mean
   VE@3 vs PCA's 0.54 — a ~14× collapse, slightly worse than the 9× at 135M. The same
   "low subspace capture for the things we'd want to read off it" story holds, on the
   model family the paper actually used.
2. **Factored-NL still wins on average and crushes PCA on the curved manifolds.**
   At 8B, factored-NL beats PCA by **+0.09 on years** and **+0.42 on geography** —
   the two manifolds where the geometry is most overtly curved (text-relative number
   ordering for years; a 2-D map-like layout for geography). On temperature and colors
   they're tied at PCA's ceiling. On `age` (69 train points, our worst-data manifold)
   PCA edges it. **Mean over the 5 manifolds: 0.64 vs 0.54 PCA**, against 0.68 vs 0.65
   at 135M. Curvature still pays off, and the prize is bigger where curvature is most
   visible.
3. **Factored-LIN collapses at 8B.** Mean VE drops from +0.15 to **−0.02**: a linear
   chart of the same dimensionality as the nonlinear chart can no longer match even
   PCA. The nonlinear chart isn't winning by burning parameters — its linear twin is
   identically parameterised in coord_dim and strictly worse than PCA. The win is in
   the curvature itself.

### 9b.2 Label R² @ N=3 (held-out, kNN; cyclic-aware on colors)

|                | model | years | age   | temperature | colors | geography |
|---             |---    |---    |---    |---          |---     |---        |
| **PCA**         | 135M | 0.77 | 0.93 | 0.97 | 0.38 | 0.67 |
| **PCA**         | 8B   | **0.98** | 0.97 | 0.99 | 0.98 | 0.86 |
| **SAE-geo (C4)** | 135M | 0.20 | 0.86 | 0.71 | 0.28 | 0.63 |
| **SAE-geo (C4)** | 8B   | 0.02 | 0.79 | 0.99 | 0.92 | 0.44 |
| **factored-NL**  | 135M | 0.29 | 0.76 | 0.94 | 0.27 | 0.58 |
| **factored-NL**  | 8B   | **0.99** | 0.94 | 0.99 | 0.73 | 0.95 |

**Surprise (and an honest correction):** at 8B, **PCA's label R² is already nearly
saturated everywhere** (0.86–0.99) — including colors, where at 135M it had been our
hardest case (0.38). Llama-3.1-8B's layer-16 representations are clean enough that a
3-D linear projection is, by itself, *legible* under cyclic-aware scoring. That moves
the goalposts: the unsupervised legibility story we leaned on for the toy model
("PCA doesn't read off label" / "factored coordinate is harder to decode raw") is
*not* the 8B story. The factored-NL coordinate is also high-R² there (mean across the
five 0.92 vs PCA's 0.96), but it doesn't get to win on PCA being unreadable, because
PCA *is* readable here.

The geography line is the cleanest demonstration of why factored still matters:
factored-NL hits **VE 0.61, label R² 0.95** at coord_dim=3, while PCA only manages
**VE 0.19, label R² 0.86** in the same 3 dims. Same dimensionality, more than triple
the variance explained, and the labels still read off the coordinate. PCA's "legible
ceiling" at 8B is real but it sits at a much lower VE than the curved chart.

### 9b.3 Legibility supervision (`legible_coord`) at 8B

> **Read §9b.3, §9b.4, §9b.5 with this caveat first.** At 8B the metrics that
> distinguish "the lever moved" from "noise" — held-out label R² and held-out
> VE@3 — all live in *very narrow bands*. PCA's R² is already 0.86–0.99 across
> all five manifolds (§9b.2), so the supervised R² gains have at most ~0.10 of
> headroom; the factored-NL VE@3 sits in a ~0.05-wide band on the linear-ish
> manifolds. The Pareto plots
> (`cache/meta-llama-3.1-8b_L16/{legible_coord,iso_parsimony,adaptive_parsimony}/pareto.png`)
> reflect this directly: trajectories cross, points cluster, and 3 seeds is **not
> enough** to confidently distinguish a real Pareto move from seed noise inside
> those bands. The tables below report the means we observed and the patterns
> that match the 135M shapes; we are **not** claiming the relative orderings
> within the narrow bands are stable. A future pass should run ≥10 seeds and
> report confidence intervals before any of these 8B numbers is leaned on. The
> headline 8B result that *does* survive this caveat is §9b.1 (the held-out
> VE-vs-N curves), where factored-NL clears the SAE baselines and PCA by margins
> that comfortably exceed seed noise on years and geography.


Same λ-sweep as Part V (`lam ∈ {0, 0.3, 1, 3, 10}`). Mean VE@3 stays flat at **0.64–0.66**
across λ — alignment is essentially free in fidelity, just like at 135M. Label R²:

| λ      | non-cyclic mean (lin R²) | colors kNN R² (cyclic) |
|---     |---                       |---                     |
| 0      | 0.81                     | 0.73                   |
| 0.3    | 0.81                     | 0.84                   |
| 1      | 0.84                     | 0.75                   |
| 3      | 0.84                     | 0.91                   |
| 10     | 0.88                     | **0.93**               |

Colours kNN R² moves from 0.73 → 0.93 across the sweep at a flat VE — the same
*shape* as the 135M run, on the manifold that was hardest there. But the
non-cyclic mean R² moves from 0.81 to 0.88 across the same λ-sweep, well inside
the 3-seed noise band suggested by the Pareto plot. We claim only that the
*pattern* matches the 135M run; we do not claim the specific row-by-row
orderings are stable until a higher-seed pass confirms them.

### 9b.4 Unsupervised isometry + parsimony (`iso_parsimony`) at 8B

| iso | par | mean VE@3 | non-cyclic lin R² | colors kNN R² |
|---  |---  |---        |---                |---            |
| 0   | 0   | 0.64      | 0.81              | 0.73          |
| 0   | 2   | 0.65      | 0.80              | **0.97**      |
| 0   | 8   | 0.61      | 0.86              | 0.77          |
| 1   | 0   | 0.64      | 0.80              | 0.85          |
| 1   | 2   | 0.64      | 0.79              | 0.83          |
| 1   | 8   | 0.64      | 0.84              | 0.77          |

At iso=0, par=2 the colors kNN R² rises to 0.97 at near-flat VE — the visible
exception to the narrow-band picture, and the row that most clearly survives the
3-seed caveat. Beyond that one row, the differences between rows in the
non-cyclic columns (0.79–0.86) and the mean-VE column (0.61–0.65) live in the
noise band. We *do* see the 135M qualified-positive shape — colours improving
when the chart is gently pushed to spend fewer dims — but the row-by-row
strength claims should be retested with more seeds before being leaned on.

### 9b.5 Adaptive parsimony (`adaptive_parsimony`) at 8B — same qualified negative

Per-chart per-dim learned gates with the Matryoshka (1,2,3) dim-cost prior:

```
Held-out VE@3 (mean over 3 seeds, coord_dim=3):
  iso  gate   years    age   temperat   colors   geography   mean*
    0     0    0.22    0.68      0.83     0.57     0.59       0.58
    0     8    0.27    0.68      0.77     0.54     0.62       0.58
    1     0    0.23    0.68      0.83     0.54     0.61       0.58
    1     8    0.22    0.69      0.82     0.53     0.59       0.57
                                                              (mean* excludes cyclic)
Active coord dims (Σ gates of dominant chart, lower = fewer):
    0     0   2.67    2.66    2.65    2.65    2.69
    0     8   2.40    2.31    2.34    1.88    2.12
    1     8   2.49    2.48    2.47    2.34    2.25
```

The active-gate-count column **does** move robustly with `lam_gate` (years 2.67 →
2.40, geography 2.69 → 2.12 at gate=8 with iso=0) — that part is real because
the gates are directly penalised. What the 135M conclusion *claimed* — that
the gates close roughly uniformly without per-manifold differentiation — is
visible in those numbers (no manifold's gate sum stays high while others
collapse). What it claimed *more strongly* — that the differential allocation
fails — also requires the VE@3 column to be tight enough that we can rule out
"the gates moved but the chart found the right dims elsewhere." VE@3 in the
table above varies by ~0.01 in the mean and ≤0.05 per manifold across the grid;
the Pareto plot makes that range look noisy. So:

- **What we can defensibly say at 8B:** the *gate-sum* effect of `lam_gate`
  reproduces (the penalty does close gates roughly uniformly), and a single
  global `lam_gate` does not produce per-manifold dim differentiation. That is
  a method observation, not a "qualified negative for legibility."
- **What we should *not* say at 8B without more seeds:** the strong claim that
  "scale fixes it" is killed because adaptive parsimony's *legibility* outcome
  is the same at 135M and 8B. The legibility-side metrics here (held-out R²)
  are inside the band the §9b.3–5 caveat above already flagged.

So the 8B run *is* consistent with the 135M qualified negative on its
strongest mechanical leg (uniform gate closure → no differential allocation),
and the legibility side reproduces the same *shape* but needs more seeds to
support row-level claims. The "method limit, not scale artefact" framing
survives at the mechanical level; the per-row tightness does not.

### 9b.6 What 8B taught us that 135M couldn't

- **The factored advantage is geometry, not parameter slack.** The linear-chart
  twin collapses at 8B (mean VE −0.02). At 135M `factored-lin` retained a small
  positive margin (mean +0.15) that could plausibly be read as "you gave it more
  parameters than PCA." At 8B that reading dies cleanly: same parameters, same
  routing, same coord_dim, and the linear version loses to PCA.
- **PCA is dramatically more legible at 8B.** Llama-3.1-8B's mid-layer activations
  concentrate cleanly enough that a 3-D linear projection alone hits 0.86–0.99 label
  R² across all five concepts. The 135M "PCA can't read colors" story doesn't
  generalise.
- **Curvature pays off where curvature is visible.** The biggest 8B factored-NL VE
  wins over PCA — years +0.09 and geography +0.42 — are on the two manifolds whose
  geometry is most overtly nonlinear. On effectively-linear `temperature` and
  `colors`, they tie at PCA's ceiling.
- **The adaptive-parsimony qualified negative reproduces at the mechanical
  level**: at 8B too, pushing `lam_gate` closes the gate-sum roughly uniformly
  across manifolds (e.g. years 2.67 → 2.40, geography 2.69 → 2.12) with no
  per-manifold dim differentiation. The legibility-side claims live in a
  narrow-band noise floor — see the §9b.3 caveat — so the *strong* version of
  "scale doesn't fix it" needs more seeds; the mechanical-uniform-closure
  version survives the noise floor.
- **Causal steering at 8B moves the margin but does not cross the decision
  boundary.** §9b.7 below: the legible-axis curve is strictly monotone with slope
  +0.090/α and is ~6× a flat random control, but unlike the 135M run — where the
  same recipe carried `logit(hot−cold)` from −1.33 (cold-dominant) at α=−3 *through
  zero* to +0.04 (hot-dominant) at α=+3 — at 8B every readout in the sweep stays
  negative (−1.31 → −0.79). We modulated the margin, we did not flip the
  prediction. That's a real qualitative difference, not just a magnitude
  attenuation; the 135M language "the legible coordinate is causal control"
  doesn't carry over to 8B verbatim. See §9b.7 for the honest reading.

### 9b.7 Steering at 8B — modulation, not control (`steer.py`)

Same recipe as Part VII (`steer.py`, aligned λ=1 legible temperature axis,
α∈[−3,+3] patched into layer-16 last-token activation, ` hot`−` cold` logit
contrast averaged over the same three readout prompts, equal-norm random
direction as control). **One seed, one manifold, one contrast pair, one model.**
Run on `NousResearch/Meta-Llama-3.1-8B` with `SAE_CACHE_TAG=meta-llama-3.1-8b_L16`;
‖v‖ = 0.88 in raw activation units, dominant chart = 0, contrast tokens
single-token (` hot`=4106, ` cold`=9439). Cache:
`cache/meta-llama-3.1-8b_L16/steer/{steering.png,results.json}`.

| α (steering strength) | −3 | −2 | −1 | 0 | +1 | +2 | +3 | slope/α |
|---|---|---|---|---|---|---|---|---|
| **legible axis**, logit(hot−cold) | −1.31 | −1.27 | −1.15 | −1.04 | −0.98 | −0.88 | −0.79 | **+0.090** |
| random control (equal norm) | −1.08 | −1.08 | −1.08 | −1.04 | −0.98 | −1.02 | −1.02 | +0.015 |

**What is real.** The legible-axis curve is **strictly monotone** across all
seven α and positive-sloped; the random-control curve is flat to within ±0.06
with no monotone trend. The legible axis's slope is **~6× the random control's**.
There IS a direction in the model's activation space, picked by the curved chart's
legible-coordinate decoder, that **the model treats as meaningfully different
from a random direction of the same norm**, and pushing on it shifts a behavioural
readout in the concept's sign.

**What is NOT real (the honest reframe of an earlier draft of this section).**
The 135M Part VII line — *"steering along the legible coordinate moves the model
from cold to hot, ~13× a random control"* — turned out to be doing two distinct
pieces of work that come apart at scale:

- *the slope ratio over random* (~13× at 135M, ~6× at 8B): a comparison of effect
  sizes. This survives at 8B as ~6×.
- *the decision-boundary crossing* (135M `logit(hot−cold)` went −1.33 → +0.04 over
  α∈[−3, +3], actually flipping the model's preferred token): a categorical claim
  about whether the steering induces a behavioural switch. **This does NOT survive
  at 8B.** Across the same α range the contrast stays −1.31 → −0.79 — the model
  prefers "cold" under every steering condition tested. The picked axis modulates
  the *margin* by which it prefers cold; it does not push it across to "hot."

Calling that "the legible coordinate is causal control" is more than the 8B data
supports. **The defensible 8B claim is "modulation, not control."** The picked
axis has a directional, monotone, above-noise causal effect on the readout, and
that's a non-trivial finding — but it falls short of demonstrating behavioural
control of the model's classification.

**Three non-exclusive readings of *why* it attenuates** (ordered most → least
supported, all speculative without further runs):

1. **The "PCA already reads it" effect from §9b.2 has a causal echo.** At 8B,
   temperature's PCA label R² is 0.99 — the concept is largely linearly readable
   from the activation itself. When the concept direction is already nearly a
   linear axis, the curved chart still picks a *valid* causal direction but no
   longer an *exceptional* one over equal-norm random; and the magnitude needed
   to flip the prediction may simply be larger than α=+3 in our raw-activation
   units.
2. **More downstream non-linearity between patch and readout.** Layer 16 of 32
   at 8B leaves 16 transformer layers to absorb / reshape a fixed-magnitude
   perturbation; layer 19 of 30 at 135M leaves 11. More layers may smear the
   intervention.
3. **bf16 at the patch site.** ‖v‖ = 0.88 is large relative to bf16's quantum at
   that activation magnitude, so this is unlikely to dominate — but it cannot be
   *excluded* without a fp32 control we did not run.

**What would actually upgrade this to "control" at 8B.** Try larger α (extend the
sweep until the prediction either flips or we can characterise an asymptote);
sweep multiple seeds for the legible chart to confirm ‖v‖ and the slope are
stable rather than seed-luck; replicate on years (richer curvature at 8B per
§9b.1 — contrast pair like ` twenty` vs ` nineteen` to dodge multi-token year
strings); compare against the *unaligned* coordinate's axis rather than only a
random control. None of those were run for this section.

**Caveats inherited from Part VII** (still applicable, sharpened): one manifold,
one contrast pair, one seed, **one model**, mean linear direction (not the full
chart map), no α-range exploration. The 135M Part VII result is itself a single
proof-of-concept; the 8B result is one more single proof-of-concept under the
same recipe, and the two together are best read as a *two-point* causal signal
whose qualitative content (modulation, with decision crossing at 135M but not
at 8B) varies with scale.

**Limitations of this run.** Single layer (16), one 8B model, 3 seeds. Steering
(Part VII) re-ran at 8B for one seed/manifold — see §9b.7 — with a qualitatively
matching but quantitatively attenuated result; the 135M slope/ratio is the upper
end of what to expect, not a universal constant. We used `NousResearch/Meta-Llama-3.1-8B`
(ungated mirror of the official weights) rather than the gated `meta-llama/Llama-3.1-8B`;
the weights are the same architecture and were verified by hash on prior published
reports, but the provenance is one step indirect. Layer 16 was picked as a default
mid-layer; the layer-sensitivity sweep (#13 follow-on) is the natural next test.

---

## 10. What worked, what didn't (at a glance) — **corrected 2026-05-29**

| Step | Outcome |
|---|---|
| Reproduce dilution/shattering | ✅ Reproduced (SAE codes plateau ~17% vs PCA 94% @16-D) |
| Curved charts vs linear charts at matched coord_dim | ✅ Curvature delta NL−LIN intact: +0.30 to +2.00 across manifolds. The straightness of atoms IS the bottleneck — the cleanest isolation of the paper's thesis. |
| Curved charts vs PCA at matched coord_dim | ◐ Narrow: factored-NL clears PCA-3 on **years only (+0.12)**. On geography/colors/temperature/age it lands below PCA-3. (Bug-era claim: +0.45 / +0.25 / +0.07 — retracted.) |
| Hard routing (straight-through one-hot) vs soft mixture | ✅ Substantial unsupervised-legibility lift: years R² 0.16→0.57, geography 0.41→0.54, colors 0.13→0.24 just by switching the router. Previously hidden by the bug. |
| Weak label alignment → legible? | ✅ λ=1 crosses PCA on 4/5 manifolds at VE cost ≈0 (0.44→0.46 mean): years 0.91 (PCA 0.75), geography 0.82 (PCA 0.60), temperature 0.93 (PCA 0.92), colors 0.37 (PCA 0.30). Age (0.75 vs PCA 0.91) is the one miss. |
| Cyclic scoring (colours) | ✅ Honest scoring (artifact removed); ⚠️ scoring alone insufficient |
| Cyclic alignment (colours) | ✅ Crosses PCA at lam≈1 (0.37 vs 0.30) |
| Unsupervised isometry alone | ❌ Negative; arc-length ≠ linear-in-coordinate. Replicated in iso_parsimony's iso=1, pars=0 row (mean VE −0.07). |
| Isometry + parsimony (global) | ◐ Qualified positive narrowed: parsimony pushes **years past PCA label-free (0.57→0.83)**. Bug-era "geography over-collapses" was an artifact and is gone. New failure mode: parsimony hurts age (0.73→0.48). |
| Adaptive parsimony (learned per-dim gates) | ◐ Qualified negative holds. Years lift now real (0.21→0.50 at gate=8, vs bug-era +0.03) but doesn't cross PCA. Gate differentiation exists at gate=2 but is **routing-driven (geography 1.21 active, years 1.94), not intrinsic-dim-driven** — opposite of the original hypothesis. |
| Steering at 135M (Part VII) | ✅ Causal *control*: monotone hot/cold shift that crosses the decision boundary, ~13× a random control. **Unaffected by the bug** — `steer.py` discards `fve`. |
| **Real-model validation (Llama-3.1-8B, layer 16)** | ⚠️ **All 8B numbers still pre-fix.** §9b.1–9b.6 were generated with the buggy `factored_eval`; they are not yet re-run under hard routing. Treat them as inflated until the RunPod re-run lands. |
| Steering at 8B (§9b.7) | ◐ Modulation, not control: monotone +0.090/α, ~6× random, never crosses the decision boundary across α∈[−3, +3] (135M's same sweep did). One seed/manifold/contrast — proof-of-concept only. **Unaffected by the bug.** |

---

## 11. Limitations (read this before believing anything)

- **The factored_eval bug (2026-05-29).** For most of the project the headline
  fidelity numbers were generated by an internally-inconsistent eval that scored
  reconstruction over the K-chart mixture (effective ~36-D subspace at K=12,
  coord_dim=3) while taking coords for label decoding from a single dominant chart
  (3-D). The "matched coord_dim" framing was false on the factored-NL VE side.
  Fix landed 2026-05-29: dominant-chart-only recon + hard-routing training (so the
  model is actually a true atlas, not a soft mixture). The 135M experiments above
  reflect the fix; the 8B experiments in Part VIII do **not yet**. The biggest
  lesson is methodological (saved as `research-comparison-smell` memory): when a
  matched-dim comparison shows a margin big enough to be the headline, audit what
  the matching constraint constrains *at metric time*, not just in the function
  signature — branch-dimension is the trap. See the retraction banner at the top
  of this document for the propagation analysis and which specific claims moved.
- **Toy scale (Parts I–VII).** SmolLM2-135M (not Llama-3.1-8B), ~5k points total,
  manifolds as small as 56–199 points. `age` (69 train points) is consistently
  unreliable; `days` was dropped entirely. Three seeds is few; the steering result is
  a single seed/manifold at each scale. **Part VIII** lifts the suite to Llama-3.1-8B
  layer 16 — one layer, one 8B model, 3 seeds, **and one seed/manifold for steering
  (§9b.7)** — so a real-model toehold on every leg of the claim, not a sensitivity
  sweep.
- **Mostly we changed the lens, not the model.** Parts I–VI re-represent fixed
  activations with different SAEs/decoders. Part VII / §9b.7 are the exceptions
  that intervene on the model — on one manifold, one contrast pair, two scales —
  so treat them as a two-point causal signal whose *qualitative content*
  (control at 135M, modulation only at 8B; see §9b.7) varies with scale.
- **The 8B Pareto results (§9b.3–5) are within a noise floor.** PCA at 8B is
  already so legible (R² 0.86–0.99) and the factored VE@3 ranges are so narrow
  that the supervised-alignment Pareto trajectories in
  `cache/meta-llama-3.1-8b_L16/{legible_coord,iso_parsimony,adaptive_parsimony}/pareto.png`
  cross and cluster rather than tracing a clean frontier. 3 seeds is not enough
  to confidently distinguish a real Pareto move from seed-luck inside those
  bands. The headline 8B result that survives this (§9b.1) is the held-out
  VE-vs-N curves, where factored-NL clears the SAE baselines and PCA by margins
  that comfortably exceed seed noise on years and geography.
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

## 12. The five-point bar — status, and what's left

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
real model, **bonus:** a causal/steering leg) — **status after the 2026-05-29 fix:**

- **Geometry fidelity at 135M** — ◐ narrow: factored-NL beats PCA-3 on **years
  only (+0.12)**. The bug-era "wins on years/geography/colors" reduces to one
  manifold under matched-dim. The curvature-vs-linear internal lever (NL−LIN
  delta +0.30 to +2.00) holds cleanly.
- **Interpretability at 135M** — ✅ supervised (4/5 manifolds cross PCA at λ=1,
  near-zero VE cost); ◐ unsupervised (parsimony pushes years past PCA; nothing
  else crosses). Hard routing alone provides a substantial unsupervised baseline
  lift (years 0.16→0.57) — a previously-hidden finding the bug had masked.
- **Causal/steering leg — split by scale.** At 135M (Part VII) the legible
  axis is a causal *control* handle: monotone hot/cold shift that crosses the
  decision boundary, ~13× a random control. At 8B (§9b.7) the same recipe
  produces *modulation only*: monotone +0.090/α, ~6× random, but the readout
  never flips across α∈[−3, +3]. One seed/manifold/contrast at each scale;
  proof-of-concept only. **Both legs unaffected by the bug** (`steer.py` discards
  the VE quantity).
- **"Ideally on a real model" — currently in regression.** Part VIII §9b.1–9b.5
  were generated pre-fix; the 8B factored VE numbers are bug-inflated by the same
  mechanism (mixture-over-K-charts vs dominant-chart-3-D). The RunPod re-run is
  the immediate next item if the 8B claims are to be defended. Steering at 8B
  (§9b.7) is unaffected.

**Explicitly still on the list / things we have not yet done:**

1. ~~**Causal / steering leg**~~ ◐ **Two single-shot proofs at two scales**
   (Part VII at 135M / §9b.7 at 8B, `steer.py`). At 135M the sweep crosses the
   decision boundary (control); at 8B it doesn't (modulation only). The
   *qualitative* claim is therefore split by scale. The follow-up that would
   collapse it back together: extend the α-range at 8B until the prediction
   either flips or asymptotes; multi-seed the legible chart; multi-manifold and
   multi-contrast sweeps; compare against the *unaligned* axis (not only a random
   control); fp32 vs bf16 control to pin reading 3 of §9b.7.
2. ~~**Coordinate-vs-label visualization.**~~ ✅ **Done** (`viz_coord.py`):
   `cache/viz/coord_vs_label.png` (predicted-vs-true label, unsupervised vs aligned)
   and `cache/viz/colors_circle.png` (the partial hue loop). The diagonal-tightening
   makes the legibility claim falsifiable at a glance.
3. ~~**Intrinsic-dimension-adaptive parsimony**~~ ◐/❌ **Attempted** (§8.3,
   `adaptive_parsimony.py`): per-chart learned per-dim gates with a fixed per-dim
   (Matryoshka-ordered) cost. **Qualified negative** — it cures the global knob's
   over-collapse (geography/colours survive) but the gates close ~uniformly, so the
   intended *differential* dim allocation never materializes and legibility is flat in
   the penalty (at 135M scale the manifolds are effectively ~1-D to a nonlinear chart).
   Incidental positive: coordinate normalization is a label-free fidelity↔legibility
   *trade* (years R² 0.28→0.64 at the cost of VE 0.79→0.45). The unsupervised frontier
   stays open; revisiting on a genuinely multi-D (real-model) geometry is the natural
   next test.
4. **A proper isometric-AE** (exact Jacobian + encoder pseudo-inverse term) instead of
   the finite-difference surrogate, to give isometry its fairest shot.
5. ~~**Validate on a real model**~~ ✅ **Done** (Part VIII / `run_all.py` on RunPod):
   Llama-3.1-8B layer 16, 3 seeds. The 135M result is **not** a small-model artefact —
   shattering and the curved-chart advantage both reproduce, the latter more strongly
   on overtly-curved manifolds (geography +0.42 mean VE over PCA at coord_dim=3).
   What's left: layer-sensitivity sweep, an 8B Llama re-run of steering, and a second
   model family (e.g. Qwen or Mistral) for cross-architecture evidence.
6. **In-the-wild router.** Train the factored model on *background* activations (not the
   curated mixture) and test whether charts discover manifolds unsupervised — the claim
   "the router is the learned analogue of feature clustering" is currently only shown on
   curated data.
7. **Group-sparse (top-k charts)** selection instead of soft softmax, for a more
   SAE-like, genuinely sparse object.
8. **Statistical rigor:** more seeds, confidence intervals, and ideally larger manifolds
   (drop/augment `age`, `days`). The 8B Pareto plots in §9b.3–5 are the most
   acute pressure point for this — see the §9b.3 caveat.
9. **Anchor against the Goodfire paper at 8B.** Before leaning further on any of
   the 8B numbers, replicate **one** Goodfire-paper figure on Llama-3.1-8B with
   the existing scripts (`subspace_capture.py` plot for Fig 4, `manifold_viz.py`
   for Fig 1) and confirm the headline shattering / 3-D PCA helix come out
   matching the paper's numbers within our pipeline. This is a cheap sanity
   anchor (~10 min on A100, no new code) and gates whether the §9b.3–5 noise
   floor is "expected at this seed count" or "our 8B pipeline is producing
   weird numbers." Pending — pod was stopped before this anchor ran.

---

## 13. Reproducibility appendix

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
| Isometry + parsimony (global) | `uv run python prototype/iso_parsimony.py --seeds 0 1 2 --lam-isos 0 1 --lam-pars 0 2 8` | `cache/iso_parsimony/` |
| Adaptive parsimony (gated) | `uv run python prototype/adaptive_parsimony.py --seeds 0 1 2 --lam-isos 0 1 --lam-gates 0 2 4 8` | `cache/adaptive_parsimony/` |
| Legibility figures | `uv run python prototype/viz_coord.py` | `cache/viz/` |
| Steering (loads model) | `SAE_MODEL_NAME=HuggingFaceTB/SmolLM2-135M SAE_LAYER=19 uv run python prototype/steer.py` | `cache/steer/` |
| **Full suite, one command** | `uv run python prototype/run_all.py --layers 19 --seeds 0 1 2` | `cache/<model>_L<layer>/…` |

The last row is the **end-to-end driver** (`prototype/run_all.py`): model-agnostic via
the same env vars, it runs `extract → background → sae → fair → legible → iso →
adaptive` per layer into a per-`(model, layer)` cache tag, skipping finished stages.
It's how the real-model validation (#13) runs on a cloud GPU — see **`RUNPOD.md`** for
the Llama-3.1-8B runbook (transfer, torch/CUDA + HF-gating caveats, layer/seed sweeps).

**Key files:** `data.py` (manifolds + extraction), `train_sae.py` / `saes.py`
(standard BatchTopK SAE), `subspace_capture.py` (the paper's metric),
`prototype/factored_sae.py` (the atlas SAE), `prototype/fair_comparison.py` (the
fair test + shared eval helpers), `prototype/legible_coord.py`,
`prototype/iso_coord.py`, `prototype/iso_parsimony.py`,
`prototype/adaptive_parsimony.py` (gated, intrinsic-dim-adaptive),
`prototype/viz_coord.py`
(legibility figures), `prototype/steer.py` (causal steering). Detailed reproduction notes
in `REPRODUCTION.md`; prototype notes in `prototype/README.md`.

---

*Last updated: 2026-05-28 (added Part VIII — Llama-3.1-8B real-model validation).
Open threads tracked in §11/§12.*
