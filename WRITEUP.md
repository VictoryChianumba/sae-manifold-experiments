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
> **Update 2026-05-30 — 8B re-run + anchor.** The 8B suite (Part VIII)
> has been fully re-run with the hard-routed atlas and 10 seeds:
> - §9b.1 fair comparison was unaffected by the noise band caveat and
>   already cleared seed noise — its numbers stand.
> - §9b.3–5 now have **per-row 2·SEM-significant** findings (see
>   `cache/meta-llama-3.1-8b_L16/seeds10_summary.md`); the "Pareto lives
>   in a noise band" caveat from earlier drafts is **retracted**.
> - §9b.7 steering is reframed around multi-seed / unaligned-axis
>   results: the "modulation, not control at 8B" single-seed claim
>   collapses — 1/3 seeds crosses zero (control), 2/3 stay at modulation;
>   the seed distribution is the story.
> - §9b.8 (new) records the #20 anchor: subspace_capture + manifold_viz
>   reproduce the paper's Fig 4 / Fig 1 shapes at 8B, so the pipeline is
>   confirmed and §9b's claims can be defended.
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
- **The legible coordinate is causal — across both scales, but the
  effect's magnitude is highly seed-dependent.** Three independent SAE
  training seeds at 8B give slopes of +0.022, +0.090, +0.288 per α —
  all positive, all sign-consistent, but a ~13× spread in magnitude.
  Seed 2 actually **crosses zero at α=+4** (control, like 135M did);
  seeds 0 and 1 stop at modulation. So the prior single-seed
  "modulation, not control" framing for 8B was over-confidently negative
  on a wide seed distribution. The **cleanest causal-axis demonstration**
  this leg has produced: an *unaligned-axis baseline* (identical
  architecture and routing, lam_label=0 instead of 1) has slope
  −0.031 (flat/wrong direction), so the legibility supervision is what
  flips a same-architecture random-ish axis into a sign-consistent causal
  one. bf16 vs fp32 doesn't matter (slopes identical). *(unaffected by
  the factored_eval bug — steer.py discards the bug-affected VE
  quantity.)*
- **The 8B pipeline reproduces the paper's signature anchors** (§9b.8,
  added 2026-05-30): `subspace_capture.py` on Llama-3.1-8B/L16 produces
  the canonical Fig-4 shattering shape (stat-SAE plateaus ~28% at k=64,
  PCA hits ~93%, random baselines flat), and `manifold_viz.py` produces
  the years 3D-PCA chronological loop. So §9b's claims rest on a
  pipeline that's producing the paper's numbers on the paper's model.

Throughout, the honest caveat: this is a **toy** (135M model, ~5k
points, a research SAE, not a scalable one), and we changed the **lens
(the SAE), not the model** — except for Part VII / §9b.7, which
intervene on the model.

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

**Result (3 seeds, hard-routed — regenerated 2026-06-08): it does not work.** The
non-cyclic mean R² never cleanly exits seed noise (0.61 at lam=0 → 0.70 peak at
lam=3 → 0.60 by lam=30); no manifold shows a *sustained* crossing of its PCA
reference (years marginally edges it at lam=3, 0.78 vs PCA 0.75, but within seed
noise and not held at other weights), and any weight large enough to reshape the
coordinate **destroys reconstruction** (held-out VE mean 0.45 → −0.43 at lam=10,
−0.12 at 30; geography VE craters to −2.00).

> **Numbers corrected 2026-06-08.** `iso_coord.py` was the one file in the atlas
> suite never migrated to hard routing — it trained a soft K-chart mixture but was
> scored on the dominant chart only (the confound the retraction banner describes).
> It now uses `hard_routing=True` and both scales were re-run. The earlier draft's
> figures (mean R² 0.55→0.63, VE 0.68→0.32) were soft-routed and are superseded by
> the values above. The **verdict is unchanged** — isometry alone neither buys
> legibility past PCA nor preserves fidelity — and is independently corroborated by
> §8.2's hard-routed iso=1/pars=0 row.

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

### 9b.3 Legibility supervision (`legible_coord`) at 8B — 10-seed re-run

> **Earlier drafts of §9b.3–5 carried a "Pareto results live in a noise
> band, ≥10 seeds needed before any row is leaned on" caveat. The
> 10-seed re-run (2026-05-30, `prototype/run_all.py --seeds 0..9` on a
> fresh A100, ~72 min total) **retracts that caveat**. The numbers
> below are mean ± SEM over 10 seeds; "↑" / "↓" mark rows where the
> *paired* (SAE − PCA) difference exceeds 2·SEM (~95% paired CI). Full
> per-cell tables in `cache/meta-llama-3.1-8b_L16/seeds10_summary.md`
> (generated by `prototype/summarize_seeds.py`).**

Same λ-sweep as Part V (`lam ∈ {0, 0.3, 1, 3, 10}`). Held-out VE@3 stays
flat at 0.54–0.57 across λ — alignment is essentially free in fidelity,
matching the 135M shape. Label R² (lin) per row:

| λ     | years           | age             | temperature     | colors (cos/sin)   | geography       | mean*    |
|---    |---              |---              |---              |---                 |---              |---       |
| 0     | 0.940 ± 0.016 · | 0.868 ± 0.050 ↓ | 0.965 ± 0.005 · | 0.899 ± 0.014 ·    | 0.523 ± 0.034 · | 0.824    |
| 0.3   | 0.947 ± 0.010 · | 0.921 ± 0.034 · | 0.966 ± 0.005 ↓ | 0.914 ± 0.004 ↓    | 0.522 ± 0.044 · | 0.839    |
| 1     | 0.947 ± 0.009 · | 0.918 ± 0.022 ↓ | 0.955 ± 0.008 ↓ | 0.915 ± 0.004 ↓    | **0.567 ± 0.038 ↑** | 0.846 |
| 3     | 0.947 ± 0.009 · | 0.927 ± 0.032 · | 0.968 ± 0.006 · | 0.918 ± 0.005 ·    | **0.613 ± 0.034 ↑** | 0.864 |
| 10    | **0.963 ± 0.005 ↑** | 0.928 ± 0.034 · | 0.979 ± 0.003 · | 0.929 ± 0.004 · | **0.776 ± 0.010 ↑** | 0.912 |
| **PCA ref** | 0.948 ± 0.003 | 0.971 ± 0.003 | 0.974 ± 0.001 | 0.925 ± 0.001    | 0.472 ± 0.010   | 0.841 |

Three findings the CIs nail:

1. **Geography is a clean, large lift** at every λ ≥ 1, with a ~30·SEM
   gap at λ=10 (0.776 ± 0.010 vs PCA 0.472 ± 0.010). This is the
   cleanest crossing of PCA the project has produced and is the
   load-bearing positive at 8B.
2. **Years also crosses PCA** at λ=10 (0.963 ± 0.005 vs PCA 0.948 ±
   0.003) — marginal but 2·SEM-clean.
3. **Most non-geography cells stay inside the PCA band (·)** at
   2·SEM. Colours and temperature show 2·SEM-significant *negative*
   moves at small λ (the supervision penalty briefly costs lin-R² where
   PCA is already 0.92–0.97); they recover by λ=10 to inside-band. The
   mean\* climbs 0.82→0.91 across λ, but per-column that climb is
   concentrated on geography. **Honest framing:** legibility supervision
   at 8B is a *refinement* on linear-ish manifolds (PCA already 0.92–0.99)
   and a *rescue* on geography (PCA's 0.47 → 0.78 with supervision); the
   earlier "improves mean* over PCA at every λ" reading was an
   averaging artifact.

### 9b.4 Unsupervised isometry + parsimony (`iso_parsimony`) at 8B — 10-seed re-run

10-seed mean ± SEM, paired-difference flag vs PCA at 2·SEM (~95% CI):

| iso | pars | years           | age             | temperature     | colors          | geography        |
|---  |---   |---              |---              |---              |---              |---               |
| 0   | 0    | 0.940 ± 0.016 · | 0.868 ± 0.050 ↓ | 0.965 ± 0.005 · | 0.899 ± 0.014 · | 0.523 ± 0.034 ·  |
| 0   | 2    | 0.942 ± 0.004 · | 0.960 ± 0.013 · | 0.955 ± 0.009 ↓ | 0.899 ± 0.014 · | 0.527 ± 0.035 ·  |
| 0   | 8    | 0.946 ± 0.010 · | 0.957 ± 0.015 · | 0.960 ± 0.008 · | 0.867 ± 0.030 · | 0.525 ± 0.032 ·  |
| 1   | 0    | 0.919 ± 0.020 · | 0.902 ± 0.041 · | 0.963 ± 0.007 · | 0.920 ± 0.002 ↓ | **0.542 ± 0.024 ↑** |
| 1   | 2    | 0.934 ± 0.011 · | 0.947 ± 0.013 · | 0.962 ± 0.008 · | 0.875 ± 0.037 · | 0.497 ± 0.054 ·  |
| 1   | 8    | 0.933 ± 0.015 · | 0.894 ± 0.044 · | 0.963 ± 0.008 · | 0.828 ± 0.041 ↓ | 0.488 ± 0.045 ·  |
| **PCA ref** | | 0.948 ± 0.003 | 0.971 ± 0.003 | 0.974 ± 0.001 | 0.925 ± 0.001 | 0.472 ± 0.010 |

**The headline at 10 seeds: a clean unsupervised PCA crossing on
geography.** `iso=1, pars=0` reaches lin-R² 0.542 ± 0.024 vs PCA 0.472
± 0.010 — a ~3·SEM gap on the **paired** difference, **without using
labels**. This is the first 2·SEM-significant unsupervised crossing of
PCA the project has shown at any scale. Isometry alone (`iso=1, pars=0`
== iso-only since parsimony is off) is the lever; adding parsimony
narrows the gap (and at `iso=1, pars=8` colours drops below PCA at
2·SEM — too much parsimony eats the cyclic structure).

The earlier-draft claim that "iso+parsimony **over-collapses age**" is
softer at 10 seeds than at 3: the relevant row (`iso=1, pars=8`) hits
age 0.894 ± 0.044, a 0.077-point drop from PCA's 0.971 ± 0.003 — real
in direction, **not 2·SEM-significant** in the paired test. So the
qualitative shape ("age dim doesn't survive heavy parsimony") holds but
the magnitude is unstable across seeds.

### 9b.5 Adaptive parsimony (`adaptive_parsimony`) at 8B — 10-seed re-run

Per-chart per-dim learned gates with the Matryoshka (1,2,3) dim-cost
prior. 10-seed mean ± SEM, paired vs PCA at 2·SEM:

| iso | gate | years           | age             | temperature     | colors           | geography        |
|---  |---   |---              |---              |---              |---               |---               |
| 0   | 0    | 0.940 ± 0.008 · | 0.931 ± 0.021 · | 0.963 ± 0.008 · | 0.772 ± 0.054 ↓  | 0.528 ± 0.042 ·  |
| 0   | 2    | 0.935 ± 0.008 · | 0.931 ± 0.026 · | 0.966 ± 0.009 · | 0.774 ± 0.050 ↓  | **0.570 ± 0.042 ↑** |
| 0   | 4    | 0.936 ± 0.008 · | 0.925 ± 0.025 · | 0.966 ± 0.007 · | 0.756 ± 0.046 ↓  | 0.562 ± 0.051 ·  |
| 0   | 8    | 0.933 ± 0.009 · | 0.927 ± 0.023 · | 0.946 ± 0.025 · | 0.716 ± 0.056 ↓  | **0.575 ± 0.039 ↑** |
| 1   | 0    | 0.927 ± 0.016 · | 0.945 ± 0.020 · | 0.946 ± 0.008 ↓ | 0.767 ± 0.051 ↓  | 0.451 ± 0.061 ·  |
| 1   | 2    | 0.925 ± 0.012 · | 0.955 ± 0.012 · | 0.940 ± 0.011 ↓ | 0.710 ± 0.061 ↓  | 0.508 ± 0.041 ·  |
| 1   | 4    | 0.933 ± 0.011 · | 0.963 ± 0.008 · | 0.947 ± 0.006 ↓ | 0.745 ± 0.058 ↓  | 0.441 ± 0.061 ·  |
| 1   | 8    | 0.932 ± 0.012 · | 0.957 ± 0.011 · | 0.942 ± 0.008 ↓ | 0.712 ± 0.052 ↓  | 0.446 ± 0.050 ·  |
| **PCA ref** | | 0.948 ± 0.003 | 0.971 ± 0.003 | 0.974 ± 0.001 | 0.925 ± 0.001 | 0.472 ± 0.010 |

The CIs reshape the prior reading:

1. **A new modest unsupervised positive on geography**: `iso=0,
   gate ∈ {2, 8}` cross PCA at 2·SEM (0.570 ± 0.042, 0.575 ± 0.039 vs
   PCA 0.472 ± 0.010). Same direction as iso_parsimony's iso=1/pars=0
   crossing, but a different mechanism (per-dim gate sparsity vs
   isometry). At 10 seeds, the unsupervised lever space for geography
   is wider than the 3-seed plot suggested.
2. **Colors is robustly hurt** by adaptive gating: every iso=0 row
   shows 2·SEM-significant ↓ on colors (lin-R² 0.71–0.77 vs PCA 0.925
   ± 0.001). The gates compress the chart in a way that breaks colour's
   cyclic structure. This is a real negative finding, not noise.
3. **iso=1 uniformly hurts temperature** at 2·SEM (rows iso=1 / gate=*
   land at 0.94 vs PCA 0.97). The isometry penalty doesn't help when
   the underlying manifold is already linear.
4. The **gate-sum mechanical claim** ("a single global `lam_gate`
   closes gates roughly uniformly across manifolds") is still
   supported by the active-dim column in `results.json`, unchanged
   from the 3-seed result.

Net: adaptive parsimony at 8B picks up a small unsupervised
geography win that the 3-seed Pareto plot couldn't distinguish from
noise; it pays for that with colours-hurt and (under iso=1)
temperature-hurt rows that the same CI test cleanly identifies. The
strong "scale fixes it" reading is still killed (no general legibility
rescue across manifolds), but adaptive is no longer purely a
qualified-negative — it has at least one 2·SEM-significant
unsupervised positive at this scale.

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
- **The adaptive-parsimony qualified negative reproduces at the
  mechanical level**: at 8B too, pushing `lam_gate` closes the gate-sum
  roughly uniformly across manifolds with no per-manifold dim
  differentiation. **But adaptive is no longer purely a
  qualified-negative at 10 seeds**: `iso=0, gate ∈ {2, 8}` crosses
  PCA's lin-R² on geography at 2·SEM (0.570/0.575 ± 0.04 vs PCA 0.472
  ± 0.01) — a small unsupervised positive that the 3-seed plot
  couldn't distinguish from noise.
- **The first 2·SEM-significant unsupervised PCA crossing.**
  iso_parsimony at `iso=1, pars=0` reaches lin-R² 0.542 ± 0.024 on
  geography vs PCA 0.472 ± 0.010 — a clean, label-free, paired-CI win.
  The unsupervised legibility frontier is no longer closed at 8B; on
  the manifold whose geometry is most overtly nonlinear, isometry
  alone (without parsimony or labels) reads PCA-better.
- **Steering at 8B is causal but seed-dependent over a large range.**
  §9b.7 below: 3 independent legible-axis seeds give slopes +0.022,
  +0.090, +0.288 per α — all positive, sign-consistent, but ~13×
  spread. Seed 2 **crosses zero at α=+4** (control), seeds 0/1 stop
  at modulation. The unaligned-axis baseline (same architecture,
  lam_label=0) is flat-or-wrong-direction across all seeds, so the
  legibility supervision IS what creates the causal axis — the
  cleanest causal-axis demonstration the project has produced. fp32
  vs bf16 indistinguishable; the asymptote (when it asymptotes) isn't
  a precision artifact. See §9b.7 for the multi-seed table.

### 9b.7 Steering at 8B — the legibility supervision IS the causal lever (`steer.py`, `steer22.py`)

> **An earlier draft of this section framed the 8B result as "modulation,
> not control" on the basis of a single seed (α∈[−3,+3] → logit −1.31 →
> −0.79; never crosses zero). The 2026-05-30 multi-seed + wider-α +
> unaligned-axis + fp32-control follow-up (`prototype/steer22.py`)
> reframes that. The seed distribution is the story.**

**Recipe (unchanged from Part VII).** Train the aligned (λ=1) factored
SAE on the manifold mixture; for the target manifold, build the
**legible axis** ŵ = (unit) probe direction that decodes the label;
project ŵ through the chart's nonlinear decoder to a raw-activation
steering vector v = std·mean_x[g_m(z+ŵ) − g_m(z)]; patch the readout
prompt's layer-16 last-token activation x' = x + α·v; read
`logit(' hot') − logit(' cold')` averaged over three readout prompts.

**Multi-seed (3 SAE training seeds, temperature target, α∈[−6,+6]):**

| seed | ‖v‖ | dom chart | logit(α=0) | logit(α=+6) | Δ at α=+6 | slope/α | crosses zero? |
|---   |---  |---        |---         |---          |---         |---      |---           |
| 0    | 0.69 | 2 | −1.04 | −0.94 | +0.10 | +0.022 | ❌ modulation |
| 1    | 0.81 | 0 | −1.04 | −0.54 | +0.50 | +0.090 | ❌ modulation |
| 2    | 0.57 | 0 | −1.04 | **+0.83** | +1.87 | **+0.288** | **✅ control at α=+4** |

All three seeds **positive-slope and sign-consistent**, but a ~13×
spread in magnitude across just 3 seeds. The prior session's single-seed
"slope +0.090" was the *median*, not an outlier in either direction.
Seed 0 (the one used in the prior session's 8B α=±3 run, and in our
first wider-α run) is the weakest; seed 2 reaches zero-crossing well
before α=+6.

**The cleanest control we have: the *unaligned-axis* baseline.** A
factored SAE with `lam_label=0` (no label supervision) has the *same
architecture and routing* as the aligned one — same charts, same
router, same coord_dim, just no concept-aligned probe. We build its
"legible axis" the same way and run the same sweep. Seed 0, same
target:

| axis (seed 0)            | slope/α | Δ at α=+6 | direction |
|---                       |---      |---        |---        |
| **legible** (λ=1)        | +0.022  | +0.10     | sign-consistent positive |
| **unaligned** (λ=0)      | **−0.031** | **−0.19** | flat / wrong sign |
| random (equal-norm)      | +0.013  | +0.04     | flat |

The unaligned axis behaves like the random control — flat or slightly
wrong direction — even though it shares the legible axis's architecture
and routing. **The legibility supervision is what creates the causal
direction**; it is not an artifact of "any direction from a curved
chart's decoder pushes the readout." That's the strongest "axis is
causal" claim this project has produced.

**Second target, ` twenty` vs ` nineteen` (years contrast):** **does
not replicate**. All 3 legible seeds give slope ≈ −0.025 (wrong
direction; +α should push toward later years / " twenty"); the
unaligned axis tracks the legible axis closely (slope −0.026). Likely
contributors: (a) years splits across multiple charts at 8B in the
factored SAE — the dominant chart varies seed-to-seed (10 / 0 / 7), so
there is no single "year direction" the chart is isolating, (b) " twenty"
/ " nineteen" are polysemous tokens (twenty dollars, twenty minutes; nineteen
ninety-five). Temperature's `' hot'` / `' cold'` are cleaner. So
"axis is causal" is *target-specific* on this evidence: clean on
temperature, inconclusive on years.

**fp32 vs bf16:** identical (slope +0.022 in both for seed=0
temperature). The α=+5 asymptote (when it does asymptote, as in seeds
0 and 1) is not a precision artifact.

**What this collapses from the prior framing.** The single-seed "8B is
modulation, 135M was control — that's a qualitative gap" reading was
built on n=1 sample from a wide distribution. At n=3 seeds, 1 of 3
**crosses zero** (control), 2 of 3 stop at modulation — i.e., the
scale-vs-seed split is "control sometimes, modulation usually" at 8B.
135M was also tested at n=1; the same seed-luck argument applies there.
The honest reading: **the qualitative gap between scales is not
supported on this evidence; what we have is a seed-noise distribution
whose modes happen to include control.**

**What survives and is load-bearing for the writeup:**
1. The legible axis is *causally axis-specific* at 8B — every legible
   seed positive-slope, the unaligned-axis baseline flat/wrong-sign.
2. The magnitude of the causal effect is highly seed-dependent — a
   single seed cannot be cited for "the slope" without a CI.
3. fp32/bf16 doesn't matter; α-range doesn't change the sign.
4. The years contrast doesn't replicate temperature — single-target
   evidence, not "axis is causal in general at 8B."

**Caveats.** 3 seeds is still few; the *direction* of the legible
effect is robust (all positive) but the magnitude has a 13× spread.
One target (temperature), one model, one layer. We used
`NousResearch/Meta-Llama-3.1-8B` (ungated mirror) — same architecture
as the gated official weights, provenance one step indirect. Layer 16
was picked as a default mid-layer; layer-sensitivity is on the open list
(#18). The cache for both sweeps is in `cache/meta-llama-3.1-8b_L16/steer/`
(α=±3 / ±6 single-seed runs) and `cache/meta-llama-3.1-8b_L16/steer22/`
(full multi-seed × unaligned × years × fp32 sweep).

### 9b.8 Anchor against the Goodfire paper at 8B (`subspace_capture`, `manifold_viz`)

A cheap sanity check before leaning further on §9b's numbers: does our
8B pipeline produce the *paper's* numbers on the *paper's* model? We
ran the two repo-shipped paper-replication scripts on Llama-3.1-8B/L16
with the fresh-trained SAE (`d_sae=32768, k=32`, 200k C4 background
tokens).

**`subspace_capture.py plot` (paper Fig 4 — shattering).** Years
manifold, k = number of selected features:

| k  | PCA | Geo-SAE | **Stat-SAE** | Random-orth | Random-over |
|--- |--- |--- |--- |--- |--- |
| 4  | 30% | 7%  | 5%  | 0% | 0% |
| 8  | 48% | 13% | 12% | 0% | 0% |
| 16 | 70% | 22% | 20% | 0% | 0% |
| 32 | 86% | 32% | 26% | 1% | 2% |
| 64 | **93%** | 40% | **28%** | 1% | 4% |

The paper's Fig-4 signature reproduces clean: stat-SAE plateaus at
~28% while PCA hits ~93% at k=64; geo > stat (decoder *directions*
span more variance than the *codes* express); random baselines stay
near zero. The PCA-to-stat-SAE ratio at N=3 is ~14× (vs 9× at 135M
in REPRODUCTION.md), so the shattering is **more dramatic at scale**
— consistent with the prior §9b.1 reading.

**`manifold_viz.py` (paper Fig 1 — years helix).** 3-D PCA of the
years manifold at 8B/L16 shows the chronological color gradient
(1800–1999) curving smoothly through PC1/PC2/PC3 (12%/7%/6% var) in a
helix-like loop. 3 PCs explain only 25% of total variance (vs much
higher at 135M), so the rendering is more diffuse, but the loop
topology of "blue (1800s) → purple (1850s) → orange (1900s) → yellow
(1990s)" is intact.

**Net:** the 8B pipeline is producing the paper's numbers on the
paper's model. The §9b.3–5 narrow bands and §9b.7 seed spread are real
findings (with the 10-seed CIs from §9b.3–5 and the 3-seed spread from
§9b.7 to characterise them), not pipeline artifacts. Cache:
`cache/meta-llama-3.1-8b_L16/{subspace_capture,manifold_viz}/`.

A scale-related implementation cost the anchor exposed: the original
`subspace_capture.find_support_greedy*` were pure-numpy CPU functions
that scaled to hours-per-manifold at 8B/32k-feature scale. The port
to torch+CUDA (commit `0a88925`) is a strict mechanical translation
behind a new `device=` kwarg; algorithm-faithful (same selected atoms,
max |VE_cpu − VE_cuda| < 2e-7 at machine epsilon).

**Limitations of this run.** Single layer (16), one 8B model. We used
`NousResearch/Meta-Llama-3.1-8B` (ungated mirror of the official
weights); same architecture, provenance one step indirect. Layer 16 was
picked as a default mid-layer; the layer-sensitivity sweep (#13 follow-on)
is the natural next test.

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
| **Real-model validation (Llama-3.1-8B, layer 16)** | ✅ **8B suite re-run 2026-05-30** with hard routing + 10 seeds. §9b.1 stands (already cleared seed noise); §9b.3–5 have per-row 2·SEM-significant findings (legible λ=10 / geography crosses PCA by ~30·SEM; iso=1/pars=0 / geography is the first **unsupervised** 2·SEM PCA-crossing in the project; adaptive iso=0/gate∈{2,8} / geography also crosses unsupervised). The earlier "Pareto noise band" caveat is retracted. |
| Anchor against the Goodfire paper at 8B (§9b.8) | ✅ Subspace_capture + manifold_viz reproduce paper Fig 4 / Fig 1 shapes (stat-SAE plateaus 28% vs PCA 93% on years at k=64; helix-like 3-D PCA loop). Pipeline confirmed. |
| Steering at 8B (§9b.7) | ◐ **Causal axis confirmed, magnitude seed-dependent.** 3 legible seeds give slopes +0.022 / +0.090 / +0.288 — all positive, ~13× spread. Seed 2 crosses zero (control); seeds 0/1 stop at modulation. Unaligned-axis baseline (same architecture, lam_label=0) is flat / wrong direction — **the cleanest "axis is causal" demonstration the project has produced**. Years contrast doesn't replicate. fp32 ≡ bf16. **Unaffected by the bug.** |

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
- **~~The 8B Pareto results (§9b.3–5) are within a noise floor.~~** —
  *Retracted 2026-05-30 by the 10-seed re-run.* The 3-seed Pareto bands
  WERE noisy, but at 10 seeds the per-row paired CIs cleanly identify
  multiple ↑ (SAE > PCA) and ↓ (SAE < PCA) rows at 2·SEM. See §9b.3–5
  and `cache/meta-llama-3.1-8b_L16/seeds10_summary.md` for the
  per-cell tables.
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
- **Causal/steering leg — axis-specific, seed-dependent magnitude.**
  At 135M (Part VII) the legible axis is a causal *control* handle:
  monotone hot/cold shift that crosses the decision boundary at α=+3.
  At 8B (§9b.7), the 3-seed re-run (`prototype/steer22.py`) gives
  slopes +0.022 / +0.090 / +0.288 — all positive, sign-consistent,
  seed 2 crosses zero (control), seeds 0/1 stop at modulation. The
  prior single-seed "scale gap" framing collapses: at 10 seeds the
  scales probably read "control sometimes, modulation usually" at
  both. The **unaligned-axis baseline** (same architecture, no label
  supervision) is flat/wrong-sign — the cleanest "axis is causal"
  control the project has produced. fp32 ≡ bf16. Years contrast
  doesn't replicate (charts split, tokens polysemous). **Unaffected by
  the bug.**
- **"Ideally on a real model" — DONE 2026-05-30 (re-run + anchor).**
  Part VIII §9b.1 stands (cleared seed noise in original 3-seed run);
  §9b.3–5 re-run at 10 seeds with per-row 2·SEM-significance flags
  (clean unsupervised geography crossing of PCA via iso=1/pars=0).
  Anchor §9b.8: `subspace_capture.py` + `manifold_viz.py` on
  Llama-3.1-8B reproduce paper Fig 4 / Fig 1 shapes. Pipeline
  confirmed; the 8B claims rest on the paper's numbers on the paper's
  model.

**Explicitly still on the list / things we have not yet done:**

1. ~~**Causal / steering leg**~~ ✅ / ◐ **Two single-shots at 135M
   plus a multi-seed/unaligned/fp32 robustness pass at 8B**
   (`steer.py`, `steer22.py`). At 135M the n=1 sweep crosses the
   decision boundary (control); at 8B (n=3) seed 2 crosses, seeds 0/1
   stop at modulation — the prior "control vs modulation" scale gap
   collapses to a seed-luck distribution. The cleanest causal-axis
   result of the project is the **unaligned-axis baseline at 8B**
   (slope −0.031, flat/wrong-sign vs the legible axis's seed-spread of
   +0.022 / +0.090 / +0.288). fp32 ≡ bf16; wider α (±6) bends back at
   α=+6 without changing the sign; years contrast doesn't replicate.
   Still on the list: multi-manifold (beyond temperature), multi-layer,
   and same-recipe multi-seed for 135M to confirm the scale-vs-seed
   reading.
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
8. ~~**Statistical rigor:** more seeds, confidence intervals…~~ ✅
   **Done for §9b.3–5 at 8B (10 seeds + per-row 2·SEM paired CIs;
   `prototype/summarize_seeds.py`).** Still on the list: more seeds for
   §9b.7 steering (currently n=3 per axis-kind/target; the 13× slope
   spread suggests n=10+ would tighten the magnitude estimate), and
   ideally larger manifolds (drop/augment `age` at 69 train points,
   `days` was already dropped).
9. ~~**Anchor against the Goodfire paper at 8B.**~~ ✅ **Done 2026-05-30
   (§9b.8).** `subspace_capture.py plot` and `manifold_viz.py` on
   Llama-3.1-8B/L16 reproduce paper Fig 4 (stat-SAE 28% plateau vs PCA
   93% at k=64 on years; PCA/SAE ratio ~14× at N=3) and Fig 1
   (helix-like 3-D PCA loop for years). Pipeline confirmed.

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
| Steering #22 robustness (multi-seed × unaligned × years × fp32) | `python prototype/steer22.py` | `cache/<tag>/steer22/` |
| 10-seed CI tables from results.json files | `python prototype/summarize_seeds.py --tag <tag>` | `cache/<tag>/seeds10_summary.md` |
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

*Last updated: 2026-05-30.* The 8B suite was re-run with 10 seeds
(§9b.3–5 noise-floor caveat retracted; per-row 2·SEM-significant flags
generated by `prototype/summarize_seeds.py`); the steering leg got a
multi-seed / unaligned-axis / years / fp32 robustness pass
(`prototype/steer22.py` — §9b.7 reframed around the seed distribution);
and the #20 paper-anchor was run (`subspace_capture` + `manifold_viz`
on Llama-3.1-8B, §9b.8 — paper Fig 4 / Fig 1 shapes reproduce).
`subspace_capture.find_support_greedy*` was ported to torch+CUDA
behind a new `device=` kwarg so the anchor could run on GPU at
8B/32k-feature scale without spending hours on CPU. Open threads
tracked in §11/§12.*
