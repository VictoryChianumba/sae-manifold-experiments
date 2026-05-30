# The story so far — goodfire SAE-manifold prototype

*A narrative recap for our own reference (the digest companion to the detailed
`../WRITEUP.md`). Reasoning-first: why each step, what we found, what it cost.*

> **⚠️ Retraction (2026-05-29).** Question 1 ("fidelity") below was answered "yes"
> on the strength of a `factored_eval` bug that scored the factored model's recon
> over a 36-D mixture-subspace union while comparing it against PCA-3. With the
> fix (dominant-chart recon + hard-routed training, both committed 2026-05-29),
> the answer to Q1 becomes a **qualified yes on years only** (+0.12 VE over PCA),
> not the broad "+0.45 / +0.25 / +0.07 on years / geography / colors" originally
> claimed. Q2 (legibility) and Q3 (steering) are partially affected — Q3 is
> unaffected at both 135M and 8B; Q2's supervised lift will shrink because the
> hard-routed baseline is already higher (years 0.16 → 0.57 unsupervised). See
> `../WRITEUP.md` retraction banner + `[[research-comparison-smell]]` memory.
> The narrative below is preserved for chronology; numbers should be re-derived
> from `cache/fair_comparison/results.json` (current = hard-routed).

## The spine of the argument

One question, asked in three escalating forms — each "yes" only allowed after we
tried hard to make it a "no":

1. Does a **curved** SAE capture concept-manifold geometry better than a standard
   (straight) one? **(fidelity)**
2. If it reconstructs better, is its representation actually **interpretable** — does
   its coordinate track the concept? **(legibility)**
3. If it's legible, is it **causal** — does pushing on it steer the model? **(control)**

---

## Part 0 — Setup and the honest constraint

The paper (*Do SAEs Capture Concept Manifolds?*) argues SAEs, built from **straight**
dictionary atoms, can't represent **curved** concept manifolds (years as a helix,
colours as a wheel, geography on a sphere) — they "shatter/dilute" the manifold across
many features instead of capturing it compactly.

We can't run Llama-3.1-8B on an 8 GB M1, so we substituted **SmolLM2-135M** (same
Llama architecture, d=576, layer 19). Standing caveat from line one: a 135M model has
weaker, noisier geometry, and we only ever change the **SAE (the lens)**, not the
model — so we lean on *relative*, within-protocol comparisons, not absolute numbers.

## Part I — Reproduce the failure (so we're not fighting a strawman)

Trained a BatchTopK SAE on C4 background, ran the paper's subspace-capture metric. On
years/helix: **PCA hits 94% of the manifold's variance in 16 dims, but the SAE's
actual codes plateau at ~17% even with 64 features.** Dilution/shattering reproduced;
a sparsity sweep showed it's intrinsic, not a k artifact.

## Part II — The idea: an "atlas" SAE

A standard SAE describes every place on Earth with straight arrows from the planet's
centre; an **atlas** says *which chart* (region), then *local coordinates* within it.
We baked that in (`FactoredSAE`): soft **router** over M charts (which manifold) +
per-chart **coordinate** (where on it) + per-chart **nonlinear decoder** (a curved
chart). The router is the learned analogue of the paper's post-hoc feature clustering.
Honest framing: a research toy, and the core lever (subspace-per-cluster) is the
paper's own remedy — so the bar is "does **curvature specifically** buy something,
fairly measured."

## Part III — The first signal, and why we refused to believe it

An early ablation hinted curved charts recovered geometry (coord→label R²: ray ~0,
3-D linear ~0.7, 3-D curved ~0.9). We listed the confounds and stopped: **capacity**
(1 vs 3 coords), **in-sample R²**, **no real SAE baseline**, **one seed**. A hint is
not a result — so we designed a fair test.

## Part IV — The fair comparison (and the pivot) — **corrected**

Five-point bar: (a) a real SAE baseline (C4 *and* mixture-retrained), (b) held-out,
(c) fixed coordinate dimensionality so **linear-vs-nonlinear charts is the only
variable**, (d) the paper's subspace-capture metric as the common yardstick,
(e) multiple seeds. The leveler: every method reconstructs held-out activations from an
**N-dimensional code**; a linear dictionary can at best reach PCA-N, **only a curved
decoder can exceed it.**

Result (held-out VE@3, 3 seeds, **hard-routed atlas, post-fix**):
- **Curvature wins at matched dim against linear charts** — nonlinear beats linear
  charts on every manifold by Δ +0.30 to +2.00 VE, many× seed noise. Single-chart
  linear decoders are very nearly incapable under hard routing (≈0 VE). The
  straightness of atoms IS the bottleneck — the cleanest isolation of the paper's
  thesis. *(This finding survived the bug fix.)*
- **vs the PCA linear ceiling: years only.** Factored-NL clears PCA-3 on years
  (+0.12 VE) and lands below PCA-3 on the other four manifolds. The bug-era
  "+0.45/+0.25/+0.07 on years/geography/colors" reduces to one narrow win.
- **On near-linear / small-data manifolds it loses.** age (−0.67 vs PCA): PCA
  saturates, the MLP underfits tiny data (69 train points). The held-out split
  correctly penalises curvature where there's none.

**The bug.** For most of the project, factored_eval was scoring reconstruction over
the K-chart mixture (~36-D effective subspace at K=12, coord_dim=3) while taking
coords from a single dominant chart (3-D). The "matched coord_dim" framing was
false. The fix: dominant-chart-only recon + hard-routed training (so the model is
actually a true atlas, not a soft mixture). The numbers above are post-fix; the
pre-fix versions are preserved in `cache/fair_comparison/results.prepatch.json`.

**The pivot still holds, narrower.** The secondary metric (held-out label R²)
shows the unsupervised curved coordinate decodes years 0.57 vs PCA 0.75, geography
0.54 vs 0.60, colors 0.24 vs 0.30 — narrower gaps than the bug-era version
reported (was: years 0.16 vs 0.75), but still a real legibility deficit on most
manifolds. Closing it is the Part V thread.

## Part V — Making the coordinate legible — **corrected**

Weak per-manifold label-alignment term (train labels only; R² scored with a *fresh*
held-out probe). **Legibility is nearly free, and crosses PCA on 4 of 5 manifolds at
λ=1:** years 0.57→0.91 (PCA 0.75), geography 0.54→0.82 (PCA 0.60), temperature
0.78→0.93 (PCA 0.92), colors 0.24→0.37 (PCA 0.30) — mean held-out VE moves 0.44→0.46
(essentially free). Only age fails to cross (small data, near-linear).

The bug-era version reported a more dramatic 0.16→0.89 jump on years, framed as
"weak supervision is the magic." The corrected decomposition: hard routing alone
provides ~half the gain (0.16→0.57 — a previously-hidden architectural win),
supervision adds the rest (0.57→0.91). Two real separable effects instead of one
conflated one. Both are still genuine; the headline shrinks by half on years but
gains breadth (it crosses PCA on more manifolds in the corrected run).

Cyclic concepts (colours/hue wraps): fixed the *scoring* first (regress onto cos/sin —
revealed the ≈0 was a scoring artifact); then fixed the *alignment* (target cos/sin).
Post-fix, colours' cyclic R² climbs 0.24→0.37 (across the lam sweep), crossing PCA's
0.30 at `λ ≈ 1`. *Lesson: legibility needs a concept-shaped target, not just a fair
score.* (Caveat: weak supervision — "probe a known concept," not discovery.)

## Part VI — Legibility *without* labels? — **corrected**

- **Isometry alone — clean negative.** Constant-speed decoder → arc-length coordinate.
  Failed: never reached PCA, and any weight strong enough to reshape the coordinate
  destroyed reconstruction (mean VE −0.07). *Why:* **arc-length ≠ linear-in-coordinate**
  — a 3-D isometric coordinate can still wind. Necessary, not sufficient. Replicated
  in iso_parsimony's iso=1, pars=0 row, so no separate file needed.
- **Isometry + parsimony — qualified positive, narrowed.** Add a scale-invariant
  participation-ratio penalty (use as few coordinate dims as the manifold needs).
  Post-fix, **parsimony pushes years past PCA label-free (0.57→0.83 at pars=8)** — one
  clean unsupervised win. The dramatic bug-era "geography over-collapses to 0.01"
  failure is **gone** (geography barely moves: 0.54→0.57 under parsimony). New failure
  visible under honest baselines: parsimony degrades age (0.73→0.48). What survives:
  one unsupervised crossing of PCA, on years. *Lesson: parsimony is the right tool for
  the simplest curved manifold; on the others it's neutral or harmful.*
- **Adaptive parsimony (learned per-dim gates) — qualified negative, with a twist.**
  Each chart gets per-dim gates so it keeps only the dims reconstruction pays for.
  Post-fix, gates produce a real R² lift on years (0.21→0.50) — bigger than the
  bug-era's ~0 movement — but still don't cross PCA on any manifold. Gate
  differentiation does appear at gate=2 (years 1.94 active, geography 1.21, colors
  1.41), but **in the *wrong direction* for the hypothesis**: geography should need
  *more* dims if it's intrinsically 2-3D. The observed pattern reflects per-chart
  needs (geography routes across multiple charts, each seeing a smaller piece),
  not intrinsic manifold dim. *Lesson: the architecture has structure but not the
  structure we hypothesised; iso_parsimony beats adaptive on years (0.83 vs 0.50).*

## Part VII — Is the coordinate causal? (steering — 135M)

The sharpest test, and the first to touch the **model**. Built a steering vector from
the legible temperature axis, added `α·v` to a readout prompt's layer-19 activation,
read `logit(" hot")−logit(" cold")`. **A clean causal handle at 135M** — the
contrast moves monotonically *through* the cold→hot crossover (slope +0.231/α,
range −1.33 → +0.04 across α∈[−3,+3], so the model's preferred token actually
flips), while an equal-norm **random control** is flat (slope +0.018, ~13×
weaker). At 135M the legible coordinate is a control handle. (§9b.7 below shows
this does *not* fully carry to 8B — read this section as the 135M proof-of-
concept, not a general result.)

## Part VIII — Real model (Llama-3.1-8B)

The acid test for which of the small-model findings were real and which were
artefacts. Run via `run_all.py` on RunPod A100, layer 16, 3 seeds. The headline
fair-comparison VE-vs-N result (§9b.1) holds cleanly; the legibility /
parsimony / steering legs need more care.

**Held up cleanly (§9b.1).** Curvature-beats-flat is now bulletproof —
`factored-LIN` collapses to −0.02 mean VE@3 at 8B (vs +0.15 at 135M), so the
factored-NL win is curvature, not parameter slack. Shattering ratio over PCA
jumps ~9× → ~14×. **This is the one 8B result we lean on.**

**Reframed framing-only (§9b.2 / §9b.6).** Part V was sold as "PCA can't read
concepts so we need labels." At 8B, PCA's label R² is 0.86–0.99 across all
five concepts — supervision becomes a *refinement* of an already-readable
baseline, not a rescue. Mechanism unchanged; motivating gap smaller.

**Reframed honestly — earlier draft overclaimed (§9b.3–5 + §9b.7).**
The Pareto plots from `legible_coord`, `iso_parsimony`, and `adaptive_parsimony`
at 8B sit inside narrow noise bands (label R² already saturated, factored
VE@3 ranges 0.05–0.10 wide); 3 seeds is not enough to confidently distinguish
"the lever moved" from "seed-luck." Row-by-row claims softened with a
noise-floor caveat. The mechanical "gates close roughly uniformly" reading of
adaptive parsimony survives; "scale doesn't fix legibility" needs more seeds.

**Steering re-run at 8B is modulation, not control (§9b.7).** Same recipe,
one seed/manifold. Slope **+0.090/α**, strictly monotone, **~6× a flat random
control** — but `logit(hot−cold)` stayed −1.31 → −0.79 across α∈[−3,+3], **never
crossed zero**. At 135M the same recipe went −1.33 → +0.04 — actually flipping
the model's preferred token. So the qualitative content of "the legible
coordinate steers the model" is split by scale: control at 135M, modulation
only at 8B. An earlier draft of this section read "qualitatively replicates";
that was too charitable. Honest version: monotone causal effect above random,
but no demonstrated behavioural switch at 8B.

**Anchor pending.** Before leaning further on any of the 8B numbers we should
replicate one Goodfire-paper figure at 8B (`subspace_capture.py` for Fig 4 /
`manifold_viz.py` for Fig 1) as a sanity anchor — to confirm the pipeline
produces the paper's numbers on the paper's model. Cheap; not yet run.

---

## Where the through-line stands — **updated 2026-05-30**

| Question | Answer | Strength |
|---|---|---|
| Curved SAE → better fidelity? | **Narrowly yes at 135M (years +0.12), louder at 8B:** at 8B factored-NL clears PCA by +0.42 on geography and +0.09 on years at coord_dim=3, while factored-LIN collapses to −0.02 (no parameter-slack defence). Curvature-vs-linear delta intact at both scales. | Solid at 8B on curved manifolds (geography, years); narrow at 135M. |
| …interpretable coordinate? | At 135M ✅ supervised (λ=1 crosses PCA on 4/5 manifolds at near-zero VE cost). At 8B (10-seed re-run): **clean ↑ on geography at every λ ≥ 1** (peaks at 0.776 ± 0.010 vs PCA 0.472 ± 0.010 — ~30·SEM gap), marginal ↑ on years at λ=10. Non-geography cells mostly inside the PCA band at 2·SEM — supervision is a *refinement* on linear-ish manifolds, a *rescue* on geography. **First clean unsupervised crossing of PCA** at 8B via iso=1/pars=0 on geography (0.542 ± 0.024). | Solid at both scales; supervised + first unsupervised. |
| …causally a control handle? | At 135M ✅ control on n=1. At 8B with multi-seed: legibility supervision creates a sign-consistent causal axis vs an unaligned-axis baseline (same architecture, lam=0, slope −0.031 = flat/wrong). Legible-axis slopes +0.022 / +0.090 / +0.288 across 3 seeds — all positive, but seed 2 crosses zero (control); seeds 0/1 stop at modulation. The prior single-seed "scale gap" framing collapses to seed-luck. fp32 ≡ bf16. Years contrast doesn't transfer (chart-splits, polysemous tokens). | Causal axis confirmed; magnitude is the open question. |
| Pipeline anchored against the paper? | ✅ **8B subspace_capture + manifold_viz reproduce paper Fig 4 / Fig 1 shapes** (stat-SAE plateaus 28% vs PCA 93% at k=64 on years, ~14× shattering ratio at N=3; chronological loop in 3-D PCA). | Anchor passed; the 8B numbers above rest on the paper's pipeline. |

**One-line story (post-2026-05-30 re-run):** *a hard-routed
atlas-of-curved-charts SAE achieves a clean matched-dim fidelity win
over PCA on the manifolds whose geometry is most overtly nonlinear
(geography +0.42 at 8B, years +0.12 at 135M); a weak label prior
pushes its coordinate past PCA on those same manifolds at near-zero VE
cost (geography 0.47 → 0.78 at 8B λ=10, ~30·SEM clean); the
unsupervised lever — isometry alone, no labels — also clears PCA on
geography at 8B; and the resulting axis is causally axis-specific
(legible-vs-unaligned slope split, fp32-invariant) with seed-dependent
magnitude. The fidelity, legibility, and causality stories all hold;
the 8B anchor confirms the pipeline is producing the paper's numbers
on the paper's model.*

**The bug, in one paragraph.** For most of this project's history, `factored_eval`
scored reconstruction over the K-chart soft mixture (~36-D effective subspace at
K=12, coord_dim=3) while taking coords for label decoding from a single dominant
chart (3-D). The "matched coord_dim" framing was false. The fix: dominant-chart-only
reconstruction + hard-routed (straight-through one-hot) training so the model is
actually a true atlas, not a soft mixture. The fidelity wins shrank dramatically;
the supervised legibility win held (and split into a separable hard-routing-baseline
+ supervision-gain decomposition); steering was unaffected. Lesson saved as
`research-comparison-smell` memory: when a matched-dim comparison shows a big
margin, audit what the matching constraint constrains *at metric time*; branch
dimension is the trap.

**Standing limitations:** still toy by some axes — `age` (69 train
points) is consistently unreliable, `days` already dropped; one layer
(16) at 8B; one contrast pair / one manifold for steering at each
scale (3 seeds for 8B steering, n=1 at 135M); "matched dimensionality"
isn't fully "matched capacity"; the legibility wins lean on weak
supervision; years-target steering doesn't replicate temperature
(distinguishes the temperature result from "axis is causal in
general"); the 13× seed-spread for the 8B steering slope means
single-seed slope numbers aren't trustworthy.

**Open threads:** more steering seeds (to tighten the magnitude
distribution and the 135M control vs 8B modulation reading); steering
on additional manifolds (geography is the next obvious test given its
clean §9b.4 win); in-the-wild router (train on background, not curated
mixture); group-sparse charts (top-k routing for a more SAE-like
object); proper isometric-AE (exact Jacobian); larger manifolds
(replace `age`); layer-sensitivity sweep at 8B; cross-architecture
(Qwen / Mistral).
