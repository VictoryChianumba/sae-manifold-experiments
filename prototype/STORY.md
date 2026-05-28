# The story so far — goodfire SAE-manifold prototype

*A narrative recap for our own reference (the digest companion to the detailed
`../WRITEUP.md`). Reasoning-first: why each step, what we found, what it cost.*

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

## Part IV — The fair comparison (and the pivot)

Five-point bar: (a) a real SAE baseline (C4 *and* mixture-retrained), (b) held-out,
(c) fixed coordinate dimensionality so **linear-vs-nonlinear charts is the only
variable**, (d) the paper's subspace-capture metric as the common yardstick,
(e) multiple seeds. The leveler: every method reconstructs held-out activations from an
**N-dimensional code**; a linear dictionary can at best reach PCA-N, **only a curved
decoder can exceed it.**

Result (held-out VE@3, 3 seeds):
- **Curvature wins at matched dim** — nonlinear beats linear charts on *every* manifold
  (Δ +0.23 to +0.75), many× seed noise. The failure really is the *straightness* of
  the atoms.
- **On curved manifolds it beats even the linear ceiling** — Factored-NL > PCA-3 on
  years (+0.45), geography (+0.25), colours (+0.07).
- **On 1-D manifolds it loses, by design** — age (−0.50), temperature (−0.11): PCA
  saturates, the MLP underfits tiny data. The held-out split correctly punishes
  curvature where there's none.

**The pivot:** the secondary metric (held-out label R²) showed the curved coordinate
decoded the label *worse* than PCA (years NL 0.16 vs PCA 0.75). We'd won **fidelity**
but not **interpretability** — the whole point. Next thread set.

## Part V — Making the coordinate legible

Weak per-manifold label-alignment term (train labels only; R² scored with a *fresh*
held-out probe). **Legibility is nearly free** — years R² 0.16→0.89 at `λ=1` with
**zero** reconstruction cost; the coordinate now **beats PCA on both axes at once**.
The earlier tension was an artifact of the unsupervised coordinate bending freely.

Cyclic concepts (colours/hue wraps): fixed the *scoring* first (regress onto cos/sin —
revealed the ≈0 was a scoring artifact, PCA recovers 0.30, but a linear probe can't
orient onto a circle); then fixed the *alignment* (target cos/sin), and colours' cyclic
R² climbed 0.13→0.54, crossing PCA at `λ≈1`. *Lesson: legibility needs a concept-shaped
target, not just a fair score.* (Caveat: weak supervision — "probe a known concept,"
not discovery.)

## Part VI — Legibility *without* labels?

- **Isometry alone — clean negative.** Constant-speed decoder → arc-length coordinate.
  Failed: never reached PCA, and any weight strong enough to reshape the coordinate
  destroyed reconstruction. *Why:* **arc-length ≠ linear-in-coordinate** — a 3-D
  isometric coordinate can still wind. Necessary, not sufficient.
- **Isometry + parsimony — qualified positive.** Add a scale-invariant
  participation-ratio penalty (use as few coordinate dims as the manifold needs).
  **Recovered label-free legibility on the genuinely low-D manifolds** (years
  0.16→0.70, temperature 0.79→0.89, near PCA) — the first time the unsupervised route
  worked — but a *single global* weight **over-collapsed** the multi-D ones (geography
  0.41→0.01, colours→0.07). *Lesson: the right parsimony = intrinsic dimension, which a
  global knob can't know.*
- **Adaptive parsimony (learned per-dim gates) — qualified negative.** Gave each chart
  a per-dim gate + a fixed per-dim cost so it keeps only the dims reconstruction pays
  for (intrinsic, not embedding, dim). It *does* cure the over-collapse — geography
  (0.30–0.45) and colours (0.14–0.23) survive, vs the global knob's 0.08/0.07 — but the
  gates close ~uniformly (~1.3 dims everywhere), so the intended *differential* dim
  allocation never appears and legibility is flat in the penalty. *Why:* at 135M scale
  a nonlinear chart reconstructs even geography from ~1 effective dim, so there's no
  differential pressure to exploit. The only new lever was incidental — per-dim
  **coordinate normalization** trades reconstruction for years legibility (R² 0.28→0.64
  at VE 0.79→0.45), a Pareto move, not a free win. *Lesson: "match parsimony to
  intrinsic dim" needs a setting where the manifolds are genuinely multi-D — pointing
  back at real-model validation.*

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

## Where the through-line stands

| Question | Answer | Strength |
|---|---|---|
| Curved SAE → better fidelity? | **Yes**, beats the PCA linear ceiling on curved manifolds; **cleaner at 8B** (factored-LIN collapses; geography +0.42, years +0.09 over PCA at coord_dim=3) | **Strong** (3 seeds, held-out, matched dim, two scales) |
| …interpretable coordinate? | **Yes with weak supervision** at 135M (incl. cyclic); ◐ label-free only on low-D at 135M; **at 8B inside a noise band** — 3 seeds, narrow R²/VE ranges (§9b.3 caveat) | Strong at 135M / **inconclusive at 8B until more seeds** |
| …causally a control handle? | At 135M ✅ control (sweep crosses decision boundary, ~13× random); at 8B ◐ modulation only (~6× random, monotone, but boundary not crossed in α∈[−3,+3]) | **Proof-of-concept; split by scale** |

**One-line story:** *a curved, factored SAE captures concept manifolds that a
standard SAE shatters — strongly at both 135M and 8B (§9b.1 is the cleanest
8B result). Once you orient its coordinate at 135M, that coordinate is both a
legible readout and a causal control. The same orientation step at 8B is
inside a 3-seed noise band on the readout side and modulates-but-does-not-flip
the model on the steering side. The shattering / curvature-wins findings
transfer cleanly; the legibility / control findings do not yet.*

**Standing limitations:** still toy by some axes — `age`/`days` too small, 3
seeds at 8B, one layer, one contrast pair / one manifold for steering at each
scale; the **8B Pareto plots are noisy enough that row-level claims should not
be cited**; "matched dimensionality" isn't fully "matched capacity"; the
legibility wins lean on weak supervision; the 8B pipeline has **not** been
anchored against a Goodfire-paper number yet.

**Open threads (tasks #14–#19, plus #20 anchor):** in-the-wild router,
group-sparse charts, proper isometric-AE, more seeds + bigger manifolds,
layer-sensitivity sweep at 8B, cross-architecture (Qwen / Mistral), and the
**anchor**: replicate one Goodfire-paper figure at 8B before re-investing in
the §9b.3–5 / §9b.7 legs.
