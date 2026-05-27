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

## Part VII — Is the coordinate causal? (steering)

The sharpest test, and the first to touch the **model**. Built a steering vector from
the legible temperature axis, added `α·v` to a readout prompt's layer-19 activation,
read `logit(" hot")−logit(" cold")`. **A clean causal handle** — the contrast moves
monotonically through the cold→hot crossover (slope +0.231/α), while an equal-norm
**random control** is flat (slope +0.018, ~13× weaker). The legible coordinate isn't
just decodable — pushing on it *steers the model*. "Representation" → "control."

---

## Where the through-line stands

| Question | Answer | Strength |
|---|---|---|
| Curved SAE → better fidelity? | **Yes**, beats the PCA linear ceiling on curved manifolds | Strong (3 seeds, held-out, matched dim) |
| …interpretable coordinate? | **Yes with weak supervision** (incl. cyclic); ◐ label-free only on low-D | Strong supervised / partial unsupervised |
| …causally a control handle? | **Yes**, ~13× a random control | Proof-of-concept (1 manifold/seed) |

**One-line story:** *a curved, factored SAE captures concept manifolds that a standard
SAE shatters, and — once you orient its coordinate — that coordinate is both a legible
readout and a causal control, on a small model.*

**Standing limitations:** toy scale (135M, ~5k points, age/days too small); we changed
the lens not the model except in Part VII; "matched dimensionality" isn't fully
"matched capacity"; the legibility wins lean on weak supervision.

**Open threads (tasks #13–#17):** ~~adaptive parsimony~~ (◐/❌ attempted — cures
over-collapse but no differential dim allocation; unsupervised frontier stays open),
real-model validation (#13, now also the natural retest for adaptive parsimony),
in-the-wild router, group-sparse charts, proper isometric-AE, more seeds.
