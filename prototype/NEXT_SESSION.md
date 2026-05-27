# Next session handoff — goodfire SAE-manifold project

## How to run
- **Project dir:** `~/Documents/goodfire`. **Branch:** `prototype/factored-manifold-sae`
  (`main` = clean reproduction checkpoint; **no remote**).
- **Always CPU** (MPS topk pathologically slow). Prefix: `SAE_DEVICE=cpu SAE_D_MODEL=576`.
  Add `SAE_MODEL_NAME=HuggingFaceTB/SmolLM2-135M SAE_LAYER=19` for anything that loads
  the model (e.g. `steer.py`). Run from repo root as `uv run python ...`.
- Background jobs: `nohup` + logfile + a `cache/.done` flag; prints are block-buffered
  so read the logfile/flag, not a live tail. Don't commit `cache/` (gitignored).

## Read first
- **`prototype/STORY.md`** — the narrative arc (Parts 0–VII), reasoning-first.
- **`WRITEUP.md`** (repo root) — the detailed living log / blog resource (numbers,
  tables, limitations, the five-point bar status, reproducibility appendix).
- The **goodfire project memory** (auto-loaded) — current state + gotchas.

## State (2026-05-27)
Parts I–VII **done** (reproduction → atlas SAE → fair comparison → supervised+cyclic
legibility → unsupervised isometry/parsimony → causal steering). Tasks **#10
(coord-vs-label viz)** and **#11 (causal/steering leg)** done. The five-point fair-
comparison bar is fully met; the broader "improvement" bar is largely met (fidelity ✅,
interpretability ✅ supervised / ◐ unsupervised, steering ✅; **real-model ❌**).

Headline: a curved factored SAE beats the PCA linear ceiling on curved manifolds; a
weak concept-shaped label prior makes its coordinate legible at ~zero fidelity cost
(incl. cyclic colours); label-free legibility works only on low-D manifolds (iso+
parsimony); and the legible axis causally steers the model (~13× a random control).

## Files (prototype/)
- `factored_sae.py` — the atlas SAE (router + per-chart coord + curved decoder).
- `fair_comparison.py` — the fair test + **shared eval helpers** (`load_split`,
  `factored_eval`, `label_score` incl. cyclic, `_pca_basis`, `CYCLIC_LABEL`, `_ms`).
- `legible_coord.py` — supervised label-alignment sweep (concept-shaped, incl. cyclic).
- `iso_coord.py` — unsupervised isometry (negative).
- `iso_parsimony.py` — isometry + participation-ratio parsimony (qualified positive).
- `viz_coord.py` — legibility figures (`cache/viz/`).
- `steer.py` — causal steering (`cache/steer/`); loads the model via nnsight.
- (repo root) `data.py`, `saes.py`, `train_sae.py`, `subspace_capture.py`.

## Remaining tasks (work through in this order; reason about WHY before each)
- **#12 Intrinsic-dimension-adaptive parsimony (NEXT).** iso+parsimony recovered
  label-free legibility on low-D manifolds but a single global weight over-collapsed
  multi-D ones (geography, colours). Let each chart learn how many coord dims it needs
  (per-chart learned/annealed target, or a Matryoshka/nested coordinate with per-dim
  gates) → keep the years/temperature win without killing geography/colours.
- **#13 Real-model validation.** Biggest external-validity threat (everything is 135M).
  Re-run the core fair comparison on a larger model / more layers.
- **#14 In-the-wild router.** Train the factored model on *background* activations (not
  the curated mixture) and test whether charts discover the manifolds unsupervised —
  validates the "router = learned feature clustering" claim that's asserted, not shown.
- **#15 Group-sparse top-k chart selection.** Replace soft softmax with top-k → a more
  honest, genuinely-sparse SAE-like object; check fidelity/legibility survive.
- **#16 Proper isometric-AE.** Exact Jacobian + encoder pseudo-inverse term (Gropp) vs
  the finite-difference surrogate — give isometry its fairest shot. Lower priority.
- **#17 Statistical rigor.** More seeds + CIs; augment/replace the smallest manifolds
  (age=69 train pts; days dropped). Do last, once the method set is frozen.

## Working agreement (how we run this project)
- Reason about **why** we add each item before building it; keep the full picture.
- Run on CPU; be **honest about negative results** (they're equally informative).
- After each task: update `WRITEUP.md` (and `STORY.md` if the arc changes), update the
  project memory, and **commit** on the prototype branch.
