# Next session handoff — goodfire SAE-manifold project

## How to run
- **Project dir:** `~/Documents/goodfire`. **Branch:** `prototype/factored-manifold-sae`
  (`main` = clean reproduction checkpoint; **no remote**).
- **Always CPU** (MPS topk pathologically slow). Prefix: `SAE_DEVICE=cpu SAE_D_MODEL=576`.
  Add `SAE_MODEL_NAME=HuggingFaceTB/SmolLM2-135M SAE_LAYER=19` for anything that loads
  the model (e.g. `steer.py`). Run from repo root as `uv run python ...`.
- Background jobs: `nohup` + logfile + a `cache/.done` flag; prints are block-buffered
  so read the logfile/flag, not a live tail. Don't commit `cache/` (gitignored).
- **Cloud GPU (RunPod) for #13+:** see **`../RUNPOD.md`**. The whole suite runs via
  **`prototype/run_all.py`** (model-agnostic, stage-based, layer-looping, idempotent;
  writes to a per-`(model,layer)` cache tag via `SAE_CACHE_TAG`). #13 targets the
  paper's **Llama-3.1-8B**. Driver CPU-tested on SmolLM2-135M.

## Read first
- **`prototype/STORY.md`** — the narrative arc (Parts 0–VII), reasoning-first.
- **`WRITEUP.md`** (repo root) — the detailed living log / blog resource (numbers,
  tables, limitations, the five-point bar status, reproducibility appendix).
- The **goodfire project memory** (auto-loaded) — current state + gotchas.

## State (2026-05-27)
Parts I–VII **done** (reproduction → atlas SAE → fair comparison → supervised+cyclic
legibility → unsupervised isometry/parsimony → causal steering). Tasks **#10
(coord-vs-label viz)**, **#11 (causal/steering leg)**, and **#12 (adaptive parsimony)**
done. The five-point fair-comparison bar is fully met; the broader "improvement" bar is
largely met (fidelity ✅, interpretability ✅ supervised / ◐ unsupervised, steering ✅;
**real-model ❌**).

**#12 adaptive parsimony — DONE, qualified negative** (`prototype/adaptive_parsimony.py`,
WRITEUP §8.3). Per-chart learned per-dim gates + fixed per-dim cost. Cures the global
knob's over-collapse (geography/colours survive: 0.30–0.45 / 0.14–0.23 vs PR's 0.08/0.07)
but gates close ~uniformly → no differential dim allocation, legibility flat in the
penalty (at 135M the manifolds are ~1-D to a nonlinear chart). Incidental: coordinate
normalization is a fidelity↔legibility *trade* (years R² 0.28→0.64 at VE 0.79→0.45).
Unsupervised frontier still open; revisit on a real (genuinely multi-D) geometry. **#13
is now NEXT** and doubles as the retest for adaptive parsimony.

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
- `iso_parsimony.py` — isometry + global participation-ratio parsimony (qualified positive).
- `adaptive_parsimony.py` — per-chart learned per-dim gates (intrinsic-dim-adaptive
  parsimony); qualified negative + the coordinate-normalization trade (`--no-coord-norm`).
- `viz_coord.py` — legibility figures (`cache/viz/`).
- `steer.py` — causal steering (`cache/steer/`); loads the model via nnsight.
- `run_all.py` — end-to-end driver for the whole suite (see `../RUNPOD.md`).
- (repo root) `data.py`, `saes.py`, `train_sae.py`, `subspace_capture.py`.

## Remaining tasks (work through in this order; reason about WHY before each)
- **#13 Real-model validation (NEXT).** Biggest external-validity threat (everything is
  135M). Re-run the core fair comparison on a larger model / more layers. Bonus: it's
  the natural retest for #12 — adaptive parsimony needs a genuinely multi-D geometry to
  show differential dim allocation, which a bigger model should provide.
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
