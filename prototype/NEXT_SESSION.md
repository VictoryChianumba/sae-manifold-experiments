# Next session handoff — goodfire SAE-manifold project

## Where we are
- **Project dir:** `~/Documents/goodfire` (moved off `~/Desktop` — macOS TCC was
  denying Desktop access mid-session; `~/Documents` is fine).
- **Git:** branch **`prototype/factored-manifold-sae`**. `main` = clean reproduction
  checkpoint (commit `c0cfdcc`). **No remote** (removed upstream
  `origin → goodfire-ai/sae-manifold` to avoid accidental push; upstream commits kept
  for provenance).
- **Env / how to run:** `uv` venv in the repo. Run from repo root as
  `uv run python ...`, **always CPU** (MPS topk is pathologically slow), with prefix:
  `SAE_DEVICE=cpu SAE_D_MODEL=576` (add `SAE_MODEL_NAME=HuggingFaceTB/SmolLM2-135M
  SAE_LAYER=19` for anything that loads the model).

## Project in one paragraph
Scaled-down local reproduction of Goodfire *Do SAEs Capture Concept Manifolds?*
(arXiv 2604.28119). Substituted **SmolLM2-135M** (LlamaForCausalLM, d=576, layer 19)
for Llama-3.1-8B (8B won't fit in 8 GB RAM). Reproduced the core finding — SAEs
shatter/dilute curved concept manifolds (years/helix: SAE codes plateau ~17% vs PCA
~94% at 16 dims). Built 3D manifold visualizations, a sparsity sweep, and clustering,
then prototyped a manifold-native SAE.

## Files
- `data.py` — env-configurable model/layer/d_model/device; concept-manifold extraction.
- `saes.py` — inference-only BatchTopK SAE + `load_sae`.
- `train_sae.py` — BatchTopK SAE **trainer** (the missing half of upstream).
- `subspace_capture.py` — `find_support_greedy` / `find_support_greedy_codes`; VE + tuning curves.
- `compare_sparsity.py` — k=16/32/64 sweep plot.
- `manifold_viz.py` — article-style 3D PCA manifold figures.
- `prototype/factored_sae.py` — **FactoredSAE**: softmax router over M charts +
  per-chart coordinate encoder + per-chart decoder (nonlinear; `--linear-charts`,
  `--coord-dim` flags). Trains on a mixture of cached manifolds. Eval = router purity
  + `coord→label R²`.
- `prototype/README.md` — prototype writeup + ablation ladder.
- `REPRODUCTION.md` — full writeup incl. the county/globe analogy.

## Cached artifacts (`cache/`, gitignored)
Manifolds `years age temperature days colors geography` (.pt);
`background_acts_40000.dat`; SAEs `sae_4608_k16/k32/k64.pt` (BatchTopK, d_sae=4608);
figures in `cache/subspace_capture/png/` and `cache/manifold_viz/`.

## Where the prototype landed — BE SKEPTICAL
Factored SAE ablation ladder (`coord→label R²`):
- **1 ray** (SAE-like): age 0.08, temp 0.01 — geometry NOT recovered (recon VE 0.978)
- **3-D subspace** (linear): age 0.72, temp 0.78
- **3-D curved** (nonlinear): age 0.95, temp 0.86 (recon VE 0.996)

This is **suggestive, NOT a demonstrated improvement.** Confounds:
1. coord_dim 1 vs 3 = **capacity confound** (3 predictors trivially beat 1).
2. R² is **in-sample** (fit & scored on same points) → inflates higher-capacity models.
3. The "1-ray" config is **not a real standard SAE** (still has router + curated data).
4. Single seed, ~5.4k points, tiny 135M model; `years` splits across charts.
Only fairer signal so far: 3-D linear vs 3-D nonlinear (same coord dim) → curvature
helps (age 0.72→0.95). Also note: we changed the **SAE (the lens)**, not the model;
and the main lever (subspace-per-cluster) is just the paper's own remedy
operationalized — low novelty.

## ITEM ONE — the fair-comparison experiment (the next task)
**Question:** does the factored/curved SAE actually recover concept-manifold geometry
better than a matched standard SAE — or is the hint an artifact?

**Requirements:**
- (a) A **real standard SAE baseline** in the same harness/data (not the pseudo
  "1-ray" config).
- (b) **Held-out** evaluation (train/test split per manifold) — kills in-sample inflation.
- (c) **Fixed coordinate dimensionality** across conditions so curvature
  (linear vs nonlinear charts) is the only variable.
- (d) Use the paper's **subspace-capture** metric as the architecture-agnostic
  yardstick (held-out manifold variance explained by N selected directions/dims),
  so SAE and factored-SAE compare at matched N.
- (e) A couple of **seeds**.

**Definition of "improvement" (the bar):** a Pareto move on
**(geometry fidelity) × (interpretability / parsimony)** at **matched sparsity &
capacity**, held-out, ideally on a real model — bonus: a **causal/steering** leg (move
along the coordinate → smooth predicted output change). **Reconstruction alone is NOT
the scoreboard** (that's the paper's whole point).

**First design decision:** pick the exact common metric for SAE vs factored-SAE.
Candidates: held-out subspace-capture variance-explained at matched #dims; or held-out
label-decodability at matched latent dims. Decide, get sign-off, then implement + run.

**Be willing to report a NEGATIVE result — it's equally informative.**

## Gotchas
- Run from repo root; prefix env vars; CPU only.
- Don't commit `cache/`, `.venv/`, `.claude/` (gitignored); `uv.lock` IS tracked.
- Keep work on the prototype branch; `main` stays the clean checkpoint.
- Background jobs: `nohup` + logfile; tail pipes buffer (read the logfile directly);
  MPS warms up slow then ~400 tok/s.
