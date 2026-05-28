# Next session handoff — goodfire SAE-manifold project

## State of the story (read this FIRST)

The 8B run (Part VIII, 2026-05-28) is the **acid test** for which 135M findings were
real and which were small-model artefacts. Most held; two reframed; one strengthened
the central claim. **Internalize this before touching the writeup or running anything
new** — the next session's edits and experiments only make sense against this picture.

### What 8B *strengthened*

1. **The curvature-beats-flat claim is now bulletproof.** At 135M, factored-LIN held a
   small positive margin (+0.15 mean VE@3), leaving "is factored-NL just winning by
   parameter slack?" half-open. At 8B, factored-LIN **collapses to −0.02** — same
   parameters, same routing, same coord_dim, and the linear twin can't even match PCA.
   The factored-NL win is curvature, not capacity.
2. **The shattering phenomenon is more dramatic on a real model.** Standard SAE
   VE@3 ÷ PCA VE@3 is **~14×** at 8B vs ~9× at 135M. The paper's headline gets
   *more* important at scale, not less.

### What 8B *reframed* (the genuine surprises)

1. **Part V's MOTIVATION was small-model-specific, not its mechanism.** At 135M we
   sold legibility supervision as "PCA can't even read concepts (colors R² 0.38) so
   we use weak labels to rescue an unreadable coordinate." At 8B, **PCA already reads
   everything** — label R² 0.86–0.99 across all five manifolds (colors 0.98, geography
   0.86). The legibility *mechanism* still works (colors kNN 0.73 → 0.93 across the
   λ-sweep at flat VE), but it's now a **refinement** of an already-readable baseline,
   not a rescue of an unreadable one. **Part V's motivating sentences need a forward-
   pointer to Part VIII §9b.2/9b.6.** This is task **#13b** below.

2. **Adaptive parsimony's "scale fixes it" excuse is dead.** §8.3's qualified negative
   (gates close uniformly, no per-manifold dim differentiation) reproduces at 8B
   identically. We had hedged at 135M that this might be because "manifolds are barely
   multi-D to a nonlinear chart at small scale." At 8B with much cleaner geometry the
   gates still close uniformly. **This is a method limit (cost shape), not a scale
   artefact.** A fix needs different machinery (group sparsity over dims, per-manifold
   dim budget, or a different prior), not a different model.

### What 8B left UN-TESTED

- **Steering (Part VII)** has zero 8B evidence. It's the strongest causal claim in
  the writeup and our most exposed orphan. **This is task #13a — top priority for
  next session.**
- **Layer sensitivity.** Layer 16 was a default. We don't know if the win is
  layer-specific.
- **Cross-architecture.** Llama only. A second family (Qwen / Mistral) would settle
  cross-arch generalisation.

### Bottom line for the writeup arc

The 135M results are **not invalidated** — they're the toy-scale lens story and
internally consistent. Part VIII contextualises them. The story now reads:

> shattering → atlas idea → curved-vs-flat win (at 135M, then *cleaner* at 8B) →
> failed unsupervised legibility → working supervised legibility (rescue at 135M,
> refinement at 8B) → causal steering at 135M → real-model validation tightens
> the geometric story and reframes one motivation.

---

## How to run

- **Project dir:** `~/Documents/goodfire`. **Branch:** `prototype/factored-manifold-sae`
  (`main` = clean reproduction checkpoint; **no remote**). Last commit: `f8b7e48`
  (Part VIII).
- **Local CPU (Mac M1 8 GB)** — prefix `SAE_DEVICE=cpu SAE_D_MODEL=576`, add
  `SAE_MODEL_NAME=HuggingFaceTB/SmolLM2-135M SAE_LAYER=19` to load the model
  (e.g. `steer.py`, `data.py`, `background.py`). MPS topk pathologically slow.
- **Cloud GPU (RunPod) for 8B work:** see **`../RUNPOD.md`**. Suite runs via
  `prototype/run_all.py` (model-agnostic, stage-based, layer-looping, idempotent).
  Default 8B env: `NousResearch/Meta-Llama-3.1-8B`, layer 16, expansion 8, k=32.
- **Read the auto-loaded project memory before doing anything** — it captures every
  GPU-cost-burning gotcha we paid for the hard way.

## Read in this order

1. **This file (`prototype/NEXT_SESSION.md`)** — state of the story + top-priority task.
2. **`WRITEUP.md`** §9b (Part VIII) — the 8B numbers and the "what 8B taught us"
   subsection (§9b.6). Then §10/§11/§12 for the patched status.
3. **`prototype/STORY.md`** — the narrative arc (Parts 0–VII; needs a Part VIII
   addendum if you do an STORY refresh).
4. **Project memory** (`goodfire-sae-manifold-repro`) — current state, gotchas, and
   the `goodfire-compute-strategy` companion note.

## Top-priority next-session tasks (do both in one pod session)

### #13a — Steering at 8B (`steer.py` on Llama-3.1-8B layer 16)

- **Why:** the only Part-VII evidence is 135M. Closing this gap upgrades the causal
  claim from "proof-of-concept on a small model" to "validated on the paper's model."
- **What:** re-run `prototype/steer.py` with the 8B env vars. Use the aligned (λ=1)
  legible temperature axis on the cached 8B factored-SAE. Same readout prompts,
  same α-sweep [−3, +3], same random-control comparison.
- **Cost estimate:** ~30 min on A100, ~$1-2. Cheapest high-value follow-up we have.
- **What to watch for:** monotone slope, ~10×+ over random control. If the slope is
  weaker at 8B, write up *that* honestly — the causal claim's scale-dependence is
  itself a finding.
- **Gotchas:**
  - `steer.py` uses `nnsight`. Layer output for a single string prompt is unbatched
    `[seq, d]`, so the indexing is `output[0][..., -1, :]` not `output[0][:, -1, :]`.
    (Same caveat as 135M — re-test on 8B.)
  - Add the *delta* `α·v` to the activation, don't replace with a lossy
    reconstruction — see Part VII's method paragraph.
  - Use the same factored-SAE cached under `cache/meta-llama-3.1-8b_L16/` — don't
    retrain on 8B.

### #13b — Part V framing patch (free, no GPU)

- **Why:** Part V's motivating sentences ("PCA isn't very legible, especially on
  colors, so we use weak labels…") read fine at 135M but contradict §9b.2/9b.6.
  A sequential reader will hit Part V before Part VIII and be misled.
- **What:** add 1–2 sentences in Part V's intro forward-pointing to §9b.6:
  > *"At small scale (135M, this section), PCA's held-out label R² is only 0.38 on
  > colors and 0.67 on geography — the coordinate is legible-in-principle but
  > readable-in-practice only with help. At real-model scale (Llama-3.1-8B, §9b.6)
  > PCA's R² rises to 0.86–0.99 across all five manifolds and supervision becomes a
  > refinement of an already-readable baseline rather than a rescue. The mechanism
  > below (weak-label probe orienting the chart's coordinate at flat VE) is the
  > same; its motivating gap is smaller at scale."*
- Also patch the Part V verdict line so the "label-free legibility works only on low-D
  manifolds" line points forward to Part VIII's adaptive-parsimony scale-confirmation.

### #13c (optional, if pod time remains) — `viz_coord.py` at 8B

- Free almost — visualisation script. Run on 8B factored-SAE to produce
  `cache/meta-llama-3.1-8b_L16/viz/coord_vs_label.png` and `colors_circle.png`. Same
  as 135M outputs in `cache/viz/`. Pure illustration; no new claims.

## Lower-priority follow-ups (after #13a-c)

- **#14 In-the-wild router.** Train the factored model on *background* (C4) activations,
  not the curated manifold mixture. Test whether charts spontaneously discover the
  manifolds. Validates the "router = learned feature clustering" claim we *asserted*
  in Part II but only checked loosely (purity).
- **#15 Group-sparse top-k chart selection.** Replace softmax router with top-k → a
  more SAE-like genuinely-sparse object. Check that fidelity/legibility survive.
- **#16 Proper isometric-AE.** Exact Jacobian + encoder pseudo-inverse term (Gropp)
  vs the finite-difference surrogate — gives isometry its fairest shot. Lower priority
  given §8.1 was a clean negative even before this refinement.
- **#17 Statistical rigor + bigger manifolds.** More seeds + CIs; augment/replace
  age (69 train pts) and days (dropped). Do last, once the method set is frozen.
- **#18 Layer sensitivity sweep at 8B.** Run `run_all.py --layers 8 12 16 20 24` on
  the 8B model. ~4 hours, ~$8. Tests whether the curvature win concentrates at one
  layer or is broad. Useful but not blocking any claim.
- **#19 Cross-architecture validation.** Same suite on Qwen-7B-base or Mistral-7B-v0.3.
  Strongest possible cross-validation but biggest spend (~$10-20).

## Hard-won gotchas (paid for in GPU dollars — DON'T re-pay)

1. **Device hardcodes silently kill GPU runs.** Multiple training/eval functions had
   `device="cpu"` defaults that override CUDA detection (5+ patches across
   `fair_comparison.py`, `train_standard_sae`, `sae_geometric_curve`,
   `sae_statistical_curve`, `_sae_greedy_basis`, `legible_coord.py`, `iso_parsimony.py`,
   `adaptive_parsimony.py`). Pattern: `device=DEVICE` for training, `model.cpu()`
   before return so the saved cache is portable.
2. **Don't `uv sync` on a CUDA pod** — clobbers the torch wheel. Use `pip install` for
   non-torch deps. Comment block above `pyproject.toml` deps marks this.
3. **`rsync` to /workspace network FS** needs `--no-perms --no-owner --no-group`
   (chown is forbidden, otherwise exit 23).
4. **`pkill` patterns must include every stage script.** A `pkill -f "run_all|fair_comparison"`
   that doesn't list `legible_coord|iso_parsimony|adaptive_parsimony|train_sae|background`
   will orphan a stage's child python, which then double-spawns when you relaunch — wasted
   us hours on dual-process CPU runs. Canonical kill list:
   `run_all.py|fair_comparison.py|legible_coord.py|iso_parsimony.py|iso_coord.py|adaptive_parsimony.py|train_sae.py|background.py|data.py`.
5. **NEVER test with `--force` against the flat `cache/` dir.** Clobbers canonical
   135M results. Always set `SAE_CACHE_TAG` (and `run_all.py` does this per layer).
6. **HF gating on `meta-llama/Llama-3.1-8B`** returns 401 silently for `from_pretrained`.
   Use `NousResearch/Meta-Llama-3.1-8B` (ungated mirror, verified-same weights) or set
   `HF_TOKEN`.
7. **Legible/iso/adaptive stages are inherently low-GPU-util** at 8B because their
   models are tiny (12 charts × small MLPs, batch 512, ~5k pts → ~10 batches/epoch).
   Don't waste cycles "fixing" 20–30% GPU utilisation here; it's kernel-launch overhead,
   not your bug. Wall time per stage on A100: legible ~16 min, iso ~25 min, adaptive
   ~53 min.

## Files (prototype/)

- `factored_sae.py` — the atlas SAE (router + per-chart coord + curved decoder).
- `fair_comparison.py` — the fair test + **shared eval helpers** (`load_split`,
  `factored_eval`, `label_score` incl. cyclic, `_pca_basis`, `CYCLIC_LABEL`, `_ms`).
  Now GPU-correct via `DEVICE` (was a multi-hour pain to debug — see gotcha #1).
- `legible_coord.py` — supervised label-alignment sweep (concept-shaped, incl. cyclic).
- `iso_coord.py` — unsupervised isometry (negative; not in `run_all.py` by default).
- `iso_parsimony.py` — isometry + global participation-ratio parsimony (qualified positive).
- `adaptive_parsimony.py` — per-chart learned per-dim gates (intrinsic-dim-adaptive
  parsimony); qualified negative + the coordinate-normalization trade (`--no-coord-norm`).
- `viz_coord.py` — legibility figures (`cache/viz/`).
- `steer.py` — causal steering (`cache/steer/`); loads the model via nnsight. **Not yet
  run at 8B (task #13a).**
- `run_all.py` — end-to-end driver for the whole suite (see `../RUNPOD.md`).
- (repo root) `data.py`, `saes.py`, `train_sae.py`, `subspace_capture.py`, `RUNPOD.md`.

## Working agreement

- Reason about **why** before building. Carry the story-of-the-story; if a result
  reframes a previous claim, patch the writeup *with a forward-pointer*, don't
  silently revise.
- Be **honest about negative and reframing results** — Part VIII §9b.6 is the
  template. Two reframings, two strengthenings, one orphan. State all four.
- After each task: update `WRITEUP.md` (and `STORY.md` if the arc changes), update
  the project memory, and **commit** on the prototype branch with a descriptive
  message. No remote — commits are the local record of state.
- **Cloud GPU sessions:** before relaunching anything after a kill, verify exactly
  one root process per stage script. Use the canonical kill pattern (gotcha #4).
  Stop the pod IMMEDIATELY when the suite finishes — `cache/run_all.out` shows
  "All requested work complete." as the success sentinel.

---

*Last updated: 2026-05-28 (Part VIII complete; #13a steering at 8B is next).*
