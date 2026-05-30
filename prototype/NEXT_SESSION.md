# Next session handoff — goodfire SAE-manifold project

## State of the story (read this FIRST)

The 8B run (Part VIII, 2026-05-28) is the **acid test** for which 135M findings were
real and which were small-model artefacts. **A late-day honest-reframe pass that
same evening (commit `acc148a`) corrected an earlier overclaim** after the user
reviewed the 8B Pareto plots and the steering plot. Read the corrected picture
below before touching the writeup or running anything new.

### What 8B *cleanly* gives us (the headline result, §9b.1)

1. **Curvature-beats-flat is now bulletproof.** At 135M, factored-LIN held a small
   positive margin (+0.15 mean VE@3), leaving "is factored-NL just winning by
   parameter slack?" half-open. At 8B factored-LIN **collapses to −0.02** — same
   parameters, same routing, same coord_dim, and the linear twin can't even match
   PCA. The factored-NL win is curvature, not capacity.
2. **Shattering is more dramatic at scale.** Standard-SAE VE@3 ÷ PCA VE@3 is **~14×**
   at 8B vs ~9× at 135M. The paper's headline gets *more* important at scale.
3. **Per-manifold:** factored-NL beats PCA by **+0.42 VE on geography** and **+0.09
   on years** at coord_dim=3 (the two manifolds whose geometry is most overtly
   nonlinear). Temperature and colors tie at PCA's ceiling.

These three points sit in `cache/meta-llama-3.1-8b_L16/fair_comparison/ve_curves.png`
and the spread comfortably exceeds 3-seed noise. **This is the §9b.1 leg of the
real-model claim — cite this when citing Part VIII.**

### What 8B *reframes the motivation of*, mechanism unchanged

- **Part V's MOTIVATION was small-model-specific.** At 135M legibility supervision
  was sold as "PCA can't read concepts (colors R² 0.38) so weak labels are a
  *rescue*." At 8B PCA's R² is 0.86–0.99 across all five manifolds — the same
  mechanism becomes a *refinement*, not a rescue. Part V's intro forward-pointer
  to §9b.6 (commit `49ec5ff`) is correct and stays.

### Where the late-day reframe **softened** earlier claims

Two pieces of today's writeup were drafted too charitably; commit `acc148a`
corrects both. Future sessions: **do not undo this without re-examining the
plots.**

1. **§9b.7 is "modulation, not control" — NOT "qualitatively replicates."**
   The 8B steering slope (+0.090/α) IS monotone and ~6× a flat random control,
   but `logit(hot−cold)` stays NEGATIVE across all α ∈ [−3, +3] (−1.31 → −0.79).
   The model prefers "cold" under every steering condition. At 135M the same
   sweep went −1.33 → +0.04 — actually flipping the prediction. The 135M
   language "the legible coordinate is causal control" earns its strength from
   that *boundary crossing*; at 8B we only modulate the margin. The honest 8B
   claim is "directional monotone effect, ~6× random, falls short of behavioural
   switch."
2. **§9b.3 / §9b.4 / §9b.5 sit inside a Pareto noise floor.** At 8B PCA already
   saturates label R² (0.86–0.99) and factored VE@3 lives in ~0.05-wide bands on
   linear-ish manifolds. The Pareto plots
   (`cache/meta-llama-3.1-8b_L16/{legible_coord,iso_parsimony,adaptive_parsimony}/pareto.png`)
   show trajectories crossing and clustering rather than tracing clean frontiers
   — the visual signature of noise in narrow bands. 3 seeds is **not enough** to
   distinguish a real Pareto move from seed-luck. We kept the *mechanical*
   gate-sum claim for adaptive parsimony (lam_gate closes gates uniformly across
   manifolds — the gates themselves are directly penalised, that part is real).
   The *legibility-side* claim "scale doesn't fix it" needs more seeds.

### What 8B still leaves UN-TESTED

- **The anchor — task #20 below.** Before re-running anything of ours with more
  seeds, replicate ONE Goodfire-paper figure at 8B (`subspace_capture.py` for
  Fig 4 / `manifold_viz.py` for Fig 1) to confirm our 8B pipeline is producing
  the paper's numbers on the paper's model. Cheap, diagnostic, no new code.
- **More seeds at 8B for §9b.3–5.** Either 10 seeds for the existing grid, or a
  narrower grid with deeper seeds. Only worth doing *after* the anchor lands
  cleanly.
- **α-range exploration for steering at 8B.** Did the curve asymptote or just
  not extend far enough? The fp32 vs bf16 control would also pin reading 3 of
  §9b.7.
- **Layer sensitivity, cross-architecture, in-the-wild router, group-sparse
  charts, proper isometric-AE, statistical rigor, bigger manifolds** — all
  still on the list (§12 items #4–#8).

### Bottom line for the writeup arc

The 135M results are **not invalidated** — they're the toy-scale lens story and
internally consistent. The 8B story now reads:

> shattering → atlas idea → curved-vs-flat win at 135M, then *cleaner* at 8B in
> the fair-comparison VE curves → supervised legibility works at 135M, at 8B
> the Pareto plots live in a noise band and need more seeds → causal control
> at 135M, modulation only at 8B → **anchor not yet run; do it before
> re-investing in the noisy 8B legs**.

---

## How to run

- **Project dir:** `~/Documents/goodfire`. **Branch:**
  `prototype/factored-manifold-sae` (`main` = clean reproduction checkpoint;
  no remote). Last commits: `acc148a` (honest reframe), `599d1a6` (overclaim
  kept in history for the audit trail), `3c6a07a` (steer runbook), `4fbff22`
  (steer.py 8B patch), `49ec5ff` (Part V/VI framing).
- **Local CPU (Mac M1 8 GB)** — prefix `SAE_DEVICE=cpu SAE_D_MODEL=576`, add
  `SAE_MODEL_NAME=HuggingFaceTB/SmolLM2-135M SAE_LAYER=19` to load the model.
- **Cloud GPU (RunPod) for 8B work:** see `../RUNPOD.md` for the general
  recipe, and `prototype/STEER_8B_RUNBOOK.md` for the specific steering-at-8B
  procedure (same pattern adapts to the anchor scripts).
- **Read the auto-loaded project memory before doing anything** — it captures
  every GPU-cost-burning gotcha.

### Pod state at end of 2026-05-28 session

- Previous pod was at `64.247.196.124:14276` on **A100-80GB** in **us-mo-1**
  region, with `/workspace` on a **Network Volume**
  (`mfs#us-mo-1.runpod.net:9421`). Pod is **DOWN** (Stopped or Terminated —
  unclear). Network volumes persist beyond pod lifetime as long as you attach
  them to the new pod and stay in **us-mo-1**.
- The pre-session capacity check: A100-80GB was out of capacity in us-mo-1.
  Fall-back hierarchy from RUNPOD.md: **RTX 4090 24 GB** (~$0.4–0.7/hr,
  cheapest), then **A100 40 GB** (~$1.5–2/hr, comparable to old pod), then
  **A6000 48 GB** (~$0.6–0.9/hr). Any 24 GB+ card works for fp16 inference.
- On new-pod boot, first command to verify volume is mounted:
  ```bash
  ls /workspace/goodfire/cache/meta-llama-3.1-8b_L16/sae_32768_k32.pt 2>/dev/null \
    && echo "volume preserved" || echo "volume EMPTY — need re-extract"
  ```

## Read in this order

1. **This file (`prototype/NEXT_SESSION.md`)** — state of the story + the
   honest-reframe correction.
2. **`WRITEUP.md`** §9b.1 (the solid 8B headline), §9b.3 (the noise-floor
   caveat block — read it BEFORE §9b.3/4/5 themselves), §9b.7 (modulation-
   not-control). Then §10/§11/§12 for the patched status.
3. **`prototype/STORY.md`** — the narrative arc, now including a Part VIII
   that lead-foots the §9b.1 win and separately calls out what doesn't hold
   yet.
4. **Project memory** (`goodfire-sae-manifold-repro`) — includes the
   overclaim-and-correction record so the next session doesn't redo the
   overclaim.

## Top-priority next-session task

### #20 — Anchor against ONE Goodfire-paper figure at 8B

- **Why:** before re-investing seeds in the §9b.3–5 Pareto noise floor or
  trying to upgrade §9b.7 to "control," confirm our 8B pipeline produces the
  paper's numbers on the paper's model. If shattering and the 3-D PCA helix
  both come out matching, we know the pipeline isn't broken and the §9b.3–5
  noise is "expected at this seed count, fix by adding seeds." If one of them
  doesn't, we have a much bigger problem than "the writeup overclaimed."
- **What to run, both scripts:**
  ```bash
  # on the new pod, in /workspace/goodfire/
  SAE_MODEL_NAME=NousResearch/Meta-Llama-3.1-8B SAE_LAYER=16 \
  SAE_D_MODEL=4096 SAE_CACHE_TAG=meta-llama-3.1-8b_L16 \
  python subspace_capture.py plot \
      --sae cache/meta-llama-3.1-8b_L16/sae_32768_k32.pt --k 32

  SAE_MODEL_NAME=NousResearch/Meta-Llama-3.1-8B SAE_LAYER=16 \
  SAE_D_MODEL=4096 SAE_CACHE_TAG=meta-llama-3.1-8b_L16 \
  python manifold_viz.py --manifold years age temperature colors geography \
      --sae cache/meta-llama-3.1-8b_L16/sae_32768_k32.pt --k 32 --d-in 4096 \
      --overlay-features 6
  ```
- **Cost estimate:** ~10 min wall, ~$0.30 on RTX 4090.
- **Patches needed:** `manifold_viz.py` device-hardcode fix shipped in commit
  `acc148a`. `subspace_capture.py` is already DEVICE-aware. Push both to the
  pod with `scp` before running.
- **Success criterion:** `subspace_capture` produces a figure where standard-
  SAE-vs-PCA VE-curves match the Goodfire paper's Fig 4 shape (SAE plateaus
  *far* below PCA). `manifold_viz` produces a 3-D PCA scatter of years that
  looks like a helix (the cover figure of the original article).
- **If it doesn't:** that's a separate diagnostic problem — the §9b.1 result
  may also be suspect. Stop and inspect, don't paper-over.

## Lower-priority follow-ups (after #20)

- **#21 More seeds for §9b.3–5 at 8B.** Run `legible_coord`, `iso_parsimony`,
  `adaptive_parsimony` with `--seeds 0..9` and compute confidence intervals
  per row. Only worth doing once the anchor confirms the pipeline.
- **#22 Wider α-range and multi-seed for §9b.7 steering at 8B.** Did the
  contrast asymptote or could it cross zero at α=+5? Also: years contrast
  (` twenty` vs ` nineteen`), unaligned-axis baseline, fp32 control.
- **#14 In-the-wild router**, **#15 group-sparse top-k**, **#16 isometric-AE**,
  **#17 more seeds + bigger manifolds**, **#18 layer-sweep at 8B**, **#19
  cross-arch (Qwen / Mistral)** — all unchanged.

## Hard-won gotchas (paid for in GPU dollars — DON'T re-pay)

1. **Device hardcodes silently kill GPU runs.** Patched across
   `fair_comparison.py`, `train_standard_sae`, `sae_geometric_curve`,
   `sae_statistical_curve`, `_sae_greedy_basis`, `legible_coord.py`,
   `iso_parsimony.py`, `adaptive_parsimony.py`, `steer.py`, `manifold_viz.py`.
   Pattern: `device=DEVICE` for training, `model.cpu()` before return so
   the saved cache is portable.
2. **Don't `uv sync` on a CUDA pod** — clobbers the torch wheel. Use
   `pip install` for non-torch deps.
3. **`rsync` to /workspace network FS** needs `--no-perms --no-owner --no-group`.
4. **`pkill` patterns must include every stage script.** Canonical kill list:
   `run_all.py|fair_comparison.py|legible_coord.py|iso_parsimony.py|iso_coord.py|adaptive_parsimony.py|train_sae.py|background.py|data.py|steer.py|subspace_capture.py|manifold_viz.py`.
5. **NEVER test with `--force` against the flat `cache/` dir.** Always set
   `SAE_CACHE_TAG`.
6. **HF gating on `meta-llama/Llama-3.1-8B`** returns 401. Use
   `NousResearch/Meta-Llama-3.1-8B` (ungated mirror) or set `HF_TOKEN`.
7. **Legible/iso/adaptive stages are low-GPU-util at 8B by design** — kernel-
   launch overhead on small batches, not a bug.
8. **`!` shell mode paste-mangles long single-line commands.** Terminal
   soft-wrap → real newlines in the shell buffer → broken syntax. Workaround
   = drive `scp`/`ssh` from the Claude Code Bash tool itself (runs on the
   Mac with access to `~/.ssh`); avoids the paste path entirely.
9. **Network Volume vs Pod Volume.** The previous pod's `/workspace` was on
   a Network Volume (`mfs#us-mo-1.runpod.net`) which persists across pod
   lifetimes. To reuse it, attach it to the new pod and stay in **us-mo-1**.
   Pod Volume (default ephemeral disk) does NOT persist.
10. **`shutdown` from inside the container fails** ("System has not been
    booted with systemd as init system"). Pods must be stopped via the
    RunPod web console or `runpodctl stop pod <ID>`.

## Files (prototype/)

- `factored_sae.py` — atlas SAE (router + per-chart coord + curved decoder).
- `fair_comparison.py` — fair test + shared eval helpers.
- `legible_coord.py` — supervised label-alignment sweep (incl. cyclic).
- `iso_coord.py` — unsupervised isometry (negative).
- `iso_parsimony.py` — isometry + global PR parsimony (qualified positive at
  135M; at 8B see noise-floor caveat).
- `adaptive_parsimony.py` — per-chart learned per-dim gates (qualified
  negative at 135M; at 8B mechanical leg holds, legibility leg needs seeds).
- `viz_coord.py` — legibility figures (`cache/viz/`).
- `steer.py` — causal steering. **At 135M = control; at 8B = modulation only.**
- `run_all.py` — end-to-end driver for the whole suite.
- `STEER_8B_RUNBOOK.md` — exact recipe for steering at 8B (preserved as the
  pattern for any "single-script run on existing 8B pod" task).
- (repo root) `data.py`, `saes.py`, `train_sae.py`, **`subspace_capture.py`
  (anchor candidate #1)**, **`manifold_viz.py` (anchor candidate #2; device
  hardcode patched in `acc148a`)**, `RUNPOD.md`.

## Working agreement

- **Reason about *why* before building.** Carry the story-of-the-story.
- **Be honest about negative and reframing results.** Today's `acc148a` is the
  template: when the data doesn't support the claim, rewrite the claim — do
  not silently dilute the language.
- After each task: update `WRITEUP.md` (and `STORY.md` if the arc changes),
  update the project memory, and commit on the prototype branch.
- **Before re-investing in a noisy 8B leg, anchor first** (task #20). The
  diagnostic cost of an anchor is ~$0.30; the diagnostic cost of more seeds
  on a leg whose pipeline might be broken is much higher.
- **Cloud GPU sessions:** before relaunching anything, verify exactly one
  root process per stage script. Stop the pod IMMEDIATELY when work
  finishes — `shutdown` from inside the container does not work; use the
  RunPod console or `runpodctl stop pod`.

---

*Last updated: 2026-05-28 evening (post `acc148a` honest reframe; #20 anchor
pending pod restart on a non-A100-80GB GPU in us-mo-1 with the existing
network volume attached).*
