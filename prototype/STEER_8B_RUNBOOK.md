# Runbook — task #13a: steering at 8B (`prototype/steer.py` on Llama-3.1-8B layer 16)

**Goal:** rerun the Part VII causal/steering leg on the paper's model so the
causal claim isn't a 135M-only proof-of-concept. **Cost target:** ≤ ~30 min,
≤ ~$2 on a single A100.

**Read first:** `RUNPOD.md` (overall pod recipe), `NEXT_SESSION.md` (gotchas
#1–#7 — all still apply), and the §9b.6 framing in `WRITEUP.md` so the
result lands in the right narrative slot.

---

## What we run, in one sentence

`prototype/steer.py` with `SAE_MODEL_NAME=NousResearch/Meta-Llama-3.1-8B`,
`SAE_LAYER=16`, `SAE_CACHE_TAG=meta-llama-3.1-8b_L16`, on one A100; output
lands in `cache/meta-llama-3.1-8b_L16/steer/{steering.png,results.json}`.

The script (a) retrains the aligned (λ=1) legible factored SAE for one seed
on the 5-manifold mixture (~1–2 min), (b) extracts the temperature
manifold's legible axis ŵ and computes the steering vector v in raw
activation units, (c) sweeps α ∈ [−3, +3] adding α·v into layer-16's
last-token activation for 3 readout prompts, reads logit(" hot") − logit
(" cold"), and (d) compares against an equal-norm random control.

---

## Pre-flight, in order

### P1. Pod selection

- **Best case:** the existing pod from the prior 8B run is still up and
  `/workspace/goodfire/cache/meta-llama-3.1-8b_L16/{years,age,temperature,colors,geography}.pt`
  exists. Total run is ~5 min on A100. Skip P3.
- **Fresh pod:** A100-40GB or RTX 4090 (24GB is enough for fp16/bf16
  inference). Use a **PyTorch CUDA template**. Add total ~10–15 min for
  activation re-extraction + model download.

### P2. SSH and code sync

`rsync` (preferred so we can iterate from the Mac without re-pushing tarballs):

```bash
# from Mac
rsync -rlvz --no-perms --no-owner --no-group \
  -e "ssh -p <PORT> -o StrictHostKeyChecking=accept-new -i ~/.ssh/id_ed25519" \
  --exclude cache --exclude .venv --exclude .git --exclude __pycache__ --exclude .DS_Store \
  ~/Documents/goodfire/ root@<IP>:/workspace/goodfire/
```

`--no-perms --no-owner --no-group` is mandatory (gotcha #3): `/workspace`
forbids `chown`.

### P3. Pod environment (skip if reusing the pod)

```bash
# on the pod
cd /workspace/goodfire
pip install nnsight transformers datasets scikit-learn matplotlib tqdm scipy pandas
python -c "import torch; print('cuda?', torch.cuda.is_available())"   # → True
```

**Do NOT `uv sync`** — clobbers the CUDA torch wheel (gotcha #2).

For the model: `NousResearch/Meta-Llama-3.1-8B` is **ungated** (gotcha #6),
so no `HF_TOKEN` needed. If somebody insists on the official mirror:
`export HF_TOKEN=hf_xxx` then use `meta-llama/Llama-3.1-8B`.

### P4. Verify or refresh manifold activations

`steer.py` reads `cache/<tag>/{years,age,temperature,colors,geography}.pt`
via `load_manifold_data`. If they're missing, `load_split` **silently
skips** the manifold (`skip {name} (not cached)` line in stdout) — that's
the only footgun in the run.

```bash
# on the pod
ls -la /workspace/goodfire/cache/meta-llama-3.1-8b_L16/*.pt 2>/dev/null
```

If empty / missing, re-extract (one-off, ~5–10 min):

```bash
cd /workspace/goodfire
SAE_MODEL_NAME=NousResearch/Meta-Llama-3.1-8B SAE_LAYER=16 \
SAE_D_MODEL=4096 SAE_CACHE_TAG=meta-llama-3.1-8b_L16 \
nohup python data.py > cache/meta-llama-3.1-8b_L16/extract.log 2>&1 &
```

Watch for `Cached {name}: torch.Size([...])` lines for all 5 manifolds.

---

## The run

Single foreground process is fine — total wall time is small enough to
babysit, and the dual-process gotcha (#4) bites only when we leave
background runs orphaned.

```bash
# on the pod, in /workspace/goodfire
SAE_MODEL_NAME=NousResearch/Meta-Llama-3.1-8B SAE_LAYER=16 \
SAE_D_MODEL=4096 SAE_CACHE_TAG=meta-llama-3.1-8b_L16 \
python prototype/steer.py 2>&1 | tee cache/meta-llama-3.1-8b_L16/steer/run.log
```

(`SAE_DEVICE` left unset → auto-selects cuda; cache lands in
`cache/<tag>/steer/`, not the flat `cache/steer/` 135M dir — no `--force`
needed for safety, but **never `--force` against the flat dir** anyway
(gotcha #5).)

### What you'll see, in order

1. `Computing legible steering vector...` — trains the factored SAE
   (~1–2 min on A100).
2. `target=temperature dom_chart=<int> ||v||=<float>` — sanity: the
   dominant chart should be a stable integer, ||v|| should be plausibly
   larger than the 135M run (8B activations have larger raw scale).
3. `Loading model...` — 16 GB download + load (~2–3 min if not cached on
   /workspace, ~30s if cached).
4. `contrast tokens: ' hot'=<id> ' cold'=<id>` — **must be a single token
   per word** for the readout to be clean. Llama-3.1's BPE gives single
   tokens for both. If the IDs look like multi-token tails, stop and
   investigate.
5. 7 α-values × 2 conditions = 14 lines of `logit(hot-cold)=±N.NNN`.
6. `Saved …/steering.png` then `Saved …/results.json`.
7. Two slope lines:
   `legible axis: logit-contrast slope per unit α = +X.XXX`,
   `random control: … = ±Y.YYY`.

### Success criterion

**Headline (matches 135M and validates the causal claim at scale):**
- Legible slope is **monotone**, +0.10 or larger per unit α (135M was
  +0.231).
- Random control slope is **flat**, |slope| ≲ 0.03.
- Ratio ≥ ~5× (135M was ~13×).

**If the 8B slope is materially weaker than 135M's** (e.g. < 0.05 or
non-monotone), that itself is a real result — write it up honestly in the
new §9b.7. Possibility space: (a) at 8B the temperature axis is one of the
"already-readable" axes where PCA's R² = 0.99 (§9b.2), so curvature buys
less causal directionality; (b) different layer pattern; (c) bf16
quantisation noise. Don't gloss it — document.

### Optional polish (only if pod time remains and the headline is clean)

- Re-run with `TARGET = "years"` (richer curvature win at 8B: VE@3 +0.09
  over PCA). Pick a readout prompt + contrast pair that maps to the year
  axis (e.g. prompt "The year is", contrast `" 2020"` vs `" 1920"` — but
  check token boundaries: years may be multi-token in Llama BPE; use
  `' twenty'` vs `' nineteen'` if so). Cheap, single-script edit.

---

## Post-flight

### Pull results back

The results are tiny (one PNG + one JSON). Use the same pull pattern as
the prior run:

```bash
# on the pod
cd /workspace/goodfire
tar czf /tmp/steer-8b.tgz cache/meta-llama-3.1-8b_L16/steer/
runpodctl send /tmp/steer-8b.tgz   # prints a one-time code

# on the Mac
cd ~/Documents/goodfire
runpodctl receive <code>
tar xzf steer-8b.tgz   # merges into cache/meta-llama-3.1-8b_L16/steer/
```

### Stop the pod

Stop immediately. The headline is in `results.json`; nothing left to do
on the pod.

### Writeup + commit

On the Mac (no GPU):

1. Add §9b.7 to `WRITEUP.md` — short subsection under Part VIII, mirroring
   the prose template in §9b.6 (1 paragraph of result + 1 of caveats).
   Cite slopes, ratio vs random, ||v||. Forward-pointer this to Part VII
   from `## 9. Part VII` so a sequential reader sees the 8B confirmation.
2. Tick off "Steering (Part VII) was not re-run at 8B" in the §9b.6
   limitations bullet and update the §10 / §12 status lines.
3. Update `prototype/STORY.md` with a Part VIII steering line.
4. Update the project memory (`goodfire-sae-manifold-repro`) — replace
   "What 8B left UN-TESTED" steering bullet with the 8B numbers.
5. Commit on `prototype/factored-manifold-sae`:
   ```bash
   git add WRITEUP.md prototype/STORY.md cache/meta-llama-3.1-8b_L16/steer/
   git commit -m "Add §9b.7: causal steering reproduces at 8B (#13a)"
   ```

---

## Hard-won gotchas — re-stated for this run

1. **Device hardcodes** — `steer.py` was patched (`vt` follows model
   device + dtype). If you copy code from steer to a new script, watch
   for this exact bug elsewhere.
2. **Don't `uv sync`** — already noted in P3.
3. **rsync needs `--no-perms --no-owner --no-group`** on `/workspace`.
4. **Single-process discipline** — this run is small, but if you ever
   re-launch, the canonical kill before relaunching:
   `pkill -f "steer.py|run_all.py|fair_comparison.py|legible_coord.py|iso_parsimony.py|iso_coord.py|adaptive_parsimony.py|train_sae.py|background.py|data.py"`.
5. **Never `--force` against the flat `cache/` dir.** `SAE_CACHE_TAG`
   namespaces output safely.
6. **HF gating workaround** — use `NousResearch/Meta-Llama-3.1-8B` (P3).
7. **Low-GPU-util is OK** — the factored-SAE retrain is small models on
   small batches; GPU util 20–30 % is kernel-launch overhead, not a bug.

---

*Last updated: 2026-05-28 (post #13b commit; #13a is the next action).*
