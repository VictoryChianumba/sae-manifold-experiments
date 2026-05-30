# RUNPOD.md — running the experiment suite on a cloud GPU

The 8 GB M1 forced the project onto SmolLM2-135M. A rented GPU lifts that limit so we
can validate on the **paper's actual model, Llama-3.1-8B** (task #13). The whole
pipeline is model-agnostic via env vars, so the *same* commands that ran on the laptop
CPU run on the pod — only the env changes.

The orchestration is **`prototype/run_all.py`** (CPU-tested end-to-end). It runs
`extract → background → sae → fair → legible → iso → adaptive` per layer, each as a
subprocess writing into a per-`(model, layer)` cache tag, and **skips any stage whose
output already exists** (so a crash resumes; re-runs are cheap).

---

## 0. Pick a pod

- **GPU:** one 24 GB card is enough for Llama-3.1-8B fp16 *inference* (RTX 4090 ≈
  $0.4–0.7/hr; A100-40GB ≈ $1.5–2/hr for headroom). We never train the LLM — only
  forward-pass it for activations — so memory, not FLOPs, is the constraint.
- **Template:** start from a **PyTorch CUDA template** so `torch` is already CUDA-built
  and matched to the pod's driver. Do **not** rebuild torch (see §2).
- **Budget:** ≈ 6–12 GPU-hr of compute for the full #13–#17 list, + ~2–4 hr one-time
  setup (mostly the ~16 GB Llama download). ≈ $10–30 total.

## 1. Get the code onto the pod (no GitHub needed)

The code is ~26 files / ~100 KB. **Never upload `cache/` or `.venv/`** — activations are
regenerated on the GPU (the whole point), and `.venv` is macOS-arm, useless on Linux.

**Recommended — `runpodctl send` (peer-to-peer, no SSH-key setup):**
```bash
# on your Mac:
cd ~/Documents/goodfire
tar czf /tmp/goodfire-code.tgz --exclude=cache --exclude=.venv --exclude=.git .
runpodctl send /tmp/goodfire-code.tgz          # prints a one-time code

# on the pod (web terminal):
runpodctl receive <code>
mkdir -p goodfire && tar xzf goodfire-code.tgz -C goodfire && cd goodfire
```

**Alternative — `rsync` over SSH** (best for repeated edit→run loops). Two gotchas,
both learned live:
1. Register your public key in the RunPod console **before creating the pod** (keys are
   injected at pod creation, not retroactively), and use **"SSH over exposed TCP"** (a
   real `sshd` on a public IP+port) — RunPod's `ssh.runpod.io` *proxy* does **not**
   support rsync/scp. If your key has a passphrase, load it once with
   `ssh-add ~/.ssh/id_ed25519` so rsync can authenticate non-interactively.
2. `/workspace` is a network FS that forbids `chown`/`chmod`, so plain `rsync -a` dies
   with `chown … Operation not permitted`. Drop ownership/perm preservation:
```bash
rsync -rlvz --no-perms --no-owner --no-group \
  -e "ssh -p <PORT> -o StrictHostKeyChecking=accept-new -i ~/.ssh/id_ed25519" \
  --exclude cache --exclude .venv --exclude .git --exclude __pycache__ --exclude .DS_Store \
  ~/Documents/goodfire/ root@<IP>:/workspace/goodfire/
```

## 2. Environment on the pod

torch is preinstalled (CUDA template). Install only the rest, **without touching torch**:
```bash
pip install nnsight transformers datasets scikit-learn matplotlib tqdm scipy pandas
python -c "import torch; print('cuda?', torch.cuda.is_available())"   # must print True
```
(`uv sync` is for the Mac, where this lockfile was resolved — it can pull a
CUDA-mismatched torch on the pod. See the note in `pyproject.toml`.)

**Gated model auth** — Llama-3.1-8B requires an HF account that has accepted Meta's
license:
```bash
export HF_TOKEN=hf_xxx        # or: huggingface-cli login
```

## 3. Env vars (the only thing that differs from the laptop)

```bash
export SAE_MODEL_NAME=meta-llama/Llama-3.1-8B   # the paper's model (this is the default)
export SAE_D_MODEL=4096                          # Llama-3.1-8B hidden size (the default)
# leave SAE_DEVICE unset -> auto-selects cuda; SAE_LAYER is set per-layer by the driver
```
These are already the defaults in `data.py`, so on a GPU pod you can often skip them
entirely. (On the Mac we override to `SAE_DEVICE=cpu SAE_D_MODEL=576
SAE_MODEL_NAME=HuggingFaceTB/SmolLM2-135M`.)

## 4. Run it

Dry-run first to see the plan (no compute):
```bash
python prototype/run_all.py --layers 16 --seeds 0 1 2 --bg-tokens 200000 --dry-run
```
Then the real run (one layer; this is #13 + re-confirms #12/legibility at scale):
```bash
nohup python prototype/run_all.py --layers 16 --seeds 0 1 2 --bg-tokens 200000 \
    > cache/run_all.out 2>&1 &
```
- **Layer-sensitivity sweep (#13):** `--layers 12 16 20 24` (each layer = its own cache
  tag `llama-3.1-8b_L<layer>/`, extracted independently).
- **Statistical-rigor pass (#17):** bump `--seeds 0 1 2 3 4 5 6 7 8 9`.
- **Subset / resume:** `--stages fair legible adaptive`; finished stages auto-skip,
  `--force` re-runs, `--keep-going` continues past a failed layer.

Per-stage logs: `cache/<tag>/logs/<stage>.log`. Roll-up: `cache/run_all_summary.json`.

## 5. Get results back

Only the (small) result JSONs + PNGs are needed for the writeup — not the activations:
```bash
# on the pod:
tar czf /tmp/goodfire-results.tgz cache/*/fair_comparison cache/*/legible_coord \
    cache/*/iso_parsimony cache/*/adaptive_parsimony cache/run_all_summary.json
runpodctl send /tmp/goodfire-results.tgz
# on your Mac: runpodctl receive <code>; tar xzf ... into the repo to merge tagged dirs.
```

## 6. What's covered / not yet

`run_all.py` runs everything that exists today: **#13** (fair comparison on the real
model, multi-layer), the **#12** adaptive-parsimony retest (a real model gives genuinely
multi-D geometry — the setting adaptive parsimony needs), supervised legibility, global
iso+parsimony, and **#17** (just more `--seeds`). Not yet implemented — add as new
stages once their scripts exist, then re-run (activations are cached, so cheap):
**#14** in-the-wild router, **#15** group-sparse top-k charts, **#16** proper
isometric-AE. Best built *after* seeing #13 data so they're informed by it.

---

*Sanity-checked locally on SmolLM2-135M/CPU: cold-start `extract→background→sae` into a
namespaced cache tag, idempotent skip, and all four analysis stages running via
subprocess. The only pod-specific risks are torch/CUDA (§2) and HF gating (§2).*
