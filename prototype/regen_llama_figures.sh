#!/usr/bin/env bash
#
# regen_llama_figures.sh — regenerate the Llama-8B figures that are missing from
# the cache, so the Llama run mirrors the SmolLM2-135M run's figure set.
#
# RUN THIS ON THE GPU BOX (RunPod), from the repo root — it needs the Llama-8B
# manifold activations, SAE checkpoint, and (for some tiers) the model itself,
# none of which live on the laptop. See RUNPOD.md.
#
# It is honest about prerequisites: figures whose inputs are present are built;
# figures whose inputs are missing are SKIPPED with an explanation (and the flag
# that would produce the missing input). Nothing is fabricated.
#
# Tiers
#   A (default, no model load) : viz/ legibility figures, subspace tuning_curves,
#                                [opt] iso_coord pareto, + PDF->PNG export.
#   B (--extract-days)         : loads the model to extract the `days` manifold,
#                                then builds days_manifold3d + days_greedy_ve.
#   C (--train-extra-saes ...) : trains extra-k SAEs, then compare_sparsity to
#                                build the *_sparsity_sweep figures (needs >=2 SAEs).
#
# Usage
#   bash prototype/regen_llama_figures.sh
#   bash prototype/regen_llama_figures.sh --with-iso-coord
#   bash prototype/regen_llama_figures.sh --extract-days
#   bash prototype/regen_llama_figures.sh --train-extra-saes 16 64
#   bash prototype/regen_llama_figures.sh --extract-days --train-extra-saes 16 64 --with-iso-coord --force
#
set -euo pipefail

# ── environment: matches the existing cache/meta-llama-3.1-8b_L16 run ─────────
export SAE_MODEL_NAME="${SAE_MODEL_NAME:-meta-llama/Llama-3.1-8B}"
export SAE_LAYER="${SAE_LAYER:-16}"
export SAE_D_MODEL="${SAE_D_MODEL:-4096}"
export SAE_CACHE_TAG="${SAE_CACHE_TAG:-meta-llama-3.1-8b_L16}"
# SAE_DEVICE inherits the box default (cuda). Export it before calling to override.

TAG="$SAE_CACHE_TAG"
CACHE="cache/$TAG"
SAE="${SAE_PATH:-$CACHE/sae_32768_k32.pt}"
K="${SAE_K:-32}"
SUBDIR="$CACHE/subspace_capture"

# Manifolds actually extracted for this Llama run (extract.log). `days` is NOT
# among them — it is handled only by the --extract-days tier.
EXISTING="years age temperature colors geography"

# ── flags ────────────────────────────────────────────────────────────────────
FORCE=0; EXTRACT_DAYS=0; WITH_ISO_COORD=0; EXTRA_KS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)            FORCE=1; shift ;;
    --extract-days)     EXTRACT_DAYS=1; shift ;;
    --with-iso-coord)   WITH_ISO_COORD=1; shift ;;
    --train-extra-saes) shift
                        while [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; do EXTRA_KS="$EXTRA_KS $1"; shift; done ;;
    -h|--help)          sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

PY="uv run python"
made=(); skipped=()
log()  { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
note() { printf '   %s\n' "$*"; }
have() { [[ -s "$1" ]]; }                       # file exists & non-empty
need_force() { [[ -s "$1" && $FORCE -eq 0 ]]; } # already there and not --force

# pdftoppm (poppler) preferred; sips fallback (macOS). One of them is enough.
pdf2png() {  # $1 = pdf, $2 = out png (no ext for pdftoppm)
  if command -v pdftoppm >/dev/null 2>&1; then
    pdftoppm -png -r 150 -singlefile "$1" "$2"
  elif command -v sips >/dev/null 2>&1; then
    sips -s format png "$1" --out "$2.png" >/dev/null
  else
    return 1
  fi
}

# ── preflight ────────────────────────────────────────────────────────────────
log "Preflight"
note "tag=$TAG  layer=$SAE_LAYER  d_model=$SAE_D_MODEL"
note "cache=$CACHE"
note "sae=$SAE  (k=$K)"
[[ -d "$CACHE" ]] || { echo "ERROR: $CACHE not found — are you on the box, at repo root?" >&2; exit 1; }
if ! have "$SAE"; then
  echo "ERROR: SAE checkpoint $SAE not found." >&2
  echo "       Found these instead:" >&2
  ls -1 "$CACHE"/sae_*.pt 2>/dev/null | sed 's/^/         /' >&2 || echo "         (none)" >&2
  echo "       Set SAE_PATH=... and SAE_K=... to point at the right checkpoint." >&2
  exit 1
fi
for m in $EXISTING; do
  have "$CACHE/$m.pt" || note "WARN: $CACHE/$m.pt missing — figures using it may be skipped/fail."
done

# ════════════════════════════════════════════════════════════════════════════
# TIER A — needs only the existing manifolds + the k$K SAE (no model load)
# ════════════════════════════════════════════════════════════════════════════

# A1: legibility figures -> viz/coord_vs_label.png, viz/colors_circle.png
log "A1  viz_coord (legibility figures)"
if need_force "$CACHE/viz/coord_vs_label.png" && need_force "$CACHE/viz/colors_circle.png"; then
  note "exist; skipping (use --force to rebuild)"; skipped+=("viz/* (already present)")
else
  $PY prototype/viz_coord.py
  made+=("viz/coord_vs_label.png" "viz/colors_circle.png")
fi

# A2: subspace tuning curves -> subspace_capture/<m>_tuning_curves.pdf  (5 manifolds)
log "A2  subspace_capture tuning (tuning curves)"
$PY subspace_capture.py tuning --sae "$SAE" --k "$K" --manifold $EXISTING
for m in $EXISTING; do have "$SUBDIR/${m}_tuning_curves.pdf" && made+=("subspace_capture/${m}_tuning_curves.pdf"); done

# A3 (optional): iso_coord pareto — exists for SmolLM as a side-experiment only.
if [[ $WITH_ISO_COORD -eq 1 ]]; then
  log "A3  iso_coord (pareto) [optional]"
  if need_force "$CACHE/iso_coord/pareto.png"; then
    note "exists; skipping (use --force)"; skipped+=("iso_coord/pareto.png (already present)")
  else
    $PY prototype/iso_coord.py --seeds 0 1 2 --lams 0 3 10 30 --manifolds $EXISTING
    made+=("iso_coord/pareto.png")
  fi
else
  skipped+=("iso_coord/pareto.png (SmolLM side-experiment; pass --with-iso-coord to mirror it)")
fi

# ════════════════════════════════════════════════════════════════════════════
# TIER B — the `days` manifold (NOT extracted for Llama). Loads the model.
# ════════════════════════════════════════════════════════════════════════════
log "B  days manifold figures"
if [[ $EXTRACT_DAYS -eq 1 ]]; then
  if ! have "$CACHE/days.pt"; then
    note "extracting days (loads $SAE_MODEL_NAME on GPU)…"
    $PY data.py --manifold days
  fi
  if have "$CACHE/days.pt"; then
    $PY manifold_viz.py --manifold days --sae "$SAE" --k "$K"
    have "$CACHE/manifold_viz/days_manifold3d.png" && made+=("manifold_viz/days_manifold3d.png")
    $PY subspace_capture.py plot --sae "$SAE" --k "$K" --manifold days
    have "$SUBDIR/days_greedy_ve.pdf" && made+=("subspace_capture/days_greedy_ve.pdf")
  else
    skipped+=("days_* (extraction produced no days.pt)")
  fi
else
  skipped+=("manifold_viz/days_manifold3d.png  (days not extracted; pass --extract-days)")
  skipped+=("subspace_capture/days_greedy_ve.pdf (days not extracted; pass --extract-days)")
fi

# ════════════════════════════════════════════════════════════════════════════
# TIER C — *_sparsity_sweep: needs >=2 SAE checkpoints at different k.
# Only sae_32768_k32.pt ships with this run. --train-extra-saes trains more.
# ════════════════════════════════════════════════════════════════════════════
log "C  subspace sparsity sweep"
SPARSITY_MANIFOLDS="years age temperature colors"   # the set SmolLM used
if [[ -n "$EXTRA_KS" ]]; then
  BG="$CACHE/background_acts_200000.dat"
  if ! have "$BG"; then
    note "ERROR: $BG not found — cannot train extra SAEs (run background.py first)."
    skipped+=("*_sparsity_sweep (no background activations to train extra SAEs)")
  else
    for k in $EXTRA_KS; do
      out="$CACHE/sae_32768_k${k}.pt"
      if have "$out"; then note "sae k=$k already present"; else
        note "training SAE k=$k (expansion 8 -> d_sae 32768)…"
        $PY train_sae.py --n-tokens 200000 --expansion-factor 8 --k "$k" --out "$out"
      fi
    done
  fi
fi
# Collect every SAE checkpoint present and pair each with its k from the filename.
SAES=(); KS=()
for f in "$CACHE"/sae_*_k*.pt; do
  [[ -e "$f" ]] || continue
  kk="$(basename "$f" | sed -E 's/.*_k([0-9]+)\.pt/\1/')"
  SAES+=("$f"); KS+=("$kk")
done
if [[ ${#SAES[@]} -ge 2 ]]; then
  $PY compare_sparsity.py --saes "${SAES[@]}" --ks "${KS[@]}" --manifold $SPARSITY_MANIFOLDS
  for m in $SPARSITY_MANIFOLDS; do have "$SUBDIR/${m}_sparsity_sweep.pdf" && made+=("subspace_capture/${m}_sparsity_sweep.pdf"); done
else
  note "only ${#SAES[@]} SAE checkpoint(s) present — sparsity sweep needs >=2 (different k)."
  note "re-run with e.g. --train-extra-saes 16 64 to build them."
  skipped+=("*_sparsity_sweep (need >=2 SAE k-variants; have ${#SAES[@]})")
fi

# ════════════════════════════════════════════════════════════════════════════
# PDF -> PNG: mirror the SmolLM run's subspace_capture/png/ folder
# ════════════════════════════════════════════════════════════════════════════
log "PNG export (subspace_capture/png/)"
mkdir -p "$SUBDIR/png"
conv=0
for pdf in "$SUBDIR"/*.pdf; do
  [[ -e "$pdf" ]] || continue
  base="$(basename "${pdf%.pdf}")"
  png="$SUBDIR/png/$base.png"
  if need_force "$png"; then continue; fi
  if pdf2png "$pdf" "$SUBDIR/png/$base"; then conv=$((conv+1)); else
    note "no pdftoppm/sips available — install poppler-utils to export PNGs."; break
  fi
done
note "converted $conv PDF(s) to PNG"

# ── summary ──────────────────────────────────────────────────────────────────
log "Summary"
echo "Built / refreshed:"
if [[ ${#made[@]} -eq 0 ]]; then echo "   (nothing new)"; else printf '   + %s\n' "${made[@]}"; fi
echo "Skipped:"
if [[ ${#skipped[@]} -eq 0 ]]; then echo "   (none)"; else printf '   - %s\n' "${skipped[@]}"; fi
echo
echo "Done. Pull the figures back to the laptop, e.g.:"
echo "   rsync -av <box>:$(pwd)/$CACHE/{viz,manifold_viz,subspace_capture,iso_coord} ./$CACHE/"
