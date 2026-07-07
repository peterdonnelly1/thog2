#!/bin/bash
set -euo pipefail

# vvv THOG
# Current scruffy OpenWebText training wrapper.
# Defaults are intentionally non-tiny: SHEET L144 H12 D768 C1024, batch 3, global accumulation 160, bfloat16, flash2.
# DENSE is reachable with -O dense; unless -L/-H/-D are explicitly supplied, dense auto-drops to GPT2-small-ish L12 H12 D768.
# Main compact knobs: -p preset, -B basis, -a attention_geometry, -m mlp_geometry, -K attention_backend.
# Presets: legacy_sheet_col | curve | mlp_block | head_aware_block | block. Basis: chebyshev | dct.
# ^^^ THOG

cd "$(dirname "$0")"
MODEL_TYPE=sheet
HOST=scruffy
DTYPE=bfloat16
BACKEND=flash2
DATA_DIR="${THOG2_OWT_DATA_DIR:-data/openwebtext}"
RUN_NAME=""
PRESET=curve
BASIS=chebyshev
ATTN=""
MLP=""
STEPS=250
BATCH=3
ACCUM=160
LAYERS=144
HEADS=12
EMBD=768
CTX=1024
DEPTH_ORDER=32
BASE_ROW_ORDER=64
CHECKPOINT_SEGMENT_SIZE=12
DRY=false
LAYERS_SET=false
HEADS_SET=false
EMBD_SET=false
usage(){ echo "Usage: $0 [-O dense|sheet] [-p preset] [-B chebyshev|dct] [-a attention_geometry] [-m mlp_geometry] [-n steps] [-b batch] [-A accum] [-L layers] [-H heads] [-D embd] [-C ctx] [-P depth_order] [-Q base_row_order] [-K auto|flash2|sdpa|math] [-T dtype] [-t data_dir] [-g run_name] [-S checkpoint_segment_size] [-x true|false]"; }
while getopts ":O:p:B:a:m:n:b:A:L:H:D:C:P:Q:K:T:t:g:S:x:h" opt; do case "$opt" in O) MODEL_TYPE="$OPTARG";; p) PRESET="$OPTARG";; B) BASIS="$OPTARG";; a) ATTN="$OPTARG";; m) MLP="$OPTARG";; n) STEPS="$OPTARG";; b) BATCH="$OPTARG";; A) ACCUM="$OPTARG";; L) LAYERS="$OPTARG"; LAYERS_SET=true;; H) HEADS="$OPTARG"; HEADS_SET=true;; D) EMBD="$OPTARG"; EMBD_SET=true;; C) CTX="$OPTARG";; P) DEPTH_ORDER="$OPTARG";; Q) BASE_ROW_ORDER="$OPTARG";; K) BACKEND="$OPTARG";; T) DTYPE="$OPTARG";; t) DATA_DIR="$OPTARG";; g) RUN_NAME="$OPTARG";; S) CHECKPOINT_SEGMENT_SIZE="$OPTARG";; x) DRY="$OPTARG";; h) usage; exit 0;; *) usage; exit 2;; esac; done
case "$MODEL_TYPE" in dense|sheet) ;; *) echo "bad MODEL_TYPE=$MODEL_TYPE" >&2; exit 2;; esac
case "$PRESET" in legacy_sheet_col|curve|mlp_block|head_aware_block|block) ;; *) echo "bad PRESET=$PRESET" >&2; exit 2;; esac
case "$BASIS" in chebyshev|dct) ;; *) echo "bad BASIS=$BASIS" >&2; exit 2;; esac
case "$BACKEND" in auto|flash2|sdpa|math) ;; *) echo "bad BACKEND=$BACKEND" >&2; exit 2;; esac
case "$DRY" in true|false) ;; *) echo "bad DRY=$DRY" >&2; exit 2;; esac
if [[ "$MODEL_TYPE" == dense ]]; then
  [[ "$LAYERS_SET" == false ]] && LAYERS=12
  [[ "$HEADS_SET" == false ]] && HEADS=12
  [[ "$EMBD_SET" == false ]] && EMBD=768
fi
BASIS_TAG=CHEBY; [[ "$BASIS" == dct ]] && BASIS_TAG=DCT
PRESET_TAG="${PRESET^^}"
[[ "$PRESET" == legacy_sheet_col ]] && PRESET_TAG=SHEET_COL
[[ "$MODEL_TYPE" == dense ]] && RUN_TAG=DENSE || RUN_TAG="${BASIS_TAG}_${PRESET_TAG}"
[[ -z "$RUN_NAME" ]] && RUN_NAME="${RUN_TAG}_OWT"
EXTRA=()
[[ -n "$ATTN" ]] && EXTRA+=(--attention-geometry "$ATTN")
[[ -n "$MLP" ]] && EXTRA+=(--mlp-geometry "$MLP")
CMD=(python -m run_thog2_owt_stage8 --model-type "$MODEL_TYPE" --host-label "$HOST" --run-name "$RUN_NAME" --data-dir "$DATA_DIR" --max-iters "$STEPS" --batch-size "$BATCH" --gradient-accumulation-steps "$ACCUM" --n-layer "$LAYERS" --n-head "$HEADS" --n-embd "$EMBD" --block-size "$CTX" --depth-order "$DEPTH_ORDER" --base-row-order "$BASE_ROW_ORDER" --geometry-preset "$PRESET" --basis-family "$BASIS" --attention-backend "$BACKEND" --dtype "$DTYPE" --wandb --wandb-mode online --activation-checkpointing --checkpoint-segment-size "$CHECKPOINT_SEGMENT_SIZE" "${EXTRA[@]}")
echo "scruffy OWT train: model=$MODEL_TYPE preset=$PRESET basis=$BASIS backend=$BACKEND dtype=$DTYPE run=$RUN_NAME L=$LAYERS H=$HEADS D=$EMBD C=$CTX batch=$BATCH accum=$ACCUM"
if [[ "$DRY" == true ]]; then "${CMD[@]}" --dry-run; printf 'DRY RUN:'; printf ' %q' "${CMD[@]}"; printf '\n'; exit 0; fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
"${CMD[@]}"
