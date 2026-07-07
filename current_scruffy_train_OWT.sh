#!/bin/bash
set -euo pipefail

# vvv THOG
# Current scruffy OpenWebText training wrapper.
# Defaults: bfloat16, attention_backend=flash2, SHEET curve, W&B online.
# Main knobs: -p preset, -B basis, -a attention_geometry, -m mlp_geometry, -K attention_backend.
# Presets: legacy_sheet_col | curve | mlp_block | head_aware_block | block.
# Basis: chebyshev | dct.
# ^^^ THOG

cd "$(dirname "$0")"
HOST=scruffy DTYPE=bfloat16 BACKEND=flash2 DATA_DIR="${THOG2_OWT_DATA_DIR:-data/openwebtext}" RUN_NAME="" PRESET=curve BASIS=chebyshev ATTN="" MLP="" STEPS=250 BATCH=3 ACCUM=160 LAYERS=144 HEADS=12 EMBD=768 CTX=1024 P=32 Q=64 DRY=false
usage(){ echo "Usage: $0 [-p preset] [-B chebyshev|dct] [-a attention_geometry] [-m mlp_geometry] [-n steps] [-b batch] [-A accum] [-L layers] [-H heads] [-D embd] [-C ctx] [-P depth_order] [-Q base_row_order] [-K auto|flash2|sdpa|math] [-T dtype] [-t data_dir] [-g run_name] [-x true|false]"; }
while getopts ":p:B:a:m:n:b:A:L:H:D:C:P:Q:K:T:t:g:x:h" opt; do case "$opt" in p) PRESET="$OPTARG";; B) BASIS="$OPTARG";; a) ATTN="$OPTARG";; m) MLP="$OPTARG";; n) STEPS="$OPTARG";; b) BATCH="$OPTARG";; A) ACCUM="$OPTARG";; L) LAYERS="$OPTARG";; H) HEADS="$OPTARG";; D) EMBD="$OPTARG";; C) CTX="$OPTARG";; P) P="$OPTARG";; Q) Q="$OPTARG";; K) BACKEND="$OPTARG";; T) DTYPE="$OPTARG";; t) DATA_DIR="$OPTARG";; g) RUN_NAME="$OPTARG";; x) DRY="$OPTARG";; h) usage; exit 0;; *) usage; exit 2;; esac; done
BASIS_TAG=CHEBY; [[ "$BASIS" == dct ]] && BASIS_TAG=DCT
PRESET_TAG="${PRESET^^}"; PRESET_TAG="${PRESET_TAG//_/-}"
[[ -z "$RUN_NAME" ]] && RUN_NAME="${BASIS_TAG}_${PRESET_TAG}_OWT"
EXTRA=(); [[ -n "$ATTN" ]] && EXTRA+=(--attention-geometry "$ATTN"); [[ -n "$MLP" ]] && EXTRA+=(--mlp-geometry "$MLP")
CMD=(python -m run_thog2_owt_stage8 --model-type sheet --host-label "$HOST" --run-name "$RUN_NAME" --data-dir "$DATA_DIR" --max-iters "$STEPS" --batch-size "$BATCH" --gradient-accumulation-steps "$ACCUM" --n-layer "$LAYERS" --n-head "$HEADS" --n-embd "$EMBD" --block-size "$CTX" --depth-order "$P" --base-row-order "$Q" --geometry-preset "$PRESET" --basis-family "$BASIS" --attention-backend "$BACKEND" --dtype "$DTYPE" --wandb --wandb-mode online --activation-checkpointing --checkpoint-segment-size 12 "${EXTRA[@]}")
echo "scruffy OWT train: preset=$PRESET basis=$BASIS backend=$BACKEND dtype=$DTYPE run=$RUN_NAME"
if [[ "$DRY" == true ]]; then "${CMD[@]}" --dry-run; printf 'DRY RUN:'; printf ' %q' "${CMD[@]}"; printf '\n'; exit 0; fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
"${CMD[@]}"
