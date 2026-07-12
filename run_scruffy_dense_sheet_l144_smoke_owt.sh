#!/bin/bash
set -euo pipefail

# vvv THOG
# Short sequential DENSE/FULL_BLOCK smoke pair using the canonical scruffy wrapper.
# ^^^ THOG

cd "$(dirname "$0")"

RUN_NAME="PICTON_SMOKE_$(date +%y%m%d_%H%M%S)"
STEPS=20
BATCH_SIZE=12
GRADIENT_ACCUMULATION_STEPS=4
EVAL_ITERS=5
EVAL_INTERVAL=10
LOG_INTERVAL=1
WARMUP_ITERS=1
N_LAYER=144
N_HEAD=6
N_EMBD=384
BLOCK_SIZE=256
O_DEPTH=32
O_ATTN_D_MODEL=64
O_ATTN_QKV_PER_CHANNEL=8
O_ATTN_OUT_PER_CHANNEL=8
O_MLP_D_MODEL=64
O_MLP_HIDDEN=256
CHECKPOINT_SEGMENT_SIZE=12
WANDB_MODE="online"
WANDB_ENABLED=true
RUN_SELECTION="both"
DRY_RUN=false

usage() {
  cat <<EOF_USAGE
Usage: $0 [options]

  -g RUN_NAME=${RUN_NAME}
  -n STEPS=${STEPS}
  -b BATCH_SIZE=${BATCH_SIZE}
  -A GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}
  -u EVAL_ITERS=${EVAL_ITERS}
  -e EVAL_INTERVAL=${EVAL_INTERVAL}
  -l LOG_INTERVAL=${LOG_INTERVAL}
  -w WARMUP_ITERS=${WARMUP_ITERS}
  -L N_LAYER=${N_LAYER}
  -H N_HEAD=${N_HEAD}
  -D N_EMBD=${N_EMBD}
  -C BLOCK_SIZE=${BLOCK_SIZE}
  -P O_DEPTH=${O_DEPTH}
  -Q O_ATTN_D_MODEL=${O_ATTN_D_MODEL}
  -J O_ATTN_QKV_PER_CHANNEL=${O_ATTN_QKV_PER_CHANNEL}
  -O O_ATTN_OUT_PER_CHANNEL=${O_ATTN_OUT_PER_CHANNEL}
  -X O_MLP_D_MODEL=${O_MLP_D_MODEL}
  -Y O_MLP_HIDDEN=${O_MLP_HIDDEN}
  -S CHECKPOINT_SEGMENT_SIZE=${CHECKPOINT_SEGMENT_SIZE}
  -M WANDB_MODE=${WANDB_MODE}
  -W WANDB_ENABLED=${WANDB_ENABLED}
  -r RUN_SELECTION=${RUN_SELECTION}                dense | full_block | both
  -x DRY_RUN=${DRY_RUN}
  -h show this help
EOF_USAGE
}

while getopts ":g:n:b:A:u:e:l:w:L:H:D:C:P:Q:J:O:X:Y:S:M:W:r:x:h" option; do
  case "$option" in
    g) RUN_NAME="$OPTARG" ;; n) STEPS="$OPTARG" ;; b) BATCH_SIZE="$OPTARG" ;;
    A) GRADIENT_ACCUMULATION_STEPS="$OPTARG" ;; u) EVAL_ITERS="$OPTARG" ;;
    e) EVAL_INTERVAL="$OPTARG" ;; l) LOG_INTERVAL="$OPTARG" ;; w) WARMUP_ITERS="$OPTARG" ;;
    L) N_LAYER="$OPTARG" ;; H) N_HEAD="$OPTARG" ;; D) N_EMBD="$OPTARG" ;; C) BLOCK_SIZE="$OPTARG" ;;
    P) O_DEPTH="$OPTARG" ;; Q) O_ATTN_D_MODEL="$OPTARG" ;; J) O_ATTN_QKV_PER_CHANNEL="$OPTARG" ;;
    O) O_ATTN_OUT_PER_CHANNEL="$OPTARG" ;; X) O_MLP_D_MODEL="$OPTARG" ;; Y) O_MLP_HIDDEN="$OPTARG" ;;
    S) CHECKPOINT_SEGMENT_SIZE="$OPTARG" ;; M) WANDB_MODE="$OPTARG" ;; W) WANDB_ENABLED="$OPTARG" ;;
    r) RUN_SELECTION="$OPTARG" ;; x) DRY_RUN="$OPTARG" ;;
    h) usage; exit 0 ;; :) echo "Option -$OPTARG requires an argument." >&2; exit 2 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage >&2; exit 2 ;;
  esac
done

case "$RUN_SELECTION" in dense|full_block|both) ;; *) echo "RUN_SELECTION must be dense, full_block, or both." >&2; exit 2 ;; esac

COMMON_ARGS=(
  -q fresh -g "$RUN_NAME" -n "$STEPS" -b "$BATCH_SIZE" -A "$GRADIENT_ACCUMULATION_STEPS"
  -u "$EVAL_ITERS" -e "$EVAL_INTERVAL" -l "$LOG_INTERVAL" -w "$WARMUP_ITERS"
  -L "$N_LAYER" -H "$N_HEAD" -D "$N_EMBD" -C "$BLOCK_SIZE"
  -P "$O_DEPTH" -Q "$O_ATTN_D_MODEL" -J "$O_ATTN_QKV_PER_CHANNEL"
  -O "$O_ATTN_OUT_PER_CHANNEL" -X "$O_MLP_D_MODEL" -Y "$O_MLP_HIDDEN"
  -S "$CHECKPOINT_SEGMENT_SIZE" -M "$WANDB_MODE" -W "$WANDB_ENABLED" -x "$DRY_RUN"
)

printf '\nPICTON DENSE/FULL_BLOCK L144 smoke pair\n'
printf '  orders: P%s Q%s J%s O%s X%s Y%s\n' "$O_DEPTH" "$O_ATTN_D_MODEL" "$O_ATTN_QKV_PER_CHANNEL" "$O_ATTN_OUT_PER_CHANNEL" "$O_MLP_D_MODEL" "$O_MLP_HIDDEN"

if [[ "$RUN_SELECTION" == dense || "$RUN_SELECTION" == both ]]; then
  ./current_scruffy_train_OWT.sh -p dense "${COMMON_ARGS[@]}"
fi
if [[ "$RUN_SELECTION" == full_block || "$RUN_SELECTION" == both ]]; then
  ./current_scruffy_train_OWT.sh -p full_block "${COMMON_ARGS[@]}"
fi
# ^^^ THOG
