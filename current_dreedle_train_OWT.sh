#!/bin/bash
set -euo pipefail

# vvv THOG
# Long single-candidate dreedle SHEET OpenWebText run.
# This targets high VRAM use on TITAN RTX by widening the SHEET geometry rather than increasing context length.
# Default candidate is wide SHEET-144: D2048/H32 with P80/Q256 at C256, backed off after larger D2304 and D2560 OOMs.
# The child SHEET wrapper's dreedle runtime profile selects float16; do not also pass -T unless deliberately overriding it.
# ^^^ THOG

cd "$(dirname "$0")"

SHEET_WRAPPER="./current_scruffy_train_SHEET_OWT.sh"
DATA_DIR="${THOG2_OWT_DATA_DIR:-$HOME/git/thog/data/openwebtext}"
RUN_NAME="KARITANE_LONG_$(date +%y%m%d_%H%M%S)"
STEPS=99999
BATCH_SIZE=12
GRADIENT_ACCUMULATION_STEPS=4
EVAL_ITERS=50
EVAL_INTERVAL=100
LOG_INTERVAL=1
WARMUP_ITERS=20
CHECKPOINT_INTERVAL=500
N_LAYER=144
N_HEAD=32
N_EMBD=2048
BLOCK_SIZE=256
DEPTH_ORDER=80
BASE_ROW_ORDER=256
ACTIVATION_CHECKPOINTING=true
CHECKPOINT_SEGMENT_SIZE=12
INSTRUMENTATION="wandb"
CURVE_ROOT="curves"
WANDB_MODE="online"
WANDB_ENABLED=true
DRY_RUN=false
CUDA_DEVICE_DEFAULT=0

usage() {
  cat <<EOF
Usage: $0 [options]

Runs one long, memory-heavy dreedle SHEET OpenWebText candidate.

Options:
  -t DATA_DIR=${DATA_DIR}
  -g RUN_NAME=${RUN_NAME}
  -n STEPS=${STEPS}
  -b BATCH_SIZE=${BATCH_SIZE}
  -A GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}
  -u EVAL_ITERS=${EVAL_ITERS}
  -e EVAL_INTERVAL=${EVAL_INTERVAL}
  -l LOG_INTERVAL=${LOG_INTERVAL}
  -w WARMUP_ITERS=${WARMUP_ITERS}
  -k CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL}
  -L N_LAYER=${N_LAYER}
  -H N_HEAD=${N_HEAD}
  -D N_EMBD=${N_EMBD}
  -C BLOCK_SIZE=${BLOCK_SIZE}
  -P DEPTH_ORDER=${DEPTH_ORDER}
  -Q BASE_ROW_ORDER=${BASE_ROW_ORDER}
  -p ACTIVATION_CHECKPOINTING=${ACTIVATION_CHECKPOINTING}
  -S CHECKPOINT_SEGMENT_SIZE=${CHECKPOINT_SEGMENT_SIZE}
  -I INSTRUMENTATION=${INSTRUMENTATION}             tensorboard | wandb | none
  -V CURVE_ROOT=${CURVE_ROOT}                       TensorBoard root directory
  -M WANDB_MODE=${WANDB_MODE}                       online | offline | disabled
  -W WANDB_ENABLED=${WANDB_ENABLED}
  -x DRY_RUN=${DRY_RUN}
  -c CUDA_VISIBLE_DEVICES default=${CUDA_DEVICE_DEFAULT}
  -h show this help
EOF
}

while getopts ":t:g:n:b:A:u:e:l:w:k:L:H:D:C:P:Q:p:S:I:V:M:W:x:c:h" option; do
  case "$option" in
    t) DATA_DIR="$OPTARG" ;;
    g) RUN_NAME="$OPTARG" ;;
    n) STEPS="$OPTARG" ;;
    b) BATCH_SIZE="$OPTARG" ;;
    A) GRADIENT_ACCUMULATION_STEPS="$OPTARG" ;;
    u) EVAL_ITERS="$OPTARG" ;;
    e) EVAL_INTERVAL="$OPTARG" ;;
    l) LOG_INTERVAL="$OPTARG" ;;
    w) WARMUP_ITERS="$OPTARG" ;;
    k) CHECKPOINT_INTERVAL="$OPTARG" ;;
    L) N_LAYER="$OPTARG" ;;
    H) N_HEAD="$OPTARG" ;;
    D) N_EMBD="$OPTARG" ;;
    C) BLOCK_SIZE="$OPTARG" ;;
    P) DEPTH_ORDER="$OPTARG" ;;
    Q) BASE_ROW_ORDER="$OPTARG" ;;
    p) ACTIVATION_CHECKPOINTING="$OPTARG" ;;
    S) CHECKPOINT_SEGMENT_SIZE="$OPTARG" ;;
    I) INSTRUMENTATION="$OPTARG" ;;
    V) CURVE_ROOT="$OPTARG" ;;
    M) WANDB_MODE="$OPTARG" ;;
    W) WANDB_ENABLED="$OPTARG" ;;
    x) DRY_RUN="$OPTARG" ;;
    c) CUDA_DEVICE_DEFAULT="$OPTARG" ;;
    h) usage; exit 0 ;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 2 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage >&2; exit 2 ;;
  esac
done

validate_positive_uint() { [[ "$1" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid $2: $1; expected a positive integer." >&2; exit 2; }; }
validate_nonnegative_uint() { [[ "$1" =~ ^[0-9]+$ ]] || { echo "Invalid $2: $1; expected a non-negative integer." >&2; exit 2; }; }
validate_true_false() { case "$1" in true|false) ;; *) echo "Invalid $2: $1; expected true or false." >&2; exit 2 ;; esac; }

for setting in "$STEPS" "$BATCH_SIZE" "$GRADIENT_ACCUMULATION_STEPS" "$EVAL_ITERS" "$EVAL_INTERVAL" "$LOG_INTERVAL" "$N_LAYER" "$N_HEAD" "$N_EMBD" "$BLOCK_SIZE" "$DEPTH_ORDER" "$BASE_ROW_ORDER" "$CHECKPOINT_SEGMENT_SIZE"; do
  validate_positive_uint "$setting" "numeric setting"
done
validate_nonnegative_uint "$WARMUP_ITERS" "WARMUP_ITERS"
validate_nonnegative_uint "$CHECKPOINT_INTERVAL" "CHECKPOINT_INTERVAL"
validate_true_false "$ACTIVATION_CHECKPOINTING" "ACTIVATION_CHECKPOINTING"
validate_true_false "$WANDB_ENABLED" "WANDB_ENABLED"
validate_true_false "$DRY_RUN" "DRY_RUN"
case "$INSTRUMENTATION" in tensorboard|wandb|none) ;; *) echo "INSTRUMENTATION must be tensorboard, wandb, or none." >&2; exit 2 ;; esac
case "$WANDB_MODE" in online|offline|disabled) ;; *) echo "WANDB_MODE must be online, offline, or disabled." >&2; exit 2 ;; esac
[[ -f "$SHEET_WRAPPER" ]] || { echo "Missing SHEET wrapper: $SHEET_WRAPPER" >&2; exit 2; }
[[ -f "$DATA_DIR/train.bin" && -f "$DATA_DIR/val.bin" ]] || { echo "DATA_DIR must contain train.bin and val.bin: $DATA_DIR" >&2; exit 2; }
(( WARMUP_ITERS < STEPS )) || { echo "WARMUP_ITERS must be less than STEPS." >&2; exit 2; }
(( N_EMBD % N_HEAD == 0 )) || { echo "N_EMBD must be divisible by N_HEAD." >&2; exit 2; }
(( DEPTH_ORDER <= N_LAYER )) || { echo "DEPTH_ORDER must not exceed N_LAYER." >&2; exit 2; }
(( BASE_ROW_ORDER <= N_EMBD )) || { echo "BASE_ROW_ORDER must not exceed N_EMBD." >&2; exit 2; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$CUDA_DEVICE_DEFAULT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export THOG2_INSTRUMENTATION="$INSTRUMENTATION"
export THOG2_CURVE_ROOT="$CURVE_ROOT"

cat <<EOF
THOG2 dreedle long SHEET OpenWebText run
  run name:                 $RUN_NAME
  data dir:                 $DATA_DIR
  geometry:                 L$N_LAYER / H$N_HEAD / D$N_EMBD / C$BLOCK_SIZE
  sheet orders:             P$DEPTH_ORDER / Q$BASE_ROW_ORDER
  steps:                    $STEPS, intended for manual stop
  batch / global accum:     $BATCH_SIZE / $GRADIENT_ACCUMULATION_STEPS
  tokens/update:            $((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * BLOCK_SIZE))
  eval:                     every $EVAL_INTERVAL updates, $EVAL_ITERS batches
  checkpoint interval:      $CHECKPOINT_INTERVAL
  activation checkpointing: $ACTIVATION_CHECKPOINTING, segment $CHECKPOINT_SEGMENT_SIZE
  CUDA_VISIBLE_DEVICES:     $CUDA_VISIBLE_DEVICES
  cuda alloc conf:          $PYTORCH_CUDA_ALLOC_CONF
  instrumentation:          $THOG2_INSTRUMENTATION
  curve root:               $THOG2_CURVE_ROOT
  wandb flags:              $WANDB_ENABLED ($WANDB_MODE)
EOF

"$SHEET_WRAPPER" \
  -R dreedle \
  -q fresh \
  -t "$DATA_DIR" \
  -g "$RUN_NAME" \
  -n "$STEPS" \
  -b "$BATCH_SIZE" \
  -A "$GRADIENT_ACCUMULATION_STEPS" \
  -u "$EVAL_ITERS" \
  -e "$EVAL_INTERVAL" \
  -l "$LOG_INTERVAL" \
  -w "$WARMUP_ITERS" \
  -k "$CHECKPOINT_INTERVAL" \
  -L "$N_LAYER" \
  -H "$N_HEAD" \
  -D "$N_EMBD" \
  -C "$BLOCK_SIZE" \
  -P "$DEPTH_ORDER" \
  -Q "$BASE_ROW_ORDER" \
  -p "$ACTIVATION_CHECKPOINTING" \
  -S "$CHECKPOINT_SEGMENT_SIZE" \
  -M "$WANDB_MODE" \
  -W "$WANDB_ENABLED" \
  -x "$DRY_RUN"
