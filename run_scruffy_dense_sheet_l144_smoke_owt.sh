#!/bin/bash

set -euo pipefail

# vvv THOG
# Run a short sequential DENSE2/SHEET smoke pair on scruffy.
# This deliberately delegates to the canonical per-architecture wrappers so there is still only one backend path.
# ^^^ THOG

cd "$(dirname "$0")"

RUN_NAME="DANEVIRK2_SMOKE"
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
DEPTH_ORDER=32
BASE_ROW_ORDER=64
ACTIVATION_CHECKPOINTING=true
CHECKPOINT_SEGMENT_SIZE=12
WANDB_MODE="online"
WANDB_ENABLED=true
RUN_SELECTION="both"
DRY_RUN=false

usage() {
  cat <<EOF
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
  -P DEPTH_ORDER=${DEPTH_ORDER}                    SHEET only
  -Q BASE_ROW_ORDER=${BASE_ROW_ORDER}              SHEET only
  -p ACTIVATION_CHECKPOINTING=${ACTIVATION_CHECKPOINTING}
  -S CHECKPOINT_SEGMENT_SIZE=${CHECKPOINT_SEGMENT_SIZE}
  -M WANDB_MODE=${WANDB_MODE}                      online | offline | disabled
  -W WANDB_ENABLED=${WANDB_ENABLED}
  -r RUN_SELECTION=${RUN_SELECTION}                dense | sheet | both
  -x DRY_RUN=${DRY_RUN}
  -h show this help
EOF
}

while getopts ":g:n:b:A:u:e:l:w:L:H:D:C:P:Q:p:S:M:W:r:x:h" option; do
  case "$option" in
    g) RUN_NAME="$OPTARG" ;;
    n) STEPS="$OPTARG" ;;
    b) BATCH_SIZE="$OPTARG" ;;
    A) GRADIENT_ACCUMULATION_STEPS="$OPTARG" ;;
    u) EVAL_ITERS="$OPTARG" ;;
    e) EVAL_INTERVAL="$OPTARG" ;;
    l) LOG_INTERVAL="$OPTARG" ;;
    w) WARMUP_ITERS="$OPTARG" ;;
    L) N_LAYER="$OPTARG" ;;
    H) N_HEAD="$OPTARG" ;;
    D) N_EMBD="$OPTARG" ;;
    C) BLOCK_SIZE="$OPTARG" ;;
    P) DEPTH_ORDER="$OPTARG" ;;
    Q) BASE_ROW_ORDER="$OPTARG" ;;
    p) ACTIVATION_CHECKPOINTING="$OPTARG" ;;
    S) CHECKPOINT_SEGMENT_SIZE="$OPTARG" ;;
    M) WANDB_MODE="$OPTARG" ;;
    W) WANDB_ENABLED="$OPTARG" ;;
    r) RUN_SELECTION="$OPTARG" ;;
    x) DRY_RUN="$OPTARG" ;;
    h) usage; exit 0 ;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 2 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))

validate_positive_uint() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]] || {
    echo "Invalid $2: $1; expected a positive integer." >&2
    exit 2
  }
}

validate_nonnegative_uint() {
  [[ "$1" =~ ^[0-9]+$ ]] || {
    echo "Invalid $2: $1; expected a non-negative integer." >&2
    exit 2
  }
}

validate_true_false() {
  case "$1" in
    true|false) ;;
    *) echo "Invalid $2: $1; expected true or false." >&2; exit 2 ;;
  esac
}

for setting in "$STEPS" "$BATCH_SIZE" "$GRADIENT_ACCUMULATION_STEPS" "$EVAL_ITERS" \
  "$EVAL_INTERVAL" "$LOG_INTERVAL" "$N_LAYER" "$N_HEAD" "$N_EMBD" "$BLOCK_SIZE" \
  "$DEPTH_ORDER" "$BASE_ROW_ORDER" "$CHECKPOINT_SEGMENT_SIZE"
do
  validate_positive_uint "$setting" "numeric setting"
done
validate_nonnegative_uint "$WARMUP_ITERS" "WARMUP_ITERS"
validate_true_false "$ACTIVATION_CHECKPOINTING" "ACTIVATION_CHECKPOINTING"
validate_true_false "$WANDB_ENABLED" "WANDB_ENABLED"
validate_true_false "$DRY_RUN" "DRY_RUN"

case "$WANDB_MODE" in online|offline|disabled) ;; *) echo "WANDB_MODE must be online, offline, or disabled." >&2; exit 2 ;; esac
case "$RUN_SELECTION" in dense|sheet|both) ;; *) echo "RUN_SELECTION must be dense, sheet, or both." >&2; exit 2 ;; esac
(( WARMUP_ITERS < STEPS )) || { echo "WARMUP_ITERS must be less than STEPS." >&2; exit 2; }
(( N_EMBD % N_HEAD == 0 )) || { echo "N_EMBD must be divisible by N_HEAD." >&2; exit 2; }
(( DEPTH_ORDER <= N_LAYER )) || { echo "DEPTH_ORDER must not exceed N_LAYER." >&2; exit 2; }
(( BASE_ROW_ORDER <= N_EMBD )) || { echo "BASE_ROW_ORDER must not exceed N_EMBD." >&2; exit 2; }

# vvv THOG
# Shared geometry and optimizer/update settings. Keep these identical for the two runs unless intentionally testing OOM behaviour.
# ^^^ THOG
COMMON_ARGS=(
  -q fresh
  -g "$RUN_NAME"
  -n "$STEPS"
  -b "$BATCH_SIZE"
  -A "$GRADIENT_ACCUMULATION_STEPS"
  -u "$EVAL_ITERS"
  -e "$EVAL_INTERVAL"
  -l "$LOG_INTERVAL"
  -w "$WARMUP_ITERS"
  -L "$N_LAYER"
  -H "$N_HEAD"
  -D "$N_EMBD"
  -C "$BLOCK_SIZE"
  -p "$ACTIVATION_CHECKPOINTING"
  -S "$CHECKPOINT_SEGMENT_SIZE"
  -M "$WANDB_MODE"
  -W "$WANDB_ENABLED"
  -x "$DRY_RUN"
)

printf '\nTHOG2 DENSE2/SHEET L144 smoke pair\n'
printf '  run name:                 %s\n' "$RUN_NAME"
printf '  geometry:                 L%s / H%s / D%s / C%s\n' "$N_LAYER" "$N_HEAD" "$N_EMBD" "$BLOCK_SIZE"
printf '  sheet orders:             P%s / Q%s\n' "$DEPTH_ORDER" "$BASE_ROW_ORDER"
printf '  steps:                    %s\n' "$STEPS"
printf '  batch / global accum:     %s / %s\n' "$BATCH_SIZE" "$GRADIENT_ACCUMULATION_STEPS"
printf '  tokens/update:            %s\n' "$((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * BLOCK_SIZE))"
printf '  activation checkpointing: %s, segment %s\n' "$ACTIVATION_CHECKPOINTING" "$CHECKPOINT_SEGMENT_SIZE"
printf '  W&B:                      %s (%s)\n' "$WANDB_ENABLED" "$WANDB_MODE"
printf '  selection:                %s\n\n' "$RUN_SELECTION"

if [[ "$RUN_SELECTION" == dense || "$RUN_SELECTION" == both ]]; then
  echo "=== DENSE2 L144 smoke run ==="
  ./current_scruffy_train_DENSE_OWT.sh "${COMMON_ARGS[@]}"
fi

if [[ "$RUN_SELECTION" == sheet || "$RUN_SELECTION" == both ]]; then
  echo "=== SHEET L144 smoke run ==="
  ./current_scruffy_train_SHEET_OWT.sh "${COMMON_ARGS[@]}" \
    -P "$DEPTH_ORDER" \
    -Q "$BASE_ROW_ORDER"
fi

echo "Done."
