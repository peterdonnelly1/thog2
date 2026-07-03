#!/bin/bash

set -uo pipefail


#  GPT-2 SMALL DENSE2 vs SHEET-144: SCRUFFY OPENWEBTEXT COMPARISON
#
#  Purpose:
#    - compare a conventional GPT-2 Small architecture against a much deeper
#      SHEET model that uses the same width, head count, context, optimiser,
#      data trace, token budget, and training controls
#    - use SHEET's compact persistent state to buy logical depth rather than
#      merely comparing two models at the same layer count
#    - run the two experiments sequentially on one GPU and preserve separate
#      canonical checkpoints, logs, result files, and W&B runs
#
#  Fixed comparison geometry:
#    DENSE2: L12  / H12 / D768 / context 1024
#    SHEET:  L144 / H12 / D768 / context 1024 / P16 / Q64
#
#  Default execution:
#    - 250 optimiser updates per model
#    - local mini-batch 3
#    - global gradient accumulation 160
#    - 491,520 training tokens per optimiser update
#    - activation checkpointing enabled, segment size 12
#    - evaluation at update 0 and every 25 updates, 5 batches per split
#    - periodic checkpoint every 25 updates
#    - W&B online
#
#  The underlying canonical runners remain:
#    current_scruffy_train_DENSE_OWT.sh
#    current_scruffy_train_SHEET_OWT.sh
#

cd "$(dirname "$0")"

RUN_MODE="fresh"
RUN_NAME="GPT2_SMALL_VS_SHEET144_A"
STEPS=250
BATCH_SIZE=3
GRADIENT_ACCUMULATION_STEPS=160
EVAL_ITERS=5
EVAL_INTERVAL=25
LOG_INTERVAL=1
WARMUP_ITERS=10
CHECKPOINT_INTERVAL=25
DEPTH_ORDER=16
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

  -q RUN_MODE=${RUN_MODE}                     fresh | resume
  -g RUN_NAME=${RUN_NAME}
  -n STEPS=${STEPS}
  -b BATCH_SIZE=${BATCH_SIZE}
  -A GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}
  -u EVAL_ITERS=${EVAL_ITERS}
  -e EVAL_INTERVAL=${EVAL_INTERVAL}
  -l LOG_INTERVAL=${LOG_INTERVAL}
  -w WARMUP_ITERS=${WARMUP_ITERS}
  -k CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL}    0 disables periodic saves
  -P DEPTH_ORDER=${DEPTH_ORDER}                    SHEET only
  -Q BASE_ROW_ORDER=${BASE_ROW_ORDER}              SHEET only
  -p ACTIVATION_CHECKPOINTING=${ACTIVATION_CHECKPOINTING}
  -S CHECKPOINT_SEGMENT_SIZE=${CHECKPOINT_SEGMENT_SIZE}
  -M WANDB_MODE=${WANDB_MODE}                 online | offline | disabled
  -W WANDB_ENABLED=${WANDB_ENABLED}
  -R RUN_SELECTION=${RUN_SELECTION}            both | dense | sheet
  -x DRY_RUN=${DRY_RUN}
  -h show this help
EOF
}

while getopts ":q:g:n:b:A:u:e:l:w:k:P:Q:p:S:M:W:R:x:h" option; do
  case "$option" in
    q) RUN_MODE="$OPTARG" ;;
    g) RUN_NAME="$OPTARG" ;;
    n) STEPS="$OPTARG" ;;
    b) BATCH_SIZE="$OPTARG" ;;
    A) GRADIENT_ACCUMULATION_STEPS="$OPTARG" ;;
    u) EVAL_ITERS="$OPTARG" ;;
    e) EVAL_INTERVAL="$OPTARG" ;;
    l) LOG_INTERVAL="$OPTARG" ;;
    w) WARMUP_ITERS="$OPTARG" ;;
    k) CHECKPOINT_INTERVAL="$OPTARG" ;;
    P) DEPTH_ORDER="$OPTARG" ;;
    Q) BASE_ROW_ORDER="$OPTARG" ;;
    p) ACTIVATION_CHECKPOINTING="$OPTARG" ;;
    S) CHECKPOINT_SEGMENT_SIZE="$OPTARG" ;;
    M) WANDB_MODE="$OPTARG" ;;
    W) WANDB_ENABLED="$OPTARG" ;;
    R) RUN_SELECTION="$OPTARG" ;;
    x) DRY_RUN="$OPTARG" ;;
    h) usage; exit 0 ;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 2 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage >&2; exit 2 ;;
  esac
done

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

case "$RUN_MODE" in
  fresh|resume) ;;
  *) echo "RUN_MODE must be fresh or resume." >&2; exit 2 ;;
esac
case "$WANDB_MODE" in
  online|offline|disabled) ;;
  *) echo "WANDB_MODE must be online, offline, or disabled." >&2; exit 2 ;;
esac
case "$RUN_SELECTION" in
  both|dense|sheet) ;;
  *) echo "RUN_SELECTION must be both, dense, or sheet." >&2; exit 2 ;;
esac

for value in "$STEPS" "$BATCH_SIZE" "$GRADIENT_ACCUMULATION_STEPS" \
  "$EVAL_ITERS" "$EVAL_INTERVAL" "$LOG_INTERVAL" "$DEPTH_ORDER" \
  "$BASE_ROW_ORDER" "$CHECKPOINT_SEGMENT_SIZE"
do
  validate_positive_uint "$value" "numeric setting"
done
validate_nonnegative_uint "$WARMUP_ITERS" "WARMUP_ITERS"
validate_nonnegative_uint "$CHECKPOINT_INTERVAL" "CHECKPOINT_INTERVAL"
validate_true_false "$ACTIVATION_CHECKPOINTING" "ACTIVATION_CHECKPOINTING"
validate_true_false "$WANDB_ENABLED" "WANDB_ENABLED"
validate_true_false "$DRY_RUN" "DRY_RUN"
(( WARMUP_ITERS < STEPS )) || {
  echo "WARMUP_ITERS must be less than STEPS." >&2
  exit 2
}

if [[ ! -f data/openwebtext/train.bin || ! -f data/openwebtext/val.bin ]]; then
  echo "ERROR: data/openwebtext must provide train.bin and val.bin." >&2
  exit 1
fi

if [[ -n "${THOG2_PYTHON:-}" ]]; then
  PYTHON_BIN="$THOG2_PYTHON"
elif [[ -x .venv/bin/python ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python"
fi

if [[ "$WANDB_ENABLED" == true && "$WANDB_MODE" != disabled ]]; then
  "$PYTHON_BIN" - <<'PY'
import wandb

assert getattr(wandb, "__file__", None), "wandb did not resolve to an installed package"
assert hasattr(wandb, "Settings"), "installed wandb package lacks wandb.Settings"
wandb.Settings()
print(f"W&B preflight ready: {wandb.__version__} ({wandb.__file__})")
PY
  WANDB_PREFLIGHT_EXIT=$?
  if (( WANDB_PREFLIGHT_EXIT != 0 )); then
    echo "ERROR: W&B preflight failed." >&2
    exit "$WANDB_PREFLIGHT_EXIT"
  fi
fi

mkdir -p logs
STAMP="$(date +%Y%m%d_%H%M%S)"
SAFE_RUN_NAME="${RUN_NAME//[^[:alnum:]._-]/_}"
ORCHESTRATOR_LOG="logs/${SAFE_RUN_NAME}_GPT2_SMALL_vs_SHEET144_${STAMP}.orchestrator.log"
exec > >(tee -a "$ORCHESTRATOR_LOG") 2>&1

TOKENS_PER_UPDATE=$((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * 1024))
TOTAL_TOKENS=$((TOKENS_PER_UPDATE * STEPS))

cat <<EOF
===== GPT-2 SMALL vs SHEET-144 COMPARISON =====
started:                 $(date)
run name/group:          $RUN_NAME
run mode:                $RUN_MODE
selection:               $RUN_SELECTION
updates/model:           $STEPS
mini-batch/GPU:          $BATCH_SIZE
global accumulation:     $GRADIENT_ACCUMULATION_STEPS
tokens/update:           $TOKENS_PER_UPDATE
total tokens/model:      $TOTAL_TOKENS
DENSE2 geometry:         L12/H12/D768/C1024
SHEET geometry:          L144/H12/D768/C1024/P${DEPTH_ORDER}/Q${BASE_ROW_ORDER}
activation checkpoint:   $ACTIVATION_CHECKPOINTING
checkpoint segment:      $CHECKPOINT_SEGMENT_SIZE
checkpoint interval:     $CHECKPOINT_INTERVAL
eval interval/batches:   $EVAL_INTERVAL / $EVAL_ITERS
warmup updates:          $WARMUP_ITERS
W&B:                     $WANDB_ENABLED ($WANDB_MODE)
Python:                  $PYTHON_BIN
orchestrator log:        $ORCHESTRATOR_LOG
================================================
EOF

COMMON_ARGS=(
  -q "$RUN_MODE"
  -g "$RUN_NAME"
  -n "$STEPS"
  -b "$BATCH_SIZE"
  -A "$GRADIENT_ACCUMULATION_STEPS"
  -G 1
  -H 12
  -D 768
  -C 1024
  -p "$ACTIVATION_CHECKPOINTING"
  -S "$CHECKPOINT_SEGMENT_SIZE"
  -w "$WARMUP_ITERS"
  -e "$EVAL_INTERVAL"
  -u "$EVAL_ITERS"
  -l "$LOG_INTERVAL"
  -W "$WANDB_ENABLED"
  -M "$WANDB_MODE"
  -x "$DRY_RUN"
)

EXTRA_ARGS=(-- --checkpoint-interval "$CHECKPOINT_INTERVAL")

DENSE_EXIT=0
SHEET_EXIT=0
DENSE_STATUS="SKIPPED"
SHEET_STATUS="SKIPPED"

if [[ "$RUN_SELECTION" == both || "$RUN_SELECTION" == dense ]]; then
  echo
  echo "===== DENSE2 GPT-2 SMALL START $(date) ====="
  ./current_scruffy_train_DENSE_OWT.sh \
    "${COMMON_ARGS[@]}" \
    -L 12 \
    "${EXTRA_ARGS[@]}"
  DENSE_EXIT=$?
  DENSE_STATUS="EXIT=$DENSE_EXIT"
  echo "===== DENSE2 GPT-2 SMALL $DENSE_STATUS $(date) ====="
fi

if [[ "$RUN_SELECTION" == both || "$RUN_SELECTION" == sheet ]]; then
  echo
  echo "===== SHEET-144 START $(date) ====="
  ./current_scruffy_train_SHEET_OWT.sh \
    "${COMMON_ARGS[@]}" \
    -L 144 \
    -P "$DEPTH_ORDER" \
    -Q "$BASE_ROW_ORDER" \
    "${EXTRA_ARGS[@]}"
  SHEET_EXIT=$?
  SHEET_STATUS="EXIT=$SHEET_EXIT"
  echo "===== SHEET-144 $SHEET_STATUS $(date) ====="
fi

echo
echo "===== COMPARISON SUMMARY ====="
echo "DENSE2 GPT-2 Small: $DENSE_STATUS"
echo "SHEET-144:          $SHEET_STATUS"
echo "finished:           $(date)"
echo "orchestrator log:   $ORCHESTRATOR_LOG"

if (( DENSE_EXIT != 0 || SHEET_EXIT != 0 )); then
  exit 1
fi
