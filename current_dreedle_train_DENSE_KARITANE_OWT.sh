#!/bin/bash
set -euo pipefail

# vvv THOG
# Best-effort DENSE2 control for KARITANE_LONG under the same practical dreedle VRAM envelope.
# Default candidate is intentionally aggressive: L72/H18/D1152/C256 with activation checkpointing and 12288 tokens/update.
# If it OOMs, first try -L 64; if it is still too large, try -D 1024 -H 16.
# ^^^ THOG

cd "$(dirname "$0")"

RUN_MODULE="run_thog2_owt_residual"
HOST_LABEL="dreedle"
DATASET_NAME="openwebtext"
DATA_DIR="${THOG2_OWT_DATA_DIR:-data/openwebtext}"
RUN_NAME="KARITANE_LONG_DENSE_$(date +%y%m%d_%H%M%S)"
STEPS=99999
BATCH_SIZE=12
GRADIENT_ACCUMULATION_STEPS=4
EVAL_ITERS=50
EVAL_INTERVAL=100
LOG_INTERVAL=1
WARMUP_ITERS=20
CHECKPOINT_INTERVAL=500
N_LAYER=72
N_HEAD=18
N_EMBD=1152
BLOCK_SIZE=256
LEARNING_RATE="3.0e-4"
MIN_LR="3.0e-5"
WEIGHT_DECAY="0.1"
BETA1="0.9"
BETA2="0.95"
GRAD_CLIP="1.0"
RESIDUAL_INIT_POLICY="depth_scaled"
RESIDUAL_INIT_DEPTH_SOURCE="true_layer_depth"
ACTIVATION_CHECKPOINTING=true
CHECKPOINT_SEGMENT_SIZE=8
DTYPE="float16"
NONFINITE_UPDATE_POLICY="skip"
MAX_NONFINITE_UPDATE_SKIPS=10
INSTRUMENTATION="wandb"
CURVE_ROOT="curves"
WANDB_MODE="online"
WANDB_ENABLED=true
DRY_RUN=false
CUDA_DEVICE_DEFAULT=0

usage() {
  cat <<EOF
Usage: $0 [options] [-- additional ${RUN_MODULE} arguments]

Runs one aggressive DENSE2 dreedle OpenWebText control for KARITANE_LONG.

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
  -a LEARNING_RATE=${LEARNING_RATE}
  -m MIN_LR=${MIN_LR}
  -p ACTIVATION_CHECKPOINTING=${ACTIVATION_CHECKPOINTING}
  -S CHECKPOINT_SEGMENT_SIZE=${CHECKPOINT_SEGMENT_SIZE}
  -T DTYPE=${DTYPE}                                  float32 | float16 | bfloat16
  -N NONFINITE_UPDATE_POLICY=${NONFINITE_UPDATE_POLICY}  raise | skip
  -K MAX_NONFINITE_UPDATE_SKIPS=${MAX_NONFINITE_UPDATE_SKIPS}
  -I INSTRUMENTATION=${INSTRUMENTATION}              tensorboard | wandb | none
  -V CURVE_ROOT=${CURVE_ROOT}
  -M WANDB_MODE=${WANDB_MODE}                        online | offline | disabled
  -W WANDB_ENABLED=${WANDB_ENABLED}
  -x DRY_RUN=${DRY_RUN}
  -c CUDA_VISIBLE_DEVICES default=${CUDA_DEVICE_DEFAULT}
  -h show this help
EOF
}

while getopts ":t:g:n:b:A:u:e:l:w:k:L:H:D:C:a:m:p:S:T:N:K:I:V:M:W:x:c:h" option; do
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
    a) LEARNING_RATE="$OPTARG" ;;
    m) MIN_LR="$OPTARG" ;;
    p) ACTIVATION_CHECKPOINTING="$OPTARG" ;;
    S) CHECKPOINT_SEGMENT_SIZE="$OPTARG" ;;
    T) DTYPE="$OPTARG" ;;
    N) NONFINITE_UPDATE_POLICY="$OPTARG" ;;
    K) MAX_NONFINITE_UPDATE_SKIPS="$OPTARG" ;;
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
shift $((OPTIND - 1))
if [[ "${1:-}" == "--" ]]; then
  shift
fi
EXTRA_ARGS=("$@")

validate_positive_uint() { [[ "$1" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid $2: $1; expected a positive integer." >&2; exit 2; }; }
validate_nonnegative_uint() { [[ "$1" =~ ^[0-9]+$ ]] || { echo "Invalid $2: $1; expected a non-negative integer." >&2; exit 2; }; }
validate_true_false() { case "$1" in true|false) ;; *) echo "Invalid $2: $1; expected true or false." >&2; exit 2 ;; esac; }

for setting in "$STEPS" "$BATCH_SIZE" "$GRADIENT_ACCUMULATION_STEPS" "$EVAL_ITERS" "$EVAL_INTERVAL" "$LOG_INTERVAL" "$N_LAYER" "$N_HEAD" "$N_EMBD" "$BLOCK_SIZE" "$CHECKPOINT_SEGMENT_SIZE"; do
  validate_positive_uint "$setting" "numeric setting"
done
validate_nonnegative_uint "$WARMUP_ITERS" "WARMUP_ITERS"
validate_nonnegative_uint "$CHECKPOINT_INTERVAL" "CHECKPOINT_INTERVAL"
validate_nonnegative_uint "$MAX_NONFINITE_UPDATE_SKIPS" "MAX_NONFINITE_UPDATE_SKIPS"
validate_true_false "$ACTIVATION_CHECKPOINTING" "ACTIVATION_CHECKPOINTING"
validate_true_false "$WANDB_ENABLED" "WANDB_ENABLED"
validate_true_false "$DRY_RUN" "DRY_RUN"
case "$DTYPE" in float32|float16|bfloat16) ;; *) echo "DTYPE must be float32, float16, or bfloat16." >&2; exit 2 ;; esac
case "$NONFINITE_UPDATE_POLICY" in raise|skip) ;; *) echo "NONFINITE_UPDATE_POLICY must be raise or skip." >&2; exit 2 ;; esac
case "$INSTRUMENTATION" in tensorboard|wandb|none) ;; *) echo "INSTRUMENTATION must be tensorboard, wandb, or none." >&2; exit 2 ;; esac
case "$WANDB_MODE" in online|offline|disabled) ;; *) echo "WANDB_MODE must be online, offline, or disabled." >&2; exit 2 ;; esac
[[ -f "$DATA_DIR/train.bin" && -f "$DATA_DIR/val.bin" ]] || { echo "DATA_DIR must contain train.bin and val.bin: $DATA_DIR" >&2; exit 2; }
(( WARMUP_ITERS < STEPS )) || { echo "WARMUP_ITERS must be less than STEPS." >&2; exit 2; }
(( N_EMBD % N_HEAD == 0 )) || { echo "N_EMBD must be divisible by N_HEAD." >&2; exit 2; }

if [[ -n "${THOG2_PYTHON:-}" ]]; then
  PYTHON_BIN="$THOG2_PYTHON"
elif [[ -x .venv/bin/python ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$CUDA_DEVICE_DEFAULT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export THOG2_INSTRUMENTATION="$INSTRUMENTATION"
export THOG2_CURVE_ROOT="$CURVE_ROOT"

CHECKPOINT_FLAG="--no-activation-checkpointing"
[[ "$ACTIVATION_CHECKPOINTING" == true ]] && CHECKPOINT_FLAG="--activation-checkpointing"
WANDB_FLAG="--no-wandb"
[[ "$WANDB_ENABLED" == true ]] && WANDB_FLAG="--wandb"

RESIDUAL_INIT_DEPTH_VALUE="$N_LAYER"
PARAMETER_ESTIMATE=$((12 * N_LAYER * N_EMBD * N_EMBD + 50304 * N_EMBD))
TOKENS_PER_UPDATE=$((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * BLOCK_SIZE))

TRAIN_ARGS=(
  --model-type dense
  --run-mode fresh
  --host-label "$HOST_LABEL"
  --run-name "$RUN_NAME"
  --dataset "$DATASET_NAME"
  --data-dir "$DATA_DIR"
  --max-iters "$STEPS"
  --batch-size "$BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --eval-iters "$EVAL_ITERS"
  --eval-interval "$EVAL_INTERVAL"
  --log-interval "$LOG_INTERVAL"
  --checkpoint-interval "$CHECKPOINT_INTERVAL"
  --warmup-iters "$WARMUP_ITERS"
  --learning-rate "$LEARNING_RATE"
  --min-lr "$MIN_LR"
  --weight-decay "$WEIGHT_DECAY"
  --beta1 "$BETA1"
  --beta2 "$BETA2"
  --grad-clip "$GRAD_CLIP"
  --n-layer "$N_LAYER"
  --n-head "$N_HEAD"
  --n-embd "$N_EMBD"
  --block-size "$BLOCK_SIZE"
  --residual-init-policy "$RESIDUAL_INIT_POLICY"
  --residual-init-depth-source "$RESIDUAL_INIT_DEPTH_SOURCE"
  --residual-init-depth-value "$RESIDUAL_INIT_DEPTH_VALUE"
  "$CHECKPOINT_FLAG"
  --checkpoint-segment-size "$CHECKPOINT_SEGMENT_SIZE"
  --dtype "$DTYPE"
  --nonfinite-update-policy "$NONFINITE_UPDATE_POLICY"
  --max-nonfinite-update-skips "$MAX_NONFINITE_UPDATE_SKIPS"
  "$WANDB_FLAG"
  --wandb-mode "$WANDB_MODE"
  "${EXTRA_ARGS[@]}"
)

LOG_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESOLVED_JSON="$($PYTHON_BIN -m "$RUN_MODULE" "${TRAIN_ARGS[@]}" --log-timestamp "$LOG_TIMESTAMP" --print-resolved-json)"
ARTIFACT_NAME="$(printf '%s' "$RESOLVED_JSON" | $PYTHON_BIN -c 'import json,sys; print(json.load(sys.stdin)["artifact_name"])')"
LOG_PATH="$(printf '%s' "$RESOLVED_JSON" | $PYTHON_BIN -c 'import json,sys; print(json.load(sys.stdin)["paths"]["log_path"])')"

cat <<EOF
THOG2 dreedle KARITANE DENSE2 OpenWebText control
  artifact:                $ARTIFACT_NAME
  run name:                $RUN_NAME
  data dir:                $DATA_DIR
  geometry:                L$N_LAYER / H$N_HEAD / D$N_EMBD / C$BLOCK_SIZE
  rough dense params:      $PARAMETER_ESTIMATE
  steps:                   $STEPS, intended for manual stop
  batch / global accum:    $BATCH_SIZE / $GRADIENT_ACCUMULATION_STEPS
  tokens/update:           $TOKENS_PER_UPDATE
  learning rate:           $LEARNING_RATE -> $MIN_LR
  eval:                    every $EVAL_INTERVAL updates, $EVAL_ITERS batches
  checkpoint interval:     $CHECKPOINT_INTERVAL
  activation checkpoint:   $ACTIVATION_CHECKPOINTING, segment $CHECKPOINT_SEGMENT_SIZE
  dtype:                   $DTYPE
  nonfinite policy:        $NONFINITE_UPDATE_POLICY, max skips $MAX_NONFINITE_UPDATE_SKIPS
  CUDA_VISIBLE_DEVICES:    $CUDA_VISIBLE_DEVICES
  cuda alloc conf:         $PYTORCH_CUDA_ALLOC_CONF
  instrumentation:         $THOG2_INSTRUMENTATION
  curve root:              $THOG2_CURVE_ROOT
  W&B:                     $WANDB_ENABLED ($WANDB_MODE)
  log:                     $LOG_PATH
EOF

COMMAND=("$PYTHON_BIN" -m "$RUN_MODULE" "${TRAIN_ARGS[@]}" --log-timestamp "$LOG_TIMESTAMP")

if [[ "$DRY_RUN" == true ]]; then
  "$PYTHON_BIN" -m "$RUN_MODULE" "${TRAIN_ARGS[@]}" --log-timestamp "$LOG_TIMESTAMP" --dry-run
  printf 'DRY RUN:'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$(dirname "$LOG_PATH")"
"${COMMAND[@]}" 2>&1 | tee "$LOG_PATH"
