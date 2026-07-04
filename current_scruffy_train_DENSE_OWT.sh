#!/bin/bash

set -euo pipefail


#  DENSE2: CANONICAL SCRUFFY OPENWEBTEXT RUN
#
#  Script called:
#    run_thog2_owt_residual.py
#
#  Purpose:
#    - train or resume the THOG2 dense control through the permanent shared runner
#    - use the DENSE2_ prefix so these runs remain distinct from THOG's DENSE_
#    - use the same public parameter names, W&B metrics, paths, and lifecycle as SHEET
#    - use THOG-compatible global gradient-accumulation semantics
#    - use segmented activation checkpointing by default, matching the THOG setup
#    - preserve a deterministic artifact identity across interruption and resume
#
#  Default geometry and execution:
#    - GPT-2 Small control: L12 / H12 / D768 / context 1024
#    - local mini-batch 3
#    - global gradient accumulation 160
#    - one GPU on scruffy
#    - activation checkpointing disabled by default for canonical dense GPT-2 execution
#    - bfloat16 autocast
#
#  Artifact layout:
#    - checkpoints/<DENSE2_artifact>/ckpt.pt
#    - logs/<YYMMDD_HHMM_DENSE2_artifact>/train.log
#    - logs/<YYMMDD_HHMM_DENSE2_artifact>/result.json
#    - W&B run name equals the canonical artifact name
#
#  Common options:
#    -q fresh|resume
#    -g RUN_NAME
#    -n MAX_ITERS
#    -b BATCH_SIZE
#    -d DATASET
#    -u EVAL_ITERS
#    -e EVAL_INTERVAL
#    -l LOG_INTERVAL
#    -w WARMUP_ITERS
#    -k CHECKPOINT_INTERVAL
#    -A GLOBAL_GRADIENT_ACCUMULATION_STEPS
#    -G NUM_GPUS
#    -L N_LAYER
#    -H N_HEAD
#    -D N_EMBD
#    -C BLOCK_SIZE
#    -r RESIDUAL_INIT_POLICY
#    -z RESIDUAL_INIT_DEPTH_SOURCE
#    -Z RESIDUAL_INIT_DEPTH_VALUE
#    -p ACTIVATION_CHECKPOINTING
#    -S CHECKPOINT_SEGMENT_SIZE
#    -M WANDB_MODE
#    -W WANDB_ENABLED
#    -x DRY_RUN
#

cd "$(dirname "$0")"

RUN_MODULE="run_thog2_owt_residual"
HOST_LABEL="scruffy"
RUN_MODE="fresh"
RUN_NAME="GPT2_SMALL_VS_SHEET144_P32_A"
STEPS=250
BATCH_SIZE=3
DATASET_NAME="openwebtext"
DATA_DIR="data/openwebtext"
EVAL_ITERS=5
EVAL_INTERVAL=20
LOG_INTERVAL=1
WARMUP_ITERS=10
CHECKPOINT_INTERVAL=20
GRADIENT_ACCUMULATION_STEPS=160
NUM_GPUS=1
N_LAYER=12
N_HEAD=12
N_EMBD=768
BLOCK_SIZE=1024
RESIDUAL_INIT_POLICY="depth_scaled"
RESIDUAL_INIT_DEPTH_SOURCE="logical_depth"
RESIDUAL_INIT_DEPTH_VALUE=12
ACTIVATION_CHECKPOINTING=false
CHECKPOINT_SEGMENT_SIZE=12
WANDB_MODE="online"
WANDB_ENABLED=true
DRY_RUN=false

usage() {
  cat <<EOF
Usage: $0 [options] [-- additional ${RUN_MODULE} arguments]

  -q RUN_MODE=${RUN_MODE}                     fresh | resume
  -g RUN_NAME=${RUN_NAME}
  -n STEPS=${STEPS}
  -b BATCH_SIZE=${BATCH_SIZE}
  -d DATASET_NAME=${DATASET_NAME}
  -u EVAL_ITERS=${EVAL_ITERS}
  -e EVAL_INTERVAL=${EVAL_INTERVAL}
  -l LOG_INTERVAL=${LOG_INTERVAL}
  -w WARMUP_ITERS=${WARMUP_ITERS}
  -k CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL}  0 disables periodic saves
  -A GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}
  -G NUM_GPUS=${NUM_GPUS}
  -L N_LAYER=${N_LAYER}
  -H N_HEAD=${N_HEAD}
  -D N_EMBD=${N_EMBD}
  -C BLOCK_SIZE=${BLOCK_SIZE}
  -r RESIDUAL_INIT_POLICY=${RESIDUAL_INIT_POLICY}              depth_scaled | unscaled
  -z RESIDUAL_INIT_DEPTH_SOURCE=${RESIDUAL_INIT_DEPTH_SOURCE}  logical_depth | constant
  -Z RESIDUAL_INIT_DEPTH_VALUE=${RESIDUAL_INIT_DEPTH_VALUE}    used only when depth source is constant
  -p ACTIVATION_CHECKPOINTING=${ACTIVATION_CHECKPOINTING}
  -S CHECKPOINT_SEGMENT_SIZE=${CHECKPOINT_SEGMENT_SIZE}
  -M WANDB_MODE=${WANDB_MODE}                 online | offline | disabled
  -W WANDB_ENABLED=${WANDB_ENABLED}
  -x DRY_RUN=${DRY_RUN}
  -h show this help
EOF
}

while getopts ":q:g:n:b:d:u:e:l:w:k:A:G:L:H:D:C:r:z:Z:p:S:M:W:x:h" option; do
  case "$option" in
    q) RUN_MODE="$OPTARG" ;;
    g) RUN_NAME="$OPTARG" ;;
    n) STEPS="$OPTARG" ;;
    b) BATCH_SIZE="$OPTARG" ;;
    d) DATASET_NAME="$OPTARG"; DATA_DIR="data/$OPTARG" ;;
    u) EVAL_ITERS="$OPTARG" ;;
    e) EVAL_INTERVAL="$OPTARG" ;;
    l) LOG_INTERVAL="$OPTARG" ;;
    w) WARMUP_ITERS="$OPTARG" ;;
    k) CHECKPOINT_INTERVAL="$OPTARG" ;;
    A) GRADIENT_ACCUMULATION_STEPS="$OPTARG" ;;
    G) NUM_GPUS="$OPTARG" ;;
    L) N_LAYER="$OPTARG" ;;
    H) N_HEAD="$OPTARG" ;;
    D) N_EMBD="$OPTARG" ;;
    C) BLOCK_SIZE="$OPTARG" ;;
    r) RESIDUAL_INIT_POLICY="$OPTARG" ;;
    z) RESIDUAL_INIT_DEPTH_SOURCE="$OPTARG" ;;
    Z) RESIDUAL_INIT_DEPTH_VALUE="$OPTARG" ;;
    p) ACTIVATION_CHECKPOINTING="$OPTARG" ;;
    S) CHECKPOINT_SEGMENT_SIZE="$OPTARG" ;;
    M) WANDB_MODE="$OPTARG" ;;
    W) WANDB_ENABLED="$OPTARG" ;;
    x) DRY_RUN="$OPTARG" ;;
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

case "$RUN_MODE" in fresh|resume) ;; *) echo "RUN_MODE must be fresh or resume." >&2; exit 2 ;; esac
case "$WANDB_MODE" in online|offline|disabled) ;; *) echo "WANDB_MODE must be online, offline, or disabled." >&2; exit 2 ;; esac
case "$RESIDUAL_INIT_POLICY" in depth_scaled|unscaled) ;; *) echo "RESIDUAL_INIT_POLICY must be depth_scaled or unscaled." >&2; exit 2 ;; esac
case "$RESIDUAL_INIT_DEPTH_SOURCE" in logical_depth|constant) ;; basis_depth) echo "basis_depth residual init is only defined for SHEET." >&2; exit 2 ;; *) echo "RESIDUAL_INIT_DEPTH_SOURCE must be logical_depth or constant for DENSE." >&2; exit 2 ;; esac
for setting in "$STEPS" "$BATCH_SIZE" "$EVAL_ITERS" "$EVAL_INTERVAL" \
  "$LOG_INTERVAL" "$GRADIENT_ACCUMULATION_STEPS" "$NUM_GPUS" \
  "$N_LAYER" "$N_HEAD" "$N_EMBD" "$BLOCK_SIZE" "$CHECKPOINT_SEGMENT_SIZE" \
  "$RESIDUAL_INIT_DEPTH_VALUE"
do
  validate_positive_uint "$setting" "numeric setting"
done
validate_nonnegative_uint "$WARMUP_ITERS" "WARMUP_ITERS"
validate_nonnegative_uint "$CHECKPOINT_INTERVAL" "CHECKPOINT_INTERVAL"
validate_true_false "$ACTIVATION_CHECKPOINTING" "ACTIVATION_CHECKPOINTING"
validate_true_false "$WANDB_ENABLED" "WANDB_ENABLED"
validate_true_false "$DRY_RUN" "DRY_RUN"
(( WARMUP_ITERS < STEPS )) || { echo "WARMUP_ITERS must be less than STEPS." >&2; exit 2; }
(( N_EMBD % N_HEAD == 0 )) || { echo "N_EMBD must be divisible by N_HEAD." >&2; exit 2; }
(( GRADIENT_ACCUMULATION_STEPS % NUM_GPUS == 0 )) || {
  echo "Global gradient accumulation must be divisible by NUM_GPUS." >&2
  exit 2
}

if [[ -n "${THOG2_PYTHON:-}" ]]; then
  PYTHON_BIN="$THOG2_PYTHON"
elif [[ -x .venv/bin/python ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python"
fi

CHECKPOINT_FLAG="--no-activation-checkpointing"
[[ "$ACTIVATION_CHECKPOINTING" == true ]] && CHECKPOINT_FLAG="--activation-checkpointing"
WANDB_FLAG="--no-wandb"
[[ "$WANDB_ENABLED" == true ]] && WANDB_FLAG="--wandb"

TRAIN_ARGS=(
  --model-type dense
  --run-mode "$RUN_MODE"
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
  --n-layer "$N_LAYER"
  --n-head "$N_HEAD"
  --n-embd "$N_EMBD"
  --block-size "$BLOCK_SIZE"
  --residual-init-policy "$RESIDUAL_INIT_POLICY"
  --residual-init-depth-source "$RESIDUAL_INIT_DEPTH_SOURCE"
  --residual-init-depth-value "$RESIDUAL_INIT_DEPTH_VALUE"
  "$CHECKPOINT_FLAG"
  --checkpoint-segment-size "$CHECKPOINT_SEGMENT_SIZE"
  "$WANDB_FLAG"
  --wandb-mode "$WANDB_MODE"
  "${EXTRA_ARGS[@]}"
)

LOG_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESOLVED_JSON="$("$PYTHON_BIN" -m "$RUN_MODULE" "${TRAIN_ARGS[@]}" \
  --log-timestamp "$LOG_TIMESTAMP" --print-resolved-json)"
ARTIFACT_NAME="$(printf '%s' "$RESOLVED_JSON" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["artifact_name"])')"
LOG_PATH="$(printf '%s' "$RESOLVED_JSON" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["paths"]["log_path"])')"

if (( NUM_GPUS == 1 )); then
  COMMAND=("$PYTHON_BIN" -m "$RUN_MODULE" "${TRAIN_ARGS[@]}" --log-timestamp "$LOG_TIMESTAMP")
else
  COMMAND=("$PYTHON_BIN" -m torch.distributed.run --standalone "--nproc-per-node=$NUM_GPUS" \
    -m "$RUN_MODULE" "${TRAIN_ARGS[@]}" --log-timestamp "$LOG_TIMESTAMP")
fi

cat <<EOF
THOG2 DENSE OpenWebText experiment
  artifact:                $ARTIFACT_NAME
  run mode:               $RUN_MODE
  geometry:               L$N_LAYER / H$N_HEAD / D$N_EMBD / C$BLOCK_SIZE
  residual init:          $RESIDUAL_INIT_POLICY / $RESIDUAL_INIT_DEPTH_SOURCE / $RESIDUAL_INIT_DEPTH_VALUE
  GPUs:                   $NUM_GPUS
  mini-batch/GPU:         $BATCH_SIZE
  global accumulation:    $GRADIENT_ACCUMULATION_STEPS
  tokens/update:          $((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * BLOCK_SIZE))
  activation checkpoint:  $ACTIVATION_CHECKPOINTING
  checkpoint segment:     $CHECKPOINT_SEGMENT_SIZE
  checkpoint interval:    $CHECKPOINT_INTERVAL
  W&B:                    $WANDB_ENABLED ($WANDB_MODE)
  log:                    $LOG_PATH
EOF

if [[ "$DRY_RUN" == true ]]; then
  "$PYTHON_BIN" -m "$RUN_MODULE" "${TRAIN_ARGS[@]}" --log-timestamp "$LOG_TIMESTAMP" --dry-run
  printf 'DRY RUN:'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$(dirname "$LOG_PATH")"
"${COMMAND[@]}" 2>&1 | tee "$LOG_PATH"
