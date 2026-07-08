#!/bin/bash
set -euo pipefail

# vvv THOG
# Current dreedle OpenWebText training wrapper for the consolidated Stage 8-capable runner.
# Defaults are deliberately non-tiny: SHEET L144 H12 D768 C1024, batch 3, global accumulation 160.
# Dreedle runtime defaults: float16, sdpa. Dense baseline is available with -O dense.
# Logging is explicit: -I tensorboard|wandb|none. TensorBoard writes under THOG2_CURVE_ROOT, default curves/.
# ^^^ THOG

cd "$(dirname "$0")"

RUN_MODULE="run_thog2_owt"
HOST_LABEL="dreedle"
MODEL_TYPE="sheet"
RUN_MODE="fresh"
RUN_NAME=""
EXPERIMENT_PREFIX="${THOG2_EXPERIMENT_PREFIX:-NELSON}"
DATASET_NAME="openwebtext"
DATA_DIR="${THOG2_OWT_DATA_DIR:-$HOME/git/thog/data/openwebtext}"
CHECKPOINT_ROOT="checkpoints"
LOG_ROOT="logs"
RESULT_ROOT="results"
WANDB_ROOT="wandb"
GEOMETRY_PRESET="curve"
BASIS_FAMILY="chebyshev"
BASIS_VERSION="auto"
ATTENTION_GEOMETRY=""
MLP_GEOMETRY=""
STEPS=250
BATCH_SIZE=3
GRADIENT_ACCUMULATION_STEPS=160
NUM_GPUS=1
EVAL_ITERS=5
EVAL_INTERVAL=20
LOG_INTERVAL=1
WARMUP_ITERS=10
CHECKPOINT_INTERVAL=20
N_LAYER=144
N_HEAD=12
N_EMBD=768
BLOCK_SIZE=1024
DEPTH_ORDER=32
BASE_ROW_ORDER=64
MLP_CHANNEL_ORDER=256
RESIDUAL_INIT_POLICY="depth_scaled"
RESIDUAL_INIT_DEPTH_SOURCE="dof_implied_depth"
RESIDUAL_INIT_DEPTH_VALUE=12
ACTIVATION_CHECKPOINTING=true
CHECKPOINT_SEGMENT_SIZE=12
DTYPE="float16"
ATTENTION_BACKEND="sdpa"
INSTRUMENTATION="tensorboard"
WANDB_MODE="online"
WANDB_ENABLED=true
DRY_RUN=false
N_LAYER_EXPLICIT=false
N_HEAD_EXPLICIT=false
N_EMBD_EXPLICIT=false

usage() {
  cat <<EOF
Usage: $0 [options] [-- extra ${RUN_MODULE} args]

Model/run:
  -O MODEL_TYPE=${MODEL_TYPE}                    dense | sheet
  -q RUN_MODE=${RUN_MODE}                        fresh | resume
  -g RUN_NAME=${RUN_NAME:-auto}
  -n STEPS=${STEPS}
  -b BATCH_SIZE=${BATCH_SIZE}
  -A GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}
  -G NUM_GPUS=${NUM_GPUS}

Schedule/logging:
  -u EVAL_ITERS=${EVAL_ITERS}
  -e EVAL_INTERVAL=${EVAL_INTERVAL}
  -l LOG_INTERVAL=${LOG_INTERVAL}
  -w WARMUP_ITERS=${WARMUP_ITERS}
  -k CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL}     0 disables periodic saves
  -I INSTRUMENTATION=${INSTRUMENTATION}             tensorboard | wandb | none
  -M WANDB_MODE=${WANDB_MODE}                       online | offline | disabled
  -W WANDB_ENABLED=${WANDB_ENABLED}                 telemetry master switch

Compact options:
  -p GEOMETRY_PRESET=${GEOMETRY_PRESET}             legacy_sheet_col | curve | head_aware_block | mlp_block | block
  -B BASIS_FAMILY=${BASIS_FAMILY}                   chebyshev | dct
  -v BASIS_VERSION=${BASIS_VERSION}
  -a ATTENTION_GEOMETRY=${ATTENTION_GEOMETRY:-preset default}
  -m MLP_GEOMETRY=${MLP_GEOMETRY:-preset default}

Shape/runtime:
  -L N_LAYER=${N_LAYER}
  -H N_HEAD=${N_HEAD}
  -D N_EMBD=${N_EMBD}
  -C BLOCK_SIZE=${BLOCK_SIZE}
  -P DEPTH_ORDER=${DEPTH_ORDER}
  -Q BASE_ROW_ORDER=${BASE_ROW_ORDER}
  -Y MLP_CHANNEL_ORDER=${MLP_CHANNEL_ORDER}
  -S CHECKPOINT_SEGMENT_SIZE=${CHECKPOINT_SEGMENT_SIZE}
  -T DTYPE=${DTYPE}                                 float32 | float16 | bfloat16
  -K ATTENTION_BACKEND=${ATTENTION_BACKEND}         auto | flash2 | sdpa | math

Residual init:
  -r RESIDUAL_INIT_POLICY=${RESIDUAL_INIT_POLICY}                 depth_scaled | unscaled
  -z RESIDUAL_INIT_DEPTH_SOURCE=${RESIDUAL_INIT_DEPTH_SOURCE}     true_layer_depth | dof_implied_depth | user_forced_depth
  -Z RESIDUAL_INIT_DEPTH_VALUE=${RESIDUAL_INIT_DEPTH_VALUE}

Paths:
  -d DATASET_NAME=${DATASET_NAME}
  -t DATA_DIR=${DATA_DIR}
  -o CHECKPOINT_ROOT=${CHECKPOINT_ROOT}
  -j LOG_ROOT=${LOG_ROOT}
  -R RESULT_ROOT=${RESULT_ROOT}
  -x DRY_RUN=${DRY_RUN}
  -h show this help
EOF
}

while getopts ":O:q:g:n:b:A:G:u:e:l:w:k:I:M:W:p:B:v:a:m:L:H:D:C:P:Q:Y:S:T:K:r:z:Z:d:t:o:j:R:x:h" option; do
  case "$option" in
    O) MODEL_TYPE="$OPTARG" ;; q) RUN_MODE="$OPTARG" ;; g) RUN_NAME="$OPTARG" ;;
    n) STEPS="$OPTARG" ;; b) BATCH_SIZE="$OPTARG" ;; A) GRADIENT_ACCUMULATION_STEPS="$OPTARG" ;; G) NUM_GPUS="$OPTARG" ;;
    u) EVAL_ITERS="$OPTARG" ;; e) EVAL_INTERVAL="$OPTARG" ;; l) LOG_INTERVAL="$OPTARG" ;; w) WARMUP_ITERS="$OPTARG" ;; k) CHECKPOINT_INTERVAL="$OPTARG" ;;
    I) INSTRUMENTATION="$OPTARG" ;; M) WANDB_MODE="$OPTARG" ;; W) WANDB_ENABLED="$OPTARG" ;;
    p) GEOMETRY_PRESET="$OPTARG" ;; B) BASIS_FAMILY="$OPTARG" ;; v) BASIS_VERSION="$OPTARG" ;; a) ATTENTION_GEOMETRY="$OPTARG" ;; m) MLP_GEOMETRY="$OPTARG" ;;
    L) N_LAYER="$OPTARG"; N_LAYER_EXPLICIT=true ;; H) N_HEAD="$OPTARG"; N_HEAD_EXPLICIT=true ;; D) N_EMBD="$OPTARG"; N_EMBD_EXPLICIT=true ;;
    C) BLOCK_SIZE="$OPTARG" ;; P) DEPTH_ORDER="$OPTARG" ;; Q) BASE_ROW_ORDER="$OPTARG" ;; Y) MLP_CHANNEL_ORDER="$OPTARG" ;; S) CHECKPOINT_SEGMENT_SIZE="$OPTARG" ;;
    T) DTYPE="$OPTARG" ;; K) ATTENTION_BACKEND="$OPTARG" ;; r) RESIDUAL_INIT_POLICY="$OPTARG" ;; z) RESIDUAL_INIT_DEPTH_SOURCE="$OPTARG" ;; Z) RESIDUAL_INIT_DEPTH_VALUE="$OPTARG" ;;
    d) DATASET_NAME="$OPTARG"; DATA_DIR="data/$OPTARG" ;; t) DATA_DIR="$OPTARG" ;; o) CHECKPOINT_ROOT="$OPTARG" ;; j) LOG_ROOT="$OPTARG" ;; R) RESULT_ROOT="$OPTARG" ;; x) DRY_RUN="$OPTARG" ;;
    h) usage; exit 0 ;; :) echo "Option -$OPTARG requires an argument." >&2; exit 2 ;; \?) echo "Unknown option: -$OPTARG" >&2; usage >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
if [[ "${1:-}" == "--" ]]; then shift; fi
EXTRA_ARGS=("$@")

validate_positive_uint() { [[ "$1" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid $2: $1; expected a positive integer." >&2; exit 2; }; }
validate_nonnegative_uint() { [[ "$1" =~ ^[0-9]+$ ]] || { echo "Invalid $2: $1; expected a non-negative integer." >&2; exit 2; }; }
validate_true_false() { case "$1" in true|false) ;; *) echo "Invalid $2: $1; expected true or false." >&2; exit 2 ;; esac; }

case "$MODEL_TYPE" in dense|sheet) ;; *) echo "MODEL_TYPE must be dense or sheet." >&2; exit 2 ;; esac
case "$RUN_MODE" in fresh|resume) ;; *) echo "RUN_MODE must be fresh or resume." >&2; exit 2 ;; esac
case "$GEOMETRY_PRESET" in legacy_sheet_col|curve|head_aware_block|mlp_block|block) ;; *) echo "Bad GEOMETRY_PRESET: $GEOMETRY_PRESET" >&2; exit 2 ;; esac
case "$BASIS_FAMILY" in chebyshev|dct) ;; *) echo "BASIS_FAMILY must be chebyshev or dct." >&2; exit 2 ;; esac
case "$ATTENTION_BACKEND" in auto|flash2|sdpa|math) ;; *) echo "Bad ATTENTION_BACKEND: $ATTENTION_BACKEND" >&2; exit 2 ;; esac
case "$INSTRUMENTATION" in tensorboard|wandb|none) ;; *) echo "INSTRUMENTATION must be tensorboard, wandb, or none." >&2; exit 2 ;; esac
case "$WANDB_MODE" in online|offline|disabled) ;; *) echo "WANDB_MODE must be online, offline, or disabled." >&2; exit 2 ;; esac
case "$RESIDUAL_INIT_POLICY" in depth_scaled|unscaled) ;; *) echo "RESIDUAL_INIT_POLICY must be depth_scaled or unscaled." >&2; exit 2 ;; esac
case "$RESIDUAL_INIT_DEPTH_SOURCE" in true_layer_depth|dof_implied_depth|user_forced_depth) ;; *) echo "Bad RESIDUAL_INIT_DEPTH_SOURCE: $RESIDUAL_INIT_DEPTH_SOURCE" >&2; exit 2 ;; esac
for setting in "$STEPS" "$BATCH_SIZE" "$GRADIENT_ACCUMULATION_STEPS" "$NUM_GPUS" "$EVAL_ITERS" "$EVAL_INTERVAL" "$LOG_INTERVAL" "$N_LAYER" "$N_HEAD" "$N_EMBD" "$BLOCK_SIZE" "$DEPTH_ORDER" "$BASE_ROW_ORDER" "$MLP_CHANNEL_ORDER" "$CHECKPOINT_SEGMENT_SIZE" "$RESIDUAL_INIT_DEPTH_VALUE"; do validate_positive_uint "$setting" "numeric setting"; done
validate_nonnegative_uint "$WARMUP_ITERS" "WARMUP_ITERS"
validate_nonnegative_uint "$CHECKPOINT_INTERVAL" "CHECKPOINT_INTERVAL"
validate_true_false "$WANDB_ENABLED" "WANDB_ENABLED"
validate_true_false "$ACTIVATION_CHECKPOINTING" "ACTIVATION_CHECKPOINTING"
validate_true_false "$DRY_RUN" "DRY_RUN"

if [[ "$MODEL_TYPE" == dense ]]; then
  [[ "$N_LAYER_EXPLICIT" == false ]] && N_LAYER=12
  [[ "$N_HEAD_EXPLICIT" == false ]] && N_HEAD=12
  [[ "$N_EMBD_EXPLICIT" == false ]] && N_EMBD=768
  [[ "$RESIDUAL_INIT_DEPTH_SOURCE" == dof_implied_depth ]] && RESIDUAL_INIT_DEPTH_SOURCE="true_layer_depth"
fi

(( WARMUP_ITERS < STEPS )) || { echo "WARMUP_ITERS must be less than STEPS." >&2; exit 2; }
(( N_EMBD % N_HEAD == 0 )) || { echo "N_EMBD must be divisible by N_HEAD." >&2; exit 2; }
if [[ "$MODEL_TYPE" == sheet ]]; then
  (( DEPTH_ORDER <= N_LAYER )) || { echo "DEPTH_ORDER must not exceed N_LAYER." >&2; exit 2; }
  (( BASE_ROW_ORDER <= N_EMBD )) || { echo "BASE_ROW_ORDER must not exceed N_EMBD." >&2; exit 2; }
  (( MLP_CHANNEL_ORDER <= 4 * N_EMBD )) || { echo "MLP_CHANNEL_ORDER must not exceed 4*N_EMBD." >&2; exit 2; }
fi
(( GRADIENT_ACCUMULATION_STEPS % NUM_GPUS == 0 )) || { echo "GRADIENT_ACCUMULATION_STEPS must be divisible by NUM_GPUS." >&2; exit 2; }

if [[ -n "${THOG2_PYTHON:-}" ]]; then PYTHON_BIN="$THOG2_PYTHON"; elif [[ -x .venv/bin/python ]]; then PYTHON_BIN=".venv/bin/python"; else PYTHON_BIN="python"; fi
BASIS_TAG="CHEBY"; [[ "$BASIS_FAMILY" == dct ]] && BASIS_TAG="DCT"
PRESET_TAG="${GEOMETRY_PRESET^^}"; [[ "$GEOMETRY_PRESET" == legacy_sheet_col ]] && PRESET_TAG="SHEET_COL"
[[ "$MODEL_TYPE" == dense ]] && RUN_TAG="DENSE" || RUN_TAG="${BASIS_TAG}_${PRESET_TAG}"
[[ -z "$RUN_NAME" ]] && RUN_NAME="${RUN_TAG}_OWT"

OPTIONAL_ARGS=()
[[ -n "$ATTENTION_GEOMETRY" ]] && OPTIONAL_ARGS+=(--attention-geometry "$ATTENTION_GEOMETRY")
[[ -n "$MLP_GEOMETRY" ]] && OPTIONAL_ARGS+=(--mlp-geometry "$MLP_GEOMETRY")
CHECKPOINT_FLAG="--no-activation-checkpointing"; [[ "$ACTIVATION_CHECKPOINTING" == true ]] && CHECKPOINT_FLAG="--activation-checkpointing"
WANDB_FLAG="--no-wandb"; [[ "$WANDB_ENABLED" == true ]] && WANDB_FLAG="--wandb"

export THOG2_INSTRUMENTATION="$INSTRUMENTATION"
export THOG2_CURVE_ROOT="${THOG2_CURVE_ROOT:-curves}"
export THOG2_MLP_CHANNEL_ORDER="$MLP_CHANNEL_ORDER"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRAIN_ARGS=(
  --model-type "$MODEL_TYPE" --run-mode "$RUN_MODE" --host-label "$HOST_LABEL" --run-name "$RUN_NAME"
  --dataset "$DATASET_NAME" --data-dir "$DATA_DIR" --checkpoint-root "$CHECKPOINT_ROOT" --log-root "$LOG_ROOT" --result-root "$RESULT_ROOT" --wandb-root "$WANDB_ROOT"
  --max-iters "$STEPS" --batch-size "$BATCH_SIZE" --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --eval-iters "$EVAL_ITERS" --eval-interval "$EVAL_INTERVAL" --log-interval "$LOG_INTERVAL" --checkpoint-interval "$CHECKPOINT_INTERVAL" --warmup-iters "$WARMUP_ITERS"
  --n-layer "$N_LAYER" --n-head "$N_HEAD" --n-embd "$N_EMBD" --block-size "$BLOCK_SIZE" --depth-order "$DEPTH_ORDER" --base-row-order "$BASE_ROW_ORDER" --mlp-channel-order "$MLP_CHANNEL_ORDER"
  --geometry-preset "$GEOMETRY_PRESET" --basis-family "$BASIS_FAMILY" --basis-version "$BASIS_VERSION" --attention-backend "$ATTENTION_BACKEND" --experiment-prefix "$EXPERIMENT_PREFIX" --dtype "$DTYPE"
  --residual-init-policy "$RESIDUAL_INIT_POLICY" --residual-init-depth-source "$RESIDUAL_INIT_DEPTH_SOURCE" --residual-init-depth-value "$RESIDUAL_INIT_DEPTH_VALUE"
  "$CHECKPOINT_FLAG" --checkpoint-segment-size "$CHECKPOINT_SEGMENT_SIZE" "$WANDB_FLAG" --wandb-mode "$WANDB_MODE" "${OPTIONAL_ARGS[@]}" "${EXTRA_ARGS[@]}"
)

LOG_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESOLVED_JSON="$($PYTHON_BIN -m "$RUN_MODULE" "${TRAIN_ARGS[@]}" --log-timestamp "$LOG_TIMESTAMP" --print-resolved-json)"
ARTIFACT_NAME="$(printf '%s' "$RESOLVED_JSON" | $PYTHON_BIN -c 'import json,sys; print(json.load(sys.stdin)["artifact_name"])')"
LOG_PATH="$(printf '%s' "$RESOLVED_JSON" | $PYTHON_BIN -c 'import json,sys; print(json.load(sys.stdin)["paths"]["log_path"])')"
COMMAND=("$PYTHON_BIN" -m "$RUN_MODULE" "${TRAIN_ARGS[@]}" --log-timestamp "$LOG_TIMESTAMP")
if (( NUM_GPUS > 1 )); then COMMAND=("$PYTHON_BIN" -m torch.distributed.run --standalone "--nproc-per-node=$NUM_GPUS" -m "$RUN_MODULE" "${TRAIN_ARGS[@]}" --log-timestamp "$LOG_TIMESTAMP"); fi

cat <<EOF
dreedle OWT train
  artifact:           $ARTIFACT_NAME
  experiment:         $EXPERIMENT_PREFIX
  model/preset/basis: $MODEL_TYPE / $GEOMETRY_PRESET / $BASIS_FAMILY
  backend/dtype:      $ATTENTION_BACKEND / $DTYPE
  instrumentation:    $INSTRUMENTATION  (tensorboard root: $THOG2_CURVE_ROOT)
  schedule:           steps=$STEPS eval_every=$EVAL_INTERVAL eval_iters=$EVAL_ITERS log_every=$LOG_INTERVAL ckpt_every=$CHECKPOINT_INTERVAL warmup=$WARMUP_ITERS
  shape:              L$N_LAYER H$N_HEAD D$N_EMBD C$BLOCK_SIZE P$DEPTH_ORDER Q$BASE_ROW_ORDER Y$MLP_CHANNEL_ORDER
  batch/accum/gpus:   $BATCH_SIZE / $GRADIENT_ACCUMULATION_STEPS / $NUM_GPUS
  log:                $LOG_PATH
EOF

if [[ "$DRY_RUN" == true ]]; then
  "$PYTHON_BIN" -m "$RUN_MODULE" "${TRAIN_ARGS[@]}" --log-timestamp "$LOG_TIMESTAMP" --dry-run
  printf 'DRY RUN:'; printf ' %q' "${COMMAND[@]}"; printf '\n'
  exit 0
fi

mkdir -p "$(dirname "$LOG_PATH")"
"${COMMAND[@]}" 2>&1 | tee "$LOG_PATH"
