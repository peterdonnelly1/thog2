#!/bin/bash
set -euo pipefail

# vvv THOG
# Current dreedle OpenWebText training wrapper for the PICTON compact-geometry contract.
# Dreedle runtime defaults: float16, sdpa. Dense baseline is available as -p dense.
# ^^^ THOG

cd "$(dirname "$0")"

RUN_MODULE="run_thog2_owt"
HOST_LABEL="dreedle"
RUN_MODE="fresh"
RUN_NAME=""
# EXPERIMENT_PREFIX="${THOG2_EXPERIMENT_PREFIX:-NELSON}"                                                                                                  # <<< THOG removed redundant environment-controlled experiment naming
EXPERIMENT_PREFIX="NO_PREFIX"                                                                                                                            # <<< THOG -g now supplies the sole human run-name prefix
DATASET_NAME="openwebtext"
DATA_DIR="${THOG2_OWT_DATA_DIR:-$HOME/git/thog/data/openwebtext}"
CHECKPOINT_ROOT="checkpoints"
LOG_ROOT="logs"
RESULT_ROOT="results"
WANDB_ROOT="wandb"
GEOMETRY_PRESET="depth"
BASIS_FAMILY="chebyshev"
BASIS_VERSION="auto"
ATTENTION_GEOMETRY=""
MLP_GEOMETRY=""
STEPS=250
STEPS_EXPLICIT=false
RESUME_FROM=""
FORK_LR_MODE=""
FORK_LEARNING_RATE=""
FORK_MIN_LR=""
FORK_REWARM_ITERS=""
NUM_GPUS_EXPLICIT=false
EVAL_ITERS_EXPLICIT=false
EVAL_INTERVAL_EXPLICIT=false
LOG_INTERVAL_EXPLICIT=false
CHECKPOINT_INTERVAL_EXPLICIT=false
INSTRUMENTATION_EXPLICIT=false
CHECKPOINT_SEGMENT_SIZE_EXPLICIT=false
FAST_DISCARD_EXPLICIT=false
DTYPE_EXPLICIT=false
ATTENTION_BACKEND_EXPLICIT=false
# vvv THOG explicit material values supplied during resume are forwarded as equality assertions
RUN_NAME_EXPLICIT=false
BATCH_SIZE_EXPLICIT=false
LEARNING_RATE_EXPLICIT=false
MIN_LR_EXPLICIT=false
ACCUMULATION_EXPLICIT=false
WARMUP_EXPLICIT=false
GEOMETRY_PRESET_EXPLICIT=false
BASIS_FAMILY_EXPLICIT=false
BASIS_VERSION_EXPLICIT=false
ATTENTION_GEOMETRY_EXPLICIT=false
MLP_GEOMETRY_EXPLICIT=false
BLOCK_SIZE_EXPLICIT=false
O_DEPTH_EXPLICIT=false
O_ATTN_D_MODEL_EXPLICIT=false
O_ATTN_QKV_PER_CHANNEL_EXPLICIT=false
O_ATTN_OUT_PER_CHANNEL_EXPLICIT=false
O_MLP_D_MODEL_EXPLICIT=false
O_MLP_HIDDEN_EXPLICIT=false
RESIDUAL_INIT_POLICY_EXPLICIT=false
RESIDUAL_INIT_DEPTH_SOURCE_EXPLICIT=false
RESIDUAL_INIT_DEPTH_VALUE_EXPLICIT=false
DATASET_EXPLICIT=false
DATA_DIR_EXPLICIT=false
LOG_ROOT_EXPLICIT=false
RESULT_ROOT_EXPLICIT=false
# ^^^ THOG
BATCH_SIZE=3
LEARNING_RATE_CODES="60"                                                                                                                               # <<< THOG LR grid codes; 70 means 7.0e-04
MIN_LR_CODE="06"                                                                                                                                         # <<< THOG minimum LR code; 1..100; 06 means 6.0e-05 and 100 means 1.0e-03
GRADIENT_ACCUMULATION_STEPS=160
NUM_GPUS=1
EVAL_ITERS=5
EVAL_INTERVAL=100
LOG_INTERVAL=1
WARMUP_ITERS=10
CHECKPOINT_INTERVAL=1000
N_LAYER=144
N_HEAD=12
N_EMBD=768
BLOCK_SIZE=1024
O_DEPTH=32
O_ATTN_D_MODEL=64
O_ATTN_QKV_PER_CHANNEL=6
O_ATTN_OUT_PER_CHANNEL=6
O_MLP_D_MODEL=64
O_MLP_HIDDEN=256
RESIDUAL_INIT_POLICY="depth_scaled"
RESIDUAL_INIT_DEPTH_SOURCE="dof_implied_depth"
RESIDUAL_INIT_DEPTH_VALUE=12
ACTIVATION_CHECKPOINTING=true
CHECKPOINT_SEGMENT_SIZE=12
FAST_DISCARD="${THOG2_FAST_DISCARD:-true}"
BYPASS_SEMANTIC_QKV_ADAPTER="${THOG2_BYPASS_SEMANTIC_QKV_ADAPTER:-true}"                                                                                  # <<< THOG default-on selectable semantic-QKV adapter bypass
DIRECT_FACTORISED_MLP="${THOG2_DIRECT_FACTORISED_MLP:-true}"                                                                                              # <<< THOG renamed default-on exact factorised MLP application
VECTORISE_PER_HEAD_MATERIALISATION="${THOG2_VECTORISE_PER_HEAD_MATERIALISATION:-true}"                                                                    # <<< THOG default-on selectable per-head batching
DTYPE="float16"
ATTENTION_BACKEND="sdpa"
INSTRUMENTATION="tensorboard"
DEPTH_CURVE_PLOTS="${THOG2_DEPTH_CURVE_PLOTS:-eval}"
DEPTH_CURVE_SAMPLE_ELEMENTS="${THOG2_DEPTH_CURVE_SAMPLE_ELEMENTS:-16384}"
DEPTH_CURVE_RENDERER="${THOG2_DEPTH_CURVE_RENDERER:-plotly}"
DEPTH_CURVE_LOCAL_HTML="${THOG2_DEPTH_CURVE_LOCAL_HTML:-true}"
DEPTH_CURVE_HTTP_PORT="${THOG2_DEPTH_CURVE_HTTP_PORT:-8787}"
DRY_RUN=false
N_LAYER_EXPLICIT=false
N_HEAD_EXPLICIT=false
N_EMBD_EXPLICIT=false

usage() {
  cat <<EOF_USAGE
Usage: $0 [options] [-- extra ${RUN_MODULE} args]

Model/run:
  -p PRESET=${GEOMETRY_PRESET}                       dense | legacy_sheet_col | depth | head_aware_block | mlp_block | full_block
                                                   single value, comma list, or quoted space list
  -q RUN_MODE=${RUN_MODE}                        fresh | resume | fork
  -g RUN_NAME=${RUN_NAME:-auto}
  -n STEPS=${STEPS}                              total optimizer steps for the run (not additional steps)
  --resume-from SELECTOR                          required for resume and fork; path, artifact_name, or YYMMDD-HHMM
  -0 / --fork-learning-rate VALUE                fork restart peak LR
  -1 / --fork-min-lr VALUE                       fork restart minimum LR
  -2 / --fork-rewarm-iters COUNT                 fork linear rewarm length
  -3 / --fork-lr-mode MODE                       continue | restart_cosine
  -b BATCH_SIZE=${BATCH_SIZE}                         single integer, comma list, or quoted space list
  -c LR_CODES=${LEARNING_RATE_CODES}                    1..1000; 70 means 7.0e-04 and 1000 means 1.0e-02; comma or quoted space list
  -f MIN_LR_CODE=${MIN_LR_CODE}                         1..100; 06 means 6.0e-05 and 100 means 1.0e-03
  -A GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}
  -G NUM_GPUS=${NUM_GPUS}

Schedule/logging:
  -u EVAL_ITERS=${EVAL_ITERS}
  -e EVAL_INTERVAL=${EVAL_INTERVAL}
  -l LOG_INTERVAL=${LOG_INTERVAL}
  -w WARMUP_ITERS=${WARMUP_ITERS}
  -k CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL}     0 disables periodic saves
  -I INSTRUMENTATION=${INSTRUMENTATION}             tensorboard | wandb | wandb_offline | none
  -F DEPTH_CURVE_PLOTS=${DEPTH_CURVE_PLOTS}         none | final | eval
  -N DEPTH_CURVE_SAMPLE_ELEMENTS=${DEPTH_CURVE_SAMPLE_ELEMENTS}
  -U DEPTH_CURVE_RENDERER=${DEPTH_CURVE_RENDERER}   matplotlib | plotly | both
  -V DEPTH_CURVE_LOCAL_HTML=${DEPTH_CURVE_LOCAL_HTML}  true | false

Compact geometry:
  -B BASIS_FAMILY=${BASIS_FAMILY}                   chebyshev | dct
  -v BASIS_VERSION=${BASIS_VERSION}
  -a ATTENTION_GEOMETRY=${ATTENTION_GEOMETRY:-preset default}
  -m MLP_GEOMETRY=${MLP_GEOMETRY:-preset default}

Shape/runtime:
  -L N_LAYER=${N_LAYER}
  -H N_HEAD=${N_HEAD}
  -D N_EMBD=${N_EMBD}
  -C BLOCK_SIZE=${BLOCK_SIZE}
  -P O_DEPTH=${O_DEPTH}                             single integer, comma list, or quoted space list; ignored by dense
  -Q O_ATTN_D_MODEL=${O_ATTN_D_MODEL}
  -J O_ATTN_QKV_PER_CHANNEL=${O_ATTN_QKV_PER_CHANNEL}
  -O O_ATTN_OUT_PER_CHANNEL=${O_ATTN_OUT_PER_CHANNEL}
  -X O_MLP_D_MODEL=${O_MLP_D_MODEL}
  -Y O_MLP_HIDDEN=${O_MLP_HIDDEN}
  -S CHECKPOINT_SEGMENT_SIZE=${CHECKPOINT_SEGMENT_SIZE}
  -E FAST_DISCARD=${FAST_DISCARD}                   true | false
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
EOF_USAGE
}

# vvv THOG normalize lifecycle long options before the established short-option parser
NORMALIZED_ARGS=()
while (( $# > 0 )); do
  case "$1" in
    --resume-from) [[ $# -ge 2 ]] || { echo "Option --resume-from requires an argument." >&2; exit 2; }; RESUME_FROM="$2"; shift 2 ;;
    --resume-from=*) RESUME_FROM="${1#*=}"; shift ;;
    --fork-lr-mode) [[ $# -ge 2 ]] || { echo "Option --fork-lr-mode requires an argument." >&2; exit 2; }; FORK_LR_MODE="$2"; shift 2 ;;
    --fork-lr-mode=*) FORK_LR_MODE="${1#*=}"; shift ;;
    --fork-learning-rate) [[ $# -ge 2 ]] || { echo "Option --fork-learning-rate requires an argument." >&2; exit 2; }; FORK_LEARNING_RATE="$2"; shift 2 ;;
    --fork-learning-rate=*) FORK_LEARNING_RATE="${1#*=}"; shift ;;
    --fork-min-lr) [[ $# -ge 2 ]] || { echo "Option --fork-min-lr requires an argument." >&2; exit 2; }; FORK_MIN_LR="$2"; shift 2 ;;
    --fork-min-lr=*) FORK_MIN_LR="${1#*=}"; shift ;;
    --fork-rewarm-iters) [[ $# -ge 2 ]] || { echo "Option --fork-rewarm-iters requires an argument." >&2; exit 2; }; FORK_REWARM_ITERS="$2"; shift 2 ;;
    --fork-rewarm-iters=*) FORK_REWARM_ITERS="${1#*=}"; shift ;;
    *) NORMALIZED_ARGS+=("$1"); shift ;;
  esac
done
set -- "${NORMALIZED_ARGS[@]}"
# ^^^ THOG

while getopts ":0:1:2:3:q:g:n:b:c:f:A:G:u:e:l:w:k:I:F:N:U:V:p:B:v:a:m:L:H:D:C:P:Q:J:O:X:Y:S:E:T:K:r:z:Z:d:t:o:j:R:x:h" option; do
  case "$option" in
    0) FORK_LEARNING_RATE="$OPTARG" ;; 1) FORK_MIN_LR="$OPTARG" ;; 2) FORK_REWARM_ITERS="$OPTARG" ;; 3) FORK_LR_MODE="$OPTARG" ;;
    q) RUN_MODE="$OPTARG" ;; g) RUN_NAME="$OPTARG"; RUN_NAME_EXPLICIT=true ;;
    n) STEPS="$OPTARG"; STEPS_EXPLICIT=true ;; b) BATCH_SIZE="$OPTARG"; BATCH_SIZE_EXPLICIT=true ;; c) LEARNING_RATE_CODES="$OPTARG"; LEARNING_RATE_EXPLICIT=true ;; f) MIN_LR_CODE="$OPTARG"; MIN_LR_EXPLICIT=true ;; A) GRADIENT_ACCUMULATION_STEPS="$OPTARG"; ACCUMULATION_EXPLICIT=true ;; G) NUM_GPUS="$OPTARG"; NUM_GPUS_EXPLICIT=true ;;
    u) EVAL_ITERS="$OPTARG"; EVAL_ITERS_EXPLICIT=true ;; e) EVAL_INTERVAL="$OPTARG"; EVAL_INTERVAL_EXPLICIT=true ;; l) LOG_INTERVAL="$OPTARG"; LOG_INTERVAL_EXPLICIT=true ;; w) WARMUP_ITERS="$OPTARG"; WARMUP_EXPLICIT=true ;; k) CHECKPOINT_INTERVAL="$OPTARG"; CHECKPOINT_INTERVAL_EXPLICIT=true ;;
    I) INSTRUMENTATION="$OPTARG"; INSTRUMENTATION_EXPLICIT=true ;; F) DEPTH_CURVE_PLOTS="$OPTARG" ;; N) DEPTH_CURVE_SAMPLE_ELEMENTS="$OPTARG" ;; U) DEPTH_CURVE_RENDERER="$OPTARG" ;; V) DEPTH_CURVE_LOCAL_HTML="$OPTARG" ;;
    p) GEOMETRY_PRESET="$OPTARG"; GEOMETRY_PRESET_EXPLICIT=true ;; B) BASIS_FAMILY="$OPTARG"; BASIS_FAMILY_EXPLICIT=true ;; v) BASIS_VERSION="$OPTARG"; BASIS_VERSION_EXPLICIT=true ;; a) ATTENTION_GEOMETRY="$OPTARG"; ATTENTION_GEOMETRY_EXPLICIT=true ;; m) MLP_GEOMETRY="$OPTARG"; MLP_GEOMETRY_EXPLICIT=true ;;
    L) N_LAYER="$OPTARG"; N_LAYER_EXPLICIT=true ;; H) N_HEAD="$OPTARG"; N_HEAD_EXPLICIT=true ;; D) N_EMBD="$OPTARG"; N_EMBD_EXPLICIT=true ;;
    C) BLOCK_SIZE="$OPTARG"; BLOCK_SIZE_EXPLICIT=true ;; P) O_DEPTH="$OPTARG"; O_DEPTH_EXPLICIT=true ;; Q) O_ATTN_D_MODEL="$OPTARG"; O_ATTN_D_MODEL_EXPLICIT=true ;; J) O_ATTN_QKV_PER_CHANNEL="$OPTARG"; O_ATTN_QKV_PER_CHANNEL_EXPLICIT=true ;; O) O_ATTN_OUT_PER_CHANNEL="$OPTARG"; O_ATTN_OUT_PER_CHANNEL_EXPLICIT=true ;; X) O_MLP_D_MODEL="$OPTARG"; O_MLP_D_MODEL_EXPLICIT=true ;; Y) O_MLP_HIDDEN="$OPTARG"; O_MLP_HIDDEN_EXPLICIT=true ;; S) CHECKPOINT_SEGMENT_SIZE="$OPTARG"; CHECKPOINT_SEGMENT_SIZE_EXPLICIT=true ;;
    E) FAST_DISCARD="$OPTARG"; FAST_DISCARD_EXPLICIT=true ;; T) DTYPE="$OPTARG"; DTYPE_EXPLICIT=true ;; K) ATTENTION_BACKEND="$OPTARG"; ATTENTION_BACKEND_EXPLICIT=true ;; r) RESIDUAL_INIT_POLICY="$OPTARG"; RESIDUAL_INIT_POLICY_EXPLICIT=true ;; z) RESIDUAL_INIT_DEPTH_SOURCE="$OPTARG"; RESIDUAL_INIT_DEPTH_SOURCE_EXPLICIT=true ;; Z) RESIDUAL_INIT_DEPTH_VALUE="$OPTARG"; RESIDUAL_INIT_DEPTH_VALUE_EXPLICIT=true ;;
    d) DATASET_NAME="$OPTARG"; DATA_DIR="data/$OPTARG"; DATASET_EXPLICIT=true; DATA_DIR_EXPLICIT=true ;; t) DATA_DIR="$OPTARG"; DATA_DIR_EXPLICIT=true ;; o) CHECKPOINT_ROOT="$OPTARG" ;; j) LOG_ROOT="$OPTARG"; LOG_ROOT_EXPLICIT=true ;; R) RESULT_ROOT="$OPTARG"; RESULT_ROOT_EXPLICIT=true ;; x) DRY_RUN="$OPTARG" ;;
    h) usage; exit 0 ;; :) echo "Option -$OPTARG requires an argument." >&2; exit 2 ;; \?) echo "Unknown option: -$OPTARG" >&2; usage >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
if [[ "${1:-}" == "--" ]]; then shift; fi
EXTRA_ARGS=("$@")
EXPERIMENT_PREFIX="${RUN_NAME:-NO_PREFIX}"                                                                                                               # <<< THOG make -g the sole experiment-prefix source

validate_positive_uint() { [[ "$1" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid $2: $1; expected a positive integer." >&2; exit 2; }; }
validate_nonnegative_uint() { [[ "$1" =~ ^[0-9]+$ ]] || { echo "Invalid $2: $1; expected a non-negative integer." >&2; exit 2; }; }
validate_true_false() { case "$1" in true|false) ;; *) echo "Invalid $2: $1; expected true or false." >&2; exit 2 ;; esac; }
# vvv THOG resume assertions cannot be grid-valued
require_single_value() {
  local normalized="${1//,/ }" values=() value
  for value in $normalized; do values+=("$value"); done
  (( ${#values[@]} == 1 )) || { echo "$2 must be a single value in resume or fork mode." >&2; exit 2; }
}
validate_resume_lr_code() {
  local value="$1" label="$2" maximum="$3"
  [[ "$value" =~ ^[0-9]{1,4}$ ]] && (( 10#$value >= 1 && 10#$value <= maximum )) || {
    echo "Invalid $label: $value; expected 1..$maximum." >&2
    exit 2
  }
}
# ^^^ THOG

# vvv THOG checkpoint-driven resume/fork path: inherit material configuration and pass only explicit overrides
run_resume_or_fork() {
  case "$RUN_MODE" in resume|fork) ;; *) return 1 ;; esac
  if [[ "$STEPS_EXPLICIT" != true ]]; then
    echo "THOG2 INFO: -n was omitted during ${RUN_MODE}!" >&2
    echo "You must specify a value for n, which must be greater than the number of steps already completed" >&2
    exit 2
  fi
  validate_positive_uint "$STEPS" "STEPS"
  [[ -n "$RESUME_FROM" ]] || { echo "${RUN_MODE} requires --resume-from SELECTOR." >&2; exit 2; }
  case "$RUN_MODE" in
    resume)
      [[ -z "$FORK_LR_MODE$FORK_LEARNING_RATE$FORK_MIN_LR$FORK_REWARM_ITERS" ]] || { echo "resume rejects fork-only LR options." >&2; exit 2; }
      ;;
    fork)
      [[ "$FORK_LR_MODE" == restart_cosine ]] || { echo "Initial fork support requires -3 restart_cosine." >&2; exit 2; }
      [[ -n "$FORK_LEARNING_RATE" && -n "$FORK_MIN_LR" && -n "$FORK_REWARM_ITERS" ]] || { echo "restart_cosine requires -0, -1, and -2." >&2; exit 2; }
      validate_nonnegative_uint "$FORK_REWARM_ITERS" "FORK_REWARM_ITERS"
      ;;
  esac

  if [[ -n "${THOG2_PYTHON:-}" ]]; then PYTHON_BIN="$THOG2_PYTHON"; elif [[ -x .venv/bin/python ]]; then PYTHON_BIN=".venv/bin/python"; else PYTHON_BIN="python"; fi
  local stored_world_size
  stored_world_size="$($PYTHON_BIN - "$RESUME_FROM" "$CHECKPOINT_ROOT" <<'PY_RESOLVE'
import sys
from sheet.checkpoint_resolver import resolve_checkpoint
from sheet.checkpoints import load_payload
from sheet.run_lifecycle import lifecycle_from_checkpoint
resolved = resolve_checkpoint(sys.argv[1], sys.argv[2])
payload = load_payload(resolved.checkpoint_path)
print(int(lifecycle_from_checkpoint(payload)["world_size"]))
PY_RESOLVE
)"
  if [[ "$NUM_GPUS_EXPLICIT" == true && "$NUM_GPUS" != "$stored_world_size" ]]; then
    echo "${RUN_MODE} world size mismatch: checkpoint=${stored_world_size}, requested=${NUM_GPUS}." >&2
    exit 2
  fi
  NUM_GPUS="$stored_world_size"

  case "$INSTRUMENTATION" in
    tensorboard) INSTRUMENTATION_BACKEND="tensorboard"; WANDB_MODE="disabled" ;;
    wandb) INSTRUMENTATION_BACKEND="wandb"; WANDB_MODE="online" ;;
    wandb_offline) INSTRUMENTATION_BACKEND="wandb"; WANDB_MODE="offline" ;;
    none) INSTRUMENTATION_BACKEND="none"; WANDB_MODE="disabled" ;;
    *) echo "INSTRUMENTATION must be tensorboard, wandb, wandb_offline, or none." >&2; exit 2 ;;
  esac

  export THOG2_DEPTH_CURVE_PLOTS="$DEPTH_CURVE_PLOTS"
  export THOG2_DEPTH_CURVE_SAMPLE_ELEMENTS="$DEPTH_CURVE_SAMPLE_ELEMENTS"
  export THOG2_DEPTH_CURVE_RENDERER="$DEPTH_CURVE_RENDERER"
  export THOG2_DEPTH_CURVE_LOCAL_HTML="$DEPTH_CURVE_LOCAL_HTML"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

  local log_timestamp resolved_json artifact_name log_path append_log start_time_friendly run_status
  local -a lifecycle_args command
  log_timestamp="$(date +%Y%m%d_%H%M%S)"
  start_time_friendly="$(date '+%H:%M  %d-%m-%y')"
  lifecycle_args=(--run-mode "$RUN_MODE" --resume-from "$RESUME_FROM" --max-iters "$STEPS" --checkpoint-root "$CHECKPOINT_ROOT" --log-timestamp "$log_timestamp")
  [[ "$RUN_MODE" == fork ]] && lifecycle_args+=(--fork-lr-mode "$FORK_LR_MODE" --fork-learning-rate "$FORK_LEARNING_RATE" --fork-min-lr "$FORK_MIN_LR" --fork-rewarm-iters "$FORK_REWARM_ITERS")
  # vvv THOG forward explicit material values as checkpoint equality assertions
  if [[ "$RUN_NAME_EXPLICIT" == true ]]; then lifecycle_args+=(--run-name "$RUN_NAME" --experiment-prefix "$EXPERIMENT_PREFIX"); fi
  if [[ "$GEOMETRY_PRESET_EXPLICIT" == true ]]; then
    require_single_value "$GEOMETRY_PRESET" "PRESET"
    if [[ "$GEOMETRY_PRESET" == dense ]]; then lifecycle_args+=(--model-type dense); else lifecycle_args+=(--model-type sheet --geometry-preset "$GEOMETRY_PRESET"); fi
  fi
  if [[ "$BATCH_SIZE_EXPLICIT" == true ]]; then require_single_value "$BATCH_SIZE" "BATCH_SIZE"; validate_positive_uint "$BATCH_SIZE" "BATCH_SIZE"; lifecycle_args+=(--batch-size "$BATCH_SIZE"); fi
  if [[ "$LEARNING_RATE_EXPLICIT" == true ]]; then require_single_value "$LEARNING_RATE_CODES" "LEARNING_RATE_CODES"; validate_resume_lr_code "$LEARNING_RATE_CODES" "LEARNING_RATE_CODES" 1000; lifecycle_args+=(--learning-rate "$((10#$LEARNING_RATE_CODES))e-5"); fi
  if [[ "$MIN_LR_EXPLICIT" == true ]]; then validate_resume_lr_code "$MIN_LR_CODE" "MIN_LR_CODE" 100; lifecycle_args+=(--min-lr "$((10#$MIN_LR_CODE))e-5"); fi
  [[ "$ACCUMULATION_EXPLICIT" == true ]] && lifecycle_args+=(--gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS")
  [[ "$WARMUP_EXPLICIT" == true ]] && lifecycle_args+=(--warmup-iters "$WARMUP_ITERS")
  [[ "$BASIS_FAMILY_EXPLICIT" == true ]] && lifecycle_args+=(--basis-family "$BASIS_FAMILY")
  [[ "$BASIS_VERSION_EXPLICIT" == true ]] && lifecycle_args+=(--basis-version "$BASIS_VERSION")
  [[ "$ATTENTION_GEOMETRY_EXPLICIT" == true ]] && lifecycle_args+=(--attention-geometry "$ATTENTION_GEOMETRY")
  [[ "$MLP_GEOMETRY_EXPLICIT" == true ]] && lifecycle_args+=(--mlp-geometry "$MLP_GEOMETRY")
  [[ "$N_LAYER_EXPLICIT" == true ]] && lifecycle_args+=(--n-layer "$N_LAYER")
  [[ "$N_HEAD_EXPLICIT" == true ]] && lifecycle_args+=(--n-head "$N_HEAD")
  [[ "$N_EMBD_EXPLICIT" == true ]] && lifecycle_args+=(--n-embd "$N_EMBD")
  [[ "$BLOCK_SIZE_EXPLICIT" == true ]] && lifecycle_args+=(--block-size "$BLOCK_SIZE")
  if [[ "$O_DEPTH_EXPLICIT" == true ]]; then require_single_value "$O_DEPTH" "O_DEPTH"; lifecycle_args+=(--o-depth "$O_DEPTH"); fi
  [[ "$O_ATTN_D_MODEL_EXPLICIT" == true ]] && lifecycle_args+=(--o-attn-d-model "$O_ATTN_D_MODEL")
  [[ "$O_ATTN_QKV_PER_CHANNEL_EXPLICIT" == true ]] && lifecycle_args+=(--o-attn-qkv-per-channel "$O_ATTN_QKV_PER_CHANNEL")
  [[ "$O_ATTN_OUT_PER_CHANNEL_EXPLICIT" == true ]] && lifecycle_args+=(--o-attn-out-per-channel "$O_ATTN_OUT_PER_CHANNEL")
  [[ "$O_MLP_D_MODEL_EXPLICIT" == true ]] && lifecycle_args+=(--o-mlp-d-model "$O_MLP_D_MODEL")
  [[ "$O_MLP_HIDDEN_EXPLICIT" == true ]] && lifecycle_args+=(--o-mlp-hidden "$O_MLP_HIDDEN")
  [[ "$RESIDUAL_INIT_POLICY_EXPLICIT" == true ]] && lifecycle_args+=(--residual-init-policy "$RESIDUAL_INIT_POLICY")
  [[ "$RESIDUAL_INIT_DEPTH_SOURCE_EXPLICIT" == true ]] && lifecycle_args+=(--residual-init-depth-source "$RESIDUAL_INIT_DEPTH_SOURCE")
  [[ "$RESIDUAL_INIT_DEPTH_VALUE_EXPLICIT" == true ]] && lifecycle_args+=(--residual-init-depth-value "$RESIDUAL_INIT_DEPTH_VALUE")
  [[ "$DATASET_EXPLICIT" == true ]] && lifecycle_args+=(--dataset "$DATASET_NAME")
  [[ "$DATA_DIR_EXPLICIT" == true ]] && lifecycle_args+=(--data-dir "$DATA_DIR")
  [[ "$LOG_ROOT_EXPLICIT" == true ]] && lifecycle_args+=(--log-root "$LOG_ROOT")
  [[ "$RESULT_ROOT_EXPLICIT" == true ]] && lifecycle_args+=(--result-root "$RESULT_ROOT")
  # ^^^ THOG
  [[ "$EVAL_ITERS_EXPLICIT" == true ]] && lifecycle_args+=(--eval-iters "$EVAL_ITERS")
  [[ "$EVAL_INTERVAL_EXPLICIT" == true ]] && lifecycle_args+=(--eval-interval "$EVAL_INTERVAL")
  [[ "$LOG_INTERVAL_EXPLICIT" == true ]] && lifecycle_args+=(--log-interval "$LOG_INTERVAL")
  [[ "$CHECKPOINT_INTERVAL_EXPLICIT" == true ]] && lifecycle_args+=(--checkpoint-interval "$CHECKPOINT_INTERVAL")
  [[ "$INSTRUMENTATION_EXPLICIT" == true ]] && lifecycle_args+=(--instrumentation "$INSTRUMENTATION_BACKEND" --wandb-mode "$WANDB_MODE")
  [[ "$CHECKPOINT_SEGMENT_SIZE_EXPLICIT" == true ]] && lifecycle_args+=(--checkpoint-segment-size "$CHECKPOINT_SEGMENT_SIZE")
  if [[ "$FAST_DISCARD_EXPLICIT" == true ]]; then
    [[ "$FAST_DISCARD" == true ]] && lifecycle_args+=(--fast-discard) || lifecycle_args+=(--no-fast-discard)
  fi
  [[ "$DTYPE_EXPLICIT" == true ]] && lifecycle_args+=(--dtype "$DTYPE")
  [[ "$ATTENTION_BACKEND_EXPLICIT" == true ]] && lifecycle_args+=(--attention-backend "$ATTENTION_BACKEND")
  lifecycle_args+=("${EXTRA_ARGS[@]}")

  resolved_json="$(WORLD_SIZE="$NUM_GPUS" $PYTHON_BIN -m "$RUN_MODULE" "${lifecycle_args[@]}" --print-resolved-json)"
  artifact_name="$(printf '%s' "$resolved_json" | $PYTHON_BIN -c 'import json,sys; print(json.load(sys.stdin)["artifact_name"])')"
  log_path="$(printf '%s' "$resolved_json" | $PYTHON_BIN -c 'import json,sys; print(json.load(sys.stdin)["paths"]["log_path"])')"
  append_log="$(printf '%s' "$resolved_json" | $PYTHON_BIN -c 'import json,sys; print("true" if json.load(sys.stdin)["append_log"] else "false")')"
  command=("$PYTHON_BIN" -m "$RUN_MODULE" "${lifecycle_args[@]}")
  if (( NUM_GPUS > 1 )); then command=("$PYTHON_BIN" -m torch.distributed.run --standalone "--nproc-per-node=$NUM_GPUS" -m "$RUN_MODULE" "${lifecycle_args[@]}"); fi

  cat <<EOF_LIFECYCLE
${HOST_LABEL} OWT ${RUN_MODE}
  start time:         $start_time_friendly
  artifact:           $artifact_name
  resume from:        $RESUME_FROM
  total steps:        $STEPS
  gpus:               $NUM_GPUS
  log:                file://$(realpath -m "$log_path")
EOF_LIFECYCLE
  if [[ "$DRY_RUN" == true ]]; then
    WORLD_SIZE="$NUM_GPUS" $PYTHON_BIN -m "$RUN_MODULE" "${lifecycle_args[@]}" --dry-run
    printf 'DRY RUN:'; printf ' %q' "${command[@]}"; printf '\n'; return 0
  fi
  mkdir -p "$(dirname "$log_path")"
  set +e
  if [[ "$append_log" == true ]]; then "${command[@]}" 2>&1 | tee -a "$log_path"; else "${command[@]}" 2>&1 | tee "$log_path"; fi
  run_status=${PIPESTATUS[0]}
  set -e
  return "$run_status"
}

case "$RUN_MODE" in fresh|resume|fork) ;; *) echo "RUN_MODE must be fresh, resume, or fork." >&2; exit 2 ;; esac
if [[ "$RUN_MODE" != fresh ]]; then run_resume_or_fork; exit $?; fi
# ^^^ THOG

O_DEPTH_VALUES=()
PRESET_VALUES=()
BATCH_SIZE_VALUES=()                                                                                                                                        # <<< THOG batch grid axis
LEARNING_RATE_CODE_VALUES=()                                                                                                                                # <<< THOG LR grid axis
HAS_DENSE_PRESET=false
HAS_COMPACT_PRESET=false
parse_positive_uint_values() {
  local normalized="${1//,/ }" value
  for value in $normalized; do validate_positive_uint "$value" "$2"; BATCH_SIZE_VALUES+=("$value"); done
  (( ${#BATCH_SIZE_VALUES[@]} > 0 )) || { echo "Invalid BATCH_SIZE list." >&2; exit 2; }
}
# vvv THOG allow high-LR experiments while retaining bounded validation for each LR control
validate_lr_code() {
  local value="$1" label="$2" maximum="$3"
  [[ "$value" =~ ^[0-9]{1,4}$ ]] && (( 10#$value >= 1 && 10#$value <= maximum )) || {
    echo "Invalid $label: $value; expected 1..$maximum." >&2
    exit 2
  }
}
parse_lr_code_values() {
  local normalized="${1//,/ }" value
  for value in $normalized; do
    validate_lr_code "$value" "LEARNING_RATE_CODES" 1000
    LEARNING_RATE_CODE_VALUES+=("$((10#$value))")
  done
  (( ${#LEARNING_RATE_CODE_VALUES[@]} > 0 )) || { echo "Invalid learning-rate code list." >&2; exit 2; }
}
# ^^^ THOG
parse_o_depth_values() {
  local normalized="${1//,/ }"
  local value
  for value in $normalized; do
    validate_positive_uint "$value" "O_DEPTH"
    O_DEPTH_VALUES+=("$value")
  done
  (( ${#O_DEPTH_VALUES[@]} > 0 )) || { echo "Invalid O_DEPTH: empty value list." >&2; exit 2; }
}
parse_geometry_preset_values() {
  local normalized="${1//,/ }"
  local value
  for value in $normalized; do
    case "$value" in
      dense) PRESET_VALUES+=("$value"); HAS_DENSE_PRESET=true ;;
      legacy_sheet_col|depth|head_aware_block|mlp_block|full_block) PRESET_VALUES+=("$value"); HAS_COMPACT_PRESET=true ;;
      *) echo "Bad PRESET: $value" >&2; exit 2 ;;
    esac
  done
  (( ${#PRESET_VALUES[@]} > 0 )) || { echo "Invalid PRESET: empty value list." >&2; exit 2; }
}
parse_o_depth_values "$O_DEPTH"
parse_geometry_preset_values "$GEOMETRY_PRESET"
parse_positive_uint_values "$BATCH_SIZE" "BATCH_SIZE"                                                                                                  # <<< THOG parse batch grid
parse_lr_code_values "$LEARNING_RATE_CODES"                                                                                                              # <<< THOG parse LR grid
validate_lr_code "$MIN_LR_CODE" "MIN_LR_CODE" 100                                                                                                          # <<< THOG validate min LR

case "$RUN_MODE" in fresh|resume|fork) ;; *) echo "RUN_MODE must be fresh, resume, or fork." >&2; exit 2 ;; esac
case "$BASIS_FAMILY" in chebyshev|dct) ;; *) echo "BASIS_FAMILY must be chebyshev or dct." >&2; exit 2 ;; esac
case "$ATTENTION_BACKEND" in auto|flash2|sdpa|math) ;; *) echo "Bad ATTENTION_BACKEND: $ATTENTION_BACKEND" >&2; exit 2 ;; esac
# vvv THOG one instrumentation selector determines both backend and W&B mode; contradictory -I/-M/-W combinations no longer exist
case "$INSTRUMENTATION" in
  tensorboard) INSTRUMENTATION_BACKEND="tensorboard"; WANDB_FLAG="--no-wandb"; WANDB_MODE="disabled" ;;
  wandb) INSTRUMENTATION_BACKEND="wandb"; WANDB_FLAG="--wandb"; WANDB_MODE="online" ;;
  wandb_offline) INSTRUMENTATION_BACKEND="wandb"; WANDB_FLAG="--wandb"; WANDB_MODE="offline" ;;
  none) INSTRUMENTATION_BACKEND="none"; WANDB_FLAG="--no-wandb"; WANDB_MODE="disabled" ;;
  *) echo "INSTRUMENTATION must be tensorboard, wandb, wandb_offline, or none." >&2; exit 2 ;;
esac
# ^^^ THOG
case "$DEPTH_CURVE_PLOTS" in none|final|eval) ;; *) echo "DEPTH_CURVE_PLOTS must be none, final, or eval." >&2; exit 2 ;; esac
case "$DEPTH_CURVE_RENDERER" in matplotlib|plotly|both) ;; *) echo "DEPTH_CURVE_RENDERER must be matplotlib, plotly, or both." >&2; exit 2 ;; esac
case "$RESIDUAL_INIT_POLICY" in depth_scaled|unscaled) ;; *) echo "RESIDUAL_INIT_POLICY must be depth_scaled or unscaled." >&2; exit 2 ;; esac
case "$RESIDUAL_INIT_DEPTH_SOURCE" in true_layer_depth|dof_implied_depth|user_forced_depth) ;; *) echo "Bad RESIDUAL_INIT_DEPTH_SOURCE: $RESIDUAL_INIT_DEPTH_SOURCE" >&2; exit 2 ;; esac
for setting in "$STEPS" "$GRADIENT_ACCUMULATION_STEPS" "$NUM_GPUS" "$EVAL_ITERS" "$LOG_INTERVAL" "$N_LAYER" "$N_HEAD" "$N_EMBD" "$BLOCK_SIZE" "$O_ATTN_D_MODEL" "$O_ATTN_QKV_PER_CHANNEL" "$O_ATTN_OUT_PER_CHANNEL" "$O_MLP_D_MODEL" "$O_MLP_HIDDEN" "$CHECKPOINT_SEGMENT_SIZE" "$RESIDUAL_INIT_DEPTH_VALUE" "$DEPTH_CURVE_SAMPLE_ELEMENTS"; do validate_positive_uint "$setting" "numeric setting"; done
validate_nonnegative_uint "$EVAL_INTERVAL" "EVAL_INTERVAL"
validate_nonnegative_uint "$WARMUP_ITERS" "WARMUP_ITERS"
validate_nonnegative_uint "$CHECKPOINT_INTERVAL" "CHECKPOINT_INTERVAL"
validate_true_false "$ACTIVATION_CHECKPOINTING" "ACTIVATION_CHECKPOINTING"
validate_true_false "$FAST_DISCARD" "FAST_DISCARD"
validate_true_false "$BYPASS_SEMANTIC_QKV_ADAPTER" "BYPASS_SEMANTIC_QKV_ADAPTER"                                                                        # <<< THOG validate wrapper-only optimisation switch
validate_true_false "$DIRECT_FACTORISED_MLP" "DIRECT_FACTORISED_MLP"                                                                                   # <<< THOG validate renamed exact MLP option
validate_true_false "$VECTORISE_PER_HEAD_MATERIALISATION" "VECTORISE_PER_HEAD_MATERIALISATION"                                                         # <<< THOG validate per-head option                                                                          # <<< THOG validate wrapper-only exact MLP application switch
validate_true_false "$DEPTH_CURVE_LOCAL_HTML" "DEPTH_CURVE_LOCAL_HTML"
validate_true_false "$DRY_RUN" "DRY_RUN"

(( WARMUP_ITERS < STEPS )) || { echo "WARMUP_ITERS must be less than STEPS." >&2; exit 2; }
(( N_EMBD % N_HEAD == 0 )) || { echo "N_EMBD must be divisible by N_HEAD." >&2; exit 2; }
HEAD_DIM=$((N_EMBD / N_HEAD))
if [[ "$HAS_COMPACT_PRESET" == true ]]; then
  for value in "${O_DEPTH_VALUES[@]}"; do (( value <= N_LAYER )) || { echo "O_DEPTH must not exceed N_LAYER: P=${value}, L=${N_LAYER}." >&2; exit 2; }; done
  (( O_ATTN_D_MODEL <= N_EMBD )) || { echo "O_ATTN_D_MODEL must not exceed N_EMBD." >&2; exit 2; }
  (( O_ATTN_QKV_PER_CHANNEL <= HEAD_DIM )) || { echo "O_ATTN_QKV_PER_CHANNEL must not exceed N_EMBD/N_HEAD." >&2; exit 2; }
  (( O_ATTN_OUT_PER_CHANNEL <= HEAD_DIM )) || { echo "O_ATTN_OUT_PER_CHANNEL must not exceed N_EMBD/N_HEAD." >&2; exit 2; }
  (( O_MLP_D_MODEL <= N_EMBD )) || { echo "O_MLP_D_MODEL must not exceed N_EMBD." >&2; exit 2; }
  (( O_MLP_HIDDEN <= 4 * N_EMBD )) || { echo "O_MLP_HIDDEN must not exceed 4*N_EMBD." >&2; exit 2; }
fi
(( GRADIENT_ACCUMULATION_STEPS % NUM_GPUS == 0 )) || { echo "GRADIENT_ACCUMULATION_STEPS must be divisible by NUM_GPUS." >&2; exit 2; }

if [[ -n "${THOG2_PYTHON:-}" ]]; then PYTHON_BIN="$THOG2_PYTHON"; elif [[ -x .venv/bin/python ]]; then PYTHON_BIN=".venv/bin/python"; else PYTHON_BIN="python"; fi
BASIS_TAG="CHEBY"; [[ "$BASIS_FAMILY" == dct ]] && BASIS_TAG="DCT"
CHECKPOINT_FLAG="--no-activation-checkpointing"; [[ "$ACTIVATION_CHECKPOINTING" == true ]] && CHECKPOINT_FLAG="--activation-checkpointing"
FAST_DISCARD_FLAG="--no-fast-discard"; [[ "$FAST_DISCARD" == true ]] && FAST_DISCARD_FLAG="--fast-discard"
BYPASS_QKV_FLAG="--no-bypass-semantic-qkv-adapter"; [[ "$BYPASS_SEMANTIC_QKV_ADAPTER" == true ]] && BYPASS_QKV_FLAG="--bypass-semantic-qkv-adapter"
DIRECT_MLP_FLAG="--no-direct-factorised-mlp"; [[ "$DIRECT_FACTORISED_MLP" == true ]] && DIRECT_MLP_FLAG="--direct-factorised-mlp"
VECTORISE_HEAD_FLAG="--no-vectorise-per-head-materialisation"; [[ "$VECTORISE_PER_HEAD_MATERIALISATION" == true ]] && VECTORISE_HEAD_FLAG="--vectorise-per-head-materialisation"

export THOG2_INSTRUMENTATION="$INSTRUMENTATION_BACKEND"
export THOG2_CURVE_ROOT="${THOG2_CURVE_ROOT:-curves}"
export THOG2_MLP_CHANNEL_ORDER="$O_MLP_HIDDEN"
export THOG2_DEPTH_CURVE_PLOTS="$DEPTH_CURVE_PLOTS"
export THOG2_DEPTH_CURVE_SAMPLE_ELEMENTS="$DEPTH_CURVE_SAMPLE_ELEMENTS"
export THOG2_DEPTH_CURVE_RENDERER="$DEPTH_CURVE_RENDERER"
export THOG2_DEPTH_CURVE_LOCAL_HTML="$DEPTH_CURVE_LOCAL_HTML"
export THOG2_FAST_DISCARD="$FAST_DISCARD"
export THOG2_BYPASS_SEMANTIC_QKV_ADAPTER="$BYPASS_SEMANTIC_QKV_ADAPTER"                                                                                  # <<< THOG pass wrapper-only optimisation switch into SheetGPTConfig
export THOG2_DIRECT_FACTORISED_MLP="$DIRECT_FACTORISED_MLP"                                                                                              # <<< THOG pass renamed option
export THOG2_VECTORISE_PER_HEAD_MATERIALISATION="$VECTORISE_PER_HEAD_MATERIALISATION"                                                                    # <<< THOG pass per-head option                                                                                    # <<< THOG pass wrapper-only exact MLP application switch into SheetGPTConfig
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run_preset_o_depth_batch_lr() {
  local geometry_preset_value="$1"
  local o_depth_value="$2"
  local batch_size_value="$3"                                                                                                                             # <<< THOG batch grid coordinate
  local learning_rate_code="$4"                                                                                                                           # <<< THOG LR grid coordinate
  local learning_rate_value="${learning_rate_code}e-5" min_lr_value="$((10#$MIN_LR_CODE))e-5"                                                         # <<< THOG decode LR codes
  local run_model_type display_model_type preset_tag run_tag run_name_value LOG_TIMESTAMP resolved_json artifact_name log_path depth_curve_local_root
  local residual_init_depth_source_value n_layer_value n_head_value n_embd_value shape_summary orders_summary start_time_friendly log_url viewer_url serve_url run_status
  local -a compact_args optional_args train_args command

  n_layer_value="$N_LAYER"; n_head_value="$N_HEAD"; n_embd_value="$N_EMBD"
  residual_init_depth_source_value="$RESIDUAL_INIT_DEPTH_SOURCE"
  optional_args=(); compact_args=()
  if [[ "$geometry_preset_value" == dense ]]; then
    run_model_type="dense"; display_model_type="dense"; preset_tag="DENSE"; run_tag="DENSE"
    [[ "$N_LAYER_EXPLICIT" == false ]] && n_layer_value=12
    [[ "$N_HEAD_EXPLICIT" == false ]] && n_head_value=12
    [[ "$N_EMBD_EXPLICIT" == false ]] && n_embd_value=768
    [[ "$residual_init_depth_source_value" == dof_implied_depth ]] && residual_init_depth_source_value="true_layer_depth"
    shape_summary="L${n_layer_value} H${n_head_value} D${n_embd_value} C${BLOCK_SIZE}"
    orders_summary="n/a"
  else
    run_model_type="sheet"; display_model_type="spectral"; preset_tag="${geometry_preset_value^^}"
    [[ "$geometry_preset_value" == legacy_sheet_col ]] && preset_tag="SHEET_COL"
    run_tag="${BASIS_TAG}_${preset_tag}"
    compact_args=(--geometry-preset "$geometry_preset_value" --basis-family "$BASIS_FAMILY" --basis-version "$BASIS_VERSION")
    [[ -n "$ATTENTION_GEOMETRY" ]] && optional_args+=(--attention-geometry "$ATTENTION_GEOMETRY")
    [[ -n "$MLP_GEOMETRY" ]] && optional_args+=(--mlp-geometry "$MLP_GEOMETRY")
    shape_summary="L${n_layer_value} H${n_head_value} D${n_embd_value} C${BLOCK_SIZE}"
    orders_summary="P${o_depth_value} Q${O_ATTN_D_MODEL} J${O_ATTN_QKV_PER_CHANNEL} O${O_ATTN_OUT_PER_CHANNEL} X${O_MLP_D_MODEL} Y${O_MLP_HIDDEN}"
  fi

  run_name_value="$RUN_NAME"; [[ -z "$run_name_value" ]] && run_name_value="${run_tag}_OWT"
  train_args=(
    --model-type "$run_model_type" --run-mode "$RUN_MODE" --host-label "$HOST_LABEL" --run-name "$run_name_value"
    --dataset "$DATASET_NAME" --data-dir "$DATA_DIR" --checkpoint-root "$CHECKPOINT_ROOT" --log-root "$LOG_ROOT" --result-root "$RESULT_ROOT" --wandb-root "$WANDB_ROOT"
    --max-iters "$STEPS" --batch-size "$batch_size_value" --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
    --eval-iters "$EVAL_ITERS" --eval-interval "$EVAL_INTERVAL" --log-interval "$LOG_INTERVAL" --checkpoint-interval "$CHECKPOINT_INTERVAL" --warmup-iters "$WARMUP_ITERS" --learning-rate "$learning_rate_value" --min-lr "$min_lr_value"
    --n-layer "$n_layer_value" --n-head "$n_head_value" --n-embd "$n_embd_value" --block-size "$BLOCK_SIZE"
    --o-depth "$o_depth_value" --o-attn-d-model "$O_ATTN_D_MODEL" --o-attn-qkv-per-channel "$O_ATTN_QKV_PER_CHANNEL" --o-attn-out-per-channel "$O_ATTN_OUT_PER_CHANNEL" --o-mlp-d-model "$O_MLP_D_MODEL" --o-mlp-hidden "$O_MLP_HIDDEN"
    "${compact_args[@]}" --attention-backend "$ATTENTION_BACKEND" --experiment-prefix "$EXPERIMENT_PREFIX" --dtype "$DTYPE"
    --residual-init-policy "$RESIDUAL_INIT_POLICY" --residual-init-depth-source "$residual_init_depth_source_value" --residual-init-depth-value "$RESIDUAL_INIT_DEPTH_VALUE"
    "$CHECKPOINT_FLAG" --checkpoint-segment-size "$CHECKPOINT_SEGMENT_SIZE" "$WANDB_FLAG" --wandb-mode "$WANDB_MODE" --instrumentation "$INSTRUMENTATION_BACKEND" "$FAST_DISCARD_FLAG" "$BYPASS_QKV_FLAG" "$DIRECT_MLP_FLAG" "$VECTORISE_HEAD_FLAG" "${optional_args[@]}" "${EXTRA_ARGS[@]}"
  )

  LOG_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  start_time_friendly="$(date '+%H:%M  %d-%m-%y')"
  resolved_json="$($PYTHON_BIN -m "$RUN_MODULE" "${train_args[@]}" --log-timestamp "$LOG_TIMESTAMP" --print-resolved-json)"
  artifact_name="$(printf '%s' "$resolved_json" | $PYTHON_BIN -c 'import json,sys; print(json.load(sys.stdin)["artifact_name"])')"
  log_path="$(printf '%s' "$resolved_json" | $PYTHON_BIN -c 'import json,sys; print(json.load(sys.stdin)["paths"]["log_path"])')"
  depth_curve_local_root="$(dirname "$log_path")/depth_curves"; export THOG2_DEPTH_CURVE_LOCAL_ROOT="$depth_curve_local_root"
  command=("$PYTHON_BIN" -m "$RUN_MODULE" "${train_args[@]}" --log-timestamp "$LOG_TIMESTAMP")
  if (( NUM_GPUS > 1 )); then command=("$PYTHON_BIN" -m torch.distributed.run --standalone "--nproc-per-node=$NUM_GPUS" -m "$RUN_MODULE" "${train_args[@]}" --log-timestamp "$LOG_TIMESTAMP"); fi
  log_url="file://$(realpath -m "$log_path")"; viewer_url="file://$(realpath -m "$depth_curve_local_root/index.html")"; serve_url="http://localhost:${DEPTH_CURVE_HTTP_PORT}/"

  cat <<EOF_RUN
dreedle OWT train
  start time:         $start_time_friendly
  artifact:           $artifact_name
  experiment:         $EXPERIMENT_PREFIX
  model/preset/basis: $display_model_type / $geometry_preset_value / $BASIS_FAMILY
  backend/dtype:      $ATTENTION_BACKEND / $DTYPE
  instrumentation:    $INSTRUMENTATION
  fast discard:       $FAST_DISCARD
  semantic adapter bypass:   $BYPASS_SEMANTIC_QKV_ADAPTER
  direct factorised MLP:    $DIRECT_FACTORISED_MLP
  vectorise per-head materialisation: $VECTORISE_PER_HEAD_MATERIALISATION
  depth curves:       $DEPTH_CURVE_PLOTS  (sample elements: $DEPTH_CURVE_SAMPLE_ELEMENTS, renderer: $DEPTH_CURVE_RENDERER, local html: $DEPTH_CURVE_LOCAL_HTML)
  depth viewer:       $viewer_url
  serve viewer:       (cd $depth_curve_local_root && python -m http.server $DEPTH_CURVE_HTTP_PORT)
  served URL:         $serve_url
  optimizer:          lr=$learning_rate_value (LR_$learning_rate_code) min_lr=$min_lr_value
  schedule:           steps=$STEPS eval_every=$EVAL_INTERVAL eval_iters=$EVAL_ITERS log_every=$LOG_INTERVAL ckpt_every=$CHECKPOINT_INTERVAL warmup=$WARMUP_ITERS
  shape:              $shape_summary
  orders:             $orders_summary
  batch/accum/gpus:   $batch_size_value / $GRADIENT_ACCUMULATION_STEPS / $NUM_GPUS
  log:                $log_url
EOF_RUN

  if [[ "$DRY_RUN" == true ]]; then
    "$PYTHON_BIN" -m "$RUN_MODULE" "${train_args[@]}" --log-timestamp "$LOG_TIMESTAMP" --dry-run
    printf 'DRY RUN:'; printf ' %q' "${command[@]}"; printf '\n'; return 0
  fi
  mkdir -p "$(dirname "$log_path")"
  set +e; "${command[@]}" 2>&1 | tee "$log_path"; run_status=${PIPESTATUS[0]}; set -e
  cat <<EOF_DONE
dreedle OWT run finished
  status:             $run_status
  artifact:           $artifact_name
  log URL:            $log_url
  depth viewer URL:   $viewer_url
EOF_DONE
  return "$run_status"
}

if (( ${#PRESET_VALUES[@]} > 1 || ${#O_DEPTH_VALUES[@]} > 1 || ${#BATCH_SIZE_VALUES[@]} > 1 || ${#LEARNING_RATE_CODE_VALUES[@]} > 1 )); then
  echo "dreedle OWT grid: p=${PRESET_VALUES[*]} P=${O_DEPTH_VALUES[*]} b=${BATCH_SIZE_VALUES[*]} LR=${LEARNING_RATE_CODE_VALUES[*]}"
fi
for geometry_preset_value in "${PRESET_VALUES[@]}"; do
  for batch_size_value in "${BATCH_SIZE_VALUES[@]}"; do
    for learning_rate_code in "${LEARNING_RATE_CODE_VALUES[@]}"; do
      if [[ "$geometry_preset_value" == dense ]]; then
        run_preset_o_depth_batch_lr "$geometry_preset_value" "${O_DEPTH_VALUES[0]}" "$batch_size_value" "$learning_rate_code"
      else
        for o_depth_value in "${O_DEPTH_VALUES[@]}"; do run_preset_o_depth_batch_lr "$geometry_preset_value" "$o_depth_value" "$batch_size_value" "$learning_rate_code"; done
      fi
    done
  done
done
# ^^^ THOG
