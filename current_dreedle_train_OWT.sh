#!/bin/bash
set -euo pipefail

# vvv THOG
# Current dreedle OpenWebText training wrapper for the PICTON compact-geometry contract.
# Dreedle runtime defaults: float16, sdpa. Dense baseline is available as -p dense.
#
# Optimizer selection is native to this full wrapper:
#   -y NAME, --optimizer NAME
#       adamw | sgd | sgd_nesterov | adafactor | rmsprop
#       Aliases: adam; nesterov; sgd-nesterov.
#   --optimizer-momentum VALUE
#       Momentum for sgd, sgd_nesterov, and rmsprop. Default: 0.9.
#
# Optimizer-specific learning-rate defaults apply only when -c and/or -f are omitted:
#   optimizer       -c / maximum LR       -f / minimum LR
#   adamw             60 / 6.0e-4           06 / 6.0e-5
#   sgd             1000 / 1.0e-2          100 / 1.0e-3
#   sgd_nesterov    1000 / 1.0e-2          100 / 1.0e-3
#   adafactor       1000 / 1.0e-2          100 / 1.0e-3
#   rmsprop          100 / 1.0e-3           10 / 1.0e-4
#
# Explicit -c and -f values override those defaults independently. Lowercase -c is
# the learning-rate code; capital -C remains the context length. Non-AdamW runs add
# OPT_<OPTIMIZER> to the artifact suffix to prevent otherwise identical collisions.
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
LAPPED_COSINE_WINDOW_LENGTH=36                                                                                                                             # <<< THOG default lapped locality scale
LAPPED_COSINE_OVERLAP_FRACTION="0.5"                                                                                                                       # <<< THOG v1 fixed overlap
ATTENTION_GEOMETRY=""
MLP_GEOMETRY=""
STEPS=250
BATCH_SIZE=3
LEARNING_RATE_CODES="60"                                                                                                                               # <<< THOG wrapper learning-rate code list; 70 means 7.0e-04
MIN_LR_CODE="06"                                                                                                                                         # <<< THOG wrapper minimum learning-rate code; 1..100; 06 means 6.0e-05 and 100 means 1.0e-03
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
DIRECT_FACTORISED_MLP="${THOG2_DIRECT_FACTORISED_MLP:-true}"                                                                                              # <<< THOG default-on exact direct application of existing THOG MLP factors
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

# vvv THOG native optimizer selection; strip optimizer-only controls before canonical option parsing
OPTIMIZER="${THOG2_OPTIMIZER:-adamw}"
OPTIMIZER_MOMENTUM="${THOG2_OPTIMIZER_MOMENTUM:-0.9}"
OPTIMIZER_LR_EXPLICIT=false
OPTIMIZER_MIN_LR_EXPLICIT=false
OPTIMIZER_FILTERED_ARGS=()
OPTIMIZER_SAW_SEPARATOR=false
while (( $# > 0 )); do
  if [[ "$OPTIMIZER_SAW_SEPARATOR" == true ]]; then
    OPTIMIZER_FILTERED_ARGS+=("$1")
    shift
    continue
  fi
  case "$1" in
    -y|--optimizer)
      (( $# >= 2 )) || { echo "$1 requires an optimizer name" >&2; exit 2; }
      OPTIMIZER="$2"
      shift 2
      ;;
    --optimizer=*)
      OPTIMIZER="${1#*=}"
      shift
      ;;
    --optimizer-momentum)
      (( $# >= 2 )) || { echo "$1 requires a numeric value" >&2; exit 2; }
      OPTIMIZER_MOMENTUM="$2"
      shift 2
      ;;
    --optimizer-momentum=*)
      OPTIMIZER_MOMENTUM="${1#*=}"
      shift
      ;;
    -c)
      (( $# >= 2 )) || { echo "-c requires a learning-rate code" >&2; exit 2; }
      OPTIMIZER_LR_EXPLICIT=true
      OPTIMIZER_FILTERED_ARGS+=("$1" "$2")
      shift 2
      ;;
    -f)
      (( $# >= 2 )) || { echo "-f requires a minimum-learning-rate code" >&2; exit 2; }
      OPTIMIZER_MIN_LR_EXPLICIT=true
      OPTIMIZER_FILTERED_ARGS+=("$1" "$2")
      shift 2
      ;;
    --)
      OPTIMIZER_FILTERED_ARGS+=("--")
      OPTIMIZER_SAW_SEPARATOR=true
      shift
      ;;
    *)
      OPTIMIZER_FILTERED_ARGS+=("$1")
      shift
      ;;
  esac
done
set -- "${OPTIMIZER_FILTERED_ARGS[@]}"

case "${OPTIMIZER,,}" in
  adam|adamw)
    OPTIMIZER="adamw"; OPTIMIZER_DEFAULT_LR_CODE="60"; OPTIMIZER_DEFAULT_MIN_LR_CODE="06" ;;
  sgd)
    OPTIMIZER="sgd"; OPTIMIZER_DEFAULT_LR_CODE="1000"; OPTIMIZER_DEFAULT_MIN_LR_CODE="100" ;;
  nesterov|sgd-nesterov|sgd_nesterov)
    OPTIMIZER="sgd_nesterov"; OPTIMIZER_DEFAULT_LR_CODE="1000"; OPTIMIZER_DEFAULT_MIN_LR_CODE="100" ;;
  adafactor)
    OPTIMIZER="adafactor"; OPTIMIZER_DEFAULT_LR_CODE="1000"; OPTIMIZER_DEFAULT_MIN_LR_CODE="100" ;;
  rmsprop)
    OPTIMIZER="rmsprop"; OPTIMIZER_DEFAULT_LR_CODE="100"; OPTIMIZER_DEFAULT_MIN_LR_CODE="10" ;;
  *)
    echo "Unsupported optimizer: $OPTIMIZER" >&2
    echo "Expected: adamw | sgd | sgd_nesterov | adafactor | rmsprop" >&2
    exit 2
    ;;
esac
[[ "$OPTIMIZER_LR_EXPLICIT" == true ]] || LEARNING_RATE_CODES="$OPTIMIZER_DEFAULT_LR_CODE"
[[ "$OPTIMIZER_MIN_LR_EXPLICIT" == true ]] || MIN_LR_CODE="$OPTIMIZER_DEFAULT_MIN_LR_CODE"
export THOG2_OPTIMIZER="$OPTIMIZER"
export THOG2_OPTIMIZER_MOMENTUM="$OPTIMIZER_MOMENTUM"
# ^^^ THOG

usage() {
  cat <<EOF_USAGE
Usage: $0 [options] [-- extra ${RUN_MODULE} args]

Model/run:
  -p PRESET=${GEOMETRY_PRESET}                       dense | legacy_sheet_col | depth | head_aware_block | mlp_block | full_block
                                                   single value, comma list, or quoted space list
  -q RUN_MODE=${RUN_MODE}                        fresh | resume
  -g RUN_NAME=${RUN_NAME:-auto}
  -n STEPS=${STEPS}
  -b BATCH_SIZE=${BATCH_SIZE}                         single integer, comma list, or quoted space list
  -c LR_CODES=${LEARNING_RATE_CODES}                    learning-rate codes 1..1000; 70 means 7.0e-04; list allowed
  -f MIN_LR_CODE=${MIN_LR_CODE}                         minimum LR code; 1..100; 06 means 6.0e-05 and 100 means 1.0e-03
  -A GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}
  -G NUM_GPUS=${NUM_GPUS}
  -y OPTIMIZER=${OPTIMIZER}                         adamw | sgd | sgd_nesterov | adafactor | rmsprop
  --optimizer-momentum VALUE=${OPTIMIZER_MOMENTUM}  momentum for SGD/Nesterov/RMSprop
                                                     defaults: adamw 60/06; sgd 1000/100;
                                                     sgd_nesterov 1000/100; adafactor 1000/100;
                                                     rmsprop 100/10. Explicit -c/-f override independently.

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
  -B BASIS_FAMILY=${BASIS_FAMILY}                   canonical: chebyshev | dct | haar | lapped_cosine; single, comma, or quoted space list
                                                    Chebyshev aliases: cheby | chebyshev_first_kind_qr
                                                    DCT aliases: dct_ii | dct_ii_orthonormal
                                                    Haar aliases: balanced_haar | haar_balanced
                                                     Lapped cosine aliases: lapped | local_cosine | lapped_local_cosine
  -v BASIS_VERSION=${BASIS_VERSION}                 auto (recommended), or exact:
                                                    chebyshev_first_kind_qr_v1
                                                    dct_ii_orthonormal_v1
                                                    haar_balanced_binary_orthonormal_v1
                                                     lapped_cosine_dc_preserving_orthonormal_v1
  -W LAPPED_COSINE_WINDOW_LENGTH=${LAPPED_COSINE_WINDOW_LENGTH}
  -i LAPPED_COSINE_OVERLAP_FRACTION=${LAPPED_COSINE_OVERLAP_FRACTION}  currently 0.5 only
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

while getopts ":q:g:n:b:c:f:A:G:u:e:l:w:k:I:F:N:U:V:p:B:v:W:i:a:m:L:H:D:C:P:Q:J:O:X:Y:S:E:T:K:r:z:Z:d:t:o:j:R:x:h" option; do
  case "$option" in
    q) RUN_MODE="$OPTARG" ;; g) RUN_NAME="$OPTARG" ;;
    n) STEPS="$OPTARG" ;; b) BATCH_SIZE="$OPTARG" ;; c) LEARNING_RATE_CODES="$OPTARG" ;; f) MIN_LR_CODE="$OPTARG" ;; A) GRADIENT_ACCUMULATION_STEPS="$OPTARG" ;; G) NUM_GPUS="$OPTARG" ;;
    u) EVAL_ITERS="$OPTARG" ;; e) EVAL_INTERVAL="$OPTARG" ;; l) LOG_INTERVAL="$OPTARG" ;; w) WARMUP_ITERS="$OPTARG" ;; k) CHECKPOINT_INTERVAL="$OPTARG" ;;
    I) INSTRUMENTATION="$OPTARG" ;; F) DEPTH_CURVE_PLOTS="$OPTARG" ;; N) DEPTH_CURVE_SAMPLE_ELEMENTS="$OPTARG" ;; U) DEPTH_CURVE_RENDERER="$OPTARG" ;; V) DEPTH_CURVE_LOCAL_HTML="$OPTARG" ;;
    p) GEOMETRY_PRESET="$OPTARG" ;; B) BASIS_FAMILY="$OPTARG" ;; v) BASIS_VERSION="$OPTARG" ;; W) LAPPED_COSINE_WINDOW_LENGTH="$OPTARG" ;; i) LAPPED_COSINE_OVERLAP_FRACTION="$OPTARG" ;; a) ATTENTION_GEOMETRY="$OPTARG" ;; m) MLP_GEOMETRY="$OPTARG" ;;
    L) N_LAYER="$OPTARG"; N_LAYER_EXPLICIT=true ;; H) N_HEAD="$OPTARG"; N_HEAD_EXPLICIT=true ;; D) N_EMBD="$OPTARG"; N_EMBD_EXPLICIT=true ;;
    C) BLOCK_SIZE="$OPTARG" ;; P) O_DEPTH="$OPTARG" ;; Q) O_ATTN_D_MODEL="$OPTARG" ;; J) O_ATTN_QKV_PER_CHANNEL="$OPTARG" ;; O) O_ATTN_OUT_PER_CHANNEL="$OPTARG" ;; X) O_MLP_D_MODEL="$OPTARG" ;; Y) O_MLP_HIDDEN="$OPTARG" ;; S) CHECKPOINT_SEGMENT_SIZE="$OPTARG" ;;
    E) FAST_DISCARD="$OPTARG" ;; T) DTYPE="$OPTARG" ;; K) ATTENTION_BACKEND="$OPTARG" ;; r) RESIDUAL_INIT_POLICY="$OPTARG" ;; z) RESIDUAL_INIT_DEPTH_SOURCE="$OPTARG" ;; Z) RESIDUAL_INIT_DEPTH_VALUE="$OPTARG" ;;
    d) DATASET_NAME="$OPTARG"; DATA_DIR="data/$OPTARG" ;; t) DATA_DIR="$OPTARG" ;; o) CHECKPOINT_ROOT="$OPTARG" ;; j) LOG_ROOT="$OPTARG" ;; R) RESULT_ROOT="$OPTARG" ;; x) DRY_RUN="$OPTARG" ;;
    h) usage; exit 0 ;; :) echo "Option -$OPTARG requires an argument." >&2; exit 2 ;; \?) echo "Unknown option: -$OPTARG" >&2; usage >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
if [[ "${1:-}" == "--" ]]; then shift; fi
EXTRA_ARGS=("$@")
# vvv THOG make optimizer identity collision-safe in artifact naming
if [[ "$OPTIMIZER" != "adamw" ]]; then
  OPTIMIZER_SUFFIX="OPT_${OPTIMIZER^^}"
  OPTIMIZER_UPDATED_EXTRA_ARGS=()
  OPTIMIZER_FOUND_ARTIFACT_SUFFIX=false
  for (( optimizer_index=0; optimizer_index < ${#EXTRA_ARGS[@]}; optimizer_index++ )); do
    optimizer_argument="${EXTRA_ARGS[optimizer_index]}"
    case "$optimizer_argument" in
      --artifact-suffix)
        (( optimizer_index + 1 < ${#EXTRA_ARGS[@]} )) || { echo "--artifact-suffix requires a value" >&2; exit 2; }
        OPTIMIZER_UPDATED_EXTRA_ARGS+=("--artifact-suffix" "${EXTRA_ARGS[optimizer_index + 1]}_${OPTIMIZER_SUFFIX}")
        optimizer_index=$((optimizer_index + 1))
        OPTIMIZER_FOUND_ARTIFACT_SUFFIX=true
        ;;
      --artifact-suffix=*)
        OPTIMIZER_UPDATED_EXTRA_ARGS+=("--artifact-suffix=${optimizer_argument#*=}_${OPTIMIZER_SUFFIX}")
        OPTIMIZER_FOUND_ARTIFACT_SUFFIX=true
        ;;
      *)
        OPTIMIZER_UPDATED_EXTRA_ARGS+=("$optimizer_argument")
        ;;
    esac
  done
  if [[ "$OPTIMIZER_FOUND_ARTIFACT_SUFFIX" == false ]]; then
    OPTIMIZER_UPDATED_EXTRA_ARGS+=("--artifact-suffix" "$OPTIMIZER_SUFFIX")
  fi
  EXTRA_ARGS=("${OPTIMIZER_UPDATED_EXTRA_ARGS[@]}")
fi
# ^^^ THOG
EXPERIMENT_PREFIX="${RUN_NAME:-NO_PREFIX}"                                                                                                               # <<< THOG make -g the sole experiment-prefix source

validate_positive_uint() { [[ "$1" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid $2: $1; expected a positive integer." >&2; exit 2; }; }
validate_nonnegative_uint() { [[ "$1" =~ ^[0-9]+$ ]] || { echo "Invalid $2: $1; expected a non-negative integer." >&2; exit 2; }; }
validate_true_false() { case "$1" in true|false) ;; *) echo "Invalid $2: $1; expected true or false." >&2; exit 2 ;; esac; }

O_DEPTH_VALUES=()
PRESET_VALUES=()
BASIS_FAMILY_VALUES=()                                                                                                                                    # <<< THOG basis-family grid axis
BASIS_TAG_VALUES=()                                                                                                                                       # <<< THOG matching artifact tags for basis-family grid
BATCH_SIZE_VALUES=()                                                                                                                                       # <<< THOG batch-size grid axis
LEARNING_RATE_CODE_VALUES=()                                                                                                                               # <<< THOG learning-rate grid axis
HAS_DENSE_PRESET=false
HAS_COMPACT_PRESET=false
parse_o_depth_values() {
  local normalized="${1//,/ }"
  local value
  for value in $normalized; do
    validate_positive_uint "$value" "O_DEPTH"
    O_DEPTH_VALUES+=("$value")
  done
  (( ${#O_DEPTH_VALUES[@]} > 0 )) || { echo "Invalid O_DEPTH: empty value list." >&2; exit 2; }
}
parse_positive_uint_values() {
  local raw="$1" label="$2" array_name="$3"
  local normalized="${raw//,/ }" value
  for value in $normalized; do
    validate_positive_uint "$value" "$label"
    eval "$array_name+=(\"$value\")"
  done
  eval "(( \${#$array_name[@]} > 0 ))" || { echo "Invalid $label: empty value list." >&2; exit 2; }
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
parse_basis_family_values() {
  local normalized="${1//,/ }" value
  for value in $normalized; do
    [[ "$value" =~ ^[a-z][a-z0-9_]*$ ]] || { echo "Invalid BASIS_FAMILY value: $value; expected a lowercase registry name or alias." >&2; exit 2; }
    BASIS_FAMILY_VALUES+=("$value")
  done
  (( ${#BASIS_FAMILY_VALUES[@]} > 0 )) || { echo "Invalid BASIS_FAMILY: empty value list." >&2; exit 2; }
}
parse_o_depth_values "$O_DEPTH"
parse_geometry_preset_values "$GEOMETRY_PRESET"
parse_basis_family_values "$BASIS_FAMILY"                                                                                                                  # <<< THOG parse basis-family grid
parse_positive_uint_values "$BATCH_SIZE" "BATCH_SIZE" BATCH_SIZE_VALUES                                                                             # <<< THOG parse batch grid
parse_lr_code_values "$LEARNING_RATE_CODES"                                                                                                            # <<< THOG parse LR grid
validate_lr_code "$MIN_LR_CODE" "MIN_LR_CODE" 100                                                                                                        # <<< THOG validate minimum LR code

case "$RUN_MODE" in fresh|resume) ;; *) echo "RUN_MODE must be fresh or resume." >&2; exit 2 ;; esac
# vvv THOG
if (( ${#BASIS_FAMILY_VALUES[@]} > 1 )) && [[ "$BASIS_VERSION" != auto ]]; then
  echo "BASIS_VERSION must be auto when BASIS_FAMILY contains multiple values." >&2
  exit 2
fi
# ^^^ THOG
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
for setting in "$STEPS" "$GRADIENT_ACCUMULATION_STEPS" "$NUM_GPUS" "$EVAL_ITERS" "$EVAL_INTERVAL" "$LOG_INTERVAL" "$N_LAYER" "$N_HEAD" "$N_EMBD" "$BLOCK_SIZE" "$O_ATTN_D_MODEL" "$O_ATTN_QKV_PER_CHANNEL" "$O_ATTN_OUT_PER_CHANNEL" "$O_MLP_D_MODEL" "$O_MLP_HIDDEN" "$CHECKPOINT_SEGMENT_SIZE" "$RESIDUAL_INIT_DEPTH_VALUE" "$DEPTH_CURVE_SAMPLE_ELEMENTS" "$LAPPED_COSINE_WINDOW_LENGTH"; do validate_positive_uint "$setting" "numeric setting"; done
# vvv THOG lapped cosine v1 accepts exactly 50 percent overlap
case "$LAPPED_COSINE_OVERLAP_FRACTION" in
  .5|0.5|0.50|0.500) LAPPED_COSINE_OVERLAP_FRACTION="0.5" ;;
  *) echo "LAPPED_COSINE_OVERLAP_FRACTION currently supports only 0.5." >&2; exit 2 ;;
esac
(( LAPPED_COSINE_WINDOW_LENGTH >= 2 && LAPPED_COSINE_WINDOW_LENGTH % 2 == 0 )) || { echo "LAPPED_COSINE_WINDOW_LENGTH must be an even integer >= 2." >&2; exit 2; }
# ^^^ THOG
validate_nonnegative_uint "$WARMUP_ITERS" "WARMUP_ITERS"
validate_nonnegative_uint "$CHECKPOINT_INTERVAL" "CHECKPOINT_INTERVAL"
validate_true_false "$ACTIVATION_CHECKPOINTING" "ACTIVATION_CHECKPOINTING"
validate_true_false "$FAST_DISCARD" "FAST_DISCARD"
validate_true_false "$BYPASS_SEMANTIC_QKV_ADAPTER" "BYPASS_SEMANTIC_QKV_ADAPTER"                                                                        # <<< THOG validate wrapper-only optimisation switch
validate_true_false "$DIRECT_FACTORISED_MLP" "DIRECT_FACTORISED_MLP"                                                                                   # <<< THOG validate renamed exact MLP switch
validate_true_false "$VECTORISE_PER_HEAD_MATERIALISATION" "VECTORISE_PER_HEAD_MATERIALISATION"                                                         # <<< THOG validate selectable head vectorisation
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
# vvv THOG
# BASIS_TAG="$("$PYTHON_BIN" -c 'import sys; from sheet.bases import basis_artifact_tag_for_family; print(basis_artifact_tag_for_family(sys.argv[1]))' "$BASIS_FAMILY")"
BASIS_FAMILY_CANONICAL_VALUES=()
for requested_basis_family in "${BASIS_FAMILY_VALUES[@]}"; do
  if ! basis_resolution="$("$PYTHON_BIN" -c 'import sys; from sheet.bases import basis_artifact_tag_for_family, normalize_registered_basis_family; family = normalize_registered_basis_family(sys.argv[1]); print(f"{family}\t{basis_artifact_tag_for_family(family)}")' "$requested_basis_family")"; then
    echo "Failed to resolve BASIS_FAMILY: $requested_basis_family" >&2
    exit 2
  fi
  IFS=$'\t' read -r basis_family_value basis_tag <<< "$basis_resolution"
  BASIS_FAMILY_CANONICAL_VALUES+=("$basis_family_value")
  BASIS_TAG_VALUES+=("$basis_tag")
done
BASIS_FAMILY_VALUES=("${BASIS_FAMILY_CANONICAL_VALUES[@]}")
# ^^^ THOG
CHECKPOINT_FLAG="--no-activation-checkpointing"; [[ "$ACTIVATION_CHECKPOINTING" == true ]] && CHECKPOINT_FLAG="--activation-checkpointing"

export THOG2_INSTRUMENTATION="$INSTRUMENTATION_BACKEND"
export THOG2_CURVE_ROOT="${THOG2_CURVE_ROOT:-curves}"
export THOG2_MLP_CHANNEL_ORDER="$O_MLP_HIDDEN"
export THOG2_DEPTH_CURVE_PLOTS="$DEPTH_CURVE_PLOTS"
export THOG2_DEPTH_CURVE_SAMPLE_ELEMENTS="$DEPTH_CURVE_SAMPLE_ELEMENTS"
export THOG2_DEPTH_CURVE_RENDERER="$DEPTH_CURVE_RENDERER"
export THOG2_DEPTH_CURVE_LOCAL_HTML="$DEPTH_CURVE_LOCAL_HTML"
export THOG2_FAST_DISCARD="$FAST_DISCARD"
export THOG2_BYPASS_SEMANTIC_QKV_ADAPTER="$BYPASS_SEMANTIC_QKV_ADAPTER"                                                                                  # <<< THOG pass wrapper-only optimisation switch into SheetGPTConfig
export THOG2_DIRECT_FACTORISED_MLP="$DIRECT_FACTORISED_MLP"                                                                                              # <<< THOG pass renamed exact MLP switch into SheetGPTConfig
export THOG2_VECTORISE_PER_HEAD_MATERIALISATION="$VECTORISE_PER_HEAD_MATERIALISATION"                                                                    # <<< THOG pass selectable head vectorisation into SheetGPTConfig
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run_grid_point() {
  local geometry_preset_value="$1"
  local o_depth_value="$2"
  local batch_size_value="$3"                                                                                                                            # <<< THOG batch grid coordinate
  local learning_rate_code="$4"                                                                                                                          # <<< THOG LR grid coordinate
  local basis_family_value="$5"                                                                                                                           # <<< THOG canonical basis-family grid coordinate
  local basis_tag="$6"                                                                                                                                    # <<< THOG matching basis artifact tag
  local learning_rate_value="${learning_rate_code}e-5" min_lr_value="$((10#$MIN_LR_CODE))e-5"                                                         # <<< THOG decode compact LR codes
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
    run_tag="${basis_tag}_${preset_tag}"
    compact_args=(--geometry-preset "$geometry_preset_value" --basis-family "$basis_family_value" --basis-version "$BASIS_VERSION" --lapped-cosine-window-length "$LAPPED_COSINE_WINDOW_LENGTH" --lapped-cosine-overlap-fraction "$LAPPED_COSINE_OVERLAP_FRACTION")
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
    "$CHECKPOINT_FLAG" --checkpoint-segment-size "$CHECKPOINT_SEGMENT_SIZE" "$WANDB_FLAG" --wandb-mode "$WANDB_MODE" "${optional_args[@]}" "${EXTRA_ARGS[@]}"
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
  model/preset/basis: $display_model_type / $geometry_preset_value / $basis_family_value
  lapped cosine:      window=$LAPPED_COSINE_WINDOW_LENGTH overlap=$LAPPED_COSINE_OVERLAP_FRACTION
  backend/dtype:      $ATTENTION_BACKEND / $DTYPE
  instrumentation:    $INSTRUMENTATION
  fast discard:       $FAST_DISCARD
  semantic adapter bypass:   $BYPASS_SEMANTIC_QKV_ADAPTER
  direct factorised MLP:       $DIRECT_FACTORISED_MLP
  vectorise per-head materialisation: $VECTORISE_PER_HEAD_MATERIALISATION
  depth curves:       $DEPTH_CURVE_PLOTS  (sample elements: $DEPTH_CURVE_SAMPLE_ELEMENTS, renderer: $DEPTH_CURVE_RENDERER, local html: $DEPTH_CURVE_LOCAL_HTML)
  depth viewer:       $viewer_url
  serve viewer:       (cd $depth_curve_local_root && python -m http.server $DEPTH_CURVE_HTTP_PORT)
  served URL:         $serve_url
  schedule:           steps=$STEPS eval_every=$EVAL_INTERVAL eval_iters=$EVAL_ITERS log_every=$LOG_INTERVAL ckpt_every=$CHECKPOINT_INTERVAL warmup=$WARMUP_ITERS
  optimiser:          $OPTIMIZER  momentum=$OPTIMIZER_MOMENTUM  lr_code=$learning_rate_code lr=$learning_rate_value min_lr_code=$MIN_LR_CODE min_lr=$min_lr_value
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

if (( ${#PRESET_VALUES[@]} > 1 || ${#BASIS_FAMILY_VALUES[@]} > 1 || ${#O_DEPTH_VALUES[@]} > 1 || ${#BATCH_SIZE_VALUES[@]} > 1 || ${#LEARNING_RATE_CODE_VALUES[@]} > 1 )); then
  echo "dreedle OWT grid: p=${PRESET_VALUES[*]} B=${BASIS_FAMILY_VALUES[*]} P=${O_DEPTH_VALUES[*]} b=${BATCH_SIZE_VALUES[*]} LR=${LEARNING_RATE_CODE_VALUES[*]}"
fi
for geometry_preset_value in "${PRESET_VALUES[@]}"; do
  if [[ "$geometry_preset_value" == dense ]]; then
    for batch_size_value in "${BATCH_SIZE_VALUES[@]}"; do
      for learning_rate_code in "${LEARNING_RATE_CODE_VALUES[@]}"; do
        run_grid_point "$geometry_preset_value" "${O_DEPTH_VALUES[0]}" "$batch_size_value" "$learning_rate_code" "${BASIS_FAMILY_VALUES[0]}" "${BASIS_TAG_VALUES[0]}"
      done
    done
  else
    for basis_index in "${!BASIS_FAMILY_VALUES[@]}"; do
      basis_family_value="${BASIS_FAMILY_VALUES[$basis_index]}"
      basis_tag="${BASIS_TAG_VALUES[$basis_index]}"
      for batch_size_value in "${BATCH_SIZE_VALUES[@]}"; do
        for learning_rate_code in "${LEARNING_RATE_CODE_VALUES[@]}"; do
          for o_depth_value in "${O_DEPTH_VALUES[@]}"; do
            run_grid_point "$geometry_preset_value" "$o_depth_value" "$batch_size_value" "$learning_rate_code" "$basis_family_value" "$basis_tag"
          done
        done
      done
    done
  fi
done
# ^^^ THOG
