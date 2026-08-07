#!/bin/bash
set -euo pipefail

# vvv THOG
# Current scruffy OpenWebText training wrapper for the PICTON compact-geometry contract.
# Scruffy runtime defaults: bfloat16, flash2. Dense baseline is available as -p dense.
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
#
# Registry:
#   geometry preset JPEG_LIKE_V1 segments MLP_UP along MLP_HIDDEN only.
#   -B selects the registered DEPTH/global basis independently.
#   --mlp-hidden-compressor selects the registered compressor inside each local segment.
#   --mlp-hidden-group-size sets physical MLP_HIDDEN positions per segment.
#   -Y remains the retained MLP_HIDDEN coefficient count; require 1 <= Y <= group size.
# ^^^ THOG

cd "$(dirname "$0")"

RUN_MODULE="run_thog2_owt"
HOST_LABEL="scruffy"
RUN_MODE="fresh"
RUN_NAME=""
# EXPERIMENT_PREFIX="${THOG2_EXPERIMENT_PREFIX:-NELSON}"                                                                                                  # <<< THOG removed redundant environment-controlled experiment naming
EXPERIMENT_PREFIX="NO_PREFIX"                                                                                                                            # <<< THOG -g now supplies the sole human run-name prefix
DATASET_NAME="openwebtext"
DATA_DIR="${THOG2_OWT_DATA_DIR:-data/openwebtext}"
CHECKPOINT_ROOT="checkpoints"
LOG_ROOT="logs"
RESULT_ROOT="results"
WANDB_ROOT="wandb"
GEOMETRY_PRESET="depth"
BASIS_FAMILY="chebyshev"
BASIS_VERSION="auto"
# vvv THOG coupled field machine HYPERBLOCK wrapper controls; long options avoid consuming the exhausted getopts alphabet
HYPERBLOCK=false
HYPERBLOCK_COMPRESSOR="chebyshev"
HYPERBLOCK_COMPRESSOR_VERSION="auto"
HYPERBLOCK_COMMON_FAMILY_ORDER=6
HYPERBLOCK_ATTENTION_FAMILY_ORDER=4
HYPERBLOCK_MLP_FAMILY_ORDER=2
HYPERBLOCK_DEPTH_ORDER=16
HYPERBLOCK_D_MODEL_ORDER=16
HYPERBLOCK_MLP_HIDDEN_ORDER=16
HYPERBLOCK_ATTENTION_HEAD_ORDER=16
HYPERBLOCK_ATTENTION_HEAD_CHANNEL_ORDER=16
HYPERBLOCK_MLP_HIDDEN_MULTIPLIER=4
HYPERBLOCK_LOOP_COUNT=1
HYPERBLOCK_LOOP_DECAY="1.0"
# ^^^ THOG
# vvv THOG PLASTIC DEPTH wrapper defaults preserve the established fixed DEPTH path
PLASTIC_ENABLED=false
PLASTIC_COARSE_PHASE="disabled"
PLASTIC_PHASE_1_N_STEPS=""
PLASTIC_PHASE_1_STARTING_LAYER_COUNT=""
PLASTIC_PHASE_1_NUMBER_OF_TRIALS=""
PLASTIC_PHASE_1_EVALUATION_STEPS_COUNT=""
PLASTIC_LAYER_COUNT_PROBE_EVERY_N_STEPS=""
PLASTIC_LAYER_COUNT_PROBE_NUMBER_OF_SAMPLED_VALID_TOKENS=1024
PLASTIC_LAYER_COUNT_PROBE_RADIUS="${THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS:-1}"
PLASTIC_LAYER_COUNT_MAX_STEP="${THOG2_PLASTIC_LAYER_COUNT_MAX_STEP:-1}"
PLASTIC_LAYER_COUNT_EXTRAPOLATION_WEIGHT="0.8"
PLASTIC_LAYERS_TO_SAMPLE=""
PLASTIC_DO_LEARN_LAYER_COUNT=false
PLASTIC_INITIAL_LAYER_COUNT=""
PLASTIC_MAX_PERMITTED_LAYERS=""
PLASTIC_LAYER_SAMPLING_INITIALISATION="equidistant"
PLASTIC_LAYER_COUNT_OBJECTIVE="lowest_loss"
PLASTIC_LAYER_COUNT_UPDATE_BRAKE=5
PLASTIC_LAYER_COUNT_PROBE_WINDOW_SIZE_AS_NUMBER_OF_PROBES=50
PLASTIC_LAYER_COUNT_PROBE_NOISE_LAMBDA="3.0"
PLASTIC_LAYER_COUNT_COST_WEIGHT="0.0"
PLASTIC_LAYER_MEMORY_BUDGET_GIB=""
PLASTIC_CUDA_ALLOCATOR_RESERVE_GIB="0.5"
PLASTIC_GEOMETRY_LEARNING_RATE_MULTIPLIER="0.1"
PLASTIC_FREEZE_GEOMETRY_DURING_WARMUP=true
PLASTIC_LOG_INTERVAL_COARSE=10
PLASTIC_COARSE_PHASE_ROLL_THROUGH=false
MAX_NONFINITE_UPDATE_SKIPS=99999
# ^^^ THOG
LAPPED_COSINE_WINDOW_LENGTH=36                                                                                                                             # <<< THOG default lapped locality scale
LAPPED_COSINE_OVERLAP_FRACTION="0.5"                                                                                                                       # <<< THOG v1 fixed overlap
MLP_HIDDEN_COMPRESSOR="${THOG2_MLP_HIDDEN_COMPRESSOR:-dct}"
MLP_HIDDEN_GROUP_SIZE="${THOG2_MLP_HIDDEN_GROUP_SIZE:-256}"
ATTENTION_GEOMETRY=""
MLP_GEOMETRY=""
STEPS=250
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
# vvv THOG layer-dropout wrapper controls; empty stratum/cardinality delegate all-active defaults to the runner
LAYER_DROPOUT_STRATUM_SIZE=""
LAYER_DROPOUT_ACTIVE_PER_STRATUM=""
LAYER_DROPOUT_RESAMPLE_STEPS=1
# ^^^ THOG
N_HEAD=12
N_EMBD=768
BLOCK_SIZE=1024
O_DEPTH=32
O_ATTN_D_MODEL=64
O_ATTN_QKV_PER_CHANNEL=6
O_ATTN_OUT_PER_CHANNEL=6
O_MLP_D_MODEL=64
O_MLP_HIDDEN=256
DEPTH_COMPRESS_LAYER_NORM_AND_BIAS=false                                                                                                                  # <<< THOG DEPTH-only repeated LayerNorm/bias participation switch
RESIDUAL_INIT_POLICY="depth_scaled"
RESIDUAL_INIT_DEPTH_SOURCE="dof_implied_depth"
RESIDUAL_INIT_DEPTH_VALUE=12
ACTIVATION_CHECKPOINTING=true
CHECKPOINT_SEGMENT_SIZE=12
FAST_DISCARD="${THOG2_FAST_DISCARD:-true}"
BYPASS_SEMANTIC_QKV_ADAPTER="${THOG2_BYPASS_SEMANTIC_QKV_ADAPTER:-true}"                                                                                  # <<< THOG default-on selectable semantic-QKV adapter bypass
DIRECT_FACTORISED_MLP="${THOG2_DIRECT_FACTORISED_MLP:-true}"                                                                                              # <<< THOG renamed default-on exact factorised MLP application
DIRECT_FACTORISED_HYPERBLOCK_MLP="${THOG2_DIRECT_FACTORISED_HYPERBLOCK_MLP:-false}"                                                                      # <<< THOG independent default-off direct HYPERBLOCK UP/DOWN application
VECTORISE_PER_HEAD_MATERIALISATION="${THOG2_VECTORISE_PER_HEAD_MATERIALISATION:-true}"                                                                    # <<< THOG default-on selectable per-head batching
DTYPE="bfloat16"
ATTENTION_BACKEND="flash2"
INSTRUMENTATION="tensorboard"
# vvv THOG depth-curve plotting is opt-in because it is diagnostic overhead
DEPTH_CURVE_PLOTS="${THOG2_DEPTH_CURVE_PLOTS:-none}"
# ^^^ THOG
DEPTH_CURVE_SAMPLE_ELEMENTS="${THOG2_DEPTH_CURVE_SAMPLE_ELEMENTS:-16384}"
DEPTH_CURVE_RENDERER="${THOG2_DEPTH_CURVE_RENDERER:-plotly}"
DEPTH_CURVE_LOCAL_HTML="${THOG2_DEPTH_CURVE_LOCAL_HTML:-true}"
DEPTH_CURVE_HTTP_PORT="${THOG2_DEPTH_CURVE_HTTP_PORT:-8787}"
DRY_RUN=false
N_LAYER_EXPLICIT=false
N_HEAD_EXPLICIT=false
N_EMBD_EXPLICIT=false

OPTIMIZER="${THOG2_OPTIMIZER:-adamw}"
OPTIMIZER_MOMENTUM="${THOG2_OPTIMIZER_MOMENTUM:-0.9}"
OPTIMIZER_LR_EXPLICIT=false
OPTIMIZER_MIN_LR_EXPLICIT=false

usage() {
  cat <<EOF_USAGE
Usage: $0 [options] [-- extra ${RUN_MODULE} args]

Model/run:
  -p PRESET=${GEOMETRY_PRESET}                       dense | legacy_sheet_col | depth | jpeg_like_v1 | head_aware_block | mlp_block | full_block
                                                   single value, comma list, or quoted space list
  -q RUN_MODE=${RUN_MODE}                        fresh | resume
  -g RUN_NAME=${RUN_NAME:-auto}
  -n STEPS=${STEPS}
  -b BATCH_SIZE=${BATCH_SIZE}                         single integer, comma list, or quoted space list
  -c LR_CODES=${LEARNING_RATE_CODES}                    1..1000; 70 means 7.0e-04 and 1000 means 1.0e-02; comma or quoted space list
  -f MIN_LR_CODE=${MIN_LR_CODE}                         1..100; 06 means 6.0e-05 and 100 means 1.0e-03
  -A GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}
  -G NUM_GPUS=${NUM_GPUS}
  -y OPTIMIZER=${OPTIMIZER}                         adamw | sgd | sgd_nesterov | adafactor | rmsprop
  --optimizer-momentum VALUE=${OPTIMIZER_MOMENTUM}  momentum for SGD/Nesterov/RMSprop
                                                     defaults: adamw 60/06; sgd 1000/100;
                                                     sgd_nesterov 1000/100; adafactor 1000/100;
                                                     rmsprop 100/10. Explicit -c/-f override independently.

Schedule/logging:
  --max-nonfinite-update-skips N=${MAX_NONFINITE_UPDATE_SKIPS}  tolerated skipped non-finite updates
  -u EVAL_ITERS=${EVAL_ITERS}
  -e EVAL_INTERVAL=${EVAL_INTERVAL}
  -l LOG_INTERVAL=${LOG_INTERVAL}
  -w WARMUP_ITERS=${WARMUP_ITERS}
  -k CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL}     0 disables periodic saves
  -I INSTRUMENTATION=${INSTRUMENTATION}             tensorboard | wandb | both | wandb_offline | none
  -F DEPTH_CURVE_PLOTS=${DEPTH_CURVE_PLOTS}         none | final | eval
  -N DEPTH_CURVE_SAMPLE_ELEMENTS=${DEPTH_CURVE_SAMPLE_ELEMENTS}
  -U DEPTH_CURVE_RENDERER=${DEPTH_CURVE_RENDERER}   matplotlib | plotly | both
  -V DEPTH_CURVE_LOCAL_HTML=${DEPTH_CURVE_LOCAL_HTML}  true | false

Systematic geometry (repeat --select-element as needed):
  --select-depth
  --select-element SELECTOR
  --option TARGET.PROPERTY=VALUE
  --explain-geometry

HYPERBLOCK:
  --hyperblock
  --hyperblock-compressor NAME=${HYPERBLOCK_COMPRESSOR}
  --hyperblock-compressor-version VERSION=${HYPERBLOCK_COMPRESSOR_VERSION}
  --hyperblock-common-family-order N=${HYPERBLOCK_COMMON_FAMILY_ORDER}
  --hyperblock-attention-family-order N=${HYPERBLOCK_ATTENTION_FAMILY_ORDER}
  --hyperblock-mlp-family-order N=${HYPERBLOCK_MLP_FAMILY_ORDER}
  --hyperblock-depth-order N=${HYPERBLOCK_DEPTH_ORDER}
  --hyperblock-d-model-order N=${HYPERBLOCK_D_MODEL_ORDER}
  --hyperblock-mlp-hidden-order N=${HYPERBLOCK_MLP_HIDDEN_ORDER}
  --hyperblock-attention-head-order N=${HYPERBLOCK_ATTENTION_HEAD_ORDER}
  --hyperblock-attention-head-channel-order N=${HYPERBLOCK_ATTENTION_HEAD_CHANNEL_ORDER}
  --hyperblock-mlp-hidden-multiplier N=${HYPERBLOCK_MLP_HIDDEN_MULTIPLIER}
  --hyperblock-loop-count N=${HYPERBLOCK_LOOP_COUNT}
  --hyperblock-loop-decay VALUE=${HYPERBLOCK_LOOP_DECAY}  exponential update decay in (0,1]
  --direct-factorised-hyperblock-mlp               enable exact UP/DOWN application without dense MLP matrices
  --no-direct-factorised-hyperblock-mlp            explicit default

PLASTIC DEPTH:
  --plastic__enabled | --no-plastic__enabled
  --plastic__coarse_phase enabled|disabled
  --plastic__phase_1_n_steps N
  --plastic__phase_1_starting_layer_count N
  --plastic__phase_1__number_of_trials N
  --plastic__phase_1_evaluation_steps_count N
  --plastic__layers_to_sample N                      fixed active count; defaults to N_LAYER
  --plastic__do_learn_layer_count | --no-plastic__do_learn_layer_count
  --plastic__initial_layer_count N                   initial learned count; defaults to N_LAYER
  --plastic__max_permitted_layers N                  required for learned count
  --plastic__layer_sampling_initialisation equidistant|random
  --plastic__layer_count_objective lowest_loss|layer_efficiency|relative_training_wall_time|memory_budget
  --plastic__layer_count_update_brake N=${PLASTIC_LAYER_COUNT_UPDATE_BRAKE}
  --plastic__layer_count_probe__probe_every_n_steps N=${PLASTIC_LAYER_COUNT_PROBE_EVERY_N_STEPS:-update brake}
  --plastic__layer_count_probe__number_of_sampled_valid_tokens N=${PLASTIC_LAYER_COUNT_PROBE_NUMBER_OF_SAMPLED_VALID_TOKENS}  0=all valid tokens in first probe microbatch
  --plastic__layer_count_probe_radius N=${PLASTIC_LAYER_COUNT_PROBE_RADIUS}
  --plastic__layer_count_max_step N=${PLASTIC_LAYER_COUNT_MAX_STEP}
  --plastic__layer_count_extrapolation_weight VALUE=${PLASTIC_LAYER_COUNT_EXTRAPOLATION_WEIGHT}
  --plastic__layer_count_probe__window_size_as_number_of_probes N=${PLASTIC_LAYER_COUNT_PROBE_WINDOW_SIZE_AS_NUMBER_OF_PROBES}
  --plastic__layer_count_probe_noise_lambda VALUE=${PLASTIC_LAYER_COUNT_PROBE_NOISE_LAMBDA}
  --plastic__layer_count_cost_weight VALUE=${PLASTIC_LAYER_COUNT_COST_WEIGHT}
  --plastic__layer_memory_budget_gib VALUE
  --plastic__cuda_allocator_reserve_gib VALUE=${PLASTIC_CUDA_ALLOCATOR_RESERVE_GIB}
  --plastic__geometry_learning_rate_multiplier VALUE=${PLASTIC_GEOMETRY_LEARNING_RATE_MULTIPLIER}
  --plastic__freeze_geometry_during_warmup | --no-plastic__freeze_geometry_during_warmup
  --plastic__log_interval_coarse N=${PLASTIC_LOG_INTERVAL_COARSE}
  --plastic__coarse_phase_roll_through | --no-plastic__coarse_phase_roll_through

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
  --mlp-hidden-compressor NAME=${MLP_HIDDEN_COMPRESSOR}  registered local compressor; comma/space list allowed
  --mlp-hidden-group-size N=${MLP_HIDDEN_GROUP_SIZE}     physical MLP_HIDDEN segment size; comma/space list allowed

Shape/runtime:
  -L N_LAYER=${N_LAYER}
  -s STRATUM_SIZE=${LAYER_DROPOUT_STRATUM_SIZE:-N_LAYER}                 layer-dropout nominal layers per stratum
  -M N_ACTIVE_PER_STRATUM=${LAYER_DROPOUT_ACTIVE_PER_STRATUM:-STRATUM_SIZE}     layer-dropout active layers selected per stratum
  --layer-dropout-resample-steps N=${LAYER_DROPOUT_RESAMPLE_STEPS}       optimizer updates per sampled layer set
  -H N_HEAD=${N_HEAD}
  -D N_EMBD=${N_EMBD}
  -C BLOCK_SIZE=${BLOCK_SIZE}
  -P O_DEPTH=${O_DEPTH}                             single integer, comma list, or quoted space list; ignored by dense
  -Q O_ATTN_D_MODEL=${O_ATTN_D_MODEL}
  -J O_ATTN_QKV_PER_CHANNEL=${O_ATTN_QKV_PER_CHANNEL}
  -O O_ATTN_OUT_PER_CHANNEL=${O_ATTN_OUT_PER_CHANNEL}
  -X O_MLP_D_MODEL=${O_MLP_D_MODEL}
  -Y O_MLP_HIDDEN=${O_MLP_HIDDEN}                    ignored by DEPTH
  --depth-compress-layer-norm-and-bias                   DEPTH only; default false
  --no-depth-compress-layer-norm-and-bias                DEPTH only; explicit default
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

# vvv THOG accept long optimizer, geometry, and layer-dropout controls without disturbing established short-option parsing
OPTIMIZER_FILTERED_ARGS=()
GEOMETRY_UI_EXTRA_ARGS=()
EXPLAIN_GEOMETRY=false
OPTIMIZER_SAW_SEPARATOR=false
while (( $# > 0 )); do
  if [[ "$OPTIMIZER_SAW_SEPARATOR" == true ]]; then
    OPTIMIZER_FILTERED_ARGS+=("$1")
    shift
    continue
  fi
  case "$1" in
    --select-depth)
      GEOMETRY_UI_EXTRA_ARGS+=("--select-depth")
      shift
      ;;
    --select-element|--option)
      (( $# >= 2 )) || { echo "$1 requires a value" >&2; exit 2; }
      GEOMETRY_UI_EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --select-element=*|--option=*)
      GEOMETRY_UI_EXTRA_ARGS+=("$1")
      shift
      ;;
    --explain-geometry)
      GEOMETRY_UI_EXTRA_ARGS+=("--explain-geometry")
      EXPLAIN_GEOMETRY=true
      shift
      ;;
    # vvv THOG direct HYPERBLOCK MLP is an independent Boolean execution option
    --direct-factorised-hyperblock-mlp)
      DIRECT_FACTORISED_HYPERBLOCK_MLP=true
      shift
      ;;
    --no-direct-factorised-hyperblock-mlp)
      DIRECT_FACTORISED_HYPERBLOCK_MLP=false
      shift
      ;;
    # ^^^ THOG
    # vvv THOG consume HYPERBLOCK long controls before getopts and build one collision-free canonical run
    --hyperblock)
      HYPERBLOCK=true
      shift
      ;;
    --no-hyperblock)
      HYPERBLOCK=false
      shift
      ;;
    --hyperblock-compressor|--hyperblock-compressor-version|--hyperblock-common-family-order|--hyperblock-attention-family-order|--hyperblock-mlp-family-order|--hyperblock-depth-order|--hyperblock-d-model-order|--hyperblock-mlp-hidden-order|--hyperblock-attention-head-order|--hyperblock-attention-head-channel-order|--hyperblock-mlp-hidden-multiplier|--hyperblock-loop-count|--hyperblock-loop-decay)
      (( $# >= 2 )) || { echo "$1 requires a value" >&2; exit 2; }
      case "$1" in
        --hyperblock-compressor) HYPERBLOCK_COMPRESSOR="$2" ;;
        --hyperblock-compressor-version) HYPERBLOCK_COMPRESSOR_VERSION="$2" ;;
        --hyperblock-common-family-order) HYPERBLOCK_COMMON_FAMILY_ORDER="$2" ;;
        --hyperblock-attention-family-order) HYPERBLOCK_ATTENTION_FAMILY_ORDER="$2" ;;
        --hyperblock-mlp-family-order) HYPERBLOCK_MLP_FAMILY_ORDER="$2" ;;
        --hyperblock-depth-order) HYPERBLOCK_DEPTH_ORDER="$2" ;;
        --hyperblock-d-model-order) HYPERBLOCK_D_MODEL_ORDER="$2" ;;
        --hyperblock-mlp-hidden-order) HYPERBLOCK_MLP_HIDDEN_ORDER="$2" ;;
        --hyperblock-attention-head-order) HYPERBLOCK_ATTENTION_HEAD_ORDER="$2" ;;
        --hyperblock-attention-head-channel-order) HYPERBLOCK_ATTENTION_HEAD_CHANNEL_ORDER="$2" ;;
        --hyperblock-mlp-hidden-multiplier) HYPERBLOCK_MLP_HIDDEN_MULTIPLIER="$2" ;;
        --hyperblock-loop-count) HYPERBLOCK_LOOP_COUNT="$2" ;;
        --hyperblock-loop-decay) HYPERBLOCK_LOOP_DECAY="$2" ;;
      esac
      shift 2
      ;;
    --hyperblock-compressor=*|--hyperblock-compressor-version=*|--hyperblock-common-family-order=*|--hyperblock-attention-family-order=*|--hyperblock-mlp-family-order=*|--hyperblock-depth-order=*|--hyperblock-d-model-order=*|--hyperblock-mlp-hidden-order=*|--hyperblock-attention-head-order=*|--hyperblock-attention-head-channel-order=*|--hyperblock-mlp-hidden-multiplier=*|--hyperblock-loop-count=*|--hyperblock-loop-decay=*)
      hyperblock_name="${1%%=*}"
      hyperblock_value="${1#*=}"
      case "$hyperblock_name" in
        --hyperblock-compressor) HYPERBLOCK_COMPRESSOR="$hyperblock_value" ;;
        --hyperblock-compressor-version) HYPERBLOCK_COMPRESSOR_VERSION="$hyperblock_value" ;;
        --hyperblock-common-family-order) HYPERBLOCK_COMMON_FAMILY_ORDER="$hyperblock_value" ;;
        --hyperblock-attention-family-order) HYPERBLOCK_ATTENTION_FAMILY_ORDER="$hyperblock_value" ;;
        --hyperblock-mlp-family-order) HYPERBLOCK_MLP_FAMILY_ORDER="$hyperblock_value" ;;
        --hyperblock-depth-order) HYPERBLOCK_DEPTH_ORDER="$hyperblock_value" ;;
        --hyperblock-d-model-order) HYPERBLOCK_D_MODEL_ORDER="$hyperblock_value" ;;
        --hyperblock-mlp-hidden-order) HYPERBLOCK_MLP_HIDDEN_ORDER="$hyperblock_value" ;;
        --hyperblock-attention-head-order) HYPERBLOCK_ATTENTION_HEAD_ORDER="$hyperblock_value" ;;
        --hyperblock-attention-head-channel-order) HYPERBLOCK_ATTENTION_HEAD_CHANNEL_ORDER="$hyperblock_value" ;;
        --hyperblock-mlp-hidden-multiplier) HYPERBLOCK_MLP_HIDDEN_MULTIPLIER="$hyperblock_value" ;;
        --hyperblock-loop-count) HYPERBLOCK_LOOP_COUNT="$hyperblock_value" ;;
        --hyperblock-loop-decay) HYPERBLOCK_LOOP_DECAY="$hyperblock_value" ;;
      esac
      unset hyperblock_name hyperblock_value
      shift
      ;;
    # ^^^ THOG
    # vvv THOG consume PLASTIC DEPTH controls before getopts and emit one canonical Python configuration
    --plastic__enabled) PLASTIC_ENABLED=true; shift ;;
    --plastic__coarse_phase_roll_through) PLASTIC_COARSE_PHASE_ROLL_THROUGH=true; shift ;;
    --no-plastic__coarse_phase_roll_through) PLASTIC_COARSE_PHASE_ROLL_THROUGH=false; shift ;;
    --no-plastic__enabled) PLASTIC_ENABLED=false; shift ;;
    --plastic__do_learn_layer_count) PLASTIC_DO_LEARN_LAYER_COUNT=true; shift ;;
    --no-plastic__do_learn_layer_count) PLASTIC_DO_LEARN_LAYER_COUNT=false; shift ;;
    --plastic__freeze_geometry_during_warmup) PLASTIC_FREEZE_GEOMETRY_DURING_WARMUP=true; shift ;;
    --no-plastic__freeze_geometry_during_warmup) PLASTIC_FREEZE_GEOMETRY_DURING_WARMUP=false; shift ;;
    # --plastic__log_interval_coarse|--plastic__layers_to_sample|--plastic__initial_layer_count|--plastic__max_permitted_layers|--plastic__layer_sampling_initialisation|--plastic__layer_count_objective|--plastic__layer_count_update_brake|--plastic__layer_count_probe__window_size_as_number_of_probes|--plastic__layer_count_probe_noise_lambda|--plastic__layer_count_extrapolation_weight|--plastic__layer_count_cost_weight|--plastic__layer_memory_budget_gib|--plastic__geometry_learning_rate_multiplier)
    --plastic__coarse_phase|--plastic__phase_1_n_steps|--plastic__phase_1_starting_layer_count|--plastic__phase_1__number_of_trials|--plastic__phase_1_evaluation_steps_count|--plastic__log_interval_coarse|--plastic__layers_to_sample|--plastic__initial_layer_count|--plastic__max_permitted_layers|--plastic__layer_sampling_initialisation|--plastic__layer_count_objective|--plastic__layer_count_update_brake|--plastic__layer_count_probe__probe_every_n_steps|--plastic__layer_count_probe__number_of_sampled_valid_tokens|--plastic__layer_count_probe_radius|--plastic__layer_count_max_step|--plastic__layer_count_probe__window_size_as_number_of_probes|--plastic__layer_count_probe_noise_lambda|--plastic__layer_count_extrapolation_weight|--plastic__layer_count_cost_weight|--plastic__layer_memory_budget_gib|--plastic__cuda_allocator_reserve_gib|--plastic__geometry_learning_rate_multiplier)
      (( $# >= 2 )) || { echo "$1 requires a value" >&2; exit 2; }
      case "$1" in
        --plastic__coarse_phase) PLASTIC_COARSE_PHASE="$2" ;;
        --plastic__phase_1_n_steps) PLASTIC_PHASE_1_N_STEPS="$2" ;;
        --plastic__phase_1_starting_layer_count) PLASTIC_PHASE_1_STARTING_LAYER_COUNT="$2" ;;
        --plastic__phase_1__number_of_trials) PLASTIC_PHASE_1_NUMBER_OF_TRIALS="$2" ;;
        --plastic__phase_1_evaluation_steps_count) PLASTIC_PHASE_1_EVALUATION_STEPS_COUNT="$2" ;;
        --plastic__log_interval_coarse) PLASTIC_LOG_INTERVAL_COARSE="$2" ;;
        --plastic__layer_count_probe__probe_every_n_steps) PLASTIC_LAYER_COUNT_PROBE_EVERY_N_STEPS="$2" ;;
        --plastic__layer_count_probe__number_of_sampled_valid_tokens) PLASTIC_LAYER_COUNT_PROBE_NUMBER_OF_SAMPLED_VALID_TOKENS="$2" ;;
        --plastic__layer_count_probe_radius) PLASTIC_LAYER_COUNT_PROBE_RADIUS="$2" ;;
        --plastic__layer_count_max_step) PLASTIC_LAYER_COUNT_MAX_STEP="$2" ;;
        --plastic__layers_to_sample) PLASTIC_LAYERS_TO_SAMPLE="$2" ;;
        --plastic__initial_layer_count) PLASTIC_INITIAL_LAYER_COUNT="$2" ;;
        --plastic__max_permitted_layers) PLASTIC_MAX_PERMITTED_LAYERS="$2" ;;
        --plastic__layer_sampling_initialisation) PLASTIC_LAYER_SAMPLING_INITIALISATION="$2" ;;
        --plastic__layer_count_objective) PLASTIC_LAYER_COUNT_OBJECTIVE="$2" ;;
        --plastic__layer_count_update_brake) PLASTIC_LAYER_COUNT_UPDATE_BRAKE="$2" ;;
        --plastic__layer_count_probe__window_size_as_number_of_probes) PLASTIC_LAYER_COUNT_PROBE_WINDOW_SIZE_AS_NUMBER_OF_PROBES="$2" ;;
        --plastic__layer_count_probe_noise_lambda) PLASTIC_LAYER_COUNT_PROBE_NOISE_LAMBDA="$2" ;;
        --plastic__layer_count_extrapolation_weight) PLASTIC_LAYER_COUNT_EXTRAPOLATION_WEIGHT="$2" ;;
        --plastic__layer_count_cost_weight) PLASTIC_LAYER_COUNT_COST_WEIGHT="$2" ;;
        --plastic__layer_memory_budget_gib) PLASTIC_LAYER_MEMORY_BUDGET_GIB="$2" ;;
        --plastic__cuda_allocator_reserve_gib) PLASTIC_CUDA_ALLOCATOR_RESERVE_GIB="$2" ;;
        --plastic__geometry_learning_rate_multiplier) PLASTIC_GEOMETRY_LEARNING_RATE_MULTIPLIER="$2" ;;
      esac
      shift 2
      ;;
    # --plastic__log_interval_coarse=*|--plastic__layers_to_sample=*|--plastic__initial_layer_count=*|--plastic__max_permitted_layers=*|--plastic__layer_sampling_initialisation=*|--plastic__layer_count_objective=*|--plastic__layer_count_update_brake=*|--plastic__layer_count_probe__window_size_as_number_of_probes=*|--plastic__layer_count_probe_noise_lambda=*|--plastic__layer_count_extrapolation_weight=*|--plastic__layer_count_cost_weight=*|--plastic__layer_memory_budget_gib=*|--plastic__geometry_learning_rate_multiplier=*)
    --plastic__coarse_phase=*|--plastic__phase_1_n_steps=*|--plastic__phase_1_starting_layer_count=*|--plastic__phase_1__number_of_trials=*|--plastic__phase_1_evaluation_steps_count=*|--plastic__log_interval_coarse=*|--plastic__layers_to_sample=*|--plastic__initial_layer_count=*|--plastic__max_permitted_layers=*|--plastic__layer_sampling_initialisation=*|--plastic__layer_count_objective=*|--plastic__layer_count_update_brake=*|--plastic__layer_count_probe__probe_every_n_steps=*|--plastic__layer_count_probe__number_of_sampled_valid_tokens=*|--plastic__layer_count_probe_radius=*|--plastic__layer_count_max_step=*|--plastic__layer_count_probe__window_size_as_number_of_probes=*|--plastic__layer_count_probe_noise_lambda=*|--plastic__layer_count_extrapolation_weight=*|--plastic__layer_count_cost_weight=*|--plastic__layer_memory_budget_gib=*|--plastic__cuda_allocator_reserve_gib=*|--plastic__geometry_learning_rate_multiplier=*)
      plastic_name="${1%%=*}"; plastic_value="${1#*=}"
      case "$plastic_name" in
        --plastic__coarse_phase) PLASTIC_COARSE_PHASE="$plastic_value" ;;
        --plastic__phase_1_n_steps) PLASTIC_PHASE_1_N_STEPS="$plastic_value" ;;
        --plastic__phase_1_starting_layer_count) PLASTIC_PHASE_1_STARTING_LAYER_COUNT="$plastic_value" ;;
        --plastic__phase_1__number_of_trials) PLASTIC_PHASE_1_NUMBER_OF_TRIALS="$plastic_value" ;;
        --plastic__phase_1_evaluation_steps_count) PLASTIC_PHASE_1_EVALUATION_STEPS_COUNT="$plastic_value" ;;
        --plastic__log_interval_coarse) PLASTIC_LOG_INTERVAL_COARSE="$plastic_value" ;;
        --plastic__layer_count_probe__probe_every_n_steps) PLASTIC_LAYER_COUNT_PROBE_EVERY_N_STEPS="$plastic_value" ;;
        --plastic__layer_count_probe__number_of_sampled_valid_tokens) PLASTIC_LAYER_COUNT_PROBE_NUMBER_OF_SAMPLED_VALID_TOKENS="$plastic_value" ;;
        --plastic__layer_count_probe_radius) PLASTIC_LAYER_COUNT_PROBE_RADIUS="$plastic_value" ;;
        --plastic__layer_count_max_step) PLASTIC_LAYER_COUNT_MAX_STEP="$plastic_value" ;;
        --plastic__layers_to_sample) PLASTIC_LAYERS_TO_SAMPLE="$plastic_value" ;;
        --plastic__initial_layer_count) PLASTIC_INITIAL_LAYER_COUNT="$plastic_value" ;;
        --plastic__max_permitted_layers) PLASTIC_MAX_PERMITTED_LAYERS="$plastic_value" ;;
        --plastic__layer_sampling_initialisation) PLASTIC_LAYER_SAMPLING_INITIALISATION="$plastic_value" ;;
        --plastic__layer_count_objective) PLASTIC_LAYER_COUNT_OBJECTIVE="$plastic_value" ;;
        --plastic__layer_count_update_brake) PLASTIC_LAYER_COUNT_UPDATE_BRAKE="$plastic_value" ;;
        --plastic__layer_count_probe__window_size_as_number_of_probes) PLASTIC_LAYER_COUNT_PROBE_WINDOW_SIZE_AS_NUMBER_OF_PROBES="$plastic_value" ;;
        --plastic__layer_count_probe_noise_lambda) PLASTIC_LAYER_COUNT_PROBE_NOISE_LAMBDA="$plastic_value" ;;
        --plastic__layer_count_extrapolation_weight) PLASTIC_LAYER_COUNT_EXTRAPOLATION_WEIGHT="$plastic_value" ;;
        --plastic__layer_count_cost_weight) PLASTIC_LAYER_COUNT_COST_WEIGHT="$plastic_value" ;;
        --plastic__layer_memory_budget_gib) PLASTIC_LAYER_MEMORY_BUDGET_GIB="$plastic_value" ;;
        --plastic__cuda_allocator_reserve_gib) PLASTIC_CUDA_ALLOCATOR_RESERVE_GIB="$plastic_value" ;;
        --plastic__geometry_learning_rate_multiplier) PLASTIC_GEOMETRY_LEARNING_RATE_MULTIPLIER="$plastic_value" ;;
      esac
      unset plastic_name plastic_value
      shift
      ;;
    # ^^^ THOG
    --depth-compress-layer-norm-and-bias)
      DEPTH_COMPRESS_LAYER_NORM_AND_BIAS=true
      shift
      ;;
    --no-depth-compress-layer-norm-and-bias)
      DEPTH_COMPRESS_LAYER_NORM_AND_BIAS=false
      shift
      ;;
    --layer-dropout-stratum-size)
      (( $# >= 2 )) || { echo "--layer-dropout-stratum-size requires a positive integer" >&2; exit 2; }
      LAYER_DROPOUT_STRATUM_SIZE="$2"
      shift 2
      ;;
    --layer-dropout-stratum-size=*)
      LAYER_DROPOUT_STRATUM_SIZE="${1#*=}"
      shift
      ;;
    --layer-dropout-active-per-stratum)
      (( $# >= 2 )) || { echo "--layer-dropout-active-per-stratum requires a positive integer" >&2; exit 2; }
      LAYER_DROPOUT_ACTIVE_PER_STRATUM="$2"
      shift 2
      ;;
    --layer-dropout-active-per-stratum=*)
      LAYER_DROPOUT_ACTIVE_PER_STRATUM="${1#*=}"
      shift
      ;;
    --layer-dropout-resample-steps)
      (( $# >= 2 )) || { echo "--layer-dropout-resample-steps requires a positive integer" >&2; exit 2; }
      LAYER_DROPOUT_RESAMPLE_STEPS="$2"
      shift 2
      ;;
    --layer-dropout-resample-steps=*)
      LAYER_DROPOUT_RESAMPLE_STEPS="${1#*=}"
      shift
      ;;
    --max-nonfinite-update-skips)
      (( $# >= 2 )) || { echo "--max-nonfinite-update-skips requires a non-negative integer" >&2; exit 2; }
      MAX_NONFINITE_UPDATE_SKIPS="$2"
      shift 2
      ;;
    --max-nonfinite-update-skips=*)
      MAX_NONFINITE_UPDATE_SKIPS="${1#*=}"
      shift
      ;;
    --optimizer)
      (( $# >= 2 )) || { echo "--optimizer requires a name" >&2; exit 2; }
      OPTIMIZER="$2"
      shift 2
      ;;
    --optimizer=*)
      OPTIMIZER="${1#*=}"
      shift
      ;;
    --optimizer-momentum)
      (( $# >= 2 )) || { echo "--optimizer-momentum requires a numeric value" >&2; exit 2; }
      OPTIMIZER_MOMENTUM="$2"
      shift 2
      ;;
    --optimizer-momentum=*)
      OPTIMIZER_MOMENTUM="${1#*=}"
      shift
      ;;
    --mlp-hidden-compressor)
      (( $# >= 2 )) || { echo "--mlp-hidden-compressor requires a registry name" >&2; exit 2; }
      MLP_HIDDEN_COMPRESSOR="$2"
      shift 2
      ;;
    --mlp-hidden-compressor=*)
      MLP_HIDDEN_COMPRESSOR="${1#*=}"
      shift
      ;;
    --mlp-hidden-group-size)
      (( $# >= 2 )) || { echo "--mlp-hidden-group-size requires a positive integer or list" >&2; exit 2; }
      MLP_HIDDEN_GROUP_SIZE="$2"
      shift 2
      ;;
    --mlp-hidden-group-size=*)
      MLP_HIDDEN_GROUP_SIZE="${1#*=}"
      shift
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
# ^^^ THOG

while getopts ":q:g:n:b:c:f:y:A:G:u:e:l:w:k:I:F:N:U:V:p:B:v:W:i:a:m:L:s:M:H:D:C:P:Q:J:O:X:Y:S:E:T:K:r:z:Z:d:t:o:j:R:x:h" option; do
  case "$option" in
    q) RUN_MODE="$OPTARG" ;; g) RUN_NAME="$OPTARG" ;;
    n) STEPS="$OPTARG" ;; b) BATCH_SIZE="$OPTARG" ;; c) LEARNING_RATE_CODES="$OPTARG"; OPTIMIZER_LR_EXPLICIT=true ;; f) MIN_LR_CODE="$OPTARG"; OPTIMIZER_MIN_LR_EXPLICIT=true ;; y) OPTIMIZER="$OPTARG" ;; A) GRADIENT_ACCUMULATION_STEPS="$OPTARG" ;; G) NUM_GPUS="$OPTARG" ;;
    u) EVAL_ITERS="$OPTARG" ;; e) EVAL_INTERVAL="$OPTARG" ;; l) LOG_INTERVAL="$OPTARG" ;; w) WARMUP_ITERS="$OPTARG" ;; k) CHECKPOINT_INTERVAL="$OPTARG" ;;
    I) INSTRUMENTATION="$OPTARG" ;; F) DEPTH_CURVE_PLOTS="$OPTARG" ;; N) DEPTH_CURVE_SAMPLE_ELEMENTS="$OPTARG" ;; U) DEPTH_CURVE_RENDERER="$OPTARG" ;; V) DEPTH_CURVE_LOCAL_HTML="$OPTARG" ;;
    p) GEOMETRY_PRESET="$OPTARG" ;; B) BASIS_FAMILY="$OPTARG" ;; v) BASIS_VERSION="$OPTARG" ;; W) LAPPED_COSINE_WINDOW_LENGTH="$OPTARG" ;; i) LAPPED_COSINE_OVERLAP_FRACTION="$OPTARG" ;; a) ATTENTION_GEOMETRY="$OPTARG" ;; m) MLP_GEOMETRY="$OPTARG" ;;
    L) N_LAYER="$OPTARG"; N_LAYER_EXPLICIT=true ;; s) LAYER_DROPOUT_STRATUM_SIZE="$OPTARG" ;; M) LAYER_DROPOUT_ACTIVE_PER_STRATUM="$OPTARG" ;; H) N_HEAD="$OPTARG"; N_HEAD_EXPLICIT=true ;; D) N_EMBD="$OPTARG"; N_EMBD_EXPLICIT=true ;;
    C) BLOCK_SIZE="$OPTARG" ;; P) O_DEPTH="$OPTARG" ;; Q) O_ATTN_D_MODEL="$OPTARG" ;; J) O_ATTN_QKV_PER_CHANNEL="$OPTARG" ;; O) O_ATTN_OUT_PER_CHANNEL="$OPTARG" ;; X) O_MLP_D_MODEL="$OPTARG" ;; Y) O_MLP_HIDDEN="$OPTARG" ;; S) CHECKPOINT_SEGMENT_SIZE="$OPTARG" ;;
    E) FAST_DISCARD="$OPTARG" ;; T) DTYPE="$OPTARG" ;; K) ATTENTION_BACKEND="$OPTARG" ;; r) RESIDUAL_INIT_POLICY="$OPTARG" ;; z) RESIDUAL_INIT_DEPTH_SOURCE="$OPTARG" ;; Z) RESIDUAL_INIT_DEPTH_VALUE="$OPTARG" ;;
    d) DATASET_NAME="$OPTARG"; DATA_DIR="data/$OPTARG" ;; t) DATA_DIR="$OPTARG" ;; o) CHECKPOINT_ROOT="$OPTARG" ;; j) LOG_ROOT="$OPTARG" ;; R) RESULT_ROOT="$OPTARG" ;; x) DRY_RUN="$OPTARG" ;;
    h) usage; exit 0 ;; :) echo "Option -$OPTARG requires an argument." >&2; exit 2 ;; \?) echo "Unknown option: -$OPTARG" >&2; usage >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
if [[ "${1:-}" == "--" ]]; then shift; fi
EXTRA_ARGS=("$@")
EXTRA_ARGS+=("${GEOMETRY_UI_EXTRA_ARGS[@]}")

# vvv THOG normalize optimizer and apply its LR defaults only when -c/-f were omitted
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
# vvv THOG PLASTIC DEPTH CUDA reserve accepts any finite non-negative scalar GiB value
validate_nonnegative_number() {
  awk -v value="$1" 'BEGIN { numeric = value + 0; exit !(value != "" && numeric >= 0.0 && numeric == numeric && numeric < 1.0e100) }' || {
    echo "Invalid $2: $1; expected a finite non-negative number." >&2
    exit 2
  }
}
# ^^^ THOG
# vvv THOG HYPERBLOCK recurrence decay is one finite scalar in the closed upper interval
validate_open_closed_unit_float() {
  awk -v value="$1" 'BEGIN { numeric = value + 0; exit !(value != "" && numeric > 0.0 && numeric <= 1.0) }' || {
    echo "Invalid $2: $1; expected a numeric value in (0, 1]." >&2
    exit 2
  }
}
# ^^^ THOG
validate_true_false() { case "$1" in true|false) ;; *) echo "Invalid $2: $1; expected true or false." >&2; exit 2 ;; esac; }

O_DEPTH_VALUES=()
PRESET_VALUES=()
BASIS_FAMILY_VALUES=()                                                                                                                                    # <<< THOG basis-family grid axis
BASIS_TAG_VALUES=()                                                                                                                                       # <<< THOG matching artifact tags for basis-family grid
MLP_HIDDEN_COMPRESSOR_VALUES=()
MLP_HIDDEN_COMPRESSOR_TAG_VALUES=()
MLP_HIDDEN_GROUP_SIZE_VALUES=()
BATCH_SIZE_VALUES=()                                                                                                                                        # <<< THOG batch grid axis
LEARNING_RATE_CODE_VALUES=()                                                                                                                                # <<< THOG LR grid axis
HAS_DENSE_PRESET=false
HAS_COMPACT_PRESET=false
HAS_JPEG_LIKE_PRESET=false
HAS_NON_DEPTH_COMPACT_PRESET=false                                                                                                                         # <<< THOG dead Q/J/O/X/Y controls must not constrain pure DEPTH runs
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
      depth) PRESET_VALUES+=("$value"); HAS_COMPACT_PRESET=true ;;
      legacy_sheet_col|head_aware_block|mlp_block|full_block) PRESET_VALUES+=("$value"); HAS_COMPACT_PRESET=true; HAS_NON_DEPTH_COMPACT_PRESET=true ;;
      jpeg_like_v1) PRESET_VALUES+=("$value"); HAS_COMPACT_PRESET=true; HAS_NON_DEPTH_COMPACT_PRESET=true; HAS_JPEG_LIKE_PRESET=true ;;
      hyperblock) PRESET_VALUES+=("$value"); HAS_COMPACT_PRESET=true; HAS_NON_DEPTH_COMPACT_PRESET=true ;;
      *) echo "Bad PRESET: $value" >&2; exit 2 ;;
    esac
  done
  (( ${#PRESET_VALUES[@]} > 0 )) || { echo "Invalid PRESET: empty value list." >&2; exit 2; }
}
parse_mlp_hidden_group_size_values() {
  local normalized="${1//,/ }" value
  for value in $normalized; do
    validate_positive_uint "$value" "MLP_HIDDEN_GROUP_SIZE"
    MLP_HIDDEN_GROUP_SIZE_VALUES+=("$value")
  done
  (( ${#MLP_HIDDEN_GROUP_SIZE_VALUES[@]} > 0 )) || { echo "Invalid MLP_HIDDEN_GROUP_SIZE list." >&2; exit 2; }
}
parse_mlp_hidden_compressor_values() {
  local normalized="${1//,/ }" value
  for value in $normalized; do
    [[ "$value" =~ ^[a-z][a-z0-9_]*$ ]] || { echo "Invalid MLP_HIDDEN_COMPRESSOR value: $value; expected a lowercase registry name or alias." >&2; exit 2; }
    MLP_HIDDEN_COMPRESSOR_VALUES+=("$value")
  done
  (( ${#MLP_HIDDEN_COMPRESSOR_VALUES[@]} > 0 )) || { echo "Invalid MLP_HIDDEN_COMPRESSOR list." >&2; exit 2; }
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
# vvv THOG --hyperblock selects exactly one architecture-wide topology rather than joining the legacy preset grid
if [[ "$HYPERBLOCK" == true ]]; then
  PRESET_VALUES=(hyperblock)
  BASIS_FAMILY="chebyshev"
  HAS_DENSE_PRESET=false
  HAS_COMPACT_PRESET=true
  HAS_JPEG_LIKE_PRESET=false
  HAS_NON_DEPTH_COMPACT_PRESET=true
fi
# ^^^ THOG
parse_basis_family_values "$BASIS_FAMILY"                                                                                                                  # <<< THOG parse basis-family grid
parse_mlp_hidden_compressor_values "$MLP_HIDDEN_COMPRESSOR"
parse_mlp_hidden_group_size_values "$MLP_HIDDEN_GROUP_SIZE"
parse_positive_uint_values "$BATCH_SIZE" "BATCH_SIZE"                                                                                                  # <<< THOG parse batch grid
parse_lr_code_values "$LEARNING_RATE_CODES"                                                                                                              # <<< THOG parse LR grid
validate_lr_code "$MIN_LR_CODE" "MIN_LR_CODE" 100                                                                                                          # <<< THOG validate min LR
# vvv THOG validate scalar layer-dropout wrapper controls before runner construction
[[ -z "$LAYER_DROPOUT_STRATUM_SIZE" ]] || validate_positive_uint "$LAYER_DROPOUT_STRATUM_SIZE" "STRATUM_SIZE"
[[ -z "$LAYER_DROPOUT_ACTIVE_PER_STRATUM" ]] || validate_positive_uint "$LAYER_DROPOUT_ACTIVE_PER_STRATUM" "N_ACTIVE_PER_STRATUM"
validate_positive_uint "$LAYER_DROPOUT_RESAMPLE_STEPS" "LAYER_DROPOUT_RESAMPLE_STEPS"
# ^^^ THOG
# vvv THOG validate PLASTIC DEPTH wrapper controls before runner construction
validate_true_false "$PLASTIC_ENABLED" "PLASTIC_ENABLED"
validate_true_false "$PLASTIC_DO_LEARN_LAYER_COUNT" "PLASTIC_DO_LEARN_LAYER_COUNT"
validate_true_false "$PLASTIC_FREEZE_GEOMETRY_DURING_WARMUP" "PLASTIC_FREEZE_GEOMETRY_DURING_WARMUP"
validate_true_false "$PLASTIC_COARSE_PHASE_ROLL_THROUGH" "PLASTIC_COARSE_PHASE_ROLL_THROUGH"
validate_positive_uint "$PLASTIC_LOG_INTERVAL_COARSE" "PLASTIC_LOG_INTERVAL_COARSE"
case "$PLASTIC_COARSE_PHASE" in enabled|disabled) ;; *) echo "PLASTIC_COARSE_PHASE must be enabled or disabled." >&2; exit 2 ;; esac
[[ -z "$PLASTIC_PHASE_1_N_STEPS" ]] || validate_positive_uint "$PLASTIC_PHASE_1_N_STEPS" "PLASTIC_PHASE_1_N_STEPS"
[[ -z "$PLASTIC_PHASE_1_STARTING_LAYER_COUNT" ]] || validate_positive_uint "$PLASTIC_PHASE_1_STARTING_LAYER_COUNT" "PLASTIC_PHASE_1_STARTING_LAYER_COUNT"
[[ -z "$PLASTIC_PHASE_1_NUMBER_OF_TRIALS" ]] || validate_positive_uint "$PLASTIC_PHASE_1_NUMBER_OF_TRIALS" "PLASTIC_PHASE_1_NUMBER_OF_TRIALS"
[[ -z "$PLASTIC_PHASE_1_EVALUATION_STEPS_COUNT" ]] || validate_positive_uint "$PLASTIC_PHASE_1_EVALUATION_STEPS_COUNT" "PLASTIC_PHASE_1_EVALUATION_STEPS_COUNT"
[[ -z "$PLASTIC_LAYER_COUNT_PROBE_EVERY_N_STEPS" ]] || validate_positive_uint "$PLASTIC_LAYER_COUNT_PROBE_EVERY_N_STEPS" "PLASTIC_LAYER_COUNT_PROBE_EVERY_N_STEPS"
validate_nonnegative_uint "$PLASTIC_LAYER_COUNT_PROBE_NUMBER_OF_SAMPLED_VALID_TOKENS" "PLASTIC_LAYER_COUNT_PROBE_NUMBER_OF_SAMPLED_VALID_TOKENS"
validate_positive_uint "$PLASTIC_LAYER_COUNT_PROBE_RADIUS" "PLASTIC_LAYER_COUNT_PROBE_RADIUS"
validate_positive_uint "$PLASTIC_LAYER_COUNT_MAX_STEP" "PLASTIC_LAYER_COUNT_MAX_STEP"
validate_nonnegative_uint "$MAX_NONFINITE_UPDATE_SKIPS" "MAX_NONFINITE_UPDATE_SKIPS"
[[ -z "$PLASTIC_LAYERS_TO_SAMPLE" ]] || validate_positive_uint "$PLASTIC_LAYERS_TO_SAMPLE" "PLASTIC_LAYERS_TO_SAMPLE"
[[ -z "$PLASTIC_INITIAL_LAYER_COUNT" ]] || validate_positive_uint "$PLASTIC_INITIAL_LAYER_COUNT" "PLASTIC_INITIAL_LAYER_COUNT"
[[ -z "$PLASTIC_MAX_PERMITTED_LAYERS" ]] || validate_positive_uint "$PLASTIC_MAX_PERMITTED_LAYERS" "PLASTIC_MAX_PERMITTED_LAYERS"
validate_nonnegative_uint "$PLASTIC_LAYER_COUNT_UPDATE_BRAKE" "PLASTIC_LAYER_COUNT_UPDATE_BRAKE"
validate_positive_uint "$PLASTIC_LAYER_COUNT_PROBE_WINDOW_SIZE_AS_NUMBER_OF_PROBES" "PLASTIC_LAYER_COUNT_PROBE_WINDOW_SIZE_AS_NUMBER_OF_PROBES"
case "$PLASTIC_LAYER_SAMPLING_INITIALISATION" in equidistant|random) ;; *) echo "PLASTIC_LAYER_SAMPLING_INITIALISATION must be equidistant or random." >&2; exit 2 ;; esac
case "$PLASTIC_LAYER_COUNT_OBJECTIVE" in lowest_loss|layer_efficiency|relative_training_wall_time|memory_budget) ;; *) echo "Bad PLASTIC_LAYER_COUNT_OBJECTIVE: $PLASTIC_LAYER_COUNT_OBJECTIVE" >&2; exit 2 ;; esac
if [[ "$PLASTIC_COARSE_PHASE" == enabled ]]; then
  [[ "$PLASTIC_ENABLED" == true ]] || { echo "--plastic__coarse_phase enabled requires --plastic__enabled." >&2; exit 2; }
  [[ "$PLASTIC_DO_LEARN_LAYER_COUNT" == true ]] || { echo "--plastic__coarse_phase enabled requires --plastic__do_learn_layer_count." >&2; exit 2; }
  [[ -n "$PLASTIC_PHASE_1_N_STEPS" && -n "$PLASTIC_PHASE_1_STARTING_LAYER_COUNT" && -n "$PLASTIC_PHASE_1_NUMBER_OF_TRIALS" && -n "$PLASTIC_PHASE_1_EVALUATION_STEPS_COUNT" ]] || { echo "enabled COARSE requires every plastic__phase_1 control." >&2; exit 2; }
fi
if [[ "$PLASTIC_DO_LEARN_LAYER_COUNT" == true ]]; then
  [[ -z "$PLASTIC_LAYERS_TO_SAMPLE" ]] || { echo "--plastic__layers_to_sample conflicts with learned layer count." >&2; exit 2; }
  [[ -n "$PLASTIC_MAX_PERMITTED_LAYERS" ]] || { echo "--plastic__max_permitted_layers is required for learned layer count." >&2; exit 2; }
else
  [[ -z "$PLASTIC_INITIAL_LAYER_COUNT" && -z "$PLASTIC_MAX_PERMITTED_LAYERS" ]] || { echo "initial/max layer count controls require --plastic__do_learn_layer_count." >&2; exit 2; }
fi
[[ "$PLASTIC_LAYER_COUNT_OBJECTIVE" != memory_budget || -n "$PLASTIC_LAYER_MEMORY_BUDGET_GIB" ]] || { echo "memory_budget requires --plastic__layer_memory_budget_gib." >&2; exit 2; }
python - "$PLASTIC_LAYER_COUNT_EXTRAPOLATION_WEIGHT" <<'PY'
import math, sys
value = float(sys.argv[1])
if not math.isfinite(value) or not (0.5 < value <= 1.0):
    raise SystemExit("PLASTIC_LAYER_COUNT_EXTRAPOLATION_WEIGHT must lie in (0.5, 1.0]")
PY
validate_nonnegative_number "$PLASTIC_CUDA_ALLOCATOR_RESERVE_GIB" "PLASTIC_CUDA_ALLOCATOR_RESERVE_GIB"
if [[ "$PLASTIC_ENABLED" == true ]]; then
  [[ "$HYPERBLOCK" == false ]] || { echo "PLASTIC DEPTH may not be combined with HYPERBLOCK." >&2; exit 2; }
  [[ "$HAS_NON_DEPTH_COMPACT_PRESET" == false && "$HAS_DENSE_PRESET" == false ]] || { echo "PLASTIC DEPTH requires every selected preset to be depth." >&2; exit 2; }
  [[ "$LAYER_DROPOUT_ACTIVE_PER_STRATUM" == "" || "$LAYER_DROPOUT_ACTIVE_PER_STRATUM" == "${LAYER_DROPOUT_STRATUM_SIZE:-$N_LAYER}" ]] || { echo "PLASTIC DEPTH v0.1 may not be combined with layer dropout." >&2; exit 2; }
fi
# ^^^ THOG

case "$RUN_MODE" in fresh|resume) ;; *) echo "RUN_MODE must be fresh or resume." >&2; exit 2 ;; esac
# vvv THOG
if [[ "$HAS_JPEG_LIKE_PRESET" == false ]] && (( ${#MLP_HIDDEN_COMPRESSOR_VALUES[@]} > 1 || ${#MLP_HIDDEN_GROUP_SIZE_VALUES[@]} > 1 )); then
  echo "MLP_HIDDEN compressor/group grids require -p jpeg_like_v1." >&2
  exit 2
fi
if (( ${#BASIS_FAMILY_VALUES[@]} > 1 )) && [[ "$BASIS_VERSION" != auto ]]; then
  echo "BASIS_VERSION must be auto when BASIS_FAMILY contains multiple values." >&2
  exit 2
fi
# ^^^ THOG
case "$ATTENTION_BACKEND" in auto|flash2|sdpa|math) ;; *) echo "Bad ATTENTION_BACKEND: $ATTENTION_BACKEND" >&2; exit 2 ;; esac
# vvv THOG one instrumentation selector determines backend and W&B mode; legacy instrumentation -M/-W meanings are retired
case "$INSTRUMENTATION" in
  tensorboard) INSTRUMENTATION_BACKEND="tensorboard"; WANDB_FLAG="--no-wandb"; WANDB_MODE="disabled" ;;
  wandb) INSTRUMENTATION_BACKEND="wandb"; WANDB_FLAG="--wandb"; WANDB_MODE="online" ;;
  both) INSTRUMENTATION_BACKEND="both"; WANDB_FLAG="--wandb"; WANDB_MODE="online" ;;
  wandb_offline) INSTRUMENTATION_BACKEND="wandb"; WANDB_FLAG="--wandb"; WANDB_MODE="offline" ;;
  none) INSTRUMENTATION_BACKEND="none"; WANDB_FLAG="--no-wandb"; WANDB_MODE="disabled" ;;
  *) echo "INSTRUMENTATION must be tensorboard, wandb, both, wandb_offline, or none." >&2; exit 2 ;;
esac
# ^^^ THOG
case "$DEPTH_CURVE_PLOTS" in none|final|eval) ;; *) echo "DEPTH_CURVE_PLOTS must be none, final, or eval." >&2; exit 2 ;; esac
case "$DEPTH_CURVE_RENDERER" in matplotlib|plotly|both) ;; *) echo "DEPTH_CURVE_RENDERER must be matplotlib, plotly, or both." >&2; exit 2 ;; esac
case "$RESIDUAL_INIT_POLICY" in depth_scaled|unscaled) ;; *) echo "RESIDUAL_INIT_POLICY must be depth_scaled or unscaled." >&2; exit 2 ;; esac
case "$RESIDUAL_INIT_DEPTH_SOURCE" in true_layer_depth|dof_implied_depth|user_forced_depth) ;; *) echo "Bad RESIDUAL_INIT_DEPTH_SOURCE: $RESIDUAL_INIT_DEPTH_SOURCE" >&2; exit 2 ;; esac
for setting in "$STEPS" "$GRADIENT_ACCUMULATION_STEPS" "$NUM_GPUS" "$EVAL_ITERS" "$EVAL_INTERVAL" "$LOG_INTERVAL" "$N_LAYER" "$N_HEAD" "$N_EMBD" "$BLOCK_SIZE" "$CHECKPOINT_SEGMENT_SIZE" "$RESIDUAL_INIT_DEPTH_VALUE" "$DEPTH_CURVE_SAMPLE_ELEMENTS" "$LAPPED_COSINE_WINDOW_LENGTH"; do validate_positive_uint "$setting" "numeric setting"; done
# vvv THOG fixed anisotropic HYPERBLOCK orders are validated before constructing any run
for setting in "$HYPERBLOCK_COMMON_FAMILY_ORDER" "$HYPERBLOCK_ATTENTION_FAMILY_ORDER" "$HYPERBLOCK_MLP_FAMILY_ORDER" "$HYPERBLOCK_DEPTH_ORDER" "$HYPERBLOCK_D_MODEL_ORDER" "$HYPERBLOCK_MLP_HIDDEN_ORDER" "$HYPERBLOCK_ATTENTION_HEAD_ORDER" "$HYPERBLOCK_ATTENTION_HEAD_CHANNEL_ORDER" "$HYPERBLOCK_MLP_HIDDEN_MULTIPLIER" "$HYPERBLOCK_LOOP_COUNT"; do validate_positive_uint "$setting" "HYPERBLOCK order or loop count"; done
validate_open_closed_unit_float "$HYPERBLOCK_LOOP_DECAY" "HYPERBLOCK_LOOP_DECAY"
# ^^^ THOG
if [[ "$HAS_NON_DEPTH_COMPACT_PRESET" == true ]]; then
  for setting in "$O_ATTN_D_MODEL" "$O_ATTN_QKV_PER_CHANNEL" "$O_ATTN_OUT_PER_CHANNEL" "$O_MLP_D_MODEL" "$O_MLP_HIDDEN"; do validate_positive_uint "$setting" "non-DEPTH compact order"; done
fi
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
validate_true_false "$DIRECT_FACTORISED_MLP" "DIRECT_FACTORISED_MLP"                                                                                   # <<< THOG validate renamed exact MLP option
validate_true_false "$DIRECT_FACTORISED_HYPERBLOCK_MLP" "DIRECT_FACTORISED_HYPERBLOCK_MLP"                                                           # <<< THOG validate independent direct HYPERBLOCK MLP option
validate_true_false "$VECTORISE_PER_HEAD_MATERIALISATION" "VECTORISE_PER_HEAD_MATERIALISATION"                                                         # <<< THOG validate per-head option                                                                          # <<< THOG validate wrapper-only exact MLP application switch
validate_true_false "$DEPTH_COMPRESS_LAYER_NORM_AND_BIAS" "DEPTH_COMPRESS_LAYER_NORM_AND_BIAS"                                                           # <<< THOG validate DEPTH vector participation switch
validate_true_false "$DEPTH_CURVE_LOCAL_HTML" "DEPTH_CURVE_LOCAL_HTML"
validate_true_false "$DRY_RUN" "DRY_RUN"

(( WARMUP_ITERS < STEPS )) || { echo "WARMUP_ITERS must be less than STEPS." >&2; exit 2; }
(( N_EMBD % N_HEAD == 0 )) || { echo "N_EMBD must be divisible by N_HEAD." >&2; exit 2; }
HEAD_DIM=$((N_EMBD / N_HEAD))
# vvv THOG reject impossible HYPERBLOCK orders before invoking Python
if [[ "$HYPERBLOCK" == true ]]; then
  (( HYPERBLOCK_COMMON_FAMILY_ORDER <= 6 )) || { echo "HYPERBLOCK_COMMON_FAMILY_ORDER must not exceed 6." >&2; exit 2; }
  (( HYPERBLOCK_ATTENTION_FAMILY_ORDER <= 4 )) || { echo "HYPERBLOCK_ATTENTION_FAMILY_ORDER must not exceed 4." >&2; exit 2; }
  (( HYPERBLOCK_MLP_FAMILY_ORDER <= 2 )) || { echo "HYPERBLOCK_MLP_FAMILY_ORDER must not exceed 2." >&2; exit 2; }
  (( HYPERBLOCK_DEPTH_ORDER <= N_LAYER )) || { echo "HYPERBLOCK_DEPTH_ORDER must not exceed N_LAYER." >&2; exit 2; }
  (( HYPERBLOCK_D_MODEL_ORDER <= N_EMBD )) || { echo "HYPERBLOCK_D_MODEL_ORDER must not exceed N_EMBD." >&2; exit 2; }
  (( HYPERBLOCK_MLP_HIDDEN_ORDER <= HYPERBLOCK_MLP_HIDDEN_MULTIPLIER * N_EMBD )) || { echo "HYPERBLOCK_MLP_HIDDEN_ORDER exceeds the physical MLP_HIDDEN length." >&2; exit 2; }
  (( HYPERBLOCK_ATTENTION_HEAD_ORDER <= N_HEAD )) || { echo "HYPERBLOCK_ATTENTION_HEAD_ORDER must not exceed N_HEAD." >&2; exit 2; }
  (( HYPERBLOCK_ATTENTION_HEAD_CHANNEL_ORDER <= HEAD_DIM )) || { echo "HYPERBLOCK_ATTENTION_HEAD_CHANNEL_ORDER must not exceed N_EMBD/N_HEAD." >&2; exit 2; }
fi
# ^^^ THOG
# vvv THOG preserve the exact pre-HYPERBLOCK compact-preset guard for source history
# if [[ "$HAS_COMPACT_PRESET" == true ]]; then
# ^^^ THOG
if [[ "$HAS_COMPACT_PRESET" == true && "$HYPERBLOCK" == false ]]; then
  for value in "${O_DEPTH_VALUES[@]}"; do (( value <= N_LAYER )) || { echo "O_DEPTH must not exceed N_LAYER: P=${value}, L=${N_LAYER}." >&2; exit 2; }; done
fi
if [[ "$HAS_NON_DEPTH_COMPACT_PRESET" == true && "$HYPERBLOCK" == false ]]; then
  (( O_ATTN_D_MODEL <= N_EMBD )) || { echo "O_ATTN_D_MODEL must not exceed N_EMBD." >&2; exit 2; }
  (( O_ATTN_QKV_PER_CHANNEL <= HEAD_DIM )) || { echo "O_ATTN_QKV_PER_CHANNEL must not exceed N_EMBD/N_HEAD." >&2; exit 2; }
  (( O_ATTN_OUT_PER_CHANNEL <= HEAD_DIM )) || { echo "O_ATTN_OUT_PER_CHANNEL must not exceed N_EMBD/N_HEAD." >&2; exit 2; }
  (( O_MLP_D_MODEL <= N_EMBD )) || { echo "O_MLP_D_MODEL must not exceed N_EMBD." >&2; exit 2; }
  (( O_MLP_HIDDEN <= 4 * N_EMBD )) || { echo "O_MLP_HIDDEN must not exceed 4*N_EMBD." >&2; exit 2; }
  if [[ "$HAS_JPEG_LIKE_PRESET" == true ]]; then
    for group_size_value in "${MLP_HIDDEN_GROUP_SIZE_VALUES[@]}"; do
      (( 4 * N_EMBD % group_size_value == 0 )) || { echo "4*N_EMBD must be divisible by MLP_HIDDEN_GROUP_SIZE: 4*D=$((4 * N_EMBD)), group=$group_size_value." >&2; exit 2; }
      (( O_MLP_HIDDEN <= group_size_value )) || { echo "O_MLP_HIDDEN/Y must not exceed MLP_HIDDEN_GROUP_SIZE: Y=$O_MLP_HIDDEN, group=$group_size_value." >&2; exit 2; }
    done
  fi
fi
# vvv THOG preserve the exact pre-HYPERBLOCK DEPTH-vector guard for source history
# if [[ "$DEPTH_COMPRESS_LAYER_NORM_AND_BIAS" == true && ( "$HAS_NON_DEPTH_COMPACT_PRESET" == true || "$HAS_DENSE_PRESET" == true ) ]]; then
# ^^^ THOG
if [[ "$DEPTH_COMPRESS_LAYER_NORM_AND_BIAS" == true && ( "$HAS_NON_DEPTH_COMPACT_PRESET" == true || "$HAS_DENSE_PRESET" == true || "$HYPERBLOCK" == true ) ]]; then
  echo "--depth-compress-layer-norm-and-bias may be used only when every selected preset is depth." >&2
  exit 2
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
MLP_HIDDEN_COMPRESSOR_CANONICAL_VALUES=()
for requested_mlp_hidden_compressor in "${MLP_HIDDEN_COMPRESSOR_VALUES[@]}"; do
  if ! compressor_resolution="$("$PYTHON_BIN" -c 'import sys; from sheet.bases import basis_artifact_tag_for_family, normalize_registered_basis_family; family = normalize_registered_basis_family(sys.argv[1]); print(f"{family}\t{basis_artifact_tag_for_family(family)}")' "$requested_mlp_hidden_compressor")"; then
    echo "Failed to resolve MLP_HIDDEN_COMPRESSOR: $requested_mlp_hidden_compressor" >&2
    exit 2
  fi
  IFS=$'\t' read -r compressor_family_value compressor_tag <<< "$compressor_resolution"
  MLP_HIDDEN_COMPRESSOR_CANONICAL_VALUES+=("$compressor_family_value")
  MLP_HIDDEN_COMPRESSOR_TAG_VALUES+=("$compressor_tag")
done
MLP_HIDDEN_COMPRESSOR_VALUES=("${MLP_HIDDEN_COMPRESSOR_CANONICAL_VALUES[@]}")
# ^^^ THOG
CHECKPOINT_FLAG="--no-activation-checkpointing"; [[ "$ACTIVATION_CHECKPOINTING" == true ]] && CHECKPOINT_FLAG="--activation-checkpointing"

export THOG2_INSTRUMENTATION="$INSTRUMENTATION_BACKEND"
export THOG2_CURVE_ROOT="${THOG2_CURVE_ROOT:-curves}"
export THOG2_MLP_CHANNEL_ORDER="$O_MLP_HIDDEN"
export THOG2_DEPTH_CURVE_PLOTS="$DEPTH_CURVE_PLOTS"
export THOG2_DEPTH_CURVE_SAMPLE_ELEMENTS="$DEPTH_CURVE_SAMPLE_ELEMENTS"
export THOG2_DEPTH_CURVE_RENDERER="$DEPTH_CURVE_RENDERER"
export THOG2_DEPTH_CURVE_LOCAL_HTML="$DEPTH_CURVE_LOCAL_HTML"
export THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS="$PLASTIC_LAYER_COUNT_PROBE_RADIUS"
export THOG2_PLASTIC_LAYER_COUNT_MAX_STEP="$PLASTIC_LAYER_COUNT_MAX_STEP"
export THOG2_FAST_DISCARD="$FAST_DISCARD"
export THOG2_BYPASS_SEMANTIC_QKV_ADAPTER="$BYPASS_SEMANTIC_QKV_ADAPTER"                                                                                  # <<< THOG pass wrapper-only optimisation switch into SheetGPTConfig
export THOG2_DIRECT_FACTORISED_MLP="$DIRECT_FACTORISED_MLP"                                                                                              # <<< THOG pass renamed option
export THOG2_DIRECT_FACTORISED_HYPERBLOCK_MLP="$DIRECT_FACTORISED_HYPERBLOCK_MLP"                                                                      # <<< THOG pass independent direct HYPERBLOCK MLP option
export THOG2_VECTORISE_PER_HEAD_MATERIALISATION="$VECTORISE_PER_HEAD_MATERIALISATION"                                                                    # <<< THOG pass per-head option                                                                                    # <<< THOG pass wrapper-only exact MLP application switch into SheetGPTConfig
#export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run_grid_point() {
  local geometry_preset_value="$1"
  local o_depth_value="$2"
  local batch_size_value="$3"                                                                                                                             # <<< THOG batch grid coordinate
  local learning_rate_code="$4"                                                                                                                           # <<< THOG LR grid coordinate
  local basis_family_value="$5"                                                                                                                           # <<< THOG canonical basis-family grid coordinate
  local basis_tag="$6"                                                                                                                                    # <<< THOG matching basis artifact tag
  local mlp_hidden_compressor_value="$7"
  local mlp_hidden_compressor_tag="$8"
  local mlp_hidden_group_size_value="$9"
  local learning_rate_value="${learning_rate_code}e-5" min_lr_value="$((10#$MIN_LR_CODE))e-5"                                                         # <<< THOG decode LR codes
  local run_model_type display_model_type preset_tag run_tag run_name_value LOG_TIMESTAMP resolved_json artifact_name log_path depth_curve_local_root
  local residual_init_depth_source_value n_layer_value n_head_value n_embd_value shape_summary orders_summary start_time_friendly log_url viewer_url serve_url run_status depth_curve_console depth_curve_done
  local -a compact_args compact_order_args optional_args train_args command

  n_layer_value="$N_LAYER"; n_head_value="$N_HEAD"; n_embd_value="$N_EMBD"
  residual_init_depth_source_value="$RESIDUAL_INIT_DEPTH_SOURCE"
  optional_args=(); compact_args=(); compact_order_args=()
  # vvv THOG PLASTIC DEPTH is emitted only on its selected DEPTH run and otherwise introduces no argument or naming changes
  if [[ "$PLASTIC_ENABLED" == true ]]; then
    optional_args+=(--plastic__enabled)
    [[ -n "$PLASTIC_LAYERS_TO_SAMPLE" ]] && optional_args+=(--plastic__layers_to_sample "$PLASTIC_LAYERS_TO_SAMPLE")
    if [[ "$PLASTIC_DO_LEARN_LAYER_COUNT" == true ]]; then optional_args+=(--plastic__do_learn_layer_count); else optional_args+=(--no-plastic__do_learn_layer_count); fi
    [[ -n "$PLASTIC_INITIAL_LAYER_COUNT" ]] && optional_args+=(--plastic__initial_layer_count "$PLASTIC_INITIAL_LAYER_COUNT")
    [[ -n "$PLASTIC_MAX_PERMITTED_LAYERS" ]] && optional_args+=(--plastic__max_permitted_layers "$PLASTIC_MAX_PERMITTED_LAYERS")
    optional_args+=(--plastic__layer_sampling_initialisation "$PLASTIC_LAYER_SAMPLING_INITIALISATION")
    optional_args+=(--plastic__layer_count_objective "$PLASTIC_LAYER_COUNT_OBJECTIVE")
    optional_args+=(--plastic__coarse_phase "$PLASTIC_COARSE_PHASE")
    if [[ "$PLASTIC_COARSE_PHASE" == enabled ]]; then
      optional_args+=(--plastic__phase_1_n_steps "$PLASTIC_PHASE_1_N_STEPS")
      optional_args+=(--plastic__phase_1_starting_layer_count "$PLASTIC_PHASE_1_STARTING_LAYER_COUNT")
      optional_args+=(--plastic__phase_1__number_of_trials "$PLASTIC_PHASE_1_NUMBER_OF_TRIALS")
      optional_args+=(--plastic__phase_1_evaluation_steps_count "$PLASTIC_PHASE_1_EVALUATION_STEPS_COUNT")
    fi
    [[ -n "$PLASTIC_LAYER_COUNT_PROBE_EVERY_N_STEPS" ]] && optional_args+=(--plastic__layer_count_probe__probe_every_n_steps "$PLASTIC_LAYER_COUNT_PROBE_EVERY_N_STEPS")
    optional_args+=(--plastic__layer_count_probe__number_of_sampled_valid_tokens "$PLASTIC_LAYER_COUNT_PROBE_NUMBER_OF_SAMPLED_VALID_TOKENS")
    optional_args+=(--plastic__layer_count_probe_radius "$PLASTIC_LAYER_COUNT_PROBE_RADIUS")
    optional_args+=(--plastic__layer_count_max_step "$PLASTIC_LAYER_COUNT_MAX_STEP")
    optional_args+=(--plastic__layer_count_update_brake "$PLASTIC_LAYER_COUNT_UPDATE_BRAKE")
    optional_args+=(--plastic__log_interval_coarse "$PLASTIC_LOG_INTERVAL_COARSE")
    if [[ "$PLASTIC_COARSE_PHASE_ROLL_THROUGH" == true ]]; then
      optional_args+=(--plastic__coarse_phase_roll_through)
    else
      optional_args+=(--no-plastic__coarse_phase_roll_through)
    fi
    optional_args+=(--plastic__layer_count_probe__window_size_as_number_of_probes "$PLASTIC_LAYER_COUNT_PROBE_WINDOW_SIZE_AS_NUMBER_OF_PROBES")
    optional_args+=(--plastic__layer_count_probe_noise_lambda "$PLASTIC_LAYER_COUNT_PROBE_NOISE_LAMBDA")
    optional_args+=(--plastic__layer_count_extrapolation_weight "$PLASTIC_LAYER_COUNT_EXTRAPOLATION_WEIGHT")
    optional_args+=(--plastic__layer_count_cost_weight "$PLASTIC_LAYER_COUNT_COST_WEIGHT")
    [[ -n "$PLASTIC_LAYER_MEMORY_BUDGET_GIB" ]] && optional_args+=(--plastic__layer_memory_budget_gib "$PLASTIC_LAYER_MEMORY_BUDGET_GIB")
    optional_args+=(--plastic__cuda_allocator_reserve_gib "$PLASTIC_CUDA_ALLOCATOR_RESERVE_GIB")
    optional_args+=(--plastic__geometry_learning_rate_multiplier "$PLASTIC_GEOMETRY_LEARNING_RATE_MULTIPLIER")
    if [[ "$PLASTIC_FREEZE_GEOMETRY_DURING_WARMUP" == true ]]; then optional_args+=(--plastic__freeze_geometry_during_warmup); else optional_args+=(--no-plastic__freeze_geometry_during_warmup); fi
  fi
  # ^^^ THOG
  # vvv THOG layer dropout is architecture-level and therefore applies to dense and compact runs alike
  [[ -n "$LAYER_DROPOUT_STRATUM_SIZE" ]] && optional_args+=(--layer-dropout-stratum-size "$LAYER_DROPOUT_STRATUM_SIZE")
  [[ -n "$LAYER_DROPOUT_ACTIVE_PER_STRATUM" ]] && optional_args+=(--layer-dropout-active-per-stratum "$LAYER_DROPOUT_ACTIVE_PER_STRATUM")
  optional_args+=(--layer-dropout-resample-steps "$LAYER_DROPOUT_RESAMPLE_STEPS")
  # ^^^ THOG
  if [[ "$geometry_preset_value" == dense ]]; then
    run_model_type="dense"; display_model_type="dense"; preset_tag="DENSE"; run_tag="DENSE"
    [[ "$N_LAYER_EXPLICIT" == false ]] && n_layer_value=12
    [[ "$N_HEAD_EXPLICIT" == false ]] && n_head_value=12
    [[ "$N_EMBD_EXPLICIT" == false ]] && n_embd_value=768
    [[ "$residual_init_depth_source_value" == dof_implied_depth ]] && residual_init_depth_source_value="true_layer_depth"
    shape_summary="L${n_layer_value} H${n_head_value} D${n_embd_value} C${BLOCK_SIZE}"
    orders_summary="n/a"
    compact_order_args=(--o-depth "$o_depth_value" --o-attn-d-model "$O_ATTN_D_MODEL" --o-attn-qkv-per-channel "$O_ATTN_QKV_PER_CHANNEL" --o-attn-out-per-channel "$O_ATTN_OUT_PER_CHANNEL" --o-mlp-d-model "$O_MLP_D_MODEL" --o-mlp-hidden "$O_MLP_HIDDEN")
  elif [[ "$geometry_preset_value" == hyperblock ]]; then
    run_model_type="sheet"; display_model_type="hyperblock"; preset_tag="HYPERBLOCK"; run_tag="HB_${HYPERBLOCK_COMPRESSOR^^}"
    compact_args=(
      --hyperblock
      --hyperblock-compressor "$HYPERBLOCK_COMPRESSOR"
      --hyperblock-compressor-version "$HYPERBLOCK_COMPRESSOR_VERSION"
      --hyperblock-common-family-order "$HYPERBLOCK_COMMON_FAMILY_ORDER"
      --hyperblock-attention-family-order "$HYPERBLOCK_ATTENTION_FAMILY_ORDER"
      --hyperblock-mlp-family-order "$HYPERBLOCK_MLP_FAMILY_ORDER"
      --hyperblock-depth-order "$HYPERBLOCK_DEPTH_ORDER"
      --hyperblock-d-model-order "$HYPERBLOCK_D_MODEL_ORDER"
      --hyperblock-mlp-hidden-order "$HYPERBLOCK_MLP_HIDDEN_ORDER"
      --hyperblock-attention-head-order "$HYPERBLOCK_ATTENTION_HEAD_ORDER"
      --hyperblock-attention-head-channel-order "$HYPERBLOCK_ATTENTION_HEAD_CHANNEL_ORDER"
      --hyperblock-mlp-hidden-multiplier "$HYPERBLOCK_MLP_HIDDEN_MULTIPLIER"
      --hyperblock-loop-count "$HYPERBLOCK_LOOP_COUNT"
      --hyperblock-loop-decay "$HYPERBLOCK_LOOP_DECAY"
    )
    compact_order_args=()
    shape_summary="L${n_layer_value} H${n_head_value} D${n_embd_value} C${BLOCK_SIZE}"
    orders_summary="HFC${HYPERBLOCK_COMMON_FAMILY_ORDER} HFA${HYPERBLOCK_ATTENTION_FAMILY_ORDER} HFM${HYPERBLOCK_MLP_FAMILY_ORDER} HL${HYPERBLOCK_DEPTH_ORDER} HD${HYPERBLOCK_D_MODEL_ORDER} HM${HYPERBLOCK_MLP_HIDDEN_ORDER} HH${HYPERBLOCK_ATTENTION_HEAD_ORDER} HC${HYPERBLOCK_ATTENTION_HEAD_CHANNEL_ORDER} loops=${HYPERBLOCK_LOOP_COUNT} decay=${HYPERBLOCK_LOOP_DECAY}"
  else
    run_model_type="sheet"; display_model_type="spectral"; preset_tag="${geometry_preset_value^^}"
    [[ "$geometry_preset_value" == legacy_sheet_col ]] && preset_tag="SHEET_COL"
    run_tag="${basis_tag}_${preset_tag}"
    [[ "$geometry_preset_value" == jpeg_like_v1 ]] && run_tag="${run_tag}_${mlp_hidden_compressor_tag}_G${mlp_hidden_group_size_value}"
    compact_args=(--geometry-preset "$geometry_preset_value" --basis-family "$basis_family_value" --basis-version "$BASIS_VERSION" --lapped-cosine-window-length "$LAPPED_COSINE_WINDOW_LENGTH" --lapped-cosine-overlap-fraction "$LAPPED_COSINE_OVERLAP_FRACTION" --mlp-hidden-compressor "$mlp_hidden_compressor_value" --mlp-hidden-group-size "$mlp_hidden_group_size_value")
    [[ -n "$ATTENTION_GEOMETRY" ]] && optional_args+=(--attention-geometry "$ATTENTION_GEOMETRY")
    [[ -n "$MLP_GEOMETRY" ]] && optional_args+=(--mlp-geometry "$MLP_GEOMETRY")
    shape_summary="L${n_layer_value} H${n_head_value} D${n_embd_value} C${BLOCK_SIZE}"
    if [[ "$geometry_preset_value" == depth ]]; then
      compact_order_args=(--o-depth "$o_depth_value" --o-attn-d-model 1 --o-attn-qkv-per-channel 1 --o-attn-out-per-channel 1 --o-mlp-d-model 1 --o-mlp-hidden 1)
      orders_summary="P${o_depth_value} DLB=${DEPTH_COMPRESS_LAYER_NORM_AND_BIAS}"
      if [[ "$DEPTH_COMPRESS_LAYER_NORM_AND_BIAS" == true ]]; then
        optional_args+=(--depth-compress-layer-norm-and-bias)
      else
        optional_args+=(--no-depth-compress-layer-norm-and-bias)
      fi
    else
      compact_order_args=(--o-depth "$o_depth_value" --o-attn-d-model "$O_ATTN_D_MODEL" --o-attn-qkv-per-channel "$O_ATTN_QKV_PER_CHANNEL" --o-attn-out-per-channel "$O_ATTN_OUT_PER_CHANNEL" --o-mlp-d-model "$O_MLP_D_MODEL" --o-mlp-hidden "$O_MLP_HIDDEN")
      orders_summary="P${o_depth_value} Q${O_ATTN_D_MODEL} J${O_ATTN_QKV_PER_CHANNEL} O${O_ATTN_OUT_PER_CHANNEL} X${O_MLP_D_MODEL} Y${O_MLP_HIDDEN}"
      [[ "$geometry_preset_value" == jpeg_like_v1 ]] && orders_summary="${orders_summary} MHG${mlp_hidden_group_size_value}"
    fi
  fi

  run_name_value="$RUN_NAME"; [[ -z "$run_name_value" ]] && run_name_value="${run_tag}_OWT"
  train_args=(
    --model-type "$run_model_type" --run-mode "$RUN_MODE" --host-label "$HOST_LABEL" --run-name "$run_name_value"
    --dataset "$DATASET_NAME" --data-dir "$DATA_DIR" --checkpoint-root "$CHECKPOINT_ROOT" --log-root "$LOG_ROOT" --result-root "$RESULT_ROOT" --wandb-root "$WANDB_ROOT"
    --max-iters "$STEPS" --batch-size "$batch_size_value" --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
    --eval-iters "$EVAL_ITERS" --eval-interval "$EVAL_INTERVAL" --log-interval "$LOG_INTERVAL" --checkpoint-interval "$CHECKPOINT_INTERVAL" --warmup-iters "$WARMUP_ITERS" --learning-rate "$learning_rate_value" --min-lr "$min_lr_value" --max-nonfinite-update-skips "$MAX_NONFINITE_UPDATE_SKIPS"
    --n-layer "$n_layer_value" --n-head "$n_head_value" --n-embd "$n_embd_value" --block-size "$BLOCK_SIZE"
    "${compact_order_args[@]}"
    "${compact_args[@]}" --attention-backend "$ATTENTION_BACKEND" --experiment-prefix "$EXPERIMENT_PREFIX" --dtype "$DTYPE"
    --residual-init-policy "$RESIDUAL_INIT_POLICY" --residual-init-depth-source "$residual_init_depth_source_value" --residual-init-depth-value "$RESIDUAL_INIT_DEPTH_VALUE"
    "$CHECKPOINT_FLAG" --checkpoint-segment-size "$CHECKPOINT_SEGMENT_SIZE" "$WANDB_FLAG" --wandb-mode "$WANDB_MODE" "${optional_args[@]}" "${EXTRA_ARGS[@]}"
  )

  if [[ "$EXPLAIN_GEOMETRY" == true ]]; then
    "$PYTHON_BIN" -m "$RUN_MODULE" "${train_args[@]}"
    return 0
  fi

  LOG_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  start_time_friendly="$(date '+%H:%M  %d-%m-%y')"
  resolved_json="$("$PYTHON_BIN" -m "$RUN_MODULE" "${train_args[@]}" --log-timestamp "$LOG_TIMESTAMP" --print-resolved-json)"
  artifact_name="$(printf '%s' "$resolved_json" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["artifact_name"])')"
  log_path="$(printf '%s' "$resolved_json" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["paths"]["log_path"])')"
  depth_curve_local_root="$(dirname "$log_path")/depth_curves"; export THOG2_DEPTH_CURVE_LOCAL_ROOT="$depth_curve_local_root"
  command=("$PYTHON_BIN" -m "$RUN_MODULE" "${train_args[@]}" --log-timestamp "$LOG_TIMESTAMP")
  if (( NUM_GPUS > 1 )); then command=("$PYTHON_BIN" -m torch.distributed.run --standalone "--nproc-per-node=$NUM_GPUS" -m "$RUN_MODULE" "${train_args[@]}" --log-timestamp "$LOG_TIMESTAMP"); fi
  log_url="file://$(realpath -m "$log_path")"; viewer_url="file://$(realpath -m "$depth_curve_local_root/index.html")"; serve_url="http://localhost:${DEPTH_CURVE_HTTP_PORT}/"
  # vvv THOG inactive depth-curve diagnostics no longer advertise hypothetical renderer or viewer settings
  if [[ "$DEPTH_CURVE_PLOTS" == none ]]; then
    depth_curve_console="  depth curves:       none"
    depth_curve_done=""
  else
    depth_curve_console="  depth curves:       $DEPTH_CURVE_PLOTS  (sample elements: $DEPTH_CURVE_SAMPLE_ELEMENTS, renderer: $DEPTH_CURVE_RENDERER, local html: $DEPTH_CURVE_LOCAL_HTML)
  depth viewer:       $viewer_url
  serve viewer:       (cd $depth_curve_local_root && python -m http.server $DEPTH_CURVE_HTTP_PORT)
  served URL:         $serve_url"
    depth_curve_done="  depth viewer URL:   $viewer_url"
  fi
  # ^^^ THOG

  # vvv THOG preserve the exact pre-HYPERBLOCK console line outside the emitted here-document
  # model/preset/basis: $display_model_type / $geometry_preset_value / $basis_family_value
  # ^^^ THOG
  #   optimizer:          $OPTIMIZER  momentum=$OPTIMIZER_MOMENTUM  lr=$learning_rate_value (LR_$learning_rate_code) min_lr=$min_lr_value  # <<< THOG aligned Python summary now owns this row
  cat <<EOF_RUN
scruffy OWT train
  start time:         $start_time_friendly
  artifact:           $artifact_name
  experiment:         $EXPERIMENT_PREFIX
  model/preset/basis: $display_model_type / $geometry_preset_value / $([[ "$geometry_preset_value" == hyperblock ]] && printf '%s' "$HYPERBLOCK_COMPRESSOR" || printf '%s' "$basis_family_value")
  lapped cosine:      window=$LAPPED_COSINE_WINDOW_LENGTH overlap=$LAPPED_COSINE_OVERLAP_FRACTION
  JPEG_LIKE_V1:       compressor=$mlp_hidden_compressor_value group=$mlp_hidden_group_size_value Y=$O_MLP_HIDDEN
  backend/dtype:      $ATTENTION_BACKEND / $DTYPE
  instrumentation:    $INSTRUMENTATION
  non-finite updates: policy=skip max_skips=$MAX_NONFINITE_UPDATE_SKIPS
  fast discard:       $FAST_DISCARD
  semantic adapter bypass:                $BYPASS_SEMANTIC_QKV_ADAPTER
  direct factorised MLP:                  $DIRECT_FACTORISED_MLP
  direct factorised HYPERBLOCK MLP:           $DIRECT_FACTORISED_HYPERBLOCK_MLP
  vectorise per-head materialisation:     $VECTORISE_PER_HEAD_MATERIALISATION
  layer dropout:      stratum=${LAYER_DROPOUT_STRATUM_SIZE:-N_LAYER} active=${LAYER_DROPOUT_ACTIVE_PER_STRATUM:-STRATUM_SIZE} resample_steps=$LAYER_DROPOUT_RESAMPLE_STEPS
  plastic depth:      enabled=$PLASTIC_ENABLED fixed=${PLASTIC_LAYERS_TO_SAMPLE:-N_LAYER} learn_count=$PLASTIC_DO_LEARN_LAYER_COUNT initial=${PLASTIC_INITIAL_LAYER_COUNT:-N_LAYER} max=${PLASTIC_MAX_PERMITTED_LAYERS:-N_LAYER} init=$PLASTIC_LAYER_SAMPLING_INITIALISATION objective=$PLASTIC_LAYER_COUNT_OBJECTIVE
  plastic coarse:     phase=$PLASTIC_COARSE_PHASE start=${PLASTIC_PHASE_1_STARTING_LAYER_COUNT:--} trials=${PLASTIC_PHASE_1_NUMBER_OF_TRIALS:--} steps=${PLASTIC_PHASE_1_N_STEPS:--} eval=${PLASTIC_PHASE_1_EVALUATION_STEPS_COUNT:--} log=$PLASTIC_LOG_INTERVAL_COARSE roll_through=$PLASTIC_COARSE_PHASE_ROLL_THROUGH
  plastic fine:       probe_every=${PLASTIC_LAYER_COUNT_PROBE_EVERY_N_STEPS:-update_brake} window=$PLASTIC_LAYER_COUNT_PROBE_WINDOW_SIZE_AS_NUMBER_OF_PROBES radius=$PLASTIC_LAYER_COUNT_PROBE_RADIUS max_step=$PLASTIC_LAYER_COUNT_MAX_STEP brake=$PLASTIC_LAYER_COUNT_UPDATE_BRAKE extrap_w=$PLASTIC_LAYER_COUNT_EXTRAPOLATION_WEIGHT
$depth_curve_console
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
scruffy OWT run finished
  status:             $run_status
  artifact:           $artifact_name
  log URL:            $log_url
$depth_curve_done
EOF_DONE
  return "$run_status"
}

if (( ${#PRESET_VALUES[@]} > 1 || ${#BASIS_FAMILY_VALUES[@]} > 1 || ${#MLP_HIDDEN_COMPRESSOR_VALUES[@]} > 1 || ${#MLP_HIDDEN_GROUP_SIZE_VALUES[@]} > 1 || ${#O_DEPTH_VALUES[@]} > 1 || ${#BATCH_SIZE_VALUES[@]} > 1 || ${#LEARNING_RATE_CODE_VALUES[@]} > 1 )); then
  echo "scruffy OWT grid: p=${PRESET_VALUES[*]} B=${BASIS_FAMILY_VALUES[*]} MHC=${MLP_HIDDEN_COMPRESSOR_VALUES[*]} MHG=${MLP_HIDDEN_GROUP_SIZE_VALUES[*]} P=${O_DEPTH_VALUES[*]} b=${BATCH_SIZE_VALUES[*]} LR=${LEARNING_RATE_CODE_VALUES[*]}"
fi
for geometry_preset_value in "${PRESET_VALUES[@]}"; do
  if [[ "$geometry_preset_value" == dense ]]; then
    for batch_size_value in "${BATCH_SIZE_VALUES[@]}"; do
      for learning_rate_code in "${LEARNING_RATE_CODE_VALUES[@]}"; do
        run_grid_point "$geometry_preset_value" "${O_DEPTH_VALUES[0]}" "$batch_size_value" "$learning_rate_code" "${BASIS_FAMILY_VALUES[0]}" "${BASIS_TAG_VALUES[0]}" "${MLP_HIDDEN_COMPRESSOR_VALUES[0]}" "${MLP_HIDDEN_COMPRESSOR_TAG_VALUES[0]}" "${MLP_HIDDEN_GROUP_SIZE_VALUES[0]}"
      done
    done
  else
    for basis_index in "${!BASIS_FAMILY_VALUES[@]}"; do
      basis_family_value="${BASIS_FAMILY_VALUES[$basis_index]}"
      basis_tag="${BASIS_TAG_VALUES[$basis_index]}"
      if [[ "$geometry_preset_value" == jpeg_like_v1 ]]; then
        for compressor_index in "${!MLP_HIDDEN_COMPRESSOR_VALUES[@]}"; do
          mlp_hidden_compressor_value="${MLP_HIDDEN_COMPRESSOR_VALUES[$compressor_index]}"
          mlp_hidden_compressor_tag="${MLP_HIDDEN_COMPRESSOR_TAG_VALUES[$compressor_index]}"
          for mlp_hidden_group_size_value in "${MLP_HIDDEN_GROUP_SIZE_VALUES[@]}"; do
            for batch_size_value in "${BATCH_SIZE_VALUES[@]}"; do
              for learning_rate_code in "${LEARNING_RATE_CODE_VALUES[@]}"; do
                for o_depth_value in "${O_DEPTH_VALUES[@]}"; do
                  run_grid_point "$geometry_preset_value" "$o_depth_value" "$batch_size_value" "$learning_rate_code" "$basis_family_value" "$basis_tag" "$mlp_hidden_compressor_value" "$mlp_hidden_compressor_tag" "$mlp_hidden_group_size_value"
                done
              done
            done
          done
        done
      else
        for batch_size_value in "${BATCH_SIZE_VALUES[@]}"; do
          for learning_rate_code in "${LEARNING_RATE_CODE_VALUES[@]}"; do
            for o_depth_value in "${O_DEPTH_VALUES[@]}"; do
              run_grid_point "$geometry_preset_value" "$o_depth_value" "$batch_size_value" "$learning_rate_code" "$basis_family_value" "$basis_tag" "${MLP_HIDDEN_COMPRESSOR_VALUES[0]}" "${MLP_HIDDEN_COMPRESSOR_TAG_VALUES[0]}" "${MLP_HIDDEN_GROUP_SIZE_VALUES[0]}"
            done
          done
        done
      fi
    done
  fi
done
# ^^^ THOG
# vvv THOG preserved superseded source lines for exact history audit
# --hyperblock-compressor|--hyperblock-compressor-version|--hyperblock-common-family-order|--hyperblock-attention-family-order|--hyperblock-mlp-family-order|--hyperblock-depth-order|--hyperblock-d-model-order|--hyperblock-mlp-hidden-order|--hyperblock-attention-head-order|--hyperblock-attention-head-channel-order|--hyperblock-mlp-hidden-multiplier)
# --hyperblock-compressor=*|--hyperblock-compressor-version=*|--hyperblock-common-family-order=*|--hyperblock-attention-family-order=*|--hyperblock-mlp-family-order=*|--hyperblock-depth-order=*|--hyperblock-d-model-order=*|--hyperblock-mlp-hidden-order=*|--hyperblock-attention-head-order=*|--hyperblock-attention-head-channel-order=*|--hyperblock-mlp-hidden-multiplier=*)
# for setting in "$HYPERBLOCK_COMMON_FAMILY_ORDER" "$HYPERBLOCK_ATTENTION_FAMILY_ORDER" "$HYPERBLOCK_MLP_FAMILY_ORDER" "$HYPERBLOCK_DEPTH_ORDER" "$HYPERBLOCK_D_MODEL_ORDER" "$HYPERBLOCK_MLP_HIDDEN_ORDER" "$HYPERBLOCK_ATTENTION_HEAD_ORDER" "$HYPERBLOCK_ATTENTION_HEAD_CHANNEL_ORDER" "$HYPERBLOCK_MLP_HIDDEN_MULTIPLIER"; do validate_positive_uint "$setting" "HYPERBLOCK order"; done
# local residual_init_depth_source_value n_layer_value n_head_value n_embd_value shape_summary orders_summary start_time_friendly log_url viewer_url serve_url run_status
# orders_summary="HFC${HYPERBLOCK_COMMON_FAMILY_ORDER} HFA${HYPERBLOCK_ATTENTION_FAMILY_ORDER} HFM${HYPERBLOCK_MLP_FAMILY_ORDER} HL${HYPERBLOCK_DEPTH_ORDER} HD${HYPERBLOCK_D_MODEL_ORDER} HM${HYPERBLOCK_MLP_HIDDEN_ORDER} HH${HYPERBLOCK_ATTENTION_HEAD_ORDER} HC${HYPERBLOCK_ATTENTION_HEAD_CHANNEL_ORDER}"
# depth curves:       $DEPTH_CURVE_PLOTS  (sample elements: $DEPTH_CURVE_SAMPLE_ELEMENTS, renderer: $DEPTH_CURVE_RENDERER, local html: $DEPTH_CURVE_LOCAL_HTML)
# served URL:         $serve_url
# optimizer:          $OPTIMIZER  momentum=$OPTIMIZER_MOMENTUM  lr=$learning_rate_value (LR_$learning_rate_code) min_lr=$min_lr_value
# depth viewer URL:   $viewer_url
# ^^^ THOG

# vvv THOG retired PLASTIC DEPTH hold-controller source preserved for history audit
# PLASTIC_LAYER_COUNT_HOLD_UPDATES=100
# --plastic-layer-count-hold-updates N=${PLASTIC_LAYER_COUNT_HOLD_UPDATES}
# --plastic__log_interval_coarse|--plastic__layers_to_sample|--plastic__initial_layer_count|--plastic__max_permitted_layers|--plastic__layer_sampling_initialisation|--plastic__layer_count_objective|--plastic-layer-count-hold-updates|--plastic__layer_count_cost_weight|--plastic__layer_memory_budget_gib|--plastic__geometry_learning_rate_multiplier)
# --plastic-layer-count-hold-updates) PLASTIC_LAYER_COUNT_HOLD_UPDATES="$2" ;;
# --plastic__log_interval_coarse=*|--plastic__layers_to_sample=*|--plastic__initial_layer_count=*|--plastic__max_permitted_layers=*|--plastic__layer_sampling_initialisation=*|--plastic__layer_count_objective=*|--plastic-layer-count-hold-updates=*|--plastic__layer_count_cost_weight=*|--plastic__layer_memory_budget_gib=*|--plastic__geometry_learning_rate_multiplier=*)
# --plastic-layer-count-hold-updates) PLASTIC_LAYER_COUNT_HOLD_UPDATES="$plastic_value" ;;
# validate_positive_uint "$PLASTIC_LAYER_COUNT_HOLD_UPDATES" "PLASTIC_LAYER_COUNT_HOLD_UPDATES"
# optional_args+=(--plastic-layer-count-hold-updates "$PLASTIC_LAYER_COUNT_HOLD_UPDATES")
# ^^^ THOG
