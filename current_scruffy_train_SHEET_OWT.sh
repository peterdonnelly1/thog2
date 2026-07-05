#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

RUN_MODULE="run_thog2_owt_residual"
HOST_LABEL="scruffy"
RUNTIME_PROFILE="auto"
DTYPE="auto"
HOST_LABEL_EXPLICIT=false
DTYPE_EXPLICIT=false
RUN_MODE="fresh"
RUN_NAME="GPT2_SMALL_VS_SHEET144_P32_RBASIS_A"
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
N_LAYER=144
N_HEAD=12
N_EMBD=768
WIDTH_SWEEP=""
BLOCK_SIZE=1024
DEPTH_ORDER=32
BASE_ROW_ORDER=64
RESIDUAL_INIT_POLICY="depth_scaled"
RESIDUAL_INIT_DEPTH_SOURCE="dof_implied_depth"
RESIDUAL_INIT_DEPTH_VALUE=12
ACTIVATION_CHECKPOINTING=true
CHECKPOINT_SEGMENT_SIZE=12
WANDB_MODE="online"
WANDB_ENABLED=true
DRY_RUN=false

usage() {
  cat <<EOF
Usage: $0 [options] [-- additional ${RUN_MODULE} arguments]

  -o HOST_LABEL=${HOST_LABEL}
  -R RUNTIME_PROFILE=${RUNTIME_PROFILE}       auto | scruffy | dreedle
  -T DTYPE=${DTYPE}                           auto | float32 | float16 | bfloat16
  -q RUN_MODE=${RUN_MODE}                     fresh | resume
  -g RUN_NAME=${RUN_NAME}
  -n STEPS=${STEPS}
  -b BATCH_SIZE=${BATCH_SIZE}
  -d DATASET_NAME=${DATASET_NAME}
  -t DATA_DIR=${DATA_DIR}
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
  -Y WIDTH_SWEEP=${WIDTH_SWEEP:-single}          D/H list or D_START:D_STOP:D_STEP, e.g. 768/12,1024/16 or 768:1536:256
  -C BLOCK_SIZE=${BLOCK_SIZE}
  -P DEPTH_ORDER=${DEPTH_ORDER}
  -Q BASE_ROW_ORDER=${BASE_ROW_ORDER}
  -r RESIDUAL_INIT_POLICY=${RESIDUAL_INIT_POLICY}              depth_scaled | unscaled
  -z RESIDUAL_INIT_DEPTH_SOURCE=${RESIDUAL_INIT_DEPTH_SOURCE}  true_layer_depth | dof_implied_depth | user_forced_depth
  -Z RESIDUAL_INIT_DEPTH_VALUE=${RESIDUAL_INIT_DEPTH_VALUE}    used only when depth source is user_forced_depth
  -p ACTIVATION_CHECKPOINTING=${ACTIVATION_CHECKPOINTING}
  -S CHECKPOINT_SEGMENT_SIZE=${CHECKPOINT_SEGMENT_SIZE}
  -M WANDB_MODE=${WANDB_MODE}                 online | offline | disabled
  -W WANDB_ENABLED=${WANDB_ENABLED}
  -x DRY_RUN=${DRY_RUN}
  -h show this help
EOF
}

while getopts ":o:R:T:q:g:n:b:d:t:u:e:l:w:k:A:G:L:H:D:Y:C:P:Q:r:z:Z:p:S:M:W:x:h" option; do
  case "$option" in
    o) HOST_LABEL="$OPTARG"; HOST_LABEL_EXPLICIT=true ;;
    R) RUNTIME_PROFILE="$OPTARG" ;;
    T) DTYPE="$OPTARG"; DTYPE_EXPLICIT=true ;;
    q) RUN_MODE="$OPTARG" ;;
    g) RUN_NAME="$OPTARG" ;;
    n) STEPS="$OPTARG" ;;
    b) BATCH_SIZE="$OPTARG" ;;
    d) DATASET_NAME="$OPTARG"; DATA_DIR="data/$OPTARG" ;;
    t) DATA_DIR="$OPTARG" ;;
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
    Y) WIDTH_SWEEP="$OPTARG" ;;
    C) BLOCK_SIZE="$OPTARG" ;;
    P) DEPTH_ORDER="$OPTARG" ;;
    Q) BASE_ROW_ORDER="$OPTARG" ;;
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
if [[ "${1:-}" == "--" ]]; then shift; fi
EXTRA_ARGS=("$@")

validate_positive_uint() { [[ "$1" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid $2: $1; expected a positive integer." >&2; exit 2; }; }
validate_nonnegative_uint() { [[ "$1" =~ ^[0-9]+$ ]] || { echo "Invalid $2: $1; expected a non-negative integer." >&2; exit 2; }; }
validate_true_false() { case "$1" in true|false) ;; *) echo "Invalid $2: $1; expected true or false." >&2; exit 2 ;; esac; }

# vvv THOG
WIDTH_SPECS=()
WIDTH_N_EMBD=""
WIDTH_N_HEAD=""

apply_runtime_profile() {
  case "$RUNTIME_PROFILE" in
    auto) ;;
    scruffy)
      [[ "$HOST_LABEL_EXPLICIT" == false ]] && HOST_LABEL="scruffy"
      [[ "$DTYPE_EXPLICIT" == false ]] && DTYPE="bfloat16"
      ;;
    dreedle)
      [[ "$HOST_LABEL_EXPLICIT" == false ]] && HOST_LABEL="dreedle"
      [[ "$DTYPE_EXPLICIT" == false ]] && DTYPE="float16"
      ;;
    *) echo "RUNTIME_PROFILE must be auto, scruffy, or dreedle." >&2; exit 2 ;;
  esac
}

build_width_specs() {
  WIDTH_SPECS=()
  if [[ -z "$WIDTH_SWEEP" ]]; then
    WIDTH_SPECS=("$N_EMBD/$N_HEAD")
    return
  fi

  local normalized="${WIDTH_SWEEP//[[:space:]]/}"
  if [[ "$normalized" =~ ^([1-9][0-9]*):([1-9][0-9]*):([1-9][0-9]*)$ ]]; then
    local start_width="${BASH_REMATCH[1]}"
    local stop_width="${BASH_REMATCH[2]}"
    local step_width="${BASH_REMATCH[3]}"
    (( start_width <= stop_width )) || { echo "WIDTH_SWEEP start must be <= stop: $WIDTH_SWEEP" >&2; exit 2; }
    for (( width = start_width; width <= stop_width; width += step_width )); do
      (( width % 64 == 0 )) || { echo "WIDTH_SWEEP range width must be divisible by 64: $width" >&2; exit 2; }
      WIDTH_SPECS+=("$width/$((width / 64))")
    done
    return
  fi

  IFS=',' read -r -a WIDTH_SPECS <<< "$normalized"
}

parse_width_spec() {
  local raw_spec="$1"
  local cleaned_spec="${raw_spec//[[:space:]]/}"
  if [[ "$cleaned_spec" =~ ^D?([1-9][0-9]*)[:/]H?([1-9][0-9]*)$ ]]; then
    WIDTH_N_EMBD="${BASH_REMATCH[1]}"
    WIDTH_N_HEAD="${BASH_REMATCH[2]}"
  else
    echo "Invalid WIDTH_SWEEP element: $raw_spec; expected D/H, D:H, D768/H12, or range D_START:D_STOP:D_STEP." >&2
    exit 2
  fi
  validate_positive_uint "$WIDTH_N_EMBD" "WIDTH_SWEEP n_embd"
  validate_positive_uint "$WIDTH_N_HEAD" "WIDTH_SWEEP n_head"
  (( WIDTH_N_EMBD % WIDTH_N_HEAD == 0 )) || { echo "WIDTH_SWEEP n_embd must be divisible by n_head: $raw_spec" >&2; exit 2; }
  (( BASE_ROW_ORDER <= WIDTH_N_EMBD )) || { echo "BASE_ROW_ORDER must not exceed swept N_EMBD: Q$BASE_ROW_ORDER > D$WIDTH_N_EMBD" >&2; exit 2; }
}
# ^^^ THOG

apply_runtime_profile

case "$RUN_MODE" in fresh|resume) ;; *) echo "RUN_MODE must be fresh or resume." >&2; exit 2 ;; esac
case "$WANDB_MODE" in online|offline|disabled) ;; *) echo "WANDB_MODE must be online, offline, or disabled." >&2; exit 2 ;; esac
case "$DTYPE" in auto|float32|float16|bfloat16) ;; *) echo "DTYPE must be auto, float32, float16, or bfloat16." >&2; exit 2 ;; esac
case "$RESIDUAL_INIT_POLICY" in depth_scaled|unscaled) ;; *) echo "RESIDUAL_INIT_POLICY must be depth_scaled or unscaled." >&2; exit 2 ;; esac
case "$RESIDUAL_INIT_DEPTH_SOURCE" in true_layer_depth|dof_implied_depth|user_forced_depth) ;; *) echo "RESIDUAL_INIT_DEPTH_SOURCE must be true_layer_depth, dof_implied_depth, or user_forced_depth." >&2; exit 2 ;; esac
for setting in "$STEPS" "$BATCH_SIZE" "$EVAL_ITERS" "$EVAL_INTERVAL" "$LOG_INTERVAL" "$GRADIENT_ACCUMULATION_STEPS" "$NUM_GPUS" "$N_LAYER" "$N_HEAD" "$N_EMBD" "$BLOCK_SIZE" "$DEPTH_ORDER" "$BASE_ROW_ORDER" "$CHECKPOINT_SEGMENT_SIZE" "$RESIDUAL_INIT_DEPTH_VALUE"; do
  validate_positive_uint "$setting" "numeric setting"
done
validate_nonnegative_uint "$WARMUP_ITERS" "WARMUP_ITERS"
validate_nonnegative_uint "$CHECKPOINT_INTERVAL" "CHECKPOINT_INTERVAL"
validate_true_false "$ACTIVATION_CHECKPOINTING" "ACTIVATION_CHECKPOINTING"
validate_true_false "$WANDB_ENABLED" "WANDB_ENABLED"
validate_true_false "$DRY_RUN" "DRY_RUN"
(( WARMUP_ITERS < STEPS )) || { echo "WARMUP_ITERS must be less than STEPS." >&2; exit 2; }
(( N_EMBD % N_HEAD == 0 )) || { echo "N_EMBD must be divisible by N_HEAD." >&2; exit 2; }
(( DEPTH_ORDER <= N_LAYER )) || { echo "DEPTH_ORDER must not exceed N_LAYER." >&2; exit 2; }
if [[ -z "$WIDTH_SWEEP" ]]; then
  (( BASE_ROW_ORDER <= N_EMBD )) || { echo "BASE_ROW_ORDER must not exceed N_EMBD." >&2; exit 2; }
fi
(( GRADIENT_ACCUMULATION_STEPS % NUM_GPUS == 0 )) || { echo "Global gradient accumulation must be divisible by NUM_GPUS." >&2; exit 2; }
build_width_specs

if [[ -n "${THOG2_PYTHON:-}" ]]; then PYTHON_BIN="$THOG2_PYTHON"; elif [[ -x .venv/bin/python ]]; then PYTHON_BIN=".venv/bin/python"; else PYTHON_BIN="python"; fi
CHECKPOINT_FLAG="--no-activation-checkpointing"; [[ "$ACTIVATION_CHECKPOINTING" == true ]] && CHECKPOINT_FLAG="--activation-checkpointing"
WANDB_FLAG="--no-wandb"; [[ "$WANDB_ENABLED" == true ]] && WANDB_FLAG="--wandb"
DTYPE_ARGS=(); [[ "$DTYPE" != auto ]] && DTYPE_ARGS=(--dtype "$DTYPE")

# vvv THOG
run_sheet_once() {
  local run_n_embd="$1"
  local run_n_head="$2"
  local log_timestamp
  local resolved_json
  local artifact_name
  local log_path
  local -a train_args
  local -a command

  train_args=(
    --model-type sheet --run-mode "$RUN_MODE" --host-label "$HOST_LABEL" --run-name "$RUN_NAME"
    --dataset "$DATASET_NAME" --data-dir "$DATA_DIR" --max-iters "$STEPS" --batch-size "$BATCH_SIZE"
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" --eval-iters "$EVAL_ITERS"
    --eval-interval "$EVAL_INTERVAL" --log-interval "$LOG_INTERVAL" --checkpoint-interval "$CHECKPOINT_INTERVAL"
    --warmup-iters "$WARMUP_ITERS" --n-layer "$N_LAYER" --n-head "$run_n_head" --n-embd "$run_n_embd" --block-size "$BLOCK_SIZE"
    --depth-order "$DEPTH_ORDER" --base-row-order "$BASE_ROW_ORDER"
    --residual-init-policy "$RESIDUAL_INIT_POLICY" --residual-init-depth-source "$RESIDUAL_INIT_DEPTH_SOURCE" --residual-init-depth-value "$RESIDUAL_INIT_DEPTH_VALUE"
    "$CHECKPOINT_FLAG" --checkpoint-segment-size "$CHECKPOINT_SEGMENT_SIZE" "$WANDB_FLAG" --wandb-mode "$WANDB_MODE" "${DTYPE_ARGS[@]}" "${EXTRA_ARGS[@]}"
  )

  log_timestamp="$(date +%Y%m%d_%H%M%S)"
  resolved_json="$("$PYTHON_BIN" -m "$RUN_MODULE" "${train_args[@]}" --log-timestamp "$log_timestamp" --print-resolved-json)"
  artifact_name="$(printf '%s' "$resolved_json" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["artifact_name"])')"
  log_path="$(printf '%s' "$resolved_json" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["paths"]["log_path"])')"

  if (( NUM_GPUS == 1 )); then
    command=("$PYTHON_BIN" -m "$RUN_MODULE" "${train_args[@]}" --log-timestamp "$log_timestamp")
  else
    command=("$PYTHON_BIN" -m torch.distributed.run --standalone "--nproc-per-node=$NUM_GPUS" -m "$RUN_MODULE" "${train_args[@]}" --log-timestamp "$log_timestamp")
  fi

  cat <<EOF
THOG2 SHEET OpenWebText experiment
  artifact:                $artifact_name
  run mode:               $RUN_MODE
  runtime profile:        $RUNTIME_PROFILE
  dtype:                  $DTYPE
  geometry:               L$N_LAYER / H$run_n_head / D$run_n_embd / C$BLOCK_SIZE
  width sweep:            ${WIDTH_SWEEP:-single}
  sheet orders:           P$DEPTH_ORDER / Q$BASE_ROW_ORDER
  residual init:          $RESIDUAL_INIT_POLICY / $RESIDUAL_INIT_DEPTH_SOURCE / $RESIDUAL_INIT_DEPTH_VALUE
  GPUs:                   $NUM_GPUS
  mini-batch/GPU:         $BATCH_SIZE
  global accumulation:    $GRADIENT_ACCUMULATION_STEPS
  tokens/update:          $((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * BLOCK_SIZE))
  activation checkpoint:  $ACTIVATION_CHECKPOINTING
  checkpoint segment:     $CHECKPOINT_SEGMENT_SIZE
  checkpoint interval:    $CHECKPOINT_INTERVAL
  W&B:                    $WANDB_ENABLED ($WANDB_MODE)
  log:                    $log_path
EOF

  if [[ "$DRY_RUN" == true ]]; then
    "$PYTHON_BIN" -m "$RUN_MODULE" "${train_args[@]}" --log-timestamp "$log_timestamp" --dry-run
    printf 'DRY RUN:'; printf ' %q' "${command[@]}"; printf '\n'
    return 0
  fi

  mkdir -p "$(dirname "$log_path")"
  "${command[@]}" 2>&1 | tee "$log_path"
}

for width_spec in "${WIDTH_SPECS[@]}"; do
  parse_width_spec "$width_spec"
  run_sheet_once "$WIDTH_N_EMBD" "$WIDTH_N_HEAD"
done
# ^^^ THOG
