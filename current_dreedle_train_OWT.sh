#!/bin/bash
set -euo pipefail

# vvv THOG
# Current dreedle OpenWebText training wrapper.
# Defaults are deliberately non-tiny: SHEET L144 H12 D768 C1024, batch 3, global accumulation 160.
# Dreedle runtime defaults: float16, sdpa. Dense baseline is available with -O dense.
# Logging is explicit: -I tensorboard|wandb|none. TensorBoard writes under THOG2_CURVE_ROOT, default curves/.
# ^^^ THOG

cd "$(dirname "$0")"

RUN_MODULE="run_thog2_owt_stage8"
HOST_LABEL="dreedle"
MODEL_TYPE="sheet"
RUN_MODE="fresh"
RUN_NAME=""
DATASET_NAME="openwebtext"
DATA_DIR="${THOG2_OWT_DATA_DIR:-$HOME/git/thog/data/openwebtext}"

GEOMETRY_PRESET="curve"
BASIS_FAMILY="chebyshev"
ATTENTION_GEOMETRY=""
MLP_GEOMETRY=""

STEPS=250
BATCH_SIZE=3
GRADIENT_ACCUMULATION_STEPS=160
N_LAYER=144
N_HEAD=12
N_EMBD=768
BLOCK_SIZE=1024
DEPTH_ORDER=32
BASE_ROW_ORDER=64
CHECKPOINT_SEGMENT_SIZE=12

DTYPE="float16"
ATTENTION_BACKEND="sdpa"
INSTRUMENTATION="tensorboard"
WANDB_MODE="online"
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

Compact options:
  -p GEOMETRY_PRESET=${GEOMETRY_PRESET}          legacy_sheet_col | curve | head_aware_block | mlp_block | block
  -B BASIS_FAMILY=${BASIS_FAMILY}                chebyshev | dct
  -a ATTENTION_GEOMETRY=${ATTENTION_GEOMETRY:-preset default}
  -m MLP_GEOMETRY=${MLP_GEOMETRY:-preset default}

Shape/runtime:
  -L N_LAYER=${N_LAYER}
  -H N_HEAD=${N_HEAD}
  -D N_EMBD=${N_EMBD}
  -C BLOCK_SIZE=${BLOCK_SIZE}
  -P DEPTH_ORDER=${DEPTH_ORDER}
  -Q BASE_ROW_ORDER=${BASE_ROW_ORDER}
  -S CHECKPOINT_SEGMENT_SIZE=${CHECKPOINT_SEGMENT_SIZE}
  -T DTYPE=${DTYPE}                              float32 | float16 | bfloat16
  -K ATTENTION_BACKEND=${ATTENTION_BACKEND}      auto | flash2 | sdpa | math

Data/logging:
  -d DATASET_NAME=${DATASET_NAME}
  -t DATA_DIR=${DATA_DIR}
  -I INSTRUMENTATION=${INSTRUMENTATION}          tensorboard | wandb | none
  -M WANDB_MODE=${WANDB_MODE}                    online | offline | disabled
  -x DRY_RUN=${DRY_RUN}
  -h show this help
EOF
}

while getopts ":O:q:g:n:b:A:p:B:a:m:L:H:D:C:P:Q:S:T:K:d:t:I:M:x:h" option; do
  case "$option" in
    O) MODEL_TYPE="$OPTARG" ;;
    q) RUN_MODE="$OPTARG" ;;
    g) RUN_NAME="$OPTARG" ;;
    n) STEPS="$OPTARG" ;;
    b) BATCH_SIZE="$OPTARG" ;;
    A) GRADIENT_ACCUMULATION_STEPS="$OPTARG" ;;
    p) GEOMETRY_PRESET="$OPTARG" ;;
    B) BASIS_FAMILY="$OPTARG" ;;
    a) ATTENTION_GEOMETRY="$OPTARG" ;;
    m) MLP_GEOMETRY="$OPTARG" ;;
    L) N_LAYER="$OPTARG"; N_LAYER_EXPLICIT=true ;;
    H) N_HEAD="$OPTARG"; N_HEAD_EXPLICIT=true ;;
    D) N_EMBD="$OPTARG"; N_EMBD_EXPLICIT=true ;;
    C) BLOCK_SIZE="$OPTARG" ;;
    P) DEPTH_ORDER="$OPTARG" ;;
    Q) BASE_ROW_ORDER="$OPTARG" ;;
    S) CHECKPOINT_SEGMENT_SIZE="$OPTARG" ;;
    T) DTYPE="$OPTARG" ;;
    K) ATTENTION_BACKEND="$OPTARG" ;;
    d) DATASET_NAME="$OPTARG"; DATA_DIR="data/$OPTARG" ;;
    t) DATA_DIR="$OPTARG" ;;
    I) INSTRUMENTATION="$OPTARG" ;;
    M) WANDB_MODE="$OPTARG" ;;
    x) DRY_RUN="$OPTARG" ;;
    h) usage; exit 0 ;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 2 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
if [[ "${1:-}" == "--" ]]; then shift; fi
EXTRA_ARGS=("$@")

case "$MODEL_TYPE" in dense|sheet) ;; *) echo "MODEL_TYPE must be dense or sheet." >&2; exit 2 ;; esac
case "$RUN_MODE" in fresh|resume) ;; *) echo "RUN_MODE must be fresh or resume." >&2; exit 2 ;; esac
case "$GEOMETRY_PRESET" in legacy_sheet_col|curve|head_aware_block|mlp_block|block) ;; *) echo "Bad GEOMETRY_PRESET: $GEOMETRY_PRESET" >&2; exit 2 ;; esac
case "$BASIS_FAMILY" in chebyshev|dct) ;; *) echo "BASIS_FAMILY must be chebyshev or dct." >&2; exit 2 ;; esac
case "$ATTENTION_BACKEND" in auto|flash2|sdpa|math) ;; *) echo "Bad ATTENTION_BACKEND: $ATTENTION_BACKEND" >&2; exit 2 ;; esac
case "$INSTRUMENTATION" in tensorboard|wandb|none) ;; *) echo "INSTRUMENTATION must be tensorboard, wandb, or none." >&2; exit 2 ;; esac
case "$WANDB_MODE" in online|offline|disabled) ;; *) echo "WANDB_MODE must be online, offline, or disabled." >&2; exit 2 ;; esac
case "$DRY_RUN" in true|false) ;; *) echo "DRY_RUN must be true or false." >&2; exit 2 ;; esac

if [[ "$MODEL_TYPE" == dense ]]; then
  [[ "$N_LAYER_EXPLICIT" == false ]] && N_LAYER=12
  [[ "$N_HEAD_EXPLICIT" == false ]] && N_HEAD=12
  [[ "$N_EMBD_EXPLICIT" == false ]] && N_EMBD=768
fi

BASIS_TAG="CHEBY"
[[ "$BASIS_FAMILY" == dct ]] && BASIS_TAG="DCT"
PRESET_TAG="${GEOMETRY_PRESET^^}"
[[ "$GEOMETRY_PRESET" == legacy_sheet_col ]] && PRESET_TAG="SHEET_COL"
[[ "$MODEL_TYPE" == dense ]] && RUN_TAG="DENSE" || RUN_TAG="${BASIS_TAG}_${PRESET_TAG}"
[[ -z "$RUN_NAME" ]] && RUN_NAME="${RUN_TAG}_OWT"

OPTIONAL_ARGS=()
[[ -n "$ATTENTION_GEOMETRY" ]] && OPTIONAL_ARGS+=(--attention-geometry "$ATTENTION_GEOMETRY")
[[ -n "$MLP_GEOMETRY" ]] && OPTIONAL_ARGS+=(--mlp-geometry "$MLP_GEOMETRY")

export THOG2_INSTRUMENTATION="$INSTRUMENTATION"
export THOG2_CURVE_ROOT="${THOG2_CURVE_ROOT:-curves}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

COMMAND=(
  python -m "$RUN_MODULE"
  --model-type "$MODEL_TYPE"
  --run-mode "$RUN_MODE"
  --host-label "$HOST_LABEL"
  --run-name "$RUN_NAME"
  --dataset "$DATASET_NAME"
  --data-dir "$DATA_DIR"
  --max-iters "$STEPS"
  --batch-size "$BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --n-layer "$N_LAYER"
  --n-head "$N_HEAD"
  --n-embd "$N_EMBD"
  --block-size "$BLOCK_SIZE"
  --depth-order "$DEPTH_ORDER"
  --base-row-order "$BASE_ROW_ORDER"
  --geometry-preset "$GEOMETRY_PRESET"
  --basis-family "$BASIS_FAMILY"
  --attention-backend "$ATTENTION_BACKEND"
  --dtype "$DTYPE"
  --wandb
  --wandb-mode "$WANDB_MODE"
  --activation-checkpointing
  --checkpoint-segment-size "$CHECKPOINT_SEGMENT_SIZE"
  "${OPTIONAL_ARGS[@]}"
  "${EXTRA_ARGS[@]}"
)

cat <<EOF
dreedle OWT train
  model/preset/basis: $MODEL_TYPE / $GEOMETRY_PRESET / $BASIS_FAMILY
  backend/dtype:      $ATTENTION_BACKEND / $DTYPE
  instrumentation:    $INSTRUMENTATION  (tensorboard root: $THOG2_CURVE_ROOT)
  run:                $RUN_NAME
  shape:              L$N_LAYER H$N_HEAD D$N_EMBD C$BLOCK_SIZE P$DEPTH_ORDER Q$BASE_ROW_ORDER
  batch/accum:        $BATCH_SIZE / $GRADIENT_ACCUMULATION_STEPS
EOF

if [[ "$DRY_RUN" == true ]]; then
  "${COMMAND[@]}" --dry-run
  printf 'DRY RUN:'; printf ' %q' "${COMMAND[@]}"; printf '\n'
  exit 0
fi

"${COMMAND[@]}"
