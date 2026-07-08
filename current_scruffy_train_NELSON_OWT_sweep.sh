#!/bin/bash
set -euo pipefail

# vvv THOG
# Run the NELSON OpenWebText comparison sweep through the consolidated OWT wrapper.
# Fairness target is equal training-token budget: every case uses the same steps, batch, accumulation, and block size.
# This is not equal compute time: DENSE pays for fully independent layers and will be much heavier per token.
# ^^^ THOG

cd "$(dirname "$0")"

EXPERIMENT_PREFIX="${THOG2_EXPERIMENT_PREFIX:-NELSON}"
STEPS="${NELSON_STEPS:-25}"
BATCH_SIZE="${NELSON_BATCH_SIZE:-3}"
GRADIENT_ACCUMULATION_STEPS="${NELSON_GRADIENT_ACCUMULATION_STEPS:-160}"
EVAL_ITERS="${NELSON_EVAL_ITERS:-5}"
EVAL_INTERVAL="${NELSON_EVAL_INTERVAL:-5}"
LOG_INTERVAL="${NELSON_LOG_INTERVAL:-1}"
WARMUP_ITERS="${NELSON_WARMUP_ITERS:-10}"
CHECKPOINT_INTERVAL="${NELSON_CHECKPOINT_INTERVAL:-0}"
N_LAYER="${NELSON_N_LAYER:-144}"
N_HEAD="${NELSON_N_HEAD:-12}"
N_EMBD="${NELSON_N_EMBD:-768}"
BLOCK_SIZE="${NELSON_BLOCK_SIZE:-1024}"
DEPTH_ORDER="${NELSON_DEPTH_ORDER:-32}"
BASE_ROW_ORDER="${NELSON_BASE_ROW_ORDER:-64}"
MLP_CHANNEL_ORDER="${NELSON_MLP_CHANNEL_ORDER:-256}"
RESIDUAL_INIT_POLICY="${NELSON_RESIDUAL_INIT_POLICY:-depth_scaled}"
RESIDUAL_INIT_DEPTH_SOURCE="${NELSON_RESIDUAL_INIT_DEPTH_SOURCE:-dof_implied_depth}"
RESIDUAL_INIT_DEPTH_VALUE="${NELSON_RESIDUAL_INIT_DEPTH_VALUE:-12}"
DRY_RUN="${NELSON_DRY_RUN:-false}"
STOP_ON_FAILURE="${NELSON_STOP_ON_FAILURE:-true}"
WANDB_MODE="${NELSON_WANDB_MODE:-online}"

usage() {
  cat <<EOF
Usage: $0 [options] [-- extra current_scruffy_train_OWT.sh args]

Runs the NELSON all-cases OWT sweep on scruffy with W&B logging enabled.

Options:
  -n STEPS=${STEPS}
  -b BATCH_SIZE=${BATCH_SIZE}
  -A GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}
  -C BLOCK_SIZE=${BLOCK_SIZE}
  -L N_LAYER=${N_LAYER}
  -H N_HEAD=${N_HEAD}
  -D N_EMBD=${N_EMBD}
  -P DEPTH_ORDER=${DEPTH_ORDER}
  -Q BASE_ROW_ORDER=${BASE_ROW_ORDER}
  -Y MLP_CHANNEL_ORDER=${MLP_CHANNEL_ORDER}
  -u EVAL_ITERS=${EVAL_ITERS}
  -e EVAL_INTERVAL=${EVAL_INTERVAL}
  -w WARMUP_ITERS=${WARMUP_ITERS}
  -k CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL}
  -x DRY_RUN=${DRY_RUN}
  -s STOP_ON_FAILURE=${STOP_ON_FAILURE}
  -h show this help
EOF
}

while getopts ":n:b:A:C:L:H:D:P:Q:Y:u:e:w:k:x:s:h" option; do
  case "$option" in
    n) STEPS="$OPTARG" ;;
    b) BATCH_SIZE="$OPTARG" ;;
    A) GRADIENT_ACCUMULATION_STEPS="$OPTARG" ;;
    C) BLOCK_SIZE="$OPTARG" ;;
    L) N_LAYER="$OPTARG" ;;
    H) N_HEAD="$OPTARG" ;;
    D) N_EMBD="$OPTARG" ;;
    P) DEPTH_ORDER="$OPTARG" ;;
    Q) BASE_ROW_ORDER="$OPTARG" ;;
    Y) MLP_CHANNEL_ORDER="$OPTARG" ;;
    u) EVAL_ITERS="$OPTARG" ;;
    e) EVAL_INTERVAL="$OPTARG" ;;
    w) WARMUP_ITERS="$OPTARG" ;;
    k) CHECKPOINT_INTERVAL="$OPTARG" ;;
    x) DRY_RUN="$OPTARG" ;;
    s) STOP_ON_FAILURE="$OPTARG" ;;
    h) usage; exit 0 ;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 2 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
if [[ "${1:-}" == "--" ]]; then shift; fi
EXTRA_ARGS=("$@")

case "$DRY_RUN" in true|false) ;; *) echo "DRY_RUN must be true or false." >&2; exit 2 ;; esac
case "$STOP_ON_FAILURE" in true|false) ;; *) echo "STOP_ON_FAILURE must be true or false." >&2; exit 2 ;; esac

TOKENS_PER_UPDATE=$((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * BLOCK_SIZE))
TOTAL_TRAINING_TOKENS=$((TOKENS_PER_UPDATE * STEPS))
SWEEP_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_LOG_DIR="logs/${SWEEP_TIMESTAMP}_${EXPERIMENT_PREFIX}_SCRUFFY_OWT_SWEEP"
mkdir -p "$SWEEP_LOG_DIR"
SUMMARY_PATH="$SWEEP_LOG_DIR/summary.tsv"
printf 'case\tstatus\tseconds\n' > "$SUMMARY_PATH"

COMMON_ARGS=(
  -n "$STEPS"
  -b "$BATCH_SIZE"
  -A "$GRADIENT_ACCUMULATION_STEPS"
  -C "$BLOCK_SIZE"
  -L "$N_LAYER"
  -H "$N_HEAD"
  -D "$N_EMBD"
  -P "$DEPTH_ORDER"
  -Q "$BASE_ROW_ORDER"
  -Y "$MLP_CHANNEL_ORDER"
  -u "$EVAL_ITERS"
  -e "$EVAL_INTERVAL"
  -l "$LOG_INTERVAL"
  -w "$WARMUP_ITERS"
  -k "$CHECKPOINT_INTERVAL"
  -r "$RESIDUAL_INIT_POLICY"
  -z "$RESIDUAL_INIT_DEPTH_SOURCE"
  -Z "$RESIDUAL_INIT_DEPTH_VALUE"
  -I wandb
  -M "$WANDB_MODE"
  -W true
  -x "$DRY_RUN"
)

run_case() {
  local label="$1"
  shift
  local start_time end_time elapsed status
  echo "===== START ${label} tokens/update=${TOKENS_PER_UPDATE} total_tokens=${TOTAL_TRAINING_TOKENS} ====="
  start_time="$(date +%s)"
  set +e
  THOG2_EXPERIMENT_PREFIX="$EXPERIMENT_PREFIX" bash current_scruffy_train_OWT.sh "${COMMON_ARGS[@]}" "$@" "${EXTRA_ARGS[@]}" 2>&1 | tee "$SWEEP_LOG_DIR/${label}.combined.log"
  status=${PIPESTATUS[0]}
  set -e
  end_time="$(date +%s)"
  elapsed=$((end_time - start_time))
  printf '%s\t%s\t%s\n' "$label" "$status" "$elapsed" >> "$SUMMARY_PATH"
  echo "===== END ${label} status=${status} seconds=${elapsed} ====="
  if [[ "$status" != 0 && "$STOP_ON_FAILURE" == true ]]; then
    exit "$status"
  fi
}

cat <<EOF
NELSON scruffy OWT sweep
  wandb:               enabled mode=${WANDB_MODE}
  fairness:            equal training tokens
  steps:               ${STEPS}
  tokens/update:       ${TOKENS_PER_UPDATE}
  total train tokens:  ${TOTAL_TRAINING_TOKENS}
  shape:               L${N_LAYER} H${N_HEAD} D${N_EMBD} C${BLOCK_SIZE} P${DEPTH_ORDER} Q${BASE_ROW_ORDER} Y${MLP_CHANNEL_ORDER}
  summary:             ${SUMMARY_PATH}
EOF

run_case DENSE -O dense -p curve -B chebyshev
run_case CHEBY_SHEET_COL -O sheet -p legacy_sheet_col -B chebyshev
run_case DCT_SHEET_COL -O sheet -p legacy_sheet_col -B dct
run_case CHEBY_CURVE -O sheet -p curve -B chebyshev
run_case DCT_CURVE -O sheet -p curve -B dct
run_case CHEBY_MLP_BLOCK -O sheet -p mlp_block -B chebyshev
run_case DCT_MLP_BLOCK -O sheet -p mlp_block -B dct
run_case CHEBY_HEAD_BLOCK -O sheet -p head_aware_block -B chebyshev
run_case DCT_HEAD_BLOCK -O sheet -p head_aware_block -B dct
run_case CHEBY_BLOCK -O sheet -p block -B chebyshev
run_case DCT_BLOCK -O sheet -p block -B dct

cat "$SUMMARY_PATH"
