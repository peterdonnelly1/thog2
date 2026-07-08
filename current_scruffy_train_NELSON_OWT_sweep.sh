#!/bin/bash
set -euo pipefail

# vvv THOG
# Run the NELSON OpenWebText comparison sweep through the consolidated OWT wrapper.
# Fairness target is near-equal training-token budget per optimizer step, not equal depth, parameter count, or wall-clock time.
# DENSE defaults to a practical L12 baseline; compact cases use larger microbatches to work the GPU harder without changing tokens/update much.
# ^^^ THOG

cd "$(dirname "$0")"

EXPERIMENT_PREFIX="${THOG2_EXPERIMENT_PREFIX:-NELSON}"
STEPS="${NELSON_STEPS:-25}"
BLOCK_SIZE="${NELSON_BLOCK_SIZE:-1024}"
DENSE_BATCH_SIZE="${NELSON_DENSE_BATCH_SIZE:-3}"
DENSE_GRADIENT_ACCUMULATION_STEPS="${NELSON_DENSE_GRADIENT_ACCUMULATION_STEPS:-160}"
COMPACT_BATCH_SIZE="${NELSON_COMPACT_BATCH_SIZE:-4}"
COMPACT_GRADIENT_ACCUMULATION_STEPS="${NELSON_COMPACT_GRADIENT_ACCUMULATION_STEPS:-120}"
TOKEN_TOLERANCE_PCT="${NELSON_TOKEN_TOLERANCE_PCT:-3}"
EVAL_ITERS="${NELSON_EVAL_ITERS:-5}"
EVAL_INTERVAL="${NELSON_EVAL_INTERVAL:-5}"
LOG_INTERVAL="${NELSON_LOG_INTERVAL:-1}"
WARMUP_ITERS="${NELSON_WARMUP_ITERS:-10}"
CHECKPOINT_INTERVAL="${NELSON_CHECKPOINT_INTERVAL:-0}"
N_LAYER="${NELSON_N_LAYER:-144}"
N_HEAD="${NELSON_N_HEAD:-12}"
N_EMBD="${NELSON_N_EMBD:-768}"
DENSE_N_LAYER="${NELSON_DENSE_N_LAYER:-12}"
DENSE_N_HEAD="${NELSON_DENSE_N_HEAD:-12}"
DENSE_N_EMBD="${NELSON_DENSE_N_EMBD:-768}"
DEPTH_ORDER="${NELSON_DEPTH_ORDER:-32}"
BASE_ROW_ORDER="${NELSON_BASE_ROW_ORDER:-64}"
MLP_CHANNEL_ORDER="${NELSON_MLP_CHANNEL_ORDER:-256}"
RESIDUAL_INIT_POLICY="${NELSON_RESIDUAL_INIT_POLICY:-depth_scaled}"
RESIDUAL_INIT_DEPTH_SOURCE="${NELSON_RESIDUAL_INIT_DEPTH_SOURCE:-dof_implied_depth}"
RESIDUAL_INIT_DEPTH_VALUE="${NELSON_RESIDUAL_INIT_DEPTH_VALUE:-12}"
DRY_RUN="${NELSON_DRY_RUN:-false}"
STOP_ON_FAILURE="${NELSON_STOP_ON_FAILURE:-true}"
WANDB_MODE="${NELSON_WANDB_MODE:-online}"
RUN_DENSE="${NELSON_RUN_DENSE:-true}"

usage() {
  cat <<EOF
Usage: $0 [options] [-- extra current_scruffy_train_OWT.sh args]

Runs the NELSON all-cases OWT sweep on scruffy with W&B logging enabled.

Options:
  -n STEPS=${STEPS}
  -C BLOCK_SIZE=${BLOCK_SIZE}
  -b COMPACT_BATCH_SIZE=${COMPACT_BATCH_SIZE}
  -A COMPACT_GRADIENT_ACCUMULATION_STEPS=${COMPACT_GRADIENT_ACCUMULATION_STEPS}
  -L N_LAYER=${N_LAYER}                         compact case layer count
  -H N_HEAD=${N_HEAD}                           compact case head count
  -D N_EMBD=${N_EMBD}                           compact case embedding width
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

Environment overrides:
  NELSON_DENSE_BATCH_SIZE=${DENSE_BATCH_SIZE}
  NELSON_DENSE_GRADIENT_ACCUMULATION_STEPS=${DENSE_GRADIENT_ACCUMULATION_STEPS}
  NELSON_DENSE_N_LAYER=${DENSE_N_LAYER}
  NELSON_DENSE_N_HEAD=${DENSE_N_HEAD}
  NELSON_DENSE_N_EMBD=${DENSE_N_EMBD}
  NELSON_TOKEN_TOLERANCE_PCT=${TOKEN_TOLERANCE_PCT}
  NELSON_RUN_DENSE=${RUN_DENSE}
EOF
}

while getopts ":n:C:b:A:L:H:D:P:Q:Y:u:e:w:k:x:s:h" option; do
  case "$option" in
    n) STEPS="$OPTARG" ;;
    C) BLOCK_SIZE="$OPTARG" ;;
    b) COMPACT_BATCH_SIZE="$OPTARG" ;;
    A) COMPACT_GRADIENT_ACCUMULATION_STEPS="$OPTARG" ;;
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
case "$RUN_DENSE" in true|false) ;; *) echo "NELSON_RUN_DENSE must be true or false." >&2; exit 2 ;; esac
if [[ ! "$TOKEN_TOLERANCE_PCT" =~ ^[0-9]+$ ]]; then
  echo "NELSON_TOKEN_TOLERANCE_PCT must be a non-negative integer percent." >&2
  exit 2
fi

DENSE_TOKENS_PER_UPDATE=$((DENSE_BATCH_SIZE * DENSE_GRADIENT_ACCUMULATION_STEPS * BLOCK_SIZE))
COMPACT_TOKENS_PER_UPDATE=$((COMPACT_BATCH_SIZE * COMPACT_GRADIENT_ACCUMULATION_STEPS * BLOCK_SIZE))
MAX_TOKENS_PER_UPDATE="$DENSE_TOKENS_PER_UPDATE"
if (( COMPACT_TOKENS_PER_UPDATE > MAX_TOKENS_PER_UPDATE )); then
  MAX_TOKENS_PER_UPDATE="$COMPACT_TOKENS_PER_UPDATE"
fi
TOKEN_DELTA=$((DENSE_TOKENS_PER_UPDATE - COMPACT_TOKENS_PER_UPDATE))
if (( TOKEN_DELTA < 0 )); then TOKEN_DELTA=$((-TOKEN_DELTA)); fi
TOKEN_TOLERANCE=$((MAX_TOKENS_PER_UPDATE * TOKEN_TOLERANCE_PCT / 100))
if (( TOKEN_DELTA > TOKEN_TOLERANCE )); then
  echo "Dense and compact tokens/update differ beyond ${TOKEN_TOLERANCE_PCT}%: dense=${DENSE_TOKENS_PER_UPDATE}, compact=${COMPACT_TOKENS_PER_UPDATE}" >&2
  echo "Use near-equal products: batch * accumulation * block_size." >&2
  exit 2
fi
TOTAL_DENSE_TRAINING_TOKENS=$((DENSE_TOKENS_PER_UPDATE * STEPS))
TOTAL_COMPACT_TRAINING_TOKENS=$((COMPACT_TOKENS_PER_UPDATE * STEPS))
SWEEP_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_LOG_DIR="logs/${SWEEP_TIMESTAMP}_${EXPERIMENT_PREFIX}_SCRUFFY_OWT_SWEEP"
mkdir -p "$SWEEP_LOG_DIR"
SUMMARY_PATH="$SWEEP_LOG_DIR/summary.tsv"
printf 'case\tstatus\tseconds\ttokens_per_update\ttotal_tokens\n' > "$SUMMARY_PATH"

BASE_ARGS=(
  -n "$STEPS"
  -C "$BLOCK_SIZE"
  -u "$EVAL_ITERS"
  -e "$EVAL_INTERVAL"
  -l "$LOG_INTERVAL"
  -w "$WARMUP_ITERS"
  -k "$CHECKPOINT_INTERVAL"
  -I wandb
  -M "$WANDB_MODE"
  -W true
  -x "$DRY_RUN"
)

COMPACT_ARGS=(
  "${BASE_ARGS[@]}"
  -b "$COMPACT_BATCH_SIZE"
  -A "$COMPACT_GRADIENT_ACCUMULATION_STEPS"
  -L "$N_LAYER"
  -H "$N_HEAD"
  -D "$N_EMBD"
  -P "$DEPTH_ORDER"
  -Q "$BASE_ROW_ORDER"
  -Y "$MLP_CHANNEL_ORDER"
  -r "$RESIDUAL_INIT_POLICY"
  -z "$RESIDUAL_INIT_DEPTH_SOURCE"
  -Z "$RESIDUAL_INIT_DEPTH_VALUE"
)

DENSE_ARGS=(
  "${BASE_ARGS[@]}"
  -b "$DENSE_BATCH_SIZE"
  -A "$DENSE_GRADIENT_ACCUMULATION_STEPS"
  -L "$DENSE_N_LAYER"
  -H "$DENSE_N_HEAD"
  -D "$DENSE_N_EMBD"
)

run_case() {
  local label="$1"
  local tokens_per_update="$2"
  local total_tokens="$3"
  shift 3
  local start_time end_time elapsed status
  echo "===== START ${label} tokens/update=${tokens_per_update} total_tokens=${total_tokens} ====="
  start_time="$(date +%s)"
  set +e
  THOG2_EXPERIMENT_PREFIX="$EXPERIMENT_PREFIX" bash current_scruffy_train_OWT.sh "$@" "${EXTRA_ARGS[@]}" 2>&1 | tee "$SWEEP_LOG_DIR/${label}.combined.log"
  status=${PIPESTATUS[0]}
  set -e
  end_time="$(date +%s)"
  elapsed=$((end_time - start_time))
  printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$status" "$elapsed" "$tokens_per_update" "$total_tokens" >> "$SUMMARY_PATH"
  echo "===== END ${label} status=${status} seconds=${elapsed} ====="
  if [[ "$status" != 0 && "$STOP_ON_FAILURE" == true ]]; then
    exit "$status"
  fi
}

cat <<EOF
NELSON scruffy OWT sweep
  wandb:               enabled mode=${WANDB_MODE}
  fairness:            near-equal tokens/update within ${TOKEN_TOLERANCE_PCT}%; not equal depth, parameter count, or wall-clock time
  dense tokens/update: ${DENSE_TOKENS_PER_UPDATE}
  compact tokens/update: ${COMPACT_TOKENS_PER_UPDATE}
  token delta:         ${TOKEN_DELTA}
  steps:               ${STEPS}
  dense total tokens:  ${TOTAL_DENSE_TRAINING_TOKENS}
  compact total tokens:${TOTAL_COMPACT_TRAINING_TOKENS}
  dense baseline:      L${DENSE_N_LAYER} H${DENSE_N_HEAD} D${DENSE_N_EMBD} C${BLOCK_SIZE} b${DENSE_BATCH_SIZE} A${DENSE_GRADIENT_ACCUMULATION_STEPS} run=${RUN_DENSE}
  compact shape:       L${N_LAYER} H${N_HEAD} D${N_EMBD} C${BLOCK_SIZE} P${DEPTH_ORDER} Q${BASE_ROW_ORDER} Y${MLP_CHANNEL_ORDER} b${COMPACT_BATCH_SIZE} A${COMPACT_GRADIENT_ACCUMULATION_STEPS}
  summary:             ${SUMMARY_PATH}
EOF

if [[ "$RUN_DENSE" == true ]]; then
  run_case DENSE_L${DENSE_N_LAYER} "$DENSE_TOKENS_PER_UPDATE" "$TOTAL_DENSE_TRAINING_TOKENS" "${DENSE_ARGS[@]}" -O dense -p curve -B chebyshev
fi
run_case CHEBY_SHEET_COL "$COMPACT_TOKENS_PER_UPDATE" "$TOTAL_COMPACT_TRAINING_TOKENS" "${COMPACT_ARGS[@]}" -O sheet -p legacy_sheet_col -B chebyshev
run_case DCT_SHEET_COL "$COMPACT_TOKENS_PER_UPDATE" "$TOTAL_COMPACT_TRAINING_TOKENS" "${COMPACT_ARGS[@]}" -O sheet -p legacy_sheet_col -B dct
run_case CHEBY_CURVE "$COMPACT_TOKENS_PER_UPDATE" "$TOTAL_COMPACT_TRAINING_TOKENS" "${COMPACT_ARGS[@]}" -O sheet -p curve -B chebyshev
run_case DCT_CURVE "$COMPACT_TOKENS_PER_UPDATE" "$TOTAL_COMPACT_TRAINING_TOKENS" "${COMPACT_ARGS[@]}" -O sheet -p curve -B dct
run_case CHEBY_MLP_BLOCK "$COMPACT_TOKENS_PER_UPDATE" "$TOTAL_COMPACT_TRAINING_TOKENS" "${COMPACT_ARGS[@]}" -O sheet -p mlp_block -B chebyshev
run_case DCT_MLP_BLOCK "$COMPACT_TOKENS_PER_UPDATE" "$TOTAL_COMPACT_TRAINING_TOKENS" "${COMPACT_ARGS[@]}" -O sheet -p mlp_block -B dct
run_case CHEBY_HEAD_BLOCK "$COMPACT_TOKENS_PER_UPDATE" "$TOTAL_COMPACT_TRAINING_TOKENS" "${COMPACT_ARGS[@]}" -O sheet -p head_aware_block -B chebyshev
run_case DCT_HEAD_BLOCK "$COMPACT_TOKENS_PER_UPDATE" "$TOTAL_COMPACT_TRAINING_TOKENS" "${COMPACT_ARGS[@]}" -O sheet -p head_aware_block -B dct
run_case CHEBY_BLOCK "$COMPACT_TOKENS_PER_UPDATE" "$TOTAL_COMPACT_TRAINING_TOKENS" "${COMPACT_ARGS[@]}" -O sheet -p block -B chebyshev
run_case DCT_BLOCK "$COMPACT_TOKENS_PER_UPDATE" "$TOTAL_COMPACT_TRAINING_TOKENS" "${COMPACT_ARGS[@]}" -O sheet -p block -B dct

cat "$SUMMARY_PATH"
