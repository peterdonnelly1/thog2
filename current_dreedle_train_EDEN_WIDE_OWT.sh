#!/bin/bash
set -euo pipefail

# vvv THOG
cd "$(dirname "$0")"

SHEET_WRAPPER="./current_scruffy_train_SHEET_OWT.sh"
DATA_DIR="${THOG2_OWT_DATA_DIR:-$HOME/git/thog/data/openwebtext}"
STEPS=20
BATCH_SIZE=12
EVAL_ITERS=5
EVAL_INTERVAL=10
LOG_INTERVAL=1
WARMUP_ITERS=2
CHECKPOINT_INTERVAL=0
GRADIENT_ACCUMULATION_STEPS=160
NUM_GPUS=1
N_LAYER=144
BLOCK_SIZE=256
ACTIVATION_CHECKPOINTING=true
CHECKPOINT_SEGMENT_SIZE=12
WANDB_MODE="online"
WANDB_ENABLED=true
DRY_RUN=false
CUDA_DEVICE_DEFAULT=0

usage() {
  cat <<EOF
Usage: $0 [options]

Runs the overnight dreedle EDEN SHEET capacity sweep:
  EDEN_WIDE2      D1792/H28..D2560/H40  P64  Q128
  EDEN_WIDE_CAP   D1536/H24,D2048/H32   P128 Q256
  EDEN_WIDE_CAP2  D2048/H32             P144 Q384

Options:
  -t DATA_DIR=${DATA_DIR}
  -n STEPS=${STEPS}
  -b BATCH_SIZE=${BATCH_SIZE}
  -u EVAL_ITERS=${EVAL_ITERS}
  -e EVAL_INTERVAL=${EVAL_INTERVAL}
  -l LOG_INTERVAL=${LOG_INTERVAL}
  -w WARMUP_ITERS=${WARMUP_ITERS}
  -k CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL}
  -A GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}
  -G NUM_GPUS=${NUM_GPUS}
  -L N_LAYER=${N_LAYER}
  -C BLOCK_SIZE=${BLOCK_SIZE}
  -p ACTIVATION_CHECKPOINTING=${ACTIVATION_CHECKPOINTING}
  -S CHECKPOINT_SEGMENT_SIZE=${CHECKPOINT_SEGMENT_SIZE}
  -M WANDB_MODE=${WANDB_MODE}                 online | offline | disabled
  -W WANDB_ENABLED=${WANDB_ENABLED}
  -x DRY_RUN=${DRY_RUN}
  -c CUDA_VISIBLE_DEVICES default=${CUDA_DEVICE_DEFAULT}
  -h show this help
EOF
}

while getopts ":t:n:b:u:e:l:w:k:A:G:L:C:p:S:M:W:x:c:h" option; do
  case "$option" in
    t) DATA_DIR="$OPTARG" ;;
    n) STEPS="$OPTARG" ;;
    b) BATCH_SIZE="$OPTARG" ;;
    u) EVAL_ITERS="$OPTARG" ;;
    e) EVAL_INTERVAL="$OPTARG" ;;
    l) LOG_INTERVAL="$OPTARG" ;;
    w) WARMUP_ITERS="$OPTARG" ;;
    k) CHECKPOINT_INTERVAL="$OPTARG" ;;
    A) GRADIENT_ACCUMULATION_STEPS="$OPTARG" ;;
    G) NUM_GPUS="$OPTARG" ;;
    L) N_LAYER="$OPTARG" ;;
    C) BLOCK_SIZE="$OPTARG" ;;
    p) ACTIVATION_CHECKPOINTING="$OPTARG" ;;
    S) CHECKPOINT_SEGMENT_SIZE="$OPTARG" ;;
    M) WANDB_MODE="$OPTARG" ;;
    W) WANDB_ENABLED="$OPTARG" ;;
    x) DRY_RUN="$OPTARG" ;;
    c) CUDA_DEVICE_DEFAULT="$OPTARG" ;;
    h) usage; exit 0 ;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 2 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage >&2; exit 2 ;;
  esac
done

validate_positive_uint() { [[ "$1" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid $2: $1; expected a positive integer." >&2; exit 2; }; }
validate_nonnegative_uint() { [[ "$1" =~ ^[0-9]+$ ]] || { echo "Invalid $2: $1; expected a non-negative integer." >&2; exit 2; }; }
validate_true_false() { case "$1" in true|false) ;; *) echo "Invalid $2: $1; expected true or false." >&2; exit 2 ;; esac; }

for setting in "$STEPS" "$BATCH_SIZE" "$EVAL_ITERS" "$EVAL_INTERVAL" "$LOG_INTERVAL" "$GRADIENT_ACCUMULATION_STEPS" "$NUM_GPUS" "$N_LAYER" "$BLOCK_SIZE" "$CHECKPOINT_SEGMENT_SIZE"; do
  validate_positive_uint "$setting" "numeric setting"
done
validate_nonnegative_uint "$WARMUP_ITERS" "WARMUP_ITERS"
validate_nonnegative_uint "$CHECKPOINT_INTERVAL" "CHECKPOINT_INTERVAL"
validate_true_false "$ACTIVATION_CHECKPOINTING" "ACTIVATION_CHECKPOINTING"
validate_true_false "$WANDB_ENABLED" "WANDB_ENABLED"
validate_true_false "$DRY_RUN" "DRY_RUN"
case "$WANDB_MODE" in online|offline|disabled) ;; *) echo "WANDB_MODE must be online, offline, or disabled." >&2; exit 2 ;; esac
[[ -f "$SHEET_WRAPPER" ]] || { echo "Missing SHEET wrapper: $SHEET_WRAPPER" >&2; exit 2; }
[[ -f "$DATA_DIR/train.bin" && -f "$DATA_DIR/val.bin" ]] || { echo "DATA_DIR must contain train.bin and val.bin: $DATA_DIR" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$CUDA_DEVICE_DEFAULT}"

mkdir -p evidence
git rev-parse HEAD | tee evidence/EDEN_WIDE_ALL_commit_sha.txt
git log --oneline -1 | tee evidence/EDEN_WIDE_ALL_commit.txt

run_eden_group() {
  local run_name="$1"
  local width_sweep="$2"
  local depth_order="$3"
  local base_row_order="$4"

  echo
  echo "===== ${run_name}: widths=${width_sweep} P${depth_order}/Q${base_row_order} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ====="

  "$SHEET_WRAPPER" \
    -R dreedle \
    -g "$run_name" \
    -n "$STEPS" \
    -b "$BATCH_SIZE" \
    -t "$DATA_DIR" \
    -u "$EVAL_ITERS" \
    -e "$EVAL_INTERVAL" \
    -l "$LOG_INTERVAL" \
    -w "$WARMUP_ITERS" \
    -k "$CHECKPOINT_INTERVAL" \
    -A "$GRADIENT_ACCUMULATION_STEPS" \
    -G "$NUM_GPUS" \
    -L "$N_LAYER" \
    -Y "$width_sweep" \
    -C "$BLOCK_SIZE" \
    -P "$depth_order" \
    -Q "$base_row_order" \
    -p "$ACTIVATION_CHECKPOINTING" \
    -S "$CHECKPOINT_SEGMENT_SIZE" \
    -M "$WANDB_MODE" \
    -W "$WANDB_ENABLED" \
    -x "$DRY_RUN"
}

run_eden_group "EDEN_WIDE2" "1792:2560:256" "64" "128"
run_eden_group "EDEN_WIDE_CAP" "1536/24,2048/32" "128" "256"
run_eden_group "EDEN_WIDE_CAP2" "2048/32" "144" "384"
# ^^^ THOG
