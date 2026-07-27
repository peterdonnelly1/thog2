#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# vvv THOG
# Scruffy layer-dropout experiment based on the stable REVAMPV1 L32 DEPTH run.
export THOG2_HOST_LABEL="${THOG2_HOST_LABEL:-scruffy_dropout}"
export THOG2_OWT_DATA_DIR="${THOG2_OWT_DATA_DIR:-data/openwebtext}"
export THOG2_NUM_GPUS="${THOG2_NUM_GPUS:-1}"
export THOG2_DTYPE="${THOG2_DTYPE:-bfloat16}"
export THOG2_ATTENTION_BACKEND="${THOG2_ATTENTION_BACKEND:-flash2}"
# ^^^ THOG

python -m run_thog2_owt --print-geometry-registry

export THOG2_WANDB_FINISH_TIMEOUT=7200
export WANDB_CONSOLE=off

./train_OWT.sh \
  -g REVAMPv1_DROPOUT_LS4_LA3_LI10 \
  -n 10000 \
  -b 16 \
  -A 8 \
  -G "$THOG2_NUM_GPUS" \
  -S 4 \
  -u 1 \
  -e 10001 \
  -l 10 \
  -w 100 \
  -k 1000 \
  -y adamw \
  -c 90 \
  -f 9 \
  -L 32 \
  -s 4 \
  -M 3 \
  --layer-dropout-resample-steps 10 \
  -H 16 \
  -D 1024 \
  -C 768 \
  -P 16 \
  -Y 64 \
  -E true \
  -r depth_scaled \
  -z dof_implied_depth \
  -I wandb \
  -F none \
  -T "$THOG2_DTYPE" \
  -K "$THOG2_ATTENTION_BACKEND" \
  -t "$THOG2_OWT_DATA_DIR" \
  --select-depth \
  --option DEPTH.compressor=chebyshev \
  --option DEPTH.order=16 \
  -- \
  --host-label "$THOG2_HOST_LABEL"
