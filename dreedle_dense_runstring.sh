#! /usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export THOG2_DREEDLE_GPU="${THOG2_DREEDLE_GPU:-0}"
export THOG2_DREEDLE_POWER_LIMIT_W="${THOG2_DREEDLE_POWER_LIMIT_W:-220}"
export THOG2_DREEDLE_APPLY_POWER_LIMIT="${THOG2_DREEDLE_APPLY_POWER_LIMIT:-true}"

export CUDA_VISIBLE_DEVICES="$THOG2_DREEDLE_GPU"
export THOG2_HOST_LABEL="${THOG2_HOST_LABEL:-dreedle_gpu${THOG2_DREEDLE_GPU}_dense_pl${THOG2_DREEDLE_POWER_LIMIT_W}}"
export THOG2_OWT_DATA_DIR="${THOG2_OWT_DATA_DIR:-$HOME/git/thog/data/openwebtext}"
export THOG2_NUM_GPUS="${THOG2_NUM_GPUS:-1}"
export THOG2_DTYPE="${THOG2_DTYPE:-float16}"
export THOG2_ATTENTION_BACKEND="${THOG2_ATTENTION_BACKEND:-sdpa}"

if [[ "$THOG2_DREEDLE_APPLY_POWER_LIMIT" == "true" && "$THOG2_DREEDLE_POWER_LIMIT_W" != "0" ]]; then
  echo "Applying Dreedle power limit: physical GPU ${THOG2_DREEDLE_GPU} -> ${THOG2_DREEDLE_POWER_LIMIT_W}W"
  sudo nvidia-smi -i "$THOG2_DREEDLE_GPU" -pl "$THOG2_DREEDLE_POWER_LIMIT_W"
fi
# ^^^ THOG

export THOG2_WANDB_FINISH_TIMEOUT=7200
export WANDB_CONSOLE=off

./train_OWT.sh \
  -g DENSE_GPU${THOG2_DREEDLE_GPU}_PL${THOG2_DREEDLE_POWER_LIMIT_W} \
  -p dense \
  -n 10000 \
  -b 12 \
  -A 4 \
  -G "$THOG2_NUM_GPUS" \
  -S 8 \
  -u 50 \
  -e 250 \
  -l 10 \
  -w 20 \
  -k 1000 \
  -c 30 \
  -f 3 \
  -L 72 \
  -H 18 \
  -D 1152 \
  -C 256 \
  -E true \
  -I wandb \
  -F none \
  -T "$THOG2_DTYPE" \
  -K "$THOG2_ATTENTION_BACKEND" \
  -t "$THOG2_OWT_DATA_DIR" \
  -- \
  --host-label "$THOG2_HOST_LABEL"