#! /usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# vvv THOG
# Dreedle dense baseline host profile. Power descriptor is based on the actual post-policy nvidia-smi power limit.
export THOG2_DREEDLE_GPU="${THOG2_DREEDLE_GPU:-0}"
export THOG2_DREEDLE_POWER_LIMIT_W="${THOG2_DREEDLE_POWER_LIMIT_W:-220}"
export THOG2_DREEDLE_APPLY_POWER_LIMIT="${THOG2_DREEDLE_APPLY_POWER_LIMIT:-true}"

dreedle_query_power_limit_w() {
  nvidia-smi -i "$1" --query-gpu=power.limit --format=csv,noheader,nounits | awk '{printf "%.0f\n", $1}'
}

dreedle_query_default_power_limit_w() {
  nvidia-smi -i "$1" --query-gpu=power.default_limit --format=csv,noheader,nounits | awk '{printf "%.0f\n", $1}'
}

export CUDA_VISIBLE_DEVICES="$THOG2_DREEDLE_GPU"
export THOG2_OWT_DATA_DIR="${THOG2_OWT_DATA_DIR:-$HOME/git/thog/data/openwebtext}"
export THOG2_NUM_GPUS="${THOG2_NUM_GPUS:-1}"
export THOG2_DTYPE="${THOG2_DTYPE:-float16}"
export THOG2_ATTENTION_BACKEND="${THOG2_ATTENTION_BACKEND:-sdpa}"

if [[ "$THOG2_DREEDLE_APPLY_POWER_LIMIT" == "true" ]]; then
  case "$THOG2_DREEDLE_POWER_LIMIT_W" in
    default)
      THOG2_DREEDLE_REQUESTED_POWER_LIMIT_W="$(dreedle_query_default_power_limit_w "$THOG2_DREEDLE_GPU")"
      echo "Resetting Dreedle power limit: physical GPU ${THOG2_DREEDLE_GPU} -> default ${THOG2_DREEDLE_REQUESTED_POWER_LIMIT_W}W"
      sudo nvidia-smi -i "$THOG2_DREEDLE_GPU" -pl "$THOG2_DREEDLE_REQUESTED_POWER_LIMIT_W"
      ;;
    0|none|false)
      echo "Leaving Dreedle power limit unchanged: physical GPU ${THOG2_DREEDLE_GPU}"
      ;;
    ''|*[!0-9]*)
      echo "Invalid THOG2_DREEDLE_POWER_LIMIT_W: ${THOG2_DREEDLE_POWER_LIMIT_W}; expected integer watts, default, 0, none, or false." >&2
      exit 2
      ;;
    *)
      echo "Applying Dreedle power limit: physical GPU ${THOG2_DREEDLE_GPU} -> ${THOG2_DREEDLE_POWER_LIMIT_W}W"
      sudo nvidia-smi -i "$THOG2_DREEDLE_GPU" -pl "$THOG2_DREEDLE_POWER_LIMIT_W"
      ;;
  esac
else
  echo "Dreedle power limit not changed by script: physical GPU ${THOG2_DREEDLE_GPU}"
fi

export THOG2_DREEDLE_EFFECTIVE_POWER_LIMIT_W="$(dreedle_query_power_limit_w "$THOG2_DREEDLE_GPU")"
export THOG2_DREEDLE_DEFAULT_POWER_LIMIT_W="$(dreedle_query_default_power_limit_w "$THOG2_DREEDLE_GPU")"
if [[ "$THOG2_DREEDLE_EFFECTIVE_POWER_LIMIT_W" == "$THOG2_DREEDLE_DEFAULT_POWER_LIMIT_W" ]]; then
  export THOG2_DREEDLE_POWER_TAG="pldefault${THOG2_DREEDLE_EFFECTIVE_POWER_LIMIT_W}"
else
  export THOG2_DREEDLE_POWER_TAG="pl${THOG2_DREEDLE_EFFECTIVE_POWER_LIMIT_W}"
fi
export THOG2_HOST_LABEL="${THOG2_HOST_LABEL:-dreedle_gpu${THOG2_DREEDLE_GPU}_dense_${THOG2_DREEDLE_POWER_TAG}}"
echo "Dreedle effective power tag: GPU${THOG2_DREEDLE_GPU}_${THOG2_DREEDLE_POWER_TAG}"
# ^^^ THOG

export THOG2_WANDB_FINISH_TIMEOUT=7200
export WANDB_CONSOLE=off

./train_OWT.sh \
  -g DENSE_GPU${THOG2_DREEDLE_GPU}_${THOG2_DREEDLE_POWER_TAG} \
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
