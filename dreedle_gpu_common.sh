#!/bin/bash
# vvv THOG
# Shared Dreedle single-GPU launch helpers. Source this only from Dreedle GPU-specific runstring wrappers.

dreedle_query_power_limit_w() {
  nvidia-smi -i "$1" --query-gpu=power.limit --format=csv,noheader,nounits | awk '{printf "%.0f\n", $1}'
}

dreedle_query_default_power_limit_w() {
  nvidia-smi -i "$1" --query-gpu=power.default_limit --format=csv,noheader,nounits | awk '{printf "%.0f\n", $1}'
}

dreedle_query_used_memory_mib() {
  nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits | awk '{printf "%d\n", $1}'
}

dreedle_query_total_memory_mib() {
  nvidia-smi -i "$1" --query-gpu=memory.total --format=csv,noheader,nounits | awk '{printf "%d\n", $1}'
}

dreedle_prepare_single_gpu_launch() {
  : "${THOG2_DREEDLE_GPU:?THOG2_DREEDLE_GPU must be set by the wrapper}"

  export THOG2_DREEDLE_POWER_LIMIT_W="${THOG2_DREEDLE_POWER_LIMIT_W:-220}"
  export THOG2_DREEDLE_APPLY_POWER_LIMIT="${THOG2_DREEDLE_APPLY_POWER_LIMIT:-true}"
  export THOG2_MIN_FREE_GPU_MEMORY_MIB="${THOG2_MIN_FREE_GPU_MEMORY_MIB:-22000}"
  export THOG2_ALLOW_BUSY_GPU="${THOG2_ALLOW_BUSY_GPU:-false}"
  export THOG2_DREEDLE_RUN_KIND="${THOG2_DREEDLE_RUN_KIND:-run}"

  if [[ "$THOG2_DREEDLE_APPLY_POWER_LIMIT" == "true" ]]; then
    case "$THOG2_DREEDLE_POWER_LIMIT_W" in
      default)
        sudo -v
        requested_power_limit_w="$(dreedle_query_default_power_limit_w "$THOG2_DREEDLE_GPU")"
        echo "Resetting Dreedle power limit: physical GPU ${THOG2_DREEDLE_GPU} -> default ${requested_power_limit_w}W"
        sudo nvidia-smi -i "$THOG2_DREEDLE_GPU" -pl "$requested_power_limit_w"
        ;;
      0|none|false)
        echo "Leaving Dreedle power limit unchanged: physical GPU ${THOG2_DREEDLE_GPU}"
        ;;
      ''|*[!0-9]*)
        echo "Invalid THOG2_DREEDLE_POWER_LIMIT_W: ${THOG2_DREEDLE_POWER_LIMIT_W}; expected integer watts, default, 0, none, or false." >&2
        exit 2
        ;;
      *)
        sudo -v
        echo "Applying Dreedle power limit: physical GPU ${THOG2_DREEDLE_GPU} -> ${THOG2_DREEDLE_POWER_LIMIT_W}W"
        sudo nvidia-smi -i "$THOG2_DREEDLE_GPU" -pl "$THOG2_DREEDLE_POWER_LIMIT_W"
        ;;
    esac
  else
    echo "Dreedle power limit not changed by script: physical GPU ${THOG2_DREEDLE_GPU}"
  fi

  sleep 2

  used_mib="$(dreedle_query_used_memory_mib "$THOG2_DREEDLE_GPU")"
  total_mib="$(dreedle_query_total_memory_mib "$THOG2_DREEDLE_GPU")"
  free_mib="$((total_mib - used_mib))"

  if [[ "$THOG2_ALLOW_BUSY_GPU" != "true" && "$free_mib" -lt "$THOG2_MIN_FREE_GPU_MEMORY_MIB" ]]; then
    echo "Refusing to start: physical GPU ${THOG2_DREEDLE_GPU} has only ${free_mib}MiB free; required ${THOG2_MIN_FREE_GPU_MEMORY_MIB}MiB." >&2
    nvidia-smi -i "$THOG2_DREEDLE_GPU"
    exit 3
  fi

  export CUDA_VISIBLE_DEVICES="$THOG2_DREEDLE_GPU"
  export THOG2_NUM_GPUS=1

  export THOG2_DREEDLE_EFFECTIVE_POWER_LIMIT_W="$(dreedle_query_power_limit_w "$THOG2_DREEDLE_GPU")"
  export THOG2_DREEDLE_DEFAULT_POWER_LIMIT_W="$(dreedle_query_default_power_limit_w "$THOG2_DREEDLE_GPU")"

  if [[ "$THOG2_DREEDLE_EFFECTIVE_POWER_LIMIT_W" == "$THOG2_DREEDLE_DEFAULT_POWER_LIMIT_W" ]]; then
    export THOG2_DREEDLE_POWER_TAG="pldefault${THOG2_DREEDLE_EFFECTIVE_POWER_LIMIT_W}"
  else
    export THOG2_DREEDLE_POWER_TAG="pl${THOG2_DREEDLE_EFFECTIVE_POWER_LIMIT_W}"
  fi

  export THOG2_HOST_LABEL="${THOG2_HOST_LABEL:-dreedle_gpu${THOG2_DREEDLE_GPU}_${THOG2_DREEDLE_RUN_KIND}_${THOG2_DREEDLE_POWER_TAG}}"

  echo "Dreedle launch GPU state:"
  nvidia-smi -i "$THOG2_DREEDLE_GPU"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "THOG2_DREEDLE_POWER_TAG=${THOG2_DREEDLE_POWER_TAG}"
  echo "THOG2_HOST_LABEL=${THOG2_HOST_LABEL}"
}
# ^^^ THOG
