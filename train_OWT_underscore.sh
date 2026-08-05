#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# vvv THOG provide an underscore-first CLI shim while the preserved shell wrapper still owns canonical execution
THOG2_UNDERSCORE_FILTERED_ARGS=()
while (( $# > 0 )); do
  case "$1" in
    --plastic_layer_count_probe_interval|--plastic-layer-count-probe-interval)
      (( $# >= 2 )) || { echo "$1 requires a positive integer" >&2; exit 2; }
      export THOG2_PLASTIC_LAYER_COUNT_PROBE_INTERVAL="$2"
      shift 2
      ;;
    --plastic_layer_count_probe_interval=*|--plastic-layer-count-probe-interval=*)
      export THOG2_PLASTIC_LAYER_COUNT_PROBE_INTERVAL="${1#*=}"
      shift
      ;;
    --*=*)
      THOG2_UNDERSCORE_NAME="${1%%=*}"
      THOG2_UNDERSCORE_VALUE="${1#*=}"
      if [[ "$THOG2_UNDERSCORE_NAME" == --* ]]; then
        THOG2_UNDERSCORE_NAME="${THOG2_UNDERSCORE_NAME//_/-}"
      fi
      THOG2_UNDERSCORE_FILTERED_ARGS+=("${THOG2_UNDERSCORE_NAME}=${THOG2_UNDERSCORE_VALUE}")
      shift
      ;;
    --*)
      THOG2_UNDERSCORE_FILTERED_ARGS+=("${1//_/-}")
      shift
      ;;
    *)
      THOG2_UNDERSCORE_FILTERED_ARGS+=("$1")
      shift
      ;;
  esac
done
unset THOG2_UNDERSCORE_NAME THOG2_UNDERSCORE_VALUE
exec ./train_OWT.sh "${THOG2_UNDERSCORE_FILTERED_ARGS[@]}"
# ^^^ THOG
