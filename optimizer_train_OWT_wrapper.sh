#!/bin/bash
set -euo pipefail

if (( $# < 1 )); then
  echo "Usage: $0 TARGET_WRAPPER [wrapper options]" >&2
  exit 2
fi

TARGET_WRAPPER="$1"
shift

OPTIMIZER="adamw"
OPTIMIZER_MOMENTUM="0.9"
LR_EXPLICIT=false
MIN_LR_EXPLICIT=false
PRE_ARGS=()
EXTRA_ARGS=()
SAW_SEPARATOR=false

while (( $# > 0 )); do
  if [[ "$SAW_SEPARATOR" == true ]]; then
    EXTRA_ARGS+=("$1")
    shift
    continue
  fi
  case "$1" in
    -y|--optimizer)
      (( $# >= 2 )) || { echo "$1 requires an optimizer name" >&2; exit 2; }
      OPTIMIZER="$2"
      shift 2
      ;;
    --optimizer=*)
      OPTIMIZER="${1#*=}"
      shift
      ;;
    --optimizer-momentum)
      (( $# >= 2 )) || { echo "$1 requires a numeric value" >&2; exit 2; }
      OPTIMIZER_MOMENTUM="$2"
      shift 2
      ;;
    --optimizer-momentum=*)
      OPTIMIZER_MOMENTUM="${1#*=}"
      shift
      ;;
    -c)
      (( $# >= 2 )) || { echo "-c requires a learning-rate code" >&2; exit 2; }
      LR_EXPLICIT=true
      PRE_ARGS+=("$1" "$2")
      shift 2
      ;;
    -f)
      (( $# >= 2 )) || { echo "-f requires a minimum-learning-rate code" >&2; exit 2; }
      MIN_LR_EXPLICIT=true
      PRE_ARGS+=("$1" "$2")
      shift 2
      ;;
    --)
      SAW_SEPARATOR=true
      shift
      ;;
    *)
      PRE_ARGS+=("$1")
      shift
      ;;
  esac
done

case "${OPTIMIZER,,}" in
  adam|adamw)
    OPTIMIZER="adamw"
    DEFAULT_LR_CODE="60"
    DEFAULT_MIN_LR_CODE="06"
    ;;
  sgd)
    OPTIMIZER="sgd"
    DEFAULT_LR_CODE="1000"
    DEFAULT_MIN_LR_CODE="100"
    ;;
  nesterov|sgd-nesterov|sgd_nesterov)
    OPTIMIZER="sgd_nesterov"
    DEFAULT_LR_CODE="1000"
    DEFAULT_MIN_LR_CODE="100"
    ;;
  adafactor)
    OPTIMIZER="adafactor"
    DEFAULT_LR_CODE="1000"
    DEFAULT_MIN_LR_CODE="100"
    ;;
  rmsprop)
    OPTIMIZER="rmsprop"
    DEFAULT_LR_CODE="100"
    DEFAULT_MIN_LR_CODE="10"
    ;;
  *)
    echo "Unsupported optimizer: $OPTIMIZER" >&2
    echo "Expected: adamw | sgd | sgd_nesterov | adafactor | rmsprop" >&2
    exit 2
    ;;
esac

if [[ "$LR_EXPLICIT" == false ]]; then
  PRE_ARGS+=("-c" "$DEFAULT_LR_CODE")
fi
if [[ "$MIN_LR_EXPLICIT" == false ]]; then
  PRE_ARGS+=("-f" "$DEFAULT_MIN_LR_CODE")
fi

if [[ "$OPTIMIZER" != "adamw" ]]; then
  OPTIMIZER_SUFFIX="OPT_${OPTIMIZER^^}"
  UPDATED_EXTRA_ARGS=()
  FOUND_ARTIFACT_SUFFIX=false
  for (( index=0; index < ${#EXTRA_ARGS[@]}; index++ )); do
    argument="${EXTRA_ARGS[index]}"
    case "$argument" in
      --artifact-suffix)
        (( index + 1 < ${#EXTRA_ARGS[@]} )) || { echo "--artifact-suffix requires a value" >&2; exit 2; }
        UPDATED_EXTRA_ARGS+=("--artifact-suffix" "${EXTRA_ARGS[index + 1]}_${OPTIMIZER_SUFFIX}")
        index=$((index + 1))
        FOUND_ARTIFACT_SUFFIX=true
        ;;
      --artifact-suffix=*)
        UPDATED_EXTRA_ARGS+=("--artifact-suffix=${argument#*=}_${OPTIMIZER_SUFFIX}")
        FOUND_ARTIFACT_SUFFIX=true
        ;;
      *)
        UPDATED_EXTRA_ARGS+=("$argument")
        ;;
    esac
  done
  if [[ "$FOUND_ARTIFACT_SUFFIX" == false ]]; then
    UPDATED_EXTRA_ARGS+=("--artifact-suffix" "$OPTIMIZER_SUFFIX")
  fi
  EXTRA_ARGS=("${UPDATED_EXTRA_ARGS[@]}")
fi

export THOG2_OPTIMIZER="$OPTIMIZER"
export THOG2_OPTIMIZER_MOMENTUM="$OPTIMIZER_MOMENTUM"

if (( ${#EXTRA_ARGS[@]} > 0 )); then
  exec "$TARGET_WRAPPER" "${PRE_ARGS[@]}" -- "${EXTRA_ARGS[@]}"
fi
exec "$TARGET_WRAPPER" "${PRE_ARGS[@]}"
