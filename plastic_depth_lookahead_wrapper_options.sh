#!/bin/bash

# vvv THOG
# Canonical train_OWT.sh controls for PLASTIC DEPTH COARSE/FINE discovery.
# train_OWT.sh normalizes underscore spellings before this helper runs.
# Radius/max-step values remain exported for legacy overlays and remain visible
# when this helper is sourced directly; train_OWT.sh routes them through its
# Python-extra boundary together with the newer COARSE/FINE controls.
THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS="${THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS:-1}"
THOG2_PLASTIC_LAYER_COUNT_MAX_STEP="${THOG2_PLASTIC_LAYER_COUNT_MAX_STEP:-1}"
THOG2_PLASTIC_LOOKAHEAD_FILTERED_ARGS=()
THOG2_PLASTIC_LOOKAHEAD_EXTRA_ARGS=()
THOG2_PLASTIC_LOOKAHEAD_COMPAT_ARGS=()
THOG2_PLASTIC_LOOKAHEAD_HELP=false
THOG2_PLASTIC_LOOKAHEAD_SAW_SEPARATOR=false
THOG2_PLASTIC_LOOKAHEAD_FROM_TRAIN_WRAPPER=false
case "${BASH_SOURCE[1]:-}" in
  *train_OWT.sh) THOG2_PLASTIC_LOOKAHEAD_FROM_TRAIN_WRAPPER=true ;;
esac

while (( $# > 0 )); do
  if [[ "$THOG2_PLASTIC_LOOKAHEAD_SAW_SEPARATOR" == true ]]; then
    THOG2_PLASTIC_LOOKAHEAD_FILTERED_ARGS+=("$1")
    shift
    continue
  fi
  case "$1" in
    --plastic-coarse-phase|--plastic-phase-1-n-steps|--plastic-phase-1-starting-layer-count|--plastic-phase-1-number-of-trials|--plastic-phase-1-evaluation-steps-count|--plastic-layer-count-probe-interval)
      (( $# >= 2 )) || { echo "$1 requires a value" >&2; exit 2; }
      THOG2_PLASTIC_LOOKAHEAD_EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --plastic-coarse-phase=*|--plastic-phase-1-n-steps=*|--plastic-phase-1-starting-layer-count=*|--plastic-phase-1-number-of-trials=*|--plastic-phase-1-evaluation-steps-count=*|--plastic-layer-count-probe-interval=*)
      THOG2_PLASTIC_LOOKAHEAD_EXTRA_ARGS+=("$1")
      shift
      ;;
    --plastic-layer-count-probe-radius)
      (( $# >= 2 )) || { echo "$1 requires a positive integer" >&2; exit 2; }
      THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS="$2"
      THOG2_PLASTIC_LOOKAHEAD_COMPAT_ARGS+=("$1" "$2")
      shift 2
      ;;
    --plastic-layer-count-probe-radius=*)
      THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS="${1#*=}"
      THOG2_PLASTIC_LOOKAHEAD_COMPAT_ARGS+=("$1")
      shift
      ;;
    --plastic-layer-count-max-step)
      (( $# >= 2 )) || { echo "$1 requires a positive integer" >&2; exit 2; }
      THOG2_PLASTIC_LAYER_COUNT_MAX_STEP="$2"
      THOG2_PLASTIC_LOOKAHEAD_COMPAT_ARGS+=("$1" "$2")
      shift 2
      ;;
    --plastic-layer-count-max-step=*)
      THOG2_PLASTIC_LAYER_COUNT_MAX_STEP="${1#*=}"
      THOG2_PLASTIC_LOOKAHEAD_COMPAT_ARGS+=("$1")
      shift
      ;;
    -h|--help)
      THOG2_PLASTIC_LOOKAHEAD_HELP=true
      THOG2_PLASTIC_LOOKAHEAD_FILTERED_ARGS+=("$1")
      shift
      ;;
    --)
      THOG2_PLASTIC_LOOKAHEAD_FILTERED_ARGS+=("--")
      THOG2_PLASTIC_LOOKAHEAD_SAW_SEPARATOR=true
      shift
      ;;
    *)
      THOG2_PLASTIC_LOOKAHEAD_FILTERED_ARGS+=("$1")
      shift
      ;;
  esac
done

# vvv THOG preserve direct-source argument identity while keeping train_OWT_core.sh insulated from Python-only long options
if [[ "$THOG2_PLASTIC_LOOKAHEAD_FROM_TRAIN_WRAPPER" == true ]]; then
  THOG2_PLASTIC_LOOKAHEAD_EXTRA_ARGS+=("${THOG2_PLASTIC_LOOKAHEAD_COMPAT_ARGS[@]}")
else
  THOG2_PLASTIC_LOOKAHEAD_FILTERED_ARGS=(
    "${THOG2_PLASTIC_LOOKAHEAD_COMPAT_ARGS[@]}"
    "${THOG2_PLASTIC_LOOKAHEAD_FILTERED_ARGS[@]}"
  )
fi
if (( ${#THOG2_PLASTIC_LOOKAHEAD_EXTRA_ARGS[@]} > 0 )); then
  if [[ "$THOG2_PLASTIC_LOOKAHEAD_SAW_SEPARATOR" == true ]]; then
    THOG2_PLASTIC_LOOKAHEAD_FILTERED_ARGS+=("${THOG2_PLASTIC_LOOKAHEAD_EXTRA_ARGS[@]}")
  else
    THOG2_PLASTIC_LOOKAHEAD_FILTERED_ARGS+=("--" "${THOG2_PLASTIC_LOOKAHEAD_EXTRA_ARGS[@]}")
  fi
fi
# ^^^ THOG

set -- "${THOG2_PLASTIC_LOOKAHEAD_FILTERED_ARGS[@]}"
unset THOG2_PLASTIC_LOOKAHEAD_FILTERED_ARGS THOG2_PLASTIC_LOOKAHEAD_EXTRA_ARGS THOG2_PLASTIC_LOOKAHEAD_COMPAT_ARGS THOG2_PLASTIC_LOOKAHEAD_SAW_SEPARATOR THOG2_PLASTIC_LOOKAHEAD_FROM_TRAIN_WRAPPER

[[ "$THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS" =~ ^[1-9][0-9]*$ ]] || {
  echo "Invalid plastic__layer_count_probe_radius: $THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS; expected a positive integer." >&2
  exit 2
}
[[ "$THOG2_PLASTIC_LAYER_COUNT_MAX_STEP" =~ ^[1-9][0-9]*$ ]] || {
  echo "Invalid plastic__layer_count_max_step: $THOG2_PLASTIC_LAYER_COUNT_MAX_STEP; expected a positive integer." >&2
  exit 2
}

export THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS
export THOG2_PLASTIC_LAYER_COUNT_MAX_STEP

if [[ "$THOG2_PLASTIC_LOOKAHEAD_HELP" == true ]]; then
  printf '%s\n' \
    'PLASTIC DEPTH COARSE/FINE:' \
    '  --plastic__coarse_phase enabled|disabled       one-shot COARSE discovery; default disabled' \
    '  --plastic__phase_1_n_steps N                   optimizer steps per COARSE trial' \
    '  --plastic__phase_1_starting_layer_count N      first doubling candidate' \
    '  --plastic__phase_1__number_of_trials N         number of doubling candidates' \
    '  --plastic__phase_1_evaluation_steps_count N    final validation batches per trial' \
    '  --plastic__layer_count_probe_interval N        FINE probe cadence; defaults to update brake' \
    "  --plastic__layer_count_probe_radius N=${THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS}       full integer FINE probe radius" \
    "  --plastic__layer_count_max_step N=${THOG2_PLASTIC_LAYER_COUNT_MAX_STEP}          maximum committed FINE movement" \
    '  Hyphenated and legacy single-underscore aliases remain accepted.' \
    ''
fi
unset THOG2_PLASTIC_LOOKAHEAD_HELP
# ^^^ THOG
