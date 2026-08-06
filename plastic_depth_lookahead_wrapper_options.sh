#!/bin/bash

# vvv THOG
# Canonical train_OWT.sh controls for PLASTIC DEPTH COARSE/FINE discovery.
# train_OWT.sh normalizes underscore spellings before this helper runs.
# Radius/max-step values are still exported for legacy overlays, but the
# canonical arguments now remain in argv so resolved config/checkpoints own them.
THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS="${THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS:-1}"
THOG2_PLASTIC_LAYER_COUNT_MAX_STEP="${THOG2_PLASTIC_LAYER_COUNT_MAX_STEP:-1}"
THOG2_PLASTIC_LOOKAHEAD_FILTERED_ARGS=()
THOG2_PLASTIC_LOOKAHEAD_HELP=false
THOG2_PLASTIC_LOOKAHEAD_SAW_SEPARATOR=false

while (( $# > 0 )); do
  if [[ "$THOG2_PLASTIC_LOOKAHEAD_SAW_SEPARATOR" == true ]]; then
    THOG2_PLASTIC_LOOKAHEAD_FILTERED_ARGS+=("$1")
    shift
    continue
  fi
  case "$1" in
    --plastic-layer-count-probe-radius)
      (( $# >= 2 )) || { echo "$1 requires a positive integer" >&2; exit 2; }
      THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS="$2"
      THOG2_PLASTIC_LOOKAHEAD_FILTERED_ARGS+=("$1" "$2")
      shift 2
      ;;
    --plastic-layer-count-probe-radius=*)
      THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS="${1#*=}"
      THOG2_PLASTIC_LOOKAHEAD_FILTERED_ARGS+=("$1")
      shift
      ;;
    --plastic-layer-count-max-step)
      (( $# >= 2 )) || { echo "$1 requires a positive integer" >&2; exit 2; }
      THOG2_PLASTIC_LAYER_COUNT_MAX_STEP="$2"
      THOG2_PLASTIC_LOOKAHEAD_FILTERED_ARGS+=("$1" "$2")
      shift 2
      ;;
    --plastic-layer-count-max-step=*)
      THOG2_PLASTIC_LAYER_COUNT_MAX_STEP="${1#*=}"
      THOG2_PLASTIC_LOOKAHEAD_FILTERED_ARGS+=("$1")
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

set -- "${THOG2_PLASTIC_LOOKAHEAD_FILTERED_ARGS[@]}"
unset THOG2_PLASTIC_LOOKAHEAD_FILTERED_ARGS THOG2_PLASTIC_LOOKAHEAD_SAW_SEPARATOR

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
