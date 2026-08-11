#!/bin/bash

# vvv THOG
# Observe core-owned PLASTIC controls for legacy environment consumers and route the
# v0.541 Python-native wall-time controls plus v0.53 same-batch and v0.56 decision controls through the established -- extra-args channel.
THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS="${THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS:-1}"
THOG2_PLASTIC_LAYER_COUNT_MAX_STEP="${THOG2_PLASTIC_LAYER_COUNT_MAX_STEP:-1}"
THOG2_PLASTIC_LAYER_COUNT__SAME_BATCH_ALL_PROBES="${THOG2_PLASTIC_LAYER_COUNT__SAME_BATCH_ALL_PROBES:-false}"
THOG2_PLASTIC_WALL_TIME_LOSS_RATE_WINDOW="${THOG2_PLASTIC_WALL_TIME_LOSS_RATE_WINDOW:-64}"
THOG2_PLASTIC_WALL_TIME_LOSS_RATE_MIN_OBSERVATIONS="${THOG2_PLASTIC_WALL_TIME_LOSS_RATE_MIN_OBSERVATIONS:-16}"
THOG2_PLASTIC_LOOKAHEAD_ORIGINAL_ARGS=("$@")
THOG2_PLASTIC_LOOKAHEAD_FILTERED_ARGS=()
THOG2_PLASTIC_WALL_TIME_EXTRA_ARGS=()
THOG2_PLASTIC_SAME_BATCH_EXTRA_ARGS=()
THOG2_PLASTIC_DECISION_EXTRA_ARGS=()
THOG2_PLASTIC_LOOKAHEAD_HELP=false
THOG2_PLASTIC_LOOKAHEAD_INDEX=0
while (( THOG2_PLASTIC_LOOKAHEAD_INDEX < ${#THOG2_PLASTIC_LOOKAHEAD_ORIGINAL_ARGS[@]} )); do
  THOG2_PLASTIC_LOOKAHEAD_ARGUMENT="${THOG2_PLASTIC_LOOKAHEAD_ORIGINAL_ARGS[$THOG2_PLASTIC_LOOKAHEAD_INDEX]}"
  case "$THOG2_PLASTIC_LOOKAHEAD_ARGUMENT" in
    --)
      THOG2_PLASTIC_LOOKAHEAD_FILTERED_ARGS+=("--")
      ((THOG2_PLASTIC_LOOKAHEAD_INDEX += 1))
      while (( THOG2_PLASTIC_LOOKAHEAD_INDEX < ${#THOG2_PLASTIC_LOOKAHEAD_ORIGINAL_ARGS[@]} )); do
        THOG2_PLASTIC_LOOKAHEAD_FILTERED_ARGS+=("${THOG2_PLASTIC_LOOKAHEAD_ORIGINAL_ARGS[$THOG2_PLASTIC_LOOKAHEAD_INDEX]}")
        ((THOG2_PLASTIC_LOOKAHEAD_INDEX += 1))
      done
      break
      ;;
    --plastic__layer_count__same_batch_all_probes)
      THOG2_PLASTIC_LAYER_COUNT__SAME_BATCH_ALL_PROBES=true
      THOG2_PLASTIC_SAME_BATCH_EXTRA_ARGS+=("$THOG2_PLASTIC_LOOKAHEAD_ARGUMENT")
      ((THOG2_PLASTIC_LOOKAHEAD_INDEX += 1))
      continue
      ;;
    --no-plastic__layer_count__same_batch_all_probes)
      THOG2_PLASTIC_LAYER_COUNT__SAME_BATCH_ALL_PROBES=false
      THOG2_PLASTIC_SAME_BATCH_EXTRA_ARGS+=("$THOG2_PLASTIC_LOOKAHEAD_ARGUMENT")
      ((THOG2_PLASTIC_LOOKAHEAD_INDEX += 1))
      continue
      ;;
    --plastic__layer_count_gradient__minimum_absolute_kendall_tau|--plastic__layer_count_gradient__minimum_absolute_kendall_tau=*)
      echo "--plastic__layer_count_gradient__minimum_absolute_kendall_tau was removed in PLASTIC v0.55; Kendall coherence is fixed at |tau| >= 0.5 for the Sen/Kendall algorithms" >&2
      exit 2
      ;;
    --plastic__layer_count_decision_algorithm__growth_side_discount)
      (( THOG2_PLASTIC_LOOKAHEAD_INDEX + 1 < ${#THOG2_PLASTIC_LOOKAHEAD_ORIGINAL_ARGS[@]} )) || {
        echo "$THOG2_PLASTIC_LOOKAHEAD_ARGUMENT requires a value in [0, 1]" >&2
        exit 2
      }
      THOG2_PLASTIC_DECISION_EXTRA_ARGS+=(
        "$THOG2_PLASTIC_LOOKAHEAD_ARGUMENT"
        "${THOG2_PLASTIC_LOOKAHEAD_ORIGINAL_ARGS[$((THOG2_PLASTIC_LOOKAHEAD_INDEX + 1))]}"
      )
      ((THOG2_PLASTIC_LOOKAHEAD_INDEX += 2))
      continue
      ;;
    --plastic__layer_count_decision_algorithm__growth_side_discount=*)
      THOG2_PLASTIC_DECISION_EXTRA_ARGS+=("$THOG2_PLASTIC_LOOKAHEAD_ARGUMENT")
      ((THOG2_PLASTIC_LOOKAHEAD_INDEX += 1))
      continue
      ;;
    --plastic__layer_count_decision_algorithm)
      (( THOG2_PLASTIC_LOOKAHEAD_INDEX + 1 < ${#THOG2_PLASTIC_LOOKAHEAD_ORIGINAL_ARGS[@]} )) || {
        echo "$THOG2_PLASTIC_LOOKAHEAD_ARGUMENT requires a value" >&2
        exit 2
      }
      THOG2_PLASTIC_DECISION_EXTRA_ARGS+=(
        "$THOG2_PLASTIC_LOOKAHEAD_ARGUMENT"
        "${THOG2_PLASTIC_LOOKAHEAD_ORIGINAL_ARGS[$((THOG2_PLASTIC_LOOKAHEAD_INDEX + 1))]}"
      )
      ((THOG2_PLASTIC_LOOKAHEAD_INDEX += 2))
      continue
      ;;
    --plastic__layer_count_decision_algorithm=*)
      THOG2_PLASTIC_DECISION_EXTRA_ARGS+=("$THOG2_PLASTIC_LOOKAHEAD_ARGUMENT")
      ((THOG2_PLASTIC_LOOKAHEAD_INDEX += 1))
      continue
      ;;
    --plastic__wall_time_equivalent_time_gain_discount|--plastic__wall_time_equivalent_time_gain_loss_rate_window|--plastic__wall_time_equivalent_time_gain_loss_rate_min_observations)
      (( THOG2_PLASTIC_LOOKAHEAD_INDEX + 1 < ${#THOG2_PLASTIC_LOOKAHEAD_ORIGINAL_ARGS[@]} )) || {
        echo "$THOG2_PLASTIC_LOOKAHEAD_ARGUMENT requires a value" >&2
        exit 2
      }
      THOG2_PLASTIC_WALL_TIME_EXTRA_ARGS+=(
        "$THOG2_PLASTIC_LOOKAHEAD_ARGUMENT"
        "${THOG2_PLASTIC_LOOKAHEAD_ORIGINAL_ARGS[$((THOG2_PLASTIC_LOOKAHEAD_INDEX + 1))]}"
      )
      case "$THOG2_PLASTIC_LOOKAHEAD_ARGUMENT" in
        --plastic__wall_time_equivalent_time_gain_loss_rate_window)
          THOG2_PLASTIC_WALL_TIME_LOSS_RATE_WINDOW="${THOG2_PLASTIC_LOOKAHEAD_ORIGINAL_ARGS[$((THOG2_PLASTIC_LOOKAHEAD_INDEX + 1))]}"
          ;;
        --plastic__wall_time_equivalent_time_gain_loss_rate_min_observations)
          THOG2_PLASTIC_WALL_TIME_LOSS_RATE_MIN_OBSERVATIONS="${THOG2_PLASTIC_LOOKAHEAD_ORIGINAL_ARGS[$((THOG2_PLASTIC_LOOKAHEAD_INDEX + 1))]}"
          ;;
      esac
      ((THOG2_PLASTIC_LOOKAHEAD_INDEX += 2))
      continue
      ;;
    --plastic__wall_time_equivalent_time_gain_discount=*|--plastic__wall_time_equivalent_time_gain_loss_rate_window=*|--plastic__wall_time_equivalent_time_gain_loss_rate_min_observations=*)
      THOG2_PLASTIC_WALL_TIME_EXTRA_ARGS+=("$THOG2_PLASTIC_LOOKAHEAD_ARGUMENT")
      case "$THOG2_PLASTIC_LOOKAHEAD_ARGUMENT" in
        --plastic__wall_time_equivalent_time_gain_loss_rate_window=*)
          THOG2_PLASTIC_WALL_TIME_LOSS_RATE_WINDOW="${THOG2_PLASTIC_LOOKAHEAD_ARGUMENT#*=}"
          ;;
        --plastic__wall_time_equivalent_time_gain_loss_rate_min_observations=*)
          THOG2_PLASTIC_WALL_TIME_LOSS_RATE_MIN_OBSERVATIONS="${THOG2_PLASTIC_LOOKAHEAD_ARGUMENT#*=}"
          ;;
      esac
      ((THOG2_PLASTIC_LOOKAHEAD_INDEX += 1))
      continue
      ;;
    --plastic__layer_count_probe_radius)
      (( THOG2_PLASTIC_LOOKAHEAD_INDEX + 1 < ${#THOG2_PLASTIC_LOOKAHEAD_ORIGINAL_ARGS[@]} )) || {
        echo "$THOG2_PLASTIC_LOOKAHEAD_ARGUMENT requires a positive integer" >&2
        exit 2
      }
      THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS="${THOG2_PLASTIC_LOOKAHEAD_ORIGINAL_ARGS[$((THOG2_PLASTIC_LOOKAHEAD_INDEX + 1))]}"
      THOG2_PLASTIC_LOOKAHEAD_FILTERED_ARGS+=(
        "$THOG2_PLASTIC_LOOKAHEAD_ARGUMENT"
        "${THOG2_PLASTIC_LOOKAHEAD_ORIGINAL_ARGS[$((THOG2_PLASTIC_LOOKAHEAD_INDEX + 1))]}"
      )
      ((THOG2_PLASTIC_LOOKAHEAD_INDEX += 2))
      continue
      ;;
    --plastic__layer_count_probe_radius=*)
      THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS="${THOG2_PLASTIC_LOOKAHEAD_ARGUMENT#*=}"
      ;;
    --plastic__layer_count__max_allowable_layer_change)
      (( THOG2_PLASTIC_LOOKAHEAD_INDEX + 1 < ${#THOG2_PLASTIC_LOOKAHEAD_ORIGINAL_ARGS[@]} )) || {
        echo "$THOG2_PLASTIC_LOOKAHEAD_ARGUMENT requires a positive integer" >&2
        exit 2
      }
      THOG2_PLASTIC_LAYER_COUNT_MAX_STEP="${THOG2_PLASTIC_LOOKAHEAD_ORIGINAL_ARGS[$((THOG2_PLASTIC_LOOKAHEAD_INDEX + 1))]}"
      THOG2_PLASTIC_LOOKAHEAD_FILTERED_ARGS+=(
        "$THOG2_PLASTIC_LOOKAHEAD_ARGUMENT"
        "${THOG2_PLASTIC_LOOKAHEAD_ORIGINAL_ARGS[$((THOG2_PLASTIC_LOOKAHEAD_INDEX + 1))]}"
      )
      ((THOG2_PLASTIC_LOOKAHEAD_INDEX += 2))
      continue
      ;;
    --plastic__layer_count__max_allowable_layer_change=*)
      THOG2_PLASTIC_LAYER_COUNT_MAX_STEP="${THOG2_PLASTIC_LOOKAHEAD_ARGUMENT#*=}"
      ;;
    -h|--help)
      THOG2_PLASTIC_LOOKAHEAD_HELP=true
      ;;
  esac
  THOG2_PLASTIC_LOOKAHEAD_FILTERED_ARGS+=("$THOG2_PLASTIC_LOOKAHEAD_ARGUMENT")
  ((THOG2_PLASTIC_LOOKAHEAD_INDEX += 1))
done

if (( ${#THOG2_PLASTIC_WALL_TIME_EXTRA_ARGS[@]} > 0 || ${#THOG2_PLASTIC_SAME_BATCH_EXTRA_ARGS[@]} > 0 || ${#THOG2_PLASTIC_DECISION_EXTRA_ARGS[@]} > 0 )); then
  THOG2_PLASTIC_LOOKAHEAD_FINAL_ARGS=()
  THOG2_PLASTIC_LOOKAHEAD_INSERTED_PYTHON_ARGS=false
  for THOG2_PLASTIC_LOOKAHEAD_ARGUMENT in "${THOG2_PLASTIC_LOOKAHEAD_FILTERED_ARGS[@]}"; do
    THOG2_PLASTIC_LOOKAHEAD_FINAL_ARGS+=("$THOG2_PLASTIC_LOOKAHEAD_ARGUMENT")
    if [[ "$THOG2_PLASTIC_LOOKAHEAD_ARGUMENT" == "--" && "$THOG2_PLASTIC_LOOKAHEAD_INSERTED_PYTHON_ARGS" == false ]]; then
      THOG2_PLASTIC_LOOKAHEAD_FINAL_ARGS+=("${THOG2_PLASTIC_SAME_BATCH_EXTRA_ARGS[@]}" "${THOG2_PLASTIC_WALL_TIME_EXTRA_ARGS[@]}" "${THOG2_PLASTIC_DECISION_EXTRA_ARGS[@]}")
      THOG2_PLASTIC_LOOKAHEAD_INSERTED_PYTHON_ARGS=true
    fi
  done
  if [[ "$THOG2_PLASTIC_LOOKAHEAD_INSERTED_PYTHON_ARGS" == false ]]; then
    THOG2_PLASTIC_LOOKAHEAD_FINAL_ARGS+=("--" "${THOG2_PLASTIC_SAME_BATCH_EXTRA_ARGS[@]}" "${THOG2_PLASTIC_WALL_TIME_EXTRA_ARGS[@]}" "${THOG2_PLASTIC_DECISION_EXTRA_ARGS[@]}")
  fi
  set -- "${THOG2_PLASTIC_LOOKAHEAD_FINAL_ARGS[@]}"
else
  set -- "${THOG2_PLASTIC_LOOKAHEAD_FILTERED_ARGS[@]}"
fi
unset THOG2_PLASTIC_LOOKAHEAD_ORIGINAL_ARGS THOG2_PLASTIC_LOOKAHEAD_FILTERED_ARGS THOG2_PLASTIC_LOOKAHEAD_FINAL_ARGS
unset THOG2_PLASTIC_WALL_TIME_EXTRA_ARGS THOG2_PLASTIC_SAME_BATCH_EXTRA_ARGS THOG2_PLASTIC_DECISION_EXTRA_ARGS THOG2_PLASTIC_LOOKAHEAD_INSERTED_PYTHON_ARGS
unset THOG2_PLASTIC_LOOKAHEAD_INDEX THOG2_PLASTIC_LOOKAHEAD_ARGUMENT

[[ "$THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS" =~ ^[1-9][0-9]*$ ]] || {
  echo "Invalid plastic__layer_count_probe_radius: $THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS; expected a positive integer." >&2
  exit 2
}
[[ "$THOG2_PLASTIC_LAYER_COUNT_MAX_STEP" =~ ^[1-9][0-9]*$ ]] || {
  echo "Invalid plastic__layer_count__max_allowable_layer_change: $THOG2_PLASTIC_LAYER_COUNT_MAX_STEP; expected a positive integer." >&2
  exit 2
}
case "$THOG2_PLASTIC_LAYER_COUNT__SAME_BATCH_ALL_PROBES" in
  true|false) ;;
  *) echo "THOG2_PLASTIC_LAYER_COUNT__SAME_BATCH_ALL_PROBES must be true or false; got: $THOG2_PLASTIC_LAYER_COUNT__SAME_BATCH_ALL_PROBES" >&2; exit 2 ;;
esac
export THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS
export THOG2_PLASTIC_LAYER_COUNT_MAX_STEP
export THOG2_PLASTIC_LAYER_COUNT__SAME_BATCH_ALL_PROBES
export THOG2_PLASTIC_WALL_TIME_LOSS_RATE_WINDOW
export THOG2_PLASTIC_WALL_TIME_LOSS_RATE_MIN_OBSERVATIONS

if [[ "$THOG2_PLASTIC_LOOKAHEAD_HELP" == true ]]; then
  printf '%s\n' \
    'PLASTIC DEPTH COARSE/FINE:' \
    '  --plastic__coarse_phase enabled|disabled       one-shot COARSE discovery; default disabled' \
    '  --plastic__phase_1_n_steps N                   optimizer steps per COARSE trial' \
    '  --plastic__phase_1_starting_layer_count N      first doubling candidate' \
    '  --plastic__phase_1__number_of_trials N         number of doubling candidates' \
    '  --plastic__phase_1_evaluation_steps_count N    final validation batches per trial' \
    '  --plastic__log_interval_coarse N=10            COARSE progress cadence' \
    '  --plastic__coarse_phase_roll_through            skip the review delay and start FINE immediately' \
    '  --no-plastic__coarse_phase_roll_through         retain the review delay; default' \
    '  --plastic__layer_count_probe__probe_every_n_steps N        FINE probe cadence; defaults to update brake' \
    "  --plastic__layer_count_probe_radius N=${THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS}       full integer FINE probe radius" \
    "  --plastic__layer_count__max_allowable_layer_change N=${THOG2_PLASTIC_LAYER_COUNT_MAX_STEP}          maximum committed FINE movement" \
    '  --plastic__layer_count__same_batch_all_probes             one fixed probe batch per strict non-overlapping evidence window' \
    '  --no-plastic__layer_count__same_batch_all_probes          established rolling/multi-batch probe path; default' \
    '  --plastic__layer_count_decision_algorithm ALGORITHM       directional_coherence (default), theil_sen_kendall_LRA, or sen_kendall__tau__stratified; objective selected separately' \
    '      For Sen slope:' \
    '        sen < 0    y tends to decrease as x increases. Adding layers tends to improve the wall-time-equivalent economic score' \
    '        sen = 0    no overall trend. Little directional economic preference' \
    '        sen > 0    y tends to increase as x increases. Removing layers tends to improve the wall-time-equivalent economic score' \
    '      For Kendall tau:' \
    '        tau = -1    perfectly decreasing ordering. Extremely strong/near-perfect indication toward adding layers' \
    '        tau ≈ -0.8  strongly decreasing. Strong indication toward adding layers' \
    '        tau ≈ -0.5  moderately decreasing. Moderate / minimum accepted indication toward adding layer' \
    '        tau ≈  0    little/no monotonic association. No reliable directional indication' \
    '        tau ≈ +0.5  moderately increasing. Moderate / minimum accepted indication toward removing layers' \
    '        tau ≈ +0.8  strongly increasing. Strong indication toward removing layers' \
    '        tau = +1    perfectly increasing ordering. Extremely strong/near-perfect indication toward removing layers' \
    '  --plastic__layer_count_decision_algorithm__growth_side_discount X       Sen/Kendall only; credit fraction [0,1] of beneficial growth-side objective evidence; default 1.0' \
    '  --plastic__wall_time_equivalent_time_gain_discount X       credited fraction of positive equivalent-time gain; default 0.9' \
    '  --plastic__wall_time_equivalent_time_gain_loss_rate_window N       rolling ordinary-training loss-rate window; default 64' \
    '  --plastic__wall_time_equivalent_time_gain_loss_rate_min_observations N       minimum observations before loss-rate fit is usable; default 16' \
    '  Hyphenated and single-underscore PLASTIC aliases are rejected.' \
    ''
fi
unset THOG2_PLASTIC_LOOKAHEAD_HELP
# ^^^ THOG
