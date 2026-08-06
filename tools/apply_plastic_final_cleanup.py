#!/usr/bin/env python3
# vvv THOG
"""Final readability and inline-marker cleanup for the PLASTIC refinement."""

from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


HELPER = '''#!/bin/bash

# vvv THOG
# Observe canonical lookahead controls for legacy environment consumers without
# reordering, renaming, or bypassing train_OWT_core.sh argument ownership.
THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS="${THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS:-1}"
THOG2_PLASTIC_LAYER_COUNT_MAX_STEP="${THOG2_PLASTIC_LAYER_COUNT_MAX_STEP:-1}"
THOG2_PLASTIC_LOOKAHEAD_ORIGINAL_ARGS=("$@")
THOG2_PLASTIC_LOOKAHEAD_HELP=false
THOG2_PLASTIC_LOOKAHEAD_INDEX=0
while (( THOG2_PLASTIC_LOOKAHEAD_INDEX < ${#THOG2_PLASTIC_LOOKAHEAD_ORIGINAL_ARGS[@]} )); do
  THOG2_PLASTIC_LOOKAHEAD_ARGUMENT="${THOG2_PLASTIC_LOOKAHEAD_ORIGINAL_ARGS[$THOG2_PLASTIC_LOOKAHEAD_INDEX]}"
  case "$THOG2_PLASTIC_LOOKAHEAD_ARGUMENT" in
    --plastic__layer_count_probe_radius)
      (( THOG2_PLASTIC_LOOKAHEAD_INDEX + 1 < ${#THOG2_PLASTIC_LOOKAHEAD_ORIGINAL_ARGS[@]} )) || {
        echo "$THOG2_PLASTIC_LOOKAHEAD_ARGUMENT requires a positive integer" >&2
        exit 2
      }
      THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS="${THOG2_PLASTIC_LOOKAHEAD_ORIGINAL_ARGS[$((THOG2_PLASTIC_LOOKAHEAD_INDEX + 1))]}"
      ((THOG2_PLASTIC_LOOKAHEAD_INDEX += 2))
      continue
      ;;
    --plastic__layer_count_probe_radius=*)
      THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS="${THOG2_PLASTIC_LOOKAHEAD_ARGUMENT#*=}"
      ;;
    --plastic__layer_count_max_step)
      (( THOG2_PLASTIC_LOOKAHEAD_INDEX + 1 < ${#THOG2_PLASTIC_LOOKAHEAD_ORIGINAL_ARGS[@]} )) || {
        echo "$THOG2_PLASTIC_LOOKAHEAD_ARGUMENT requires a positive integer" >&2
        exit 2
      }
      THOG2_PLASTIC_LAYER_COUNT_MAX_STEP="${THOG2_PLASTIC_LOOKAHEAD_ORIGINAL_ARGS[$((THOG2_PLASTIC_LOOKAHEAD_INDEX + 1))]}"
      ((THOG2_PLASTIC_LOOKAHEAD_INDEX += 2))
      continue
      ;;
    --plastic__layer_count_max_step=*)
      THOG2_PLASTIC_LAYER_COUNT_MAX_STEP="${THOG2_PLASTIC_LOOKAHEAD_ARGUMENT#*=}"
      ;;
    -h|--help)
      THOG2_PLASTIC_LOOKAHEAD_HELP=true
      ;;
  esac
  ((THOG2_PLASTIC_LOOKAHEAD_INDEX += 1))
done
set -- "${THOG2_PLASTIC_LOOKAHEAD_ORIGINAL_ARGS[@]}"
unset THOG2_PLASTIC_LOOKAHEAD_ORIGINAL_ARGS
unset THOG2_PLASTIC_LOOKAHEAD_INDEX
unset THOG2_PLASTIC_LOOKAHEAD_ARGUMENT

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
    '  --plastic__log_interval_coarse N=10            COARSE progress cadence' \
    '  --plastic__coarse_phase_roll_through            skip the review delay and start FINE immediately' \
    '  --no-plastic__coarse_phase_roll_through         retain the review delay; default' \
    '  --plastic__layer_count_probe_interval N        FINE probe cadence; defaults to update brake' \
    "  --plastic__layer_count_probe_radius N=${THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS}       full integer FINE probe radius" \
    "  --plastic__layer_count_max_step N=${THOG2_PLASTIC_LAYER_COUNT_MAX_STEP}          maximum committed FINE movement" \
    '  Hyphenated and single-underscore PLASTIC aliases are rejected.' \
    ''
fi
unset THOG2_PLASTIC_LOOKAHEAD_HELP
# ^^^ THOG
'''


def main() -> None:
    helper_path = ROOT / "plastic_depth_lookahead_wrapper_options.sh"
    helper_path.write_text(HELPER, encoding="utf-8")

    marker_files = (
        "sheet/__init__.py",
        "run_thog2_owt_core.py",
        "sheet/run_config.py",
        "sheet/training_config.py",
        "sheet/plastic_depth_coarse.py",
        "sheet/plastic_depth_coarse_runner.py",
        "sheet/plastic_depth_lifecycle.py",
        "sheet/plastic_depth_console_minor_patch.py",
        "sheet/plastic_depth_coarse_runtime_recovery_patch.py",
    )
    subprocess.run(
        ("python", "tools/align_thog_inline_markers.py", *marker_files),
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
# ^^^ THOG
