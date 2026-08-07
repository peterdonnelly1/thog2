#!/usr/bin/env python3
# vvv THOG
"""Make the shell core authoritative for every canonical PLASTIC control."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, content: str) -> None:
    (ROOT / path).write_text(content, encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    content = read(path)
    if new in content:
        return
    count = content.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one anchor, found {count}: {old[:120]!r}")
    write(path, content.replace(old, new, 1))


def rewrite_lookahead_helper() -> None:
    write(
        "plastic_depth_lookahead_wrapper_options.sh",
        '''#!/bin/bash

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
unset THOG2_PLASTIC_LOOKAHEAD_ORIGINAL_ARGS THOG2_PLASTIC_LOOKAHEAD_INDEX THOG2_PLASTIC_LOOKAHEAD_ARGUMENT

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
    '  --plastic__layer_count_probe__probe_every_n_steps N        FINE probe cadence; defaults to update brake' \
    "  --plastic__layer_count_probe_radius N=${THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS}       full integer FINE probe radius" \
    "  --plastic__layer_count_max_step N=${THOG2_PLASTIC_LAYER_COUNT_MAX_STEP}          maximum committed FINE movement" \
    '  Hyphenated and single-underscore PLASTIC aliases are rejected.' \
    ''
fi
unset THOG2_PLASTIC_LOOKAHEAD_HELP
# ^^^ THOG
''',
    )


def make_shell_core_authoritative() -> None:
    path = "train_OWT_core.sh"
    content = read(path)
    defaults_anchor = 'PLASTIC_ENABLED=false\n'
    defaults = (
        'PLASTIC_COARSE_PHASE="disabled"\n'
        'PLASTIC_PHASE_1_N_STEPS=""\n'
        'PLASTIC_PHASE_1_STARTING_LAYER_COUNT=""\n'
        'PLASTIC_PHASE_1_NUMBER_OF_TRIALS=""\n'
        'PLASTIC_PHASE_1_EVALUATION_STEPS_COUNT=""\n'
        'PLASTIC_LAYER_COUNT_PROBE_EVERY_N_STEPS=""\n'
        'PLASTIC_LAYER_COUNT_PROBE_RADIUS="${THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS:-1}"\n'
        'PLASTIC_LAYER_COUNT_MAX_STEP="${THOG2_PLASTIC_LAYER_COUNT_MAX_STEP:-1}"\n'
    )
    if "PLASTIC_COARSE_PHASE=" not in content:
        content = content.replace(defaults_anchor, defaults_anchor + defaults, 1)

    help_anchor = 'PLASTIC DEPTH:\n  --plastic__enabled | --no-plastic__enabled\n'
    help_addition = (
        '  --plastic__coarse_phase enabled|disabled\n'
        '  --plastic__phase_1_n_steps N\n'
        '  --plastic__phase_1_starting_layer_count N\n'
        '  --plastic__phase_1__number_of_trials N\n'
        '  --plastic__phase_1_evaluation_steps_count N\n'
    )
    if '  --plastic__phase_1_n_steps N\n' not in content:
        content = content.replace(help_anchor, help_anchor + help_addition, 1)
    if '  --plastic__layer_count_probe__probe_every_n_steps N' not in content:
        content = content.replace(
            '  --plastic__layer_count_update_brake N=${PLASTIC_LAYER_COUNT_UPDATE_BRAKE}\n',
            '  --plastic__layer_count_update_brake N=${PLASTIC_LAYER_COUNT_UPDATE_BRAKE}\n'
            '  --plastic__layer_count_probe__probe_every_n_steps N=${PLASTIC_LAYER_COUNT_PROBE_EVERY_N_STEPS:-update brake}\n'
            '  --plastic__layer_count_probe_radius N=${PLASTIC_LAYER_COUNT_PROBE_RADIUS}\n'
            '  --plastic__layer_count_max_step N=${PLASTIC_LAYER_COUNT_MAX_STEP}\n',
            1,
        )

    value_group_old = '--plastic__log_interval_coarse|--plastic__layers_to_sample|--plastic__initial_layer_count|--plastic__max_permitted_layers|--plastic__layer_sampling_initialisation|--plastic__layer_count_objective|--plastic__layer_count_update_brake|--plastic__layer_count_probe__window_size_as_number_of_probes|--plastic__layer_count_probe_noise_lambda|--plastic__layer_count_cost_weight|--plastic__layer_memory_budget_gib|--plastic__cuda_allocator_reserve_gib|--plastic__geometry_learning_rate_multiplier)'
    value_group_new = '--plastic__coarse_phase|--plastic__phase_1_n_steps|--plastic__phase_1_starting_layer_count|--plastic__phase_1__number_of_trials|--plastic__phase_1_evaluation_steps_count|--plastic__log_interval_coarse|--plastic__layers_to_sample|--plastic__initial_layer_count|--plastic__max_permitted_layers|--plastic__layer_sampling_initialisation|--plastic__layer_count_objective|--plastic__layer_count_update_brake|--plastic__layer_count_probe__probe_every_n_steps|--plastic__layer_count_probe_radius|--plastic__layer_count_max_step|--plastic__layer_count_probe__window_size_as_number_of_probes|--plastic__layer_count_probe_noise_lambda|--plastic__layer_count_cost_weight|--plastic__layer_memory_budget_gib|--plastic__cuda_allocator_reserve_gib|--plastic__geometry_learning_rate_multiplier)'
    content = content.replace(value_group_old, value_group_new)

    cases_anchor = '      case "$1" in\n        --plastic__log_interval_coarse) PLASTIC_LOG_INTERVAL_COARSE="$2" ;;\n'
    cases_new = (
        '      case "$1" in\n'
        '        --plastic__coarse_phase) PLASTIC_COARSE_PHASE="$2" ;;\n'
        '        --plastic__phase_1_n_steps) PLASTIC_PHASE_1_N_STEPS="$2" ;;\n'
        '        --plastic__phase_1_starting_layer_count) PLASTIC_PHASE_1_STARTING_LAYER_COUNT="$2" ;;\n'
        '        --plastic__phase_1__number_of_trials) PLASTIC_PHASE_1_NUMBER_OF_TRIALS="$2" ;;\n'
        '        --plastic__phase_1_evaluation_steps_count) PLASTIC_PHASE_1_EVALUATION_STEPS_COUNT="$2" ;;\n'
        '        --plastic__log_interval_coarse) PLASTIC_LOG_INTERVAL_COARSE="$2" ;;\n'
        '        --plastic__layer_count_probe__probe_every_n_steps) PLASTIC_LAYER_COUNT_PROBE_EVERY_N_STEPS="$2" ;;\n'
        '        --plastic__layer_count_probe_radius) PLASTIC_LAYER_COUNT_PROBE_RADIUS="$2" ;;\n'
        '        --plastic__layer_count_max_step) PLASTIC_LAYER_COUNT_MAX_STEP="$2" ;;\n'
    )
    if '        --plastic__coarse_phase) PLASTIC_COARSE_PHASE="$2" ;;' not in content:
        content = content.replace(cases_anchor, cases_new, 1)

    equals_old = '--plastic__log_interval_coarse=*|--plastic__layers_to_sample=*|--plastic__initial_layer_count=*|--plastic__max_permitted_layers=*|--plastic__layer_sampling_initialisation=*|--plastic__layer_count_objective=*|--plastic__layer_count_update_brake=*|--plastic__layer_count_probe__window_size_as_number_of_probes=*|--plastic__layer_count_probe_noise_lambda=*|--plastic__layer_count_cost_weight=*|--plastic__layer_memory_budget_gib=*|--plastic__cuda_allocator_reserve_gib=*|--plastic__geometry_learning_rate_multiplier=*)'
    equals_new = '--plastic__coarse_phase=*|--plastic__phase_1_n_steps=*|--plastic__phase_1_starting_layer_count=*|--plastic__phase_1__number_of_trials=*|--plastic__phase_1_evaluation_steps_count=*|--plastic__log_interval_coarse=*|--plastic__layers_to_sample=*|--plastic__initial_layer_count=*|--plastic__max_permitted_layers=*|--plastic__layer_sampling_initialisation=*|--plastic__layer_count_objective=*|--plastic__layer_count_update_brake=*|--plastic__layer_count_probe__probe_every_n_steps=*|--plastic__layer_count_probe_radius=*|--plastic__layer_count_max_step=*|--plastic__layer_count_probe__window_size_as_number_of_probes=*|--plastic__layer_count_probe_noise_lambda=*|--plastic__layer_count_cost_weight=*|--plastic__layer_memory_budget_gib=*|--plastic__cuda_allocator_reserve_gib=*|--plastic__geometry_learning_rate_multiplier=*)'
    content = content.replace(equals_old, equals_new)

    equals_cases_anchor = '      case "$plastic_name" in\n        --plastic__log_interval_coarse) PLASTIC_LOG_INTERVAL_COARSE="$plastic_value" ;;\n'
    equals_cases_new = (
        '      case "$plastic_name" in\n'
        '        --plastic__coarse_phase) PLASTIC_COARSE_PHASE="$plastic_value" ;;\n'
        '        --plastic__phase_1_n_steps) PLASTIC_PHASE_1_N_STEPS="$plastic_value" ;;\n'
        '        --plastic__phase_1_starting_layer_count) PLASTIC_PHASE_1_STARTING_LAYER_COUNT="$plastic_value" ;;\n'
        '        --plastic__phase_1__number_of_trials) PLASTIC_PHASE_1_NUMBER_OF_TRIALS="$plastic_value" ;;\n'
        '        --plastic__phase_1_evaluation_steps_count) PLASTIC_PHASE_1_EVALUATION_STEPS_COUNT="$plastic_value" ;;\n'
        '        --plastic__log_interval_coarse) PLASTIC_LOG_INTERVAL_COARSE="$plastic_value" ;;\n'
        '        --plastic__layer_count_probe__probe_every_n_steps) PLASTIC_LAYER_COUNT_PROBE_EVERY_N_STEPS="$plastic_value" ;;\n'
        '        --plastic__layer_count_probe_radius) PLASTIC_LAYER_COUNT_PROBE_RADIUS="$plastic_value" ;;\n'
        '        --plastic__layer_count_max_step) PLASTIC_LAYER_COUNT_MAX_STEP="$plastic_value" ;;\n'
    )
    if '        --plastic__coarse_phase) PLASTIC_COARSE_PHASE="$plastic_value" ;;' not in content:
        content = content.replace(equals_cases_anchor, equals_cases_new, 1)

    validation_anchor = 'validate_positive_uint "$PLASTIC_LOG_INTERVAL_COARSE" "PLASTIC_LOG_INTERVAL_COARSE"\n'
    validation = (
        'case "$PLASTIC_COARSE_PHASE" in enabled|disabled) ;; *) echo "PLASTIC_COARSE_PHASE must be enabled or disabled." >&2; exit 2 ;; esac\n'
        '[[ -z "$PLASTIC_PHASE_1_N_STEPS" ]] || validate_positive_uint "$PLASTIC_PHASE_1_N_STEPS" "PLASTIC_PHASE_1_N_STEPS"\n'
        '[[ -z "$PLASTIC_PHASE_1_STARTING_LAYER_COUNT" ]] || validate_positive_uint "$PLASTIC_PHASE_1_STARTING_LAYER_COUNT" "PLASTIC_PHASE_1_STARTING_LAYER_COUNT"\n'
        '[[ -z "$PLASTIC_PHASE_1_NUMBER_OF_TRIALS" ]] || validate_positive_uint "$PLASTIC_PHASE_1_NUMBER_OF_TRIALS" "PLASTIC_PHASE_1_NUMBER_OF_TRIALS"\n'
        '[[ -z "$PLASTIC_PHASE_1_EVALUATION_STEPS_COUNT" ]] || validate_positive_uint "$PLASTIC_PHASE_1_EVALUATION_STEPS_COUNT" "PLASTIC_PHASE_1_EVALUATION_STEPS_COUNT"\n'
        '[[ -z "$PLASTIC_LAYER_COUNT_PROBE_EVERY_N_STEPS" ]] || validate_positive_uint "$PLASTIC_LAYER_COUNT_PROBE_EVERY_N_STEPS" "PLASTIC_LAYER_COUNT_PROBE_EVERY_N_STEPS"\n'
        'validate_positive_uint "$PLASTIC_LAYER_COUNT_PROBE_RADIUS" "PLASTIC_LAYER_COUNT_PROBE_RADIUS"\n'
        'validate_positive_uint "$PLASTIC_LAYER_COUNT_MAX_STEP" "PLASTIC_LAYER_COUNT_MAX_STEP"\n'
    )
    if 'case "$PLASTIC_COARSE_PHASE" in enabled|disabled)' not in content:
        content = content.replace(validation_anchor, validation_anchor + validation, 1)

    learned_anchor = 'if [[ "$PLASTIC_DO_LEARN_LAYER_COUNT" == true ]]; then\n'
    coarse_validation = (
        'if [[ "$PLASTIC_COARSE_PHASE" == enabled ]]; then\n'
        '  [[ "$PLASTIC_ENABLED" == true ]] || { echo "--plastic__coarse_phase enabled requires --plastic__enabled." >&2; exit 2; }\n'
        '  [[ "$PLASTIC_DO_LEARN_LAYER_COUNT" == true ]] || { echo "--plastic__coarse_phase enabled requires --plastic__do_learn_layer_count." >&2; exit 2; }\n'
        '  [[ -n "$PLASTIC_PHASE_1_N_STEPS" && -n "$PLASTIC_PHASE_1_STARTING_LAYER_COUNT" && -n "$PLASTIC_PHASE_1_NUMBER_OF_TRIALS" && -n "$PLASTIC_PHASE_1_EVALUATION_STEPS_COUNT" ]] || { echo "enabled COARSE requires every plastic__phase_1 control." >&2; exit 2; }\n'
        'fi\n'
    )
    if 'enabled COARSE requires every plastic__phase_1 control' not in content:
        content = content.replace(learned_anchor, coarse_validation + learned_anchor, 1)

    optional_anchor = '    optional_args+=(--plastic__layer_count_update_brake "$PLASTIC_LAYER_COUNT_UPDATE_BRAKE")\n'
    optional = (
        '    optional_args+=(--plastic__coarse_phase "$PLASTIC_COARSE_PHASE")\n'
        '    if [[ "$PLASTIC_COARSE_PHASE" == enabled ]]; then\n'
        '      optional_args+=(--plastic__phase_1_n_steps "$PLASTIC_PHASE_1_N_STEPS")\n'
        '      optional_args+=(--plastic__phase_1_starting_layer_count "$PLASTIC_PHASE_1_STARTING_LAYER_COUNT")\n'
        '      optional_args+=(--plastic__phase_1__number_of_trials "$PLASTIC_PHASE_1_NUMBER_OF_TRIALS")\n'
        '      optional_args+=(--plastic__phase_1_evaluation_steps_count "$PLASTIC_PHASE_1_EVALUATION_STEPS_COUNT")\n'
        '    fi\n'
        '    [[ -n "$PLASTIC_LAYER_COUNT_PROBE_EVERY_N_STEPS" ]] && optional_args+=(--plastic__layer_count_probe__probe_every_n_steps "$PLASTIC_LAYER_COUNT_PROBE_EVERY_N_STEPS")\n'
        '    optional_args+=(--plastic__layer_count_probe_radius "$PLASTIC_LAYER_COUNT_PROBE_RADIUS")\n'
        '    optional_args+=(--plastic__layer_count_max_step "$PLASTIC_LAYER_COUNT_MAX_STEP")\n'
    )
    if 'optional_args+=(--plastic__coarse_phase "$PLASTIC_COARSE_PHASE")' not in content:
        content = content.replace(optional_anchor, optional + optional_anchor, 1)

    startup_anchor = '  plastic depth:      enabled=$PLASTIC_ENABLED fixed=${PLASTIC_LAYERS_TO_SAMPLE:-N_LAYER} learn_count=$PLASTIC_DO_LEARN_LAYER_COUNT initial=${PLASTIC_INITIAL_LAYER_COUNT:-N_LAYER} max=${PLASTIC_MAX_PERMITTED_LAYERS:-N_LAYER} init=$PLASTIC_LAYER_SAMPLING_INITIALISATION objective=$PLASTIC_LAYER_COUNT_OBJECTIVE\n'
    startup = (
        '  plastic coarse:     phase=$PLASTIC_COARSE_PHASE start=${PLASTIC_PHASE_1_STARTING_LAYER_COUNT:--} trials=${PLASTIC_PHASE_1_NUMBER_OF_TRIALS:--} steps=${PLASTIC_PHASE_1_N_STEPS:--} eval=${PLASTIC_PHASE_1_EVALUATION_STEPS_COUNT:--} log=$PLASTIC_LOG_INTERVAL_COARSE roll_through=$PLASTIC_COARSE_PHASE_ROLL_THROUGH\n'
        '  plastic fine:       probe_interval=${PLASTIC_LAYER_COUNT_PROBE_EVERY_N_STEPS:-update_brake} radius=$PLASTIC_LAYER_COUNT_PROBE_RADIUS max_step=$PLASTIC_LAYER_COUNT_MAX_STEP brake=$PLASTIC_LAYER_COUNT_UPDATE_BRAKE\n'
    )
    if '  plastic coarse:' not in content:
        content = content.replace(startup_anchor, startup_anchor + startup, 1)

    export_anchor = 'export THOG2_FAST_DISCARD="$FAST_DISCARD"\n'
    exports = (
        'export THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS="$PLASTIC_LAYER_COUNT_PROBE_RADIUS"\n'
        'export THOG2_PLASTIC_LAYER_COUNT_MAX_STEP="$PLASTIC_LAYER_COUNT_MAX_STEP"\n'
    )
    if 'export THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS="$PLASTIC_LAYER_COUNT_PROBE_RADIUS"' not in content:
        content = content.replace(export_anchor, exports + export_anchor, 1)
    write(path, content)


def registered_help_and_console_boundary() -> None:
    path = "train_OWT.sh"
    content = read(path)
    old = '''      "$THOG2_REGISTRY_PYTHON" -m run_thog2_owt --print-geometry-registry
      printf '\ncanonical train_OWT.sh options\n------------------------------\n'
      bash ./train_OWT_core.sh -h
      printf '\nTorch compilation:\n  --torch-compile false|true|regional        false=eager, true=whole-model, regional=checkpoint-segment compile\n'
'''
    new = '''      "$THOG2_REGISTRY_PYTHON" -m run_thog2_owt --print-geometry-registry
      printf '\ncanonical train_OWT.sh options\n------------------------------\n'
      bash ./train_OWT_core.sh -h
      printf '\nregistered runner hyperparameters\n---------------------------------\n'
      "$THOG2_REGISTRY_PYTHON" -c 'from run_thog2_owt_core import build_parser; print(build_parser().format_help(), end="")'
      printf '\nTorch compilation:\n  --torch-compile false|true|regional        false=eager, true=whole-model, regional=checkpoint-segment compile\n'
'''
    if "registered runner hyperparameters" not in content:
        if old not in content:
            raise RuntimeError("registered help anchor missing")
        content = content.replace(old, new, 1)
    write(path, content)

    path = "sheet/__init__.py"
    content = read(path)
    content = content.replace(
        '# <<< THOG force bold explicit RGB yellow across validation-loss label and value',
        '# <<< THOG reset then force terminal bright-yellow plus bold across validation-loss label and value',
    )
    write(path, content)

    path = "sheet/plastic_depth_console_minor_patch.py"
    content = read(path).replace(
        'and int(completed_updates) <= int(getattr(config, "warmup_updates", 0))',
        'and int(completed_updates) < int(getattr(config, "warmup_updates", 0))',
    )
    write(path, content)


def tests() -> None:
    path = "tests/test_plastic_depth_wrapper_options.py"
    content = read(path).replace(
        '"4|2|--plastic__layer_count_probe_radius 4 --plastic__layer_count_max_step=2 marker",',
        '"4|2|--plastic__layer_count_probe_radius 4 --plastic__layer_count_max_step=2 marker",',
    )
    write(path, content)

    path = "tests/test_plastic_cli_console_refinements.py"
    content = read(path)
    if "test_warmup_brake_ends_at_the_actual_schedule_boundary" not in content:
        content += '''\n\ndef test_warmup_brake_ends_at_the_actual_schedule_boundary() -> None:
    from types import SimpleNamespace
    from sheet import plastic_depth_console_minor_patch as console

    trainer = SimpleNamespace(
        config=SimpleNamespace(
            plastic__enabled=True,
            plastic__do_learn_layer_count=True,
            plastic__freeze_geometry_during_warmup=True,
            warmup_updates=100,
        )
    )
    assert console._row_has_warmup_brake(trainer, 99)
    assert not console._row_has_warmup_brake(trainer, 100)


def test_registered_help_is_generated_from_the_complete_parser() -> None:
    source = (ROOT / "train_OWT.sh").read_text(encoding="utf-8")
    assert "registered runner hyperparameters" in source
    assert "build_parser().format_help()" in source


def test_shell_core_owns_every_coarse_and_fine_control() -> None:
    source = (ROOT / "train_OWT_core.sh").read_text(encoding="utf-8")
    for option in (
        "--plastic__coarse_phase",
        "--plastic__phase_1_n_steps",
        "--plastic__phase_1_starting_layer_count",
        "--plastic__phase_1__number_of_trials",
        "--plastic__phase_1_evaluation_steps_count",
        "--plastic__layer_count_probe__probe_every_n_steps",
        "--plastic__layer_count_probe_radius",
        "--plastic__layer_count_max_step",
        "--plastic__log_interval_coarse",
        "--plastic__coarse_phase_roll_through",
    ):
        assert option in source
    helper = (ROOT / "plastic_depth_lookahead_wrapper_options.sh").read_text(encoding="utf-8")
    assert "LOOKAHEAD_EXTRA_ARGS" not in helper
    assert "Python-extra boundary" not in helper
'''
    write(path, content)

    path = "tests/test_plastic_depth_lifecycle.py"
    content = read(path)
    if "    plastic__coarse_phase_roll_through: bool = False\n" not in content:
        content = content.replace(
            '    plastic__initial_layer_count: int = 2\n',
            '    plastic__initial_layer_count: int = 2\n    plastic__coarse_phase_roll_through: bool = False\n',
            1,
        )
    if "test_roll_through_skips_pause_and_builds_fine_immediately" not in content:
        content += '''\n\ndef test_roll_through_skips_pause_and_builds_fine_immediately() -> None:
    coordinator = _Coordinator()
    builds = []
    output = io.StringIO()

    def builder(**kwargs):
        builds.append(kwargs["phase"])
        return PlasticFreshTrainingState(
            trainer=SimpleNamespace(config=kwargs["resolved_config"]),
            phase=kwargs["phase"],
            active_layer_count=kwargs["active_layer_count"],
            instrumentation_namespace=kwargs["instrumentation_namespace"],
            fingerprint={},
        )

    outcome = run_plastic_coarse_fine_lifecycle(
        trainer_factory=lambda *_: None,
        resolved_config=_Config(plastic__coarse_phase_roll_through=True),
        train_tokens=object(),
        validation_tokens=object(),
        coarse_config=ResolvedPlasticCoarseConfig(True, (2,), 1, 1),
        objective="lowest_loss",
        maximum_layers=8,
        cost_weight=0.0,
        memory_budget_gib=None,
        geometry_initialisation="equidistant",
        console_stream=output,
        fresh_state_builder=builder,
        trial_runner=lambda state, **_: PlasticCoarseTrialResult(
            trial_index=1,
            layers=state.active_layer_count,
            status="success",
            validation_losses=(3.0,),
            training_elapsed_seconds=1.0,
            training_steps=1,
            tokens_per_update=100,
        ),
        state_destroyer=lambda state: setattr(state, "trainer", None),
        pause_runner=lambda **_: pytest.fail("roll-through must not invoke the pause runner"),
        coordinator_factory=lambda _: coordinator,
    )

    assert builds == ["coarse", "fine"]
    assert outcome.pause_result.disposition == "roll_through"
    assert outcome.provenance["pause"]["disposition"] == "roll_through"
    assert "starting FINE immediately" in output.getvalue()
    assert coordinator.barriers == 1
    outcome.close_coordinator()
'''
    write(path, content)


def main() -> None:
    rewrite_lookahead_helper()
    make_shell_core_authoritative()
    registered_help_and_console_boundary()
    tests()


if __name__ == "__main__":
    main()
# ^^^ THOG
