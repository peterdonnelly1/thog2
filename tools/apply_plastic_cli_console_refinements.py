#!/usr/bin/env python3
# vvv THOG
"""Apply the strict PLASTIC CLI, COARSE console, startup and artifact refinements."""

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
        raise RuntimeError(f"{path}: expected one replacement anchor, found {count}: {old[:100]!r}")
    write(path, content.replace(old, new, 1))


def replace_all(path: str, replacements: dict[str, str]) -> None:
    content = read(path)
    for old, new in replacements.items():
        content = content.replace(old, new)
    write(path, content)


def insert_after(path: str, anchor: str, addition: str) -> None:
    content = read(path)
    if addition.strip() in content:
        return
    count = content.count(anchor)
    if count != 1:
        raise RuntimeError(f"{path}: expected one insertion anchor, found {count}: {anchor!r}")
    write(path, content.replace(anchor, anchor + addition, 1))


def strict_argparse_registration() -> None:
    path = "sheet/argparse_underscore_alias_patch.py"
    write(
        path,
        '''# vvv THOG
"""Register PLASTIC controls only under their exact canonical double-underscore names."""

from __future__ import annotations

import argparse
from typing import Any


_ORIGINAL_ADD_ARGUMENT = argparse.ArgumentParser.add_argument


def _canonical_plastic_option(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
    destination = kwargs.get("dest")
    if not isinstance(destination, str) or not destination.startswith("plastic__"):
        return args
    rewritten = []
    replaced = False
    for argument in args:
        if isinstance(argument, str) and argument.startswith("--plastic-"):
            if replaced:
                raise RuntimeError(f"multiple PLASTIC option strings registered for {destination!r}")
            rewritten.append("--" + destination)
            replaced = True
        else:
            rewritten.append(argument)
    if not replaced and ("--" + destination) not in rewritten:
        raise RuntimeError(
            f"PLASTIC argparse registration must use its canonical destination: {destination!r}"
        )
    return tuple(rewritten)


def _add_argument_with_strict_plastic_names(
    self: argparse.ArgumentParser,
    *args: Any,
    **kwargs: Any,
):
    return _ORIGINAL_ADD_ARGUMENT(
        self,
        *_canonical_plastic_option(tuple(args), kwargs),
        **kwargs,
    )


if argparse.ArgumentParser.add_argument is not _add_argument_with_strict_plastic_names:
    argparse.ArgumentParser.add_argument = _add_argument_with_strict_plastic_names
# ^^^ THOG
''',
    )


def strict_wrapper_names() -> None:
    path = "train_OWT.sh"
    old_start = "# vvv THOG accept canonical double-underscore and legacy single-underscore long options while preserving established hyphen aliases\n"
    old_end = "# ^^^ THOG\n\n# vvv THOG expose exact PLASTIC lookahead controls through the one canonical wrapper"
    content = read(path)
    if "reject non-canonical PLASTIC aliases before any wrapper parses them" not in content:
        start = content.index(old_start)
        end = content.index(old_end, start)
        replacement = '''# vvv THOG reject non-canonical PLASTIC aliases before any wrapper parses them
thog2_normalize_nonplastic_long_option() {
  local option_name="$1"
  while [[ "$option_name" == *"__"* ]]; do
    option_name="${option_name//__/_}"
  done
  printf '%s' "${option_name//_/-}"
}
THOG2_STRICT_LONG_ARGS=()
while (( $# > 0 )); do
  case "$1" in
    --plastic__*|--no-plastic__*)
      THOG2_STRICT_LONG_ARGS+=("$1")
      shift
      ;;
    --plastic-*|--no-plastic-*|--plastic_[!_]*|--no-plastic_[!_]*)
      echo "Non-canonical PLASTIC option rejected: $1; use the exact --plastic__... or --no-plastic__... spelling." >&2
      exit 2
      ;;
    --*=*)
      THOG2_LONG_NAME="${1%%=*}"
      THOG2_LONG_VALUE="${1#*=}"
      THOG2_LONG_NAME="$(thog2_normalize_nonplastic_long_option "$THOG2_LONG_NAME")"
      THOG2_STRICT_LONG_ARGS+=("${THOG2_LONG_NAME}=${THOG2_LONG_VALUE}")
      shift
      ;;
    --*)
      THOG2_STRICT_LONG_ARGS+=("$(thog2_normalize_nonplastic_long_option "$1")")
      shift
      ;;
    *)
      THOG2_STRICT_LONG_ARGS+=("$1")
      shift
      ;;
  esac
done
set -- "${THOG2_STRICT_LONG_ARGS[@]}"
unset THOG2_STRICT_LONG_ARGS THOG2_LONG_NAME THOG2_LONG_VALUE
unset -f thog2_normalize_nonplastic_long_option
# ^^^ THOG

# vvv THOG expose exact PLASTIC lookahead controls through the one canonical wrapper'''
        content = content[:start] + replacement + content[end + len(old_end):]
        write(path, content)

    replacements = {
        "--plastic-coarse-phase": "--plastic__coarse_phase",
        "--plastic-phase-1-n-steps": "--plastic__phase_1_n_steps",
        "--plastic-phase-1-starting-layer-count": "--plastic__phase_1_starting_layer_count",
        "--plastic-phase-1-number-of-trials": "--plastic__phase_1__number_of_trials",
        "--plastic-phase-1-evaluation-steps-count": "--plastic__phase_1_evaluation_steps_count",
        "--plastic-layer-count-probe-interval": "--plastic__layer_count_probe__probe_every_n_steps",
        "--plastic-layer-count-probe-radius": "--plastic__layer_count_probe_radius",
        "--plastic-layer-count-max-step": "--plastic__layer_count_max_step",
    }
    replace_all("plastic_depth_lookahead_wrapper_options.sh", replacements)
    helper = read("plastic_depth_lookahead_wrapper_options.sh")
    if "--plastic__log_interval_coarse" not in helper:
        helper = helper.replace(
            "--plastic__coarse_phase|--plastic__phase_1_n_steps|--plastic__phase_1_starting_layer_count|--plastic__phase_1__number_of_trials|--plastic__phase_1_evaluation_steps_count|--plastic__layer_count_probe__probe_every_n_steps)",
            "--plastic__coarse_phase|--plastic__phase_1_n_steps|--plastic__phase_1_starting_layer_count|--plastic__phase_1__number_of_trials|--plastic__phase_1_evaluation_steps_count|--plastic__layer_count_probe__probe_every_n_steps|--plastic__log_interval_coarse)",
        ).replace(
            "--plastic__coarse_phase=*|--plastic__phase_1_n_steps=*|--plastic__phase_1_starting_layer_count=*|--plastic__phase_1__number_of_trials=*|--plastic__phase_1_evaluation_steps_count=*|--plastic__layer_count_probe__probe_every_n_steps=*)",
            "--plastic__coarse_phase=*|--plastic__phase_1_n_steps=*|--plastic__phase_1_starting_layer_count=*|--plastic__phase_1__number_of_trials=*|--plastic__phase_1_evaluation_steps_count=*|--plastic__layer_count_probe__probe_every_n_steps=*|--plastic__log_interval_coarse=*)",
        ).replace(
            "    --plastic__layer_count_probe_radius)",
            "    --plastic__coarse_phase_roll_through|--no-plastic__coarse_phase_roll_through)\n      THOG2_PLASTIC_LOOKAHEAD_EXTRA_ARGS+=(\"$1\")\n      shift\n      ;;\n    --plastic__layer_count_probe_radius)",
        ).replace(
            "    '  --plastic__phase_1_evaluation_steps_count N    final validation batches per trial' \\\n",
            "    '  --plastic__phase_1_evaluation_steps_count N    final validation batches per trial' \\\n    '  --plastic__log_interval_coarse N=10             COARSE progress cadence' \\\n    '  --plastic__coarse_phase_roll_through             skip the review delay and start FINE immediately' \\\n    '  --no-plastic__coarse_phase_roll_through          retain the review delay; default' \\\n",
        ).replace(
            "    '  Hyphenated and legacy single-underscore aliases remain accepted.' \\\n",
            "    '  Hyphenated and single-underscore PLASTIC aliases are rejected.' \\\n",
        )
        write("plastic_depth_lookahead_wrapper_options.sh", helper)


def core_config_fields() -> None:
    # Source-level parser names are canonical as well as the runtime registration.
    replacements = {
        "--plastic-enabled": "--plastic__enabled",
        "--plastic-coarse-phase": "--plastic__coarse_phase",
        "--plastic-phase-1-n-steps": "--plastic__phase_1_n_steps",
        "--plastic-phase-1-starting-layer-count": "--plastic__phase_1_starting_layer_count",
        "--plastic-phase-1-number-of-trials": "--plastic__phase_1__number_of_trials",
        "--plastic-phase-1-evaluation-steps-count": "--plastic__phase_1_evaluation_steps_count",
        "--plastic-layers-to-sample": "--plastic__layers_to_sample",
        "--plastic-do-learn-layer-count": "--plastic__do_learn_layer_count",
        "--plastic-initial-layer-count": "--plastic__initial_layer_count",
        "--plastic-max-permitted-layers": "--plastic__max_permitted_layers",
        "--plastic-layer-sampling-initialisation": "--plastic__layer_sampling_initialisation",
        "--plastic-layer-count-objective": "--plastic__layer_count_objective",
        "--plastic-layer-count-update-brake": "--plastic__layer_count_update_brake",
        "--plastic-layer-count-probe-interval": "--plastic__layer_count_probe__probe_every_n_steps",
        "--plastic-layer-count-probe-radius": "--plastic__layer_count_probe_radius",
        "--plastic-layer-count-max-step": "--plastic__layer_count_max_step",
        "--plastic-layer-count-probe-noise-window": "--plastic__layer_count_probe__window_size_as_number_of_probes",
        "--plastic-layer-count-probe-noise-min-observations": "--plastic__layer_count_min_probes",
        "--plastic-layer-count-probe-noise-lambda": "--plastic__layer_count_probe_noise_lambda",
        "--plastic-layer-count-cost-weight": "--plastic__layer_count_cost_weight",
        "--plastic-layer-memory-budget-gib": "--plastic__layer_memory_budget_gib",
        "--plastic-cuda-allocator-reserve-gib": "--plastic__cuda_allocator_reserve_gib",
        "--plastic-geometry-learning-rate-multiplier": "--plastic__geometry_learning_rate_multiplier",
        "--plastic-freeze-geometry-during-warmup": "--plastic__freeze_geometry_during_warmup",
    }
    replace_all("run_thog2_owt_core.py", replacements)

    insert_after(
        "run_thog2_owt_core.py",
        '    parser.add_argument("--plastic__coarse_phase", dest="plastic__coarse_phase", choices=("enabled", "disabled"), default="disabled")\n',
        '    parser.add_argument("--plastic__coarse_phase_roll_through", dest="plastic__coarse_phase_roll_through", action=argparse.BooleanOptionalAction, default=False)\n'
        '    parser.add_argument("--plastic__log_interval_coarse", dest="plastic__log_interval_coarse", type=int, default=10)\n',
    )
    insert_after(
        "run_thog2_owt_core.py",
        "        plastic__coarse_phase=arguments.plastic__coarse_phase,\n",
        "        plastic__coarse_phase_roll_through=arguments.plastic__coarse_phase_roll_through,\n"
        "        plastic__log_interval_coarse=arguments.plastic__log_interval_coarse,\n",
    )
    replace_all(
        "run_thog2_owt_core.py",
        {'parser.add_argument("--max-nonfinite-update-skips", type=int, default=10)': 'parser.add_argument("--max-nonfinite-update-skips", type=int, default=99999)'},
    )

    for path, fields_anchor in (
        ("sheet/run_config.py", '    "plastic__coarse_phase",\n'),
        ("sheet/training_config.py", '    "plastic__coarse_phase",\n'),
    ):
        insert_after(
            path,
            fields_anchor,
            '    "plastic__coarse_phase_roll_through",\n'
            '    "plastic__log_interval_coarse",\n',
        )

    insert_after(
        "sheet/run_config.py",
        '    plastic__coarse_phase: str = "disabled"\n',
        '    plastic__coarse_phase_roll_through: bool = False\n'
        '    plastic__log_interval_coarse: int = 10\n',
    )
    insert_after(
        "sheet/training_config.py",
        '    plastic__coarse_phase: str = "disabled"\n',
        '    plastic__coarse_phase_roll_through: bool = False\n'
        '    plastic__log_interval_coarse: int = 10\n',
    )
    replace_all(
        "sheet/run_config.py",
        {'max_nonfinite_update_skips: int = 10': 'max_nonfinite_update_skips: int = 99999'},
    )
    replace_all(
        "sheet/training_config.py",
        {'max_nonfinite_update_skips: int = 10': 'max_nonfinite_update_skips: int = 99999'},
    )

    for path in ("sheet/run_config.py", "sheet/training_config.py"):
        insert_after(
            path,
            '        if not isinstance(self.plastic__enabled, bool):\n',
            '            pass\n',
        )
        content = read(path)
        content = content.replace(
            '        if not isinstance(self.plastic__enabled, bool):\n            pass\n',
            '        if not isinstance(self.plastic__enabled, bool):\n',
        )
        bool_anchor = (
            '        if not isinstance(self.plastic__do_learn_layer_count, bool):\n'
        )
        if "plastic__coarse_phase_roll_through must be bool" not in content:
            insertion = (
                '        if not isinstance(self.plastic__coarse_phase_roll_through, bool):\n'
                '            raise ValueError("plastic__coarse_phase_roll_through must be bool")\n'
                '        if (\n'
                '            isinstance(self.plastic__log_interval_coarse, bool)\n'
                '            or not isinstance(self.plastic__log_interval_coarse, int)\n'
                '            or self.plastic__log_interval_coarse < 1\n'
                '        ):\n'
                '            raise ValueError("plastic__log_interval_coarse must be a positive integer")\n'
            )
            content = content.replace(bool_anchor, insertion + bool_anchor, 1)
        write(path, content)

    # Persist and transfer both scheduling controls.
    insert_after(
        "sheet/plastic_depth.py",
        '    coarse_phase: str = "disabled",\n',
        '    coarse_phase_roll_through: bool = False,\n'
        '    log_interval_coarse: int = 10,\n',
    )
    insert_after(
        "sheet/plastic_depth.py",
        '        "plastic__coarse_phase": coarse_phase,\n',
        '        "plastic__coarse_phase_roll_through": bool(coarse_phase_roll_through),\n'
        '        "plastic__log_interval_coarse": int(log_interval_coarse),\n',
    )
    for path in ("sheet/run_config.py", "sheet/training_config.py"):
        insert_after(
            path,
            '                coarse_phase=self.plastic__coarse_phase,\n',
            '                coarse_phase_roll_through=self.plastic__coarse_phase_roll_through,\n'
            '                log_interval_coarse=self.plastic__log_interval_coarse,\n',
        )

    insert_after(
        "sheet/run_config.py",
        '            plastic__coarse_phase=self.plastic__coarse_phase,\n',
        '            plastic__coarse_phase_roll_through=self.plastic__coarse_phase_roll_through,\n'
        '            plastic__log_interval_coarse=self.plastic__log_interval_coarse,\n',
    )


def artifact_descriptor() -> None:
    path = "sheet/run_config.py"
    old_fit = '''        fit_fields = [
            f"A_{self.gradient_accumulation_steps}",
            f"b_{self.batch_size}",
            f"c_{self._learning_rate_code(self.learning_rate)}",
            f"d_{dataset_label(self.dataset)}",
            f"f_{self._learning_rate_code(self.min_lr)}",
            f"w_{self.warmup_iters}",
        ]'''
    new_fit = '''        fit_fields = [
            f"d_{dataset_label(self.dataset)}",
            f"A_{self.gradient_accumulation_steps}",
            f"b_{self.batch_size}",
            f"c_{self._learning_rate_code(self.learning_rate)}",
            f"f_{self._learning_rate_code(self.min_lr)}",
            f"w_{self.warmup_iters}",
        ]'''
    replace_once(path, old_fit, new_fit)

    old_plastic = '''            plastic_fields = [
                f"PLN_{self.plastic__initial_active_layers}",
                f"PLM_{self.n_layer}",
                f"PLI_{self.plastic__layer_sampling_initialisation}",
                f"PLO_{self.plastic__layer_count_objective}",
            ]'''
    new_plastic = '''            sampling_label = {"equidistant": "equ", "random": "rndm"}[self.plastic__layer_sampling_initialisation]
            objective_label = {
                "lowest_loss": "loss",
                "relative_training_wall_time": "time",
                "layer_efficiency": "lyrs",
                "memory_budget": "mem",
            }[self.plastic__layer_count_objective]
            plastic_fields = [
                f"LN_{self.plastic__initial_active_layers}",
                f"LM_{self.n_layer}",
                f"LI_{sampling_label}",
                f"LO_{objective_label}",
            ]'''
    replace_once(path, old_plastic, new_plastic)
    replace_all(
        path,
        {
            'f"PLC_{': 'f"LC_{',
            'f"PLCS_{': 'f"LCS_{',
            'f"PLCT_{': 'f"LCT_{',
            'f"PLCE_{': 'f"LCE_{',
            'f"PLPI_{': 'f"LPI_{',
            'f"PLPR_{': 'f"LPR_{',
            'f"PLMS_{': 'f"LMS_{',
            'f"PLB_{self.plastic__layer_count_update_brake}': 'f"LB_{self.plastic__layer_count_update_brake}',
            'f"PLNW_{': 'f"LNW_{',
            'f"PLNM_{': 'f"LNM_{',
            'f"PLNL_{': 'f"LNL_{',
            'f"PLW_{': 'f"LW_{',
            'f"PLB_{self._artifact_float(self.plastic__layer_memory_budget_gib)}': 'f"LMB_{self._artifact_float(self.plastic__layer_memory_budget_gib)}',
            'f"PLG_{': 'f"LG_{',
            'plastic_fields.append("PLF_0")': 'plastic_fields.append("LF_0")',
            'sections.append("_".join(plastic_fields))': 'sections.append("P__" + "_".join(plastic_fields))',
        },
    )


def coarse_console_and_rollthrough() -> None:
    path = "sheet/plastic_depth_coarse_runner.py"
    replace_once(
        path,
        '''            f"PLASTIC COARSE - trial {trial_index}/{trial_count}",
            f"  layers:      {layers}",
            f"  training:    {n_steps} steps, starting at step 0",''',
        '''            f"TRIAL {trial_index}/{trial_count}",
            f"  layers:      {layers}",
            f"  steps:       {n_steps}",''',
    )
    replace_once(
        path,
        '''    if progress_sink is not None:
        status = "local step zero" if completed_at_start == 0 else "resumed"
        progress_sink(
            f"C {trial_index:02d} step {completed_at_start:6d}/{n_steps:<6d} "
            f"layers={state.active_layer_count:<4d} {status}"
        )''',
        '''    log_interval_coarse = int(getattr(trainer.config, "plastic__log_interval_coarse", 10))
    if progress_sink is not None:
        status = "local step zero" if completed_at_start == 0 else "resumed"
        progress_sink(
            f"C {trial_index:02d} {completed_at_start:6d}/{n_steps:<6d} "
            f"{status:<22} {float(prior_training_elapsed_seconds):8.1f}s"
        )''',
    )
    replace_once(
        path,
        '''            if progress_sink is not None:
                progress_sink(
                    f"C {trial_index:02d} step {local_step:6d}/{n_steps:<6d} "
                    f"layers={state.active_layer_count:<4d} "
                    f"loss={training_loss:.6f}"
                )''',
        '''            progress_due = (
                local_step == 1
                or local_step == n_steps
                or local_step % log_interval_coarse == 0
            )
            if progress_sink is not None and progress_due:
                elapsed_seconds = (
                    float(prior_training_elapsed_seconds)
                    + max(0.0, float(clock() - started))
                )
                progress_sink(
                    f"C {trial_index:02d} {local_step:6d}/{n_steps:<6d} "
                    f"loss={training_loss:.6f} {elapsed_seconds:8.1f}s"
                )''',
    )

    lifecycle = "sheet/plastic_depth_lifecycle.py"
    insert_after(
        lifecycle,
        '        trial_count = len(coarse_config.candidate_layers)\n',
        '        if coordinator.is_primary:\n'
        '            _emit(console_stream, "COARSE TRIALS")\n'
        '            _emit(\n'
        '                console_stream,\n'
        '                "  layer counts: " + ", ".join(str(value) for value in coarse_config.candidate_layers),\n'
        '            )\n',
    )
    old_pause = '''        pause_result = run_distributed_plastic_coarse_review_pause(
            coordinator,
            duration_seconds=pause_duration_seconds,
            output=console_stream,
            pause_runner=pause_runner,
        )'''
    new_pause = '''        if bool(getattr(resolved_config, "plastic__coarse_phase_roll_through", False)):
            pause_result = PlasticCoarsePauseResult(
                disposition="roll_through",
                elapsed_seconds=0.0,
                remaining_seconds=0.0,
            )
            if coordinator.is_primary:
                _emit(console_stream, "COARSE roll-through enabled; starting FINE immediately.")
            coordinator.barrier()
        else:
            pause_result = run_distributed_plastic_coarse_review_pause(
                coordinator,
                duration_seconds=pause_duration_seconds,
                output=console_stream,
                pause_runner=pause_runner,
            )'''
    replace_once(lifecycle, old_pause, new_pause)

    report = "sheet/plastic_depth_coarse.py"
    replace_once(
        report,
        '        marker = " <<< WINNER" if row is winner else ""\n',
        '        marker = (\n'
        '            f" {_WINNER_STYLE_START}<<< WINNER{_STYLE_END}"\n'
        '            if row is winner and ansi\n'
        '            else (" <<< WINNER" if row is winner else "")\n'
        '        )\n',
    )
    replace_once(
        report,
        '        status = "failed" if result.status == "failed" else ("ok" if row.selectable else "unselectable")\n',
        '        if result.status == "failed":\n'
        '            reason = " ".join(str(result.error_message or "no reason recorded").split())\n'
        '            status = f"failed - because {result.error_class or \'Exception\'}: {reason}"\n'
        '        else:\n'
        '            status = "ok" if row.selectable else "unselectable"\n',
    )
    replace_once(
        report,
        '            f"{within}{_format_optional(row.score, 18, 6)} {status:>9}{marker}"\n',
        '            f"{within}{_format_optional(row.score, 18, 6)} {status}{marker}"\n',
    )
    replace_once(
        report,
        '''        if row is winner and ansi:
            line = f"{_WINNER_STYLE_START}{line}{_STYLE_END}"
        lines.append(line)''',
        '''        lines.append(line)''',
    )

    recovery = "sheet/plastic_depth_coarse_runtime_recovery_patch.py"
    content = read(recovery)
    if "inline failure reasons now belong to the canonical result row" not in content:
        start = content.index("def _render_plastic_coarse_report_with_failures")
        end = content.index("\n\ndef _log_scalars_without_rewinding_wandb", start)
        function = '''def _render_plastic_coarse_report_with_failures(*args: Any, **kwargs: Any) -> str:
    # vvv THOG inline failure reasons now belong to the canonical result row
    return _ORIGINAL_RENDER_PLASTIC_COARSE_REPORT(*args, **kwargs)
    # ^^^ THOG
'''
        content = content[:start] + function + content[end:]
        write(recovery, content)


def console_colours_and_brakes() -> None:
    replace_all(
        "sheet/__init__.py",
        {'_stage6_trainer._PROGRESS_VALIDATION_FIELD_STYLE_START = "\\033[1;38;2;255;255;0m"': '_stage6_trainer._PROGRESS_VALIDATION_FIELD_STYLE_START = "\\033[0;1;93m"'},
    )
    path = "sheet/plastic_depth_console_minor_patch.py"
    insert_after(path, '_PALE_RED = "\\033[38;2;255;150;150m"\n', '_PALE_CYAN = "\\033[38;2;150;220;255m"\n')
    insert_after(
        path,
        '''def _row_has_active_update_brake(trainer: Any, completed_updates: int) -> bool:
''',
        '',
    )
    content = read(path)
    if "def _row_has_warmup_brake" not in content:
        anchor = "# ^^^ THOG\n\n\ndef _remove_probe_fields"
        addition = '''# ^^^ THOG


# vvv THOG expose the separate warmup gate whenever learned count movement is prohibited
def _row_has_warmup_brake(trainer: Any, completed_updates: int) -> bool:
    config = getattr(trainer, "config", None)
    return (
        bool(getattr(config, "plastic__enabled", False))
        and bool(getattr(config, "plastic__do_learn_layer_count", False))
        and bool(getattr(config, "plastic__freeze_geometry_during_warmup", False))
        and int(completed_updates) <= int(getattr(config, "warmup_updates", 0))
    )
# ^^^ THOG


def _remove_probe_fields'''
        if anchor not in content:
            raise RuntimeError("warmup brake insertion anchor missing")
        content = content.replace(anchor, addition, 1)
    if 'values["plastic_warmup_brake_active"]' not in content:
        old = '''    if _row_has_active_update_brake(self, completed_updates):
        values["plastic_update_brake_active"] = True
    else:
        values.pop("plastic_update_brake_active", None)
    return values'''
        new = '''    if _row_has_active_update_brake(self, completed_updates):
        values["plastic_update_brake_active"] = True
    else:
        values.pop("plastic_update_brake_active", None)
    if _row_has_warmup_brake(self, completed_updates):
        values["plastic_warmup_brake_active"] = True
    else:
        values.pop("plastic_warmup_brake_active", None)
    return values'''
        if old not in content:
            raise RuntimeError("warmup payload anchor missing")
        content = content.replace(old, new, 1)
    if "<<< warmup braked enabled" not in content:
        old = '''    if bool(payload.get("plastic_update_brake_active", False)):
        line = f"{line.rstrip()}  {_PALE_RED}<<< update brake on{_RESET}"
    return line'''
        new = '''    if bool(payload.get("plastic_update_brake_active", False)):
        line = f"{line.rstrip()}  {_PALE_RED}<<< update brake on{_RESET}"
    if bool(payload.get("plastic_warmup_brake_active", False)):
        line = f"{line.rstrip()}  {_PALE_CYAN}<<< warmup braked enabled{_RESET}"
    return line'''
        if old not in content:
            raise RuntimeError("warmup suffix anchor missing")
        content = content.replace(old, new, 1)
    write(path, content)


def core_wrapper_help_and_startup() -> None:
    path = "train_OWT_core.sh"
    content = read(path)
    if "MAX_NONFINITE_UPDATE_SKIPS=" not in content:
        content = content.replace(
            "PLASTIC_FREEZE_GEOMETRY_DURING_WARMUP=true\n",
            "PLASTIC_FREEZE_GEOMETRY_DURING_WARMUP=true\n"
            "PLASTIC_LOG_INTERVAL_COARSE=10\n"
            "PLASTIC_COARSE_PHASE_ROLL_THROUGH=false\n"
            "MAX_NONFINITE_UPDATE_SKIPS=99999\n",
            1,
        )
    replacements = {
        "--plastic-enabled": "--plastic__enabled",
        "--no-plastic-enabled": "--no-plastic__enabled",
        "--plastic-layers-to-sample": "--plastic__layers_to_sample",
        "--plastic-do-learn-layer-count": "--plastic__do_learn_layer_count",
        "--no-plastic-do-learn-layer-count": "--no-plastic__do_learn_layer_count",
        "--plastic-initial-layer-count": "--plastic__initial_layer_count",
        "--plastic-max-permitted-layers": "--plastic__max_permitted_layers",
        "--plastic-layer-sampling-initialisation": "--plastic__layer_sampling_initialisation",
        "--plastic-layer-count-objective": "--plastic__layer_count_objective",
        "--plastic-layer-count-update-brake": "--plastic__layer_count_update_brake",
        "--plastic-layer-count-probe-noise-window": "--plastic__layer_count_probe__window_size_as_number_of_probes",
        "--plastic-layer-count-probe-noise-min-observations": "--plastic__layer_count_min_probes",
        "--plastic-layer-count-probe-noise-lambda": "--plastic__layer_count_probe_noise_lambda",
        "--plastic-layer-count-cost-weight": "--plastic__layer_count_cost_weight",
        "--plastic-layer-memory-budget-gib": "--plastic__layer_memory_budget_gib",
        "--plastic-cuda-allocator-reserve-gib": "--plastic__cuda_allocator_reserve_gib",
        "--plastic-geometry-learning-rate-multiplier": "--plastic__geometry_learning_rate_multiplier",
        "--plastic-freeze-geometry-during-warmup": "--plastic__freeze_geometry_during_warmup",
        "--no-plastic-freeze-geometry-during-warmup": "--no-plastic__freeze_geometry_during_warmup",
    }
    for old, new in replacements.items():
        content = content.replace(old, new)
    if "--plastic__log_interval_coarse" not in content:
        content = content.replace(
            "  --plastic__freeze_geometry_during_warmup | --no-plastic__freeze_geometry_during_warmup\n",
            "  --plastic__freeze_geometry_during_warmup | --no-plastic__freeze_geometry_during_warmup\n"
            "  --plastic__log_interval_coarse N=${PLASTIC_LOG_INTERVAL_COARSE}\n"
            "  --plastic__coarse_phase_roll_through | --no-plastic__coarse_phase_roll_through\n",
            1,
        )
    if "--max-nonfinite-update-skips" not in content:
        content = content.replace(
            "Schedule/logging:\n",
            "Schedule/logging:\n  --max-nonfinite-update-skips N=${MAX_NONFINITE_UPDATE_SKIPS}  tolerated skipped non-finite updates\n",
            1,
        )
    # Parse the two new PLASTIC controls and non-finite limit in the wrapper.
    if "--plastic__coarse_phase_roll_through)" not in content:
        content = content.replace(
            "    --plastic__enabled) PLASTIC_ENABLED=true; shift ;;\n",
            "    --plastic__enabled) PLASTIC_ENABLED=true; shift ;;\n"
            "    --plastic__coarse_phase_roll_through) PLASTIC_COARSE_PHASE_ROLL_THROUGH=true; shift ;;\n"
            "    --no-plastic__coarse_phase_roll_through) PLASTIC_COARSE_PHASE_ROLL_THROUGH=false; shift ;;\n",
            1,
        )
        content = content.replace(
            "--plastic__layers_to_sample|--plastic__initial_layer_count|--plastic__max_permitted_layers|",
            "--plastic__log_interval_coarse|--plastic__layers_to_sample|--plastic__initial_layer_count|--plastic__max_permitted_layers|",
        ).replace(
            "        --plastic__layers_to_sample) PLASTIC_LAYERS_TO_SAMPLE=\"$2\" ;;\n",
            "        --plastic__log_interval_coarse) PLASTIC_LOG_INTERVAL_COARSE=\"$2\" ;;\n"
            "        --plastic__layers_to_sample) PLASTIC_LAYERS_TO_SAMPLE=\"$2\" ;;\n",
            1,
        ).replace(
            "--plastic__layers_to_sample=*|--plastic__initial_layer_count=*|--plastic__max_permitted_layers=*|",
            "--plastic__log_interval_coarse=*|--plastic__layers_to_sample=*|--plastic__initial_layer_count=*|--plastic__max_permitted_layers=*|",
        ).replace(
            "        --plastic__layers_to_sample) PLASTIC_LAYERS_TO_SAMPLE=\"$plastic_value\" ;;\n",
            "        --plastic__log_interval_coarse) PLASTIC_LOG_INTERVAL_COARSE=\"$plastic_value\" ;;\n"
            "        --plastic__layers_to_sample) PLASTIC_LAYERS_TO_SAMPLE=\"$plastic_value\" ;;\n",
            1,
        )
    if "--max-nonfinite-update-skips)" not in content:
        content = content.replace(
            "    --optimizer)\n",
            "    --max-nonfinite-update-skips)\n"
            "      (( $# >= 2 )) || { echo \"--max-nonfinite-update-skips requires a non-negative integer\" >&2; exit 2; }\n"
            "      MAX_NONFINITE_UPDATE_SKIPS=\"$2\"\n"
            "      shift 2\n"
            "      ;;\n"
            "    --max-nonfinite-update-skips=*)\n"
            "      MAX_NONFINITE_UPDATE_SKIPS=\"${1#*=}\"\n"
            "      shift\n"
            "      ;;\n"
            "    --optimizer)\n",
            1,
        )
    if "validate_positive_uint \"$PLASTIC_LOG_INTERVAL_COARSE\"" not in content:
        content = content.replace(
            "validate_true_false \"$PLASTIC_FREEZE_GEOMETRY_DURING_WARMUP\" \"PLASTIC_FREEZE_GEOMETRY_DURING_WARMUP\"\n",
            "validate_true_false \"$PLASTIC_FREEZE_GEOMETRY_DURING_WARMUP\" \"PLASTIC_FREEZE_GEOMETRY_DURING_WARMUP\"\n"
            "validate_true_false \"$PLASTIC_COARSE_PHASE_ROLL_THROUGH\" \"PLASTIC_COARSE_PHASE_ROLL_THROUGH\"\n"
            "validate_positive_uint \"$PLASTIC_LOG_INTERVAL_COARSE\" \"PLASTIC_LOG_INTERVAL_COARSE\"\n"
            "validate_nonnegative_uint \"$MAX_NONFINITE_UPDATE_SKIPS\" \"MAX_NONFINITE_UPDATE_SKIPS\"\n",
            1,
        )
    write(path, content)


def tests() -> None:
    path = ROOT / "tests/test_plastic_cli_console_refinements.py"
    path.write_text(
        '''# vvv THOG
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import run_thog2_owt_core as core
from sheet.plastic_depth_coarse import PlasticCoarseTrialResult, ScoredPlasticCoarseTrial, render_plastic_coarse_report
from sheet.plastic_depth_coarse_runner import render_plastic_coarse_trial_header
from sheet.run_config import OwtRunConfig


ROOT = Path(__file__).resolve().parents[1]


def test_core_parser_accepts_only_double_underscore_plastic_names() -> None:
    parser = core.build_parser()
    parsed = parser.parse_args(["--plastic__enabled", "--plastic__coarse_phase_roll_through"])
    assert parsed.plastic__enabled is True
    assert parsed.plastic__coarse_phase_roll_through is True
    with pytest.raises(SystemExit):
        parser.parse_args(["--plastic-enabled"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--plastic_enabled"])


def test_wrapper_rejects_noncanonical_plastic_aliases() -> None:
    for alias in ("--plastic-enabled", "--plastic_enabled"):
        completed = subprocess.run(
            ("bash", "./train_OWT.sh", alias, "-h"),
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 2
        assert "Non-canonical PLASTIC option rejected" in completed.stderr


def test_artifact_uses_dataset_first_and_one_plastic_group_prefix() -> None:
    config = OwtRunConfig(
        model_type="sheet",
        n_layer=32,
        n_head=4,
        n_embd=64,
        block_size=32,
        batch_size=2,
        gradient_accumulation_steps=3,
        max_iters=20,
        warmup_iters=2,
        plastic__enabled=True,
        plastic__coarse_phase="enabled",
        plastic__phase_1_starting_layer_count=4,
        plastic__phase_1_n_steps=20,
        plastic__phase_1__number_of_trials=4,
        plastic__phase_1_evaluation_steps_count=10,
        plastic__do_learn_layer_count=True,
        plastic__max_permitted_layers=32,
    )
    fragment = config.parameter_artifact_fragment()
    assert fragment.startswith("d_owt_A_3_b_2_c_60_f_6_w_2")
    assert "__P__LN_4_LM_32_LI_equ_LO_loss_LC_4_LCS_20_LCT_4_LCE_10" in fragment
    assert "PLN_" not in fragment
    assert "PLI_" not in fragment


def test_coarse_header_and_winner_marker_contract() -> None:
    header = render_plastic_coarse_trial_header(
        trial_index=1,
        trial_count=4,
        layers=4,
        n_steps=20,
        evaluation_steps_count=10,
        objective="lowest_loss",
        geometry_initialisation="equidistant",
    )
    assert header.startswith("TRIAL 1/4")
    assert "steps:       20" in header
    assert "starting at step" not in header

    result = PlasticCoarseTrialResult(
        trial_index=1,
        layers=4,
        status="success",
        validation_losses=(3.0,),
        training_losses=(4.0,),
        training_elapsed_seconds=1.0,
        training_steps=1,
        tokens_per_update=10,
    )
    winner = ScoredPlasticCoarseTrial(result, "lowest_loss", "loss_score", 3.0, True, None, None)
    report = render_plastic_coarse_report((winner,), winner, training_steps=1, evaluation_steps_count=1, ansi=True)
    winner_line = report.splitlines()[-1]
    assert winner_line.startswith("    1")
    assert "\\x1b[1;92m<<< WINNER\\x1b[0m" in winner_line


def test_defaults() -> None:
    parser = core.build_parser()
    values = parser.parse_args([])
    assert values.max_nonfinite_update_skips == 99999
    assert values.plastic__log_interval_coarse == 10
    assert values.plastic__coarse_phase_roll_through is False
# ^^^ THOG
''',
        encoding="utf-8",
    )


def main() -> None:
    strict_argparse_registration()
    strict_wrapper_names()
    core_config_fields()
    artifact_descriptor()
    coarse_console_and_rollthrough()
    console_colours_and_brakes()
    core_wrapper_help_and_startup()
    tests()


if __name__ == "__main__":
    main()
# ^^^ THOG
