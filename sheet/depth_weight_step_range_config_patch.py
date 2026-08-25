# vvv THOG
"""Register, preserve and report INSTRA weight capture-window hyperparameters."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Mapping, Optional, Sequence

from . import local_chart_store as _local_chart_store


_START_DESTINATION = "instrumentation__depth_weight_curves__start_step"
_END_DESTINATION = "instrumentation__depth_weight_curves__end_step"
_START_OPTION = f"--{_START_DESTINATION}"
_END_OPTION = f"--{_END_DESTINATION}"
_START_ENVIRONMENT = "THOG2_INSTRUMENTATION_DEPTH_WEIGHT_CURVES_START_STEP"
_END_ENVIRONMENT = "THOG2_INSTRUMENTATION_DEPTH_WEIGHT_CURVES_END_STEP"
_CLI_INSTALLED_ATTRIBUTE = "_thog_depth_weight_step_range_arguments_installed"
_EXPLICIT_STEP_RANGE: dict[str, int] = {}

_DEPTH_WEIGHT_ENVIRONMENT_ROWS = (
    (
        "instrumentation__depth_weight_curves__scalar_weights_per_matrix",
        "THOG2_INSTRUMENTATION_DEPTH_WEIGHT_CURVES_SCALAR_WEIGHTS_PER_MATRIX",
        "3",
    ),
    (
        "instrumentation__depth_weight_curves__depth_evaluation_points",
        "THOG2_INSTRUMENTATION_DEPTH_WEIGHT_CURVES_DEPTH_EVALUATION_POINTS",
        "256",
    ),
    (
        "instrumentation__depth_weight_curves__time_mode",
        "THOG2_INSTRUMENTATION_DEPTH_WEIGHT_CURVES_TIME_MODE",
        "latest",
    ),
    (
        "instrumentation__depth_weight_curves__history_length",
        "THOG2_INSTRUMENTATION_DEPTH_WEIGHT_CURVES_HISTORY_LENGTH",
        "20",
    ),
    (
        "instrumentation__depth_weight_curves__log_every_n_steps",
        "THOG2_INSTRUMENTATION_DEPTH_WEIGHT_CURVES_LOG_EVERY_N_STEPS",
        "100",
    ),
    (
        "instrumentation__depth_weight_curves__same_coordinates_all_runs",
        "THOG2_INSTRUMENTATION_DEPTH_WEIGHT_CURVES_SAME_COORDINATES_ALL_RUNS",
        "false",
    ),
    (
        "instrumentation__depth_weight_curves__destination",
        "THOG2_INSTRUMENTATION_DEPTH_WEIGHT_CURVES_DESTINATION",
        "local",
    ),
    (_START_DESTINATION, _START_ENVIRONMENT, None),
    (_END_DESTINATION, _END_ENVIRONMENT, None),
)
_INSTRUMENTATION_CONSOLE_PRINTED = False


def _nonnegative_integer(value: str) -> int:
    try:
        resolved = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer; got {value!r}") from error
    if resolved < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer; got {value!r}")
    return resolved


def _raw_option_value(arguments: Sequence[str], option: str) -> Optional[int]:
    resolved: Optional[int] = None
    prefix = f"{option}="
    for index, argument in enumerate(arguments):
        candidate: Optional[str] = None
        if argument == option:
            if index + 1 < len(arguments):
                candidate = arguments[index + 1]
        elif argument.startswith(prefix):
            candidate = argument[len(prefix):]
        if candidate is None:
            continue
        try:
            numeric = int(candidate)
        except (TypeError, ValueError):
            continue
        if numeric >= 0:
            resolved = numeric
    return resolved


def _capture_original_argv_step_range(arguments: Sequence[str]) -> None:
    # Capture before the runner's layered argparse passes can replace an explicit
    # value with this compatibility parser's None default.
    for destination, option, environment in (
        (_START_DESTINATION, _START_OPTION, _START_ENVIRONMENT),
        (_END_DESTINATION, _END_OPTION, _END_ENVIRONMENT),
    ):
        value = _raw_option_value(arguments, option)
        if value is None:
            continue
        _EXPLICIT_STEP_RANGE[destination] = value
        os.environ[environment] = str(value)


_capture_original_argv_step_range(tuple(sys.argv[1:]))


def _ensure_cli_arguments(parser: argparse.ArgumentParser) -> None:
    if bool(getattr(parser, _CLI_INSTALLED_ATTRIBUTE, False)):
        return
    if _START_OPTION not in parser._option_string_actions:
        parser.add_argument(_START_OPTION, dest=_START_DESTINATION, type=_nonnegative_integer, default=None)
    if _END_OPTION not in parser._option_string_actions:
        parser.add_argument(_END_OPTION, dest=_END_DESTINATION, type=_nonnegative_integer, default=None)
    setattr(parser, _CLI_INSTALLED_ATTRIBUTE, True)


def _publish_step_range_environment(namespace: argparse.Namespace) -> None:
    # argparse is layered repeatedly in this runner; a later parser seeing its
    # default None must never erase an explicit range captured by an earlier pass.
    for destination, environment in (
        (_START_DESTINATION, _START_ENVIRONMENT),
        (_END_DESTINATION, _END_ENVIRONMENT),
    ):
        value = getattr(namespace, destination, None)
        if value is None:
            continue
        resolved = int(value)
        _EXPLICIT_STEP_RANGE[destination] = resolved
        os.environ[environment] = str(resolved)


_ORIGINAL_PARSE_KNOWN_ARGS = argparse.ArgumentParser.parse_known_args


def _parse_known_args_with_weight_step_range(
    self: argparse.ArgumentParser,
    args: Optional[Sequence[str]] = None,
    namespace: Optional[argparse.Namespace] = None,
):
    _ensure_cli_arguments(self)
    parsed, remaining = _ORIGINAL_PARSE_KNOWN_ARGS(self, args=args, namespace=namespace)
    _publish_step_range_environment(parsed)
    return parsed, remaining


argparse.ArgumentParser.parse_known_args = _parse_known_args_with_weight_step_range


_ORIGINAL_LOCAL_CHART_STORE_INIT = _local_chart_store.LocalChartStore.__init__


def _local_chart_store_init_with_weight_step_range(
    self: Any,
    path: Any,
    *,
    run_name: str,
    run_id: Optional[str] = None,
    wandb_run_id: Optional[str] = None,
    wandb_url: Optional[str] = None,
    config: Mapping[str, Any],
) -> None:
    enriched_config = dict(config)
    for destination, environment in (
        (_START_DESTINATION, _START_ENVIRONMENT),
        (_END_DESTINATION, _END_ENVIRONMENT),
    ):
        value = _EXPLICIT_STEP_RANGE.get(destination)
        if value is None:
            environment_value = os.environ.get(environment)
            value = None if environment_value is None else int(environment_value)
        if value is not None:
            enriched_config[destination] = int(value)
    _ORIGINAL_LOCAL_CHART_STORE_INIT(
        self,
        path,
        run_name=run_name,
        run_id=run_id,
        wandb_run_id=wandb_run_id,
        wandb_url=wandb_url,
        config=enriched_config,
    )


_local_chart_store.LocalChartStore.__init__ = _local_chart_store_init_with_weight_step_range


def _instrumentation_console_rows(config: Any) -> tuple[tuple[str, Any], ...]:
    rows: dict[str, Any] = {}
    try:
        config_values = vars(config)
    except TypeError:
        config_values = {}
    for name, value in config_values.items():
        if str(name).startswith("instrumentation__"):
            rows[str(name)] = value
    for name, environment, default in _DEPTH_WEIGHT_ENVIRONMENT_ROWS:
        value = os.environ.get(environment)
        if value is None:
            value = default
        rows[name] = value
    return tuple(sorted(rows.items()))


def _console_value(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _print_instrumentation_section(config: Any) -> None:
    rows = _instrumentation_console_rows(config)
    if not rows:
        return
    width = max(len(f"{name}:") for name, _value in rows) + 3
    print("instrumentation", flush=True)
    for name, value in rows:
        print(f"  {name + ':':<{width}}{_console_value(value)}", flush=True)


# Stage6 is imported here only after the core sheet modules required above exist.
# The wrapper evaluates configuration at run start, after every argparse layer has
# published its effective values, and prints once on the primary process.
from . import stage6_trainer as _stage6_trainer

_ORIGINAL_STAGE6_PRINT_PROGRESS = _stage6_trainer.Stage6Trainer._print_progress


def _print_progress_with_instrumentation(self: Any, run_id: str, event: str, **payload: Any) -> None:
    global _INSTRUMENTATION_CONSOLE_PRINTED
    if (
        event == "run_started"
        and not _INSTRUMENTATION_CONSOLE_PRINTED
        and bool(getattr(getattr(self, "distributed", None), "is_primary", True))
    ):
        _print_instrumentation_section(getattr(self, "config", None))
        print(flush=True)
        _INSTRUMENTATION_CONSOLE_PRINTED = True
    return _ORIGINAL_STAGE6_PRINT_PROGRESS(self, run_id, event, **payload)


_stage6_trainer.Stage6Trainer._print_progress = _print_progress_with_instrumentation
# ^^^ THOG
