# vvv THOG
"""Register and persist display-only INSTRA weight-step window hyperparameters."""

from __future__ import annotations

import argparse
import os
from typing import Any, Mapping, Optional, Sequence

from . import local_chart_store as _local_chart_store


_START_DESTINATION = "instrumentation__depth_weight_curves__start_step"
_END_DESTINATION = "instrumentation__depth_weight_curves__end_step"
_START_OPTION = f"--{_START_DESTINATION}"
_END_OPTION = f"--{_END_DESTINATION}"
_START_ENVIRONMENT = "THOG2_INSTRUMENTATION_DEPTH_WEIGHT_CURVES_START_STEP"
_END_ENVIRONMENT = "THOG2_INSTRUMENTATION_DEPTH_WEIGHT_CURVES_END_STEP"
_CLI_INSTALLED_ATTRIBUTE = "_thog_depth_weight_step_range_arguments_installed"


def _nonnegative_integer(value: str) -> int:
    try:
        resolved = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer; got {value!r}") from error
    if resolved < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer; got {value!r}")
    return resolved


def _ensure_cli_arguments(parser: argparse.ArgumentParser) -> None:
    if bool(getattr(parser, _CLI_INSTALLED_ATTRIBUTE, False)):
        return
    if _START_OPTION not in parser._option_string_actions:
        parser.add_argument(_START_OPTION, dest=_START_DESTINATION, type=_nonnegative_integer, default=None)
    if _END_OPTION not in parser._option_string_actions:
        parser.add_argument(_END_OPTION, dest=_END_DESTINATION, type=_nonnegative_integer, default=None)
    setattr(parser, _CLI_INSTALLED_ATTRIBUTE, True)


def _publish_step_range_environment(namespace: argparse.Namespace) -> None:
    for destination, environment in (
        (_START_DESTINATION, _START_ENVIRONMENT),
        (_END_DESTINATION, _END_ENVIRONMENT),
    ):
        value = getattr(namespace, destination, None)
        if value is None:
            os.environ.pop(environment, None)
        else:
            os.environ[environment] = str(int(value))


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
        value = os.environ.get(environment)
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
# ^^^ THOG
