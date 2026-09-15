# vvv THOG register full-step timing as real THOG argparse controls without adding it to model/checkpoint identity
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Optional, Sequence


_ENABLED_DESTINATION = "premat_instra__full_step_timing_capture_and_chart"
_CAPTURE_DESTINATION = "premat_instra__full_step_timing_capture_and_chart_capture"
_ENABLED_OPTION = f"--{_ENABLED_DESTINATION}"
_CAPTURE_OPTION = f"--{_CAPTURE_DESTINATION}"
_CLI_INSTALLED_ATTRIBUTE = "_thog_full_step_timing_arguments_installed"

_FULL_STEP_TIMING_ENABLED = "disable"
_FULL_STEP_TIMING_CAPTURE_UPDATE: Optional[int] = None


class _FullStepCaptureAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) != 2 or str(values[0]).strip().lower() != "step":
            parser.error(f"{option_string or _CAPTURE_OPTION} syntax is: {_CAPTURE_OPTION} step <integer>")
        try:
            update = int(str(values[1]).strip())
        except ValueError:
            parser.error(f"{option_string or _CAPTURE_OPTION} requires a positive integer step; got: {values[1]}")
        if update < 1:
            parser.error(f"{option_string or _CAPTURE_OPTION} requires a positive integer step; got: {update}")
        setattr(namespace, self.dest, update)


def _ensure_cli_arguments(parser: argparse.ArgumentParser) -> None:
    if bool(getattr(parser, _CLI_INSTALLED_ATTRIBUTE, False)):
        return
    if _ENABLED_OPTION not in parser._option_string_actions:
        parser.add_argument(
            _ENABLED_OPTION,
            dest=_ENABLED_DESTINATION,
            choices=("enable", "disable"),
            default="disable",
            help="lightweight full-update timing/chart capture; default disable",
        )
    if _CAPTURE_OPTION not in parser._option_string_actions:
        parser.add_argument(
            _CAPTURE_OPTION,
            dest=_CAPTURE_DESTINATION,
            action=_FullStepCaptureAction,
            nargs=2,
            metavar=("step", "UPDATE"),
            default=None,
            help="optional exact optimizer update; syntax: step <integer>; default final step",
        )
    setattr(parser, _CLI_INSTALLED_ATTRIBUTE, True)


def _raw_explicit_state(arguments: Sequence[str]) -> tuple[Optional[str], Optional[int]]:
    enabled: Optional[str] = None
    capture_update: Optional[int] = None
    index = 0
    while index < len(arguments):
        argument = str(arguments[index])
        if argument == _ENABLED_OPTION and index + 1 < len(arguments):
            candidate = str(arguments[index + 1]).strip().lower()
            if candidate in {"enable", "disable"}:
                enabled = candidate
            index += 2
            continue
        if argument.startswith(_ENABLED_OPTION + "="):
            candidate = argument.split("=", 1)[1].strip().lower()
            if candidate in {"enable", "disable"}:
                enabled = candidate
            index += 1
            continue
        if argument == _CAPTURE_OPTION and index + 2 < len(arguments):
            if str(arguments[index + 1]).strip().lower() == "step":
                try:
                    candidate_update = int(str(arguments[index + 2]).strip())
                except ValueError:
                    candidate_update = 0
                if candidate_update > 0:
                    capture_update = candidate_update
            index += 3
            continue
        index += 1
    return enabled, capture_update


_EXPLICIT_ENABLED, _EXPLICIT_CAPTURE_UPDATE = _raw_explicit_state(tuple(sys.argv[1:]))


def full_step_timing_enabled() -> bool:
    return _FULL_STEP_TIMING_ENABLED == "enable"


def full_step_timing_capture_update() -> Optional[int]:
    return _FULL_STEP_TIMING_CAPTURE_UPDATE


def _publish_cli_state(parser: argparse.ArgumentParser, namespace: argparse.Namespace) -> None:
    global _FULL_STEP_TIMING_ENABLED, _FULL_STEP_TIMING_CAPTURE_UPDATE
    parsed_enabled = str(getattr(namespace, _ENABLED_DESTINATION, "disable")).strip().lower()
    parsed_capture = getattr(namespace, _CAPTURE_DESTINATION, None)
    enabled = _EXPLICIT_ENABLED if _EXPLICIT_ENABLED is not None else parsed_enabled
    capture_update = _EXPLICIT_CAPTURE_UPDATE if _EXPLICIT_CAPTURE_UPDATE is not None else parsed_capture
    if capture_update is not None and enabled != "enable":
        parser.error(f"{_CAPTURE_OPTION} requires {_ENABLED_OPTION} enable")
    _FULL_STEP_TIMING_ENABLED = enabled
    _FULL_STEP_TIMING_CAPTURE_UPDATE = None if capture_update is None else int(capture_update)

    # processing_update_timing_patch is installed near the end of sheet package
    # initialisation.  Normal runner parsing occurs after that point, so the
    # full-step overlay can now safely bind itself to the already-imported runner.
    if "sheet.processing_update_timing_patch" in sys.modules:
        full_step_patch = importlib.import_module("sheet.processing_full_step_timing_patch")
        installer = getattr(full_step_patch, "_install_runner_attach_binding", None)
        if callable(installer):
            installer()


_ORIGINAL_PARSE_KNOWN_ARGS = argparse.ArgumentParser.parse_known_args


def _parse_known_args_with_full_step_timing(
    self: argparse.ArgumentParser,
    args: Optional[Sequence[str]] = None,
    namespace: Optional[argparse.Namespace] = None,
):
    _ensure_cli_arguments(self)
    parsed, remaining = _ORIGINAL_PARSE_KNOWN_ARGS(self, args=args, namespace=namespace)
    _publish_cli_state(self, parsed)
    return parsed, remaining


argparse.ArgumentParser.parse_known_args = _parse_known_args_with_full_step_timing


# vvv THOG the dashboard needs the viewer overlay even though it never parses a training command line
_ORIGINAL_PATH_READ_BYTES = Path.read_bytes


def _dashboard_asset_read_bytes(path: Path) -> bytes:
    payload = _ORIGINAL_PATH_READ_BYTES(path)
    if (
        Path(sys.argv[0]).name == "run_thog2_local_dashboard.py"
        and path.name == "dashboard_processing.js"
        and path.parent.name == "local_dashboard_assets"
    ):
        patch = path.with_name("dashboard_processing_full_step_patch.js")
        if patch.is_file():
            payload += b"\n" + _ORIGINAL_PATH_READ_BYTES(patch)
    return payload


if Path(sys.argv[0]).name == "run_thog2_local_dashboard.py":
    Path.read_bytes = _dashboard_asset_read_bytes
# ^^^ THOG


__all__ = [
    "full_step_timing_capture_update",
    "full_step_timing_enabled",
]
# ^^^ THOG
