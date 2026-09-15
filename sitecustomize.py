# vvv THOG
"""Process-local CLI aliasing for underscore long options.

Python imports sitecustomize before normal module execution when the repository
root is on sys.path. This keeps the canonical argparse definitions untouched
while accepting user-facing underscore spellings such as --select_depth.
"""

from __future__ import annotations

import argparse
import builtins
import os
import sys
from typing import List, Optional, Sequence


_PROBE_INTERVAL_ENVIRONMENT_KEY = "THOG2_PLASTIC_LAYER_COUNT_PROBE__PROBE_EVERY_N_STEPS"
_PROBE_INTERVAL_OPTION = "--plastic-layer-count-probe-interval"
_ORIGINAL_PARSE_ARGS = argparse.ArgumentParser.parse_args
_ORIGINAL_PARSE_KNOWN_ARGS = argparse.ArgumentParser.parse_known_args
_ORIGINAL_FORMAT_HELP = argparse.ArgumentParser.format_help

# vvv THOG full-step timing is a public CLI feature; the former environment variable is deliberately no longer a user control
_FULL_STEP_TIMING_OPTION = "--premat_instra__full_step_timing_capture_and_chart"
_FULL_STEP_TIMING_CAPTURE_OPTION = "--premat_instra__full_step_timing_capture_and_chart_capture"
_FULL_STEP_TIMING_ENABLED_ENVIRONMENT_KEY = "THOG2_INTERNAL_PREMAT_INSTRA_FULL_STEP_TIMING_CAPTURE_AND_CHART"
_FULL_STEP_TIMING_CAPTURE_ENVIRONMENT_KEY = "THOG2_INTERNAL_PREMAT_INSTRA_FULL_STEP_TIMING_CAPTURE_AND_CHART_CAPTURE"
_LEGACY_FULL_STEP_TIMING_ENVIRONMENT_KEY = "THOG2_PROCESSING_UPDATE_TIMING_UPDATE"
os.environ.pop(_LEGACY_FULL_STEP_TIMING_ENVIRONMENT_KEY, None)
os.environ.pop(_FULL_STEP_TIMING_ENABLED_ENVIRONMENT_KEY, None)
os.environ.pop(_FULL_STEP_TIMING_CAPTURE_ENVIRONMENT_KEY, None)
# ^^^ THOG


def _normalise_long_option(argument: str) -> str:
    if argument == "--" or not argument.startswith("--"):
        return argument
    option, separator, value = argument.partition("=")
    # These public namespaces deliberately use double underscores.  The shell
    # wrapper validates their exact spelling and preserves it, so the Python
    # compatibility normaliser must not turn ``__`` into ``--`` before the
    # canonical argparse actions and late option overlays see the argument.
    if option.startswith(
        (
            "--plastic__",
            "--no-plastic__",
            "--chaos_bump__",
            "--no-chaos_bump__",
            "--instrumentation__",
            "--premat_",
        )
    ):
        return option + (separator + value if separator else "")
    return option.replace("_", "-") + (separator + value if separator else "")


def _extract_probe_interval(arguments: Sequence[str]) -> List[str]:
    remaining: List[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        normalised = _normalise_long_option(argument)
        option, separator, value = normalised.partition("=")
        if option == _PROBE_INTERVAL_OPTION:
            if separator:
                resolved = value
                index += 1
            else:
                if index + 1 >= len(arguments):
                    raise SystemExit(f"{argument} requires a value")
                resolved = arguments[index + 1]
                index += 2
            try:
                interval = int(str(resolved).strip())
            except ValueError as error:
                raise SystemExit(f"{argument} requires a positive integer; got: {resolved}") from error
            if interval < 1:
                raise SystemExit(f"{argument} requires a positive integer; got: {resolved}")
            os.environ[_PROBE_INTERVAL_ENVIRONMENT_KEY] = str(interval)
            continue
        remaining.append(normalised)
        index += 1
    return remaining


# vvv THOG consume the exact full-step timing CLI before core argparse; capture defaults to the trainer's final update when omitted

def _extract_full_step_timing(arguments: Sequence[str]) -> List[str]:
    remaining: List[str] = []
    enabled: Optional[str] = None
    capture_update: Optional[int] = None
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        option, separator, value = argument.partition("=")
        if option == _FULL_STEP_TIMING_OPTION:
            if separator:
                resolved = value
                index += 1
            else:
                if index + 1 >= len(arguments):
                    raise SystemExit(f"{argument} requires enable or disable")
                resolved = arguments[index + 1]
                index += 2
            resolved = str(resolved).strip().lower()
            if resolved not in {"enable", "disable"}:
                raise SystemExit(f"{argument} requires enable or disable; got: {resolved}")
            enabled = resolved
            continue
        if option == _FULL_STEP_TIMING_CAPTURE_OPTION:
            if separator:
                raise SystemExit(f"{option} syntax is: {option} step <integer>")
            if index + 2 >= len(arguments) or str(arguments[index + 1]).strip().lower() != "step":
                raise SystemExit(f"{option} syntax is: {option} step <integer>")
            resolved = arguments[index + 2]
            try:
                capture_update = int(str(resolved).strip())
            except ValueError as error:
                raise SystemExit(f"{option} requires a positive integer step; got: {resolved}") from error
            if capture_update < 1:
                raise SystemExit(f"{option} requires a positive integer step; got: {capture_update}")
            index += 3
            continue
        remaining.append(argument)
        index += 1
    if capture_update is not None and enabled != "enable":
        raise SystemExit(f"{_FULL_STEP_TIMING_CAPTURE_OPTION} requires {_FULL_STEP_TIMING_OPTION} enable")
    if enabled is not None:
        os.environ[_FULL_STEP_TIMING_ENABLED_ENVIRONMENT_KEY] = enabled
    if capture_update is not None:
        os.environ[_FULL_STEP_TIMING_CAPTURE_ENVIRONMENT_KEY] = str(capture_update)
    return remaining
# ^^^ THOG


def _normalise_arguments(arguments: Optional[Sequence[str]]) -> List[str]:
    source = sys.argv[1:] if arguments is None else list(arguments)
    # vvv THOG consume full-step timing controls after established underscore/PLASTIC handling
    # return _extract_probe_interval(source)
    return _extract_full_step_timing(_extract_probe_interval(source))
    # ^^^ THOG


def _parse_args_with_underscore_aliases(self: argparse.ArgumentParser, args: Optional[Sequence[str]] = None, namespace=None):
    return _ORIGINAL_PARSE_ARGS(self, _normalise_arguments(args), namespace)


def _parse_known_args_with_underscore_aliases(self: argparse.ArgumentParser, args: Optional[Sequence[str]] = None, namespace=None):
    return _ORIGINAL_PARSE_KNOWN_ARGS(self, _normalise_arguments(args), namespace)


argparse.ArgumentParser.parse_args = _parse_args_with_underscore_aliases
argparse.ArgumentParser.parse_known_args = _parse_known_args_with_underscore_aliases

# vvv THOG advertise the consumed full-step controls on the canonical OWT argparse help surface
def _format_help_with_full_step_timing(self: argparse.ArgumentParser) -> str:
    text = _ORIGINAL_FORMAT_HELP(self)
    if str(getattr(self, "description", "")) != "Train or resume one canonical THOG2 OpenWebText run":
        return text
    return (
        text.rstrip()
        + "\n\nPremat Instra full-step timing:\n"
        + "  --premat_instra__full_step_timing_capture_and_chart enable|disable\n"
        + "      lightweight full-update timing/chart capture; default disable\n"
        + "  --premat_instra__full_step_timing_capture_and_chart_capture step <integer>\n"
        + "      optional exact optimizer update; default final step\n"
    )


argparse.ArgumentParser.format_help = _format_help_with_full_step_timing
# ^^^ THOG

# vvv THOG install the full-step timing overlay only after the established processing patch has completely initialized
_ORIGINAL_IMPORT_FOR_FULL_STEP_TIMING = builtins.__import__
_FULL_STEP_TIMING_PATCH_INSTALLED = False


def _import_with_full_step_timing_patch(name, globals=None, locals=None, fromlist=(), level=0):
    global _FULL_STEP_TIMING_PATCH_INSTALLED
    module = _ORIGINAL_IMPORT_FOR_FULL_STEP_TIMING(name, globals, locals, fromlist, level)
    processing_patch = sys.modules.get("sheet.processing_update_timing_patch")
    if (
        not _FULL_STEP_TIMING_PATCH_INSTALLED
        and processing_patch is not None
        and getattr(processing_patch, "_ORIGINAL_ATTACH_TELEMETRY", None) is not None
    ):
        _FULL_STEP_TIMING_PATCH_INSTALLED = True
        builtins.__import__ = _ORIGINAL_IMPORT_FOR_FULL_STEP_TIMING
        _ORIGINAL_IMPORT_FOR_FULL_STEP_TIMING(
            "sheet.processing_full_step_timing_patch",
            globals,
            locals,
            (),
            0,
        )
    return module


builtins.__import__ = _import_with_full_step_timing_patch
# ^^^ THOG
# ^^^ THOG
