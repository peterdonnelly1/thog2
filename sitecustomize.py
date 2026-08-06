# vvv THOG
"""Process-local CLI aliasing for underscore long options.

Python imports sitecustomize before normal module execution when the repository
root is on sys.path. This keeps the canonical argparse definitions untouched
while accepting user-facing underscore spellings such as --select_depth.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence


_PROBE_INTERVAL_ENVIRONMENT_KEY = "THOG2_PLASTIC_LAYER_COUNT_PROBE_WINDOW_SIZE"
_PROBE_INTERVAL_OPTION = "--plastic-layer-count-probe-interval"
_ORIGINAL_PARSE_ARGS = argparse.ArgumentParser.parse_args
_ORIGINAL_PARSE_KNOWN_ARGS = argparse.ArgumentParser.parse_known_args


def _normalise_long_option(argument: str) -> str:
    if argument == "--" or not argument.startswith("--"):
        return argument
    option, separator, value = argument.partition("=")
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


def _normalise_arguments(arguments: Optional[Sequence[str]]) -> List[str]:
    source = sys.argv[1:] if arguments is None else list(arguments)
    return _extract_probe_interval(source)


def _parse_args_with_underscore_aliases(self: argparse.ArgumentParser, args: Optional[Sequence[str]] = None, namespace=None):
    return _ORIGINAL_PARSE_ARGS(self, _normalise_arguments(args), namespace)


def _parse_known_args_with_underscore_aliases(self: argparse.ArgumentParser, args: Optional[Sequence[str]] = None, namespace=None):
    return _ORIGINAL_PARSE_KNOWN_ARGS(self, _normalise_arguments(args), namespace)


argparse.ArgumentParser.parse_args = _parse_args_with_underscore_aliases
argparse.ArgumentParser.parse_known_args = _parse_known_args_with_underscore_aliases
# ^^^ THOG
