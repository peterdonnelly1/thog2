# vvv THOG
"""Accept underscore spellings for long argparse options without removing legacy hyphen spellings."""

from __future__ import annotations

import argparse
from typing import Any


_ORIGINAL_ADD_ARGUMENT = argparse.ArgumentParser.add_argument


def _underscore_long_alias(option: str) -> str | None:
    if not option.startswith("--"):
        return None
    name = option[2:]
    if "-" not in name:
        return None
    return "--" + name.replace("-", "_")


def _add_argument_with_underscore_aliases(self: argparse.ArgumentParser, *args: Any, **kwargs: Any):
    expanded = []
    seen = set()
    for argument in args:
        expanded.append(argument)
        if isinstance(argument, str):
            seen.add(argument)
            alias = _underscore_long_alias(argument)
            if alias is not None and alias not in seen and alias not in args:
                expanded.append(alias)
                seen.add(alias)
    return _ORIGINAL_ADD_ARGUMENT(self, *expanded, **kwargs)


if argparse.ArgumentParser.add_argument is not _add_argument_with_underscore_aliases:
    argparse.ArgumentParser.add_argument = _add_argument_with_underscore_aliases
# ^^^ THOG
