# vvv THOG
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
