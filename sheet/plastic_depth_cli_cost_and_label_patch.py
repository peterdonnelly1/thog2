# vvv THOG
"""Set the PLASTIC wall-time cost default and compact long probe offset labels."""

from __future__ import annotations

import argparse
import inspect
import re
from typing import Any, Callable, Tuple

from . import stage6_trainer as _stage6
from .run_config import OwtRunConfig
from .training_config import TrainingConfig


_DEFAULT_LAYER_COUNT_COST_WEIGHT = 0.02
_OFFSET_VECTOR_LABEL = re.compile(
    r"(?P<field>probe_losses|loss_gain|score_z) "
    r"\[(?P<label>[^\]]+)\] ="
)


# vvv THOG keep direct dataclass construction consistent with the public CLI default
def _replace_callable_default(
    function: Callable[..., Any],
    parameter_name: str,
    value: Any,
) -> None:
    parameters = tuple(inspect.signature(function).parameters.values())
    positional = tuple(
        parameter
        for parameter in parameters
        if parameter.kind
        in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
    )
    defaults = list(function.__defaults__ or ())
    default_start = len(positional) - len(defaults)
    parameter_index = next(
        index
        for index, parameter in enumerate(positional)
        if parameter.name == parameter_name
    )
    default_index = parameter_index - default_start
    if default_index < 0:
        raise RuntimeError(
            f"{parameter_name} is not represented in callable positional defaults"
        )
    defaults[default_index] = value
    function.__defaults__ = tuple(defaults)


def _replace_dataclass_default(cls: type[Any], field_name: str, value: Any) -> None:
    cls.__dataclass_fields__[field_name].default = value
    _replace_callable_default(cls.__init__, field_name, value)


_replace_dataclass_default(
    TrainingConfig,
    "plastic__layer_count_cost_weight",
    _DEFAULT_LAYER_COUNT_COST_WEIGHT,
)
_replace_dataclass_default(
    OwtRunConfig,
    "plastic__layer_count_cost_weight",
    _DEFAULT_LAYER_COUNT_COST_WEIGHT,
)
# ^^^ THOG


# vvv THOG replace the argparse default only for the canonical PLASTIC cost-weight option
_ORIGINAL_ADD_ARGUMENT = argparse.ArgumentParser.add_argument


def _add_argument_with_plastic_cost_default(
    parser: argparse.ArgumentParser,
    *name_or_flags: str,
    **kwargs: Any,
):
    if "--plastic__layer_count_cost_weight" in name_or_flags:
        kwargs["default"] = _DEFAULT_LAYER_COUNT_COST_WEIGHT
    return _ORIGINAL_ADD_ARGUMENT(parser, *name_or_flags, **kwargs)


argparse.ArgumentParser.add_argument = _add_argument_with_plastic_cost_default
# ^^^ THOG


# vvv THOG compact long contiguous probe descriptions without changing vector values or radius-one labels
def _compact_offset_vector_labels(line: str) -> str:
    def replace_label(match: re.Match[str]) -> str:
        items = tuple(
            item.strip()
            for item in match.group("label").split(",")
            if item.strip()
        )
        if len(items) <= 3:
            return match.group(0)
        return (
            f"{match.group('field')} "
            f"[{items[0]} ... {items[-1]}] ="
        )

    return _OFFSET_VECTOR_LABEL.sub(replace_label, line)


_ORIGINAL_FORMAT_PROGRESS_LINE = _stage6.format_progress_line


def _format_progress_line_with_compact_offset_labels(
    run_id: str,
    event: str,
    payload: dict[str, Any],
) -> str:
    return _compact_offset_vector_labels(
        _ORIGINAL_FORMAT_PROGRESS_LINE(run_id, event, payload)
    )


_stage6.format_progress_line = _format_progress_line_with_compact_offset_labels
# ^^^ THOG


__all__: Tuple[str, ...] = (
    "_DEFAULT_LAYER_COUNT_COST_WEIGHT",
    "_compact_offset_vector_labels",
)
# ^^^ THOG
