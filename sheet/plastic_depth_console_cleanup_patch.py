# vvv THOG
"""Final PLASTIC DEPTH console label and colour cleanup."""

from __future__ import annotations

import re
from typing import Any, Optional, Tuple

from . import stage6_trainer as _stage6


_GREEN = "\033[1;38;2;0;255;0m"
_RED = "\033[1;31m"
_RESET = "\033[0m"

_ORIGINAL_FORMAT_PROGRESS_LINE = _stage6.format_progress_line


def _colour_probe_loss(value: str, relation: Optional[float]) -> str:
    if relation is None:
        return value
    if relation > 0.0:
        return f"{_GREEN}{value}{_RESET}"
    if relation < 0.0:
        return f"{_RED}{value}{_RESET}"
    return value


def _colour_probe_losses(match: re.Match[str]) -> str:
    label = match.group("label")
    body = match.group("body")
    values = [item for item in body.split(",")]
    if len(values) != 3:
        return f"probe_losses [{label}] = [{body}]"
    numeric_values = []
    for item in values:
        text = item.strip()
        if text == "-":
            numeric_values.append(None)
            continue
        try:
            numeric_values.append(float(text))
        except ValueError:
            numeric_values.append(None)
    current = numeric_values[1]
    if current is None:
        return f"probe_losses [{label}] = [{body}]"
    relations: Tuple[Optional[float], Optional[float], Optional[float]] = (
        None if numeric_values[0] is None else current - numeric_values[0],
        None,
        None if numeric_values[2] is None else current - numeric_values[2],
    )
    coloured = [
        _colour_probe_loss(value, relation)
        for value, relation in zip(values, relations)
    ]
    return f"probe_losses [{label}] = [{','.join(coloured)}]"


def _strip_loss_gain(line: str) -> str:
    return re.sub(
        r"\s+loss_gain \[[^\]]+\] = \[[^\]]+\]",
        "",
        line,
    )


def _format_progress_line_with_plastic_console_cleanup(run_id: str, event: str, payload: dict[str, Any]) -> str:
    line = _ORIGINAL_FORMAT_PROGRESS_LINE(run_id, event, payload)
    line = _strip_loss_gain(line)
    line = line.replace("current_layer_count =", "layers =")
    line = line.replace("\tlayer indices =", "\tsample_pos_100 =")
    line = line.replace("  layer indices =", "  sample_pos_100 =")
    line = re.sub(
        r"probe_losses \[(?P<label>[^\]]+)\] = \[(?P<body>[^\]]+)\]",
        _colour_probe_losses,
        line,
    )
    return line


_stage6.format_progress_line = _format_progress_line_with_plastic_console_cleanup
# ^^^ THOG
