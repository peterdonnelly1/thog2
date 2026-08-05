# vvv THOG
"""Final PLASTIC DEPTH console label, colour, and elapsed-time cleanup."""

from __future__ import annotations

import math
import re
from typing import Any, Optional, Tuple

from . import stage6_trainer as _stage6


_GREEN = "\033[1;38;2;0;255;0m"
_RESET = "\033[0m"

_ORIGINAL_FORMAT_PROGRESS_LINE = _stage6.format_progress_line


# vvv THOG show copyable HH:MM:SS plus raw seconds in one fixed-width elapsed field
def _progress_elapsed_with_raw_seconds(value: Any, completed_updates: Any) -> str:
    elapsed_seconds = max(0, int(round(float(str(value).strip()))))
    hours, remainder_seconds = divmod(elapsed_seconds, 60 * 60)
    minutes, seconds = divmod(remainder_seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d} {elapsed_seconds:7d}s"


_stage6._progress_elapsed = _progress_elapsed_with_raw_seconds
# ^^^ THOG


def _colour_probe_loss(value: str, relation: Optional[float]) -> str:
    if relation is None:
        return value
    if relation > 0.0:
        return f"{_GREEN}{value}{_RESET}"
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
        None if numeric_values[0] is None else numeric_values[0] - current,
        None,
        None if numeric_values[2] is None else numeric_values[2] - current,
    )
    coloured = [
        _colour_probe_loss(value, relation)
        for value, relation in zip(values, relations)
    ]
    return f"probe_losses [{label}] = [{','.join(coloured)}]"


def _colour_positive_score(value: str) -> str:
    text = value.strip()
    if text in {"", "-"}:
        return value
    suffix = "i" if text.endswith("i") else ""
    numeric_text = text[:-1] if suffix else text
    try:
        numeric = float(numeric_text)
    except ValueError:
        return value
    if math.isfinite(numeric) and numeric > 0.0:
        return f"{_GREEN}{value}{_RESET}"
    return value


def _colour_score_z(match: re.Match[str]) -> str:
    label = match.group("label")
    body = match.group("body")
    values = [item for item in body.split(",")]
    coloured = [_colour_positive_score(value) for value in values]
    return f"score_z [{label}] = [{','.join(coloured)}]"


def _strip_loss_gain(line: str) -> str:
    return re.sub(
        r"\s+loss_gain \[[^\]]+\] = \[[^\]]+\]",
        "",
        line,
    )


def _strip_sampled_values(line: str) -> str:
    return re.sub(
        r"\s+sampled_values = \[[^\]]*\]",
        "",
        line,
    )


def _format_progress_line_with_plastic_console_cleanup(run_id: str, event: str, payload: dict[str, Any]) -> str:
    line = _ORIGINAL_FORMAT_PROGRESS_LINE(run_id, event, payload)
    line = _strip_loss_gain(line)
    line = _strip_sampled_values(line)
    line = line.replace("learning rate=", "lr=")
    line = line.replace("current_layer_count =", "layers =")
    line = line.replace("\tlayer indices =", "\tsampled =")
    line = line.replace("  layer indices =", "  sampled =")
    line = line.replace("\tsample_pos_100 =", "\tsampled =")
    line = line.replace("  sample_pos_100 =", "  sampled =")
    line = line.replace("\tsample_layer =", "\tsampled =")
    line = line.replace("  sample_layer =", "  sampled =")
    line = re.sub(
        r"probe_losses \[(?P<label>[^\]]+)\] = \[(?P<body>[^\]]+)\]",
        _colour_probe_losses,
        line,
    )
    line = re.sub(
        r"score_z \[(?P<label>[^\]]+)\] = \[(?P<body>[^\]]+)\]",
        _colour_score_z,
        line,
    )
    return line.rstrip()


_stage6.format_progress_line = _format_progress_line_with_plastic_console_cleanup
# ^^^ THOG
