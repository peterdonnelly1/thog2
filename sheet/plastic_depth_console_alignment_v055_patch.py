# vvv THOG
"""Final v0.55 PLASTIC operator-row geometry after Sen/Kendall console formatting."""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any

from . import stage6_trainer as _stage6


_ORIGINAL_FORMAT_PROGRESS_LINE = _stage6.format_progress_line
_PROGRESS_PREFIX = re.compile(
    r"^(?P<style>(?:\x1b\[[0-9;]*m)*)"
    r"(?P<kind>[TV])\s+"
    r"(?P<step>\d+)\s+"
    r"(?P<timestamp>(?:\d{6}[:-])?\d{4})\s+"
    r"(?P<seconds>\d+)s\s+"
    r"(?P<hours>[+-]?(?:\d+(?:\.\d*)?|\.\d+))h?"
)
_STEP_DELTA = re.compile(r"Δ=\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))s")
_TOKENS_PER_SECOND = re.compile(r"tok/s=\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))")
_CONSUMED_TOKENS = re.compile(r"toks=\s*(?P<value>[0-9,]+)")
_TRAINING_LOSS = re.compile(r"(?<!validation )(?<!training )loss=\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))")
_VALIDATION_LOSS = re.compile(r"validation loss=\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))")
_LEARNING_RATE = re.compile(r"lr=\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?\d+)")
_GRADIENT_NORM = re.compile(r"g/n=\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))")
_LAYER_COUNT = re.compile(r"layers\s+(?P<value>\d+)")
_LOSS_DELTA = re.compile(
    r"Δ=\s*(?P<value>n/a|[+-]?(?:\d+(?:\.\d*)?|\.\d+))"
    r"(?P<reset>\x1b\[[0-9;]*m)?(?=\s+lr=)"
)
_PROBE_BLOCK = re.compile(r"(?P<probe>P\d+\s+probe_Δloss\b)")
_PROBE_SHIFT_COLUMNS = 27


def _progress_timestamp_hhmm() -> str:
    return datetime.now().strftime("%H%M")


def _progress_elapsed_seconds_and_decimal_hours(value: Any, completed_updates: Any) -> str:
    del completed_updates
    elapsed_seconds = max(0, int(round(float(str(value).strip()))))
    return f"{elapsed_seconds:7d}s {elapsed_seconds / 3600.0:8.3f}h"


def _format_progress_prefix(line: str) -> str:
    match = _PROGRESS_PREFIX.match(line)
    if match is None:
        return line
    step = int(match.group("step"))
    elapsed_seconds = int(match.group("seconds"))
    elapsed_hours = float(match.group("hours"))
    timestamp = match.group("timestamp")[-4:]
    prefix = (
        f"{match.group('style')}{match.group('kind')} {step:6d}  {timestamp}  "
        f"{elapsed_seconds:7d}s {elapsed_hours:8.3f}h"
    )
    return prefix + "  " + line[match.end() :].lstrip(" \t")


def _format_loss_delta(match: re.Match[str]) -> str:
    raw = match.group("value")
    reset = match.group("reset") or ""
    if raw == "n/a":
        rendered = f"{'n/a':>8}"
    else:
        rendered = f"{float(raw):+8.3f}"
    return f"Δ={rendered}{reset}"


def _restore_fixed_numeric_fields(line: str) -> str:
    rendered = _STEP_DELTA.sub(
        lambda match: f"Δ={float(match.group('value')):5.1f}s",
        line,
        count=1,
    )
    rendered = _TOKENS_PER_SECOND.sub(
        lambda match: f"tok/s={float(match.group('value')):6.0f}",
        rendered,
        count=1,
    )
    rendered = _CONSUMED_TOKENS.sub(
        lambda match: f"toks={int(match.group('value').replace(',', '')):11,d}",
        rendered,
        count=1,
    )
    rendered = _TRAINING_LOSS.sub(
        lambda match: f"loss={float(match.group('value')):8.4f}",
        rendered,
        count=1,
    )
    rendered = _VALIDATION_LOSS.sub(
        lambda match: f"validation loss={float(match.group('value')):8.4f}",
        rendered,
        count=1,
    )
    rendered = _LOSS_DELTA.sub(_format_loss_delta, rendered, count=1)
    rendered = _LEARNING_RATE.sub(
        lambda match: f"lr={float(match.group('value')):10.3e}",
        rendered,
        count=1,
    )
    rendered = _GRADIENT_NORM.sub(
        lambda match: f"g/n={float(match.group('value')):8.3f}",
        rendered,
        count=1,
    )
    rendered = _LAYER_COUNT.sub(
        lambda match: f"layers {int(match.group('value')):3d}",
        rendered,
        count=1,
    )
    return rendered


def _shift_probe_block_right(line: str) -> str:
    match = _PROBE_BLOCK.search(line)
    if match is None:
        return line
    return line[: match.start()] + (" " * _PROBE_SHIFT_COLUMNS) + line[match.start() :]


def _format_progress_line_with_final_v055_alignment(
    run_id: str,
    event: str,
    payload: dict[str, Any],
) -> str:
    line = _ORIGINAL_FORMAT_PROGRESS_LINE(run_id, event, payload)
    if event not in {"optimizer_progress", "evaluation_completed"}:
        return line
    line = _format_progress_prefix(line)
    line = _restore_fixed_numeric_fields(line)
    line = _shift_probe_block_right(line)
    return line


_stage6._progress_timestamp = _progress_timestamp_hhmm
_stage6._progress_elapsed = _progress_elapsed_seconds_and_decimal_hours
_stage6.format_progress_line = _format_progress_line_with_final_v055_alignment


__all__ = [
    "_format_progress_line_with_final_v055_alignment",
    "_format_progress_prefix",
    "_progress_elapsed_seconds_and_decimal_hours",
    "_progress_timestamp_hhmm",
    "_restore_fixed_numeric_fields",
    "_shift_probe_block_right",
]
# ^^^ THOG
