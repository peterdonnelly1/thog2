# vvv THOG
"""Final human-facing PLASTIC progress-row postfix cleanup."""

from __future__ import annotations

import re
from typing import Any, Dict

from . import stage6_trainer as _stage6


_ORIGINAL_FORMAT_PROGRESS_LINE = _stage6.format_progress_line
_POSTFIX_ANNOTATION = re.compile(
    r"(?P<spacing>[ \t]*)(?P<colour>\x1b\[[0-9;]*m)?"
    r"(?P<text><<< (?:update brake on|warmup braked? enabled))"
    r"(?P<reset>\x1b\[[0-9;]*m)?"
)
_NEUTRAL_LRA = re.compile(r"(?P<prefix>L/R/A=\[[^\]]+\]/\d+)=>-(?=$|[ \t])")


def _finalize_plastic_postfixes(line: str) -> str:
    rendered = _NEUTRAL_LRA.sub(r"\g<prefix>=>stet", line)
    annotations = []

    def collect_annotation(match: re.Match[str]) -> str:
        text = match.group("text").replace("warmup braked enabled", "warmup brake enabled")
        annotations.append(
            f"{match.group('colour') or ''}{text}{match.group('reset') or ''}"
        )
        return ""

    rendered = _POSTFIX_ANNOTATION.sub(collect_annotation, rendered).rstrip(" \t")
    if annotations:
        rendered = f"{rendered}  {'  '.join(annotations)}"
    return rendered


def _format_progress_line_with_final_postfixes(
    run_id: str,
    event: str,
    payload: Dict[str, Any],
) -> str:
    return _finalize_plastic_postfixes(
        _ORIGINAL_FORMAT_PROGRESS_LINE(run_id, event, payload)
    )


_stage6.format_progress_line = _format_progress_line_with_final_postfixes


__all__ = ["_finalize_plastic_postfixes"]
# ^^^ THOG
