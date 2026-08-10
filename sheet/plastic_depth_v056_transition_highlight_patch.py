# vvv THOG
"""Highlight the authoritative new PLASTIC DEPTH layer count exactly once on its transition row."""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional

import constants as _constants

from . import stage6_trainer as _stage6


_LAYER_FIELD = re.compile(r"(?P<label>\blayers\s+)(?P<count>\d+)\b")
_TRANSITION_PAYLOAD_KEY = "plastic_layer_count_transition_highlight"
_ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD = _stage6.Stage6Trainer._prepare_console_progress_payload
_ORIGINAL_FORMAT_PROGRESS_LINE = _stage6.format_progress_line


def _console_int(value: Any) -> int:
    return int(str(value).strip().replace(",", ""))


def _latest_committed_count_transition(trainer: Any) -> Optional[Mapping[str, Any]]:
    for event in reversed(getattr(trainer, "events", ())):
        if getattr(event, "name", None) != "plastic_depth_count_decision":
            continue
        payload = getattr(event, "payload", None)
        if not isinstance(payload, Mapping):
            continue
        try:
            previous = int(payload["previous_active_layers"])
            selected = int(payload["selected_active_layers"])
        except (KeyError, TypeError, ValueError):
            continue
        if previous != selected:
            return payload
    return None


def _prepare_console_progress_payload_with_transition_highlight(
    self: Any,
    event: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    values = _ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD(self, event, payload)
    values.pop(_TRANSITION_PAYLOAD_KEY, None)
    if event != "optimizer_progress":
        return values
    try:
        completed_updates = _console_int(
            values.get("completed_updates", payload.get("completed_updates"))
        )
    except (TypeError, ValueError):
        return values
    transition = _latest_committed_count_transition(self)
    if transition is None:
        return values
    try:
        selected = int(transition["selected_active_layers"])
        transition_update = int(transition.get("last_count_change_update", -1))
    except (KeyError, TypeError, ValueError):
        return values
    if transition_update == completed_updates:
        values[_TRANSITION_PAYLOAD_KEY] = selected
    return values


def _format_progress_line_with_transition_highlight(
    run_id: str,
    event: str,
    payload: Dict[str, Any],
) -> str:
    local = dict(payload)
    highlighted_count = local.pop(_TRANSITION_PAYLOAD_KEY, None)
    line = _ORIGINAL_FORMAT_PROGRESS_LINE(run_id, event, local)
    if event != "optimizer_progress" or highlighted_count is None:
        return line
    target = int(highlighted_count)

    def replace(match: re.Match[str]) -> str:
        if int(match.group("count")) != target:
            return match.group(0)
        return (
            f"{match.group('label')}"
            f"{_constants.BOLD}{_constants.YELLOW}{match.group('count')}{_constants.R}"
        )

    return _LAYER_FIELD.sub(replace, line, count=1)


_stage6.Stage6Trainer._prepare_console_progress_payload = (
    _prepare_console_progress_payload_with_transition_highlight
)
_stage6.format_progress_line = _format_progress_line_with_transition_highlight


__all__ = [
    "_format_progress_line_with_transition_highlight",
    "_latest_committed_count_transition",
    "_prepare_console_progress_payload_with_transition_highlight",
]
# ^^^ THOG
