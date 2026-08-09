# vvv THOG
"""Final operator-facing PLASTIC probe/header visibility policy."""

from __future__ import annotations

import math
import re
import sys
from typing import Any, Dict, Mapping, Optional, Sequence

import constants as _constants

from . import plastic_depth_same_batch_all_probes_patch as _same_batch
from . import stage6_trainer as _stage6
from . import trainer_step as _trainer_step


_ORIGINAL_COMMIT_INLINE_UPDATE = _trainer_step.TrainerStepMixin._commit_plastic_depth_inline_update
_ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD = _stage6.Stage6Trainer._prepare_console_progress_payload
_ORIGINAL_FORMAT_PROGRESS_LINE = _stage6.format_progress_line
_ORIGINAL_STAGE6_INIT = _stage6.Stage6Trainer.__init__
_STARTUP_VISIBILITY_INSTALLED = False
_PROBE_COLUMN = 360
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
_P_SECTION = re.compile(r"P\s*(?P<number>\d+)\s+probe_Δloss")
_SAMPLED_FIELD = re.compile(r"sampled = \[[^\]]*\]")
_GRADIENT_NORM_FIELD = re.compile(r"g nrm=\s*(?P<value>[^\s]+)")
_SAME_BATCH_MARKER = re.compile(
    r"[ \t]+same_batch W\d+:\d+/\d+ B=[0-9A-Fa-f]+"
)
_DIRECTION_MARKERS = (
    f"{_constants.DOWN_ARROW}|{_constants.UP_ARROW}|? =",
    "⇩|⇧|? =",
    "↓|↑|? =",
)
_DIRECTION_OUTCOME = re.compile(
    rf"(?P<prefix>=>)(?:\x1b\[[0-9;]*m)*(?P<glyph>[{re.escape(_constants.DOWN_ARROW + _constants.UP_ARROW)}])(?:\x1b\[[0-9;]*m)*"
)
_SAMPLED_BY_RUN_ID: Dict[str, tuple[str, ...]] = {}


def _window_local_provenance(ordinal: int) -> tuple[int, ...]:
    resolved = int(ordinal)
    if resolved < 1:
        return ()
    return tuple(range(1, resolved + 1))


def _latest_same_batch_audit_for_update(trainer: Any, update_number: int) -> Optional[Mapping[str, Any]]:
    rows = getattr(trainer, "plastic_depth_count_audit", None)
    if not isinstance(rows, list):
        return None
    for row in reversed(rows):
        if not bool(row.get("same_batch_all_probes", False)):
            continue
        if int(row.get("update_number", -1)) != int(update_number):
            continue
        return row
    return None


def _commit_plastic_depth_inline_update_with_window_local_provenance(
    self: Any,
    context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    transition = _ORIGINAL_COMMIT_INLINE_UPDATE(self, context)
    if (
        context is None
        or not _same_batch._runtime_enabled()
        or not bool(context.get("plastic_same_batch_precomputed", False))
    ):
        return transition

    rows = getattr(self, "plastic_depth_count_audit", None)
    if not isinstance(rows, list) or not rows:
        return transition
    row = rows[-1]
    decision = context.get("decision")
    if decision is None or int(row.get("update_number", -1)) != int(decision.update_number):
        return transition

    ordinal = int(context["plastic_same_batch_window_ordinal"])
    global_provenance = tuple(int(value) for value in row.get("probe_window_provenance", ()))
    row["probe_global_sequence"] = int(context["plastic_probe_sequence"])
    row["probe_global_provenance"] = global_provenance
    row["probe_window_provenance"] = _window_local_provenance(ordinal)
    return transition


_trainer_step.TrainerStepMixin._commit_plastic_depth_inline_update = (
    _commit_plastic_depth_inline_update_with_window_local_provenance
)


def _prepare_console_progress_payload_with_same_batch_identity(
    self: Any,
    event: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    values = _ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD(self, event, payload)
    if event != "optimizer_progress":
        return values
    try:
        completed_updates = int(values.get("completed_updates", payload.get("completed_updates")))
    except (TypeError, ValueError):
        return values

    if _same_batch._runtime_enabled():
        row = _latest_same_batch_audit_for_update(self, completed_updates)
        if row is not None:
            ordinal = int(row["probe_window_ordinal"])
            values["plastic_probe_sequence"] = ordinal
            values["plastic_probe_provenance"] = _window_local_provenance(ordinal)
            values["plastic_same_batch_window_id"] = int(row["probe_window_id"])
            values["plastic_same_batch_window_ordinal"] = ordinal
            values["plastic_same_batch_window_size"] = int(row["probe_window_size"])
            values["plastic_same_batch_batch_digest"] = str(row["probe_batch_digest"])

    if "plastic_probe_losses" in values:
        required = max(
            1,
            int(
                getattr(
                    self.config,
                    "plastic__layer_count_probe__window_size_as_number_of_probes",
                    1,
                )
            ),
        )
        provenance = tuple(values.get("plastic_probe_provenance", ()))
        values["plastic_probe_decision_ready"] = len(provenance) >= required
    return values


_stage6.Stage6Trainer._prepare_console_progress_payload = (
    _prepare_console_progress_payload_with_same_batch_identity
)


def _format_sample_value(value: Any) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        return str(numeric)
    return f"{numeric:.1f}"


def _highlight_new_sample_changes(
    line: str,
    *,
    run_id: str,
    event: str,
    sample_points: Optional[Sequence[Any]],
) -> str:
    if event != "optimizer_progress" or sample_points is None:
        return line
    current = tuple(_format_sample_value(value) for value in sample_points)
    previous = _SAMPLED_BY_RUN_ID.get(run_id)
    _SAMPLED_BY_RUN_ID[run_id] = current
    match = _SAMPLED_FIELD.search(line)
    if match is None:
        return line
    rendered = []
    for index, value in enumerate(current):
        changed = previous is not None and index < len(previous) and value != previous[index]
        rendered.append(
            f"{_constants.YELLOW}{value}{_constants.R}"
            if changed
            else value
        )
    replacement = f"sampled = [{', '.join(rendered)}]"
    return line[: match.start()] + replacement + line[match.end() :]


def _hide_unqualified_decision_summary(line: str, *, decision_ready: Optional[bool]) -> str:
    if decision_ready is not False:
        return line
    starts = [line.find(marker) for marker in _DIRECTION_MARKERS]
    starts = [position for position in starts if position >= 0]
    if not starts:
        return line
    start = min(starts)
    preserved_tail_positions = [
        line.find("  same_batch ", start),
        line.find("  <<<", start),
    ]
    preserved_tail_positions = [position for position in preserved_tail_positions if position >= 0]
    end = min(preserved_tail_positions) if preserved_tail_positions else len(line)
    return f"{line[:start].rstrip()}{line[end:]}"


def _use_full_size_direction_glyphs(line: str) -> str:
    return (
        line.replace("⇩", _constants.DOWN_ARROW)
        .replace("↓", _constants.DOWN_ARROW)
        .replace("⇧", _constants.UP_ARROW)
        .replace("↑", _constants.UP_ARROW)
    )


def _highlight_direction_outcome(line: str) -> str:
    return _DIRECTION_OUTCOME.sub(
        lambda match: (
            f"{match.group('prefix')}{_constants.BOLD}{_constants.YELLOW}"
            f"{match.group('glyph')}{_constants.R}"
        ),
        line,
    )


def _fix_gradient_norm_width(line: str) -> str:
    return _GRADIENT_NORM_FIELD.sub(
        lambda match: f"g nrm={match.group('value'):>7}",
        line,
        count=1,
    )


def _visible_length(text: str) -> int:
    return len(_ANSI_ESCAPE.sub("", text))


def _truncate_visible(text: str, width: int) -> str:
    if _visible_length(text) <= width:
        return text
    rendered = []
    visible = 0
    position = 0
    while position < len(text) and visible < width:
        escape = _ANSI_ESCAPE.match(text, position)
        if escape is not None:
            rendered.append(escape.group(0))
            position = escape.end()
            continue
        rendered.append(text[position])
        visible += 1
        position += 1
    rendered.append(_constants.R)
    return "".join(rendered)


def _align_probe_section(line: str) -> str:
    match = _P_SECTION.search(line)
    if match is None:
        return line
    probe = line[match.start() :]
    probe = _P_SECTION.sub(
        lambda value: f"P{value.group('number')}  probe_Δloss",
        probe,
        count=1,
    )
    prefix = line[: match.start()].rstrip(" \t")
    prefix_width = _PROBE_COLUMN - 1
    prefix = _truncate_visible(prefix, prefix_width)
    visible = _visible_length(prefix)
    if visible < prefix_width:
        prefix = f"{prefix}{' ' * (prefix_width - visible)}"
    return f"{prefix}{probe}"


def _format_progress_line_with_same_batch_identity(
    run_id: str,
    event: str,
    payload: Dict[str, Any],
) -> str:
    local_payload = dict(payload)
    window_id = local_payload.pop("plastic_same_batch_window_id", None)
    ordinal = local_payload.pop("plastic_same_batch_window_ordinal", None)
    window_size = local_payload.pop("plastic_same_batch_window_size", None)
    batch_digest = local_payload.pop("plastic_same_batch_batch_digest", None)
    decision_ready = local_payload.pop("plastic_probe_decision_ready", None)
    sample_points = local_payload.get("depth_sample_points")
    line = _ORIGINAL_FORMAT_PROGRESS_LINE(run_id, event, local_payload)
    line = _highlight_new_sample_changes(
        line,
        run_id=run_id,
        event=event,
        sample_points=sample_points,
    )
    line = _hide_unqualified_decision_summary(line, decision_ready=decision_ready)
    line = _use_full_size_direction_glyphs(line)
    line = _highlight_direction_outcome(line)
    line = _fix_gradient_norm_width(line)
    if (
        int(getattr(_constants, "DEBUG", 0)) > 9
        and event == "optimizer_progress"
        and window_id is not None
        and ordinal is not None
        and window_size is not None
        and batch_digest is not None
        and "probe_Δloss" in line
    ):
        line = (
            f"{line}  same_batch W{int(window_id)}:{int(ordinal)}/{int(window_size)} "
            f"B={str(batch_digest)[:8]}"
        )
    else:
        line = _SAME_BATCH_MARKER.sub("", line)
    return _align_probe_section(line)


_stage6.format_progress_line = _format_progress_line_with_same_batch_identity


def _startup_runner_module(modules: Optional[Mapping[str, Any]] = None) -> Optional[Any]:
    available = sys.modules if modules is None else modules
    for module_name in ("run_thog2_owt", "__main__"):
        runner = available.get(module_name)
        if runner is not None and hasattr(runner, "_print_plastic_option"):
            return runner
    return None


def _install_startup_visibility() -> None:
    global _STARTUP_VISIBILITY_INSTALLED
    if _STARTUP_VISIBILITY_INSTALLED:
        return
    runner = _startup_runner_module()
    if runner is None:
        return
    original = runner._print_plastic_option

    def print_plastic_option_with_same_batch(label: str, value: str) -> None:
        original(label, value)
        if label == "plastic__layer_count_probe__window_size_as_number_of_probes:":
            original(
                "plastic__layer_count__same_batch_all_probes:",
                "true" if _same_batch._runtime_enabled() else "false",
            )

    runner._print_plastic_option = print_plastic_option_with_same_batch
    _STARTUP_VISIBILITY_INSTALLED = True


def _stage6_init_with_same_batch_startup_visibility(self: Any, *args: Any, **kwargs: Any) -> None:
    _ORIGINAL_STAGE6_INIT(self, *args, **kwargs)
    _install_startup_visibility()


_stage6.Stage6Trainer.__init__ = _stage6_init_with_same_batch_startup_visibility


__all__ = [
    "_align_probe_section",
    "_hide_unqualified_decision_summary",
    "_install_startup_visibility",
    "_latest_same_batch_audit_for_update",
    "_startup_runner_module",
    "_window_local_provenance",
]
# ^^^ THOG
