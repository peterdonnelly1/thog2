# vvv THOG
"""Final PLASTIC DEPTH progress-row alignment and probe-state visibility."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Mapping, Optional

from . import plastic_depth_console_cleanup_patch as _cleanup
from . import plastic_depth_lookahead_patch as _lookahead
from . import stage6_trainer as _stage6


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
_PALE_RED = "\033[38;2;255;150;150m"
_PALE_CYAN = "\033[38;2;150;220;255m"
_RESET = "\033[0m"
# _ALIGNMENT_LABELS = ("probe_losses", "score_z", "sampled =")                                                                                         # <<< THOG preserve the pre-public-formatter alignment field set
_ALIGNMENT_LABELS = ("probe_losses", "score_z", "change_z", "sampled =")                                                                              # <<< THOG align every active PLASTIC suffix spelling after the final public formatter
_ALIGNMENT_BY_RUN_ID: Dict[str, Dict[str, int]] = {}
_ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD = _stage6.Stage6Trainer._prepare_console_progress_payload
_ORIGINAL_FORMAT_PROGRESS_LINE = _stage6.format_progress_line
_ORIGINAL_COLOUR_POSITIVE_SCORE = _cleanup._colour_positive_score
_ORIGINAL_PRINT_PROGRESS = _stage6.Stage6Trainer._print_progress


def _console_int(value: Any) -> int:
    return int(str(value).strip().replace(",", ""))


def _latest_count_decision_payload(trainer: Any) -> Optional[Mapping[str, Any]]:
    for event in reversed(getattr(trainer, "events", ())):
        if getattr(event, "name", None) == "plastic_depth_count_decision":
            return getattr(event, "payload", None)
    return None


def _row_is_current_probe(trainer: Any, completed_updates: int) -> bool:
    config = getattr(trainer, "config", None)
    if not bool(getattr(config, "plastic__enabled", False)):
        return False
    if not bool(getattr(config, "plastic__do_learn_layer_count", False)):
        return False
    lattice = trainer._plastic_depth_lattice()
    if lattice is None:
        return False
    last_decision = getattr(lattice, "last_count_decision_update", None)
    if last_decision is None:
        return False
    last_decision_update = int(last_decision.item()) if hasattr(last_decision, "item") else int(last_decision)
    return int(completed_updates) == last_decision_update


# vvv THOG derive brake state from committed trainer state so non-probe logged rows inside the brake window are visibly marked
def _row_has_active_update_brake(trainer: Any, completed_updates: int) -> bool:
    config = getattr(trainer, "config", None)
    if not bool(getattr(config, "plastic__enabled", False)):
        return False
    if not bool(getattr(config, "plastic__do_learn_layer_count", False)):
        return False
    update_brake = int(getattr(config, "plastic__layer_count_update_brake", 0))
    last_change_update = int(
        getattr(
            getattr(trainer, "state", None),
            "plastic_depth_last_count_change_update",
            -1,
        )
    )
    updates_since_change = int(completed_updates) - last_change_update
    return update_brake > 0 and last_change_update >= 0 and 0 < updates_since_change < update_brake
# ^^^ THOG


# vvv THOG expose the separate warmup gate whenever learned count movement is prohibited
def _row_has_warmup_brake(trainer: Any, completed_updates: int) -> bool:
    config = getattr(trainer, "config", None)
    return (
        bool(getattr(config, "plastic__enabled", False))
        and bool(getattr(config, "plastic__do_learn_layer_count", False))
        and bool(getattr(config, "plastic__freeze_geometry_during_warmup", False))
        and int(completed_updates) < int(getattr(config, "warmup_updates", 0))
    )
# ^^^ THOG


def _remove_probe_fields(values: Dict[str, Any]) -> None:
    for key in (
        "plastic_probe_losses",
        "plastic_probe_offsets",
        "plastic_probe_edge_offsets",
        "plastic_loss_gain",
        "plastic_score_z",
    ):
        values.pop(key, None)


def _prepare_console_progress_payload_with_probe_visibility(
    self: Any,
    event: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    values = _ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD(self, event, payload)
    if event not in {"optimizer_progress", "evaluation_completed"}:
        return values
    if "completed_updates" not in values:
        return values
    completed_updates = _console_int(values["completed_updates"])
    if not _row_is_current_probe(self, completed_updates):
        _remove_probe_fields(values)
        # values.pop("plastic_update_brake_active", None)                                                                                                  # <<< THOG preserve old probe-only brake suppression for source history
    # decision_payload = _latest_count_decision_payload(self)                                                                                              # <<< THOG preserve old decision-event-only brake source
    # if decision_payload is not None and bool(decision_payload.get("brake_active", False)):
    #     values["plastic_update_brake_active"] = True
    if _row_has_active_update_brake(self, completed_updates):
        values["plastic_update_brake_active"] = True
    else:
        values.pop("plastic_update_brake_active", None)
    if _row_has_warmup_brake(self, completed_updates):
        values["plastic_warmup_brake_active"] = True
    else:
        values.pop("plastic_warmup_brake_active", None)
    return values


_stage6.Stage6Trainer._prepare_console_progress_payload = _prepare_console_progress_payload_with_probe_visibility
# ^^^ THOG


# vvv THOG reserve 0+/0- for non-zero scores that would otherwise round to 0.00 at two decimal places
def _format_score_z_with_zero_sign(value: Any) -> str:
    if value is None:
        return f"{'-':>9}"
    numeric = float(value)
    if not math.isfinite(numeric):
        return f"{str(numeric):>9}"
    if 0.0 < numeric < 0.005:
        return f"{'0+':>9}"
    if -0.005 < numeric < 0.0:
        return f"{'0-':>9}"
    magnitude = abs(numeric)
    if magnitude >= 1000.0:
        return f"{numeric:+9.2e}"
    return f"{numeric:+9.2f}"


def _colour_positive_score_with_zero_sign(value: str) -> str:
    if value.strip() == "0+":
        return f"{_cleanup._GREEN}{value}{_cleanup._RESET}"
    return _ORIGINAL_COLOUR_POSITIVE_SCORE(value)


_lookahead._format_score_z = _format_score_z_with_zero_sign
_cleanup._colour_positive_score = _colour_positive_score_with_zero_sign
# ^^^ THOG


# vvv THOG align PLASTIC suffix fields by visible terminal column and annotate active count brakes
def _visible_width(value: str) -> int:
    return len(_ANSI_ESCAPE.sub("", value).expandtabs(8))


def _field_start(line: str, label: str) -> Optional[int]:
    raw_index = line.find(label)
    if raw_index < 0:
        return None
    return _visible_width(line[:raw_index])


def _align_field_to_column(line: str, label: str, target_column: int) -> str:
    raw_index = line.find(label)
    if raw_index < 0:
        return line
    prefix = line[:raw_index].rstrip(" \t")
    suffix = line[raw_index:]
    prefix_width = _visible_width(prefix)
    padding = max(2, int(target_column) - prefix_width)
    return prefix + (" " * padding) + suffix


def _record_alignment(run_id: str, line: str) -> None:
    # positions = {                                                                                                                                         # <<< THOG preserve overwrite-on-every-T-row behaviour that discarded absent probe columns
    #     label: position
    #     for label in _ALIGNMENT_LABELS
    #     if (position := _field_start(line, label)) is not None
    # }
    positions = dict(_ALIGNMENT_BY_RUN_ID.get(str(run_id), {}))
    positions.update(
        {
            label: position
            for label in _ALIGNMENT_LABELS
            if (position := _field_start(line, label)) is not None
        }
    )
    if positions:
        _ALIGNMENT_BY_RUN_ID[str(run_id)] = positions


def _align_validation_row(run_id: str, line: str) -> str:
    positions = _ALIGNMENT_BY_RUN_ID.get(str(run_id), {})
    aligned = line
    for label in _ALIGNMENT_LABELS:
        target = positions.get(label)
        if target is not None:
            aligned = _align_field_to_column(aligned, label, target)
    return aligned


def _format_progress_line_with_minor_plastic_console_changes(
    run_id: str,
    event: str,
    payload: Dict[str, Any],
) -> str:
    line = _ORIGINAL_FORMAT_PROGRESS_LINE(run_id, event, payload)
    if event == "optimizer_progress":
        _record_alignment(run_id, line)
    elif event == "evaluation_completed":
        line = _align_validation_row(run_id, line)
    if bool(payload.get("plastic_update_brake_active", False)):
        line = f"{line.rstrip()}  {_PALE_RED}<<< update brake on{_RESET}"
    if bool(payload.get("plastic_warmup_brake_active", False)):
        line = f"{line.rstrip()}  {_PALE_CYAN}<<< warmup braked enabled{_RESET}"
    return line


_stage6.format_progress_line = _format_progress_line_with_minor_plastic_console_changes
# ^^^ THOG


# vvv THOG apply alignment after every later public formatter has inserted its own tabs and suffix fields
def _align_final_progress_line(run_id: str, event: str, line: str) -> str:
    if event == "optimizer_progress":
        _record_alignment(run_id, line)
        return line
    if event == "evaluation_completed":
        return _align_validation_row(run_id, line)
    return line


def _print_progress_with_final_plastic_alignment(
    self: Any,
    run_id: str,
    event: str,
    **payload: Any,
) -> None:
    if not self.distributed.is_primary:
        return
    values = _stage6.Stage6Trainer._prepare_console_progress_payload(self, event, payload)
    line = _stage6.format_progress_line(run_id, event, values)
    print(_align_final_progress_line(run_id, event, line), flush=True)
    if event == "run_started":
        print(flush=True)


_stage6.Stage6Trainer._print_progress = _print_progress_with_final_plastic_alignment
# ^^^ THOG
