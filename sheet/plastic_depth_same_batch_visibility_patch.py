# vvv THOG
"""Operator-visible same-batch PLASTIC window identity and window-local provenance."""

from __future__ import annotations

import sys
from typing import Any, Dict, Mapping, Optional, Sequence

from . import plastic_depth_same_batch_all_probes_patch as _same_batch
from . import stage6_trainer as _stage6
from . import trainer_step as _trainer_step


_ORIGINAL_COMMIT_INLINE_UPDATE = _trainer_step.TrainerStepMixin._commit_plastic_depth_inline_update
_ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD = _stage6.Stage6Trainer._prepare_console_progress_payload
_ORIGINAL_FORMAT_PROGRESS_LINE = _stage6.format_progress_line
_ORIGINAL_STAGE6_INIT = _stage6.Stage6Trainer.__init__
_STARTUP_VISIBILITY_INSTALLED = False


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
    if event != "optimizer_progress" or not _same_batch._runtime_enabled():
        return values
    try:
        completed_updates = int(values.get("completed_updates", payload.get("completed_updates")))
    except (TypeError, ValueError):
        return values
    row = _latest_same_batch_audit_for_update(self, completed_updates)
    if row is None:
        return values

    ordinal = int(row["probe_window_ordinal"])
    values["plastic_probe_sequence"] = ordinal
    values["plastic_probe_provenance"] = _window_local_provenance(ordinal)
    values["plastic_same_batch_window_id"] = int(row["probe_window_id"])
    values["plastic_same_batch_window_ordinal"] = ordinal
    values["plastic_same_batch_window_size"] = int(row["probe_window_size"])
    values["plastic_same_batch_batch_digest"] = str(row["probe_batch_digest"])
    return values


_stage6.Stage6Trainer._prepare_console_progress_payload = (
    _prepare_console_progress_payload_with_same_batch_identity
)


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
    line = _ORIGINAL_FORMAT_PROGRESS_LINE(run_id, event, local_payload)
    if (
        event == "optimizer_progress"
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
    return line


_stage6.format_progress_line = _format_progress_line_with_same_batch_identity


def _install_startup_visibility() -> None:
    global _STARTUP_VISIBILITY_INSTALLED
    if _STARTUP_VISIBILITY_INSTALLED:
        return
    runner = sys.modules.get("run_thog2_owt")
    if runner is None or not hasattr(runner, "_print_plastic_option"):
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
    "_install_startup_visibility",
    "_latest_same_batch_audit_for_update",
    "_window_local_provenance",
]
# ^^^ THOG
