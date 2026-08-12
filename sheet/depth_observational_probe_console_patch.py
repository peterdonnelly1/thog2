# vvv THOG
"""Expose read-only fixed-run DEPTH probe losses on the ordinary progress row."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from . import depth_weight_curves_and_observational_probes_patch as _depth
from . import stage6_trainer as _stage6


_ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD = _stage6.Stage6Trainer._prepare_console_progress_payload


def _latest_observational_probe_report(trainer: Any) -> Optional[Dict[str, Any]]:
    if bool(getattr(getattr(trainer, "config", None), "plastic__do_learn_layer_count", False)):
        return None
    for item in reversed(tuple(getattr(trainer, "events", ()))):
        if getattr(item, "name", None) != "plastic_depth_count_decision":
            continue
        payload = getattr(item, "payload", None)
        if not isinstance(payload, Mapping) or not bool(payload.get("observational_only", False)):
            continue
        current = int(payload["previous_active_layers"])
        candidates = tuple(payload.get("candidates", ()))
        counts = tuple(int(candidate["active_layers"]) for candidate in candidates)
        losses = tuple(
            None
            if candidate.get("validation_loss") is None
            else float(candidate["validation_loss"])
            for candidate in candidates
        )
        return {
            "probe_update": int(payload.get("probe_update", -1)),
            "current_layer_count": current,
            "offsets": tuple(count - current for count in counts),
            "losses": losses,
        }
    return None


def _prepare_console_progress_payload_with_observational_probe(
    self: Any,
    event: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    values = _ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD(self, event, payload)
    if event != "optimizer_progress":
        return values
    if not _depth._observational_probe_enabled(self):
        return values
    report = _latest_observational_probe_report(self)
    if report is None:
        return values
    completed_updates = int(
        str(values.get("completed_updates", getattr(self.state, "completed_updates", -1)))
        .strip()
        .replace(",", "")
    )
    if int(report["probe_update"]) != completed_updates:
        return values
    values["current_layer_count"] = int(report["current_layer_count"])
    values["plastic_probe_offsets"] = tuple(int(value) for value in report["offsets"])
    values["plastic_probe_losses"] = tuple(report["losses"])
    return values


_stage6.Stage6Trainer._prepare_console_progress_payload = _prepare_console_progress_payload_with_observational_probe
# ^^^ THOG


__all__ = [
    "_latest_observational_probe_report",
    "_prepare_console_progress_payload_with_observational_probe",
]
# ^^^ THOG
