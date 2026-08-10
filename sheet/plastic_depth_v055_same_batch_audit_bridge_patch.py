# vvv THOG
"""Make same-batch framework holds explicit and independently replayable in the v0.55 Sen/Kendall audit."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from . import plastic_depth_same_batch_all_probes_patch as _same_batch
from . import plastic_depth_sen_kendall_v055_audit_fix_patch as _audit_v055
from . import plastic_depth_sen_kendall_v055_patch as _v055


_FRAMEWORK_HOLD_REASON = "framework_hold_reason"
_FRAMEWORK_RAW_SELECTED_COUNT = "framework_raw_selected_count"
_ORIGINAL_FORCE_FRAMEWORK_HOLD = _same_batch._force_framework_hold
_ORIGINAL_SEN_KENDALL_WINNING_COUNT_FROM_AUDIT = _audit_v055._sen_kendall_winning_count_from_audit


def _force_framework_hold_with_v055_audit_metadata(context: Dict[str, Any], *, reason: str) -> None:
    decision = context.get("decision")
    raw_selected_count = None if decision is None else int(decision.selected_count)
    _ORIGINAL_FORCE_FRAMEWORK_HOLD(context, reason=reason)
    report = context.get("plastic_directional_report")
    if not isinstance(report, dict):
        return
    if str(report.get("algorithm", "")) not in _v055.SEN_KENDALL_ALGORITHMS:
        return
    if raw_selected_count is None:
        raise RuntimeError("PLASTIC v0.55 same-batch framework hold lost the raw selected count")
    report[_FRAMEWORK_HOLD_REASON] = str(reason)
    report[_FRAMEWORK_RAW_SELECTED_COUNT] = int(raw_selected_count)


def _sen_kendall_winning_count_from_audit_with_framework_hold(audit: Mapping[str, Any]) -> int:
    raw_winning_count = int(_ORIGINAL_SEN_KENDALL_WINNING_COUNT_FROM_AUDIT(audit))
    report = audit.get("directional_report")
    if not isinstance(report, Mapping):
        return raw_winning_count
    hold_reason = str(report.get(_FRAMEWORK_HOLD_REASON, "")).strip()
    if not hold_reason:
        return raw_winning_count
    current_count = int(audit["previous_count"])
    recorded_raw = report.get(_FRAMEWORK_RAW_SELECTED_COUNT)
    if recorded_raw is None:
        raise ValueError(
            "PLASTIC v0.55 audit framework hold lacks its pre-hold selected count; "
            f"reason={hold_reason!r}"
        )
    recorded_raw_count = int(recorded_raw)
    if recorded_raw_count != raw_winning_count:
        raise ValueError(
            "PLASTIC v0.55 audit framework-hold raw decision mismatch: "
            f"recorded_raw={recorded_raw_count}, replayed_raw={raw_winning_count}, reason={hold_reason!r}"
        )
    if abs(recorded_raw_count - current_count) > 1:
        raise ValueError(
            "PLASTIC v0.55 audit framework-hold raw decision must be current or adjacent; "
            f"current={current_count}, recorded_raw={recorded_raw_count}"
        )
    return current_count


_same_batch._force_framework_hold = _force_framework_hold_with_v055_audit_metadata
_audit_v055._sen_kendall_winning_count_from_audit = (
    _sen_kendall_winning_count_from_audit_with_framework_hold
)


__all__ = [
    "_force_framework_hold_with_v055_audit_metadata",
    "_sen_kendall_winning_count_from_audit_with_framework_hold",
]
# ^^^ THOG
