# vvv THOG
"""Make the retained PLASTIC FINE audit independently replay v0.55 Sen/Kendall decisions without score_z."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

from . import plastic_depth_audit_patch as _audit
from . import plastic_depth_sen_kendall_v055_patch as _v055


_ORIGINAL_WINNING_PROBE_COUNT = _audit._winning_probe_count
_ORIGINAL_REPLAY = _audit.replay_plastic_depth_count_audit


def _audit_algorithm(audit: Mapping[str, Any]) -> str:
    directional = audit.get("directional_report")
    if isinstance(directional, Mapping):
        value = str(directional.get("algorithm", "")).strip()
        if value:
            return value
    return ""


def _sen_kendall_significant_candidate_from_evidence(
    evidence: Sequence[Any],
    *,
    current_count: int,
    brake_active: bool,
) -> int:
    if bool(brake_active):
        return int(current_count)
    candidates = []
    for item in evidence:
        feasible = bool(item["feasible"] if isinstance(item, Mapping) else item.feasible)
        significant = bool(item["significant"] if isinstance(item, Mapping) else item.significant)
        if not feasible or not significant:
            continue
        candidate = int(item["candidate_count"] if isinstance(item, Mapping) else item.candidate_count)
        candidates.append(candidate)
    unique = tuple(sorted(set(candidates)))
    if not unique:
        return int(current_count)
    if len(unique) != 1:
        raise ValueError(
            "PLASTIC v0.55 Sen/Kendall audit expected at most one significant adjacent candidate; "
            f"got {unique}"
        )
    candidate = int(unique[0])
    if abs(candidate - int(current_count)) != 1:
        raise ValueError(
            "PLASTIC v0.55 Sen/Kendall audit significant candidate must be adjacent; "
            f"current={current_count}, candidate={candidate}"
        )
    return candidate


def _winning_probe_count_v055(decision: Any, current_count: int) -> int:
    if _v055._runtime_algorithm() not in _v055.SEN_KENDALL_ALGORITHMS:
        return _ORIGINAL_WINNING_PROBE_COUNT(decision, current_count)
    return _sen_kendall_significant_candidate_from_evidence(
        decision.evidence,
        current_count=int(current_count),
        brake_active=bool(decision.brake_active),
    )


def _finite_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _paired_difference_from_audit(
    audit: Mapping[str, Any],
    *,
    candidate_count: int,
) -> Optional[float]:
    for item in audit.get("robust_evidence", ()):
        if int(item["candidate_count"]) != int(candidate_count):
            continue
        if not bool(item.get("feasible", False)):
            return None
        return _finite_float(item.get("paired_difference"))
    return None


def _stratified_winning_count_from_report(
    audit: Mapping[str, Any],
    report: Mapping[str, Any],
) -> int:
    current = int(audit["previous_count"])
    if bool(audit["brake_active"]) or not bool(report.get("window_ready", False)):
        return current
    sen = _finite_float(report.get("sen_slope_seconds_per_layer"))
    ken = _finite_float(report.get("kendall_tau"))
    adjacent = _finite_float(report.get("adjacent_score_seconds"))
    if sen is None or ken is None or adjacent is None or adjacent >= 0.0:
        return current
    if sen > 0.0 and ken >= _v055.FIXED_MINIMUM_ABSOLUTE_KENDALL_TAU:
        return current - 1
    if sen < 0.0 and ken <= -_v055.FIXED_MINIMUM_ABSOLUTE_KENDALL_TAU:
        return current + 1
    return current


def _lra_winning_count_from_report(
    audit: Mapping[str, Any],
    report: Mapping[str, Any],
) -> int:
    current = int(audit["previous_count"])
    if bool(audit["brake_active"]) or not bool(report.get("window_ready", False)):
        return current
    votes = tuple(str(value) for value in report.get("direction_window_votes", ()))
    if not votes:
        return current
    left_votes = sum(value == "L" for value in votes)
    right_votes = sum(value == "R" for value in votes)
    direction = -1 if left_votes * 2 > len(votes) else 1 if right_votes * 2 > len(votes) else 0
    if direction == 0:
        return current
    adjacent = _paired_difference_from_audit(
        audit,
        candidate_count=current + direction,
    )
    if adjacent is None or adjacent >= 0.0:
        return current
    return current + direction


def _sen_kendall_winning_count_from_audit(audit: Mapping[str, Any]) -> int:
    report = audit.get("directional_report")
    if not isinstance(report, Mapping):
        raise ValueError("PLASTIC v0.55 Sen/Kendall audit lacks its directional report")
    algorithm = str(report.get("algorithm", ""))
    if algorithm == _v055.STRATIFIED_ALGORITHM:
        return _stratified_winning_count_from_report(audit, report)
    if algorithm == _v055.LRA_ALGORITHM:
        return _lra_winning_count_from_report(audit, report)
    raise ValueError(f"unsupported PLASTIC v0.55 Sen/Kendall audit algorithm: {algorithm!r}")


def _replay_sen_kendall_audit(audit: Mapping[str, Any]) -> Dict[str, object]:
    current_count = int(audit["previous_count"])
    max_step = int(audit["max_step"])
    if max_step != 1:
        raise ValueError(
            "PLASTIC v0.55 Sen/Kendall audit requires max_step=1; "
            f"got {max_step}"
        )
    brake_active = bool(audit["brake_active"])
    warmup_brake_active = bool(audit.get("warmup_brake_active", False))
    winning_probe_count = _sen_kendall_winning_count_from_audit(audit)
    committed_count = (
        current_count
        if warmup_brake_active
        else current_count + max(-1, min(1, winning_probe_count - current_count))
    )
    reason = _audit._decision_reason(
        current_count=current_count,
        winning_probe_count=winning_probe_count,
        committed_count=committed_count,
        brake_active=brake_active,
        warmup_brake_active=warmup_brake_active,
    )
    replay = {
        "winning_probe_count": winning_probe_count,
        "committed_count": committed_count,
        "decision_reason": reason,
    }
    expected = {
        name: audit[name]
        for name in (
            "winning_probe_count",
            "committed_count",
            "decision_reason",
        )
    }
    if replay != expected:
        raise ValueError(
            "PLASTIC FINE audit replay mismatch: "
            f"recorded={expected}, replayed={replay}"
        )
    return replay


def replay_plastic_depth_count_audit_v055(audit: Mapping[str, Any]) -> Dict[str, object]:
    if _audit_algorithm(audit) not in _v055.SEN_KENDALL_ALGORITHMS:
        return _ORIGINAL_REPLAY(audit)
    return _replay_sen_kendall_audit(audit)


_audit._winning_probe_count = _winning_probe_count_v055
_audit.replay_plastic_depth_count_audit = replay_plastic_depth_count_audit_v055
# ^^^ THOG


__all__ = [
    "_audit_algorithm",
    "_lra_winning_count_from_report",
    "_replay_sen_kendall_audit",
    "_sen_kendall_significant_candidate_from_evidence",
    "_sen_kendall_winning_count_from_audit",
    "_stratified_winning_count_from_report",
    "replay_plastic_depth_count_audit_v055",
]
# ^^^ THOG
