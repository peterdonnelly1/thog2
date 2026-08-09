# vvv THOG
"""Make the retained PLASTIC FINE audit replay v0.55 Sen/Kendall decisions without score_z."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

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
    winning_probe_count = _sen_kendall_significant_candidate_from_evidence(
        audit["robust_evidence"],
        current_count=current_count,
        brake_active=brake_active,
    )
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
    "_replay_sen_kendall_audit",
    "_sen_kendall_significant_candidate_from_evidence",
    "replay_plastic_depth_count_audit_v055",
]
# ^^^ THOG
