# vvv THOG
from __future__ import annotations

from types import SimpleNamespace

from sheet import plastic_depth_sen_kendall_v055_audit_fix_patch as audit_fix
from sheet import plastic_depth_sen_kendall_v055_patch as v055


def _evidence(*, candidate_count: int, significant: bool):
    return SimpleNamespace(
        candidate_count=candidate_count,
        feasible=True,
        significant=significant,
        standardized_improvement=None,
    )


def _audit_evidence(*, candidate_count: int, significant: bool):
    return {
        "candidate_count": candidate_count,
        "feasible": True,
        "significant": significant,
        "standardized_improvement": None,
    }


def test_sen_kendall_winner_does_not_require_score_z(monkeypatch) -> None:
    monkeypatch.setenv(v055._tsk._ALGORITHM_ENV, v055.STRATIFIED_ALGORITHM)
    decision = SimpleNamespace(
        evidence=(
            _evidence(candidate_count=21, significant=True),
            _evidence(candidate_count=23, significant=False),
        ),
        brake_active=False,
    )
    assert audit_fix._winning_probe_count_v055(decision, 22) == 21


def test_replay_matches_actual_stratified_22_to_21_commit_without_score_z() -> None:
    audit = {
        "previous_count": 22,
        "max_step": 1,
        "brake_active": False,
        "warmup_brake_active": False,
        "robust_evidence": (
            _audit_evidence(candidate_count=21, significant=True),
            _audit_evidence(candidate_count=23, significant=False),
        ),
        "directional_report": {
            "algorithm": v055.STRATIFIED_ALGORITHM,
            "sen_slope_seconds_per_layer": 0.482,
            "kendall_tau": 0.81,
            "adjacent_score_seconds": -0.864,
        },
        "winning_probe_count": 21,
        "committed_count": 21,
        "decision_reason": "winning_probe_committed",
    }
    assert audit_fix.replay_plastic_depth_count_audit_v055(audit) == {
        "winning_probe_count": 21,
        "committed_count": 21,
        "decision_reason": "winning_probe_committed",
    }


def test_replay_rejects_non_adjacent_sen_kendall_candidate() -> None:
    audit = {
        "previous_count": 22,
        "max_step": 1,
        "brake_active": False,
        "warmup_brake_active": False,
        "robust_evidence": (
            _audit_evidence(candidate_count=20, significant=True),
        ),
        "directional_report": {"algorithm": v055.STRATIFIED_ALGORITHM},
        "winning_probe_count": 20,
        "committed_count": 21,
        "decision_reason": "max_step_limited",
    }
    try:
        audit_fix.replay_plastic_depth_count_audit_v055(audit)
    except ValueError as error:
        assert "must be adjacent" in str(error)
    else:
        raise AssertionError("non-adjacent Sen/Kendall audit candidate was accepted")
# ^^^ THOG
