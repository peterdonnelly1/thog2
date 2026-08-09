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


def _audit_evidence(
    *,
    candidate_count: int,
    significant: bool,
    paired_difference=None,
):
    return {
        "candidate_count": candidate_count,
        "feasible": True,
        "significant": significant,
        "standardized_improvement": None,
        "paired_difference": paired_difference,
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
            "window_ready": True,
            "sen_slope_seconds_per_layer": 0.482,
            "kendall_tau": 0.81,
            "adjacent_score_seconds": -0.864,
        },
        "winning_probe_count": 21,
        "committed_count": 21,
        "decision_reason": "robust_gate_committed",
    }
    assert audit_fix.replay_plastic_depth_count_audit_v055(audit) == {
        "winning_probe_count": 21,
        "committed_count": 21,
        "decision_reason": "robust_gate_committed",
    }


def test_stratified_replay_holds_until_window_is_complete() -> None:
    audit = {
        "previous_count": 22,
        "max_step": 1,
        "brake_active": False,
        "warmup_brake_active": False,
        "robust_evidence": (),
        "directional_report": {
            "algorithm": v055.STRATIFIED_ALGORITHM,
            "window_ready": False,
            "sen_slope_seconds_per_layer": 0.482,
            "kendall_tau": 0.81,
            "adjacent_score_seconds": -0.864,
        },
        "winning_probe_count": 22,
        "committed_count": 22,
        "decision_reason": "robust_gate_hold",
    }
    assert audit_fix.replay_plastic_depth_count_audit_v055(audit)["committed_count"] == 22


def test_lra_replay_uses_strict_majority_plus_adjacent_economic_check() -> None:
    audit = {
        "previous_count": 22,
        "max_step": 1,
        "brake_active": False,
        "warmup_brake_active": False,
        "robust_evidence": (
            _audit_evidence(candidate_count=21, significant=True, paired_difference=-0.400),
            _audit_evidence(candidate_count=23, significant=False, paired_difference=+0.100),
        ),
        "directional_report": {
            "algorithm": v055.LRA_ALGORITHM,
            "window_ready": True,
            "direction_window_votes": ("L", "L", "A"),
        },
        "winning_probe_count": 21,
        "committed_count": 21,
        "decision_reason": "robust_gate_committed",
    }
    assert audit_fix.replay_plastic_depth_count_audit_v055(audit)["winning_probe_count"] == 21


def test_audit_recording_rejects_non_adjacent_significant_candidate() -> None:
    evidence = (_audit_evidence(candidate_count=20, significant=True),)
    try:
        audit_fix._sen_kendall_significant_candidate_from_evidence(
            evidence,
            current_count=22,
            brake_active=False,
        )
    except ValueError as error:
        assert "must be adjacent" in str(error)
    else:
        raise AssertionError("non-adjacent Sen/Kendall audit candidate was accepted")
# ^^^ THOG
