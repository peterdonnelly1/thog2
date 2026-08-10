from __future__ import annotations

from copy import deepcopy

import pytest

from sheet import plastic_depth_controller as controller
from sheet import plastic_depth_sen_kendall_v055_patch as v055
from sheet import plastic_depth_sen_kendall_v055_audit_fix_patch as audit_v055
from sheet import plastic_depth_v055_same_batch_audit_bridge_patch as bridge


def _decision(*, current: int = 10, selected: int = 11):
    evidence = (
        controller.PlasticDepthPairedDirectionEvidence(
            candidate_count=current - 1,
            direction=-1,
            paired_difference=0.4,
            observation_count=2,
            median=None,
            mad=None,
            sigma=None,
            standardized_improvement=None,
            significant=False,
            feasible=True,
        ),
        controller.PlasticDepthPairedDirectionEvidence(
            candidate_count=current + 1,
            direction=1,
            paired_difference=-0.2,
            observation_count=2,
            median=None,
            mad=None,
            sigma=None,
            standardized_improvement=None,
            significant=True,
            feasible=True,
        ),
    )
    return controller.PlasticDepthRobustCountDecision(
        selected_count=selected,
        current_count=current,
        update_number=20,
        brake_active=False,
        last_count_change_update=-1,
        histories={},
        evidence=evidence,
    )


def _stratified_report() -> dict:
    return {
        "algorithm": v055.STRATIFIED_ALGORITHM,
        "window_ready": True,
        "sen_slope_seconds_per_layer": -1.25,
        "kendall_tau": -0.8,
        "adjacent_score_seconds": -0.2,
        "selected_count": 11,
    }


def test_same_batch_hold_records_raw_v055_decision_and_replays_hold() -> None:
    raw_report = _stratified_report()
    context = {
        "current_count": 10,
        "selected_count": 11,
        "decision": _decision(),
        "plastic_v055_sen_kendall_report": deepcopy(raw_report),
        "plastic_directional_report": deepcopy(raw_report),
    }

    bridge._force_framework_hold_with_v055_audit_metadata(
        context,
        reason="incomplete_same_batch_window",
    )

    assert context["selected_count"] == 10
    assert context["decision"].selected_count == 10
    assert all(not item.significant for item in context["decision"].evidence)
    for report_key in (
        "plastic_v055_sen_kendall_report",
        "plastic_directional_report",
    ):
        report = context[report_key]
        assert report["selected_count"] == 10
        assert report["framework_hold_reason"] == "incomplete_same_batch_window"
        assert report["framework_raw_selected_count"] == 11

    # vvv THOG reproduce the live v0.56 commit state: the compatibility directional report has been removed, leaving only the authoritative Sen/Kendall report
    audit = {
        "previous_count": 10,
        "max_step": 1,
        "brake_active": False,
        "warmup_brake_active": False,
        "sen_kendall_report": context["plastic_v055_sen_kendall_report"],
        "winning_probe_count": 10,
        "committed_count": 10,
        "decision_reason": "robust_gate_hold",
    }
    assert audit_v055.replay_plastic_depth_count_audit_v055(audit) == {
        "winning_probe_count": 10,
        "committed_count": 10,
        "decision_reason": "robust_gate_hold",
    }
    # ^^^ THOG


def test_same_batch_hold_audit_rejects_tampered_raw_v055_decision() -> None:
    report = _stratified_report()
    report["framework_hold_reason"] = "incomplete_same_batch_window"
    report["framework_raw_selected_count"] = 9
    audit = {
        "previous_count": 10,
        "brake_active": False,
        "directional_report": deepcopy(report),
    }

    with pytest.raises(ValueError, match="framework-hold raw decision mismatch"):
        bridge._sen_kendall_winning_count_from_audit_with_framework_hold(audit)


def test_no_framework_hold_preserves_raw_v055_audit_winner() -> None:
    audit = {
        "previous_count": 10,
        "brake_active": False,
        "directional_report": _stratified_report(),
    }

    assert bridge._sen_kendall_winning_count_from_audit_with_framework_hold(audit) == 11
