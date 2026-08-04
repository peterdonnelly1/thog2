# vvv THOG
from __future__ import annotations

import math

import pytest

from sheet.plastic_depth_controller import (
    PLASTIC_DEPTH_MAD_SIGMA_FLOOR,
    choose_plastic_depth_count_with_mad,
)


def _score_report(*, lower: float = 1.1, current: float = 1.0, upper: float = 1.1):
    return (
        {"active_layers": 2, "score": lower, "feasible": math.isfinite(lower)},
        {"active_layers": 3, "score": current, "feasible": math.isfinite(current)},
        {"active_layers": 4, "score": upper, "feasible": math.isfinite(upper)},
    )


def test_mad_gate_collects_required_observations_before_transition() -> None:
    histories = {}
    for update_number in range(1, 3):
        decision = choose_plastic_depth_count_with_mad(
            current_count=3,
            score_report=_score_report(lower=0.8),
            histories=histories,
            noise_window=8,
            minimum_observations=3,
            noise_lambda=1.0,
            update_number=update_number,
            last_count_change_update=-1,
            update_brake=0,
        )
        histories = decision.histories
        assert decision.selected_count == 3
    decision = choose_plastic_depth_count_with_mad(
        current_count=3,
        score_report=_score_report(lower=0.8),
        histories=histories,
        noise_window=8,
        minimum_observations=3,
        noise_lambda=1.0,
        update_number=3,
        last_count_change_update=-1,
        update_brake=0,
    )
    assert decision.selected_count == 2
    assert decision.evidence[0].observation_count == 3
    assert decision.evidence[0].significant


def test_zero_mad_uses_positive_scale_floor() -> None:
    decision = choose_plastic_depth_count_with_mad(
        current_count=3,
        score_report=_score_report(lower=0.5),
        histories={"3:-1": (-0.5, -0.5, -0.5)},
        noise_window=8,
        minimum_observations=1,
        noise_lambda=1.0,
        update_number=4,
        last_count_change_update=-1,
        update_brake=0,
    )
    evidence = decision.evidence[0]
    assert evidence.mad == 0.0
    assert evidence.sigma is not None
    assert evidence.sigma >= PLASTIC_DEPTH_MAD_SIGMA_FLOOR
    assert math.isfinite(evidence.standardized_improvement)
    assert decision.selected_count == 2


def test_update_brake_blocks_transition_but_preserves_new_evidence() -> None:
    decision = choose_plastic_depth_count_with_mad(
        current_count=3,
        score_report=_score_report(upper=0.5),
        histories={},
        noise_window=8,
        minimum_observations=1,
        noise_lambda=0.0,
        update_number=12,
        last_count_change_update=10,
        update_brake=5,
    )
    assert decision.brake_active
    assert decision.selected_count == 3
    assert decision.histories["3:+1"] == (-0.5,)

    released = choose_plastic_depth_count_with_mad(
        current_count=3,
        score_report=_score_report(upper=0.5),
        histories=decision.histories,
        noise_window=8,
        minimum_observations=1,
        noise_lambda=0.0,
        update_number=15,
        last_count_change_update=10,
        update_brake=5,
    )
    assert not released.brake_active
    assert released.selected_count == 4


def test_exact_standardized_tie_prefers_lower_count() -> None:
    decision = choose_plastic_depth_count_with_mad(
        current_count=3,
        score_report=_score_report(lower=0.5, upper=0.5),
        histories={},
        noise_window=8,
        minimum_observations=1,
        noise_lambda=0.0,
        update_number=1,
        last_count_change_update=-1,
        update_brake=0,
    )
    assert decision.selected_count == 2


def test_histories_are_count_direction_specific_and_windowed() -> None:
    decision = choose_plastic_depth_count_with_mad(
        current_count=3,
        score_report=_score_report(lower=1.2, upper=1.3),
        histories={"3:-1": (1.0, 1.1, 1.2), "2:+1": (9.0,)},
        noise_window=3,
        minimum_observations=3,
        noise_lambda=3.0,
        update_number=4,
        last_count_change_update=-1,
        update_brake=0,
    )
    assert decision.histories["3:-1"] == pytest.approx((1.1, 1.2, 0.2))
    assert decision.histories["3:+1"] == pytest.approx((0.3,))
    assert decision.histories["2:+1"] == pytest.approx((9.0,))


def test_infeasible_direction_does_not_create_history() -> None:
    decision = choose_plastic_depth_count_with_mad(
        current_count=3,
        score_report=_score_report(lower=float("inf"), upper=0.8),
        histories={},
        noise_window=8,
        minimum_observations=1,
        noise_lambda=0.0,
        update_number=1,
        last_count_change_update=-1,
        update_brake=0,
    )
    assert "3:-1" not in decision.histories
    assert decision.evidence[0].feasible is False
    assert decision.selected_count == 4


def test_invalid_controller_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="noise_window"):
        choose_plastic_depth_count_with_mad(
            current_count=3,
            score_report=_score_report(),
            histories={},
            noise_window=0,
            minimum_observations=1,
            noise_lambda=1.0,
            update_number=1,
            last_count_change_update=-1,
            update_brake=0,
        )
    with pytest.raises(ValueError, match="non-finite"):
        choose_plastic_depth_count_with_mad(
            current_count=3,
            score_report=_score_report(),
            histories={"3:-1": (float("nan"),)},
            noise_window=8,
            minimum_observations=1,
            noise_lambda=1.0,
            update_number=1,
            last_count_change_update=-1,
            update_brake=0,
        )
# ^^^ THOG
