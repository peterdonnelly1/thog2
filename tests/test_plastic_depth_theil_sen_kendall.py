# vvv THOG
from __future__ import annotations

import math

import pytest

from sheet.plastic_depth_theil_sen_kendall_patch import (
    DEFAULT_MINIMUM_ABSOLUTE_KENDALL_TAU,
    GRADIENT_ALGORITHM,
    LEGACY_DIRECTIONAL_ALGORITHM,
    _gradient_probe_classification,
    choose_plastic_depth_count_with_theil_sen_kendall,
    kendall_tau_b,
    theil_sen_slope,
)


def _score_report(current_count: int, scores: dict[int, float], *, bootstrap: bool = False):
    return tuple(
        {
            "active_layers": count,
            "feasible": math.isfinite(score),
            "score": score,
            "wall_time_algorithm": "wall_time_equivalent_time_gain",
            "wall_time_bootstrap": bootstrap,
        }
        for count, score in sorted(scores.items())
    )


def _classification(scores: dict[int, float], *, current_count: int = 10, tau: float = 0.5):
    return _gradient_probe_classification(
        current_count=current_count,
        score_report=_score_report(current_count, scores),
        minimum_absolute_kendall_tau=tau,
    )


def _decision(scores: dict[int, float], *, current_count: int = 10, histories=None, window: int = 1, tau: float = 0.5, max_step: int = 1):
    return choose_plastic_depth_count_with_theil_sen_kendall(
        current_count=current_count,
        score_report=_score_report(current_count, scores),
        histories={} if histories is None else histories,
        noise_window=window,
        noise_lambda=0.0,
        update_number=100,
        last_count_change_update=-1,
        update_brake=0,
        max_step=max_step,
        minimum_absolute_kendall_tau=tau,
    )


def test_theil_sen_known_median_pairwise_slope():
    assert theil_sen_slope(((0, 0), (1, 1), (2, 4))) == pytest.approx(2.0)


def test_theil_sen_outlier_preserves_direction():
    assert theil_sen_slope(((-3, -3), (-2, -2), (-1, -1), (0, 0), (1, 100))) > 0.0


def test_kendall_perfect_increasing():
    tau, *_ = kendall_tau_b(((0, 0), (1, 1), (2, 2), (3, 3)))
    assert tau == pytest.approx(1.0)


def test_kendall_perfect_decreasing():
    tau, *_ = kendall_tau_b(((0, 3), (1, 2), (2, 1), (3, 0)))
    assert tau == pytest.approx(-1.0)


def test_kendall_tau_b_handles_y_ties():
    tau, concordant, discordant, x_ties, y_ties = kendall_tau_b(((0, 0), (1, 0), (2, 1)))
    assert tau == pytest.approx(2.0 / math.sqrt(6.0))
    assert (concordant, discordant, x_ties, y_ties) == (2, 0, 0, 1)


def test_clean_three_point_left():
    report = _classification({9: -0.001, 10: 0.0, 11: 0.002})
    assert report["per_probe_vote"] == -1.0
    assert report["theil_sen_slope_seconds_per_layer"] > 0.0
    assert report["kendall_tau"] == pytest.approx(1.0)


def test_clean_three_point_right():
    report = _classification({9: 0.003, 10: 0.0, 11: -0.002})
    assert report["per_probe_vote"] == 1.0
    assert report["theil_sen_slope_seconds_per_layer"] < 0.0
    assert report["kendall_tau"] == pytest.approx(-1.0)


def test_both_adjacent_worse_is_ambiguous():
    assert _classification({9: 0.001, 10: 0.0, 11: 0.002})["per_probe_vote"] == 0.0


def test_adjacent_left_better_but_gradient_right_is_ambiguous():
    report = _classification({6: 10.0, 7: 9.0, 8: 8.0, 9: -0.01, 10: 0.0, 11: 0.1})
    assert report["theil_sen_slope_seconds_per_layer"] < 0.0
    assert report["kendall_tau"] <= -0.5
    assert report["left_adjacent_score_seconds"] < report["current_score_seconds"]
    assert report["right_adjacent_score_seconds"] > report["current_score_seconds"]
    assert report["per_probe_vote"] == 0.0


def test_adjacent_right_better_but_gradient_left_is_ambiguous():
    report = _classification({7: -0.8, 8: -0.6, 9: 0.01, 10: 0.0, 11: -0.01})
    assert report["theil_sen_slope_seconds_per_layer"] > 0.0
    assert report["per_probe_vote"] == 0.0


def test_tau_below_threshold_rejects_wiggly_landscape():
    scores = {6: -5.0, 7: -4.0, 8: -1.0, 9: -2.0, 10: 0.0, 11: -3.0}
    report = _classification(scores, tau=0.5)
    assert abs(report["kendall_tau"]) < 0.5
    assert report["per_probe_vote"] == 0.0


def test_tau_exactly_threshold_is_accepted():
    scores = {6: -5.0, 7: -4.0, 8: -1.0, 9: -2.0, 10: 0.0, 11: -3.0}
    report = _classification(scores, tau=7.0 / 15.0)
    assert report["kendall_tau"] == pytest.approx(7.0 / 15.0)
    assert report["per_probe_vote"] == -1.0


def test_far_right_scores_do_not_change_per_probe_decision():
    base = {7: -0.4, 8: -0.3, 9: -0.1, 10: 0.0, 11: 0.2}
    favourable_far_right = {**base, 12: -100.0, 13: -200.0, 14: -300.0}
    unfavourable_far_right = {**base, 12: 100.0, 13: 200.0, 14: 300.0}
    a = _classification(favourable_far_right)
    b = _classification(unfavourable_far_right)
    assert a["per_probe_vote"] == b["per_probe_vote"] == -1.0
    assert a["theil_sen_slope_seconds_per_layer"] == b["theil_sen_slope_seconds_per_layer"]
    assert a["kendall_tau"] == b["kendall_tau"]
    assert a["decision_fit_offsets"] == b["decision_fit_offsets"] == (-3, -2, -1, 0, 1)


def test_more_consistent_left_interpolation_retains_left():
    assert _classification({9: -0.1, 10: 0.0, 11: 0.1})["per_probe_vote"] == -1.0
    assert _classification({6: -0.4, 7: -0.3, 8: -0.2, 9: -0.1, 10: 0.0, 11: 0.1})["per_probe_vote"] == -1.0


def test_bootstrap_raw_loss_scores_never_vote():
    report = _gradient_probe_classification(
        current_count=10,
        score_report=_score_report(10, {9: -1.0, 10: 0.0, 11: 1.0}, bootstrap=True),
        minimum_absolute_kendall_tau=0.5,
    )
    assert report["fit_point_count"] == 0
    assert report["per_probe_vote"] == 0.0


def test_nonfinite_candidate_is_excluded():
    report = _classification({8: -0.2, 9: -0.1, 10: 0.0, 11: float("inf")})
    assert 1 not in report["decision_fit_offsets"]
    assert report["per_probe_vote"] == -1.0


def test_lower_boundary_without_l_minus_1_cannot_vote_left():
    report = _classification({10: 0.0, 11: 0.1}, current_count=10)
    assert report["per_probe_vote"] == 0.0


def test_upper_boundary_without_l_plus_1_cannot_vote_right():
    report = _classification({8: 0.2, 9: 0.1, 10: 0.0}, current_count=10)
    assert report["per_probe_vote"] == 0.0


def test_temporal_window_one_commits_clean_right():
    decision = _decision({9: 0.2, 10: 0.0, 11: -0.2}, window=1)
    assert decision.selected_count == 11


def test_temporal_window_three_is_not_ready_after_one_probe():
    decision = _decision({9: 0.2, 10: 0.0, 11: -0.2}, window=3)
    assert decision.selected_count == 10
    assert any(key.endswith("@TSK") for key in decision.histories)


def test_temporal_strict_majority_three_accepts_a_r_r():
    report = _score_report(10, {9: 0.2, 10: 0.0, 11: -0.2})
    first = choose_plastic_depth_count_with_theil_sen_kendall(
        current_count=10, score_report=report, histories={}, noise_window=3,
        noise_lambda=0.0, update_number=1, last_count_change_update=-1,
        update_brake=0, max_step=1, minimum_absolute_kendall_tau=0.5,
    )
    histories = dict(first.histories)
    histories["10:@TSK"] = (0.0,)
    second = choose_plastic_depth_count_with_theil_sen_kendall(
        current_count=10, score_report=report, histories=histories, noise_window=3,
        noise_lambda=0.0, update_number=2, last_count_change_update=-1,
        update_brake=0, max_step=1, minimum_absolute_kendall_tau=0.5,
    )
    third = choose_plastic_depth_count_with_theil_sen_kendall(
        current_count=10, score_report=report, histories=second.histories, noise_window=3,
        noise_lambda=0.0, update_number=3, last_count_change_update=-1,
        update_brake=0, max_step=1, minimum_absolute_kendall_tau=0.5,
    )
    assert third.selected_count == 11


def test_update_brake_blocks_otherwise_valid_move():
    decision = choose_plastic_depth_count_with_theil_sen_kendall(
        current_count=10,
        score_report=_score_report(10, {9: 0.2, 10: 0.0, 11: -0.2}),
        histories={}, noise_window=1, noise_lambda=0.0, update_number=10,
        last_count_change_update=9, update_brake=5, max_step=1,
        minimum_absolute_kendall_tau=0.5,
    )
    assert decision.brake_active
    assert decision.selected_count == 10


def test_max_step_one_bounds_multi_layer_left_winner():
    decision = _decision({6: -4.0, 7: -3.0, 8: -2.0, 9: -1.0, 10: 0.0, 11: 1.0}, max_step=1)
    assert decision.selected_count == 9


def test_non_one_max_step_remains_supported_for_interpolation_left():
    decision = _decision({6: -4.0, 7: -3.0, 8: -2.0, 9: -1.0, 10: 0.0, 11: 1.0}, max_step=3)
    assert 7 <= decision.selected_count < 10


def test_far_right_candidate_cannot_be_selected_even_when_locally_attractive():
    decision = _decision({9: 0.2, 10: 0.0, 11: -0.1, 12: -100.0, 13: -200.0}, max_step=3)
    assert decision.selected_count == 11


def test_algorithm_constants_and_default_threshold_are_stable():
    assert GRADIENT_ALGORITHM == "wall_time__theil_sen_kendall_LRA"
    assert LEGACY_DIRECTIONAL_ALGORITHM == "directional_coherence"
    assert DEFAULT_MINIMUM_ABSOLUTE_KENDALL_TAU == 0.5
# ^^^ THOG
