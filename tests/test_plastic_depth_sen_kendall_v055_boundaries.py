# vvv THOG
from __future__ import annotations

import math

from sheet import plastic_depth_sen_kendall_v055_patch as v055
from sheet import plastic_depth_wall_time_equivalent_time_gain_patch as wall_time


def _score_report(scores: dict[int, float]):
    return tuple(
        {
            "active_layers": count,
            "feasible": math.isfinite(score),
            "score": score,
            "wall_time_algorithm": wall_time.WALL_TIME_ALGORITHM,
            "wall_time_bootstrap": False,
        }
        for count, score in sorted(scores.items())
    )


def test_infeasible_far_left_point_does_not_invalidate_complete_stratum():
    decision = v055.choose_plastic_depth_count_with_stratified_sen_kendall_v055(
        current_count=10,
        score_report=_score_report({8: float("inf"), 9: 0.2, 10: 0.0, 11: -0.1}),
        histories={},
        noise_window=1,
        noise_lambda=0.0,
        update_number=1,
        last_count_change_update=-1,
        update_brake=0,
        max_step=1,
        minimum_observations=None,
    )
    assert decision.selected_count == 11


def test_infeasible_adjacent_left_still_prevents_shrink():
    decision = v055.choose_plastic_depth_count_with_stratified_sen_kendall_v055(
        current_count=10,
        score_report=_score_report({7: -0.4, 8: -0.3, 9: float("inf"), 10: 0.0, 11: 0.2}),
        histories={},
        noise_window=1,
        noise_lambda=0.0,
        update_number=1,
        last_count_change_update=-1,
        update_brake=0,
        max_step=1,
        minimum_observations=None,
    )
    assert decision.selected_count == 10
# ^^^ THOG
