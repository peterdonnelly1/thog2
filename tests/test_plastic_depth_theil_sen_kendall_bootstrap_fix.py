# vvv THOG
from __future__ import annotations

import math

from sheet import plastic_depth_directional_coherence_patch as directional
from sheet import plastic_depth_sen_kendall_v055_patch as v055
from sheet import plastic_depth_theil_sen_kendall_bootstrap_fix_patch as bootstrap_fix
from sheet import plastic_depth_theil_sen_kendall_patch as tsk
from sheet import plastic_depth_wall_time_equivalent_time_gain_patch as wall_time
from sheet import trainer_step


def _score_report(scores: dict[int, float], *, bootstrap: bool):
    return tuple(
        {
            "active_layers": count,
            "feasible": math.isfinite(score),
            "score": score,
            "wall_time_algorithm": wall_time.WALL_TIME_ALGORITHM,
            "wall_time_bootstrap": bootstrap,
        }
        for count, score in sorted(scores.items())
    )


def test_tsk_bootstrap_stays_put_and_never_falls_back_to_directional_coherence(monkeypatch):
    monkeypatch.setenv(tsk._ALGORITHM_ENV, v055.LRA_ALGORITHM)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("legacy directional bootstrap executed under TSK")

    monkeypatch.setattr(directional, "_robust_scale", forbidden)
    monkeypatch.setattr(directional, "_directional_support", forbidden)
    report = _score_report({9: 0.2, 10: 0.0, 11: -0.2}, bootstrap=True)
    assert bootstrap_fix._wall_time_score_report_is_bootstrap(report)
    assert tsk._gradient_probe_classification(
        current_count=10,
        score_report=report,
        minimum_absolute_kendall_tau=0.5,
    )["fit_point_count"] == 0
    decision = trainer_step.choose_plastic_depth_count_with_mad(
        current_count=10,
        score_report=report,
        histories={},
        noise_window=1,
        noise_lambda=0.0,
        update_number=10,
        last_count_change_update=-1,
        update_brake=0,
        max_step=1,
        extrapolation_weight=1.0,
        minimum_observations=None,
    )
    assert decision.selected_count == 10
    assert all(item.standardized_improvement is None for item in decision.evidence)


def test_real_equivalent_time_scores_are_not_classified_as_bootstrap():
    report = _score_report({9: -0.2, 10: 0.0, 11: 0.2}, bootstrap=False)
    assert not bootstrap_fix._wall_time_score_report_is_bootstrap(report)
    gradient_report = tsk._gradient_probe_classification(
        current_count=10,
        score_report=report,
        minimum_absolute_kendall_tau=0.5,
    )
    assert math.isfinite(float(gradient_report["theil_sen_slope_seconds_per_layer"]))
    assert math.isfinite(float(gradient_report["kendall_tau"]))
# ^^^ THOG
