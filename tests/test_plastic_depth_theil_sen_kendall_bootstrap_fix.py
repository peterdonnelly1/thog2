# vvv THOG
from __future__ import annotations

import math
from types import SimpleNamespace

from sheet import plastic_depth_theil_sen_kendall_bootstrap_fix_patch as bootstrap_fix
from sheet import plastic_depth_theil_sen_kendall_patch as gradient
from sheet import plastic_depth_wall_time_equivalent_time_gain_patch as wall_time


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


def test_tsk_mode_uses_v0531_directional_bootstrap_for_first_count_change(monkeypatch):
    monkeypatch.setenv(gradient._ALGORITHM_ENV, gradient.GRADIENT_ALGORITHM)
    trainer = SimpleNamespace(config=SimpleNamespace(plastic__layer_count__max_allowable_layer_change=1))
    token = wall_time._ACTIVE_TRAINER.set(trainer)
    try:
        report = _score_report({9: 0.2, 10: 0.0, 11: -0.2}, bootstrap=True)
        assert bootstrap_fix._wall_time_score_report_is_bootstrap(report)
        assert gradient._gradient_probe_classification(
            current_count=10,
            score_report=report,
            minimum_absolute_kendall_tau=0.5,
        )["fit_point_count"] == 0
        decision = bootstrap_fix._choose_count_with_v0531_bootstrap(
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
    finally:
        wall_time._ACTIVE_TRAINER.reset(token)
    assert decision.selected_count == 11


def test_real_equivalent_time_scores_are_not_classified_as_bootstrap():
    report = _score_report({9: -0.2, 10: 0.0, 11: 0.2}, bootstrap=False)
    assert not bootstrap_fix._wall_time_score_report_is_bootstrap(report)
    gradient_report = gradient._gradient_probe_classification(
        current_count=10,
        score_report=report,
        minimum_absolute_kendall_tau=0.5,
    )
    assert math.isfinite(float(gradient_report["theil_sen_slope_seconds_per_layer"]))
    assert math.isfinite(float(gradient_report["kendall_tau"]))
# ^^^ THOG
