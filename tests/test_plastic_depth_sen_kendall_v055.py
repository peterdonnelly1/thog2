# vvv THOG
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from sheet import plastic_depth_sen_kendall_v055_patch as v055
from sheet import plastic_depth_theil_sen_kendall_patch as tsk
from sheet import plastic_depth_wall_time_equivalent_time_gain_patch as wall_time
from sheet import trainer_step


def _score_report(scores: dict[int, float], *, bootstrap: bool = False):
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


def _decision(function, scores, *, histories=None, window=1, current=10, update=100, brake=0, last=-1):
    return function(
        current_count=current,
        score_report=_score_report(scores),
        histories={} if histories is None else histories,
        noise_window=window,
        noise_lambda=0.0,
        update_number=update,
        last_count_change_update=last,
        update_brake=brake,
        max_step=1,
        minimum_observations=None,
    )


def test_v055_algorithm_names_are_exact_and_old_name_is_retired(monkeypatch):
    assert v055.LRA_ALGORITHM == "wall_time__theil_sen_kendall_LRA"
    assert v055.STRATIFIED_ALGORITHM == "wall_time__sen_kendall__tau__stratified"
    monkeypatch.setenv(tsk._ALGORITHM_ENV, "wall_time__gradient__theil_sen_kendall_slope_tau")
    with pytest.raises(ValueError, match="renamed"):
        v055._runtime_algorithm()


def test_tau_cli_control_is_rejected():
    assert v055._contains_retired_tau(["--plastic__layer_count_gradient__minimum_absolute_kendall_tau=0.6"])
    with pytest.raises(SystemExit, match="removed in PLASTIC v0.55"):
        v055._parse_args_v055(
            __import__("argparse").ArgumentParser(add_help=False),
            ["--plastic__layer_count_gradient__minimum_absolute_kendall_tau", "0.6"],
        )


def test_fixed_kendall_coherence_is_half():
    assert v055.FIXED_MINIMUM_ABSOLUTE_KENDALL_TAU == pytest.approx(0.5)


def test_stratified_sen_pools_only_within_probe_slopes():
    strata = (
        ((-1.0, 1000.0), (0.0, 1001.0), (1.0, 1002.0)),
        ((-1.0, -1000.0), (0.0, -999.0), (1.0, -998.0)),
    )
    assert v055.stratified_sen_slope(strata) == pytest.approx(1.0)


def test_stratified_kendall_pools_within_probe_ordering():
    increasing = ((-1.0, -1.0), (0.0, 0.0), (1.0, 1.0))
    tau, concordant, discordant, ties = v055.stratified_kendall_tau_b((increasing, increasing))
    assert tau == pytest.approx(1.0)
    assert (concordant, discordant, ties) == (6, 0, 0)


def test_stratified_kendall_opposite_strata_cancel():
    increasing = ((-1.0, -1.0), (0.0, 0.0), (1.0, 1.0))
    decreasing = ((-1.0, 1.0), (0.0, 0.0), (1.0, -1.0))
    tau, *_ = v055.stratified_kendall_tau_b((increasing, decreasing))
    assert tau == pytest.approx(0.0)


def test_lra_window_retains_majority_but_has_no_score_z_evidence():
    first = _decision(
        v055.choose_plastic_depth_count_with_tsk_lra_v055,
        {9: 0.2, 10: 0.0, 11: -0.2},
        window=2,
        update=1,
    )
    assert first.selected_count == 10
    second = _decision(
        v055.choose_plastic_depth_count_with_tsk_lra_v055,
        {9: 0.2, 10: 0.0, 11: -0.2},
        histories=first.histories,
        window=2,
        update=2,
    )
    assert second.selected_count == 11
    assert all(item.standardized_improvement is None for item in second.evidence)
    assert all(item.mad is None and item.sigma is None for item in second.evidence)


def test_stratified_window_moves_right_after_complete_window():
    first = _decision(
        v055.choose_plastic_depth_count_with_stratified_sen_kendall_v055,
        {8: 0.4, 9: 0.2, 10: 0.0, 11: -0.2, 12: -0.4},
        window=2,
        update=1,
    )
    assert first.selected_count == 10
    second = _decision(
        v055.choose_plastic_depth_count_with_stratified_sen_kendall_v055,
        {8: 0.3, 9: 0.15, 10: 0.0, 11: -0.1, 12: -0.2},
        histories=first.histories,
        window=2,
        update=2,
    )
    assert second.selected_count == 11
    assert all(item.standardized_improvement is None for item in second.evidence)


def test_stratified_far_right_evidence_participates_in_full_radius_fit():
    base = {8: 0.4, 9: 0.2, 10: 0.0, 11: -0.2}
    growth_supporting = _decision(
        v055.choose_plastic_depth_count_with_stratified_sen_kendall_v055,
        {**base, 12: -10000.0, 13: -20000.0},
    )
    growth_opposing = _decision(
        v055.choose_plastic_depth_count_with_stratified_sen_kendall_v055,
        {**base, 12: 10000.0, 13: 20000.0},
    )
    assert growth_supporting.selected_count == 11
    assert growth_opposing.selected_count == 10


def test_stratified_adjacent_check_can_veto_broad_right_slope():
    decision = _decision(
        v055.choose_plastic_depth_count_with_stratified_sen_kendall_v055,
        {6: 10.0, 7: 8.0, 8: 6.0, 9: 2.0, 10: 0.0, 11: 0.1},
    )
    assert decision.selected_count == 10


def test_bootstrap_rows_never_drive_either_v055_algorithm():
    report = _score_report({9: 0.2, 10: 0.0, 11: -0.2}, bootstrap=True)
    for function in (
        v055.choose_plastic_depth_count_with_tsk_lra_v055,
        v055.choose_plastic_depth_count_with_stratified_sen_kendall_v055,
    ):
        decision = function(
            current_count=10,
            score_report=report,
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


def test_runtime_selector_is_v055_not_bootstrap_fallback():
    assert trainer_step.choose_plastic_depth_count_with_mad is v055._choose_count_v055


def test_provisional_timing_fit_uses_one_count_without_decision_fallback(monkeypatch):
    monkeypatch.setenv(tsk._ALGORITHM_ENV, v055.STRATIFIED_ALGORITHM)

    class Item:
        def item(self):
            return 0.5

    trainer = SimpleNamespace()
    trainer._plastic_depth_lattice = lambda: SimpleNamespace(optimizer_step_time_ema=Item())
    setattr(
        trainer,
        wall_time._STATE_ATTRIBUTE,
        {
            "timing_n": 4,
            "timing_sum_x": 40.0,
            "timing_sum_y": 20.0,
            "timing_sum_x2": 400.0,
            "timing_sum_xy": 200.0,
            "timing_sum_y2": 100.0,
            "timing_distinct_counts": {10},
        },
    )
    fit = v055._timing_fit_v055(trainer)
    assert fit is not None
    assert fit["provisional"] == 1.0
    assert fit["intercept"] == pytest.approx(0.5)
    assert fit["slope"] == pytest.approx(0.45)


def test_console_removes_score_z_and_uses_therefore(monkeypatch):
    monkeypatch.setenv(tsk._ALGORITHM_ENV, v055.LRA_ALGORITHM)
    monkeypatch.setattr(
        v055,
        "_ORIGINAL_FORMAT_PROGRESS_LINE",
        lambda *_args, **_kwargs: (
            "P2 probe_Δloss [L-1 ... L+1] = [-0.1, 4.0, +0.1]  "
            "score_z [L-1 ... L+1] = [+1.0, -, -2.0]  "
            "ts=-0.100s/layer tau=-1.00  ⇩|⇧|? =[0/2/0]/2=>R (P1,2)"
        ),
    )
    rendered = v055._format_progress_line_v055(
        "run", "optimizer_progress",
        {
            "plastic_v055_algorithm": v055.LRA_ALGORITHM,
            "plastic_v055_sen": -0.1,
            "plastic_v055_ken": -1.0,
            "plastic_v055_adj": -0.2,
            "plastic_v055_conclusion": "R",
            "plastic_v055_selected_count": 11,
            "plastic_v055_current_count": 10,
            "plastic_v055_probe_ids": (1, 2),
            "plastic_v055_left_votes": 0,
            "plastic_v055_right_votes": 2,
            "plastic_v055_ambiguous_votes": 0,
            "plastic_v055_vote_total": 2,
        },
    )
    assert "probe_Δloss" in rendered
    assert "score_z" not in rendered
    assert "ts=" not in rendered and "tau=" not in rendered
    assert "sen=-0.100 ken=-1.00 adj=-0.200" in rendered
    assert "∴ ▼|▲|? =[0/2/0]/2 ∴ ▲ (P1,2)" in rendered
# ^^^ THOG
