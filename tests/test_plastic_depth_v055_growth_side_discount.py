from __future__ import annotations

import argparse

import pytest

from sheet import plastic_depth_sen_kendall_v055_patch as v055
from sheet import plastic_depth_theil_sen_kendall_patch as tsk
from sheet import plastic_depth_v055_growth_side_discount_patch as growth
from sheet import plastic_depth_wall_time_equivalent_time_gain_patch as wall_time


def _score(count: int, score: float) -> dict:
    return {
        "active_layers": count,
        "score": score,
        "feasible": True,
        "wall_time_algorithm": wall_time.WALL_TIME_ALGORITHM,
        "wall_time_bootstrap": False,
    }


def test_growth_discount_only_attenuates_beneficial_right_side(monkeypatch) -> None:
    monkeypatch.setenv(growth._RUNTIME_ENV, "0.5")
    raw, fitted = growth.growth_discounted_score_map(
        current_count=10,
        score_report=(
            _score(8, 8.0),
            _score(9, 9.0),
            _score(10, 10.0),
            _score(11, 8.0),
            _score(12, 12.0),
        ),
    )

    assert raw == {8: 8.0, 9: 9.0, 10: 10.0, 11: 8.0, 12: 12.0}
    assert fitted[8] == pytest.approx(8.0)
    assert fitted[9] == pytest.approx(9.0)
    assert fitted[10] == pytest.approx(10.0)
    assert fitted[11] == pytest.approx(9.0)
    assert fitted[12] == pytest.approx(12.0)


def test_lra_tsk_fit_uses_every_feasible_radius_point(monkeypatch) -> None:
    monkeypatch.setenv(tsk._ALGORITHM_ENV, v055.LRA_ALGORITHM)
    monkeypatch.setenv(growth._RUNTIME_ENV, "0.75")
    report = tsk._gradient_probe_classification(
        current_count=10,
        score_report=tuple(_score(count, 20.0 - count) for count in range(7, 14)),
        minimum_absolute_kendall_tau=0.5,
    )

    assert report["decision_fit_offsets"] == (-3, -2, -1, 0, 1, 2, 3)
    assert report["fit_point_count"] == 7
    assert report["growth_side_discount"] == pytest.approx(0.75)


def test_stratified_histories_store_discounted_fit_but_raw_adjacent(monkeypatch) -> None:
    monkeypatch.setenv(tsk._ALGORITHM_ENV, v055.STRATIFIED_ALGORITHM)
    monkeypatch.setenv(growth._RUNTIME_ENV, "0.5")
    decision = growth.choose_plastic_depth_count_with_stratified_sen_kendall_growth_discount(
        current_count=10,
        score_report=(
            _score(8, 13.0),
            _score(9, 11.0),
            _score(10, 10.0),
            _score(11, 10.5),
            _score(12, 8.0),
        ),
        histories={},
        noise_window=1,
        noise_lambda=0.0,
        update_number=1,
        last_count_change_update=-1,
        update_brake=0,
    )

    assert decision.selected_count == 10
    assert decision.histories[v055._stratified_history_key(10, 2)] == pytest.approx((-1.0,))
    assert decision.histories[growth._raw_adjacent_history_key(10, 1)] == pytest.approx((0.5,))
    assert decision.histories[v055._stratified_history_key(10, -1)] == pytest.approx((1.0,))


def test_public_growth_discount_option_is_parsed(monkeypatch) -> None:
    monkeypatch.delenv(growth._RUNTIME_ENV, raising=False)
    monkeypatch.delenv(growth._EXPLICIT_ENV, raising=False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--plastic__enabled", action="store_true")
    namespace = parser.parse_args(
        [
            "--plastic__enabled",
            "--plastic__layer_count_decision_algorithm__growth_side_discount",
            "0.6",
        ]
    )

    assert namespace.plastic__enabled is True
    assert namespace.plastic__layer_count_decision_algorithm__growth_side_discount == pytest.approx(0.6)
    assert growth._runtime_growth_side_discount() == pytest.approx(0.6)


def test_growth_discount_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError):
        growth._validate_growth_side_discount(-0.01)
    with pytest.raises(ValueError):
        growth._validate_growth_side_discount(1.01)
