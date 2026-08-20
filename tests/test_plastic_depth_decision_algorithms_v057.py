from __future__ import annotations

from types import SimpleNamespace

import pytest

from sheet import plastic_depth_decision_algorithms_v057_patch as v057
from sheet import plastic_depth_theil_sen_kendall_patch as tsk


def _score_report(scores: dict[int, float]):
    return tuple(
        {
            "active_layers": count,
            "validation_loss": loss,
            "feasible": True,
            "score": loss,
        }
        for count, loss in sorted(scores.items())
    )


def _selector_kwargs(scores: dict[int, float], **overrides):
    values = {
        "current_count": 10,
        "score_report": _score_report(scores),
        "histories": {},
        "noise_window": 1,
        "noise_lambda": 0.0,
        "extrapolation_weight": 0.8,
        "update_number": 10,
        "last_count_change_update": -1,
        "update_brake": 0,
        "max_step": 1,
        "minimum_observations": None,
    }
    values.update(overrides)
    return values


def test_algorithm_vocabulary_contains_existing_and_new_modes() -> None:
    assert tsk.LEGACY_DIRECTIONAL_ALGORITHM in v057.DECISION_ALGORITHMS
    assert "theil_sen_kendall_LRA" in v057.DECISION_ALGORITHMS
    assert "sen_kendall__tau__stratified" in v057.DECISION_ALGORITHMS
    assert v057.SEN_ALGORITHM in v057.DECISION_ALGORITHMS
    assert v057.KENDALL_ALGORITHM in v057.DECISION_ALGORITHMS
    assert v057.JUMP_TO_LOWEST_LOSS_ALGORITHM in v057.DECISION_ALGORITHMS


def test_retired_and_unknown_algorithm_names_fail_without_recursion(monkeypatch) -> None:
    monkeypatch.setenv(
        tsk._ALGORITHM_ENV,
        "wall_time__sen_kendall__tau__stratified",
    )
    with pytest.raises(ValueError, match="retired in PLASTIC v0.56"):
        v057._runtime_algorithm()
    with pytest.raises(ValueError, match="must be one of"):
        v057._set_runtime_algorithm("not_an_algorithm")


def test_standalone_sen_uses_its_threshold_and_raw_adjacent_gate(monkeypatch) -> None:
    monkeypatch.setenv(tsk._ALGORITHM_ENV, v057.SEN_ALGORITHM)
    monkeypatch.setenv(v057._SEN_THRESHOLD_ENV, "0.1")
    decision = v057._choose_count_v057(
        **_selector_kwargs({8: 10.4, 9: 10.2, 10: 10.0, 11: 9.8, 12: 9.6})
    )
    assert decision.selected_count == 11

    monkeypatch.setenv(v057._SEN_THRESHOLD_ENV, "0.25")
    held = v057._choose_count_v057(
        **_selector_kwargs({8: 10.4, 9: 10.2, 10: 10.0, 11: 9.8, 12: 9.6})
    )
    assert held.selected_count == 10


def test_standalone_kendall_uses_its_threshold(monkeypatch) -> None:
    monkeypatch.setenv(tsk._ALGORITHM_ENV, v057.KENDALL_ALGORITHM)
    monkeypatch.setenv(v057._KENDALL_THRESHOLD_ENV, "0.8")
    decision = v057._choose_count_v057(
        **_selector_kwargs({8: 10.4, 9: 10.2, 10: 10.0, 11: 9.8, 12: 9.6})
    )
    assert decision.selected_count == 11

    monkeypatch.setenv(v057._KENDALL_THRESHOLD_ENV, "1.0")
    held = v057._choose_count_v057(
        **_selector_kwargs({8: 10.4, 9: 9.9, 10: 10.0, 11: 9.8, 12: 9.7})
    )
    assert held.selected_count == 10


def test_standalone_modes_retain_update_brake(monkeypatch) -> None:
    monkeypatch.setenv(tsk._ALGORITHM_ENV, v057.SEN_ALGORITHM)
    monkeypatch.setenv(v057._SEN_THRESHOLD_ENV, "0")
    decision = v057._choose_count_v057(
        **_selector_kwargs(
            {8: 10.4, 9: 10.2, 10: 10.0, 11: 9.8, 12: 9.6},
            update_number=12,
            last_count_change_update=10,
            update_brake=3,
        )
    )
    assert decision.brake_active is True
    assert decision.selected_count == 10


def test_jump_uses_raw_loss_not_configured_objective_score(monkeypatch) -> None:
    monkeypatch.setenv(tsk._ALGORITHM_ENV, v057.JUMP_TO_LOWEST_LOSS_ALGORITHM)
    monkeypatch.setenv(v057._JUMP_THRESHOLD_ENV, "1.0")
    report = (
        {"active_layers": 8, "validation_loss": 9.7, "feasible": False, "score": 100.0},
        {"active_layers": 10, "validation_loss": 10.0, "feasible": True, "score": 0.0},
        {"active_layers": 12, "validation_loss": 9.6, "feasible": False, "score": 200.0},
    )
    decision = v057._choose_count_v057(
        **_selector_kwargs({}, score_report=report, max_step=1)
    )
    assert decision.selected_count == 12


def test_jump_threshold_tie_and_direct_distance_are_deterministic(monkeypatch) -> None:
    monkeypatch.setenv(tsk._ALGORITHM_ENV, v057.JUMP_TO_LOWEST_LOSS_ALGORITHM)
    monkeypatch.setenv(v057._JUMP_THRESHOLD_ENV, "5.0")
    held = v057._choose_count_v057(
        **_selector_kwargs({8: 9.51, 10: 10.0, 12: 9.51}, max_step=1)
    )
    assert held.selected_count == 10

    monkeypatch.setenv(v057._JUMP_THRESHOLD_ENV, "4.9")
    selected = v057._choose_count_v057(
        **_selector_kwargs({8: 9.51, 10: 10.0, 12: 9.51}, max_step=1)
    )
    assert selected.selected_count == 8


def test_jump_update_brake_is_still_a_safety_gate(monkeypatch) -> None:
    monkeypatch.setenv(tsk._ALGORITHM_ENV, v057.JUMP_TO_LOWEST_LOSS_ALGORITHM)
    monkeypatch.setenv(v057._JUMP_THRESHOLD_ENV, "0")
    decision = v057._choose_count_v057(
        **_selector_kwargs(
            {8: 8.0, 10: 10.0, 12: 9.0},
            update_number=12,
            last_count_change_update=10,
            update_brake=3,
        )
    )
    assert decision.brake_active is True
    assert decision.selected_count == 10


def test_threshold_parser_and_persistence_are_algorithm_specific(monkeypatch) -> None:
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--plastic__enabled", action="store_true")
    namespace = parser.parse_args(
        [
            "--plastic__enabled",
            v057._SEN_THRESHOLD_OPTION,
            "0.025",
        ]
    )
    assert getattr(namespace, v057.SEN_THRESHOLD_KEY) == pytest.approx(0.025)

    monkeypatch.setenv(tsk._ALGORITHM_ENV, v057.SEN_ALGORITHM)
    values = v057._persistent_v057(
        lambda _config: {"existing": 1},
        SimpleNamespace(plastic__enabled=True),
    )
    assert values[tsk._ALGORITHM_KEY] == v057.SEN_ALGORITHM
    assert values[v057.SEN_THRESHOLD_KEY] == pytest.approx(0.025)
    assert v057.KENDALL_THRESHOLD_KEY not in values
    assert v057.JUMP_THRESHOLD_KEY not in values


def test_sitecustomize_preserves_exact_public_double_underscore_names() -> None:
    import sitecustomize

    assert sitecustomize._normalise_long_option("--plastic__enabled") == "--plastic__enabled"
    assert (
        sitecustomize._normalise_long_option(
            "--instrumentation__delta_loss_v_layer_heatmap__destination=local"
        )
        == "--instrumentation__delta_loss_v_layer_heatmap__destination=local"
    )
    assert sitecustomize._normalise_long_option("--select_depth") == "--select-depth"


def test_jump_validation_allows_nonunit_max_step(monkeypatch) -> None:
    monkeypatch.setenv(tsk._ALGORITHM_ENV, v057.JUMP_TO_LOWEST_LOSS_ALGORITHM)
    v057._validate_v057_config(
        SimpleNamespace(
            plastic__enabled=True,
            plastic__do_learn_layer_count=True,
            plastic__layer_count__max_allowable_layer_change=7,
        )
    )


def test_standalone_validation_requires_unit_max_step(monkeypatch) -> None:
    monkeypatch.setenv(tsk._ALGORITHM_ENV, v057.SEN_ALGORITHM)
    with pytest.raises(ValueError, match="max_allowable_layer_change=1"):
        v057._validate_v057_config(
            SimpleNamespace(
                plastic__enabled=True,
                plastic__do_learn_layer_count=True,
                plastic__layer_count__max_allowable_layer_change=2,
            )
        )
