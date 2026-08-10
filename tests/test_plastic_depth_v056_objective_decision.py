from __future__ import annotations

from types import SimpleNamespace

import pytest

from sheet import plastic_depth as plastic_depth
from sheet import plastic_depth_directional_coherence_patch as directional
from sheet import plastic_depth_theil_sen_kendall_patch as tsk
from sheet import plastic_depth_v055_growth_side_discount_patch as growth
from sheet import plastic_depth_v056_objective_decision_patch as v056
from sheet import plastic_depth_wall_time_equivalent_time_gain_patch as wall_time
from tests.test_plastic_depth import plastic_training_config


OBJECTIVES = (
    "lowest_loss",
    "layer_efficiency",
    "relative_training_wall_time",
    "memory_budget",
)


def _score_report(scores: dict[int, float], *, bootstrap: bool = False):
    return tuple(
        {
            "active_layers": count,
            "validation_loss": score,
            "feasible": True,
            "score": score,
            "wall_time_bootstrap": bootstrap,
        }
        for count, score in sorted(scores.items())
    )


def _selector_kwargs(scores: dict[int, float]):
    return {
        "current_count": 10,
        "score_report": _score_report(scores),
        "histories": {},
        "noise_window": 1,
        "noise_lambda": 0.0,
        "update_number": 1,
        "last_count_change_update": -1,
        "update_brake": 0,
        "max_step": 1,
        "minimum_observations": None,
    }


def test_v056_algorithm_names_are_objective_neutral(monkeypatch) -> None:
    assert v056.LRA_ALGORITHM == "theil_sen_kendall_LRA"
    assert v056.STRATIFIED_ALGORITHM == "sen_kendall__tau__stratified"
    for retired in (
        "wall_time__gradient__theil_sen_kendall_slope_tau",
        "wall_time__theil_sen_kendall_LRA",
        "wall_time__sen_kendall__tau__stratified",
    ):
        monkeypatch.setenv(tsk._ALGORITHM_ENV, retired)
        with pytest.raises(ValueError, match="retired in PLASTIC v0.56"):
            v056._runtime_algorithm()


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("algorithm", v056.DECISION_ALGORITHMS)
def test_all_objective_decision_combinations_validate(monkeypatch, objective, algorithm) -> None:
    monkeypatch.setenv(tsk._ALGORITHM_ENV, algorithm)
    config = plastic_training_config(
        plastic__layers_to_sample=None,
        plastic__do_learn_layer_count=True,
        plastic__initial_layer_count=10,
        plastic__max_permitted_layers=20,
        plastic__layer_count_objective=objective,
        plastic__layer_memory_budget_gib=12.0 if objective == "memory_budget" else None,
        plastic__layer_count__max_allowable_layer_change=1,
    )
    v056._validate_v056_config(config)
    assert config.plastic__layer_count_objective == objective
    assert v056._runtime_algorithm() == algorithm


def test_objective_score_map_uses_generic_feasible_score_and_skips_bootstrap() -> None:
    report = (
        {"active_layers": 8, "feasible": True, "score": 1.25},
        {"active_layers": 9, "feasible": False, "score": -100.0},
        {"active_layers": 10, "feasible": True, "score": 1.0},
        {"active_layers": 11, "feasible": True, "score": 0.9, "wall_time_bootstrap": True},
        {"active_layers": 12, "feasible": True, "score": float("inf")},
    )
    assert v056.objective_score_map(report) == {8: 1.25, 10: 1.0}


def test_memory_budget_base_scorer_excludes_over_budget_candidate() -> None:
    within = plastic_depth.PlasticDepthCandidateMeasurement(
        active_layers=10,
        validation_loss=5.0,
        peak_allocated_gib=7.0,
    )
    over = plastic_depth.PlasticDepthCandidateMeasurement(
        active_layers=11,
        validation_loss=4.0,
        peak_allocated_gib=9.0,
    )
    _, report = plastic_depth.choose_plastic_depth_candidate(
        (within, over),
        objective="memory_budget",
        maximum_layers=20,
        cost_weight=0.02,
        reference_training_time=None,
        memory_budget_gib=8.0,
    )
    scores = v056.objective_score_map(report)
    assert scores == {10: pytest.approx(5.0)}


def test_growth_discount_applies_to_generic_objective_score(monkeypatch) -> None:
    monkeypatch.setenv(growth._RUNTIME_ENV, "0.5")
    raw, fitted = growth.growth_discounted_score_map(
        current_count=10,
        score_report=_score_report({9: 12.0, 10: 10.0, 11: 8.0, 12: 12.0}),
    )
    assert raw == {9: 12.0, 10: 10.0, 11: 8.0, 12: 12.0}
    assert fitted[9] == pytest.approx(12.0)
    assert fitted[10] == pytest.approx(10.0)
    assert fitted[11] == pytest.approx(9.0)
    assert fitted[12] == pytest.approx(12.0)


def test_tsk_path_never_calls_legacy_direction_calculation(monkeypatch) -> None:
    monkeypatch.setenv(tsk._ALGORITHM_ENV, v056.STRATIFIED_ALGORITHM)

    def forbidden(**_kwargs):
        raise AssertionError("legacy directional machinery executed under TSK")

    monkeypatch.setattr(v056, "_ORIGINAL_UPDATED_HISTORIES_AND_DIRECTION", forbidden)
    histories, report = v056._updated_histories_and_direction_v056(
        current_count=10,
        score_report=_score_report({9: 1.0, 10: 0.0, 11: -1.0}),
        histories={"10:@SK_STRAT_V055:+0": (0.0,)},
        noise_window=2,
        extrapolation_weight=0.8,
    )
    assert histories == {"10:@SK_STRAT_V055:+0": (0.0,)}
    assert report["algorithm"] == v056.STRATIFIED_ALGORITHM


def test_directional_coherence_retains_legacy_robust_z_path(monkeypatch) -> None:
    monkeypatch.setenv(tsk._ALGORITHM_ENV, tsk.LEGACY_DIRECTIONAL_ALGORITHM)
    calls = {"count": 0}

    def legacy(**_kwargs):
        calls["count"] += 1
        return {}, {"algorithm": "directional_coherence"}

    monkeypatch.setattr(v056, "_ORIGINAL_UPDATED_HISTORIES_AND_DIRECTION", legacy)
    histories, report = v056._updated_histories_and_direction_v056(
        current_count=10,
        score_report=_score_report({9: 1.0, 10: 0.0, 11: -1.0}),
        histories={},
        noise_window=2,
        extrapolation_weight=0.8,
    )
    assert calls["count"] == 1
    assert histories == {}
    assert report["algorithm"] == "directional_coherence"


def test_tsk_evidence_contains_no_robust_z_statistics(monkeypatch) -> None:
    monkeypatch.setenv(tsk._ALGORITHM_ENV, v056.STRATIFIED_ALGORITHM)
    decision = growth.choose_plastic_depth_count_with_stratified_sen_kendall_growth_discount(
        **_selector_kwargs({8: 0.4, 9: 0.2, 10: 0.0, 11: -0.2, 12: -0.4})
    )
    assert decision.selected_count == 11
    assert all(item.median is None for item in decision.evidence)
    assert all(item.mad is None for item in decision.evidence)
    assert all(item.sigma is None for item in decision.evidence)
    assert all(item.standardized_improvement is None for item in decision.evidence)


def test_tsk_probe_report_never_reads_score_evidence(monkeypatch) -> None:
    monkeypatch.setenv(tsk._ALGORITHM_ENV, v056.LRA_ALGORITHM)

    class ForbiddenEvidence:
        def __iter__(self):
            raise AssertionError("score_z evidence traversed under TSK")

    event = SimpleNamespace(
        name="plastic_depth_count_decision",
        payload={
            "previous_active_layers": 10,
            "selected_active_layers": 10,
            "objective": "lowest_loss",
            "decision_candidate_counts": (9, 10, 11),
            "candidates": (
                {"active_layers": 9, "validation_loss": 5.1},
                {"active_layers": 10, "validation_loss": 5.0},
                {"active_layers": 11, "validation_loss": 4.9},
            ),
            "score_evidence": ForbiddenEvidence(),
        },
    )
    trainer = SimpleNamespace(
        config=SimpleNamespace(plastic__do_learn_layer_count=True),
        events=[event],
    )
    report = v056._latest_probe_report_v056(trainer)
    assert report is not None
    assert report["score_z"] is None
    assert report["offsets"] == (-1, 0, 1)


def test_wall_time_score_units_remain_objective_specific(monkeypatch) -> None:
    trainer = SimpleNamespace(
        config=SimpleNamespace(plastic__layer_count_objective="relative_training_wall_time")
    )
    assert v056._active_objective(trainer) == "relative_training_wall_time"
    assert v056._score_units("relative_training_wall_time") == "equivalent_seconds"
    assert v056._score_units("lowest_loss") == "loss"
    assert v056._score_units("layer_efficiency") == "layer_efficiency_score"
