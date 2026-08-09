# vvv THOG
from __future__ import annotations

import os

import pytest

from sheet import plastic_depth_sen_kendall_v055_patch as v055
from sheet import plastic_depth_theil_sen_kendall_patch as tsk
from sheet import trainer_step


@pytest.fixture(autouse=True)
def _isolated_sen_kendall_runtime_environment():
    previous_algorithm = os.environ.pop(tsk._ALGORITHM_ENV, None)
    previous_tau = os.environ.pop(tsk._TAU_ENV, None)
    try:
        yield
    finally:
        os.environ.pop(tsk._ALGORITHM_ENV, None)
        os.environ.pop(tsk._TAU_ENV, None)
        if previous_algorithm is not None:
            os.environ[tsk._ALGORITHM_ENV] = previous_algorithm
        if previous_tau is not None:
            os.environ[tsk._TAU_ENV] = previous_tau


def test_sen_kendall_algorithms_are_opt_in_and_legacy_is_default(monkeypatch):
    monkeypatch.delenv(tsk._ALGORITHM_ENV, raising=False)
    assert v055._runtime_algorithm() == tsk.LEGACY_DIRECTIONAL_ALGORITHM
    for algorithm in (v055.LRA_ALGORITHM, v055.STRATIFIED_ALGORITHM):
        monkeypatch.setenv(tsk._ALGORITHM_ENV, algorithm)
        assert v055._runtime_algorithm() == algorithm


def test_retired_gradient_algorithm_name_is_rejected(monkeypatch):
    monkeypatch.setenv(
        tsk._ALGORITHM_ENV,
        "wall_time__gradient__theil_sen_kendall_slope_tau",
    )
    with pytest.raises(ValueError, match="renamed in PLASTIC v0.55"):
        v055._runtime_algorithm()


def test_v055_retargets_v054_compatibility_constant_to_lra():
    assert tsk.GRADIENT_ALGORITHM == v055.LRA_ALGORITHM
    assert v055.FIXED_MINIMUM_ABSOLUTE_KENDALL_TAU == pytest.approx(0.5)
    assert tsk._runtime_minimum_absolute_kendall_tau() == pytest.approx(0.5)


def test_runtime_selector_surface_is_owned_by_v055_not_bootstrap_fallback():
    assert trainer_step.choose_plastic_depth_count_with_mad is v055._choose_count_v055


def test_v055_lra_classifier_is_deterministic_for_identical_ddp_inputs():
    score_report = (
        {"active_layers": 9, "feasible": True, "score": 0.2, "wall_time_algorithm": "wall_time_equivalent_time_gain", "wall_time_bootstrap": False},
        {"active_layers": 10, "feasible": True, "score": 0.0, "wall_time_algorithm": "wall_time_equivalent_time_gain", "wall_time_bootstrap": False},
        {"active_layers": 11, "feasible": True, "score": -0.2, "wall_time_algorithm": "wall_time_equivalent_time_gain", "wall_time_bootstrap": False},
    )
    first = tsk._gradient_probe_classification(
        current_count=10,
        score_report=score_report,
        minimum_absolute_kendall_tau=v055.FIXED_MINIMUM_ABSOLUTE_KENDALL_TAU,
    )
    second = tsk._gradient_probe_classification(
        current_count=10,
        score_report=score_report,
        minimum_absolute_kendall_tau=v055.FIXED_MINIMUM_ABSOLUTE_KENDALL_TAU,
    )
    assert first == second
# ^^^ THOG
