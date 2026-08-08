# vvv THOG
from __future__ import annotations

import math
import re
from types import SimpleNamespace

import constants
from sheet import plastic_depth_wall_time_equivalent_time_gain_patch as patch
from sheet.plastic_depth import PlasticDepthCandidateMeasurement


def _plain(value: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", value)


def test_linear_timing_fit_recovers_constant_marginal_layer_cost() -> None:
    points = (
        (8.0, 1.8),
        (8.0, 1.8),
        (12.0, 2.2),
        (12.0, 2.2),
        (16.0, 2.6),
        (16.0, 2.6),
    )
    fit = patch._linear_fit_points(points)
    assert fit is not None
    assert math.isclose(fit["slope"], 0.1, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(fit["intercept"], 1.0, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(fit["r_squared"], 1.0, rel_tol=0.0, abs_tol=1.0e-12)


def test_equivalent_time_gain_haircuts_only_alleged_benefit() -> None:
    details = patch._equivalent_time_score(
        current_probe_loss=4.0,
        candidate_probe_loss=3.99,
        loss_improvement_rate=0.002,
        predicted_current_update_seconds=2.0,
        predicted_candidate_update_seconds=2.05,
        horizon_updates=7,
        discount=0.9,
    )
    assert math.isclose(details["raw_equivalent_time_gain_seconds"], 5.0)
    assert math.isclose(details["credited_equivalent_time_gain_seconds"], 4.5)
    assert math.isclose(details["extra_wall_time_seconds"], 0.35)
    assert math.isclose(details["score_seconds"], -4.15)

    harm = patch._equivalent_time_score(
        current_probe_loss=4.0,
        candidate_probe_loss=4.01,
        loss_improvement_rate=0.002,
        predicted_current_update_seconds=2.0,
        predicted_candidate_update_seconds=1.95,
        horizon_updates=7,
        discount=0.9,
    )
    assert math.isclose(harm["raw_equivalent_time_gain_seconds"], -5.0)
    assert math.isclose(harm["credited_equivalent_time_gain_seconds"], -5.0)


def test_wall_time_horizon_respects_history_and_brake() -> None:
    config = SimpleNamespace(
        plastic__layer_count_probe__probe_every_n_steps=4,
        plastic__layer_count_probe__window_size_as_number_of_probes=2,
        plastic__layer_count_update_brake=0,
    )
    assert patch._wall_time_horizon_updates(config) == 7

    config.plastic__layer_count_update_brake = 30
    assert patch._wall_time_horizon_updates(config) == 31


def test_ready_equivalent_time_algorithm_clears_bootstrap_histories() -> None:
    config = SimpleNamespace(
        plastic__layer_count_probe__probe_every_n_steps=4,
        plastic__layer_count_probe__window_size_as_number_of_probes=2,
        plastic__layer_count_update_brake=0,
        warmup_updates=0,
    )
    trainer_state = SimpleNamespace(
        completed_updates=16,
        plastic_depth_probe_histories={"9:+1": [-0.1]},
    )
    lattice = SimpleNamespace(current_active_layers=9)
    trainer = SimpleNamespace(config=config, state=trainer_state)
    trainer._plastic_depth_lattice = lambda: lattice
    trainer._plastic_depth_inline_update_context = {"current_count": 9}

    for update in range(1, 17):
        count = 8 if update <= 8 else 9
        elapsed = 1.8 if count == 8 else 1.9
        patch._record_ordinary_update_sample(
            trainer,
            update_number=update,
            active_layers=count,
            elapsed_seconds=elapsed,
            training_loss=4.0 - 0.01 * update,
        )

    measurements = (
        PlasticDepthCandidateMeasurement(active_layers=8, validation_loss=4.003),
        PlasticDepthCandidateMeasurement(active_layers=9, validation_loss=4.000),
        PlasticDepthCandidateMeasurement(active_layers=10, validation_loss=3.995),
    )
    selected, report = patch._choose_wall_time_equivalent_time_gain(
        trainer,
        measurements,
    )
    assert selected.active_layers in {8, 9, 10}
    assert trainer.state.plastic_depth_probe_histories == {}
    assert all(item["wall_time_algorithm"] == patch.WALL_TIME_ALGORITHM for item in report)
    assert all(item["wall_time_bootstrap"] is False for item in report)
    assert all(item["wall_time_discount"] == 0.9 for item in report)
    current = next(item for item in report if item["active_layers"] == 9)
    assert math.isclose(float(current["score"]), 0.0, abs_tol=1.0e-12)


def test_console_direction_symbols_gradient_label_and_sample_width() -> None:
    neutral = patch._finalize_console_v0531(
        "T 1 grad norm= 0.250 layers = 8\tsampled = [1.0]  "
        "L/R/A=[1/0/2]/3=>stet"
    )
    assert "g nrm= 0.250" in neutral
    assert "layers = 8   \tsampled =" in neutral
    assert "↓|↑|? =[1/0/2]/3=>●" in neutral

    left = patch._finalize_console_v0531("L/R/A=[2/0/1]/3=>L")
    assert _plain(left) == "↓|↑|? =[2/0/1]/3=>↓"
    assert f"{constants.BOLD}{constants.YELLOW}↓{constants.R}" in left

    right = patch._finalize_console_v0531("L/R/A=[0/2/1]/3=>R")
    assert _plain(right) == "↓|↑|? =[0/2/1]/3=>↑"
    assert f"{constants.BOLD}{constants.YELLOW}↑{constants.R}" in right


def test_score_z_fixed_decimal_values_are_zero_padded_but_scientific_is_untouched() -> None:
    line = (
        "score_z [L-5 ... L+5] = "
        "[-5.20, -4.72, -31.09, +0.42, -1.22, -1.00e-04]"
    )
    rendered = patch._finalize_console_v0531(line)
    assert (
        rendered
        == "score_z [L-5 ... L+5] = "
        "[-05.20, -04.72, -31.09, +00.42, -01.22, -1.00e-04]"
    )


def test_wall_time_telemetry_is_absent_when_plastic_is_disabled() -> None:
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            plastic__enabled=False,
            plastic__do_learn_layer_count=False,
        )
    )
    assert patch._wall_time_telemetry_values(trainer, completed_update=1) == {}


def test_plastic_timing_synchronize_is_removed_for_non_memory_goals() -> None:
    fake = SimpleNamespace(
        config=SimpleNamespace(plastic__layer_count_objective="relative_training_wall_time")
    )
    assert patch._plastic_depth_device_synchronize_v0531(fake) is None
# ^^^ THOG
