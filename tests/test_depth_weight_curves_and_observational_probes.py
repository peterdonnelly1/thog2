from __future__ import annotations

import argparse
from types import SimpleNamespace

import constants
import pytest
import torch

from sheet import depth_observational_probe_wandb_patch as observational_wandb
from sheet import depth_weight_curves_and_observational_probes_patch as depth_curves
from sheet.depth_trajectory import DepthTrajectory
from sheet.geometry import SheetGeometryConfig
from sheet.semantic_materializer import MLP_EXPANSION_WEIGHT


# vvv THOG small public DEPTH fixture exercises the real coefficient representation without allocating a training-size model
def _trajectory() -> DepthTrajectory:
    geometry = SheetGeometryConfig(
        n_layer=4,
        n_embd=8,
        n_head=2,
        depth_order=4,
        base_row_order=1,
        bias=True,
    )
    return DepthTrajectory(
        geometry,
        runtime_dtype=torch.float32,
        depth_compress_layer_norm_and_bias=False,
    )
# ^^^ THOG


def _trainer(trajectory: DepthTrajectory, *, run_seed: int = 1234, cadence=5, learn=False):
    return SimpleNamespace(
        raw_model=SimpleNamespace(trajectory=trajectory),
        config=SimpleNamespace(
            model_seed=run_seed,
            plastic__do_learn_layer_count=learn,
            plastic__layer_count_probe__probe_every_n_steps=cadence,
        ),
    )


def _telemetry(name: str):
    return SimpleNamespace(name=name, group="test-group")


# vvv THOG instrumentation CLI controls are discoverable yet remain execution-only environment controls rather than model identity fields
def test_depth_weight_curve_cli_controls_publish_environment(monkeypatch) -> None:
    for suffix in (
        "SCALAR_WEIGHTS_PER_MATRIX",
        "DEPTH_EVALUATION_POINTS",
        "TIME_MODE",
        "HISTORY_LENGTH",
        "LOG_EVERY_N_STEPS",
        "SAME_COORDINATES_ALL_RUNS",
    ):
        monkeypatch.delenv(depth_curves._environment_name(suffix), raising=False)

    parser = argparse.ArgumentParser(add_help=False)
    values, remaining = parser.parse_known_args(
        [
            "--instrumentation__depth_weight_curves__scalar_weights_per_matrix",
            "5",
            "--instrumentation__depth_weight_curves__depth_evaluation_points",
            "64",
            "--instrumentation__depth_weight_curves__time_mode",
            "accumulate",
            "--instrumentation__depth_weight_curves__history_length",
            "7",
            "--instrumentation__depth_weight_curves__log_every_n_steps",
            "11",
            "--instrumentation__depth_weight_curves__same_coordinates_all_runs",
        ]
    )

    assert remaining == []
    assert values.instrumentation__depth_weight_curves__scalar_weights_per_matrix == 5
    assert depth_curves._scalar_weights_per_matrix() == 5
    assert depth_curves._depth_evaluation_points() == 64
    assert depth_curves._time_mode() == "accumulate"
    assert depth_curves._history_length() == 7
    assert depth_curves._log_every_n_steps() == 11
    assert depth_curves._same_coordinates_all_runs() is True
# ^^^ THOG


# vvv THOG train_OWT normalizes underscore runs to single hyphens before forwarding, so the hidden aliases must accept the exact wrapper-produced spellings
def test_depth_weight_curve_cli_controls_accept_wrapper_normalized_spellings(monkeypatch) -> None:
    for suffix in (
        "SCALAR_WEIGHTS_PER_MATRIX",
        "DEPTH_EVALUATION_POINTS",
        "TIME_MODE",
        "HISTORY_LENGTH",
        "LOG_EVERY_N_STEPS",
        "SAME_COORDINATES_ALL_RUNS",
    ):
        monkeypatch.delenv(depth_curves._environment_name(suffix), raising=False)

    parser = argparse.ArgumentParser(add_help=False)
    values, remaining = parser.parse_known_args(
        [
            "--instrumentation-depth-weight-curves-scalar-weights-per-matrix",
            "1",
            "--instrumentation-depth-weight-curves-depth-evaluation-points",
            "256",
            "--instrumentation-depth-weight-curves-time-mode",
            "accumulate",
            "--instrumentation-depth-weight-curves-history-length",
            "12",
            "--instrumentation-depth-weight-curves-log-every-n-steps",
            "1",
            "--instrumentation-depth-weight-curves-same-coordinates-all-runs",
        ]
    )

    assert remaining == []
    assert values.instrumentation__depth_weight_curves__scalar_weights_per_matrix == 1
    assert depth_curves._scalar_weights_per_matrix() == 1
    assert depth_curves._depth_evaluation_points() == 256
    assert depth_curves._time_mode() == "accumulate"
    assert depth_curves._history_length() == 12
    assert depth_curves._log_every_n_steps() == 1
    assert depth_curves._same_coordinates_all_runs() is True
# ^^^ THOG


# vvv THOG default selection is deterministic within one run but deliberately changes its seed with run identity
def test_scalar_selection_is_fixed_per_run_and_run_specific(monkeypatch) -> None:
    monkeypatch.setenv(
        depth_curves._environment_name("SAME_COORDINATES_ALL_RUNS"),
        "false",
    )
    monkeypatch.setenv(
        depth_curves._environment_name("SCALAR_WEIGHTS_PER_MATRIX"),
        "3",
    )
    trajectory = _trajectory()
    trainer = _trainer(trajectory)
    telemetry_a = _telemetry("run-a")
    telemetry_b = _telemetry("run-b")

    first = depth_curves._selected_scalar_coordinates(trainer, telemetry_a)
    repeated = depth_curves._selected_scalar_coordinates(trainer, telemetry_a)
    other_run = depth_curves._selected_scalar_coordinates(trainer, telemetry_b)

    assert first == repeated
    assert first["seed"] != other_run["seed"]
    assert len(first["attn_q_head_N"]) == 3
    assert len(first["mlp_up"]) == 3
    assert len(first["mlp_down"]) == 3
    head_dim = trajectory.config.n_embd // trajectory.config.n_head
    head = int(first["attention_head"])
    assert all(
        head * head_dim <= row < (head + 1) * head_dim
        for row, _column in first["attn_q_head_N"]
    )
# ^^^ THOG


# vvv THOG cross-run fixed mode ignores run identity and therefore selects the same scalar coordinates across runs
def test_scalar_selection_can_be_fixed_across_runs(monkeypatch) -> None:
    monkeypatch.setenv(
        depth_curves._environment_name("SAME_COORDINATES_ALL_RUNS"),
        "true",
    )
    monkeypatch.setenv(
        depth_curves._environment_name("SCALAR_WEIGHTS_PER_MATRIX"),
        "3",
    )
    trajectory = _trajectory()
    trainer = _trainer(trajectory)
    first = depth_curves._selected_scalar_coordinates(trainer, _telemetry("run-a"))
    second = depth_curves._selected_scalar_coordinates(trainer, _telemetry("run-b"))
    assert first == second
# ^^^ THOG


# vvv THOG continuous scalar evaluation returns the requested dense public-depth ruler rather than only active integer layers
def test_continuous_scalar_curve_uses_dense_public_depth_ruler(monkeypatch) -> None:
    monkeypatch.setenv(depth_curves._environment_name("DEPTH_EVALUATION_POINTS"), "17")
    trajectory = _trajectory()
    family = depth_curves._evaluate_scalar_family(
        trajectory,
        MLP_EXPANSION_WEIGHT,
        ((0, 0),),
    )
    coordinates = family["depth_coordinates"]
    assert len(coordinates) == 17
    assert coordinates[0] == pytest.approx(1.0)
    assert coordinates[-1] == pytest.approx(100.0)
    assert family["values"].shape == (1, 17)
# ^^^ THOG


# vvv THOG fixed-run probes remain opt-in via explicit cadence when learned layer-count control is absent
def test_observational_probe_enablement_requires_explicit_cadence() -> None:
    trajectory = _trajectory()
    assert depth_curves._observational_probe_enabled(_trainer(trajectory, cadence=None)) is False
    assert depth_curves._observational_probe_enabled(_trainer(trajectory, cadence=5)) is True
    assert depth_curves._observational_probe_enabled(_trainer(trajectory, cadence=5, learn=True)) is False
# ^^^ THOG


# vvv THOG DEBUG thresholds separate the new depth charts from the legacy sampled-coefficient forensic chart
def test_depth_chart_debug_thresholds(monkeypatch) -> None:
    monkeypatch.setattr(constants, "DEBUG", 2)
    assert depth_curves._depth_weight_charts_enabled() is False
    assert depth_curves._legacy_sampled_coefficient_chart_enabled() is False

    monkeypatch.setattr(constants, "DEBUG", 3)
    assert depth_curves._depth_weight_charts_enabled() is True
    assert depth_curves._legacy_sampled_coefficient_chart_enabled() is False

    monkeypatch.setattr(constants, "DEBUG", 9)
    assert depth_curves._depth_weight_charts_enabled() is True
    assert depth_curves._legacy_sampled_coefficient_chart_enabled() is False

    monkeypatch.setattr(constants, "DEBUG", 10)
    assert depth_curves._legacy_sampled_coefficient_chart_enabled() is True
# ^^^ THOG


# vvv THOG RHS favourable negative deltas use the deliberately darker green without changing the left-side favourable colour contract
def test_rhs_negative_delta_green_is_darker() -> None:
    assert depth_curves._DARKER_RHS_GREEN == "\033[38;2;0;180;0m"
# ^^^ THOG


# vvv THOG accumulated chart rows remain bounded below W&B's hard table ceiling while retaining complete scalar curves
def test_accumulated_chart_rows_fit_wandb_table_limit(monkeypatch) -> None:
    monkeypatch.setenv(depth_curves._environment_name("SCALAR_WEIGHTS_PER_MATRIX"), "3")
    monkeypatch.setenv(depth_curves._environment_name("DEPTH_EVALUATION_POINTS"), "256")
    monkeypatch.setenv(depth_curves._environment_name("HISTORY_LENGTH"), "20")
    trajectory = _trajectory()
    trainer = _trainer(trajectory)
    telemetry = _telemetry("run-a")
    snapshots = []
    for update in range(20):
        snapshots.append(depth_curves._depth_snapshot(trainer, telemetry, update))
    rows = depth_curves._depth_chart_rows(snapshots, "mlp_up")
    assert len(rows) <= 9_999
    assert len(rows) % 256 == 0
# ^^^ THOG


# vvv THOG observational probe events are converted into W&B probe records even when PLASTIC itself is disabled
def test_observational_probe_events_are_wandb_visible() -> None:
    event = SimpleNamespace(
        name="plastic_depth_count_decision",
        completed_updates=4,
        payload={
            "previous_active_layers": 4,
            "selected_active_layers": 4,
            "observational_only": True,
            "candidates": (
                {"active_layers": 3, "validation_loss": 5.1},
                {"active_layers": 4, "validation_loss": 5.0},
                {"active_layers": 5, "validation_loss": 4.9},
            ),
        },
    )
    record = observational_wandb._probe_record_from_event(event)
    assert record is not None
    assert record["active_layers"] == 4
    assert record["selected_layers"] == 4
# ^^^ THOG
