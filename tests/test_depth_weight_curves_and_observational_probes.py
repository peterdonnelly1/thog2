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


# vvv THOG cross-run-fixed mode deliberately removes run identity from scalar/head selection
def test_scalar_selection_can_be_identical_across_runs(monkeypatch) -> None:
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


# vvv THOG continuous chart snapshots evaluate each fixed scalar at arbitrary dense depth positions, not at the model's four actual layers
def test_depth_weight_snapshot_uses_dense_continuous_depth(monkeypatch) -> None:
    monkeypatch.setenv(
        depth_curves._environment_name("SAME_COORDINATES_ALL_RUNS"),
        "true",
    )
    monkeypatch.setenv(
        depth_curves._environment_name("SCALAR_WEIGHTS_PER_MATRIX"),
        "2",
    )
    monkeypatch.setenv(
        depth_curves._environment_name("DEPTH_EVALUATION_POINTS"),
        "31",
    )
    trajectory = _trajectory()
    trainer = _trainer(trajectory)

    snapshot = depth_curves._depth_weight_snapshot(
        trainer,
        _telemetry("run-a"),
        optimizer_update=17,
    )

    assert snapshot["optimizer_update"] == 17
    assert set(snapshot["families"]) == {"attn_q_head_N", "mlp_up", "mlp_down"}
    for family in snapshot["families"].values():
        assert len(family["depth_coordinates"]) == 31
        assert family["depth_coordinates"][0] == pytest.approx(1.0)
        assert family["depth_coordinates"][-1] == pytest.approx(100.0)
        assert len(family["curves"]) == 2
        assert all(len(curve["values"]) == 31 for curve in family["curves"])
        assert all(
            torch.isfinite(torch.tensor(curve["values"])).all().item()
            for curve in family["curves"]
        )
# ^^^ THOG


# vvv THOG observational views may ask the fixed four-layer trajectory for a fifth/sixth logical depth without changing its persistent configured layer count
def test_observational_materialization_extends_beyond_fixed_layer_count() -> None:
    trajectory = _trajectory()
    original_count = int(trajectory.config.n_layer)
    trajectory._thog_observational_depth_coordinates = torch.linspace(
        1.0,
        100.0,
        6,
        dtype=torch.float64,
    )
    try:
        generated = trajectory.materialize(MLP_EXPANSION_WEIGHT, 5)
        layer_norm = trajectory.materialize("ln_1_weight", 5)
    finally:
        delattr(trajectory, "_thog_observational_depth_coordinates")

    assert tuple(generated.shape) == (32, 8)
    assert tuple(layer_norm.shape) == (1, 8)
    assert bool(torch.isfinite(generated).all().item())
    assert bool(torch.isfinite(layer_norm).all().item())
    assert int(trajectory.config.n_layer) == original_count == 4
# ^^^ THOG


# vvv THOG an explicit probe cadence opts a fixed DEPTH run into observational probing; learned-count PLASTIC remains owned by its established path
def test_observational_probe_opt_in_requires_cadence_and_fixed_count() -> None:
    trajectory = _trajectory()
    assert depth_curves._observational_probe_enabled(
        _trainer(trajectory, cadence=5, learn=False)
    )
    assert not depth_curves._observational_probe_enabled(
        _trainer(trajectory, cadence=None, learn=False)
    )
    assert not depth_curves._observational_probe_enabled(
        _trainer(trajectory, cadence=5, learn=True)
    )
# ^^^ THOG


# vvv THOG legacy sampled-coefficient runtime visibility changes exactly at DEBUG>9
def test_legacy_coefficient_chart_debug_threshold(monkeypatch) -> None:
    monkeypatch.setattr(constants, "DEBUG", 9)
    assert not depth_curves._legacy_coefficient_chart_enabled()
    monkeypatch.setattr(constants, "DEBUG", 10)
    assert depth_curves._legacy_coefficient_chart_enabled()
# ^^^ THOG


# vvv THOG attached runtime suppresses old coefficient refresh below DEBUG>10 without changing the historical helper API used by regression tests
def test_attached_runtime_gates_old_coefficient_refresh(monkeypatch) -> None:
    telemetry = SimpleNamespace(
        _thog_runtime_legacy_coefficient_debug_gate=True,
    )
    monkeypatch.setattr(constants, "DEBUG", 3)
    assert not observational_wandb._should_refresh_coefficient_chart_runtime_gated(
        telemetry,
        evaluation=False,
    )
# ^^^ THOG


# vvv THOG only the favourable growth-side negative delta uses the new darker RGB green
def test_right_negative_probe_delta_uses_darker_green() -> None:
    rendered = depth_curves._render_probe_delta_values_with_darker_rhs(
        (-1, 0, 1),
        (4.9, 5.0, 4.8),
    )

    assert rendered is not None
    left, current, right = rendered.split(", ")
    assert depth_curves._DARKER_RHS_GREEN not in left
    assert depth_curves._DARKER_RHS_GREEN not in current
    assert depth_curves._DARKER_RHS_GREEN in right
# ^^^ THOG
