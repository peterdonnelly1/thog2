from __future__ import annotations

import argparse
from types import SimpleNamespace

import constants
import pytest
import torch

from sheet import depth_weight_curves_and_observational_probes_patch as depth_curves
from sheet import plastic_depth_wandb_probe_curves_patch as probe_wandb
from sheet.depth_trajectory import DepthTrajectory
from sheet.geometry import SheetGeometryConfig


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


def _trainer(
    trajectory: DepthTrajectory,
    *,
    run_seed: int = 1234,
    cadence=5,
    learn=False,
    plastic=False,
):
    return SimpleNamespace(
        raw_model=SimpleNamespace(trajectory=trajectory),
        config=SimpleNamespace(
            model_seed=run_seed,
            plastic__enabled=plastic,
            plastic__do_learn_layer_count=learn,
            plastic__layer_count_probe__probe_every_n_steps=cadence,
        ),
    )


def _telemetry(name: str):
    return SimpleNamespace(name=name, group="test-group")


def _clear_depth_curve_environment(monkeypatch) -> None:
    for suffix in (
        "SCALAR_WEIGHTS_PER_MATRIX",
        "DEPTH_EVALUATION_POINTS",
        "TIME_MODE",
        "HISTORY_LENGTH",
        "LOG_EVERY_N_STEPS",
        "SAME_COORDINATES_ALL_RUNS",
        "START_STEP",
        "END_STEP",
    ):
        monkeypatch.delenv(depth_curves._environment_name(suffix), raising=False)


# vvv THOG canonical instrumentation CLI controls publish execution-only environment controls rather than model identity fields
def test_depth_weight_curve_cli_controls_publish_environment(monkeypatch) -> None:
    _clear_depth_curve_environment(monkeypatch)
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


# vvv THOG train_OWT collapses underscore runs to single hyphens before forwarding, so exact wrapper-produced spellings must parse
def test_depth_weight_curve_cli_controls_accept_wrapper_normalized_spellings(monkeypatch) -> None:
    _clear_depth_curve_environment(monkeypatch)
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


# vvv THOG the established argparse compatibility layer may preserve double hyphens created from double underscores, so that spelling is accepted too
def test_depth_weight_curve_cli_controls_accept_parser_normalized_spellings(monkeypatch) -> None:
    _clear_depth_curve_environment(monkeypatch)
    parser = argparse.ArgumentParser(add_help=False)
    values, remaining = parser.parse_known_args(
        [
            "--instrumentation--depth-weight-curves--scalar-weights-per-matrix",
            "2",
            "--instrumentation--depth-weight-curves--depth-evaluation-points",
            "32",
            "--instrumentation--depth-weight-curves--time-mode",
            "latest",
            "--instrumentation--depth-weight-curves--history-length",
            "4",
            "--instrumentation--depth-weight-curves--log-every-n-steps",
            "3",
            "--instrumentation--depth-weight-curves--same-coordinates-all-runs",
        ]
    )

    assert remaining == []
    assert values.instrumentation__depth_weight_curves__scalar_weights_per_matrix == 2
    assert depth_curves._scalar_weights_per_matrix() == 2
    assert depth_curves._depth_evaluation_points() == 32
    assert depth_curves._time_mode() == "latest"
    assert depth_curves._history_length() == 4
    assert depth_curves._log_every_n_steps() == 3
    assert depth_curves._same_coordinates_all_runs() is True
# ^^^ THOG


def test_weight_snapshot_capture_window_is_inclusive_and_anchors_cadence(monkeypatch) -> None:
    _clear_depth_curve_environment(monkeypatch)
    monkeypatch.setenv(depth_curves._environment_name("LOG_EVERY_N_STEPS"), "25")
    monkeypatch.setenv(depth_curves._environment_name("START_STEP"), "300")
    monkeypatch.setenv(depth_curves._environment_name("END_STEP"), "400")

    captured = [step for step in range(1, 425) if depth_curves._weight_snapshot_due(step)]

    assert captured == [300, 325, 350, 375, 400]


def test_weight_snapshot_capture_window_forces_unaligned_end(monkeypatch) -> None:
    _clear_depth_curve_environment(monkeypatch)
    monkeypatch.setenv(depth_curves._environment_name("LOG_EVERY_N_STEPS"), "40")
    monkeypatch.setenv(depth_curves._environment_name("START_STEP"), "300")
    monkeypatch.setenv(depth_curves._environment_name("END_STEP"), "405")

    captured = [step for step in range(250, 450) if depth_curves._weight_snapshot_due(step)]

    assert captured == [300, 340, 380, 405]


def test_unbounded_weight_snapshot_cadence_is_unchanged(monkeypatch) -> None:
    _clear_depth_curve_environment(monkeypatch)
    monkeypatch.setenv(depth_curves._environment_name("LOG_EVERY_N_STEPS"), "100")

    captured = [step for step in range(1, 251) if depth_curves._weight_snapshot_due(step)]

    assert captured == [1, 100, 200]


def test_weight_snapshot_capture_window_rejects_reversed_bounds(monkeypatch) -> None:
    _clear_depth_curve_environment(monkeypatch)
    monkeypatch.setenv(depth_curves._environment_name("START_STEP"), "400")
    monkeypatch.setenv(depth_curves._environment_name("END_STEP"), "300")

    with pytest.raises(ValueError, match="END_STEP.*greater than or equal"):
        depth_curves._weight_snapshot_due(350)


# vvv THOG default selection is deterministic within one run but deliberately changes its seed with run identity
def test_scalar_selection_is_fixed_per_run_and_run_specific(monkeypatch) -> None:
    monkeypatch.setenv(depth_curves._environment_name("SAME_COORDINATES_ALL_RUNS"), "false")
    monkeypatch.setenv(depth_curves._environment_name("SCALAR_WEIGHTS_PER_MATRIX"), "3")
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
    monkeypatch.setenv(depth_curves._environment_name("SAME_COORDINATES_ALL_RUNS"), "true")
    monkeypatch.setenv(depth_curves._environment_name("SCALAR_WEIGHTS_PER_MATRIX"), "3")
    trajectory = _trajectory()
    trainer = _trainer(trajectory)
    first = depth_curves._selected_scalar_coordinates(trainer, _telemetry("run-a"))
    second = depth_curves._selected_scalar_coordinates(trainer, _telemetry("run-b"))
    assert {key: value for key, value in first.items() if key != "matched_selection_root"} == {
        key: value for key, value in second.items() if key != "matched_selection_root"
    }
    assert first["matched_selection_root"] != second["matched_selection_root"]
# ^^^ THOG


# vvv THOG continuous scalar snapshot is relabelled onto the executed 1..L layer-index ruler and separately records exact integer layer positions
def test_continuous_scalar_curve_uses_executed_layer_index_ruler(monkeypatch) -> None:
    monkeypatch.setenv(depth_curves._environment_name("DEPTH_EVALUATION_POINTS"), "17")
    monkeypatch.setenv(depth_curves._environment_name("SCALAR_WEIGHTS_PER_MATRIX"), "1")
    trajectory = _trajectory()
    snapshot = depth_curves._depth_weight_snapshot(
        _trainer(trajectory),
        _telemetry("run-a"),
        optimizer_update=1,
    )
    family = snapshot["families"]["mlp_up"]
    coordinates = family["depth_coordinates"]
    assert len(coordinates) == 17
    assert coordinates[0] == pytest.approx(1.0)
    assert coordinates[-1] == pytest.approx(4.0)
    assert family["executed_layer_coordinates"] == (1.0, 2.0, 3.0, 4.0)
    assert len(family["curves"]) == 1
    assert len(family["curves"][0]["values"]) == 17
# ^^^ THOG


# vvv THOG fixed-run probes remain opt-in via explicit cadence when learned layer-count control is absent
def test_observational_probe_enablement_requires_explicit_cadence() -> None:
    trajectory = _trajectory()
    assert depth_curves._observational_probe_enabled(_trainer(trajectory, cadence=None)) is False
    assert depth_curves._observational_probe_enabled(_trainer(trajectory, cadence=5)) is True
    assert depth_curves._observational_probe_enabled(_trainer(trajectory, cadence=5, plastic=True)) is True
    assert depth_curves._observational_probe_enabled(_trainer(trajectory, cadence=5, learn=True, plastic=True)) is False
    assert depth_curves._observational_probe_enabled(_trainer(trajectory, cadence=5, learn=True, plastic=False)) is True
# ^^^ THOG


# vvv THOG DEBUG thresholds separate new depth chart work from the legacy sampled-coefficient forensic chart
def test_depth_chart_debug_thresholds(monkeypatch) -> None:
    calls = []
    trainer = _trainer(_trajectory())
    telemetry = SimpleNamespace(run=object(), module=object())
    monkeypatch.setenv(depth_curves._environment_name("DESTINATION"), "wandb")

    def fake_snapshot(*_args, **_kwargs):
        calls.append(1)
        return {}

    monkeypatch.setattr(depth_curves, "_depth_weight_snapshot", fake_snapshot)
    monkeypatch.setattr(constants, "DEBUG", 2)
    depth_curves._log_depth_weight_snapshot(trainer, telemetry, optimizer_update=1)
    assert calls == []
    assert depth_curves._legacy_coefficient_chart_enabled() is False

    monkeypatch.setattr(constants, "DEBUG", 3)
    depth_curves._log_depth_weight_snapshot(trainer, telemetry, optimizer_update=1)
    assert calls == [1]
    assert depth_curves._legacy_coefficient_chart_enabled() is False

    monkeypatch.setattr(constants, "DEBUG", 9)
    assert depth_curves._legacy_coefficient_chart_enabled() is False

    monkeypatch.setattr(constants, "DEBUG", 10)
    assert depth_curves._legacy_coefficient_chart_enabled() is True
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
        snapshots.append(
            depth_curves._depth_weight_snapshot(
                trainer,
                telemetry,
                optimizer_update=update,
            )
        )
    rows = depth_curves._depth_chart_rows(snapshots, "mlp_up")
    assert len(rows) <= 9_999
    assert len(rows) % 256 == 0
# ^^^ THOG


# vvv THOG observational probe events use the established W&B record conversion even when the selected layer count is unchanged
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
    record = probe_wandb._probe_record_from_event(event)
    assert record is not None
    assert record["active_layers"] == 4
    assert record["selected_layers"] == 4
# ^^^ THOG
