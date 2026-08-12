from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest
import torch

from sheet import depth_weight_curves_and_observational_probes_patch as depth_curves
from sheet import depth_weight_curves_v2_patch as depth_curves_v2
from sheet.depth_trajectory import DepthTrajectory
from sheet.geometry import SheetGeometryConfig


# vvv THOG compact public DEPTH fixture exercises all six semantic matrix families without allocating a training-size model
def _trajectory(*, n_layer: int = 4) -> DepthTrajectory:
    geometry = SheetGeometryConfig(
        n_layer=n_layer,
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


def _trainer(trajectory: DepthTrajectory, *, run_seed: int = 1234):
    return SimpleNamespace(
        raw_model=SimpleNamespace(trajectory=trajectory),
        config=SimpleNamespace(model_seed=run_seed),
    )


def _telemetry(name: str):
    return SimpleNamespace(name=name, group="test-group")
# ^^^ THOG


# vvv THOG Q/K/V use the same selected head by output-row slice while attention output uses that same head by input-column slice
def test_all_attention_charts_share_one_head(monkeypatch) -> None:
    monkeypatch.setenv(depth_curves._environment_name("SCALAR_WEIGHTS_PER_MATRIX"), "3")
    monkeypatch.setenv(depth_curves._environment_name("SAME_COORDINATES_ALL_RUNS"), "false")
    trajectory = _trajectory()
    selection = depth_curves_v2._selected_scalar_coordinates_v2(
        _trainer(trajectory),
        _telemetry("run-a"),
    )

    head = int(selection["attention_head"])
    head_dim = trajectory.config.n_embd // trajectory.config.n_head
    head_start = head * head_dim
    head_stop = (head + 1) * head_dim
    for chart_name in ("attn_q_head_N", "attn_k_head_N", "attn_v_head_N"):
        assert len(selection[chart_name]) == 3
        assert all(head_start <= row < head_stop for row, _column in selection[chart_name])
    assert len(selection["attn_out_head_N"]) == 3
    assert all(
        head_start <= column < head_stop
        for _row, column in selection["attn_out_head_N"]
    )
# ^^^ THOG


# vvv THOG explicit fixed-coordinate mode preserves the shared head N and every scalar coordinate across run identities
def test_same_coordinates_all_runs_preserves_full_selection(monkeypatch) -> None:
    monkeypatch.setenv(depth_curves._environment_name("SCALAR_WEIGHTS_PER_MATRIX"), "2")
    monkeypatch.setenv(depth_curves._environment_name("SAME_COORDINATES_ALL_RUNS"), "true")
    trajectory = _trajectory()
    trainer = _trainer(trajectory)
    first = depth_curves_v2._selected_scalar_coordinates_v2(trainer, _telemetry("run-a"))
    second = depth_curves_v2._selected_scalar_coordinates_v2(trainer, _telemetry("run-b"))
    assert first == second
# ^^^ THOG


# vvv THOG the visible x-axis is 1..executed layer count while exact integer executed positions are retained separately for marker overlays
def test_snapshot_uses_executed_layer_index_ruler(monkeypatch) -> None:
    monkeypatch.setenv(depth_curves._environment_name("SCALAR_WEIGHTS_PER_MATRIX"), "1")
    monkeypatch.setenv(depth_curves._environment_name("DEPTH_EVALUATION_POINTS"), "17")
    trajectory = _trajectory()
    snapshot = depth_curves_v2._depth_weight_snapshot_v2(
        _trainer(trajectory),
        _telemetry("run-a"),
        optimizer_update=7,
    )

    assert set(snapshot["families"]) == {
        "attn_q_head_N",
        "attn_k_head_N",
        "attn_v_head_N",
        "attn_out_head_N",
        "mlp_up",
        "mlp_down",
    }
    family = snapshot["families"]["mlp_down"]
    assert len(family["depth_coordinates"]) == 17
    assert family["depth_coordinates"][0] == pytest.approx(1.0)
    assert family["depth_coordinates"][-1] == pytest.approx(4.0)
    assert family["executed_layer_coordinates"] == (1.0, 2.0, 3.0, 4.0)
    assert len(family["curves"][0]["values"]) == 17
    assert len(family["curves"][0]["executed_values"]) == 4
# ^^^ THOG


# vvv THOG accumulated Plotly history makes age visually and interactively explicit while marking only the newest curve at executed layers
def test_plotly_history_emphasises_newest_and_oldest(monkeypatch) -> None:
    monkeypatch.setenv(depth_curves._environment_name("SCALAR_WEIGHTS_PER_MATRIX"), "1")
    monkeypatch.setenv(depth_curves._environment_name("DEPTH_EVALUATION_POINTS"), "17")
    trajectory = _trajectory()
    trainer = _trainer(trajectory)
    telemetry = _telemetry("run-a")
    snapshots = (
        depth_curves_v2._depth_weight_snapshot_v2(trainer, telemetry, optimizer_update=10),
        depth_curves_v2._depth_weight_snapshot_v2(trainer, telemetry, optimizer_update=11),
    )
    figure = depth_curves_v2._build_depth_plotly_figure(snapshots, "attn_q_head_N")

    line_traces = [trace for trace in figure.data if trace.mode == "lines"]
    marker_traces = [trace for trace in figure.data if trace.mode == "markers"]
    assert len(line_traces) == 2
    assert len(marker_traces) == 1
    assert "oldest U10" in line_traces[0].name
    assert "newest U11" in line_traces[1].name
    assert float(line_traces[0].opacity) < float(line_traces[1].opacity)
    assert float(line_traces[0].line.width) < float(line_traces[1].line.width)
    assert line_traces[0].line.color == line_traces[1].line.color
    assert tuple(marker_traces[0].x) == (1.0, 2.0, 3.0, 4.0)
    assert figure.layout.hovermode == "closest"
    assert figure.layout.xaxis.dtick == 1
    assert figure.layout.xaxis.tickangle == 0
    assert figure.layout.xaxis.automargin is True
    assert f"attn_q_head_{snapshots[-1]['attention_head']}" in figure.layout.title.text
# ^^^ THOG


# vvv THOG large DEPTH charts retain every layer coordinate in the data while limiting the visible ruler to readable, unclipped integer ticks
def test_plotly_large_layer_ruler_limits_visible_tick_density(monkeypatch) -> None:
    monkeypatch.setenv(depth_curves._environment_name("SCALAR_WEIGHTS_PER_MATRIX"), "1")
    monkeypatch.setenv(depth_curves._environment_name("DEPTH_EVALUATION_POINTS"), "145")
    trajectory = _trajectory(n_layer=144)
    snapshot = depth_curves_v2._depth_weight_snapshot_v2(
        _trainer(trajectory),
        _telemetry("run-a"),
        optimizer_update=12,
    )
    figure = depth_curves_v2._build_depth_plotly_figure((snapshot,), "mlp_down")

    marker_trace = next(trace for trace in figure.data if trace.mode == "markers")
    assert len(marker_trace.x) == 144
    assert tuple(figure.layout.xaxis.range) == (1.0, 144.0)
    assert figure.layout.xaxis.tickmode == "auto"
    assert figure.layout.xaxis.nticks == 18
    assert figure.layout.xaxis.dtick is None
    assert figure.layout.xaxis.tickformat == ".0f"
    assert figure.layout.xaxis.tickangle == 0
    assert figure.layout.xaxis.automargin is True
# ^^^ THOG


# vvv THOG the public cross-run coordinate lock accepts explicit true|false after the exact train_OWT single-hyphen normalization
def test_same_coordinates_cli_accepts_explicit_true_false(monkeypatch) -> None:
    environment = depth_curves._environment_name("SAME_COORDINATES_ALL_RUNS")
    monkeypatch.delenv(environment, raising=False)

    parser_false = argparse.ArgumentParser(add_help=False)
    values_false, remaining_false = parser_false.parse_known_args(
        [
            "--instrumentation-depth-weight-curves-same-coordinates-all-runs",
            "false",
        ]
    )
    assert remaining_false == []
    assert values_false.instrumentation__depth_weight_curves__same_coordinates_all_runs is False
    assert depth_curves._same_coordinates_all_runs() is False

    parser_true = argparse.ArgumentParser(add_help=False)
    values_true, remaining_true = parser_true.parse_known_args(
        [
            "--instrumentation-depth-weight-curves-same-coordinates-all-runs",
            "true",
        ]
    )
    assert remaining_true == []
    assert values_true.instrumentation__depth_weight_curves__same_coordinates_all_runs is True
    assert depth_curves._same_coordinates_all_runs() is True
# ^^^ THOG
