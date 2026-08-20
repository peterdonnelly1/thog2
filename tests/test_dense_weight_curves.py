from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from model import GPTConfig
from sheet import dense_weight_curves_patch as dense_curves
from sheet import depth_weight_curves_and_observational_probes_patch as depth_curves
from sheet import depth_weight_curves_v2_patch as depth_curves_v2
from sheet import depth_weight_curves_v2_runtime_seam_patch as runtime_seam
from sheet.local_chart_store import LocalChartReader, close_local_chart_store
from sheet.training_model import TrainingDenseGPT


def _model() -> TrainingDenseGPT:
    return TrainingDenseGPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=3,
            n_head=2,
            n_embd=4,
            dropout=0.0,
            bias=True,
        )
    )


def _trainer(model: TrainingDenseGPT):
    return SimpleNamespace(
        raw_model=model,
        config=SimpleNamespace(model_seed=1234),
    )


def _telemetry(name: str = "dense-run"):
    return SimpleNamespace(name=name, group="test-group")


def test_dense_snapshot_reuses_all_six_families_with_integer_layer_points(monkeypatch) -> None:
    monkeypatch.setenv(depth_curves._environment_name("SCALAR_WEIGHTS_PER_MATRIX"), "1")
    model = _model()
    snapshot = dense_curves._dense_weight_snapshot(
        _trainer(model),
        _telemetry(),
        optimizer_update=7,
    )

    assert snapshot["model_type"] == "dense"
    assert snapshot["trajectory_kind"] == dense_curves._DENSE_KIND
    assert set(snapshot["families"]) == set(depth_curves_v2._CHART_FAMILIES)
    for family in snapshot["families"].values():
        assert family["depth_coordinates"] == (1.0, 2.0, 3.0)
        assert family["executed_layer_coordinates"] == (1.0, 2.0, 3.0)
        assert len(family["curves"]) == 1
        assert len(family["curves"][0]["values"]) == 3


def test_dense_qkv_rows_read_the_correct_combined_projection_slice(monkeypatch) -> None:
    monkeypatch.setenv(depth_curves._environment_name("SCALAR_WEIGHTS_PER_MATRIX"), "1")
    monkeypatch.setenv(depth_curves._environment_name("SAME_COORDINATES_ALL_RUNS"), "true")
    model = _model()
    trainer = _trainer(model)
    telemetry = _telemetry()
    selection = dense_curves._dense_selection(trainer, telemetry)
    width = model.config.n_embd
    for layer_index, block in enumerate(model.transformer.h, start=1):
        with torch.no_grad():
            for row in range(3 * width):
                block.attn.c_attn.weight[row].fill_(1000.0 * layer_index + row)
    snapshot = dense_curves._dense_weight_snapshot(
        trainer,
        telemetry,
        optimizer_update=8,
    )
    for chart_name, slice_index in (
        ("attn_q_head_N", 0),
        ("attn_k_head_N", 1),
        ("attn_v_head_N", 2),
    ):
        semantic_row, _column = selection[chart_name][0]
        expected = tuple(
            1000.0 * layer + semantic_row + slice_index * width
            for layer in (1, 2, 3)
        )
        assert snapshot["families"][chart_name]["curves"][0]["values"] == expected


def test_dense_figure_uses_unconnected_crosses(monkeypatch) -> None:
    monkeypatch.setenv(depth_curves._environment_name("SCALAR_WEIGHTS_PER_MATRIX"), "1")
    model = _model()
    trainer = _trainer(model)
    telemetry = _telemetry()
    snapshots = (
        dense_curves._dense_weight_snapshot(trainer, telemetry, optimizer_update=10),
        dense_curves._dense_weight_snapshot(trainer, telemetry, optimizer_update=11),
    )
    figure = depth_curves_v2._build_depth_plotly_figure(snapshots, "mlp_down")

    assert len(figure.data) == 2
    assert all(trace.mode == "markers" for trace in figure.data)
    assert all(trace.marker.symbol == "x" for trace in figure.data)
    assert all(trace.line.width is None for trace in figure.data)
    assert tuple(figure.layout.xaxis.range) == (0.5, 3.5)
    assert "DENSE learned scalar weights" in figure.layout.title.text
    assert "× = discrete materialised layer weight" in figure.layout.title.text


def test_dense_and_depth_selection_lock_is_run_identity_independent(monkeypatch) -> None:
    monkeypatch.setenv(depth_curves._environment_name("SCALAR_WEIGHTS_PER_MATRIX"), "2")
    monkeypatch.setenv(depth_curves._environment_name("SAME_COORDINATES_ALL_RUNS"), "true")
    model = _model()
    first = dense_curves._dense_selection(_trainer(model), _telemetry("run-a"))
    second = dense_curves._dense_selection(_trainer(model), _telemetry("run-b"))
    assert first == second


def test_dense_local_sink_uses_existing_snapshot_store(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("THOG2_INSTRUMENTATION_LOCAL_ROOT", str(tmp_path))
    monkeypatch.setenv(depth_curves._environment_name("DESTINATION"), "local")
    monkeypatch.setenv(depth_curves._environment_name("HISTORY_LENGTH"), "2")
    monkeypatch.setenv(depth_curves._environment_name("TIME_MODE"), "accumulate")
    monkeypatch.setenv(depth_curves._environment_name("SCALAR_WEIGHTS_PER_MATRIX"), "1")
    monkeypatch.setattr(depth_curves._constants, "DEBUG", 3)
    telemetry = SimpleNamespace(
        name="dense_local_test",
        group="test-group",
        config={},
        run=None,
        module=None,
    )
    trainer = _trainer(_model())

    for step in (10, 20, 30):
        runtime_seam._log_depth_weight_snapshot_with_patchable_snapshot(
            trainer,
            telemetry,
            optimizer_update=step,
        )

    reader = LocalChartReader(telemetry._thog_local_chart_store.path)
    snapshots = reader.depth_weight_snapshots()
    assert tuple(snapshot["optimizer_update"] for snapshot in snapshots) == (20, 30)
    assert snapshots[-1]["trajectory_kind"] == dense_curves._DENSE_KIND
    close_local_chart_store(telemetry)
