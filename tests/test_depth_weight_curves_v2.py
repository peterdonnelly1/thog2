from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest
import torch

from sheet import depth_weight_curves_and_observational_probes_patch as depth_curves
from sheet import depth_weight_curves_v2_patch as depth_curves_v2
from sheet import depth_weight_curves_v2_runtime_seam_patch as runtime_seam
from sheet.depth_trajectory import DepthTrajectory
from sheet.geometry import SheetGeometryConfig
from sheet.local_chart_store import LocalChartReader, close_local_chart_store


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


def test_matched_logical_coordinates_follow_matrix_orientation() -> None:
    from sheet import matched_weight_selection_patch as matched_weights

    forward = {"attn_q_head_N", "attn_k_head_N", "attn_v_head_N", "mlp_up"}
    reverse = {"attn_out_head_N", "mlp_down"}

    for chart_name in forward:
        matrix_coordinate = matched_weights._logical_to_matrix(chart_name, 17, 29)
        assert matrix_coordinate == (29, 17)
        assert matched_weights._matrix_to_logical(chart_name, *matrix_coordinate) == (17, 29)

    for chart_name in reverse:
        matrix_coordinate = matched_weights._logical_to_matrix(chart_name, 17, 29)
        assert matrix_coordinate == (17, 29)
        assert matched_weights._matrix_to_logical(chart_name, *matrix_coordinate) == (17, 29)


def test_matched_weight_selection_is_persisted_per_run(monkeypatch, tmp_path) -> None:
    from sheet import matched_weight_selection_patch as matched_weights

    monkeypatch.setenv("THOG2_INSTRUMENTATION_LOCAL_ROOT", str(tmp_path))
    first = SimpleNamespace(name="same-name", run=SimpleNamespace(id="run-one"))
    second = SimpleNamespace(name="same-name", run=SimpleNamespace(id="run-two"))
    first_root = matched_weights._selection_root_for_telemetry(first)
    second_root = matched_weights._selection_root_for_telemetry(second)

    assert first_root != second_root
    matched_weights.write_weight_selection(
        {"user_selected": True, "model_feature": 1010, "intermediate_feature": 12},
        first_root,
    )
    assert matched_weights.read_weight_selection(first_root)["model_feature"] == 1010
    assert matched_weights.read_weight_selection(second_root) == {
        "protocol": matched_weights.WEIGHT_SELECTION_PROTOCOL,
        "user_selected": False,
        "model_feature": 0,
        "intermediate_feature": 0,
    }


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
    assert {key: value for key, value in first.items() if key != "matched_selection_root"} == {
        key: value for key, value in second.items() if key != "matched_selection_root"
    }
    assert first["matched_selection_root"] != second["matched_selection_root"]
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


def test_plastic_sampling_filters_curve_geometry_without_owning_capture_schedule() -> None:
    active = torch.tensor([4.0, 19.0, 43.0, 88.0], dtype=torch.float32)
    trajectory = SimpleNamespace(
        plastic_enabled=True,
        plastic_sampling=SimpleNamespace(
            active_public_coordinates=lambda: active,
        ),
    )

    coordinates = depth_curves_v2._executed_public_coordinates(
        trajectory,
        torch.zeros(1, dtype=torch.float32),
    )

    assert coordinates.dtype == torch.float64
    assert coordinates.tolist() == pytest.approx(active.tolist())
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
    assert line_traces[0].meta["instra_thog_weight"] is True
    assert line_traces[0].meta["instra_thog_optimizer_update"] == 10
    assert tuple(line_traces[0].meta["instra_thog_integer_x"]) == (1.0, 2.0, 3.0, 4.0)
    assert tuple(line_traces[0].meta["instra_thog_integer_y"]) == snapshots[0]["families"]["attn_q_head_N"]["curves"][0]["executed_values"]
    assert tuple(marker_traces[0].x) == (1.0, 2.0, 3.0, 4.0)
    assert marker_traces[0].meta["instra_thog_executed_overlay"] is True
    assert marker_traces[0].meta["instra_thog_optimizer_update"] == 11
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


# vvv THOG destination selection is independent of W&B and local depth snapshots remain bounded by history length
def test_depth_curve_destination_defaults_local_and_accepts_explicit_none(monkeypatch) -> None:
    environment = depth_curves._environment_name("DESTINATION")
    monkeypatch.delenv(environment, raising=False)
    parser = argparse.ArgumentParser(add_help=False)
    defaults, remaining = parser.parse_known_args([])
    assert remaining == []
    assert defaults.instrumentation__depth_weight_curves__destination == "local"
    assert depth_curves._destination() == "local"

    explicit, remaining = parser.parse_known_args(
        ["--instrumentation__depth_weight_curves__destination", "none"]
    )
    assert remaining == []
    assert explicit.instrumentation__depth_weight_curves__destination == "none"
    assert depth_curves._destination() == "none"


def test_local_depth_curve_sink_uses_no_wandb_and_bounds_history(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("THOG2_INSTRUMENTATION_LOCAL_ROOT", str(tmp_path))
    monkeypatch.setenv(depth_curves._environment_name("DESTINATION"), "local")
    monkeypatch.setenv(depth_curves._environment_name("HISTORY_LENGTH"), "2")
    monkeypatch.setenv(depth_curves._environment_name("TIME_MODE"), "accumulate")
    monkeypatch.setattr(depth_curves._constants, "DEBUG", 3)
    trajectory = _trajectory()
    trainer = _trainer(trajectory)
    telemetry = SimpleNamespace(
        name="local_depth_test",
        group="test-group",
        config={},
        run=None,
        module=None,
    )

    for step in (10, 20, 30):
        runtime_seam._log_depth_weight_snapshot_with_patchable_snapshot(
            trainer,
            telemetry,
            optimizer_update=step,
        )

    store = telemetry._thog_local_chart_store
    reader = LocalChartReader(store.path)
    snapshots = reader.depth_weight_snapshots()
    assert tuple(snapshot["optimizer_update"] for snapshot in snapshots) == (20, 30)
    assert set(snapshots[-1]["families"]) == set(depth_curves_v2._CHART_FAMILIES)
    close_local_chart_store(telemetry)
# ^^^ THOG


def test_local_depth_curve_failure_disables_capture_without_stopping_training(
    monkeypatch,
    capsys,
) -> None:
    calls = []

    class FailingStore:
        def append_depth_weight_snapshot(self, *_args, **_kwargs) -> None:
            raise OSError("local disk unavailable")

    monkeypatch.setattr(depth_curves._constants, "DEBUG", 3)
    monkeypatch.setattr(depth_curves, "_destination", lambda: "local")
    monkeypatch.setattr(
        depth_curves,
        "_depth_weight_snapshot",
        lambda *_args, **_kwargs: calls.append("snapshot") or {
            "optimizer_update": 10,
            "families": {},
        },
    )
    monkeypatch.setattr(
        runtime_seam,
        "ensure_local_chart_store",
        lambda _telemetry: FailingStore(),
    )
    telemetry = SimpleNamespace(run=None, module=None)

    runtime_seam._log_depth_weight_snapshot_with_patchable_snapshot(
        object(),
        telemetry,
        optimizer_update=10,
    )
    runtime_seam._log_depth_weight_snapshot_with_patchable_snapshot(
        object(),
        telemetry,
        optimizer_update=11,
    )

    assert calls == ["snapshot"]
    assert telemetry._thog_local_depth_weight_capture_disabled is True
    output = capsys.readouterr().out
    assert output.count("local DEPTH weight logging failed") == 1


# vvv THOG repeated scalar ids must retain the coupling recorded at their own optimizer update, not inherit the newest snapshot's indices
def test_matched_weight_metadata_is_keyed_by_update_and_scalar(monkeypatch) -> None:
    from sheet import matched_weight_selection_patch as matched_weights

    def trace(update: int):
        return SimpleNamespace(
            name="r0_c0",
            hovertemplate="r0_c0",
            meta={
                "instra_thog_scalar_id": "r0_c0",
                "instra_thog_optimizer_update": update,
            },
        )

    figure = SimpleNamespace(
        data=[trace(10), trace(20)],
        layout=SimpleNamespace(title=SimpleNamespace(text="DEPTH — attention query")),
    )
    monkeypatch.setattr(
        matched_weights,
        "_ORIGINAL_FIGURE_BUILDER",
        lambda _snapshots, _chart_name: figure,
    )
    snapshots = tuple(
        {
            "optimizer_update": update,
            "weight_selection": {"feature_count": 32},
            "families": {
                "attn_q_head_N": {
                    "curves": ({
                        "scalar_id": "r0_c0",
                        "model_feature": model_feature,
                        "intermediate_feature": intermediate_feature,
                        "selection_kind": kind,
                    },),
                },
            },
        }
        for update, model_feature, intermediate_feature, kind in (
            (10, 1, 2, "random"),
            (20, 8, 9, "user"),
        )
    )

    prepared = matched_weights._build_figure_with_matched_selection(
        snapshots,
        "attn_q_head_N",
    )

    assert prepared.data[0].meta["instra_weight_model_feature"] == 1
    assert prepared.data[0].meta["instra_weight_intermediate_feature"] == 2
    assert prepared.data[0].meta["instra_weight_selection_kind"] == "random"
    assert prepared.data[1].meta["instra_weight_model_feature"] == 8
    assert prepared.data[1].meta["instra_weight_intermediate_feature"] == 9
    assert prepared.data[1].meta["instra_weight_selection_kind"] == "user"
# ^^^ THOG
