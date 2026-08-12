# vvv THOG
from __future__ import annotations

from types import SimpleNamespace

import constants
import pytest

import run_thog2_owt_core as runner
from sheet import plastic_depth_wandb_probe_curves_patch as curves
from sheet.run_config import OwtRunConfig
from sheet.trainer_state import TrainerEvent


def _event(
    *,
    completed_updates: int,
    probe_sequence: int,
    current: int = 4,
) -> TrainerEvent:
    return TrainerEvent(
        "plastic_depth_count_decision",
        completed_updates,
        {
            "previous_active_layers": current,
            "selected_active_layers": current,
            "probe_sequence": probe_sequence,
            "candidates": (
                {"active_layers": current - 1, "validation_loss": 4.98},
                {"active_layers": current, "validation_loss": 5.00},
                {"active_layers": current + 1, "validation_loss": 5.03},
            ),
        },
    )


def _enabled_config(**overrides):
    values = {
        "model_type": "sheet",
        "plastic__enabled": True,
        "plastic__do_learn_layer_count": True,
        "plastic__initial_layer_count": 4,
        "plastic__max_permitted_layers": 8,
        "o_depth": 8,
        "instrumentation__delta_loss_v_layer_heatmap": True,
    }
    values.update(overrides)
    return OwtRunConfig(**values)


def test_cli_uses_exact_true_false_values_and_conservative_defaults() -> None:
    arguments = runner.build_parser().parse_args(
        [
            "--instrumentation__delta_loss_v_layer_heatmap",
            "true",
            "--instrumentation__delta_loss_v_layer_heatmap_abs_limit",
            "0.125",
            "--instrumentation__delta_loss_v_layer_heatmap_log_every_n_probes",
            "400",
        ]
    )
    assert arguments.instrumentation__delta_loss_v_layer_heatmap is True
    assert arguments.instrumentation__delta_loss_v_layer_heatmap_abs_limit == pytest.approx(0.125)
    assert arguments.instrumentation__delta_loss_v_layer_heatmap_log_every_n_probes == 400

    defaults = runner.build_parser().parse_args([])
    assert defaults.instrumentation__delta_loss_v_layer_heatmap is False
    assert defaults.instrumentation__delta_loss_v_layer_heatmap_abs_limit == pytest.approx(0.05)
    assert defaults.instrumentation__delta_loss_v_layer_heatmap_log_every_n_probes == 250

    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(
            ["--instrumentation__delta_loss_v_layer_heatmap", "yes"]
        )


def test_run_config_rejects_ineffective_or_unbounded_heatmap_controls() -> None:
    config = _enabled_config()
    canonical = config.canonical_dict(world_size=1)
    assert canonical["instrumentation__delta_loss_v_layer_heatmap"] is True
    assert canonical["instrumentation__delta_loss_v_layer_heatmap_abs_limit"] == pytest.approx(0.05)
    assert canonical["instrumentation__delta_loss_v_layer_heatmap_log_every_n_probes"] == 250

    with pytest.raises(ValueError, match="learned PLASTIC layer count"):
        OwtRunConfig(
            model_type="sheet",
            instrumentation__delta_loss_v_layer_heatmap=True,
        )
    with pytest.raises(ValueError, match="requires W&B"):
        _enabled_config(wandb_enabled=False)
    with pytest.raises(ValueError, match="finite and positive"):
        _enabled_config(instrumentation__delta_loss_v_layer_heatmap_abs_limit=0.0)
    with pytest.raises(ValueError, match="positive integer"):
        _enabled_config(
            instrumentation__delta_loss_v_layer_heatmap_log_every_n_probes=0
        )


def test_heatmap_uses_absolute_layers_fixed_missing_cells_and_exact_probe_rows() -> None:
    record = curves._probe_record_from_event(
        _event(completed_updates=8, probe_sequence=7)
    )
    assert record is not None
    assert record["probe_id"] == "P7"

    heatmap_record = curves._delta_loss_heatmap_record(record, maximum_layers=8)
    rendered = curves._delta_loss_heatmap_render_data(
        (heatmap_record,),
        maximum_layers=8,
    )

    assert rendered["x"] == tuple(range(1, 9))
    assert rendered["y"] == ("P7 | update 9 | active 4",)
    assert rendered["z"][0][:2] == (None, None)
    assert rendered["z"][0][2] == pytest.approx(-0.02)
    assert rendered["z"][0][3] == pytest.approx(0.0)
    assert rendered["z"][0][4] == pytest.approx(0.03)
    assert rendered["z"][0][5:] == (None, None, None)


def test_heatmap_progressively_decimates_to_512_exact_rows_and_keeps_endpoints() -> None:
    source = []
    for probe_sequence in range(1, 701):
        record = curves._probe_record_from_event(
            _event(
                completed_updates=probe_sequence * 2 - 1,
                probe_sequence=probe_sequence,
            )
        )
        assert record is not None
        source.append(curves._delta_loss_heatmap_record(record, maximum_layers=8))

    rendered = curves._delta_loss_heatmap_render_data(source, maximum_layers=8)

    assert rendered["source_rows"] == 700
    assert rendered["rendered_rows"] == 512
    assert rendered["y"][0].startswith("P1 | update 2")
    assert rendered["y"][-1].startswith("P700 | update 1400")
    assert len(set(rendered["y"])) == 512


def test_plotly_figure_has_fixed_symmetric_scale_requested_palette_and_active_trace() -> None:
    plotly = pytest.importorskip("plotly.graph_objects")
    record = curves._probe_record_from_event(
        _event(completed_updates=8, probe_sequence=7)
    )
    assert record is not None
    history = (curves._delta_loss_heatmap_record(record, maximum_layers=8),)

    figure = curves._delta_loss_heatmap_figure(
        history,
        maximum_layers=8,
        abs_limit=0.05,
        go_module=plotly,
    )

    heatmap, active = figure.data
    assert heatmap.zmin == pytest.approx(-0.05)
    assert heatmap.zmax == pytest.approx(0.05)
    assert heatmap.zmid == pytest.approx(0.0)
    assert heatmap.colorscale[0][1] == "rgb(0,255,0)"
    assert heatmap.colorscale[1][1] == "rgb(88,88,88)"
    assert heatmap.colorscale[-1][1] == "rgb(255,0,0)"
    assert active.line.color == "white"
    assert tuple(active.x) == (4,)


def test_heatmap_upload_cadence_is_early_then_every_250_probes() -> None:
    telemetry = SimpleNamespace(
        config={
            "instrumentation__delta_loss_v_layer_heatmap_log_every_n_probes": 250
        }
    )
    history = curves._ensure_delta_loss_heatmap_state(telemetry)

    for probe_sequence in range(1, 250):
        history.append({"probe_id": f"P{probe_sequence}"})
    telemetry._delta_loss_heatmap_last_logged_total = 100
    assert not curves._should_refresh_delta_loss_heatmap(telemetry)

    history.append({"probe_id": "P250"})
    assert curves._should_refresh_delta_loss_heatmap(telemetry)
    telemetry._delta_loss_heatmap_last_logged_total = 250

    history.extend({"probe_id": f"P{value}"} for value in range(251, 500))
    assert not curves._should_refresh_delta_loss_heatmap(telemetry)
    assert curves._should_refresh_delta_loss_heatmap(telemetry, force=True)


class _FakeRun:
    def __init__(self):
        self.calls = []

    def log(self, payload, step=None):
        self.calls.append((payload, step))


def test_explicit_heatmap_logs_without_activating_legacy_debug_charts(
    monkeypatch,
) -> None:
    monkeypatch.setattr(constants, "DEBUG", 0)
    monkeypatch.setattr(
        curves,
        "_delta_loss_heatmap_figure",
        lambda *_args, **_kwargs: "plotly-figure",
    )
    telemetry = SimpleNamespace(
        run=_FakeRun(),
        module=SimpleNamespace(),
        config={
            "instrumentation__delta_loss_v_layer_heatmap": True,
            "instrumentation__delta_loss_v_layer_heatmap_abs_limit": 0.05,
            "instrumentation__delta_loss_v_layer_heatmap_log_every_n_probes": 250,
        },
    )
    record = curves._probe_record_from_event(
        _event(completed_updates=8, probe_sequence=7)
    )
    assert record is not None
    curves._append_delta_loss_heatmap_records(
        telemetry,
        (record,),
        maximum_layers=8,
    )

    curves._log_rolling_probe_charts(
        telemetry,
        step=9,
        include_probe_charts=False,
        include_coefficient_chart=False,
        include_delta_loss_heatmap=True,
    )

    assert telemetry.run.calls == [
        ({curves._DELTA_LOSS_HEATMAP_KEY: "plotly-figure"}, 9)
    ]


def test_heatmap_only_attachment_scans_probe_events_without_sampling_coefficients(
    monkeypatch,
) -> None:
    monkeypatch.setattr(constants, "DEBUG", 0)
    monkeypatch.setattr(curves, "_ORIGINAL_ATTACH_TELEMETRY", lambda *_args: None)
    monkeypatch.setattr(
        curves,
        "_delta_loss_heatmap_figure",
        lambda *_args, **_kwargs: "plotly-figure",
    )
    def original_timed(function):
        return function(), 0.25

    trainer = SimpleNamespace(
        _timed=original_timed,
        train_one_update=lambda: {},
        _print_progress=lambda *_args, **_kwargs: None,
        distributed=SimpleNamespace(is_primary=True),
        config=SimpleNamespace(plastic__enabled=True, n_layer=8),
        state=SimpleNamespace(completed_updates=9),
        events=[_event(completed_updates=8, probe_sequence=7)],
    )
    telemetry = SimpleNamespace(
        run=_FakeRun(),
        module=SimpleNamespace(),
        config={
            "instrumentation__delta_loss_v_layer_heatmap": True,
            "instrumentation__delta_loss_v_layer_heatmap_abs_limit": 0.05,
            "instrumentation__delta_loss_v_layer_heatmap_log_every_n_probes": 250,
        },
    )

    curves.attach_telemetry_with_plastic_probe_curves(trainer, telemetry)
    trainer._print_progress("run", "optimizer_progress", completed_updates=9)

    assert trainer._timed is original_timed
    assert not hasattr(telemetry, "_plastic_coefficient_curve_history")
    assert len(telemetry._delta_loss_heatmap_history) == 1
    assert telemetry.run.calls == [
        ({curves._DELTA_LOSS_HEATMAP_KEY: "plotly-figure"}, 9)
    ]
# ^^^ THOG
