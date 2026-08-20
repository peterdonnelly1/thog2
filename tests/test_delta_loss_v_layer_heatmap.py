# vvv THOG
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import constants
import pytest

import run_thog2_lifecycle as lifecycle
import run_thog2_owt_core as runner
from sheet import depth_weight_curves_and_observational_probes_patch as depth_probes
from sheet import depth_observational_probe_wandb_patch as observational_wandb
from sheet import plastic_depth_wandb_probe_curves_patch as curves
from sheet.local_chart_store import LocalChartReader, LocalChartStore, close_local_chart_store
from sheet.run_config import OwtRunConfig
from sheet.stage6_trainer import _linear_heatmap_probe_progress_due
from sheet.trainer_state import TrainerEvent


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _event(
    *,
    completed_updates: int,
    probe_sequence: int,
    current: int = 4,
    radius: int = 1,
    observational_only: bool = False,
    selected: int | None = None,
    brake_active: bool = False,
) -> TrainerEvent:
    candidates = tuple(
        {
            "active_layers": candidate,
            "validation_loss": 5.0 + 0.01 * (candidate - current),
        }
        for candidate in range(max(1, current - radius), current + radius + 1)
    )
    return TrainerEvent(
        "plastic_depth_count_decision",
        completed_updates,
        {
            "previous_active_layers": current,
            "selected_active_layers": current if selected is None else selected,
            "brake_active": brake_active,
            "probe_sequence": probe_sequence,
            "probe_update": completed_updates + 1,
            "observational_only": observational_only,
            "candidates": candidates,
        },
    )


def _heatmap_config(*, plastic_enabled: bool, do_learn_layer_count: bool) -> OwtRunConfig:
    values = {
        "model_type": "sheet",
        "n_layer": 4,
        "n_head": 2,
        "n_embd": 16,
        "o_depth": 4,
        "plastic__enabled": plastic_enabled,
        "plastic__do_learn_layer_count": do_learn_layer_count,
        "instrumentation__delta_loss_v_layer_heatmap": "log",
    }
    if plastic_enabled and do_learn_layer_count:
        values["plastic__initial_layer_count"] = 4
        values["plastic__max_permitted_layers"] = 8
    return OwtRunConfig(**values)


def test_cli_uses_log_or_linear_modes_and_viewer_owns_history_window() -> None:
    arguments = runner.build_parser().parse_args(
        [
            "--instrumentation__delta_loss_v_layer_heatmap",
            "linear",
            "--instrumentation__delta_loss_v_layer_heatmap__destination",
            "wandb",
            "--instrumentation__delta_loss_v_layer_heatmap_abs_limit",
            "0.125",
            "--instrumentation__delta_loss_v_layer_heatmap_log_every_n_probes",
            "400",
        ]
    )
    assert arguments.instrumentation__delta_loss_v_layer_heatmap == "linear"
    assert arguments.instrumentation__delta_loss_v_layer_heatmap__destination == "wandb"
    assert not hasattr(arguments, "instrumentation__delta_loss_v_layer_heatmap_linear")
    assert arguments.instrumentation__delta_loss_v_layer_heatmap_abs_limit == pytest.approx(0.125)
    assert arguments.instrumentation__delta_loss_v_layer_heatmap_log_every_n_probes == 400

    defaults = runner.build_parser().parse_args([])
    assert defaults.instrumentation__delta_loss_v_layer_heatmap is None
    assert defaults.instrumentation__delta_loss_v_layer_heatmap__destination == "local"
    assert not hasattr(defaults, "instrumentation__delta_loss_v_layer_heatmap_linear")
    assert defaults.instrumentation__delta_loss_v_layer_heatmap_abs_limit == pytest.approx(0.05)
    assert defaults.instrumentation__delta_loss_v_layer_heatmap_log_every_n_probes == 250

    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(
            ["--instrumentation__delta_loss_v_layer_heatmap", "true"]
        )
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(
            ["--instrumentation__delta_loss_v_layer_heatmap_linear", "500"]
        )


def test_heatmap_controls_are_operational_across_resume_and_fork() -> None:
    arguments = lifecycle.build_parser().parse_args(
        [
            "--instrumentation__delta_loss_v_layer_heatmap",
            "linear",
            "--instrumentation__delta_loss_v_layer_heatmap__destination",
            "local",
            "--instrumentation__delta_loss_v_layer_heatmap_abs_limit",
            "0.075",
            "--instrumentation__delta_loss_v_layer_heatmap_log_every_n_probes",
            "300",
        ]
    )
    assert arguments.instrumentation__delta_loss_v_layer_heatmap == "linear"
    assert arguments.instrumentation__delta_loss_v_layer_heatmap__destination == "local"
    for destination in (
        "instrumentation__delta_loss_v_layer_heatmap",
        "instrumentation__delta_loss_v_layer_heatmap__destination",
        "instrumentation__delta_loss_v_layer_heatmap_abs_limit",
        "instrumentation__delta_loss_v_layer_heatmap_log_every_n_probes",
    ):
        assert destination in lifecycle._OPERATIONAL_CONFIG_DESTINATIONS


@pytest.mark.parametrize(
    ("plastic_enabled", "do_learn_layer_count"),
    ((False, False), (False, True), (True, False), (True, True)),
)
def test_heatmap_configuration_is_independent_of_plastic_mutation_switches(
    plastic_enabled: bool,
    do_learn_layer_count: bool,
) -> None:
    config = _heatmap_config(
        plastic_enabled=plastic_enabled,
        do_learn_layer_count=do_learn_layer_count,
    )
    canonical = config.canonical_dict(world_size=1)
    assert canonical["instrumentation__delta_loss_v_layer_heatmap"] == "log"
    assert canonical["instrumentation__delta_loss_v_layer_heatmap__destination"] == "local"


def test_run_config_accepts_local_without_wandb_and_rejects_invalid_controls() -> None:
    local = OwtRunConfig(
        model_type="sheet",
        wandb_enabled=False,
        wandb_mode="disabled",
        instrumentation__delta_loss_v_layer_heatmap="log",
        instrumentation__delta_loss_v_layer_heatmap__destination="local",
    )
    assert local.instrumentation__delta_loss_v_layer_heatmap__destination == "local"
    with pytest.raises(ValueError, match="requires W&B"):
        OwtRunConfig(
            model_type="sheet",
            wandb_enabled=False,
            instrumentation__delta_loss_v_layer_heatmap="log",
            instrumentation__delta_loss_v_layer_heatmap__destination="wandb",
        )
    with pytest.raises(ValueError, match="positive optimizer step"):
        OwtRunConfig(
            model_type="sheet",
            instrumentation__delta_loss_v_layer_heatmap_linear=0,
        )
    with pytest.raises(ValueError, match="finite and positive"):
        OwtRunConfig(
            model_type="sheet",
            instrumentation__delta_loss_v_layer_heatmap_abs_limit=0.0,
        )
    with pytest.raises(ValueError, match="positive integer"):
        OwtRunConfig(
            model_type="sheet",
            instrumentation__delta_loss_v_layer_heatmap_log_every_n_probes=0,
        )


def test_heatmap_uses_absolute_layers_and_expands_to_observed_growth_candidates() -> None:
    telemetry = SimpleNamespace()
    first = curves._probe_record_from_event(
        _event(completed_updates=8, probe_sequence=7, current=4, radius=1)
    )
    wide = curves._probe_record_from_event(
        _event(
            completed_updates=18,
            probe_sequence=8,
            current=48,
            radius=12,
            observational_only=True,
        )
    )
    assert first is not None
    assert wide is not None

    curves._append_delta_loss_heatmap_records(telemetry, (first,), maximum_layers=8)
    curves._append_delta_loss_heatmap_records(telemetry, (wide,), maximum_layers=60)
    rendered = curves._delta_loss_heatmap_render_data(
        telemetry._delta_loss_heatmap_history,
        maximum_layers=telemetry._delta_loss_heatmap_maximum_layers,
    )

    assert rendered["x"] == (1, 2)
    assert rendered["x_steps"] == (9, 19)
    assert rendered["y"] == tuple(range(1, 61))
    assert len(rendered["z"]) == 60
    assert rendered["z"][8] == (None, None)
    assert rendered["z"][35][1] == pytest.approx(-0.12)
    assert rendered["z"][47][1] == pytest.approx(0.0)
    assert rendered["z"][59][1] == pytest.approx(0.12)


def test_heatmap_records_decision_brake_and_chaos_runtime_metadata() -> None:
    telemetry = SimpleNamespace()
    trainer = SimpleNamespace(events=[
        TrainerEvent(
            "chaos_bump_sampling_started",
            9,
            {
                "start_update": 10,
                "end_update": 11,
                "duration_steps": 2,
                "movement_fractions": (0.01, 0.05, 0.03),
            },
        ),
        _event(
            completed_updates=9,
            probe_sequence=1,
            current=4,
            selected=5,
            brake_active=True,
        ),
        TrainerEvent(
            "chaos_bump_sampling_ended",
            10,
            {"completed_update": 10},
        ),
        _event(completed_updates=10, probe_sequence=2, current=5),
    ])

    records = curves._consume_new_delta_loss_heatmap_records(trainer, telemetry)

    assert len(records) == 2
    assert records[0]["decision_committed"] is True
    assert records[0]["brake_active"] is True
    assert records[0]["chaos_bump"] == {
        "state": "active",
        "magnitude_percent": pytest.approx(5.0),
        "step": 1,
        "duration": 2,
    }
    assert records[1]["chaos_bump"] == {"state": "reverted"}


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
    assert rendered["probe_labels"][0].startswith("P1 | update 2")
    assert rendered["probe_labels"][-1].startswith("P700 | update 1400")
    assert len(set(rendered["probe_labels"])) == 512


def test_plotly_figure_has_fixed_scale_white_square_step_layout_and_active_trace() -> None:
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
    assert figure.layout.paper_bgcolor == "white"
    assert figure.layout.plot_bgcolor == "white"
    assert figure.layout.xaxis.title.text == "step"
    assert tuple(figure.layout.xaxis.ticktext) == ("9",)
    assert figure.layout.yaxis.title.text == "absolute candidate layer count"
    assert figure.layout.yaxis.scaleanchor == "x"
    assert figure.layout.yaxis.scaleratio == pytest.approx(1.0)
    assert len(heatmap.z) == 8
    assert all(len(layer_row) == 1 for layer_row in heatmap.z)
    assert active.line.color == "black"
    assert tuple(active.x) == (1,)
    assert tuple(active.y) == (4,)


def test_heatmap_upload_cadence_is_early_then_every_250_probes() -> None:
    telemetry = SimpleNamespace(
        config={
            "instrumentation__delta_loss_v_layer_heatmap": "log",
            "instrumentation__delta_loss_v_layer_heatmap__destination": "wandb",
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


def test_linear_heatmap_uploads_every_probe_without_a_capture_cap() -> None:
    telemetry = SimpleNamespace(
        config={
            "instrumentation__delta_loss_v_layer_heatmap": "linear",
            "instrumentation__delta_loss_v_layer_heatmap_linear": 20,
        }
    )
    history = curves._ensure_delta_loss_heatmap_state(telemetry)
    history.append({"probe_id": "P1", "optimizer_update": 10})
    assert curves._should_refresh_delta_loss_heatmap(telemetry)

    telemetry._delta_loss_heatmap_last_logged_total = 1
    history.append({"probe_id": "P2", "optimizer_update": 20})
    assert curves._should_refresh_delta_loss_heatmap(telemetry)

    telemetry._delta_loss_heatmap_last_logged_total = 2
    history.append({"probe_id": "P3", "optimizer_update": 30})
    assert curves._should_refresh_delta_loss_heatmap(telemetry)
    assert curves._should_refresh_delta_loss_heatmap(telemetry, force=True)

    telemetry.config["instrumentation__delta_loss_v_layer_heatmap_linear"] = None
    assert curves._should_refresh_delta_loss_heatmap(telemetry)


def test_linear_heatmap_promotes_every_probe_step_without_a_capture_cap() -> None:
    config = SimpleNamespace(
        instrumentation__delta_loss_v_layer_heatmap="linear",
        instrumentation__delta_loss_v_layer_heatmap_linear=20,
    )
    assert _linear_heatmap_probe_progress_due(
        config=config,
        completed_updates=20,
        probe_update=20,
    )
    assert _linear_heatmap_probe_progress_due(
        config=config,
        completed_updates=21,
        probe_update=21,
    )
    assert not _linear_heatmap_probe_progress_due(
        config=config,
        completed_updates=19,
        probe_update=None,
    )

    config.instrumentation__delta_loss_v_layer_heatmap_linear = None
    assert _linear_heatmap_probe_progress_due(
        config=config,
        completed_updates=1_000_000,
        probe_update=1_000_000,
    )

    config.instrumentation__delta_loss_v_layer_heatmap = "log"
    assert not _linear_heatmap_probe_progress_due(
        config=config,
        completed_updates=20,
        probe_update=20,
    )


def test_observational_probe_marks_its_actual_optimizer_step(monkeypatch) -> None:
    monkeypatch.setattr(
        depth_probes,
        "_ORIGINAL_TRAIN_ONE_UPDATE",
        lambda _trainer: {"skipped_update": 0.0},
    )
    monkeypatch.setattr(depth_probes, "_observational_probe_enabled", lambda _trainer: True)
    monkeypatch.setattr(
        depth_probes,
        "_observational_probe_due",
        lambda _trainer, update: update == 37,
    )
    observed = []
    monkeypatch.setattr(
        depth_probes,
        "_run_observational_probe",
        lambda _trainer, *, update: observed.append(update),
    )
    trainer = SimpleNamespace(state=SimpleNamespace(completed_updates=37))

    metrics = depth_probes._train_one_update_with_observational_probe(trainer)

    assert metrics == {"skipped_update": 0.0}
    assert observed == [37]
    assert trainer._depth_probe_optimizer_update == 37


class _FakeRun:
    def __init__(self) -> None:
        self.calls = []

    def log(self, payload, step=None) -> None:
        self.calls.append((payload, step))


@pytest.mark.parametrize(
    ("plastic_enabled", "do_learn_layer_count", "observational_enabled"),
    (
        (False, False, True),
        (False, True, True),
        (True, False, True),
        (True, True, False),
    ),
)
def test_attachment_logs_heatmap_in_all_plastic_modes_without_legacy_debug_charts(
    monkeypatch,
    plastic_enabled: bool,
    do_learn_layer_count: bool,
    observational_enabled: bool,
) -> None:
    monkeypatch.setattr(constants, "DEBUG", 0)
    monkeypatch.setattr(
        observational_wandb,
        "_ORIGINAL_ATTACH_TELEMETRY",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        observational_wandb._depth,
        "_observational_probe_enabled",
        lambda _trainer: observational_enabled,
    )
    captured = {}

    def fake_figure(_history, *, maximum_layers, abs_limit):
        captured["maximum_layers"] = maximum_layers
        captured["abs_limit"] = abs_limit
        return "plotly-figure"

    monkeypatch.setattr(curves, "_delta_loss_heatmap_figure", fake_figure)
    trainer = SimpleNamespace(
        _print_progress=lambda *_args, **_kwargs: None,
        distributed=SimpleNamespace(is_primary=True),
        config=SimpleNamespace(
            n_layer=48,
            plastic__enabled=plastic_enabled,
            plastic__do_learn_layer_count=do_learn_layer_count,
        ),
        state=SimpleNamespace(completed_updates=10),
        events=[
            _event(
                completed_updates=9,
                probe_sequence=1,
                current=48,
                radius=12,
                observational_only=True,
            )
        ],
    )
    telemetry = SimpleNamespace(
        run=_FakeRun(),
        module=SimpleNamespace(),
        config={
            "instrumentation__delta_loss_v_layer_heatmap": "log",
            "instrumentation__delta_loss_v_layer_heatmap__destination": "wandb",
            "instrumentation__delta_loss_v_layer_heatmap_abs_limit": 0.05,
            "instrumentation__delta_loss_v_layer_heatmap_log_every_n_probes": 250,
        },
    )

    observational_wandb._attach_telemetry_with_observational_probe_charts(
        trainer,
        telemetry,
    )
    trainer._print_progress("run", "optimizer_progress", completed_updates=10)

    assert captured == {"maximum_layers": 60, "abs_limit": pytest.approx(0.05)}
    assert telemetry.run.calls == [
        ({curves._DELTA_LOSS_HEATMAP_KEY: "plotly-figure"}, 10)
    ]


@pytest.mark.parametrize(
    ("plastic_enabled", "do_learn_layer_count", "observational_enabled"),
    (
        (False, False, True),
        (False, True, True),
        (True, False, True),
        (True, True, False),
    ),
)
def test_attachment_writes_local_heatmap_without_wandb(
    monkeypatch,
    tmp_path: Path,
    plastic_enabled: bool,
    do_learn_layer_count: bool,
    observational_enabled: bool,
) -> None:
    monkeypatch.setenv("THOG2_INSTRUMENTATION_LOCAL_ROOT", str(tmp_path))
    monkeypatch.setattr(
        observational_wandb,
        "_ORIGINAL_ATTACH_TELEMETRY",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        observational_wandb._depth,
        "_observational_probe_enabled",
        lambda _trainer: observational_enabled,
    )
    trainer = SimpleNamespace(
        _print_progress=lambda *_args, **_kwargs: None,
        distributed=SimpleNamespace(is_primary=True),
        config=SimpleNamespace(
            n_layer=48,
            plastic__enabled=plastic_enabled,
            plastic__do_learn_layer_count=do_learn_layer_count,
        ),
        state=SimpleNamespace(completed_updates=10),
        events=[
            _event(
                completed_updates=9,
                probe_sequence=1,
                current=48,
                radius=12,
                observational_only=True,
            )
        ],
    )
    telemetry = SimpleNamespace(
        name="local_heatmap_test",
        run=_FakeRun(),
        module=SimpleNamespace(),
        config={
            "instrumentation__delta_loss_v_layer_heatmap": "linear",
            "instrumentation__delta_loss_v_layer_heatmap__destination": "local",
            "instrumentation__delta_loss_v_layer_heatmap_linear": None,
            "instrumentation__delta_loss_v_layer_heatmap_abs_limit": 0.05,
        },
    )

    observational_wandb._attach_telemetry_with_observational_probe_charts(
        trainer,
        telemetry,
    )
    trainer._print_progress("run", "optimizer_progress", completed_updates=10)

    store = telemetry._thog_local_chart_store
    reader = LocalChartReader(store.path)
    status = reader.status()
    history = reader.heatmap_history()
    assert status["heatmap_count"] == 1
    assert status["heatmap_maximum_update"] == 10
    assert len(history[0]["values"]) == 60
    assert history[0]["values"][47] == pytest.approx(0.0)
    assert telemetry.run.calls == []
    close_local_chart_store(telemetry)


def test_local_heatmap_reader_selects_earliest_or_latest_probe_window(tmp_path: Path) -> None:
    store = LocalChartStore(
        tmp_path / "charts.sqlite3",
        run_name="window-test",
        config={},
    )
    store.append_heatmap_records(
        {
            "probe_id": f"P{step}",
            "optimizer_update": step * 3,
            "active_layers": 4,
            "selected_layers": 4,
            "current_loss": 5.0,
            "shrink": ((0, 0.0, 4, 0),),
            "growth": ((0, 0.0, 4, 0), (1, -0.01, 5, 1)),
        }
        for step in range(1, 6)
    )
    reader = LocalChartReader(store.path)

    earliest = reader.heatmap_history_window(probe_count=2, window_mode="from_zero")
    latest = reader.heatmap_history_window(probe_count=2, window_mode="rolling")

    assert tuple(row["optimizer_update"] for row in earliest) == (3, 6)
    assert tuple(row["optimizer_update"] for row in latest) == (12, 15)
    store.close()


def test_wrapper_accepts_dormant_plastic_controls_and_forwards_both_instrumentation_surfaces() -> None:
    environment = dict(os.environ)
    environment["THOG2_PYTHON"] = sys.executable
    result = subprocess.run(
        [
            "bash",
            "train_OWT.sh",
            "-x",
            "true",
            "-I",
            "wandb",
            "-g",
            "FIXED_OBSERVATIONAL_HEATMAP",
            "-n",
            "2",
            "-w",
            "0",
            "-b",
            "1",
            "-A",
            "1",
            "-L",
            "4",
            "-H",
            "2",
            "-D",
            "8",
            "-C",
            "8",
            "-P",
            "4",
            "-S",
            "1",
            "--no-plastic__enabled",
            "--no-plastic__do_learn_layer_count",
            "--plastic__initial_layer_count",
            "4",
            "--plastic__max_permitted_layers",
            "8",
            "--plastic__layer_count_probe__probe_every_n_steps",
            "10",
            "--plastic__layer_count_probe__number_of_sampled_valid_tokens",
            "0",
            "--plastic__layer_count_probe_radius",
            "2",
            "--",
            "--instrumentation__depth_weight_curves__scalar_weights_per_matrix",
            "1",
            "--instrumentation__depth_weight_curves__depth_evaluation_points",
            "16",
            "--instrumentation__depth_weight_curves__time_mode",
            "accumulate",
            "--instrumentation__depth_weight_curves__history_length",
            "10",
            "--instrumentation__depth_weight_curves__log_every_n_steps",
            "1",
            "--instrumentation__depth_weight_curves__same_coordinates_all_runs",
            "true",
            "--instrumentation__depth_weight_curves__destination",
            "local",
            "--instrumentation__delta_loss_v_layer_heatmap",
            "linear",
            "--instrumentation__delta_loss_v_layer_heatmap__destination",
            "local",
            "--instrumentation__delta_loss_v_layer_heatmap_abs_limit",
            "0.05",
            "--instrumentation__delta_loss_v_layer_heatmap_log_every_n_probes",
            "250",
        ],
        cwd=_REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "initial/max layer count controls require" not in result.stderr
    dry_run = result.stdout.split("DRY RUN:", 1)[1]
    for expected in (
        "--no-plastic__enabled",
        "--no-plastic__do_learn_layer_count",
        "--plastic__layer_count_probe__probe_every_n_steps 10",
        "--plastic__layer_count_probe__number_of_sampled_valid_tokens 0",
        "--plastic__layer_count_probe_radius 2",
        "--instrumentation__depth_weight_curves__scalar_weights_per_matrix 1",
        "--instrumentation__depth_weight_curves__destination local",
        "--instrumentation__delta_loss_v_layer_heatmap linear",
        "--instrumentation__delta_loss_v_layer_heatmap__destination local",
    ):
        assert expected in dry_run
# ^^^ THOG
