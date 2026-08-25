from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import run_thog2_local_dashboard as dashboard
from sheet import local_chart_store as local_store
from sheet import wandb_telemetry
from sheet import local_chart_lifecycle_patch as lifecycle
from sheet import depth_weight_curves_and_observational_probes_patch as depth_curves
from sheet.local_chart_store import LocalChartStore, ensure_local_chart_store


def test_finite_capture_count_includes_both_explicit_boundaries() -> None:
    assert lifecycle._finite_capture_count(
        start_step=300,
        end_step=400,
        cadence=1,
    ) == 101
    assert lifecycle._finite_capture_count(
        start_step=300,
        end_step=405,
        cadence=100,
    ) == 3


def test_capture_window_promotes_retention_instead_of_aborting(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv(depth_curves._environment_name("TIME_MODE"), "accumulate")
    monkeypatch.setenv(depth_curves._environment_name("HISTORY_LENGTH"), "100")
    monkeypatch.setenv(depth_curves._environment_name("LOG_EVERY_N_STEPS"), "1")
    monkeypatch.setenv(depth_curves._environment_name("START_STEP"), "300")
    monkeypatch.setenv(depth_curves._environment_name("END_STEP"), "400")

    assert lifecycle._ensure_capture_retention() == 101
    assert depth_curves._history_length() == 101
    assert "from 100 to 101" in capsys.readouterr().out

    monkeypatch.setenv(depth_curves._environment_name("HISTORY_LENGTH"), "101")
    assert lifecycle._ensure_capture_retention() == 101
    assert capsys.readouterr().out == ""


def test_latest_mode_preserves_requested_retention(monkeypatch, capsys) -> None:
    monkeypatch.setenv(depth_curves._environment_name("TIME_MODE"), "latest")
    monkeypatch.setenv(depth_curves._environment_name("HISTORY_LENGTH"), "100")
    monkeypatch.setenv(depth_curves._environment_name("LOG_EVERY_N_STEPS"), "1")
    monkeypatch.setenv(depth_curves._environment_name("START_STEP"), "300")
    monkeypatch.setenv(depth_curves._environment_name("END_STEP"), "400")

    assert lifecycle._ensure_capture_retention() == 100
    assert depth_curves._history_length() == 100
    assert capsys.readouterr().out == ""


def test_weight_instrumentation_phase_is_independent_of_plastic_configuration() -> None:
    assert lifecycle._weight_phase(11, start_step=300, end_step=400) == "preparing"
    assert lifecycle._weight_phase(300, start_step=300, end_step=400) == "recording"
    assert lifecycle._weight_phase(400, start_step=300, end_step=400) == "recording"
    assert lifecycle._weight_phase(401, start_step=300, end_step=400) == "monitoring"


def test_eager_attach_registers_run_before_first_weight_snapshot(monkeypatch) -> None:
    calls = []

    class FakeStore:
        _has_recorded_data = False
        _has_heatmap_records = False

        def configure_weight_capture(self, **values) -> None:
            calls.append(("configure", values))

        def heartbeat(self, optimizer_update, *, run_state, force=False) -> None:
            calls.append(("heartbeat", optimizer_update, run_state, force))

    class FakeTrainer:
        def __init__(self) -> None:
            self.raw_model = object()
            self.state = SimpleNamespace(completed_updates=11)
            self.distributed = SimpleNamespace(is_primary=True)

        def train_one_update(self):
            return {}

        def _timed(self, function):
            return function(), 0.0

    telemetry = SimpleNamespace(
        config={},
        _thog_local_chart_store=FakeStore(),
    )
    trainer = FakeTrainer()
    monkeypatch.setattr(lifecycle, "_ORIGINAL_ATTACH_TELEMETRY", lambda *_args: None)
    monkeypatch.setattr(lifecycle, "_weight_curves_supported", lambda _trainer: True)
    monkeypatch.setattr(lifecycle, "ensure_local_chart_store", lambda _telemetry: telemetry._thog_local_chart_store)
    monkeypatch.setattr(depth_curves, "_destination", lambda: "local")
    monkeypatch.setenv(depth_curves._environment_name("TIME_MODE"), "accumulate")
    monkeypatch.setenv(depth_curves._environment_name("HISTORY_LENGTH"), "100")
    monkeypatch.setenv(depth_curves._environment_name("LOG_EVERY_N_STEPS"), "1")
    monkeypatch.setattr(lifecycle, "_capture_bounds", lambda: (300, 400))

    lifecycle._attach_telemetry_with_local_lifecycle(trainer, telemetry)

    assert calls == [
        (
            "configure",
            {
                "start_step": 300,
                "end_step": 400,
                "cadence": 1,
                "history_length": 101,
            },
        ),
        ("heartbeat", 11, "preparing", True),
    ]
    assert depth_curves._history_length() == 101


def test_disabled_weight_instrumentation_does_not_promote_retention(monkeypatch) -> None:
    class FakeTrainer:
        raw_model = object()

    telemetry = SimpleNamespace(config={})
    monkeypatch.setattr(lifecycle, "_ORIGINAL_ATTACH_TELEMETRY", lambda *_args: None)
    monkeypatch.setattr(lifecycle, "_weight_curves_supported", lambda _trainer: True)
    monkeypatch.setattr(depth_curves, "_destination", lambda: "none")
    monkeypatch.setenv(depth_curves._environment_name("TIME_MODE"), "accumulate")
    monkeypatch.setenv(depth_curves._environment_name("HISTORY_LENGTH"), "100")
    monkeypatch.setenv(depth_curves._environment_name("LOG_EVERY_N_STEPS"), "1")
    monkeypatch.setenv(depth_curves._environment_name("START_STEP"), "300")
    monkeypatch.setenv(depth_curves._environment_name("END_STEP"), "400")

    lifecycle._attach_telemetry_with_local_lifecycle(FakeTrainer(), telemetry)

    assert depth_curves._history_length() == 100


def test_heatmap_only_capture_enters_recording_phase(tmp_path: Path) -> None:
    store = LocalChartStore(
        tmp_path / "charts.sqlite3",
        run_name="heatmap-only",
        config={},
    )
    assert store._has_heatmap_records is False

    store.append_heatmap_records((
        {
            "optimizer_update": 7,
            "probe_id": "probe-7",
            "active_layers": 8,
            "selected_layers": 8,
            "shrink": ((1, -0.01, 7, -1),),
            "growth": ((1, 0.01, 9, 1),),
        },
    ))

    assert store._has_heatmap_records is True
    assert lifecycle._active_phase(
        store,
        7,
        weight_local=False,
        heatmap_local=True,
        start_step=None,
        end_step=None,
    ) == "recording"
    store.close()
    assert local_store.LocalChartReader(store.path).metadata()["current_update"] == "7"


def test_local_chart_announcement_uses_complete_dashboard_launcher(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("THOG2_INSTRUMENTATION_LOCAL_ROOT", str(tmp_path))
    telemetry = SimpleNamespace(
        name="launcher-test",
        run=None,
        config={},
    )

    ensure_local_chart_store(telemetry)

    output = capsys.readouterr().out
    assert "python -m run_thog2_dashboard --run " in output
    assert "python -m run_thog2_local_dashboard --run " not in output
    telemetry._thog_local_chart_store.close()


def test_terminal_state_flushes_latest_throttled_model_update(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monotonic_values = iter((10.0, 10.1))
    monkeypatch.setattr(local_store.time, "monotonic", lambda: next(monotonic_values))
    path = tmp_path / "charts.sqlite3"
    store = LocalChartStore(path, run_name="fast-run", config={})

    store.heartbeat(1, run_state="recording", force=True)
    store.heartbeat(2, run_state="recording")
    assert local_store.LocalChartReader(path).metadata()["current_update"] == "1"

    store.close(final_state="finished")

    metadata = local_store.LocalChartReader(path).metadata()
    assert metadata["current_update"] == "2"
    assert metadata["run_state"] == "finished"


def test_telemetry_finish_closes_remaining_sinks_after_local_store_failure(
    monkeypatch,
    capsys,
) -> None:
    calls = []

    class FakeRun:
        def finish(self, **_kwargs) -> None:
            calls.append("wandb")

    class FakeWriter:
        def flush(self) -> None:
            calls.append("flush")

        def close(self) -> None:
            calls.append("tensorboard")

    telemetry = SimpleNamespace(
        run=FakeRun(),
        writer=FakeWriter(),
    )
    monkeypatch.setattr(
        wandb_telemetry,
        "close_local_chart_store",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk failure")),
    )

    wandb_telemetry.WandbTelemetry.finish(telemetry)

    assert calls == ["wandb", "flush", "tensorboard"]
    assert telemetry.run is None
    assert telemetry.writer is None
    assert "local chart store could not be closed" in capsys.readouterr().out


def test_status_uses_heartbeat_and_reports_evidence_based_data_loss(tmp_path: Path) -> None:
    path = tmp_path / "run" / "charts.sqlite3"
    store = LocalChartStore(path, run_name="run", config={})
    store.configure_weight_capture(
        start_step=300,
        end_step=400,
        cadence=1,
        history_length=101,
    )
    state = dashboard.RunDashboardState(path)

    store.heartbeat(11, run_state="preparing", force=True)
    preparing = state.status()
    assert preparing["run_state"] == "preparing"
    assert preparing["maximum_update"] == 11
    assert preparing["data_lost"] is False

    revision = preparing["revision"]
    store.heartbeat(300, run_state="recording", force=True)
    missing = state.status()
    assert missing["maximum_update"] == 300
    assert missing["data_lost"] is True
    assert missing["revision"] == revision

    store.append_depth_weight_snapshot(
        {"optimizer_update": 300, "families": {}},
        history_length=101,
    )
    recorded = state.status()
    assert recorded["data_lost"] is False
    assert recorded["chart_maximum_update"] == 300

    store.heartbeat(401, run_state="monitoring", force=True)
    monitoring = state.status()
    assert monitoring["run_state"] == "monitoring"
    assert monitoring["maximum_update"] == 401

    store.close(final_state="stopped")
    assert state.status()["run_state"] == "stopped"
