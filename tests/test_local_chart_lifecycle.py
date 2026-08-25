from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import run_thog2_local_dashboard as dashboard
from sheet import local_chart_lifecycle_patch as lifecycle
from sheet import depth_weight_curves_and_observational_probes_patch as depth_curves
from sheet.local_chart_store import LocalChartStore


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


def test_capture_window_rejects_silent_history_truncation(monkeypatch) -> None:
    monkeypatch.setenv(depth_curves._environment_name("TIME_MODE"), "accumulate")
    monkeypatch.setenv(depth_curves._environment_name("HISTORY_LENGTH"), "100")
    monkeypatch.setenv(depth_curves._environment_name("LOG_EVERY_N_STEPS"), "1")
    monkeypatch.setenv(depth_curves._environment_name("START_STEP"), "300")
    monkeypatch.setenv(depth_curves._environment_name("END_STEP"), "400")

    with pytest.raises(ValueError, match="requires at least 101, got 100"):
        lifecycle._validate_capture_retention()

    monkeypatch.setenv(depth_curves._environment_name("HISTORY_LENGTH"), "101")
    lifecycle._validate_capture_retention()


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
    monkeypatch.setattr(depth_curves, "_time_mode", lambda: "accumulate")
    monkeypatch.setattr(depth_curves, "_history_length", lambda: 101)
    monkeypatch.setattr(depth_curves, "_log_every_n_steps", lambda: 1)
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
