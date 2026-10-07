"""Model activity must recover after a timeout, including existing WIDTH jobs."""
from types import SimpleNamespace

import pytest

import run_thog2_local_dashboard as dashboard
from sheet import local_chart_store
from sheet.local_dashboard_responsiveness import install


OLD = "2026-10-07T00:00:00+00:00"
FRESH = "2026-10-07T00:30:00+00:00"


def metadata(store, **values):
    store.connection.executemany("INSERT OR REPLACE INTO metadata VALUES (?, ?)", values.items())
    store.connection.commit()


def test_loss_only_progress_refreshes_the_writer_heartbeat(tmp_path, monkeypatch):
    clock = [OLD]
    monkeypatch.setattr(local_chart_store, "_utc_timestamp", lambda: clock[0])
    store = local_chart_store.LocalChartStore(tmp_path / "charts.sqlite3", run_name="width", config={"geometry_preset": "width"})
    try:
        clock[0] = FRESH
        store.append_training_loss(10, 6.2)
        assert local_chart_store.LocalChartReader(store.path).metadata()["heartbeat_at"] == FRESH
        status = dashboard.RunDashboardState(store.path).status()
        assert status["heartbeat_at"] == FRESH
        assert status["updated_at"] == FRESH
        assert status["maximum_update"] == 10
        assert status["run_state"] == "recording"
    finally:
        store.close()


def test_existing_loss_only_run_recovers_with_a_cached_status_reader(tmp_path):
    store = local_chart_store.LocalChartStore(tmp_path / "width" / "charts.sqlite3", run_name="width", config={})
    class State(dashboard.RunDashboardState):
        pass
    runtime = SimpleNamespace(RunDashboardState=State)
    install(runtime)
    state = State(store.path)
    try:
        metadata(store, run_state="recording", heartbeat_at=OLD, data_updated_at=OLD, updated_at=OLD)
        assert state.status()["heartbeat_at"] == OLD
        # This is the metadata left by a running job with the old writer code.
        store.append_training_loss(20, 5.9)
        metadata(store, heartbeat_at=OLD, data_updated_at=FRESH, updated_at=FRESH)
        recovered = state.status()
        assert recovered["heartbeat_at"] == FRESH
        assert recovered["updated_at"] == FRESH
        assert recovered["maximum_update"] == 20
        assert recovered["last_loss"] == 5.9
        assert state.status() == recovered
        # Re-reading or touching the database cannot manufacture model activity.
        store.path.touch()
        assert state.status()["heartbeat_at"] == FRESH
        assert local_chart_store.LocalChartReader(store.path).metadata()["heartbeat_at"] == OLD
    finally:
        store.close()


@pytest.mark.parametrize(("heartbeat", "data_time", "expected"), [
    (FRESH, OLD, FRESH),
    (OLD, FRESH, FRESH),
    ("2026-10-07T11:00:00+11:00", FRESH, FRESH),
    ("invalid", FRESH, FRESH),
    (OLD, "invalid", OLD),
])
def test_status_uses_the_newest_valid_model_timestamp(tmp_path, heartbeat, data_time, expected):
    store = local_chart_store.LocalChartStore(tmp_path / "charts.sqlite3", run_name="run", config={})
    try:
        metadata(store, heartbeat_at=heartbeat, data_updated_at=data_time, updated_at=OLD)
        assert dashboard.RunDashboardState(store.path).status()["heartbeat_at"] == expected
    finally:
        store.close()


@pytest.mark.parametrize(("final_state", "update", "expected"), [("finished", 50, "finished"), ("stopped", 20, "stopped")])
def test_fresh_data_does_not_reactivate_a_terminal_run(tmp_path, final_state, update, expected):
    store = local_chart_store.LocalChartStore(tmp_path / "charts.sqlite3", run_name="run", config={"max_iters": 50})
    store.append_training_loss(update, 5.8)
    store.close(final_state=final_state)
    status = dashboard.RunDashboardState(store.path).status()
    assert status["run_state"] == expected
