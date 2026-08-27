# vvv THOG
from __future__ import annotations

import threading
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace

from sheet.local_chart_store import LocalChartReader, LocalChartStore
from sheet import local_dashboard_weight_step_range_patch as step_range


def _snapshot(update: int) -> dict:
    return {
        "optimizer_update": update,
        "families": {"chart": {"curves": [{"scalar_id": "r0_c0", "x": [1], "y": [float(update)]}]}},
    }


class _Figure:
    def __init__(self, updates: list[int]) -> None:
        self.updates = updates

    def to_plotly_json(self) -> dict:
        return {"updates": self.updates}


def _fake_dashboard(builder_calls: list[tuple[int, ...]]):
    class FakeState:
        def __init__(self, reader: LocalChartReader) -> None:
            self.reader = reader
            self.lock = threading.Lock()

        def status(self) -> dict:
            value = self.reader.status()
            return {
                **value,
                "revision": [
                    value["depth_snapshot_count"],
                    value["depth_maximum_update"],
                ],
            }

    class BaseHandler:
        path = ""

        def __init__(self) -> None:
            self.sent = None

        def _send_json(self, value, status=HTTPStatus.OK) -> None:
            self.sent = (status, value)

        def do_GET(self) -> None:
            self.sent = (HTTPStatus.I_AM_A_TEAPOT, {"base": True})

    def build(snapshots, _chart_name):
        updates = tuple(int(snapshot["optimizer_update"]) for snapshot in snapshots)
        builder_calls.append(updates)
        return _Figure(list(updates))

    return SimpleNamespace(
        RunDashboardState=FakeState,
        _handler_for=lambda _catalog: BaseHandler,
        depth_curves=SimpleNamespace(
            _CHART_FAMILIES=("chart",),
            _build_depth_plotly_figure=build,
        ),
        HTTPStatus=HTTPStatus,
    )


def test_retained_weight_step_range_reads_only_requested_snapshots(tmp_path: Path) -> None:
    store = LocalChartStore(
        tmp_path / "charts.sqlite3",
        run_name="range_test",
        run_id="range_test",
        config={},
    )
    for update in (100, 110, 130):
        store.append_depth_weight_snapshot(_snapshot(update), history_length=3)

    reader = LocalChartReader(store.path)
    status = reader.status()
    assert status["depth_minimum_update"] == 100
    assert status["depth_maximum_update"] == 130
    assert step_range._depth_minimum_update(reader) == 100
    assert [
        item["optimizer_update"]
        for item in step_range._depth_snapshots_in_range(
            reader,
            minimum_update=105,
            maximum_update=130,
        )
    ] == [110, 130]
    assert step_range._depth_snapshots_in_range(
        reader,
        minimum_update=120,
        maximum_update=120,
    ) == ()

    store.append_depth_weight_snapshot(_snapshot(140), history_length=3)
    assert reader.status()["depth_minimum_update"] == 110
    assert step_range._depth_minimum_update(reader) == 110
    assert [
        item["optimizer_update"]
        for item in step_range._depth_snapshots_in_range(
            reader,
            minimum_update=0,
            maximum_update=999,
        )
    ] == [110, 130, 140]
    store.close()


def test_range_payload_cache_http_route_and_future_fill(tmp_path: Path) -> None:
    store = LocalChartStore(
        tmp_path / "charts.sqlite3",
        run_name="route_test",
        run_id="route_test",
        config={},
    )
    for update in (10, 20, 30, 40):
        store.append_depth_weight_snapshot(_snapshot(update), history_length=4)

    builder_calls: list[tuple[int, ...]] = []
    dashboard = _fake_dashboard(builder_calls)
    step_range.install(dashboard)
    state = dashboard.RunDashboardState(LocalChartReader(store.path))

    status = state.status()
    assert status["depth_minimum_update"] == 10

    payload = step_range._ranged_depth_payload(
        dashboard,
        state,
        minimum_update=20,
        maximum_update=30,
    )
    assert payload["weight_step_range"] == {
        "minimum": 20,
        "maximum": 30,
        "snapshot_count": 2,
    }
    assert payload["depth"]["chart"]["updates"] == [20, 30]
    assert builder_calls == [(20, 30)]

    cached = step_range._ranged_depth_payload(
        dashboard,
        state,
        minimum_update=20,
        maximum_update=30,
    )
    assert cached is payload
    assert builder_calls == [(20, 30)]

    catalog = SimpleNamespace(state_for_run=lambda run_name: state if run_name == "route_test" else None)
    handler_type = dashboard._handler_for(catalog)
    handler = handler_type()
    handler.path = "/api/figure-family?run=route_test&family=depth&step_min=20&step_max=30&current_only=1"
    handler.do_GET()
    assert handler.sent[0] == HTTPStatus.OK
    assert handler.sent[1]["depth"]["chart"]["updates"] == [20, 30]

    bad = handler_type()
    bad.path = "/api/figure-family?run=route_test&family=depth&step_min=30&step_max=20"
    bad.do_GET()
    assert bad.sent[0] == HTTPStatus.BAD_REQUEST

    # A future display window is initially empty. When the trainer later records a
    # snapshot inside it, the same range key must invalidate on depth revision and
    # immediately start returning data rather than reusing the cached empty payload.
    future = step_range._ranged_depth_payload(
        dashboard,
        state,
        minimum_update=60,
        maximum_update=70,
    )
    assert future["weight_step_range"]["snapshot_count"] == 0
    assert future["depth"] == {}

    store.append_depth_weight_snapshot(_snapshot(50), history_length=4)
    assert state.status()["depth_minimum_update"] == 20
    refreshed = step_range._ranged_depth_payload(
        dashboard,
        state,
        minimum_update=20,
        maximum_update=50,
    )
    assert refreshed["depth"]["chart"]["updates"] == [20, 30, 40, 50]
    assert builder_calls[-1] == (20, 30, 40, 50)

    store.append_depth_weight_snapshot(_snapshot(60), history_length=4)
    future_filled = step_range._ranged_depth_payload(
        dashboard,
        state,
        minimum_update=60,
        maximum_update=70,
    )
    assert future_filled["weight_step_range"]["snapshot_count"] == 1
    assert future_filled["depth"]["chart"]["updates"] == [60]
    assert builder_calls[-1] == (60,)
    store.close()


def test_status_reports_effective_promoted_history_capacity(tmp_path: Path) -> None:
    store = LocalChartStore(
        tmp_path / "charts.sqlite3",
        run_name="promoted-retention",
        run_id="promoted-retention",
        config={"instrumentation__depth_weight_curves__history_length": 100},
    )
    store.configure_weight_capture(
        start_step=300,
        end_step=400,
        cadence=1,
        history_length=101,
    )

    dashboard = _fake_dashboard([])
    step_range.install(dashboard)
    state = dashboard.RunDashboardState(LocalChartReader(store.path))

    assert state.status()["configuration"][
        "instrumentation__depth_weight_curves__history_length"
    ] == 101
    store.close()
# ^^^ THOG
