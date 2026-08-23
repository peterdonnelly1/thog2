# vvv THOG
"""Serve retained weight snapshots for an explicit optimiser-step window."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import parse_qs, urlparse

from . import local_chart_store as _store


_WEIGHT_VIEW_CONFIGURATION_KEYS = (
    "instrumentation__depth_weight_curves__history_length",
    "instrumentation__depth_weight_curves__start_step",
    "instrumentation__depth_weight_curves__end_step",
)


def _depth_minimum_update(reader: Any) -> int | None:
    connection = _store._open_database(reader.path, readonly=True)
    try:
        row = connection.execute(
            "SELECT MIN(optimizer_update) AS minimum_update FROM depth_weight_snapshots"
        ).fetchone()
    finally:
        connection.close()
    value = None if row is None else row["minimum_update"]
    return None if value is None else int(value)


def _depth_snapshots_in_range(
    reader: Any,
    *,
    minimum_update: int,
    maximum_update: int,
) -> tuple[dict[str, Any], ...]:
    connection = _store._open_database(reader.path, readonly=True)
    try:
        rows = connection.execute(
            """
            SELECT payload
            FROM depth_weight_snapshots
            WHERE optimizer_update BETWEEN ? AND ?
            ORDER BY optimizer_update
            """,
            (int(minimum_update), int(maximum_update)),
        ).fetchall()
    finally:
        connection.close()
    return tuple(_store._decode_payload(row["payload"]) for row in rows)


def _ranged_depth_payload(
    dashboard: Any,
    state: Any,
    *,
    minimum_update: int,
    maximum_update: int,
) -> dict[str, Any]:
    status = state.status()
    revision = (
        tuple(status.get("revision", ())),
        status.get("depth_snapshot_count"),
        status.get("depth_maximum_update"),
        int(minimum_update),
        int(maximum_update),
    )
    with state.lock:
        if getattr(state, "_thog2_ranged_depth_revision", None) != revision:
            snapshots = _depth_snapshots_in_range(
                state.reader,
                minimum_update=minimum_update,
                maximum_update=maximum_update,
            )
            figures: dict[str, Any] = {}
            if snapshots:
                available = snapshots[-1].get("families", {})
                for chart_name in dashboard.depth_curves._CHART_FAMILIES:
                    if chart_name not in available:
                        continue
                    figures[chart_name] = dashboard.depth_curves._build_depth_plotly_figure(
                        snapshots,
                        chart_name,
                    ).to_plotly_json()
            state._thog2_ranged_depth_revision = revision
            state._thog2_ranged_depth_payload = {
                "depth": figures,
                "weight_step_range": {
                    "minimum": int(minimum_update),
                    "maximum": int(maximum_update),
                    "snapshot_count": len(snapshots),
                },
            }
        return state._thog2_ranged_depth_payload


def install(dashboard: Any) -> None:
    if getattr(dashboard, "_thog2_weight_step_range_patch_installed", False):
        return
    dashboard._thog2_weight_step_range_patch_installed = True

    original_status = dashboard.RunDashboardState.status

    def status_with_depth_minimum(self: Any) -> dict[str, Any]:
        status = original_status(self)
        revision = (
            status.get("depth_snapshot_count"),
            status.get("depth_maximum_update"),
        )
        if getattr(self, "_thog2_depth_minimum_revision", None) != revision:
            self._thog2_depth_minimum_revision = revision
            self._thog2_depth_minimum_update = (
                _depth_minimum_update(self.reader)
                if int(status.get("depth_snapshot_count") or 0) > 0
                else None
            )
        # vvv THOG expose only the Weights display configuration needed by the existing group controller; model/checkpoint configuration remains untouched
        metadata = self.reader.metadata()
        stored_configuration = json.loads(metadata.get("config_json", "{}"))
        configuration = dict(status.get("configuration") or {})
        for key in _WEIGHT_VIEW_CONFIGURATION_KEYS:
            if key in stored_configuration:
                configuration[key] = stored_configuration[key]
        # ^^^ THOG
        depth_minimum_update = getattr(self, "_thog2_depth_minimum_update", None)
        if (
            status.get("depth_minimum_update") == depth_minimum_update
            and status.get("configuration") == configuration
        ):
            return status
        return {
            **status,
            "depth_minimum_update": depth_minimum_update,
            "configuration": configuration,
        }

    dashboard.RunDashboardState.status = status_with_depth_minimum

    original_handler_for = dashboard._handler_for

    def handler_for_weight_step_range(catalog: Any) -> Any:
        base_handler = original_handler_for(catalog)

        class WeightStepRangeHandler(base_handler):
            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path != "/api/figure-family":
                    return super().do_GET()
                query = parse_qs(parsed.query)
                if query.get("family", [""])[0] != "depth" or "step_min" not in query:
                    return super().do_GET()
                run_name = query.get("run", [""])[0]
                if not run_name:
                    self._send_json(
                        {"error": "run is required"},
                        status=dashboard.HTTPStatus.BAD_REQUEST,
                    )
                    return
                try:
                    minimum_update = int(query.get("step_min", [""])[0])
                    raw_maximum = query.get("step_max", [""])[0]
                    maximum_update = minimum_update if raw_maximum == "" else int(raw_maximum)
                    if minimum_update < 0 or maximum_update < 0:
                        raise ValueError("weight step bounds must be non-negative integers")
                    if maximum_update < minimum_update:
                        raise ValueError("weight step maximum must be >= minimum")
                    state = catalog.state_for_run(run_name)
                    self._send_json(
                        _ranged_depth_payload(
                            dashboard,
                            state,
                            minimum_update=minimum_update,
                            maximum_update=maximum_update,
                        )
                    )
                except (FileNotFoundError, KeyError) as error:
                    self._send_json({"error": str(error)}, status=dashboard.HTTPStatus.NOT_FOUND)
                except ValueError as error:
                    self._send_json({"error": str(error)}, status=dashboard.HTTPStatus.BAD_REQUEST)
                except Exception as error:
                    self._send_json(
                        {"error": str(error)},
                        status=dashboard.HTTPStatus.INTERNAL_SERVER_ERROR,
                    )

        return WeightStepRangeHandler

    dashboard._handler_for = handler_for_weight_step_range


__all__ = ["install"]
# ^^^ THOG
