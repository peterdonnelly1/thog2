# vvv THOG
"""Serve only the latest weight snapshot when INSTRA requests current weights."""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, urlparse

from . import local_chart_store as _store


def _latest_depth_snapshot(reader: Any) -> dict[str, Any] | None:
    connection = _store._open_database(reader.path, readonly=True)
    try:
        row = connection.execute(
            """
            SELECT payload
            FROM depth_weight_snapshots
            ORDER BY optimizer_update DESC
            LIMIT 1
            """
        ).fetchone()
    finally:
        connection.close()
    return None if row is None else _store._decode_payload(row["payload"])


def install(dashboard: Any) -> None:
    if getattr(dashboard, "_thog2_current_weights_performance_patch_installed", False):
        return
    dashboard._thog2_current_weights_performance_patch_installed = True
    original_handler_for = dashboard._handler_for

    def latest_depth_payload(state: Any) -> dict[str, Any]:
        status = state.status()
        revision = (
            status.get("depth_snapshot_count"),
            status.get("depth_maximum_update"),
        )
        with state.lock:
            if getattr(state, "_thog2_latest_depth_revision", None) != revision:
                snapshot = _latest_depth_snapshot(state.reader)
                figures: dict[str, Any] = {}
                if snapshot:
                    available = snapshot.get("families", {})
                    for chart_name in dashboard.depth_curves._CHART_FAMILIES:
                        if chart_name not in available:
                            continue
                        figures[chart_name] = dashboard.depth_curves._build_depth_plotly_figure(
                            (snapshot,),
                            chart_name,
                        ).to_plotly_json()
                state._thog2_latest_depth_revision = revision
                state._thog2_latest_depth_payload = {"depth": figures}
            return state._thog2_latest_depth_payload

    def handler_for_current_weights(catalog: Any) -> Any:
        base_handler = original_handler_for(catalog)

        class CurrentWeightsHandler(base_handler):
            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path != "/api/figure-family":
                    return super().do_GET()
                query = parse_qs(parsed.query)
                family = query.get("family", [""])[0]
                current_only = query.get("current_only", ["0"])[0].strip().lower()
                if family != "depth" or current_only not in {"1", "true", "yes", "on"}:
                    return super().do_GET()
                run_name = query.get("run", [""])[0]
                if not run_name:
                    self._send_json(
                        {"error": "run is required"},
                        status=dashboard.HTTPStatus.BAD_REQUEST,
                    )
                    return
                try:
                    state = catalog.state_for_run(run_name)
                    self._send_json(latest_depth_payload(state))
                except (FileNotFoundError, KeyError) as error:
                    self._send_json(
                        {"error": str(error)},
                        status=dashboard.HTTPStatus.NOT_FOUND,
                    )
                except Exception as error:
                    self._send_json(
                        {"error": str(error)},
                        status=dashboard.HTTPStatus.INTERNAL_SERVER_ERROR,
                    )

        return CurrentWeightsHandler

    dashboard._handler_for = handler_for_current_weights


__all__ = ["install"]
# ^^^ THOG
