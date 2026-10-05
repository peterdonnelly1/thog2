# vvv THOG construct only the requested weight chart, with the viewer's snapshot selection applied before Plotly serialization
"""Demand-driven weight figures for the current dashboard runtime."""
from __future__ import annotations

from collections import OrderedDict
import json
import sqlite3
import threading
from urllib.parse import parse_qs, urlparse


def install(dashboard):
    original_handler_for = dashboard._handler_for

    def handler_for(catalog):
        parent_handler = original_handler_for(catalog)
        weight_cache = OrderedDict()
        cache_lock = threading.Lock()
        cache_bytes = 0

        def forget_weights(database_path, _run_ids):
            nonlocal cache_bytes
            with cache_lock:
                for key in list(weight_cache):
                    if key[0] == str(database_path):
                        cache_bytes -= weight_cache.pop(key)[2]
        if not hasattr(catalog, "chart_cache_clearers"):
            catalog.chart_cache_clearers = []
        catalog.chart_cache_clearers.append(forget_weights)

        class DemandHandler(parent_handler):
            def do_GET(self):
                nonlocal cache_bytes
                parsed = urlparse(self.path)
                if parsed.path not in {"/api/weight-figure", "/api/processing-status"}:
                    return super().do_GET()
                query = parse_qs(parsed.query)
                try:
                    run_id = query.get("run", [""])[0]
                    if parsed.path == "/api/processing-status":
                        state = catalog.state_for_run(run_id)
                        directory = state.database_path.parent / "processing"
                        from sheet.local_dashboard_responsiveness import file_signature
                        files = {name: file_signature(directory / name) for name in
                                 ("processing_data.json", "update_timing.json", "processing_premat_compatibility.json")}
                        return self._send_json({"trace_available": files["processing_data.json"] is not None,
                                                "timing_available": files["update_timing.json"] is not None,
                                                "revision": files})
                    chart_name = query.get("chart", [""])[0]
                    if not run_id or chart_name not in dashboard.depth_curves._CHART_FAMILIES:
                        raise ValueError("run and a valid weight chart are required")
                    current_only = query.get("current_only", ["0"])[0] == "1"
                    snapshot_count = int(query.get("snapshots", ["0"])[0])
                    if snapshot_count < 0:
                        raise ValueError("snapshots must be non-negative")
                    from_zero = query.get("window", ["rolling"])[0] == "from_zero"
                    state = catalog.state_for_run(run_id)
                    status = state.status()
                    source_signature = (status.get("depth_snapshot_count"), status.get("depth_maximum_update"),
                                        chart_name, current_only, snapshot_count, from_zero)
                    with state.lock:
                        from sheet.local_dashboard_responsiveness import file_signature
                        signature = source_signature + (file_signature(state.database_path), file_signature(str(state.database_path) + "-wal"))
                        cache_key = (str(state.database_path), chart_name)
                        with cache_lock:
                            cached = weight_cache.get(cache_key)
                            if cached:
                                weight_cache.move_to_end(cache_key)
                        if cached and cached[0] == signature:
                            payload = cached[1]
                        else:
                            connection = state.reader._connection()
                            try:
                                limit = 1 if current_only else snapshot_count
                                sql = "SELECT optimizer_update, payload FROM depth_weight_snapshots ORDER BY optimizer_update "
                                sql += "ASC" if from_zero and not current_only else "DESC"
                                rows = connection.execute(sql + (" LIMIT ?" if limit else ""), (limit,) if limit else ()).fetchall()
                            finally:
                                connection.close()
                            rows = sorted(rows, key=lambda row: row["optimizer_update"])
                            from sheet.local_chart_store import _decode_payload
                            snapshots = [_decode_payload(row["payload"]) for row in rows]
                            figure = None
                            if snapshots and chart_name in snapshots[-1].get("families", {}):
                                figure = dashboard.depth_curves._build_depth_plotly_figure(snapshots, chart_name).to_plotly_json()
                            payload = {"depth": {chart_name: figure} if figure is not None else {},
                                       "weight_step_range": {"minimum": rows[0]["optimizer_update"] if rows else None,
                                                             "maximum": rows[-1]["optimizer_update"] if rows else None,
                                                             "snapshot_count": len(rows)}}
                            from plotly.utils import PlotlyJSONEncoder
                            size = 4 * len(json.dumps(payload, cls=PlotlyJSONEncoder, separators=(",", ":")))
                            # Closing the last SQLite reader can checkpoint/remove the WAL without changing this result.
                            signature = source_signature + (file_signature(state.database_path), file_signature(str(state.database_path) + "-wal"))
                            with cache_lock:
                                previous = weight_cache.pop(cache_key, None)
                                if previous:
                                    cache_bytes -= previous[2]
                                if size <= 64 * 1024 * 1024:
                                    weight_cache[cache_key] = (signature, payload, size)
                                    cache_bytes += size
                                while len(weight_cache) > 16 or cache_bytes > 64 * 1024 * 1024:
                                    cache_bytes -= weight_cache.popitem(last=False)[1][2]
                    return self._send_json(payload)
                except (ValueError, TypeError) as error:
                    return self._send_json({"error": str(error)}, status=dashboard.HTTPStatus.BAD_REQUEST)
                except (FileNotFoundError, KeyError) as error:
                    return self._send_json({"error": str(error)}, status=dashboard.HTTPStatus.NOT_FOUND)
                except (sqlite3.DatabaseError, OSError) as error:
                    return self._send_json({"error": str(error)}, status=dashboard.HTTPStatus.SERVICE_UNAVAILABLE)

        return DemandHandler

    dashboard._handler_for = handler_for
# ^^^ THOG
