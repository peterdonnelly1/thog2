# vvv THOG
"""Performance patch for the local THOG2 dashboard.

Avoid repeated run-catalog walks, cache unchanged run status reads, cache
heatmap/depth Plotly figures independently, and expose family-sized figure
payloads so a heatmap update does not serialize all six coefficient charts again.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


def install(dashboard: Any) -> None:
    if getattr(dashboard, "_thog2_dashboard_performance_patch_installed", False):
        return
    dashboard._thog2_dashboard_performance_patch_installed = True

    original_status = dashboard.RunDashboardState.status

    def status_mtime_cached(self: Any) -> dict[str, Any]:
        modified = dashboard._modified_time(self.database_path)
        cached = getattr(self, "_thog2_status_cache", None)
        if cached is not None and cached[0] == modified:
            return cached[1]
        value = original_status(self)
        self._thog2_status_cache = (modified, value)
        return value

    dashboard.RunDashboardState.status = status_mtime_cached

    original_runs = dashboard.DashboardCatalog.runs
    original_state_for_run = dashboard.DashboardCatalog.state_for_run

    def runs_with_lookup_cache(self: Any) -> dict[str, Any]:
        payload = original_runs(self)
        lookup: dict[str, Any] = {}
        for run in payload.get("runs", ()):
            run_directory = str(run.get("run_directory") or "").strip()
            if not run_directory:
                continue
            database_path = (Path(run_directory) / dashboard.LOCAL_CHART_DATABASE_NAME).resolve()
            state = self.states.get(database_path)
            if state is None:
                continue
            dashboard_id = str(run.get("dashboard_run_id") or "").strip()
            if dashboard_id:
                lookup[dashboard_id] = state
            # These aliases are useful for direct URLs/legacy callers. Preserve the
            # first match so duplicate artifact names do not randomly change target.
            for key in ("local_run_id", "wandb_run_id", "artifact_name", "run_name"):
                alias = str(run.get(key) or "").strip()
                if alias and alias not in lookup:
                    lookup[alias] = state
        self._thog2_state_lookup = lookup
        return payload

    def state_for_run_cached(self: Any, run_name: str) -> Any:
        lookup = getattr(self, "_thog2_state_lookup", None)
        if isinstance(lookup, dict):
            state = lookup.get(str(run_name))
            if state is not None and state.database_path.exists():
                return state
        state = original_state_for_run(self, run_name)
        if not isinstance(lookup, dict):
            lookup = {}
            self._thog2_state_lookup = lookup
        lookup[str(run_name)] = state
        return state

    dashboard.DashboardCatalog.runs = runs_with_lookup_cache
    dashboard.DashboardCatalog.state_for_run = state_for_run_cached

    def heatmap_revision(status: dict[str, Any]) -> tuple[Any, ...]:
        settings = status.get("heatmap_settings") or {}
        return (
            status.get("heatmap_count"),
            status.get("heatmap_maximum_update"),
            settings.get("abs_limit"),
        )

    def depth_revision(status: dict[str, Any]) -> tuple[Any, ...]:
        return (
            status.get("depth_snapshot_count"),
            status.get("depth_maximum_update"),
        )

    def heatmap_payload(
        self: Any,
        status: dict[str, Any],
        *,
        probe_count: int = 100,
        window_mode: str = "rolling",
    ) -> dict[str, Any]:
        resolved_count = max(1, min(512, int(probe_count)))
        resolved_window = str(window_mode).strip().lower()
        if resolved_window not in {"from_zero", "rolling"}:
            raise ValueError("heatmap window_mode must be from_zero or rolling")
        revision = (*heatmap_revision(status), resolved_count, resolved_window)
        with self.lock:
            cache = getattr(self, "_thog2_heatmap_payloads", None)
            if not isinstance(cache, dict):
                cache = {}
                self._thog2_heatmap_payloads = cache
            if revision not in cache:
                window_reader = getattr(self.reader, "heatmap_history_window", None)
                if callable(window_reader):
                    selected_history = window_reader(
                        probe_count=resolved_count,
                        window_mode=resolved_window,
                    )
                else:
                    history = self.reader.heatmap_history()
                    selected_history = (
                        history[:resolved_count]
                        if resolved_window == "from_zero"
                        else history[-resolved_count:]
                    )
                figure = None
                maximum_layers = 0
                if selected_history:
                    maximum_layers = max(len(record["values"]) for record in selected_history)
                    metadata = self.reader.metadata()
                    configuration = json.loads(metadata.get("config_json", "{}"))
                    abs_limit = float(
                        configuration.get(
                            "instrumentation__delta_loss_v_layer_heatmap_abs_limit",
                            0.05,
                        )
                    )
                    figure = dashboard.probe_curves._delta_loss_heatmap_figure(
                        selected_history,
                        maximum_layers=maximum_layers,
                        abs_limit=abs_limit,
                    ).to_plotly_json()
                cache.clear()
                cache[revision] = {
                    "heatmap": figure,
                    "heatmap_dimensions": {
                        "layers": maximum_layers,
                        "probes": len(selected_history),
                        "source_probes": int(status.get("heatmap_count") or 0),
                        "probe_count": resolved_count,
                        "window_mode": resolved_window,
                    },
                }
            return cache[revision]

    def depth_payload(self: Any, status: dict[str, Any]) -> dict[str, Any]:
        revision = depth_revision(status)
        with self.lock:
            if getattr(self, "_thog2_depth_revision", None) != revision:
                snapshots = self.reader.depth_weight_snapshots()
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
                self._thog2_depth_revision = revision
                self._thog2_depth_payload = {"depth": figures}
            return self._thog2_depth_payload

    def figure_family(
        self: Any,
        family: str,
        *,
        probe_count: int = 100,
        window_mode: str = "rolling",
    ) -> dict[str, Any]:
        status = self.status()
        if family == "heatmap":
            return heatmap_payload(
                self,
                status,
                probe_count=probe_count,
                window_mode=window_mode,
            )
        if family == "depth":
            return depth_payload(self, status)
        raise ValueError(f"unknown figure family: {family}")

    def figures_split_cached(self: Any) -> dict[str, Any]:
        status = self.status()
        heatmap = heatmap_payload(self, status)
        depth = depth_payload(self, status)
        combined = {
            "heatmap": heatmap.get("heatmap"),
            "heatmap_dimensions": heatmap.get("heatmap_dimensions", {"layers": 0, "probes": 0}),
            "depth": depth.get("depth", {}),
        }
        # Keep the established cache fields coherent for any code that still reads
        # them directly, while family caches remain the source of truth.
        self.cached_revision = tuple(status.get("revision", ()))
        self.cached_figures = combined
        return combined

    dashboard.RunDashboardState.figure_family = figure_family
    dashboard.RunDashboardState.figures = figures_split_cached

    original_handler_for = dashboard._handler_for

    def handler_for_with_family_endpoint(catalog: Any) -> Any:
        base_handler = original_handler_for(catalog)

        class PerformanceHandler(base_handler):
            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path != "/api/figure-family":
                    return super().do_GET()
                query = parse_qs(parsed.query)
                run_name = query.get("run", [""])[0]
                family = query.get("family", [""])[0]
                if not run_name or family not in {"heatmap", "depth"}:
                    self._send_json(
                        {"error": "run and family=heatmap|depth are required"},
                        status=dashboard.HTTPStatus.BAD_REQUEST,
                    )
                    return
                try:
                    state = catalog.state_for_run(run_name)
                    probe_count = int(query.get("probe_count", ["100"])[0])
                    window_mode = query.get("window_mode", ["rolling"])[0]
                    self._send_json(
                        state.figure_family(
                            family,
                            probe_count=probe_count,
                            window_mode=window_mode,
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

        return PerformanceHandler

    dashboard._handler_for = handler_for_with_family_endpoint
# ^^^ THOG
