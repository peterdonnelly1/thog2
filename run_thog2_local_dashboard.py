# vvv THOG
"""Serve live THOG chart runs and interactive figures from compact local data."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sqlite3
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from plotly.offline import get_plotlyjs
from plotly.utils import PlotlyJSONEncoder

from sheet import depth_weight_curves_v2_patch as depth_curves
from sheet import plastic_depth_wandb_probe_curves_patch as probe_curves
from sheet.local_chart_store import (
    LOCAL_CHART_DATABASE_NAME,
    LocalChartReader,
)


_ASSET_ROOT = Path(__file__).resolve().parent / "sheet" / "local_dashboard_assets"
_ASSET_NAMES = frozenset(("dashboard.css", "dashboard.js"))


def _modified_time(path: Path) -> float:
    candidates = (path, Path(f"{path}-wal"), Path(f"{path}-shm"))
    return max(
        (candidate.stat().st_mtime for candidate in candidates if candidate.exists()),
        default=0.0,
    )


def _timestamp_from_epoch(value: float) -> str:
    if value <= 0.0:
        return ""
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


class RunDashboardState:
    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.reader = LocalChartReader(self.database_path)
        self.lock = threading.Lock()
        self.cached_revision: Optional[Tuple[Any, ...]] = None
        self.cached_figures: Dict[str, Any] = {
            "heatmap": None,
            "heatmap_dimensions": {"layers": 0, "probes": 0},
            "depth": {},
        }

    def status(self) -> Dict[str, Any]:
        status = self.reader.status()
        metadata = self.reader.metadata()
        revision = (
            status["heatmap_count"],
            status["heatmap_maximum_update"],
            status["depth_snapshot_count"],
            status["depth_maximum_update"],
            metadata.get("updated_at"),
        )
        configuration = json.loads(metadata.get("config_json", "{}"))
        maximum_update = max(
            (
                value
                for value in (
                    status["heatmap_maximum_update"],
                    status["depth_maximum_update"],
                )
                if value is not None
            ),
            default=None,
        )
        modified_at = _timestamp_from_epoch(_modified_time(self.database_path))
        return {
            **status,
            "run_name": metadata.get(
                "artifact_name",
                metadata.get("run_name", self.database_path.parent.name),
            ),
            "artifact_name": metadata.get(
                "artifact_name",
                metadata.get("run_name", self.database_path.parent.name),
            ),
            "local_run_id": metadata.get("local_run_id", self.database_path.parent.name),
            "wandb_run_id": metadata.get("wandb_run_id", ""),
            "wandb_url": metadata.get("wandb_url", ""),
            "run_state": metadata.get("run_state", "unknown"),
            "created_at": metadata.get("created_at", modified_at),
            "updated_at": metadata.get("updated_at", modified_at),
            "host_label": str(configuration.get("host_label", "")),
            "model_type": str(configuration.get("model_type", "")),
            "maximum_update": maximum_update,
            "database_bytes": int(self.database_path.stat().st_size),
            "revision": revision,
        }

    def figures(self) -> Dict[str, Any]:
        status = self.status()
        revision = tuple(status["revision"])
        with self.lock:
            if revision == self.cached_revision:
                return self.cached_figures
            heatmap_history = self.reader.heatmap_history()
            heatmap_figure = None
            maximum_layers = 0
            if heatmap_history:
                maximum_layers = max(len(record["values"]) for record in heatmap_history)
                metadata = self.reader.metadata()
                configuration = json.loads(metadata.get("config_json", "{}"))
                abs_limit = float(
                    configuration.get(
                        "instrumentation__delta_loss_v_layer_heatmap_abs_limit",
                        0.05,
                    )
                )
                heatmap_figure = probe_curves._delta_loss_heatmap_figure(
                    heatmap_history,
                    maximum_layers=maximum_layers,
                    abs_limit=abs_limit,
                ).to_plotly_json()
            snapshots = self.reader.depth_weight_snapshots()
            depth_figures = {}
            if snapshots:
                available = snapshots[-1].get("families", {})
                for chart_name in depth_curves._CHART_FAMILIES:
                    if chart_name not in available:
                        continue
                    depth_figures[chart_name] = depth_curves._build_depth_plotly_figure(
                        snapshots,
                        chart_name,
                    ).to_plotly_json()
            self.cached_revision = revision
            self.cached_figures = {
                "heatmap": heatmap_figure,
                "heatmap_dimensions": {
                    "layers": maximum_layers,
                    "probes": min(len(heatmap_history), 512),
                },
                "depth": depth_figures,
            }
            return self.cached_figures


class DashboardCatalog:
    def __init__(self, *, root: Path, requested_run: Optional[str] = None) -> None:
        self.root = Path(root)
        self.requested_run = requested_run
        self.lock = threading.Lock()
        self.states: Dict[Path, RunDashboardState] = {}

    def _candidate_paths(self) -> Tuple[Path, ...]:
        if not self.root.is_dir():
            return ()
        candidates = tuple(
            sorted(
                self.root.glob(f"**/{LOCAL_CHART_DATABASE_NAME}"),
                key=_modified_time,
                reverse=True,
            )
        )
        if not self.requested_run:
            return candidates
        supplied = Path(self.requested_run)
        if supplied.is_file():
            return (supplied,)
        matches = []
        for path in candidates:
            if self.requested_run in {path.parent.name, path.parent.parent.name}:
                matches.append(path)
                continue
            try:
                metadata = LocalChartReader(path).metadata()
            except (OSError, sqlite3.DatabaseError):
                continue
            if self.requested_run in {
                metadata.get("local_run_id"),
                metadata.get("wandb_run_id"),
                metadata.get("artifact_name"),
                metadata.get("run_name"),
            }:
                matches.append(path)
        return tuple(matches)

    def _state_for_path(self, path: Path) -> RunDashboardState:
        resolved = path.resolve()
        with self.lock:
            state = self.states.get(resolved)
            if state is None:
                state = RunDashboardState(resolved)
                self.states[resolved] = state
            return state

    def runs(self) -> Dict[str, Any]:
        runs = []
        for path in self._candidate_paths():
            try:
                status = self._state_for_path(path).status()
            except (OSError, sqlite3.DatabaseError, ValueError, json.JSONDecodeError):
                continue
            runs.append(status)
        runs.sort(key=lambda run: str(run["updated_at"]), reverse=True)
        return {
            "runs": runs,
            "waiting": not bool(runs),
            "requested_run": self.requested_run,
            "root": str(self.root.resolve()),
        }

    def state_for_run(self, run_name: str) -> RunDashboardState:
        for path in self._candidate_paths():
            state = self._state_for_path(path)
            try:
                status = state.status()
            except (OSError, sqlite3.DatabaseError, ValueError, json.JSONDecodeError):
                continue
            if run_name in {
                str(status["run_name"]),
                str(status["local_run_id"]),
                str(status["wandb_run_id"]),
            }:
                return state
        raise KeyError(f"local chart run not found: {run_name}")


def _handler_for(catalog: DashboardCatalog):
    plotly_javascript = get_plotlyjs().encode("utf-8")
    index_html = (_ASSET_ROOT / "index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def _send(
            self,
            body: bytes,
            *,
            content_type: str,
            status: HTTPStatus = HTTPStatus.OK,
            cache_control: str = "no-store",
        ) -> None:
            self.send_response(int(status))
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache_control)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(
            self,
            value: Any,
            *,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            body = json.dumps(
                value,
                cls=PlotlyJSONEncoder,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            self._send(
                body,
                content_type="application/json; charset=utf-8",
                status=status,
            )

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)
            try:
                if path in {"/", "/runs"} or path.startswith("/runs/"):
                    self._send(index_html, content_type="text/html; charset=utf-8")
                    return
                if path == "/plotly.min.js":
                    self._send(
                        plotly_javascript,
                        content_type="text/javascript; charset=utf-8",
                        cache_control="public, max-age=86400",
                    )
                    return
                if path.startswith("/assets/"):
                    asset_name = Path(path).name
                    if asset_name not in _ASSET_NAMES:
                        raise FileNotFoundError(asset_name)
                    asset_path = _ASSET_ROOT / asset_name
                    content_type = mimetypes.guess_type(asset_path.name)[0]
                    self._send(
                        asset_path.read_bytes(),
                        content_type=f"{content_type or 'application/octet-stream'}; charset=utf-8",
                    )
                    return
                if path == "/api/runs":
                    self._send_json(catalog.runs())
                    return
                if path in {"/api/status", "/api/figures"}:
                    run_name = query.get("run", [""])[0]
                    if not run_name:
                        self._send_json(
                            {"error": "run query parameter is required"},
                            status=HTTPStatus.BAD_REQUEST,
                        )
                        return
                    state = catalog.state_for_run(run_name)
                    value = state.status() if path == "/api/status" else state.figures()
                    self._send_json(value)
                    return
                self._send(
                    b"not found\n",
                    content_type="text/plain; charset=utf-8",
                    status=HTTPStatus.NOT_FOUND,
                )
            except (FileNotFoundError, KeyError) as error:
                self._send_json({"error": str(error)}, status=HTTPStatus.NOT_FOUND)
            except Exception as error:
                self._send_json(
                    {"error": str(error)},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def log_message(self, _format: str, *_arguments: Any) -> None:
            return

    return Handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve live THOG2 charts from compact local run data"
    )
    parser.add_argument(
        "--run",
        help="optional artifact name or charts.sqlite3 path to wait for and show",
    )
    parser.add_argument("--root", type=Path, default=Path("logs"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6007)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    catalog = DashboardCatalog(root=arguments.root, requested_run=arguments.run)
    server = ThreadingHTTPServer(
        (str(arguments.host), int(arguments.port)),
        _handler_for(catalog),
    )
    url = f"http://{arguments.host}:{arguments.port}/"
    print(f"THOG2 local instrumentation: {url}", flush=True)
    if arguments.run:
        print(f"waiting for local chart run: {arguments.run}", flush=True)
    else:
        print(f"watching for local chart runs below: {arguments.root.resolve()}", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# ^^^ THOG
