# vvv THOG
"""Serve live, interactive THOG heatmap and DEPTH charts from compact local data."""

from __future__ import annotations

import argparse
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

from plotly.offline import get_plotlyjs
from plotly.utils import PlotlyJSONEncoder

from sheet import depth_weight_curves_v2_patch as depth_curves
from sheet import plastic_depth_wandb_probe_curves_patch as probe_curves
from sheet.local_chart_store import (
    LOCAL_CHART_DATABASE_NAME,
    LocalChartReader,
    local_chart_database_path,
)


_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>THOG2 local instrumentation</title>
  <script src="/plotly.min.js"></script>
  <style>
    :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
    body { margin: 0; background: #f4f6f8; color: #202124; }
    header { position: sticky; top: 0; z-index: 4; padding: 14px 22px; background: rgba(255,255,255,.96); border-bottom: 1px solid #dfe3e7; }
    h1 { display: inline; margin: 0; font-size: 20px; font-weight: 650; }
    #status { float: right; padding-top: 3px; color: #5f6368; font-size: 13px; }
    main { padding: 18px; }
    .panel { margin-bottom: 18px; overflow: hidden; background: white; border: 1px solid #dfe3e7; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
    .plot { min-height: 420px; width: 100%; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(560px, 1fr)); gap: 18px; }
    .empty { padding: 48px 24px; color: #6b7280; text-align: center; }
    @media (max-width: 640px) { .grid { grid-template-columns: 1fr; } #status { float: none; display: block; } }
  </style>
</head>
<body>
  <header><h1>THOG2 local instrumentation</h1><span id="status">connecting...</span></header>
  <main>
    <section class="panel"><div id="heatmap" class="plot"><div class="empty">Waiting for the first layer-count probe.</div></div></section>
    <section id="depth_grid" class="grid"></section>
  </main>
  <script>
    const plot_config = {
      responsive: true,
      scrollZoom: true,
      displaylogo: false,
      toImageButtonOptions: {format: 'png', scale: 2}
    };
    let revision = null;

    function render_figure(element_id, figure) {
      const node = document.getElementById(element_id);
      if (!figure) {
        return;
      }
      figure.layout = figure.layout || {};
      figure.layout.uirevision = figure.layout.uirevision || element_id;
      Plotly.react(node, figure.data, figure.layout, plot_config);
    }

    function depth_element(chart_name) {
      const element_id = `depth_${chart_name}`;
      let node = document.getElementById(element_id);
      if (node) {
        return node;
      }
      const panel = document.createElement('section');
      panel.className = 'panel';
      node = document.createElement('div');
      node.id = element_id;
      node.className = 'plot';
      panel.appendChild(node);
      document.getElementById('depth_grid').appendChild(panel);
      return node;
    }

    async function refresh() {
      try {
        const status_response = await fetch('/api/status', {cache: 'no-store'});
        const status = await status_response.json();
        document.getElementById('status').textContent =
          `${status.run_name} · heatmap ${status.heatmap_count} probes · depth ${status.depth_snapshot_count} snapshots`;
        const next_revision = JSON.stringify(status.revision);
        if (next_revision === revision) {
          return;
        }
        revision = next_revision;
        const figure_response = await fetch('/api/figures', {cache: 'no-store'});
        const figures = await figure_response.json();
        render_figure('heatmap', figures.heatmap);
        for (const [chart_name, figure] of Object.entries(figures.depth)) {
          depth_element(chart_name);
          render_figure(`depth_${chart_name}`, figure);
        }
      } catch (error) {
        document.getElementById('status').textContent = `viewer error: ${error}`;
      }
    }

    refresh();
    setInterval(refresh, 3000);
  </script>
</body>
</html>
"""


def _resolve_database(run: Optional[str], *, root: Path) -> Path:
    if run:
        supplied = Path(run)
        if supplied.is_file():
            return supplied
        candidate = local_chart_database_path(run, root=root)
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"local chart database not found: {candidate}")
    candidates = tuple(root.glob(f"*/{LOCAL_CHART_DATABASE_NAME}"))
    if not candidates:
        raise FileNotFoundError(f"no local chart databases found below {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


class DashboardState:
    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.reader = LocalChartReader(self.database_path)
        self.lock = threading.Lock()
        self.cached_revision: Optional[Tuple[Any, ...]] = None
        self.cached_figures: Dict[str, Any] = {"heatmap": None, "depth": {}}

    def status(self) -> Dict[str, Any]:
        status = self.reader.status()
        metadata = self.reader.metadata()
        revision = (
            status["heatmap_count"],
            status["heatmap_maximum_update"],
            status["depth_snapshot_count"],
            status["depth_maximum_update"],
        )
        return {
            **status,
            "run_name": metadata.get("run_name", self.database_path.parent.name),
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
            if heatmap_history:
                maximum_layers = max(len(record["values"]) for record in heatmap_history)
                metadata = self.reader.metadata()
                config = json.loads(metadata.get("config_json", "{}"))
                abs_limit = float(
                    config.get(
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
                "depth": depth_figures,
            }
            return self.cached_figures


def _handler_for(state: DashboardState):
    plotly_javascript = get_plotlyjs().encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, *, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(int(status))
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
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
            path = urlparse(self.path).path
            try:
                if path == "/":
                    self._send(_PAGE.encode("utf-8"), content_type="text/html; charset=utf-8")
                    return
                if path == "/plotly.min.js":
                    self._send(plotly_javascript, content_type="text/javascript; charset=utf-8")
                    return
                if path == "/api/status":
                    self._send_json(state.status())
                    return
                if path == "/api/figures":
                    self._send_json(state.figures())
                    return
                self._send(b"not found\n", content_type="text/plain; charset=utf-8", status=HTTPStatus.NOT_FOUND)
            except Exception as error:
                self._send_json(
                    {"error": str(error)},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def log_message(self, _format: str, *_arguments: Any) -> None:
            return

    return Handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve live THOG2 charts from local compact data")
    parser.add_argument("--run", help="artifact name or charts.sqlite3 path; default is latest")
    parser.add_argument("--root", type=Path, default=Path("logs"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6007)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    database_path = _resolve_database(arguments.run, root=arguments.root)
    state = DashboardState(database_path)
    server = ThreadingHTTPServer(
        (str(arguments.host), int(arguments.port)),
        _handler_for(state),
    )
    url = f"http://{arguments.host}:{arguments.port}/"
    print(f"THOG2 local instrumentation: {url}", flush=True)
    print(f"data: {database_path.resolve()}", flush=True)
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
