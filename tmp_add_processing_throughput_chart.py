# vvv THOG temporary exact-source transformer for Processing tok/s convenience chart
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one occurrence, found {count}: {old[:180]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def thog_line(code: str, explanation: str) -> str:
    marker = "# <<< THOG " + explanation
    if len(code) >= 155:
        raise RuntimeError(f"THOG line too long for column-156 marker: {code}")
    return code + (" " * (155 - len(code))) + marker + "\n"


# One canonical throughput calculation, shared by console and Processing persistence.
replace_once(
    "sheet/stage6_source.py",
    '''def training_metric_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:\n''',
    '''# vvv THOG console and Processing use one resumed-session throughput definition\ndef progress_tokens_per_second(payload: Mapping[str, Any]) -> Optional[float]:\n    elapsed = payload.get("cumulative_training_seconds", payload.get("training_seconds"))\n    throughput_tokens = payload.get("session_consumed_tokens")\n    if throughput_tokens is None:\n        throughput_tokens = payload.get("consumed_tokens")\n    if elapsed is None or throughput_tokens is None:\n        return None\n    elapsed_value = float(elapsed)\n    if elapsed_value <= 0.0:\n        return None\n    return float(throughput_tokens) / elapsed_value\n# ^^^ THOG\n\n\ndef training_metric_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:\n''',
)

replace_once(
    "run_thog2_owt_core.py",
    'from sheet.stage6_trainer import Stage6Trainer\n',
    'from sheet.stage6_trainer import Stage6Trainer\n'
    + thog_line('from sheet.stage6_source import progress_tokens_per_second', 'share resumed-session tok/s definition with Processing instrumentation'),
)
replace_once(
    "run_thog2_owt_core.py",
    '''def add_console_tokens_per_second(payload: Dict[str, Any]) -> Dict[str, Any]:\n    values = dict(payload)\n    elapsed = values.get("cumulative_training_seconds", values.get("training_seconds"))\n    throughput_tokens = values.pop("session_consumed_tokens", None)\n    if throughput_tokens is None:\n        throughput_tokens = values.get("consumed_tokens")\n    if elapsed is None or throughput_tokens is None:\n        return values\n    elapsed_value = float(elapsed)\n    if elapsed_value <= 0.0:\n        return values\n    values["tok/s"] = float(throughput_tokens) / elapsed_value\n    return values\n''',
    '''def add_console_tokens_per_second(payload: Dict[str, Any]) -> Dict[str, Any]:\n    values = dict(payload)\n    tokens_per_second = progress_tokens_per_second(values)\n    values.pop("session_consumed_tokens", None)\n    if tokens_per_second is not None:\n        values["tok/s"] = tokens_per_second\n    return values\n''',
)

# Lightweight local history only for Processing-enabled runs.
replace_once(
    "sheet/local_chart_store.py",
    'LOCAL_CHART_SCHEMA_VERSION = 3',
    'LOCAL_CHART_SCHEMA_VERSION = 4',
)
replace_once(
    "sheet/local_chart_store.py",
    '''            CREATE TABLE IF NOT EXISTS premat_snapshots (\n                optimizer_update INTEGER PRIMARY KEY,\n                payload BLOB NOT NULL\n            );\n''',
    '''            CREATE TABLE IF NOT EXISTS premat_snapshots (\n                optimizer_update INTEGER PRIMARY KEY,\n                payload BLOB NOT NULL\n            );\n            CREATE TABLE IF NOT EXISTS processing_throughput (\n                optimizer_update INTEGER PRIMARY KEY,\n                tokens_per_second REAL NOT NULL\n            );\n''',
)
replace_once(
    "sheet/local_chart_store.py",
    '''        self._has_premat_records = bool(\n            self.connection.execute("SELECT EXISTS(SELECT 1 FROM premat_snapshots)").fetchone()[0]\n        )\n        self._has_recorded_data = self._has_heatmap_records or self._has_depth_records or self._has_premat_records\n''',
    '''        self._has_premat_records = bool(\n            self.connection.execute("SELECT EXISTS(SELECT 1 FROM premat_snapshots)").fetchone()[0]\n        )\n        # vvv THOG Processing throughput is ordinary retained chart evidence, not a new measurement path\n        self._has_processing_throughput_records = bool(\n            self.connection.execute("SELECT EXISTS(SELECT 1 FROM processing_throughput)").fetchone()[0]\n        )\n        self._has_recorded_data = (\n            self._has_heatmap_records\n            or self._has_depth_records\n            or self._has_premat_records\n            or self._has_processing_throughput_records\n        )\n        # ^^^ THOG\n''',
)
replace_once(
    "sheet/local_chart_store.py",
    '''    def update_premat_aggregate(\n''',
    '''    # vvv THOG retain the console-equivalent net-throughput scoreboard for Processing runs\n    def append_processing_throughput(\n        self,\n        optimizer_update: int,\n        tokens_per_second: float,\n    ) -> None:\n        value = _safe_runtime_metric(tokens_per_second)\n        if value is None:\n            return\n        update = int(optimizer_update)\n        self.connection.execute(\n            "INSERT OR REPLACE INTO processing_throughput(optimizer_update, tokens_per_second) VALUES (?, ?)",\n            (update, value),\n        )\n        self._latest_observed_update = max(self._latest_observed_update, update)\n        self._touch()\n        self._has_processing_throughput_records = True\n        self.connection.commit()\n    # ^^^ THOG\n\n    def update_premat_aggregate(\n''',
)
replace_once(
    "sheet/local_chart_store.py",
    '''    def latest_premat_snapshot(self) -> Optional[Dict[str, Any]]:\n''',
    '''    # vvv THOG old databases remain readable; they simply have no Processing throughput history\n    def processing_throughput(self) -> Tuple[Dict[str, Any], ...]:\n        connection = self._connection()\n        try:\n            try:\n                rows = connection.execute(\n                    "SELECT optimizer_update, tokens_per_second FROM processing_throughput ORDER BY optimizer_update"\n                ).fetchall()\n            except sqlite3.OperationalError:\n                rows = ()\n        finally:\n            connection.close()\n        return tuple(\n            {\n                "optimizer_update": int(row["optimizer_update"]),\n                "tokens_per_second": float(row["tokens_per_second"]),\n            }\n            for row in rows\n        )\n    # ^^^ THOG\n\n    def latest_premat_snapshot(self) -> Optional[Dict[str, Any]]:\n''',
)

# Persist exact session-based tok/s from the already-existing progress payload.
replace_once(
    "sheet/wandb_telemetry.py",
    '''    training_metric_payload,\n)\n''',
    '''    training_metric_payload,\n    progress_tokens_per_second,\n)\n''',
)
replace_once(
    "sheet/wandb_telemetry.py",
    '''    def log_event(self, event: str, payload: Mapping[str, Any]) -> None:\n        # vvv THOG persist bounded premat snapshots before scalar-backend early returns\n''',
    '''    def log_event(self, event: str, payload: Mapping[str, Any]) -> None:\n        # vvv THOG Processing duplicates the console tok/s scoreboard without adding another measurement\n        if (\n            event == "optimizer_progress"\n            and str(self.config.get("premat_processing_logging", "disabled")) == "enabled"\n        ):\n            tokens_per_second = progress_tokens_per_second(payload)\n            if tokens_per_second is not None:\n                ensure_local_chart_store(self).append_processing_throughput(\n                    int(payload.get("completed_updates", 0)),\n                    tokens_per_second,\n                )\n        # ^^^ THOG\n        # vvv THOG persist bounded premat snapshots before scalar-backend early returns\n''',
)
replace_once(
    "sheet/wandb_telemetry.py",
    '''            if "consumed_tokens" in telemetry_payload:\n                telemetry_payload["consumed_tokens"] = (\n                    int(telemetry_payload["consumed_tokens"]) * multiplier\n                )\n            telemetry.log_event(event, telemetry_payload)\n''',
    '''            if "consumed_tokens" in telemetry_payload:\n                telemetry_payload["consumed_tokens"] = (\n                    int(telemetry_payload["consumed_tokens"]) * multiplier\n                )\n            # vvv THOG session tokens need the same global multiplier so Processing tok/s exactly matches console semantics under DDP\n            if "session_consumed_tokens" in telemetry_payload:\n                telemetry_payload["session_consumed_tokens"] = (\n                    int(telemetry_payload["session_consumed_tokens"]) * multiplier\n                )\n            # ^^^ THOG\n            telemetry.log_event(event, telemetry_payload)\n''',
)

# Add retained throughput history to the existing Processing response and revision.
replace_once(
    "run_thog2_local_dashboard.py",
    '''    def processing(self) -> Dict[str, Any]:\n        path = self.database_path.parent / "processing" / "processing_data.json"\n        if not path.is_file():\n            return {"available": False, "revision": None, "data": None}\n        stat_result = path.stat()\n        return {\n            "available": True,\n            "revision": f"{stat_result.st_mtime_ns}:{stat_result.st_size}",\n            "data": json.loads(path.read_text()),\n        }\n''',
    '''    def processing(self) -> Dict[str, Any]:\n        path = self.database_path.parent / "processing" / "processing_data.json"\n        if not path.is_file():\n            return {"available": False, "revision": None, "data": None}\n        stat_result = path.stat()\n        # vvv THOG combine immutable Nsight evidence with the lightweight optimizer-progress throughput history\n        throughput = self.reader.processing_throughput()\n        data = json.loads(path.read_text())\n        data["throughput"] = list(throughput)\n        throughput_tail = throughput[-1] if throughput else None\n        throughput_revision = (\n            "0"\n            if throughput_tail is None\n            else f"{len(throughput)}:{throughput_tail['optimizer_update']}:{throughput_tail['tokens_per_second']:.12g}"\n        )\n        # ^^^ THOG\n        return {\n            "available": True,\n            "revision": f"{stat_result.st_mtime_ns}:{stat_result.st_size}:{throughput_revision}",\n            "data": data,\n        }\n''',
)

# Third standard INSTRA card in Processing.
insert_anchor = '''            <article class="processing-card chart-card processing-contention-card" data-chart="processing_contention">\n'''
throughput_card = '''            <!-- vvv THOG duplicate the run-level net-throughput scoreboard beside the Nsight diagnostics for convenience -->\n            <article class="processing-card chart-card processing-throughput-card" data-chart="processing_throughput">\n              <header class="chart-card-header"><div class="chart-heading-copy"><h2>Training throughput (tok/s)</h2></div><div class="chart-card-actions"><button class="maximize-button" data-maximize="processing_throughput" type="button" aria-label="Maximize training throughput" title="Maximize chart"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true" style="pointer-events:none;vertical-align:middle"><rect x="4" y="4" width="16" height="16" rx="1"/></svg></button></div></header>\n              <div class="processing-plot" id="processing_throughput_plot"></div>\n              <div class="panel-resizer panel-resizer-east" data-resize="east" title="Drag to resize chart width"></div>\n              <div class="panel-resizer panel-resizer-south" data-resize="south" title="Drag to resize chart height"></div>\n              <div class="panel-resizer panel-resizer-corner" data-resize="both" title="Drag to resize chart"></div>\n            </article>\n            <!-- ^^^ THOG -->\n'''
replace_once("sheet/local_dashboard_assets/index.html", insert_anchor, throughput_card + insert_anchor)

replace_once(
    "sheet/local_dashboard_assets/dashboard_processing.css",
    '.processing-grid .processing-contention-card { flex: 1 1 420px; }\n',
    '.processing-grid .processing-contention-card { flex: 1 1 420px; }\n.processing-grid .processing-throughput-card { flex: 1 1 420px; }                                           /* THOG net-throughput scoreboard */\n',
)

replace_once(
    "sheet/local_dashboard_assets/dashboard_processing.js",
    '''function processing_render_contention(payload) {\n''',
    '''// vvv THOG convenience copy of the run-level net-throughput scoreboard\nfunction processing_render_throughput(payload) {\n  const rows = (payload.throughput || []).filter(row => (\n    Number.isFinite(Number(row.optimizer_update))\n    && Number.isFinite(Number(row.tokens_per_second))\n  ));\n  const traces = rows.length ? [{\n    type: "scatter",\n    mode: rows.length === 1 ? "markers" : "lines+markers",\n    name: "tok/s",\n    x: rows.map(row => Number(row.optimizer_update)),\n    y: rows.map(row => Number(row.tokens_per_second)),\n    hovertemplate: "update %{x}<br>%{y:,.0f} tok/s<extra></extra>",\n  }] : [];\n  Plotly.react("processing_throughput_plot", traces, {\n    margin: {l: 72, r: 24, t: 12, b: 54},\n    xaxis: {title: "optimizer update", dtick: rows.length <= 20 ? 1 : undefined},\n    yaxis: {title: "tokens / second", rangemode: "tozero", separatethousands: true},\n    showlegend: false,\n    annotations: rows.length ? [] : [{\n      text: "No retained tok/s samples for this run",\n      showarrow: false,\n      xref: "paper", yref: "paper", x: 0.5, y: 0.5,\n    }],\n  }, plot_config);\n}\n// ^^^ THOG\n\nfunction processing_render_contention(payload) {\n''',
)
replace_once(
    "sheet/local_dashboard_assets/dashboard_processing.js",
    '''  processing_render_timeline(payload);\n  processing_render_contention(payload);\n  processing_render_summary(payload);\n''',
    '''  processing_render_timeline(payload);\n  processing_render_throughput(payload);\n  processing_render_contention(payload);\n  processing_render_summary(payload);\n''',
)
replace_once(
    "sheet/local_dashboard_assets/dashboard_processing.js",
    '    for (const chart_name of ["processing_timeline", "processing_contention"]) {\n',
    '    for (const chart_name of ["processing_timeline", "processing_throughput", "processing_contention"]) {\n',
)

# Regression coverage.
replace_once(
    "tests/test_premat_processing.py",
    'from sheet.premat_processing import (\n',
    'from sheet.local_chart_store import LocalChartReader, LocalChartStore\nfrom sheet.premat_processing import (\n',
)
replace_once(
    "tests/test_premat_processing.py",
    '''def test_processing_charts_use_standard_instra_panel_contract() -> None:\n    html = Path("sheet/local_dashboard_assets/index.html").read_text(encoding="utf-8")\n    css = Path("sheet/local_dashboard_assets/dashboard_processing.css").read_text(encoding="utf-8")\n    for chart_name in ("processing_timeline", "processing_contention"):\n''',
    '''def test_processing_charts_use_standard_instra_panel_contract() -> None:\n    html = Path("sheet/local_dashboard_assets/index.html").read_text(encoding="utf-8")\n    css = Path("sheet/local_dashboard_assets/dashboard_processing.css").read_text(encoding="utf-8")\n    for chart_name in ("processing_timeline", "processing_throughput", "processing_contention"):\n''',
)
replace_once(
    "tests/test_premat_processing.py",
    '''    assert ".processing-grid.chart-grid" in css\n\n\n''',
    '''    assert ".processing-grid.chart-grid" in css\n\n\ndef test_processing_throughput_round_trips_through_local_store(tmp_path: Path) -> None:\n    database = tmp_path / "charts.sqlite3"\n    store = LocalChartStore(database, run_name="fixture", config={})\n    store.append_processing_throughput(1, 12345.5)\n    store.append_processing_throughput(2, 13001.25)\n    store.close()\n    assert LocalChartReader(database).processing_throughput() == (\n        {"optimizer_update": 1, "tokens_per_second": 12345.5},\n        {"optimizer_update": 2, "tokens_per_second": 13001.25},\n    )\n\n\n''',
)

log_path = ROOT / "THOG2_DYNAMIC_PREMATERIALISATION_LOG.md"
log = log_path.read_text(encoding="utf-8")
entry = "\n- Processing now includes an INSTRA-standard `Training throughput (tok/s)` convenience chart. It retains the same resumed-session definition as console tok/s at optimizer-progress points; no second throughput measurement is introduced. The lightweight history is stored locally only for Processing-enabled runs and is exposed with the existing Processing response.\n"
if entry.strip() not in log:
    log_path.write_text(log.rstrip() + entry, encoding="utf-8")

print("Processing tok/s chart update applied")
# ^^^ THOG
