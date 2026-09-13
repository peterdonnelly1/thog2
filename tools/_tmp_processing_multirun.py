from __future__ import annotations

from pathlib import Path


SERVER = Path("run_thog2_local_dashboard.py")
PROCESSING_JS = Path("sheet/local_dashboard_assets/dashboard_processing.js")
TESTS = Path("tests/test_premat_processing.py")


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


server = SERVER.read_text(encoding="utf-8")
if "def processing_throughput(self)" not in server:
    anchor = "    # ^^^ THOG\n\n    def figures(self) -> Dict[str, Any]:\n"
    addition = '''    # ^^^ THOG\n\n    # vvv THOG expose the lightweight retained tok/s series independently so Workspace multirun never fetches Nsight trace payloads\n    def processing_throughput(self) -> Dict[str, Any]:\n        return {"throughput": list(self.reader.processing_throughput())}\n    # ^^^ THOG\n\n    def figures(self) -> Dict[str, Any]:\n'''
    server = replace_once(server, anchor, addition, label="processing throughput method anchor")

route_old = '                if path in {"/api/status", "/api/figures", "/api/premat", "/api/processing"}:\n'
route_new = '''                # if path in {"/api/status", "/api/figures", "/api/premat", "/api/processing"}:                                              # <<< THOG preserve the prior Processing API route set\n                if path in {"/api/status", "/api/figures", "/api/premat", "/api/processing", "/api/processing-throughput"}:                              # <<< THOG add lightweight throughput-only Workspace fetches\n'''
if '"/api/processing-throughput"' not in server:
    server = replace_once(server, route_old, route_new, label="processing route set")

dispatch_old = '''                    elif path == "/api/processing":\n                        value = state.processing()\n                    else:\n'''
dispatch_new = '''                    elif path == "/api/processing":\n                        value = state.processing()\n                    elif path == "/api/processing-throughput":                                                                                           # <<< THOG keep multirun tok/s fetches independent of the selected run's Nsight bundle\n                        value = state.processing_throughput()\n                    else:\n'''
if 'value = state.processing_throughput()' not in server:
    server = replace_once(server, dispatch_old, dispatch_new, label="processing route dispatch")
SERVER.write_text(server, encoding="utf-8")


js = PROCESSING_JS.read_text(encoding="utf-8")
start_marker = "// vvv THOG convenience copy of the run-level net-throughput scoreboard\n"
end_marker = "// ^^^ THOG\n"
start = js.find(start_marker)
if start < 0:
    if "processing_throughput_workspace_runs" not in js:
        raise RuntimeError("processing throughput block start not found")
else:
    end = js.find(end_marker, start)
    if end < 0:
        raise RuntimeError("processing throughput block end not found")
    end += len(end_marker)
    new_block = r'''// vvv THOG Training throughput alone has meaningful Workspace semantics; Nsight timeline/overlap/summary remain selected-run diagnostics
function processing_throughput_workspace_runs() {
  if (app.workspace_mode === true) {
    return (app.runs || []).filter(run => is_visible(run_identifier(run)));
  }
  const selected = typeof current_run === "function" ? current_run() : null;
  if (selected) return [selected];
  const run_id = processing_current_run();
  return run_id ? [{dashboard_run_id: run_id, artifact_name: run_id}] : [];
}

function processing_valid_throughput_rows(rows) {
  return (rows || []).filter(row => (
    Number.isFinite(Number(row.optimizer_update))
    && Number.isFinite(Number(row.tokens_per_second))
  ));
}

async function processing_throughput_rows_for_run(run, current_payload) {
  const run_id = String(run_identifier(run));
  if (run_id === processing_current_run()) {
    return processing_valid_throughput_rows(current_payload.throughput);
  }
  try {
    const response = await fetch_json(`/api/processing-throughput?run=${encodeURIComponent(run_id)}`);
    return processing_valid_throughput_rows(response.throughput);
  } catch (_error) {
    return [];
  }
}

async function processing_render_throughput(payload) {
  const runs = processing_throughput_workspace_runs();
  const resolved = await Promise.all(runs.map(async run => ({
    run,
    run_id: String(run_identifier(run)),
    rows: await processing_throughput_rows_for_run(run, payload),
  })));
  const populated = resolved.filter(entry => entry.rows.length);
  const traces = populated.map(entry => {
    const name = String(entry.run.artifact_name || entry.run.run_name || entry.run_id);
    const colour = colour_for_run(entry.run_id);
    return {
      type: "scatter",
      mode: entry.rows.length === 1 ? "markers" : "lines+markers",
      name,
      meta: {instra_workspace_run_id: entry.run_id},
      x: entry.rows.map(row => Number(row.optimizer_update)),
      y: entry.rows.map(row => Number(row.tokens_per_second)),
      line: {color: colour, width: 2.4},
      marker: {color: colour},
      hovertemplate: "update %{x}<br>%{y:,.0f} tok/s<extra>%{fullData.name}</extra>",
    };
  });
  const maximum_points = Math.max(0, ...populated.map(entry => entry.rows.length));
  const workspace = app.workspace_mode === true;
  await processing_plot("processing_throughput_plot", traces, {
    margin: {l: 72, r: 24, t: 12, b: 54},
    xaxis: {title: "optimizer update", dtick: maximum_points <= 20 ? 1 : undefined},
    yaxis: {title: "tokens / second", rangemode: "tozero", separatethousands: true},
    showlegend: traces.length > 1,
    legend: {orientation: "h", y: 1.10},
    annotations: traces.length ? [] : [{
      text: workspace ? "No retained tok/s samples for visible Workspace runs" : "No retained tok/s samples for this run",
      showarrow: false,
      xref: "paper", yref: "paper", x: 0.5, y: 0.5,
    }],
  }, plot_config);
}
// ^^^ THOG
'''
    js = js[:start] + new_block + js[end:]
PROCESSING_JS.write_text(js, encoding="utf-8")


tests = TESTS.read_text(encoding="utf-8")
marker = "def test_processing_throughput_workspace_multirun_contract()"
if marker not in tests:
    tests += r'''

# vvv THOG Training throughput overlays visible Workspace runs while capture-specific Processing charts remain selected-run diagnostics
def test_processing_throughput_workspace_multirun_contract() -> None:
    processing_js = Path("sheet/local_dashboard_assets/dashboard_processing.js").read_text(encoding="utf-8")
    server = Path("run_thog2_local_dashboard.py").read_text(encoding="utf-8")
    assert "/api/processing-throughput" in server
    assert "def processing_throughput(self)" in server
    assert "processing_throughput_workspace_runs" in processing_js
    assert "app.workspace_mode === true" in processing_js
    assert "is_visible(run_identifier(run))" in processing_js
    assert "colour_for_run(entry.run_id)" in processing_js
    assert "/api/processing-throughput?run=" in processing_js
    assert processing_js.count("/api/processing-throughput?run=") == 1
    assert "processing_render_timeline(payload)" in processing_js
    assert "processing_render_contention(payload)" in processing_js
    assert "processing_render_summary(payload)" in processing_js
# ^^^ THOG
'''
TESTS.write_text(tests, encoding="utf-8")
