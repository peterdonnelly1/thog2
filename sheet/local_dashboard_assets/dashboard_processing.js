// vvv THOG
"use strict";

const processing_view = {
  run_id: null,
  revision: null,
  timer: null,
  available: false,
  trace_available: false,
  charts_tab_visible: true,
};

// vvv THOG Processing cards use the ordinary INSTRA plot-mount readiness contract instead of direct Plotly.react on empty divs
async function processing_plot(mount_id, traces, layout) {
  const mount = by_id(mount_id);
  if (!mount) return;
  const resolved_layout = {...layout, autosize: true};
  if (mount.dataset.plotReady === "true") {
    await Plotly.react(mount, traces, resolved_layout, plot_config);
  } else {
    mount.replaceChildren();
    await Plotly.newPlot(mount, traces, resolved_layout, plot_config);
    mount.dataset.plotReady = "true";
  }
}

function processing_sync_visibility() {
  const group = by_id("processing_chart_group");
  if (!group) return;
  group.hidden = !(processing_view.charts_tab_visible && processing_view.available);
  const timeline = by_id("processing_timeline_card");
  const contention = by_id("processing_contention_card");
  if (timeline) timeline.hidden = !processing_view.trace_available;
  if (contention) contention.hidden = !processing_view.trace_available;
}

window.processing_apply_detail_tab = charts_selected => {
  processing_view.charts_tab_visible = Boolean(charts_selected);
  processing_sync_visibility();
};
// ^^^ THOG

function processing_escape(value) {
  return String(value ?? "").replace(/[&<>"']/g, character => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[character]));
}

function processing_current_run() {
  return String(app.current_run_id || "");
}

function processing_download_url(path) {
  return `/api/local-file?run=${encodeURIComponent(processing_current_run())}&path=${encodeURIComponent(`processing/${path}`)}&download=1`;
}

function processing_set_downloads(metadata) {
  const files = metadata?.files || {};
  const links = {
    processing_download_bundle: files.bundle,
    processing_download_samples: files.samples,
    processing_download_intervals: files.intervals,
    processing_download_summary: files.summary,
    processing_download_metadata: files.metadata,
    processing_download_raw: files.raw_trace,
  };
  for (const [id, filename] of Object.entries(links)) {
    const element = by_id(id);
    if (!element) continue;
    if (!filename) {
      element.hidden = true;
      continue;
    }
    element.href = processing_download_url(filename);
    element.hidden = false;
  }
}

function processing_trace_groups(intervals) {
  const order = ["MAIN:consume", "MAIN:materialize", "MAIN:other", "PREMAT:materialize"];
  const groups = new Map();
  for (const row of intervals || []) {
    const owner = String(row.owner || "MAIN");
    const operation = String(row.operation || "other");
    const key = `${owner}:${operation}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }
  return [...groups.entries()].sort((left, right) => {
    const a = order.indexOf(left[0]);
    const b = order.indexOf(right[0]);
    return (a < 0 ? 99 : a) - (b < 0 ? 99 : b);
  });
}

async function processing_render_timeline(payload) {
  const traces = [];
  for (const [key, rows] of processing_trace_groups(payload.intervals)) {
    const x = [];
    const y = [];
    const hover = [];
    const lane = key.startsWith("PREMAT:") ? 0 : 1;
    for (const row of rows) {
      const start = Number(row.start_us) / 1000.0;
      const end = Number(row.end_us) / 1000.0;
      const detail = `${row.owner} · ${row.operation}${row.family ? ` · ${row.family}` : ""}${row.layer !== "" && row.layer !== null ? ` · L${Number(row.layer) + 1}` : ""}<br>${processing_escape(row.kernel_name)}<br>${(end - start).toFixed(4)} ms`;
      x.push(start, end, null);
      y.push(lane, lane, null);
      hover.push(detail, detail, "");
    }
    traces.push({
      type: "scattergl",
      mode: "lines",
      name: key.replace(":", " "),
      x,
      y,
      hovertext: hover,
      hoverinfo: "text",
      line: {width: key.startsWith("PREMAT:") ? 7 : 6},
      yaxis: "y",
    });
  }
  const metric_specs = [
    ["sm_active_pct", "SM Active"],
    ["sm_issue_pct", "SM Issue"],
    ["tensor_active_pct", "Tensor Active"],
    ["active_sm_unused_warp_slots_pct", "Unused warp slots"],
  ];
  for (const [key, label] of metric_specs) {
    const rows = (payload.samples || []).filter(row => row[key] !== "" && row[key] !== null && row[key] !== undefined && Number.isFinite(Number(row[key])));
    if (!rows.length) continue;
    traces.push({
      type: "scattergl",
      mode: "lines",
      name: label,
      x: rows.map(row => Number(row.time_us) / 1000.0),
      y: rows.map(row => Number(row[key])),
      hovertemplate: `${label}: %{y:.1f}%<br>%{x:.3f} ms<extra></extra>`,
      yaxis: "y2",
    });
  }
  const capture_ms = Number(payload.metadata?.capture_duration_ms || 0);
  await processing_plot("processing_timeline_plot", traces, {
    margin: {l: 70, r: 34, t: 12, b: 48},
    hovermode: "x unified",
    legend: {orientation: "h", y: 1.08},
    xaxis: {title: "capture time (ms)", range: capture_ms > 0 ? [0, capture_ms] : undefined},
    yaxis: {
      domain: [0.0, 0.24],
      tickmode: "array",
      tickvals: [0, 1],
      ticktext: ["PREMAT", "MAIN"],
      range: [-0.5, 1.5],
      fixedrange: true,
    },
    yaxis2: {
      domain: [0.34, 1.0],
      title: "processing (%)",
      range: [0, 100],
      fixedrange: true,
    },
  }, plot_config);
}

// vvv THOG convenience copy of the run-level net-throughput scoreboard
async function processing_render_throughput(payload) {
  const rows = (payload.throughput || []).filter(row => (
    Number.isFinite(Number(row.optimizer_update))
    && Number.isFinite(Number(row.tokens_per_second))
  ));
  const traces = rows.length ? [{
    type: "scatter",
    mode: rows.length === 1 ? "markers" : "lines+markers",
    name: "tok/s",
    x: rows.map(row => Number(row.optimizer_update)),
    y: rows.map(row => Number(row.tokens_per_second)),
    hovertemplate: "update %{x}<br>%{y:,.0f} tok/s<extra></extra>",
  }] : [];
  await processing_plot("processing_throughput_plot", traces, {
    margin: {l: 72, r: 24, t: 12, b: 54},
    xaxis: {title: "optimizer update", dtick: rows.length <= 20 ? 1 : undefined},
    yaxis: {title: "tokens / second", rangemode: "tozero", separatethousands: true},
    showlegend: false,
    annotations: rows.length ? [] : [{
      text: "No retained tok/s samples for this run",
      showarrow: false,
      xref: "paper", yref: "paper", x: 0.5, y: 0.5,
    }],
  }, plot_config);
}
// ^^^ THOG

async function processing_render_contention(payload) {
  const families = [...new Set((payload.summary || []).map(row => String(row.family || "?")))];
  const traces = families.map(family => {
    const rows = payload.summary.filter(row => String(row.family || "?") === family);
    return {
      type: "scatter",
      mode: "markers",
      name: family,
      x: rows.map(row => Number(row.premat_overlap_pct)),
      y: rows.map(row => Number(row.duration_ms)),
      text: rows.map(row => `L${Number(row.layer) + 1}`),
      hovertemplate: `${family} %{text}<br>PREMAT overlap of Main GPU kernels %{x:.1f}%<br>Main GPU kernel duration %{y:.4f} ms<extra></extra>`,
    };
  });
  await processing_plot("processing_contention_plot", traces, {
    margin: {l: 70, r: 24, t: 12, b: 58},
    xaxis: {title: "Main GPU kernel time overlapped by PREMAT (%)", range: [0, 100]},
    yaxis: {title: "Main GPU kernel duration (ms)"},
    legend: {orientation: "h", y: 1.08},
  }, plot_config);
}

function processing_mean(rows, key) {
  const values = rows.filter(row => row[key] !== "" && row[key] !== null && row[key] !== undefined).map(row => Number(row[key])).filter(Number.isFinite);
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

function processing_format(value, digits = 2, suffix = "") {
  return Number.isFinite(value) ? `${value.toFixed(digits)}${suffix}` : "—";
}

function processing_render_summary(payload) {
  const body = by_id("processing_summary_body");
  const families = [...new Set((payload.summary || []).map(row => String(row.family || "?")))];
  body.innerHTML = families.map(family => {
    const rows = payload.summary.filter(row => String(row.family || "?") === family);
    return `<tr><td><strong>${processing_escape(family)}</strong></td><td>${rows.length}</td><td>${processing_format(processing_mean(rows, "duration_ms"), 4, " ms")}</td><td>${processing_format(processing_mean(rows, "premat_overlap_pct"), 1, "%")}</td><td>${processing_format(processing_mean(rows, "sm_active_pct_mean"), 1, "%")}</td><td>${processing_format(processing_mean(rows, "sm_issue_pct_mean"), 1, "%")}</td><td>${processing_format(processing_mean(rows, "tensor_active_pct_mean"), 1, "%")}</td><td>${processing_format(processing_mean(rows, "active_sm_unused_warp_slots_pct_mean"), 1, "%")}</td></tr>`;
  }).join("") || '<tr><td colspan="8">No labelled Main consuming operations in this capture.</td></tr>';
}

// vvv THOG tok/s is live; Nsight cards are not instantiated until normalized trace data exists after the profiled child completes
async function processing_render(payload, trace_available) {
  processing_view.available = true;
  processing_view.trace_available = Boolean(trace_available);
  processing_sync_visibility();
  if (typeof apply_saved_panel_sizes === "function") apply_saved_panel_sizes();
  const capture = payload.metadata?.capture || {};
  const throughput = payload.throughput || [];
  const latest_throughput = throughput.length ? throughput[throughput.length - 1] : null;
  by_id("processing_step").textContent = String(capture.optimizer_update ?? latest_throughput?.optimizer_update ?? "—");
  if (trace_available) {
    const warning_count = (payload.metadata?.warnings || []).length;
    by_id("processing_status").textContent = `${Number(payload.metadata?.capture_frequency_hz || 0).toLocaleString()} Hz · ${Number(payload.metadata?.capture_duration_ms || 0).toFixed(2)} ms capture${warning_count ? ` · ${warning_count} warning${warning_count === 1 ? "" : "s"}` : ""}`;
    processing_set_downloads(payload.metadata);
    await processing_render_timeline(payload);
    await processing_render_contention(payload);
    processing_render_summary(payload);
  } else {
    by_id("processing_status").textContent = "Live throughput · Nsight charts appear after run completion";
    processing_set_downloads(null);
  }
  await processing_render_throughput(payload);
  processing_sync_visibility();
  requestAnimationFrame(() => {
    const chart_names = trace_available
      ? ["processing_timeline", "processing_throughput", "processing_contention"]
      : ["processing_throughput"];
    for (const chart_name of chart_names) {
      const card = document.querySelector(`.chart-card[data-chart="${chart_name}"]`);
      if (card && typeof resize_plot_in_card === "function") resize_plot_in_card(card);
    }
  });
}
// ^^^ THOG

async function processing_refresh() {
  const run_id = processing_current_run();
  if (!run_id) {
    processing_view.available = false;
    processing_view.trace_available = false;
    processing_sync_visibility();
    processing_view.run_id = null;
    processing_view.revision = null;
    return;
  }
  try {
    const payload = await fetch_json(`/api/processing?run=${encodeURIComponent(run_id)}`);
    if (!payload.available) {
      processing_view.available = false;
      processing_view.trace_available = false;
      processing_sync_visibility();
      processing_view.run_id = run_id;
      processing_view.revision = null;
      return;
    }
    if (processing_view.run_id === run_id && processing_view.revision === payload.revision) return;
    processing_view.run_id = run_id;
    processing_view.revision = payload.revision;
    await processing_render(payload.data, payload.trace_available === true);
  } catch (error) {
    console.warn("Processing data refresh failed", error);
  }
}

processing_view.timer = window.setInterval(processing_refresh, 1500);
window.addEventListener("load", processing_refresh);
// ^^^ THOG
