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
  const matrix_summary = by_id("processing_matrix_summary_card");                                                                                      // <<< THOG summary is a separate trace-backed Processing card
  if (timeline) timeline.hidden = !processing_view.trace_available;
  if (contention) contention.hidden = !processing_view.trace_available;
  if (matrix_summary) matrix_summary.hidden = !processing_view.trace_available;
}

window.processing_apply_detail_tab = charts_selected => {
  processing_view.charts_tab_visible = Boolean(charts_selected);
  processing_sync_visibility();
};

// vvv THOG Processing Plotly mounts follow actual card geometry; this closes the maximize/restore race left by one-shot layout resizing
const processing_resize_observers = [];

function processing_resize_ready_card(card) {
  if (!card || card.offsetParent === null) return;
  const mount = card.querySelector(".plot-mount");
  if (!mount || mount.dataset.plotReady !== "true") return;
  requestAnimationFrame(() => {
    if (card.offsetParent !== null && mount.dataset.plotReady === "true") Plotly.Plots.resize(mount);
  });
}

function processing_install_resize_observers() {
  if (typeof ResizeObserver !== "function" || processing_resize_observers.length) return;
  for (const chart_name of ["processing_timeline", "processing_contention", "processing_throughput"]) {
    const card = document.querySelector(`.chart-card[data-chart="${chart_name}"]`);
    if (!card) continue;
    const observer = new ResizeObserver(() => processing_resize_ready_card(card));
    observer.observe(card);
    processing_resize_observers.push(observer);
  }
}
// ^^^ THOG
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

// vvv THOG Training throughput alone has meaningful Workspace semantics; Nsight timeline/overlap/summary remain selected-run diagnostics
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
      mode: entry.rows.length === 1 ? "markers" : "lines",                                                                                     // <<< THOG match ordinary INSTRA curves: no per-point markers on multi-point throughput lines
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

async function processing_render_contention(payload) {
  const family_order = ["QKV", "O", "UP", "DOWN"];
  const summary = processing_resolved_matrix_summary(payload);                                                                                             // <<< THOG drive overlap graphic from true kernel-union summary rather than semantic-span scatter points
  const families = family_order.filter(family => summary[family]);
  const premat_pct = families.map(family => Number(summary[family].premat_concurrent_with_main_pct));
  const main_pct = families.map(family => Number(summary[family].main_busy_concurrent_with_premat_pct));
  const maximum = Math.max(0, ...premat_pct.filter(Number.isFinite), ...main_pct.filter(Number.isFinite));
  const axis_maximum = Math.min(100, Math.max(5, maximum * 1.18));                                                                                         // <<< THOG keep isolated-matrix low-overlap runs readable instead of wasting a fixed 0..100 axis
  const traces = families.length ? [
    {
      type: "bar", orientation: "h", name: "PREMAT work concurrent with Main",
      y: families, x: premat_pct,
      hovertemplate: "%{y}<br>%{x:.2f}% of PREMAT GPU work coincides with any Main kernel<extra></extra>",
    },
    {
      type: "bar", orientation: "h", name: "Main busy time concurrent with PREMAT",
      y: families, x: main_pct,
      hovertemplate: "%{y}<br>%{x:.2f}% of Main GPU busy time coincides with PREMAT<extra></extra>",
    },
  ] : [];
  await processing_plot("processing_contention_plot", traces, {
    margin: {l: 70, r: 24, t: 18, b: 58},
    barmode: "group",
    xaxis: {title: "temporal overlap (%)", range: [0, axis_maximum], rangemode: "tozero"},
    yaxis: {categoryorder: "array", categoryarray: [...family_order].reverse()},
    legend: {orientation: "h", y: 1.12},
    annotations: families.length ? [] : [{text: "No PREMAT materialisation intervals in this capture", showarrow: false, xref: "paper", yref: "paper", x: 0.5, y: 0.5}],
  }, plot_config);
}

// vvv THOG fixed four-column PREMAT matrix scoreboard; non-targeted families stay blank rather than looking like zero-valued experiments
function processing_format(value, digits = 2, suffix = "") {
  return Number.isFinite(value) ? `${value.toFixed(digits)}${suffix}` : "—";
}

// vvv THOG legacy captures already contain sufficient kernel intervals; derive the new scoreboard client-side instead of forcing a rerun
function processing_merge_intervals(intervals) {
  const sorted = [...intervals]
    .map(([start, end]) => [Number(start), Number(end)])
    .filter(([start, end]) => Number.isFinite(start) && Number.isFinite(end) && end > start)
    .sort((left, right) => left[0] - right[0]);
  const merged = [];
  for (const [start, end] of sorted) {
    const tail = merged[merged.length - 1];
    if (!tail || start > tail[1]) merged.push([start, end]);
    else tail[1] = Math.max(tail[1], end);
  }
  return merged;
}

function processing_interval_duration_us(intervals) {
  return processing_merge_intervals(intervals).reduce((sum, [start, end]) => sum + end - start, 0);
}

function processing_interval_overlap_us(left_intervals, right_intervals) {
  const left = processing_merge_intervals(left_intervals);
  const right = processing_merge_intervals(right_intervals);
  let i = 0;
  let j = 0;
  let total = 0;
  while (i < left.length && j < right.length) {
    total += Math.max(0, Math.min(left[i][1], right[j][1]) - Math.max(left[i][0], right[j][0]));
    if (left[i][1] < right[j][1]) i += 1;
    else j += 1;
  }
  return total;
}

function processing_matrix_summary_from_intervals(intervals) {
  const families = ["QKV", "O", "UP", "DOWN"];
  const rows = intervals || [];
  const main_rows = rows.filter(row => row.owner === "MAIN");
  const main_consume_rows = main_rows.filter(row => row.operation === "consume");
  const as_intervals = selected => selected.map(row => [Number(row.start_us), Number(row.end_us)]);
  const main_intervals = as_intervals(main_rows);
  const main_consume_intervals = as_intervals(main_consume_rows);
  const main_busy_us = processing_interval_duration_us(main_intervals);
  const result = Object.fromEntries(families.map(family => [family, null]));
  for (const family of families) {
    const premat_rows = rows.filter(row => row.owner === "PREMAT" && row.operation === "materialize" && row.family === family);
    if (!premat_rows.length) continue;
    const premat_intervals = as_intervals(premat_rows);
    const premat_busy_us = processing_interval_duration_us(premat_intervals);
    const by_op = new Map();
    for (const row of premat_rows) {
      if (row.op_id === "" || row.op_id === null || row.op_id === undefined) continue;
      const key = Number(row.op_id);
      if (!by_op.has(key)) by_op.set(key, []);
      by_op.get(key).push([Number(row.start_us), Number(row.end_us)]);
    }
    const op_durations = [...by_op.values()].map(processing_interval_duration_us);
    const family_consumes = main_consume_rows.filter(row => row.family === family);
    const consume_ops = new Set(family_consumes.filter(row => row.op_id !== "" && row.op_id !== null && row.op_id !== undefined).map(row => Number(row.op_id)));
    const layers = [...new Set(premat_rows.filter(row => row.layer !== "" && row.layer !== null && row.layer !== undefined).map(row => Number(row.layer)))];
    const ready_leads_us = [];
    for (const layer of layers) {
      const premat_layer = premat_rows.filter(row => Number(row.layer) === layer);
      const consume_layer = family_consumes.filter(row => Number(row.layer) === layer);
      if (!consume_layer.length) continue;
      ready_leads_us.push(Math.min(...consume_layer.map(row => Number(row.start_us))) - Math.max(...premat_layer.map(row => Number(row.end_us))));
    }
    const consume_overlap_us = processing_interval_overlap_us(premat_intervals, main_consume_intervals);
    const any_main_overlap_us = processing_interval_overlap_us(premat_intervals, main_intervals);
    result[family] = {
      premat_materialisations: by_op.size,
      main_consume_operations: consume_ops.size,
      premat_gpu_ms_total: premat_busy_us / 1000,
      mean_reconstruction_ms: op_durations.length ? op_durations.reduce((sum, value) => sum + value, 0) / op_durations.length / 1000 : 0,
      main_consume_overlap_ms: consume_overlap_us / 1000,
      any_main_overlap_ms: any_main_overlap_us / 1000,
      premat_concurrent_with_main_pct: premat_busy_us > 0 ? 100 * any_main_overlap_us / premat_busy_us : 0,
      main_busy_concurrent_with_premat_pct: main_busy_us > 0 ? 100 * any_main_overlap_us / main_busy_us : 0,
      mean_ready_lead_ms: ready_leads_us.length ? ready_leads_us.reduce((sum, value) => sum + value, 0) / ready_leads_us.length / 1000 : 0,
    };
  }
  return result;
}
// ^^^ THOG

function processing_resolved_matrix_summary(payload) {
  return payload.matrix_summary || processing_matrix_summary_from_intervals(payload.intervals || []);                                                   // <<< THOG one canonical current-or-legacy summary feeds both the overlap graphic and table
}

function processing_render_summary(payload) {
  const body = by_id("processing_matrix_summary_body");
  if (!body) return;
  const families = ["QKV", "O", "UP", "DOWN"];
  const summary = processing_resolved_matrix_summary(payload);                                                                                          // <<< THOG backfill scoreboard for already-captured Processing bundles
  const value_for = (family, key, formatter) => {
    const row = summary[family];
    if (!row) return "—";
    return formatter(row[key], row);
  };
  const signed_ms = value => {
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    return `${number >= 0 ? "+" : ""}${number.toFixed(3)} ms`;
  };
  const rows = [
    ["PREMAT materialisations / Main consumes", "premat_materialisations", (value, row) => `${Number(value) || 0}/${Number(row.main_consume_operations) || 0}`],
    ["PREMAT GPU work", "premat_gpu_ms_total", value => processing_format(Number(value), 3, " ms")],
    ["Mean reconstruction", "mean_reconstruction_ms", value => processing_format(Number(value), 3, " ms")],
    ["Overlap with Main consume kernels", "main_consume_overlap_ms", value => processing_format(Number(value), 3, " ms")],
    ["Overlap with any Main kernel", "any_main_overlap_ms", value => processing_format(Number(value), 3, " ms")],
    ["PREMAT work concurrent with Main", "premat_concurrent_with_main_pct", value => processing_format(Number(value), 1, "%")],
    ["Main busy time concurrent with PREMAT", "main_busy_concurrent_with_premat_pct", value => processing_format(Number(value), 2, "%")],
    ["Mean ready-before-consumption lead", "mean_ready_lead_ms", value => signed_ms(value)],
  ];
  body.innerHTML = rows.map(([label, key, formatter]) => (`<tr><th>${processing_escape(label)}</th>${families.map(family => `<td>${processing_escape(value_for(family, key, formatter))}</td>`).join("")}</tr>`)).join("");
}
// ^^^ THOG

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
window.addEventListener("load", () => {
  processing_install_resize_observers();                                                                                                                  // <<< THOG observe Processing card geometry before first completed-trace render
  processing_refresh();
});
// ^^^ THOG
