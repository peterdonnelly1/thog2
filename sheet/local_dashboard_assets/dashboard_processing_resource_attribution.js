// vvv THOG
"use strict";

processing_view.resource_available = false;
processing_view.compatibility_available = false;
processing_view.resource_axis_sync = false;

const processing_resource_metric_specs = [
  ["sm_active_pct", "SM Active"],
  ["sm_issue_pct", "SM Issue"],
  ["tensor_active_pct", "Tensor Active"],
  ["active_sm_unused_warp_slots_pct", "Unused warp slots"],
  ["dram_read_pct", "DRAM read"],
  ["dram_write_pct", "DRAM write"],
  ["gr_active_pct", "GR/compute active"],
  ["l2_active_pct", "L2 active"],
];

function processing_resource_ensure_download_link() {
  let link = by_id("processing_download_stream_resources");
  if (link) return link;
  const sample_link = by_id("processing_download_samples");
  if (!sample_link?.parentElement) return null;
  link = sample_link.cloneNode(false);
  link.id = "processing_download_stream_resources";
  link.textContent = "Stream resources";
  link.hidden = true;
  sample_link.insertAdjacentElement("afterend", link);
  return link;
}

function processing_resource_ensure_card() {
  let card = by_id("processing_resource_card");
  if (card) return card;
  const timeline = by_id("processing_timeline_card");
  const grid = timeline?.parentElement || by_id("processing_chart_group")?.querySelector(".processing-grid");
  if (!grid) return null;
  card = document.createElement("article");
  card.className = "processing-card chart-card processing-resource-card";
  card.id = "processing_resource_card";
  card.dataset.chart = "processing_resource";
  card.hidden = true;
  card.style.setProperty("flex", "0 0 100%", "important");
  card.style.setProperty("width", "100%", "important");
  card.style.setProperty("max-width", "100%", "important");
  card.innerHTML = `
    <header class="chart-card-header">
      <div class="chart-heading-copy">
        <h2>Stream resource pressure</h2>
        <p>Exact stream attribution only · device-wide counters remain combined wherever Main and PREMAT overlap.</p>
        <p id="processing_resource_idle_summary"></p>
      </div>
    </header>
    <div class="processing-plot-shell"><div class="processing-plot plot-mount" id="processing_resource_plot"></div></div>
    <div class="panel-resizer panel-resizer-east" data-resize="east" title="Drag to resize chart width"></div>
    <div class="panel-resizer panel-resizer-south" data-resize="south" title="Drag to resize chart height"></div>
    <div class="panel-resizer panel-resizer-corner" data-resize="both" title="Drag to resize chart"></div>`;
  if (timeline) timeline.insertAdjacentElement("afterend", card);
  else grid.appendChild(card);
  if (timeline) {
    timeline.style.setProperty("flex", "0 0 100%", "important");
    timeline.style.setProperty("width", "100%", "important");
    timeline.style.setProperty("max-width", "100%", "important");
  }
  if (typeof ResizeObserver === "function") {
    const observer = new ResizeObserver(() => processing_resize_ready_card(card));
    observer.observe(card);
    processing_resize_observers.push(observer);
  }
  return card;
}

function processing_resource_number(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function processing_resource_context(row) {
  const fragments = [];
  if (row.main_operations) fragments.push(`MAIN ${processing_escape(row.main_operations)}${row.main_layers ? ` L${processing_escape(row.main_layers)}` : ""}${row.main_families ? ` ${processing_escape(row.main_families)}` : ""}`);
  if (row.premat_operations) fragments.push(`PREMAT ${processing_escape(row.premat_operations)}${row.premat_layers ? ` L${processing_escape(row.premat_layers)}` : ""}${row.premat_families ? ` ${processing_escape(row.premat_families)}` : ""}`);
  return fragments.length ? fragments.join("<br>") : "No attributed target kernels";
}

function processing_resource_panes(rows) {
  const has_premat = rows.some(row => row.attribution_state === "PREMAT_ONLY");
  const has_combined = rows.some(row => ["MAIN_PREMAT_OVERLAP", "MIXED_SEQUENTIAL", "OTHER_OR_UNKNOWN"].includes(String(row.attribution_state)));
  if (!has_premat && !has_combined) {
    return [{key: "main", label: "MAIN", states: new Set(["MAIN_ONLY"]), domain: [0.0, 1.0], yaxis: "y"}];
  }
  if (has_premat && !has_combined) {
    return [
      {key: "main", label: "MAIN", states: new Set(["MAIN_ONLY"]), domain: [0.54, 1.0], yaxis: "y"},
      {key: "premat", label: "PREMAT", states: new Set(["PREMAT_ONLY"]), domain: [0.0, 0.46], yaxis: "y2"},
    ];
  }
  if (!has_premat && has_combined) {
    return [
      {key: "main", label: "MAIN", states: new Set(["MAIN_ONLY"]), domain: [0.54, 1.0], yaxis: "y"},
      {key: "combined", label: "COMBINED / UNATTRIBUTABLE", states: new Set(["MAIN_PREMAT_OVERLAP", "MIXED_SEQUENTIAL", "OTHER_OR_UNKNOWN"]), domain: [0.0, 0.46], yaxis: "y2"},
    ];
  }
  return [
    {key: "main", label: "MAIN", states: new Set(["MAIN_ONLY"]), domain: [0.70, 1.0], yaxis: "y"},
    {key: "premat", label: "PREMAT", states: new Set(["PREMAT_ONLY"]), domain: [0.35, 0.64], yaxis: "y2"},
    {key: "combined", label: "COMBINED / UNATTRIBUTABLE", states: new Set(["MAIN_PREMAT_OVERLAP", "MIXED_SEQUENTIAL", "OTHER_OR_UNKNOWN"]), domain: [0.0, 0.29], yaxis: "y3"},
  ];
}

function processing_resource_trace(rows, pane, metric_key, metric_label, showlegend) {
  const x = [];
  const y = [];
  const hover = [];
  for (const row of rows) {
    const value = processing_resource_number(row[metric_key]);
    const accepted = pane.states.has(String(row.attribution_state));
    if (!accepted || value === null) {
      x.push(null);
      y.push(null);
      hover.push("");
      continue;
    }
    const time_ms = Number(row.time_us) / 1000.0;
    x.push(time_ms);
    y.push(value);
    hover.push(
      `${time_ms.toFixed(3)} ms<br>${processing_escape(row.attribution_state)}<br>${processing_resource_context(row)}<br>`
      + `${processing_escape(metric_label)}: ${value.toFixed(1)}%<br>`
      + `MAIN coverage ${Number(row.main_coverage_pct || 0).toFixed(1)}% · PREMAT coverage ${Number(row.premat_coverage_pct || 0).toFixed(1)}%`
    );
  }
  if (!y.some(value => value !== null)) return null;
  return {
    type: "scattergl",
    mode: "lines",
    name: metric_label,
    legendgroup: metric_key,
    showlegend,
    x,
    y,
    hovertext: hover,
    hoverinfo: "text",
    connectgaps: false,
    yaxis: pane.yaxis,
  };
}

function processing_resource_idle_shapes(payload) {
  return (payload.main_idle_intervals || [])
    .filter(row => Number(row.duration_us) > 0)
    .map(row => ({
      type: "rect",
      xref: "x",
      yref: "paper",
      x0: Number(row.start_us) / 1000.0,
      x1: Number(row.end_us) / 1000.0,
      y0: 0,
      y1: 1,
      line: {width: 0},
      fillcolor: "rgba(128,128,128,0.07)",
      layer: "below",
    }));
}

function processing_resource_update_idle_summary(payload) {
  const element = by_id("processing_resource_idle_summary");
  if (!element) return;
  const rows = (payload.main_idle_intervals || []).filter(row => Number(row.duration_us) > 0);
  if (!rows.length) {
    element.textContent = "Main idle: no positive-duration kernel gaps in this capture.";
    return;
  }
  const longest = Math.max(...rows.map(row => Number(row.duration_us))) / 1000.0;
  const total = rows.reduce((sum, row) => sum + Number(row.duration_us), 0) / 1000.0;
  element.textContent = `Main idle: ${rows.length} exact kernel-gap interval${rows.length === 1 ? "" : "s"} · longest ${longest.toFixed(3)} ms · total ${total.toFixed(3)} ms`;
}

async function processing_resource_render(payload) {
  const card = processing_resource_ensure_card();
  if (!card) return;
  const rows = Array.isArray(payload.stream_resources) ? payload.stream_resources : [];
  processing_view.resource_available = Number(payload.metadata?.schema_version || 1) >= 2 && rows.length > 0;
  card.hidden = !(processing_view.charts_tab_visible && processing_view.resource_available);
  if (!processing_view.resource_available) return;
  processing_resource_update_idle_summary(payload);
  const panes = processing_resource_panes(rows);
  const traces = [];
  for (const pane of panes) {
    for (const [metric_key, metric_label] of processing_resource_metric_specs) {
      const trace = processing_resource_trace(rows, pane, metric_key, metric_label, pane.key === "main");
      if (trace) traces.push(trace);
    }
  }
  const capture_ms = Number(payload.metadata?.capture_duration_ms || 0);
  const layout = {
    margin: {l: 92, r: 34, t: 18, b: 52},
    hovermode: "closest",
    legend: {orientation: "h", y: 1.10},
    xaxis: {title: "capture time (ms)", range: capture_ms > 0 ? [0, capture_ms] : undefined},
    shapes: processing_resource_idle_shapes(payload),
  };
  panes.forEach((pane, index) => {
    const axis_name = index === 0 ? "yaxis" : `yaxis${index + 1}`;
    layout[axis_name] = {
      domain: pane.domain,
      title: pane.label,
      range: [0, 100],
      ticksuffix: "%",
      fixedrange: true,
    };
  });
  await processing_plot("processing_resource_plot", traces, layout);
  processing_resource_link_time_axes();
}

function processing_resource_extract_xrange(event) {
  if (event["xaxis.autorange"] === true) return {autorange: true};
  const left = processing_resource_number(event["xaxis.range[0]"]);
  const right = processing_resource_number(event["xaxis.range[1]"]);
  return left !== null && right !== null ? {range: [left, right]} : null;
}

function processing_resource_link_one(source, target) {
  if (!source || !target || source.dataset.processingResourceLinked === "true") return;
  source.dataset.processingResourceLinked = "true";
  source.on("plotly_relayout", event => {
    if (processing_view.resource_axis_sync) return;
    const update = processing_resource_extract_xrange(event || {});
    if (!update) return;
    processing_view.resource_axis_sync = true;
    const relayout = update.autorange ? {"xaxis.autorange": true} : {"xaxis.range": update.range};
    Promise.resolve(Plotly.relayout(target, relayout)).finally(() => { processing_view.resource_axis_sync = false; });
  });
}

function processing_resource_link_time_axes() {
  const timeline = by_id("processing_timeline_plot");
  const resource = by_id("processing_resource_plot");
  processing_resource_link_one(timeline, resource);
  processing_resource_link_one(resource, timeline);
}

// vvv THOG schema-v2 execution timeline is execution-only; legacy schema-v1 captures keep the established composite timeline.
const processing_render_timeline_before_resource_attribution = processing_render_timeline;
processing_render_timeline = async function(payload) {
  if (Number(payload.metadata?.schema_version || 1) < 2) {
    return processing_render_timeline_before_resource_attribution(payload);
  }
  const traces = [];
  const lane_order = ["UNKNOWN", "OTHER", "PREMAT", "MAIN"];
  const active_owners = new Set((payload.intervals || []).map(row => String(row.owner || "UNKNOWN")));
  const owners = lane_order.filter(owner => active_owners.has(owner));
  const lane_for = owner => Math.max(0, owners.indexOf(owner));
  for (const [key, rows] of processing_trace_groups(payload.intervals)) {
    const owner = String(rows[0]?.owner || "UNKNOWN");
    const x = [];
    const y = [];
    const hover = [];
    for (const row of rows) {
      const start = Number(row.start_us) / 1000.0;
      const end = Number(row.end_us) / 1000.0;
      const source = row.owner_source ? ` · ${row.owner_source}` : "";
      const detail = `${processing_escape(row.owner)} · ${processing_escape(row.operation)}${row.family ? ` · ${processing_escape(row.family)}` : ""}${row.layer !== "" && row.layer !== null ? ` · L${Number(row.layer) + 1}` : ""}${source}<br>${processing_escape(row.kernel_name)}<br>${(end - start).toFixed(4)} ms`;
      x.push(start, end, null);
      y.push(lane_for(owner), lane_for(owner), null);
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
      line: {width: owner === "PREMAT" ? 7 : 6},
    });
  }
  const capture_ms = Number(payload.metadata?.capture_duration_ms || 0);
  await processing_plot("processing_timeline_plot", traces, {
    margin: {l: 92, r: 34, t: 12, b: 48},
    hovermode: "closest",
    legend: {orientation: "h", y: 1.08},
    xaxis: {title: "capture time (ms)", range: capture_ms > 0 ? [0, capture_ms] : undefined},
    yaxis: {
      tickmode: "array",
      tickvals: owners.map((_owner, index) => index),
      ticktext: owners,
      range: [-0.5, Math.max(0.5, owners.length - 0.5)],
      fixedrange: true,
    },
  });
};
// ^^^ THOG

const processing_set_downloads_before_resource_attribution = processing_set_downloads;
processing_set_downloads = function(metadata) {
  processing_set_downloads_before_resource_attribution(metadata);
  const link = processing_resource_ensure_download_link();
  if (!link) return;
  const filename = metadata?.files?.stream_resources;
  if (!filename) {
    link.hidden = true;
    return;
  }
  link.href = processing_download_url(filename);
  link.hidden = false;
};

const processing_sync_visibility_before_resource_attribution = processing_sync_visibility;
processing_sync_visibility = function() {
  processing_sync_visibility_before_resource_attribution();
  const card = by_id("processing_resource_card");
  if (card) card.hidden = !(processing_view.charts_tab_visible && (processing_view.resource_available || processing_view.compatibility_available));
};

const processing_render_before_resource_attribution = processing_render;
processing_render = async function(payload, trace_available) {
  await processing_render_before_resource_attribution(payload, trace_available);
  if (trace_available) await processing_resource_render(payload);
  else processing_view.resource_available = false;
  const group_count = by_id("processing_group_count");
  if (group_count && processing_view.resource_available) {
    const count = Number(group_count.textContent);
    if (Number.isFinite(count)) group_count.textContent = String(count + 1);
  }
  processing_sync_visibility();
  requestAnimationFrame(() => processing_resize_ready_card(by_id("processing_resource_card")));
};

window.addEventListener("load", () => {
  processing_resource_ensure_download_link();
  processing_resource_ensure_card();
});
// ^^^ THOG

// vvv THOG GPU resource compatibility tool v1

Object.assign(chart_titles, {
  processing_timeline: "GPU processing timeline",
  processing_resource: "Stream resource pressure",
  processing_contention: "PREMAT/Main temporal overlap",
  processing_throughput: "Training throughput (tok/s)",
  processing_matrix_summary: "Matrix summary",
});

const processing_gpu_resource_specs = Object.freeze([
  {key: "tensor_active_pct", label: "Tensor Active", visible: true, colour: "#6f42c1"},
  {key: "sm_issue_pct", label: "SM Issue", visible: true, colour: "#0067c5"},
  {key: "compute_warps_in_flight_pct", label: "Compute warp residency", visible: true, colour: "#00a6d6"},
  {key: "dram_read_pct", label: "DRAM read", visible: true, colour: "#f39c12"},
  {key: "dram_write_pct", label: "DRAM write", visible: true, colour: "#9b59b6"},
  {key: "active_sm_unused_warp_slots_pct", label: "Active-SM allocated warp slots", visible: true, colour: "#ff1493", invert: true},
  {key: "idle_sm_unused_warp_slots_pct", label: "Idle-SM warp-slot headroom", visible: false, colour: "#c45a00"},
  {key: "l2_active_pct", label: "L2 active", visible: false, colour: "#7f8c8d"},
  {key: "sm_active_pct", label: "SM Active", visible: false, colour: "#4d79a7"},
  {key: "gr_active_pct", label: "GR engine active", visible: false, colour: "#8c6d5a"},
]);

const processing_gpu_bar_palette = Object.freeze([
  "#0067c5", "#ff1493", "#7b61ff", "#f39c12", "#00a6d6", "#6f42c1", "#8c6d5a", "#607d8b",
]);

processing_view.gpu_axis_sync = false;
processing_view.throughput_axis_mode = localStorage.getItem("thog2_processing_throughput_x_mode") || "step";
processing_view.throughput_z_offset = 0;
processing_view.throughput_last_payload = null;

function processing_gpu_display_text(value) {
  return String(value ?? "").replaceAll("NOMAT", "MAT");
}

function processing_gpu_inject_style() {
  if (by_id("processing_gpu_resource_tool_style")) return;
  const style = document.createElement("style");
  style.id = "processing_gpu_resource_tool_style";
  style.textContent = `
    .processing-card.maximized { display:flex !important; flex-direction:column !important; }
    .processing-card.maximized > .chart-card-header {
      position:relative !important; flex:0 0 auto !important; min-height:58px !important;
      height:auto !important; padding-right:8px !important; overflow:visible !important;
    }
    .processing-card.maximized .chart-card-actions {
      position:static !important; flex:0 0 auto !important; margin-left:auto !important;
      display:flex !important; align-items:center !important; gap:6px !important;
    }
    .processing-card.maximized .maximize-button,
    .processing-card.maximized .chart-settings-button {
      position:static !important; inset:auto !important; right:auto !important; top:auto !important;
      box-shadow:none !important;
    }
    .processing-card.maximized > .processing-plot-shell,
    .processing-card.maximized > .plot-shell.processing-plot-shell {
      position:relative !important; inset:auto !important; flex:1 1 auto !important;
      width:100% !important; height:auto !important; min-height:0 !important;
    }
    .processing-card.maximized .processing-plot { min-height:0 !important; height:100% !important; }
    .processing-resource-card .chart-card-header { flex:0 0 auto; min-height:72px; height:auto; align-items:flex-start; }
    .processing-resource-body { flex:1 1 auto; min-height:0; display:flex; flex-direction:column; }
    .processing-resource-main-shell { flex:1 1 auto; min-height:150px; position:relative !important; inset:auto !important; }
    .processing-resource-clock-shell { flex:0 0 108px; min-height:86px; position:relative !important; inset:auto !important; border-top:1px solid rgba(127,127,127,.12); }
    .processing-resource-compat-shell { flex:0 0 auto; min-height:92px; position:relative !important; inset:auto !important; border-top:1px solid rgba(127,127,127,.12); }
    .processing-resource-card.maximized .processing-resource-clock-shell { flex-basis:128px; }
    .processing-resource-card.maximized .processing-resource-compat-shell { min-height:112px; }
    .processing-resource-card.processing-resource-compat-only .processing-resource-main-shell,
    .processing-resource-card.processing-resource-compat-only .processing-resource-clock-shell { display:none !important; }
    .processing-resource-card.processing-resource-compat-only .processing-resource-compat-shell { flex:1 1 auto; min-height:150px; }
    .processing-throughput-axis-select {
      height:28px; max-width:150px; border:1px solid rgba(127,127,127,.32); border-radius:5px;
      background:#fff; color:inherit; padding:0 6px; font-size:11px;
    }
    .processing-throughput-z-button {
      width:29px; height:28px; border:1px solid rgba(127,127,127,.32); border-radius:5px;
      background:#f7f7f8; color:inherit; cursor:pointer; font-weight:700;
    }
    .processing-throughput-z-button:hover { background:#eceafc; border-color:#b8afea; color:#4732b7; }
    .processing-resource-detail {
      flex:0 0 auto; display:grid; grid-template-columns:repeat(auto-fit,minmax(175px,1fr));
      gap:4px 14px; padding:7px 12px; border-bottom:1px solid rgba(127,127,127,.16);
      background:rgba(75,92,120,.045); font-size:11px; line-height:1.35;
    }
    .processing-resource-detail[hidden] { display:none !important; }
    .processing-resource-detail strong { color:#222; }
  `;
  document.head.appendChild(style);
}

const processing_plot_before_gpu_resource_tool = processing_plot;
processing_plot = async function(mount_id, traces, layout) {
  let bar_index = 0;
  for (const trace of traces || []) {
    if (trace?.type !== "bar") continue;
    trace.marker = {...(trace.marker || {})};
    if (trace.marker.color === undefined || trace.marker.color === null || trace.marker.color === "") {
      trace.marker.color = processing_gpu_bar_palette[bar_index % processing_gpu_bar_palette.length];
      bar_index += 1;
    }
  }
  return processing_plot_before_gpu_resource_tool(mount_id, traces, layout);
};

const processing_resize_ready_card_before_gpu_resource_tool = processing_resize_ready_card;
processing_resize_ready_card = function(card) {
  if (!card || card.offsetParent === null) return;
  const mounts = [...card.querySelectorAll(".plot-mount")].filter(mount => mount.dataset.plotReady === "true");
  if (!mounts.length) {
    processing_resize_ready_card_before_gpu_resource_tool(card);
    return;
  }
  requestAnimationFrame(() => {
    for (const mount of mounts) {
      if (card.offsetParent !== null && mount.dataset.plotReady === "true") Plotly.Plots.resize(mount);
    }
  });
};

function processing_gpu_resize_visible_cards() {
  document.querySelectorAll("#processing_chart_group .processing-card").forEach(processing_resize_ready_card);
}

function processing_gpu_schedule_resize() {
  for (const delay of [0, 60, 180, 350]) setTimeout(processing_gpu_resize_visible_cards, delay);
}

function processing_gpu_maximize_button(chart_name, label) {
  const button = document.createElement("button");
  button.className = "maximize-button";
  button.dataset.maximize = chart_name;
  button.type = "button";
  button.setAttribute("aria-label", `Maximize ${label}`);
  button.title = "Maximize chart";
  button.innerHTML = typeof chart_size_icon === "function" ? chart_size_icon(false) : "□";
  return button;
}

const processing_resource_ensure_card_before_gpu_resource_tool = processing_resource_ensure_card;
processing_resource_ensure_card = function() {
  const card = processing_resource_ensure_card_before_gpu_resource_tool();
  if (!card) return card;
  const header = card.querySelector(".chart-card-header");
  if (header) {
    let actions = header.querySelector(".chart-card-actions");
    if (!actions) {
      actions = document.createElement("div");
      actions.className = "chart-card-actions";
      header.appendChild(actions);
    }
    if (!actions.querySelector(".maximize-button")) {
      actions.appendChild(processing_gpu_maximize_button("processing_resource", "stream resource pressure"));
    }
  }
  let body = card.querySelector(".processing-resource-body");
  if (!body) {
    const main_shell = card.querySelector(":scope > .processing-plot-shell");
    if (main_shell) {
      body = document.createElement("div");
      body.className = "processing-resource-body";
      main_shell.classList.add("processing-resource-main-shell");
      main_shell.insertAdjacentElement("beforebegin", body);
      body.appendChild(main_shell);
    }
  }
  if (body && !by_id("processing_resource_clock_plot")) {
    const clock_shell = document.createElement("div");
    clock_shell.className = "processing-plot-shell processing-resource-clock-shell";
    clock_shell.innerHTML = '<div class="processing-plot plot-mount" id="processing_resource_clock_plot"></div>';
    body.appendChild(clock_shell);
  }
  if (body && !by_id("processing_resource_detail")) {
    const detail = document.createElement("div");
    detail.id = "processing_resource_detail";
    detail.className = "processing-resource-detail";
    detail.hidden = true;
    body.insertBefore(detail, body.firstChild);
  }
  if (body && !by_id("processing_resource_compatibility_plot")) {
    const compatibility_shell = document.createElement("div");
    compatibility_shell.className = "processing-plot-shell processing-resource-compat-shell";
    compatibility_shell.id = "processing_resource_compatibility_shell";
    compatibility_shell.hidden = true;
    compatibility_shell.innerHTML = '<div class="processing-plot plot-mount" id="processing_resource_compatibility_plot"></div>';
    body.appendChild(compatibility_shell);
  }
  if (typeof ResizeObserver === "function" && !card._processing_gpu_resize_observer) {
    const observer = new ResizeObserver(() => processing_resize_ready_card(card));
    observer.observe(card);
    card._processing_gpu_resize_observer = observer;
    if (Array.isArray(processing_resize_observers)) processing_resize_observers.push(observer);
  }
  return card;
};

function processing_gpu_resource_trace(rows, pane, spec, showlegend, metric_catalog) {
  const x = [];
  const y = [];
  const hover = [];
  for (const row of rows) {
    const raw_value = processing_resource_number(row[spec.key]);
    const value = raw_value === null ? null : (spec.invert ? 100 - raw_value : raw_value);
    if (!pane.states.has(String(row.attribution_state)) || value === null) {
      x.push(null); y.push(null); hover.push("");
      continue;
    }
    const time_ms = Number(row.time_us) / 1000.0;
    x.push(time_ms);
    y.push(value);
    hover.push(
      `${time_ms.toFixed(3)} ms<br>${processing_escape(row.attribution_state)}<br>${processing_resource_context(row)}<br>`
      + `${processing_escape(spec.label)}: ${value.toFixed(1)}%<br>`
      + `Nsight metric: ${processing_escape(metric_catalog?.[spec.key]?.nsight_name || "unavailable")}<br>`
      + `MAIN coverage ${Number(row.main_coverage_pct || 0).toFixed(1)}% · PREMAT coverage ${Number(row.premat_coverage_pct || 0).toFixed(1)}%`
    );
  }
  if (!y.some(value => value !== null)) return null;
  return {
    type: "scattergl", mode: "lines", name: spec.label,
    legendgroup: spec.key, uid: `${pane.key}:${spec.key}`, showlegend, x, y, hovertext: hover, hoverinfo: "text",
    connectgaps: false, yaxis: pane.yaxis,
    line: {width: 1.7, color: spec.colour},
    visible: spec.visible ? true : "legendonly",
  };
}

function processing_gpu_compatibility_rows(payload) {
  const source = payload?.premat_compatibility;
  if (Array.isArray(source)) return source;
  if (source && Array.isArray(source.rows)) return source.rows;
  return [];
}

function processing_gpu_compatibility_rank(value) {
  return {RED: 0, YELLOW: 1, ORANGE: 2, GREEN: 3}[String(value || "").toUpperCase()] ?? 99;
}

function processing_gpu_matching_compatibility(rows, interval) {
  const candidates = rows.filter(row => {
    if (String(row.main_operation || "") && String(row.main_operation) !== String(interval.operation || "")) return false;
    if (String(row.main_family || "") && String(row.main_family) !== String(interval.family || "")) return false;
    return true;
  });
  const has_layer = row => row.main_layer !== "" && row.main_layer !== null && row.main_layer !== undefined;
  const exact_layer = candidates.filter(row => has_layer(row) && Number(row.main_layer) === Number(interval.layer));
  const generic = candidates.filter(row => !has_layer(row));
  const selected = exact_layer.length ? exact_layer : (generic.length ? generic : candidates);
  return selected.sort((left, right) => processing_gpu_compatibility_rank(left.compatibility_class) - processing_gpu_compatibility_rank(right.compatibility_class))[0] || null;
}

async function processing_gpu_render_compatibility(payload) {
  const shell = by_id("processing_resource_compatibility_shell");
  const mount = by_id("processing_resource_compatibility_plot");
  if (!shell || !mount) return;
  const compatibility = processing_gpu_compatibility_rows(payload);
  if (!compatibility.length) {
    shell.hidden = true;
    return;
  }
  const families = [...new Set(compatibility.map(row => String(row.premat_family || "PREMAT")).filter(Boolean))];
  const class_colours = {GREEN: "#2e9d57", ORANGE: "#f28c28", YELLOW: "#f2cc0c", RED: "#d63c3c"};
  const intervals = Array.isArray(payload.intervals) ? payload.intervals : [];
  if (!intervals.length) {
    const grouped = new Map();
    for (const row of compatibility) {
      const klass = String(row.compatibility_class || "").toUpperCase();
      if (!class_colours[klass]) continue;
      if (!grouped.has(klass)) grouped.set(klass, {x: [], y: [], hover: []});
      const trace = grouped.get(klass);
      const main_layer = row.main_layer === "" || row.main_layer === null || row.main_layer === undefined ? "—" : Number(row.main_layer) + 1;
      const premat_layer = row.premat_layer === "" || row.premat_layer === null || row.premat_layer === undefined ? "—" : Number(row.premat_layer) + 1;
      trace.x.push(1);
      trace.y.push(`MAIN ${row.main_family || "?"} L${main_layer} → PREMAT ${row.premat_family || "?"} L${premat_layer}`);
      trace.hover.push(
        `${klass} · aggregate SM-budget pair ${row.pair_can_co_reside ? "can" : "cannot"} co-reside<br>`
        + `PREMAT blocks with full MAIN residency: ${Number(row.premat_blocks_with_full_main_residency || 0)}<br>`
        + `MAIN block: ${Number(row.main_threads_per_block || 0)} threads · ${Number(row.main_warps_per_block || 0)} warps · ${Number(row.main_registers_per_block || 0).toLocaleString()} regs · ${Number(row.main_shared_mem_bytes || 0).toLocaleString()} B shared<br>`
        + `PREMAT block: ${Number(row.premat_threads_per_block || 0)} threads · ${Number(row.premat_warps_per_block || 0)} warps · ${Number(row.premat_registers_per_block || 0).toLocaleString()} regs · ${Number(row.premat_shared_mem_bytes || 0).toLocaleString()} B shared<br>`
        + `limiter: ${processing_escape(row.limiting_resource || "—")}`
      );
    }
    const traces = [...grouped.entries()].map(([klass, item]) => ({
      type: "bar", orientation: "h", name: klass, x: item.x, y: item.y,
      hovertext: item.hover, hoverinfo: "text", marker: {color: class_colours[klass]},
      width: 0.62, showlegend: false,
    }));
    shell.hidden = false;
    shell.style.flexBasis = `${Math.max(150, 84 + compatibility.length * 34)}px`;
    await processing_plot("processing_resource_compatibility_plot", traces, {
      margin: {l: 210, r: 34, t: 34, b: 24}, barmode: "group", hovermode: "closest",
      title: {text: "Theoretical MAIN/PREMAT aggregate SM-budget co-residency · Nsight Compute", x: 0.01, xanchor: "left", font: {size: 11}},
      xaxis: {visible: false, range: [0, 1]},
      yaxis: {automargin: true},
    });
    return;
  }
  const traces_by_class = new Map();
  for (const interval of intervals) {
    if (String(interval.owner) !== "MAIN") continue;
    for (const premat_family of families) {
      const match = processing_gpu_matching_compatibility(
        compatibility.filter(row => String(row.premat_family || "PREMAT") === premat_family),
        interval,
      );
      if (!match) continue;
      const klass = String(match.compatibility_class || "").toUpperCase();
      if (!class_colours[klass]) continue;
      const key = `${klass}:${premat_family}`;
      if (!traces_by_class.has(key)) traces_by_class.set(key, {klass, premat_family, x: [], base: [], y: [], hover: []});
      const trace = traces_by_class.get(key);
      const start = Number(interval.start_us) / 1000.0;
      const duration = Math.max(0, Number(interval.end_us) - Number(interval.start_us)) / 1000.0;
      trace.x.push(duration);
      trace.base.push(start);
      trace.y.push(`PREMAT ${premat_family}`);
      trace.hover.push(
        `MAIN ${processing_escape(interval.operation)} ${processing_escape(interval.family || "")} → PREMAT ${processing_escape(premat_family)}<br>`
        + `${klass} · aggregate SM-budget pair ${match.pair_can_co_reside ? "can" : "cannot"} co-reside<br>`
        + `PREMAT blocks with full MAIN residency: ${Number(match.premat_blocks_with_full_main_residency || 0)}<br>`
        + `MAIN block: ${Number(match.main_threads_per_block || 0)} threads · ${Number(match.main_warps_per_block || 0)} warps · ${Number(match.main_registers_per_block || 0).toLocaleString()} regs · ${Number(match.main_shared_mem_bytes || 0).toLocaleString()} B shared<br>`
        + `PREMAT block: ${Number(match.premat_threads_per_block || 0)} threads · ${Number(match.premat_warps_per_block || 0)} warps · ${Number(match.premat_registers_per_block || 0).toLocaleString()} regs · ${Number(match.premat_shared_mem_bytes || 0).toLocaleString()} B shared<br>`
        + `limiter: ${processing_escape(match.limiting_resource || "—")} · representative NCU MAIN layer ${match.main_layer === "" ? "—" : Number(match.main_layer) + 1}`
      );
    }
  }
  const traces = [...traces_by_class.values()].map(item => ({
    type: "bar", orientation: "h", name: item.klass,
    x: item.x, base: item.base, y: item.y, hovertext: item.hover, hoverinfo: "text",
    marker: {color: class_colours[item.klass]}, width: 0.62,
    legendgroup: item.klass, showlegend: false,
  }));
  shell.hidden = false;
  shell.style.flexBasis = `${Math.max(92, 52 + families.length * 28)}px`;
  const capture_ms = Number(payload.metadata?.capture_duration_ms || 0);
  await processing_plot("processing_resource_compatibility_plot", traces, {
    margin: {l: 92, r: 34, t: 24, b: 34}, barmode: "overlay", hovermode: "closest",
    title: {text: "Theoretical MAIN/PREMAT aggregate SM-budget co-residency", x: 0.01, xanchor: "left", font: {size: 11}},
    xaxis: {title: "capture time (ms)", range: capture_ms > 0 ? [0, capture_ms] : undefined},
    yaxis: {categoryorder: "array", categoryarray: families.map(family => `PREMAT ${family}`), automargin: true},
  });
}

async function processing_resource_render(payload) {
  const card = processing_resource_ensure_card();
  if (!card) return;
  const rows = Array.isArray(payload.stream_resources) ? payload.stream_resources : [];
  processing_view.resource_available = Number(payload.metadata?.schema_version || 1) >= 2 && rows.length > 0;
  processing_view.compatibility_available = processing_gpu_compatibility_rows(payload).length > 0;
  const any_resource_view = processing_view.resource_available || processing_view.compatibility_available;
  card.hidden = !(processing_view.charts_tab_visible && any_resource_view);
  card.classList.toggle("processing-resource-compat-only", !processing_view.resource_available && processing_view.compatibility_available);
  const heading = card.querySelector(".chart-heading-copy h2");
  const subtitle = card.querySelector(".chart-heading-copy > p:not(#processing_resource_idle_summary)");
  if (heading) heading.textContent = processing_view.resource_available ? "Stream resource pressure" : "PREMAT compatibility";
  if (subtitle) subtitle.textContent = processing_view.resource_available
    ? "Exact stream attribution only · device-wide counters remain combined wherever Main and PREMAT overlap."
    : "Nsight Compute structural result · aggregate SM-budget co-residency; not a timing measurement.";
  if (!processing_view.resource_available) {
    const idle = by_id("processing_resource_idle_summary");
    if (idle) idle.textContent = "";
    await processing_gpu_render_compatibility(payload);
    processing_gpu_schedule_resize();
    return;
  }
  processing_resource_update_idle_summary(payload);
  const panes = processing_resource_panes(rows);
  const traces = [];
  const metric_catalog = payload.metadata?.metric_catalog || {};
  for (const spec of processing_gpu_resource_specs) {
    if (metric_catalog[spec.key] && metric_catalog[spec.key].available === false) continue;
    let legend_written = false;
    for (const pane of panes) {
      const trace = processing_gpu_resource_trace(rows, pane, spec, !legend_written, metric_catalog);
      if (!trace) continue;
      traces.push(trace);
      legend_written = true;
    }
  }
  const capture_ms = Number(payload.metadata?.capture_duration_ms || 0);
  const layout = {
    margin: {l: 92, r: 34, t: 18, b: 48}, hovermode: "closest",
    legend: {orientation: "h", y: 1.10, groupclick: "togglegroup"},
    uirevision: "processing-resource-v1",
    xaxis: {title: "capture time (ms)", range: capture_ms > 0 ? [0, capture_ms] : undefined},
    shapes: processing_resource_idle_shapes(payload),
  };
  panes.forEach((pane, index) => {
    const axis_name = index === 0 ? "yaxis" : `yaxis${index + 1}`;
    layout[axis_name] = {domain: pane.domain, title: pane.label, range: [0, 100], ticksuffix: "%", fixedrange: true};
  });
  await processing_plot("processing_resource_plot", traces, layout);

  const clock_rows = rows.filter(row => processing_resource_number(row.gpc_clock_mhz) !== null);
  const clock_mount = by_id("processing_resource_clock_plot");
  const clock_shell = clock_mount?.closest(".processing-resource-clock-shell");
  if (clock_shell) clock_shell.hidden = !clock_rows.length;
  if (clock_rows.length) await processing_plot("processing_resource_clock_plot", [{
    type: "scattergl", mode: "lines", name: "GPC clock",
    x: clock_rows.map(row => Number(row.time_us) / 1000.0),
    y: clock_rows.map(row => Number(row.gpc_clock_mhz)),
    line: {width: 1.6, color: "#5c677d"},
    hovertemplate: "GPC clock %{y:.0f} MHz<br>%{x:.3f} ms<extra></extra>",
  }], {
    margin: {l: 92, r: 34, t: 20, b: 32}, showlegend: false,
    xaxis: {range: capture_ms > 0 ? [0, capture_ms] : undefined, showticklabels: true},
    yaxis: {title: "GPC MHz", rangemode: "tozero", fixedrange: true},
  });
  await processing_gpu_render_compatibility(payload);
  processing_gpu_link_time_axes();
  processing_gpu_link_inspection(payload);
  processing_gpu_schedule_resize();
}

function processing_gpu_link_time_axes() {
  const current_mounts = () => [
    by_id("processing_timeline_plot"), by_id("processing_resource_plot"),
    by_id("processing_resource_clock_plot"), by_id("processing_resource_compatibility_plot"),
  ].filter(mount => mount && mount.dataset.plotReady === "true" && mount.offsetParent !== null);
  for (const source of current_mounts()) {
    if (source._processing_gpu_axis_linked === true) continue;
    source._processing_gpu_axis_linked = true;
    source.on("plotly_relayout", event => {
      if (processing_view.gpu_axis_sync) return;
      const update = processing_resource_extract_xrange(event || {});
      if (!update) return;
      processing_view.gpu_axis_sync = true;
      const relayout = update.autorange ? {"xaxis.autorange": true} : {"xaxis.range": update.range};
      Promise.all(current_mounts().filter(target => target !== source).map(target => Promise.resolve(Plotly.relayout(target, relayout))))
        .finally(() => { processing_view.gpu_axis_sync = false; });
    });
  }
}
processing_resource_link_time_axes = processing_gpu_link_time_axes;

function processing_gpu_point_time(point) {
  if (Array.isArray(point?.customdata) && Number.isFinite(Number(point.customdata[6])) && Number.isFinite(Number(point.customdata[7]))) {
    return (Number(point.customdata[6]) + Number(point.customdata[7])) / 2;
  }
  const base = Number(point?.base);
  const width = Number(point?.x);
  if (Number.isFinite(base) && Number.isFinite(width)) return base + width / 2;
  const x = Number(point?.x);
  return Number.isFinite(x) ? x : null;
}

function processing_gpu_set_linked_cursor(time_ms, selected) {
  const mounts = [
    by_id("processing_timeline_plot"), by_id("processing_resource_plot"),
    by_id("processing_resource_clock_plot"), by_id("processing_resource_compatibility_plot"),
  ].filter(mount => mount && mount.dataset.plotReady === "true" && mount.offsetParent !== null);
  for (const mount of mounts) {
    const shapes = [...(mount.layout?.shapes || [])].filter(shape => shape?.name !== "processing-linked-cursor");
    if (Number.isFinite(time_ms)) shapes.push({
      name: "processing-linked-cursor", type: "line", xref: "x", yref: "paper",
      x0: time_ms, x1: time_ms, y0: 0, y1: 1,
      line: {color: selected ? "#171717" : "rgba(23,23,23,.55)", width: selected ? 2 : 1, dash: selected ? "solid" : "dot"},
      layer: "above",
    });
    Plotly.relayout(mount, {shapes});
  }
}

function processing_gpu_selected_detail(payload, time_ms) {
  const detail = by_id("processing_resource_detail");
  if (!detail) return;
  const rows = Array.isArray(payload.stream_resources) ? payload.stream_resources : [];
  const sample = rows.find(row => Number(row.sample_start_us) / 1000 <= time_ms && Number(row.sample_end_us) / 1000 >= time_ms)
    || rows.reduce((best, row) => !best || Math.abs(Number(row.time_us) / 1000 - time_ms) < Math.abs(Number(best.time_us) / 1000 - time_ms) ? row : best, null);
  const interval = (payload.intervals || []).find(row => Number(row.start_us) / 1000 <= time_ms && Number(row.end_us) / 1000 >= time_ms && String(row.owner) === "MAIN")
    || (payload.intervals || []).find(row => Number(row.start_us) / 1000 <= time_ms && Number(row.end_us) / 1000 >= time_ms);
  const compatibility = interval ? processing_gpu_matching_compatibility(processing_gpu_compatibility_rows(payload), interval) : null;
  const operation_stats = interval ? (payload.operation_resource_stats || []).filter(row => (
    String(row.op_id) === String(interval.op_id)
    && String(row.owner) === String(interval.owner)
  )) : [];
  const metric_catalog = payload.metadata?.metric_catalog || {};
  const metrics = processing_gpu_resource_specs.flatMap(spec => {
    const raw = processing_resource_number(sample?.[spec.key]);
    if (raw === null || metric_catalog[spec.key]?.available === false) return [];
    const value = spec.invert ? 100 - raw : raw;
    return [`${spec.label} ${value.toFixed(1)}%`];
  });
  const lifecycle = (payload.premat_lifecycle || []).filter(row => Math.abs(Number(row.capture_time_ms) - time_ms) <= 0.25).map(row => row.event);
  const fragments = [
    `<span><strong>${time_ms.toFixed(3)} ms</strong> · ${processing_escape(sample?.attribution_state || "no sampled attribution")}</span>`,
    `<span>${processing_resource_context(sample || {})}</span>`,
    `<span>${processing_escape(metrics.join(" · ") || "No finite resource metric in this bin")}</span>`,
  ];
  if (compatibility) fragments.push(`<span>NCU: <strong>${processing_escape(compatibility.compatibility_class || "—")}</strong> · limiter ${processing_escape(compatibility.limiting_resource || "—")} · MAIN ${Number(compatibility.main_registers_per_block || 0).toLocaleString()} regs/${Number(compatibility.main_shared_mem_bytes || 0).toLocaleString()} B smem · PREMAT ${Number(compatibility.premat_registers_per_block || 0).toLocaleString()} regs/${Number(compatibility.premat_shared_mem_bytes || 0).toLocaleString()} B smem</span>`);
  if (operation_stats.length) fragments.push(`<span>Interval weighted: ${processing_escape(operation_stats.slice(0, 6).map(row => {
    const spec = processing_gpu_resource_specs.find(candidate => candidate.key === row.metric);
    const label = spec?.label || row.metric;
    if (spec?.invert) return `${label} mean ${(100 - Number(row.mean)).toFixed(1)}, range ${(100 - Number(row.max)).toFixed(1)}–${(100 - Number(row.min)).toFixed(1)}`;
    return `${label} mean ${Number(row.mean).toFixed(1)}, p95 ${Number(row.p95).toFixed(1)}`;
  }).join(" · "))}</span>`);
  if (lifecycle.length) fragments.push(`<span>Lifecycle: ${processing_escape([...new Set(lifecycle)].join(", "))}</span>`);
  detail.innerHTML = fragments.join("");
  detail.hidden = false;
}

function processing_gpu_link_inspection(payload) {
  processing_view.processing_payload = payload;
  const mounts = [
    by_id("processing_timeline_plot"), by_id("processing_resource_plot"),
    by_id("processing_resource_clock_plot"), by_id("processing_resource_compatibility_plot"),
  ].filter(mount => mount && mount.dataset.plotReady === "true");
  for (const mount of mounts) {
    if (mount._processing_inspection_handlers) {
      for (const [name, handler] of Object.entries(mount._processing_inspection_handlers)) mount.removeListener?.(name, handler);
    }
    const hover = event => {
      const time_ms = processing_gpu_point_time(event?.points?.[0]);
      if (time_ms === null) return;
      processing_view.processing_hover_time = time_ms;
      if (processing_view.processing_hover_frame) return;
      processing_view.processing_hover_frame = requestAnimationFrame(() => {
        processing_view.processing_hover_frame = 0;
        processing_gpu_set_linked_cursor(processing_view.processing_hover_time, false);
      });
    };
    const unhover = () => processing_gpu_set_linked_cursor(processing_view.processing_selected_time, true);
    const click = event => {
      const time_ms = processing_gpu_point_time(event?.points?.[0]);
      if (time_ms === null) return;
      processing_view.processing_selected_time = time_ms;
      processing_gpu_set_linked_cursor(time_ms, true);
      processing_gpu_selected_detail(processing_view.processing_payload || payload, time_ms);
    };
    mount.on("plotly_hover", hover);
    mount.on("plotly_unhover", unhover);
    mount.on("plotly_click", click);
    mount._processing_inspection_handlers = {plotly_hover: hover, plotly_unhover: unhover, plotly_click: click};
  }
}

function processing_gpu_operation_colour(operation, owner) {
  if (String(operation) === "materialize") return "#ff1493";
  const colours = {
    consume: "#0067c5", attention: "#7b61ff", layernorm: "#00a6d6",
    activation: "#f39c12", residual: "#607d8b", lm_head: "#6f42c1",
    loss: "#d98b2b", misc: "#8c8c8c", other: "#8c8c8c",
  };
  if (colours[String(operation)]) return colours[String(operation)];
  return String(owner) === "PREMAT" ? "#ff1493" : "#4d79a7";
}

async function processing_render_timeline(payload) {
  const groups = processing_trace_groups(payload.intervals || []);
  const traces = [];
  const lane_order = ["MAIN", "PREMAT", "OTHER", "UNKNOWN"];
  const owners = lane_order.filter(owner => (payload.intervals || []).some(row => String(row.owner || "UNKNOWN") === owner));
  for (const [key, rows] of groups) {
    if (!rows.length) continue;
    const owner = String(rows[0].owner || "UNKNOWN");
    const operation = String(rows[0].operation || "misc");
    traces.push({
      type: "bar", orientation: "h", name: key.replace(":", " "),
      x: rows.map(row => Math.max(0, Number(row.end_us) - Number(row.start_us)) / 1000.0),
      base: rows.map(row => Number(row.start_us) / 1000.0),
      y: rows.map(() => owner), width: 0.66,
      marker: {
        color: processing_gpu_operation_colour(operation, owner),
        line: operation === "materialize" ? {color: "#4a0030", width: 1.1} : {width: 0},
      },
      customdata: rows.map(row => [
        row.operation, row.family,
        row.layer === "" || row.layer === null || row.layer === undefined ? "—" : Number(row.layer) + 1,
        row.kernel_name, row.owner_source, row.op_id, Number(row.start_us) / 1000.0,
        Number(row.end_us) / 1000.0, row.owner,
      ]),
      hovertemplate: "%{y} · %{customdata[0]} · %{customdata[1]} · L%{customdata[2]}<br>%{customdata[3]}<br>%{x:.4f} ms<extra></extra>",
    });
  }
  const lifecycle = Array.isArray(payload.premat_lifecycle) ? [...payload.premat_lifecycle] : [];
  if (lifecycle.length && !owners.includes("PREMAT")) owners.push("PREMAT");
  for (const summary of payload.premat_lifecycle_summary || []) {
    for (const [field, event] of [["gpu_start_ms", "gpu_start"], ["gpu_end_ms", "gpu_end"]]) {
      if (summary[field] === "" || summary[field] === null || summary[field] === undefined || !Number.isFinite(Number(summary[field]))) continue;
      lifecycle.push({...summary, event, capture_time_ms:Number(summary[field]), owner:"PREMAT", state:"GPU", outcome:summary.final_outcome, reason:"exact NSYS CUDA interval"});
    }
  }
  const lifecycle_groups = new Map();
  for (const row of lifecycle) {
    const event = String(row.event || "event");
    const category = event.startsWith("deadline_") ? "deadline" : event;
    if (!lifecycle_groups.has(category)) lifecycle_groups.set(category, []);
    lifecycle_groups.get(category).push(row);
  }
  const lifecycle_colours = {
    admission_considered: "#607d8b", admission_deferred: "#d98b2b", materialising: "#ff1493",
    gpu_start: "#c21875", gpu_end: "#8e24aa", available: "#2e9d57", deadline: "#222", critical_path_wait: "#d63c3c",
    consuming: "#0067c5", consumed: "#6f42c1", pass_end_release: "#8c8c8c",
  };
  for (const [category, rows] of lifecycle_groups) {
    traces.push({
      type: "scatter", mode: "markers", name: `lifecycle ${category}`,
      x: rows.map(row => Number(row.capture_time_ms)),
      y: rows.map(row => owners.includes(String(row.owner || "").toUpperCase()) ? String(row.owner).toUpperCase() : "PREMAT"),
      marker: {size: 8, symbol: category === "deadline" ? "triangle-down" : "diamond", color: lifecycle_colours[category] || "#555", line: {color: "#fff", width: 1}},
      customdata: rows.map(row => [row.event, row.family, row.layer === "" ? "—" : Number(row.layer) + 1, row.outcome, row.reason, row.timing_basis, row.job_id]),
      hovertemplate: "%{customdata[0]} · %{customdata[1]} · L%{customdata[2]}<br>%{customdata[3]} · %{customdata[4]}<br>%{customdata[5]} · %{customdata[6]}<extra></extra>",
      legendgroup: "lifecycle",
      visible: ["materialising", "gpu_start", "gpu_end", "available", "deadline", "critical_path_wait", "consuming", "consumed"].includes(category) ? true : "legendonly",
    });
  }
  const capture_ms = Number(payload.metadata?.capture_duration_ms || 0);
  await processing_plot("processing_timeline_plot", traces, {
    margin: {l: 92, r: 34, t: 12, b: 48}, barmode: "overlay", hovermode: "closest",
    legend: {orientation: "h", y: 1.08},
    xaxis: {title: "capture time (ms)", range: capture_ms > 0 ? [0, capture_ms] : undefined},
    yaxis: {categoryorder: "array", categoryarray: owners, automargin: true},
  });
  processing_gpu_link_time_axes();
  processing_gpu_link_inspection(payload);
}

function processing_gpu_ensure_throughput_controls() {
  const card = document.querySelector('.processing-throughput-card[data-chart="processing_throughput"]');
  const actions = card?.querySelector(".chart-card-actions");
  if (!actions) return;
  if (!by_id("processing_throughput_x_mode")) {
    const select = document.createElement("select");
    select.id = "processing_throughput_x_mode";
    select.className = "processing-throughput-axis-select";
    select.title = "Training throughput x axis";
    for (const [value, label] of [
      ["step", "Steps"], ["relative_wall", "Relative wall time"],
      ["relative_process", "Relative process time"], ["wall_time", "Wall time"],
    ]) {
      const option = document.createElement("option");
      option.value = value; option.textContent = label; select.appendChild(option);
    }
    select.value = processing_view.throughput_axis_mode;
    select.addEventListener("change", () => {
      processing_view.throughput_axis_mode = select.value;
      localStorage.setItem("thog2_processing_throughput_x_mode", select.value);
      if (processing_view.throughput_last_payload) processing_render_throughput(processing_view.throughput_last_payload);
    });
    actions.insertBefore(select, actions.firstChild);
  }
  if (!by_id("processing_throughput_z_button")) {
    const button = document.createElement("button");
    button.id = "processing_throughput_z_button";
    button.className = "processing-throughput-z-button";
    button.type = "button";
    button.textContent = "z";
    button.title = "Cycle which throughput curve is drawn on top";
    button.addEventListener("click", processing_gpu_cycle_throughput_z);
    actions.insertBefore(button, actions.querySelector(".maximize-button"));
  }
}

function processing_gpu_cycle_throughput_z() {
  processing_view.throughput_z_offset += 1;
  if (processing_view.throughput_last_payload) {
    processing_render_throughput(processing_view.throughput_last_payload);
  }
}

window.processing_gpu_cycle_throughput_z = processing_gpu_cycle_throughput_z;

function processing_gpu_throughput_value(row, mode) {
  if (mode === "step") return Number(row.optimizer_update);
  const key = mode === "relative_wall" ? "relative_wall_seconds" : mode === "relative_process" ? "process_time_seconds" : "wall_time";
  const raw = row[key];
  if (raw === null || raw === undefined || raw === "") return Number.NaN;
  const value = Number(raw);
  if (!Number.isFinite(value)) return Number.NaN;
  return mode === "wall_time" ? value * 1000.0 : value / 3600.0;
}

function processing_gpu_throughput_mode_available(rows, mode) {
  if (mode === "step") return true;
  return (rows || []).some(row => Number.isFinite(processing_gpu_throughput_value(row, mode)));
}

async function processing_render_throughput(payload) {
  processing_view.throughput_last_payload = payload;
  processing_gpu_ensure_throughput_controls();
  const runs = processing_throughput_workspace_runs();
  const resolved = await Promise.all(runs.map(async run => ({
    run, run_id: String(run_identifier(run)), rows: await processing_throughput_rows_for_run(run, payload),
  })));
  const populated = resolved.filter(entry => entry.rows.length);
  const select = by_id("processing_throughput_x_mode");
  if (select) {
    for (const option of select.options) {
      option.disabled = option.value !== "step" && !populated.every(entry => (
        entry.rows.some(row => Number.isFinite(processing_gpu_throughput_value(row, option.value)))
      ));
    }
    if (select.selectedOptions[0]?.disabled) {
      select.value = "step";
      processing_view.throughput_axis_mode = "step";
      localStorage.setItem("thog2_processing_throughput_x_mode", "step");
    }
  }
  const mode = processing_view.throughput_axis_mode;
  const rotated = populated.length ? populated.map((_entry, index) => populated[(index + processing_view.throughput_z_offset) % populated.length]) : [];
  const traces = [];
  for (const entry of rotated) {
    const valid_rows = entry.rows.filter(row => Number.isFinite(processing_gpu_throughput_value(row, mode)));
    if (!valid_rows.length) continue;
    const name = processing_gpu_display_text(entry.run.artifact_name || entry.run.run_name || entry.run_id);
    const colour = colour_for_run(entry.run_id);
    traces.push({
      type: "scatter", mode: valid_rows.length === 1 ? "markers" : "lines", name,
      meta: {instra_workspace_run_id: entry.run_id},
      x: valid_rows.map(row => processing_gpu_throughput_value(row, mode)),
      y: valid_rows.map(row => Number(row.tokens_per_second)),
      line: {color: colour, width: 2.4}, marker: {color: colour},
      hovertemplate: mode === "wall_time"
        ? "%{x|%Y-%m-%d %H:%M:%S}<br>%{y:,.0f} tok/s<extra>%{fullData.name}</extra>"
        : "%{x:.4g}<br>%{y:,.0f} tok/s<extra>%{fullData.name}</extra>",
    });
  }
  processing_view.training_throughput_available = traces.length > 0;
  const maximum_points = Math.max(0, ...populated.map(entry => entry.rows.length));
  const workspace = app.workspace_mode === true;
  const x_titles = {step: "optimizer update", relative_wall: "relative wall time (hours)", relative_process: "relative process time (hours)", wall_time: "wall time"};
  await processing_plot("processing_throughput_plot", traces, {
    margin: {l: 72, r: 24, t: traces.length > 1 ? Math.min(132, 20 + traces.length * 18) : 12, b: 54},
    xaxis: {title: x_titles[mode], type: mode === "wall_time" ? "date" : undefined, dtick: mode === "step" && maximum_points <= 20 ? 1 : undefined},
    yaxis: {title: "tokens / second", rangemode: "tozero", separatethousands: true},
    showlegend: traces.length > 1,
    legend: {orientation: "v", x: 0, xanchor: "left", y: 1.02, yanchor: "bottom", font:{size:9}},
    annotations: traces.length ? [] : [{
      text: workspace ? "No retained tok/s samples for visible Workspace runs" : "No retained tok/s samples for this run",
      showarrow: false, xref: "paper", yref: "paper", x: 0.5, y: 0.5,
    }],
  });
  const z_button = by_id("processing_throughput_z_button");
  if (z_button) {
    const top = traces.length ? traces[traces.length - 1].name : "—";
    z_button.title = `Cycle throughput draw order · currently on top: ${top}`;
  }
}

const processing_set_chart_artifacts_before_gpu_resource_tool = processing_set_chart_artifacts;
processing_set_chart_artifacts = function(payload) {
  processing_set_chart_artifacts_before_gpu_resource_tool(payload);
  for (const id of ["processing_timeline_artifact", "processing_contention_artifact"]) {
    const element = by_id(id);
    if (element) element.textContent = processing_gpu_display_text(element.textContent);
  }
};

function processing_gpu_replace_display_nomat(root) {
  if (!root) return;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  for (const node of nodes) {
    if (node.nodeValue?.includes("NOMAT")) node.nodeValue = node.nodeValue.replaceAll("NOMAT", "MAT");
  }
}

function processing_gpu_ensure_download_links(payload) {
  const sample_link = by_id("processing_download_stream_resources") || by_id("processing_download_samples");
  if (!sample_link?.parentElement) return;
  const files = payload?.premat_compatibility_files || {};
  for (const [id, label, filename] of [
    ["processing_download_ncu_resources", "NCU resources", files.kernel_resources],
    ["processing_download_compatibility", "Compatibility", files.csv || files.json],
  ]) {
    let link = by_id(id);
    if (!link) {
      link = sample_link.cloneNode(false); link.id = id; link.textContent = label;
      sample_link.parentElement.appendChild(link);
    }
    link.hidden = !filename;
    if (filename) link.href = processing_download_url(filename);
  }
}

const processing_render_before_gpu_resource_tool = processing_render;
processing_render = async function(payload, trace_available) {
  await processing_render_before_gpu_resource_tool(payload, trace_available);
  processing_gpu_ensure_throughput_controls();
  processing_gpu_ensure_download_links(payload);
  processing_gpu_replace_display_nomat(by_id("processing_chart_group"));
  processing_gpu_schedule_resize();
};

document.addEventListener("click", event => {
  if (event.target.closest?.("#processing_chart_group .maximize-button")) processing_gpu_schedule_resize();
}, true);

window.addEventListener("load", () => {
  processing_gpu_inject_style();
  processing_resource_ensure_card();
  processing_gpu_ensure_throughput_controls();
  const group = by_id("processing_chart_group");
  if (group && typeof MutationObserver === "function") {
    const observer = new MutationObserver(() => processing_gpu_replace_display_nomat(group));
    observer.observe(group, {childList: true, subtree: true, characterData: true});
  }
});

// vvv THOG Processing overlap chart uses non-semantic colours; red/green remain reserved for compatibility status.
const processing_render_contention_before_gpu_resource_tool = processing_render_contention;
processing_render_contention = async function(payload) {
  await processing_render_contention_before_gpu_resource_tool(payload);
  const mount = by_id("processing_contention_plot");
  if (!mount || mount.dataset.plotReady !== "true" || !Array.isArray(mount.data)) return;
  const colours = ["#0067c5", "#ff1493", "#7b61ff", "#f39c12"];
  const updates = [];
  const indices = [];
  mount.data.forEach((trace, index) => {
    if (trace?.type !== "bar") return;
    updates.push(colours[updates.length % colours.length]);
    indices.push(index);
  });
  if (indices.length) await Plotly.restyle(mount, {"marker.color": updates}, indices);
};
// ^^^ THOG

// ^^^ THOG GPU resource compatibility tool v1
