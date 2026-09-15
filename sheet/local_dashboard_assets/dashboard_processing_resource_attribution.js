// vvv THOG
"use strict";

processing_view.resource_available = false;
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
  if (card) card.hidden = !(processing_view.charts_tab_visible && processing_view.resource_available);
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
