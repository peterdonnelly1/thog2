// vvv THOG
"use strict";

const chart_titles = {
  heatmap: "Layer-count Δloss heatmap",
  attn_q_head_N: "Attention query scalar trajectories",
  attn_k_head_N: "Attention key scalar trajectories",
  attn_v_head_N: "Attention value scalar trajectories",
  attn_out_head_N: "Attention output scalar trajectories",
  mlp_up: "MLP expansion scalar trajectories",
  mlp_down: "MLP contraction scalar trajectories",
};

const chart_groups = {
  depth: Object.keys(chart_titles),
};

const default_palette = [
  "#865ed6", "#e76f38", "#f4ab83", "#ffb73d", "#91bf57", "#4d9d71",
  "#218d80", "#78c2b6", "#48b4cf", "#578adb", "#7a59d1", "#df79d9",
  "#cf47c5", "#ae317e", "#a96f54", "#9fa7ad",
];

const plot_config = {
  responsive: true,
  scrollZoom: true,
  displaylogo: false,
  modeBarButtonsToRemove: ["lasso2d", "select2d"],
  toImageButtonOptions: {format: "png", scale: 2},
};

const panel_layout_version = "heatmap-row-and-three-by-two-curves-v1";

const app = {
  runs: [],
  requested_run: null,
  recommended_run_id: null,
  root: "",
  current_run_id: null,
  current_status: null,
  figures: null,
  figure_revision: null,
  refresh_in_flight: false,
  manual_selection: false,
  initial_route_run_id: route_run_id(),
  colours: load_json("thog2_local_run_colours", {}),
  visibility: load_json("thog2_local_run_visibility", {}),
  panel_sizes: load_json("thog2_local_panel_sizes", {}),
  axis_ranges: load_json("thog2_local_chart_axis_ranges", {}),
  file_source: localStorage.getItem("instra_file_source") === "wandb" ? "wandb" : "instra",
  file_path: "",
  file_payload: null,
  file_loading: false,
  file_request_serial: 0,
  page_size: load_number("thog2_local_page_size", 50),
  current_page: 1,
  sort_descending: localStorage.getItem("thog2_local_sort_descending") !== "0",
  crash_timeout_minutes: load_number("thog2_local_crash_timeout_minutes", 15),
  selected: new Set(),
  group_by_host: false,
  maximized_chart: null,
  colour_run_id: null,
  menu_run_id: null,
  picker_hue: 250,
  picker_saturation: 56,
  picker_value: 84,
  axis_chart_name: null,
  chart_settings_render_override: null,
  chart_settings_preview_serial: 0,
  chart_settings_preview_timer: null,
};
if (!app.axis_ranges || typeof app.axis_ranges !== "object" || Array.isArray(app.axis_ranges)) app.axis_ranges = {};

function by_id(id) { return document.getElementById(id); }

function load_json(key, fallback) {
  try {
    const value = JSON.parse(localStorage.getItem(key));
    return value === null ? fallback : value;
  } catch (_error) {
    return fallback;
  }
}

function save_json(key, value) { localStorage.setItem(key, JSON.stringify(value)); }

function load_number(key, fallback) {
  const value = Number(localStorage.getItem(key));
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function route_run_id() {
  const match = /^\/runs\/([^/]+)$/.exec(window.location.pathname);
  return match ? decodeURIComponent(match[1]) : null;
}

async function fetch_json(url, options = {}) {
  const response = await fetch(url, {cache: "no-store", ...options});
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `${response.status} ${response.statusText}`);
  return value;
}

function hash_text(value) {
  let hash = 2166136261;
  for (const character of value) {
    hash ^= character.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function colour_for_run(run_id) {
  return app.colours[run_id] || default_palette[hash_text(run_id) % default_palette.length];
}

function is_visible(run_id) { return app.visibility[run_id] !== false; }
function run_identifier(run) { return String(run.dashboard_run_id || run.local_run_id || run.wandb_run_id || run.run_name); }
function current_run() { return app.runs.find(run => run_identifier(run) === app.current_run_id) || app.current_status; }

function display_run_state(run) {
  if (run.run_state !== "running") return run.run_state;
  const last_write = Date.parse(run.updated_at || run.created_at || "");
  if (!Number.isFinite(last_write)) return "running";
  const timeout_ms = app.crash_timeout_minutes * 60 * 1000;
  return Date.now() - last_write > timeout_ms ? "crashed" : "running";
}

function format_integer(value) {
  return value === null || value === undefined ? "—" : Number(value).toLocaleString();
}

function format_bytes(value) {
  const bytes = Number(value || 0);
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

function format_time(value) {
  if (!value) return "—";
  const timestamp = new Date(value);
  const delta_seconds = Math.max(0, Math.round((Date.now() - timestamp.getTime()) / 1000));
  if (delta_seconds < 10) return "just now";
  if (delta_seconds < 60) return `${delta_seconds}s ago`;
  if (delta_seconds < 3600) return `${Math.floor(delta_seconds / 60)}m ago`;
  if (delta_seconds < 86400) return `${Math.floor(delta_seconds / 3600)}h ago`;
  return timestamp.toLocaleString();
}

function text_cell(value, class_name = "") {
  const cell = document.createElement("td");
  cell.textContent = value;
  if (class_name) cell.className = class_name;
  return cell;
}

function icon_svg(kind) {
  const markup = {
    eye_open: '<path d="M1.5 8s2.8-4.5 8.5-4.5S18.5 8 18.5 8s-2.8 4.5-8.5 4.5S1.5 8 1.5 8Z"/><circle cx="10" cy="8" r="2.3"/>',
    eye_closed: '<path d="M2 6.4c2.4 2.8 5 4 8 4s5.6-1.2 8-4"/><path d="M4.5 9.1 3.3 11M8 10.2l-.4 2.2M12 10.2l.4 2.2M15.5 9.1l1.2 1.9"/>',
    running: '<path d="M4.2 4.6A4.4 4.4 0 0 1 11.7 3L13 4.4M11.8 9.4A4.4 4.4 0 0 1 4.3 11L3 9.6"/><path d="M13 1.8v2.6h-2.6M3 12.2V9.6h2.6"/>',
    finished: '<circle cx="8" cy="8" r="5.2"/><path d="m5.2 8 1.8 1.8 3.8-4"/>',
    crashed: '<circle cx="8" cy="8" r="5.2"/><path d="m6 6 4 4M10 6l-4 4"/>',
    unknown: '<circle cx="8" cy="8" r="5.2"/><path d="M6.6 6.2A1.6 1.6 0 0 1 8.1 5c1 0 1.8.6 1.8 1.5 0 1.4-1.9 1.4-1.9 2.7M8 11.2h.01"/>',
  }[kind];
  const template = document.createElement("template");
  const view_box = kind.startsWith("eye_") ? "0 0 20 16" : "0 0 16 16";
  template.innerHTML = `<svg viewBox="${view_box}" aria-hidden="true">${markup}</svg>`;
  return template.content.firstElementChild;
}

function filtered_runs() {
  const query = by_id("run_search").value.trim().toLowerCase();
  const filter = by_id("state_filter").value;
  const sort = by_id("run_sort").value;
  const runs = app.runs.filter(run => {
    const searchable = `${run.artifact_name} ${run.wandb_run_id} ${run.local_run_id} ${run.host_label}`.toLowerCase();
    return (!query || searchable.includes(query)) && (filter === "all" || display_run_state(run) === filter);
  });
  runs.sort((left, right) => {
    let comparison = 0;
    if (sort === "name") comparison = String(left.artifact_name).localeCompare(String(right.artifact_name));
    else if (sort === "heatmap") comparison = Number(left.heatmap_count) - Number(right.heatmap_count);
    else if (sort === "depth") comparison = Number(left.depth_snapshot_count) - Number(right.depth_snapshot_count);
    else if (sort === "updated") comparison = String(left.updated_at).localeCompare(String(right.updated_at));
    else comparison = String(left.created_at).localeCompare(String(right.created_at));
    if (!comparison) comparison = run_identifier(left).localeCompare(run_identifier(right));
    return app.sort_descending ? -comparison : comparison;
  });
  return runs;
}

function reset_pagination() {
  app.current_page = 1;
  render_runs();
}

function append_run_row(body, run) {
  const run_id = run_identifier(run);
  const row = document.createElement("tr");
  row.dataset.runId = run_id;
  row.style.setProperty("--run-colour", colour_for_run(run_id));
  row.classList.toggle("run-hidden", !is_visible(run_id));
  row.classList.toggle("current-run", run_id === app.current_run_id);

  const check_cell = document.createElement("td");
  check_cell.className = "check-column";
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.checked = app.selected.has(run_id);
  checkbox.setAttribute("aria-label", `Select ${run.artifact_name}`);
  checkbox.addEventListener("change", () => {
    if (checkbox.checked) app.selected.add(run_id); else app.selected.delete(run_id);
  });
  check_cell.appendChild(checkbox);
  row.appendChild(check_cell);

  const visibility_cell = document.createElement("td");
  visibility_cell.className = "visibility-column";
  const eye = document.createElement("button");
  eye.type = "button";
  eye.className = "eye-button";
  eye.classList.toggle("off", !is_visible(run_id));
  eye.appendChild(icon_svg(is_visible(run_id) ? "eye_open" : "eye_closed"));
  eye.title = is_visible(run_id) ? "Hide run" : "Show run";
  eye.setAttribute("aria-label", eye.title);
  eye.addEventListener("click", () => {
    app.visibility[run_id] = !is_visible(run_id);
    save_json("thog2_local_run_visibility", app.visibility);
    render_runs();
  });
  visibility_cell.appendChild(eye);
  row.appendChild(visibility_cell);

  const name_cell = document.createElement("td");
  const colour = document.createElement("button");
  colour.type = "button";
  colour.className = "colour-dot";
  colour.style.background = colour_for_run(run_id);
  colour.title = "Change run colour";
  colour.setAttribute("aria-label", `Change colour for ${run.artifact_name}`);
  colour.addEventListener("click", event => {
    event.stopPropagation();
    open_colour_picker(run_id, colour);
  });
  const name = document.createElement("button");
  name.type = "button";
  name.className = "run-link";
  name.textContent = run.artifact_name;
  name.title = run.artifact_name;
  name.addEventListener("click", () => select_run(run_id, {manual: true}));
  name_cell.append(colour, name);
  row.appendChild(name_cell);

  const wandb_cell = document.createElement("td");
  wandb_cell.className = "run-id";
  if (run.wandb_url) {
    const link = document.createElement("a");
    link.href = run.wandb_url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = run.wandb_run_id || "open";
    link.title = "Open W&B run";
    wandb_cell.appendChild(link);
  } else {
    wandb_cell.textContent = run.wandb_run_id || (String(run.local_run_id).startsWith("local-") ? run.local_run_id : "legacy");
  }
  if (run.is_legacy_layout) {
    const legacy = document.createElement("span");
    legacy.className = "legacy-tag";
    legacy.textContent = "old store";
    wandb_cell.appendChild(legacy);
  }
  row.appendChild(wandb_cell);

  const state_cell = document.createElement("td");
  const badge = document.createElement("span");
  const shown_state = display_run_state(run);
  badge.className = `state-badge ${shown_state}`;
  const state_icon = document.createElement("span");
  state_icon.className = "state-icon";
  const state_icon_name = ["running", "finished", "crashed"].includes(shown_state) ? shown_state : "unknown";
  state_icon.appendChild(icon_svg(state_icon_name));
  badge.append(state_icon, document.createTextNode(shown_state));
  if (shown_state === "crashed") {
    badge.title = `No instra chart data for more than ${app.crash_timeout_minutes} minutes`;
  }
  state_cell.appendChild(badge);
  row.appendChild(state_cell);
  row.appendChild(text_cell(run.host_label || "—"));
  row.appendChild(text_cell(format_integer(run.heatmap_count), "numeric-column"));
  row.appendChild(text_cell(format_integer(run.depth_snapshot_count), "numeric-column"));
  row.appendChild(text_cell(format_integer(run.maximum_update), "numeric-column"));
  row.appendChild(text_cell(format_time(run.updated_at)));
  const menu_cell = document.createElement("td");
  menu_cell.className = "menu-column";
  const menu_button = document.createElement("button");
  menu_button.type = "button";
  menu_button.className = "run-menu-button";
  menu_button.textContent = "…";
  menu_button.title = "Run actions";
  menu_button.setAttribute("aria-label", `Actions for ${run.artifact_name}`);
  menu_button.setAttribute("aria-expanded", "false");
  menu_button.addEventListener("click", event => {
    event.stopPropagation();
    open_run_menu(run_id, menu_button);
  });
  menu_cell.appendChild(menu_button);
  row.appendChild(menu_cell);
  row.addEventListener("click", event => {
    if (!event.target.closest("button, input, a")) select_run(run_id, {manual: true});
  });
  body.appendChild(row);
}

function render_runs() {
  const body = by_id("runs_body");
  body.replaceChildren();
  const runs = filtered_runs();
  const page_count = Math.max(1, Math.ceil(runs.length / app.page_size));
  app.current_page = Math.max(1, Math.min(app.current_page, page_count));
  const page_start = (app.current_page - 1) * app.page_size;
  const page_runs = runs.slice(page_start, page_start + app.page_size);
  by_id("run_count").textContent = String(app.runs.length);
  by_id("listed_count").textContent = `${runs.length} listed`;
  by_id("page_size").value = String(app.page_size);
  by_id("page_range").textContent = runs.length
    ? `${page_start + 1}-${page_start + page_runs.length} of ${runs.length}`
    : "0 of 0";
  by_id("previous_page").disabled = app.current_page <= 1;
  by_id("next_page").disabled = app.current_page >= page_count;
  const empty = by_id("empty_runs");
  empty.hidden = runs.length !== 0;
  if (!runs.length) {
    by_id("empty_runs_title").textContent = app.runs.length ? "No matching runs" : "Waiting for a local run";
    by_id("empty_runs_detail").textContent = app.runs.length
      ? "Change the search or filter to see runs."
      : "instra is ready. Start training and the W&B-linked run will appear here automatically.";
    return;
  }
  let previous_group = null;
  for (const run of page_runs) {
    const group = run.host_label || "Unlabelled host";
    if (app.group_by_host && group !== previous_group) {
      const group_row = document.createElement("tr");
      group_row.className = "group-row";
      const group_cell = document.createElement("td");
      group_cell.colSpan = 11;
      group_cell.textContent = group;
      group_row.appendChild(group_cell);
      body.appendChild(group_row);
      previous_group = group;
    }
    append_run_row(body, run);
  }
}

function should_follow_recommendation(recommended) {
  if (!recommended || app.manual_selection || recommended === app.current_run_id) return false;
  const selected = current_run();
  const candidate = app.runs.find(run => run_identifier(run) === recommended);
  if (!selected || !candidate) return true;
  if (candidate.run_state === "running" && selected.run_state !== "running") return true;
  if (candidate.wandb_run_id && selected.is_legacy_layout) return true;
  if (
    candidate.run_state === "running"
    && selected.run_state === "running"
    && String(candidate.created_at) > String(selected.created_at)
  ) return true;
  return false;
}

async function refresh_catalog() {
  try {
    const catalog = await fetch_json("/api/runs");
    app.runs = catalog.runs;
    app.requested_run = catalog.requested_run;
    app.recommended_run_id = catalog.recommended_run_id;
    app.root = catalog.root;
    const watch_text = catalog.waiting
      ? `Waiting · ${catalog.root}`
      : `${catalog.runs.length} run${catalog.runs.length === 1 ? "" : "s"} · ${catalog.root}`;
    by_id("watch_status").textContent = watch_text;
    by_id("topbar_state").textContent = watch_text;

    if (app.current_run_id && !app.runs.some(run => run_identifier(run) === app.current_run_id)) {
      app.file_request_serial += 1;
      app.current_run_id = null;
      app.manual_selection = false;
      app.file_path = "";
      app.file_payload = null;
      app.file_loading = false;
      reset_run_charts();
    }

    const route_candidate = app.initial_route_run_id && app.runs.some(run => run_identifier(run) === app.initial_route_run_id)
      ? app.initial_route_run_id
      : null;
    if (!app.current_run_id && route_candidate) {
      select_run(route_candidate, {manual: true, replace_history: true});
      app.initial_route_run_id = null;
    }
    if (!app.current_run_id && app.recommended_run_id) {
      select_run(app.recommended_run_id, {manual: false, replace_history: true});
    } else if (should_follow_recommendation(app.recommended_run_id)) {
      const old_id = app.current_run_id;
      select_run(app.recommended_run_id, {manual: false, replace_history: true});
      show_toast(`Switched from stale ${old_id} data to active W&B run ${app.recommended_run_id}.`);
    }

    render_runs();
    render_run_heading();
    render_empty_state();
  } catch (error) {
    by_id("watch_status").textContent = `Viewer error: ${error.message}`;
    by_id("topbar_state").textContent = "Viewer error";
  }
}

function clear_plot(mount) {
  if (mount.dataset.plotReady === "true") Plotly.purge(mount);
  mount.replaceChildren();
  delete mount.dataset.plotReady;
}

function clone_figure(figure) { return JSON.parse(JSON.stringify(figure)); }

function transpose_matrix(matrix) {
  if (!Array.isArray(matrix) || !matrix.length) return matrix;
  const width = Math.max(...matrix.map(row => Array.isArray(row) ? row.length : 0));
  return Array.from({length: width}, (_unused, column) =>
    matrix.map(row => Array.isArray(row) ? row[column] : null)
  );
}

function evenly_spaced_indices(length, limit) {
  if (length <= 0 || limit <= 0) return [];
  if (length <= limit) return Array.from({length}, (_unused, index) => index);
  if (limit === 1) return [length - 1];
  const stride = Math.ceil((length - 1) / (limit - 1));
  const indices = [];
  for (let index = 0; index < length; index += stride) indices.push(index);
  if (indices.at(-1) !== length - 1) indices.push(length - 1);
  return indices;
}

function transpose_heatmap(prepared) {
  const original_xaxis = {...(prepared.layout.xaxis || {})};
  const original_yaxis = {...(prepared.layout.yaxis || {})};
  let heatmap_trace = null;
  let active_layer_trace = null;
  for (const trace of prepared.data || []) {
    if (trace.type === "heatmap") {
      heatmap_trace = trace;
      const original_x = trace.x;
      trace.x = trace.y;
      trace.y = original_x;
      trace.z = transpose_matrix(trace.z);
      if (trace.customdata) trace.customdata = transpose_matrix(trace.customdata);
      trace.zsmooth = false;
      trace.xgap = 0;
      trace.ygap = 0;
      trace.hovertemplate = "step=%{customdata}<br>candidate layers=%{x}<br>Δloss=%{z:.8f}<extra></extra>";
      trace.colorbar = trace.colorbar || {};
      trace.colorbar.thickness = 12;
      trace.colorbar.len = 0.82;
    } else if (trace.x && trace.y) {
      active_layer_trace = trace;
      const original_x = trace.x;
      trace.x = trace.y;
      trace.y = original_x;
      const y_min = Number(original_xaxis.range?.[0] ?? 0.5);
      const initial_layer_count = Number(trace.x?.[0]);
      if (Number.isFinite(y_min) && Number.isFinite(initial_layer_count)) {
        trace.x = [initial_layer_count, ...trace.x];
        trace.y = [y_min, ...trace.y];
        if (Array.isArray(trace.customdata) && trace.customdata.length) {
          trace.customdata = [trace.customdata[0], ...trace.customdata];
        }
      }
      trace.line = {...(trace.line || {}), color: "white", width: 2};
      trace.hovertemplate = "step=%{customdata}<br>active layers=%{x}<extra></extra>";
    }
  }
  prepared.layout.xaxis = original_yaxis;
  prepared.layout.yaxis = original_xaxis;
  prepared.layout.xaxis.title = {text: "absolute candidate layer count", standoff: 46};
  prepared.layout.yaxis.title = {text: "step"};
  delete prepared.layout.xaxis.scaleanchor;
  delete prepared.layout.xaxis.scaleratio;
  prepared.layout.xaxis.constrain = "domain";
  prepared.layout.yaxis.constrain = "domain";
  prepared.layout.yaxis.constraintoward = "bottom";
  prepared.layout.yaxis.scaleanchor = "x";
  prepared.layout.yaxis.scaleratio = 1;

  const row_coordinates = heatmap_trace?.y || [];
  const step_values = (heatmap_trace?.customdata || []).map((row, index) =>
    Array.isArray(row) && row.length ? row[0] : row_coordinates[index]
  );
  const tick_indices = evenly_spaced_indices(row_coordinates.length, 20);
  prepared.layout.yaxis.tickmode = "array";
  prepared.layout.yaxis.tickvals = tick_indices.map(index => row_coordinates[index]);
  prepared.layout.yaxis.ticktext = tick_indices.map(index => String(step_values[index]));

  const initial_layer_count = Number(active_layer_trace?.x?.[0]);
  if (Number.isFinite(initial_layer_count)) {
    prepared.layout.annotations = [...(prepared.layout.annotations || []), {
      x: initial_layer_count,
      xref: "x",
      y: 0,
      yref: "paper",
      yshift: -27,
      yanchor: "top",
      text: `<b>${initial_layer_count}</b>`,
      showarrow: false,
      font: {size: 14, color: "#30343b"},
    }];
  }
}

function chart_default_labels(chart_name) {
  return chart_name === "heatmap"
    ? {x_source: "Candidate layer count", x_label: "absolute candidate layer count", y_source: "Optimizer step", y_label: "optimizer step"}
    : {x_source: "Layer sample number", x_label: "layer sample number", y_source: "Generated weight value", y_label: "weight value"};
}

function stored_chart_settings(chart_name) {
  const stored = app.axis_ranges && typeof app.axis_ranges === "object"
    ? app.axis_ranges[chart_name]
    : null;
  return stored && typeof stored === "object" && !Array.isArray(stored) ? stored : {};
}

function normalize_chart_settings(chart_name, supplied = null) {
  const stored = supplied || stored_chart_settings(chart_name);
  const labels = chart_default_labels(chart_name);
  const normalized = {
    title: typeof stored.title === "string" && stored.title.trim() ? stored.title.trim() : chart_titles[chart_name],
    x_label: typeof stored.x_label === "string" && stored.x_label.trim() ? stored.x_label.trim() : labels.x_label,
    y_label: typeof stored.y_label === "string" && stored.y_label.trim() ? stored.y_label.trim() : labels.y_label,
    max_snapshots: 0,
    exclude_outliers: stored.exclude_outliers === true,
    smoothing: 0,
    line_width: 1,
    chart_type: ["lines", "lines_markers", "markers"].includes(stored.chart_type) ? stored.chart_type : "lines",
    show_grid: stored.show_grid !== false,
  };
  for (const field of ["x_min", "x_max", "y_min", "y_max"]) {
    const value = Number(stored[field]);
    if (stored[field] !== null && stored[field] !== "" && Number.isFinite(value)) normalized[field] = value;
  }
  const max_snapshots = Number(stored.max_snapshots);
  if (Number.isInteger(max_snapshots) && max_snapshots > 0) normalized.max_snapshots = max_snapshots;
  const smoothing = Number(stored.smoothing);
  if (Number.isFinite(smoothing)) normalized.smoothing = Math.min(0.95, Math.max(0, smoothing));
  const line_width = Number(stored.line_width);
  if (Number.isFinite(line_width)) normalized.line_width = Math.min(3, Math.max(0.5, line_width));
  const heatmap_row_height = Number(stored.heatmap_row_height);
  if (Number.isFinite(heatmap_row_height)) normalized.heatmap_row_height = Math.min(24, Math.max(1, heatmap_row_height));
  return normalized;
}

function chart_axis_settings(chart_name, supplied = null) {
  const normalized = normalize_chart_settings(chart_name, supplied);
  const axes = {};
  for (const field of ["x_min", "x_max", "y_min", "y_max"]) {
    if (Number.isFinite(normalized[field])) axes[field] = normalized[field];
  }
  return axes;
}

function has_chart_axis_settings(chart_name) {
  return Object.keys(stored_chart_settings(chart_name)).length > 0;
}

function numeric_trace_bounds(prepared, axis_name) {
  let minimum = Infinity;
  let maximum = -Infinity;
  for (const trace of prepared.data || []) {
    for (const raw_value of trace[axis_name] || []) {
      const value = Number(raw_value);
      if (!Number.isFinite(value)) continue;
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
    }
  }
  if (!Number.isFinite(minimum) || !Number.isFinite(maximum)) return [0, 1];
  if (minimum !== maximum) return [minimum, maximum];
  const padding = Math.max(1, Math.abs(minimum) * 0.05);
  return [minimum - padding, maximum + padding];
}

function automatic_axis_bounds(prepared, axis_name) {
  const axis = prepared.layout[`${axis_name}axis`] || {};
  const range = axis.range || [];
  const range_minimum = Number(range[0]);
  const range_maximum = Number(range[1]);
  if (Number.isFinite(range_minimum) && Number.isFinite(range_maximum) && range_minimum < range_maximum) {
    return [range_minimum, range_maximum];
  }
  return numeric_trace_bounds(prepared, axis_name);
}

function heatmap_step_coordinate(prepared, requested_step, boundary) {
  const heatmap_trace = (prepared.data || []).find(trace => trace.type === "heatmap");
  const coordinates = heatmap_trace?.y || [];
  const steps = (heatmap_trace?.customdata || []).map((row, index) => {
    const value = Array.isArray(row) && row.length ? Number(row[0]) : Number(coordinates[index]);
    return Number.isFinite(value) ? value : Number(coordinates[index]);
  });
  if (!coordinates.length || coordinates.length !== steps.length) return requested_step;
  if (boundary === "minimum") {
    const index = steps.findIndex(step => step >= requested_step);
    return Number(coordinates[index < 0 ? coordinates.length - 1 : index]) - 0.5;
  }
  for (let index = steps.length - 1; index >= 0; index -= 1) {
    if (steps[index] <= requested_step) return Number(coordinates[index]) + 0.5;
  }
  return Number(coordinates[0]) + 0.5;
}

function apply_chart_axis_settings(prepared, chart_name, settings) {
  for (const axis_name of ["x", "y"]) {
    const minimum_field = `${axis_name}_min`;
    const maximum_field = `${axis_name}_max`;
    const has_minimum = Number.isFinite(settings[minimum_field]);
    const has_maximum = Number.isFinite(settings[maximum_field]);
    if (!has_minimum && !has_maximum) continue;
    const automatic = automatic_axis_bounds(prepared, axis_name);
    let minimum = has_minimum ? settings[minimum_field] : automatic[0];
    let maximum = has_maximum ? settings[maximum_field] : automatic[1];
    if (chart_name === "heatmap" && axis_name === "y") {
      if (has_minimum) minimum = heatmap_step_coordinate(prepared, minimum, "minimum");
      if (has_maximum) maximum = heatmap_step_coordinate(prepared, maximum, "maximum");
    } else if (chart_name === "heatmap" && axis_name === "x") {
      if (has_minimum) minimum -= 0.5;
      if (has_maximum) maximum += 0.5;
    }
    if (!(minimum < maximum)) {
      const padding = Math.max(1, Math.abs(has_minimum ? minimum : maximum) * 0.05);
      if (has_minimum && !has_maximum) maximum = minimum + padding;
      else if (!has_minimum && has_maximum) minimum = maximum - padding;
    }
    const axis = prepared.layout[`${axis_name}axis`] || {};
    axis.autorange = false;
    axis.range = [minimum, maximum];
    prepared.layout[`${axis_name}axis`] = axis;
  }
}

function trace_optimizer_update(trace) {
  const description = `${trace?.name || ""} ${trace?.hovertemplate || ""}`;
  const match = description.match(/(?:^|[^A-Za-z0-9])U(\d+)(?:\D|$)/);
  return match ? Number(match[1]) : null;
}

function available_snapshot_updates(figure) {
  const updates = [];
  const seen = new Set();
  for (const trace of figure?.data || []) {
    const update = trace_optimizer_update(trace);
    if (!Number.isFinite(update) || seen.has(update)) continue;
    seen.add(update);
    updates.push(update);
  }
  return updates;
}

function limit_curve_snapshots(prepared, maximum) {
  if (!Number.isInteger(maximum) || maximum < 1) return;
  const updates = available_snapshot_updates(prepared);
  if (updates.length <= maximum) return;
  const retained = new Set(updates.slice(-maximum));
  prepared.data = (prepared.data || []).filter(trace => {
    const update = trace_optimizer_update(trace);
    return !Number.isFinite(update) || retained.has(update);
  });
}

function smoothed_values(values, smoothing) {
  if (!(smoothing > 0)) return values;
  const output = [];
  let previous = null;
  for (const raw_value of values || []) {
    const value = Number(raw_value);
    if (!Number.isFinite(value)) {
      output.push(raw_value);
      continue;
    }
    previous = previous === null ? value : smoothing * previous + (1 - smoothing) * value;
    output.push(previous);
  }
  return output;
}

function numeric_quantile(sorted_values, fraction) {
  if (!sorted_values.length) return null;
  const position = Math.min(sorted_values.length - 1, Math.max(0, fraction * (sorted_values.length - 1)));
  const left = Math.floor(position);
  const right = Math.ceil(position);
  if (left === right) return sorted_values[left];
  const weight = position - left;
  return sorted_values[left] * (1 - weight) + sorted_values[right] * weight;
}

function apply_outlier_resistant_y_range(prepared) {
  const values = [];
  for (const trace of prepared.data || []) {
    for (const raw_value of trace.y || []) {
      const value = Number(raw_value);
      if (Number.isFinite(value)) values.push(value);
    }
  }
  if (values.length < 20) return;
  values.sort((left, right) => left - right);
  const minimum = numeric_quantile(values, 0.01);
  const maximum = numeric_quantile(values, 0.99);
  if (!Number.isFinite(minimum) || !Number.isFinite(maximum) || !(minimum < maximum)) return;
  const padding = Math.max((maximum - minimum) * 0.06, Math.abs(maximum) * 1e-6, 1e-12);
  prepared.layout.yaxis = {...(prepared.layout.yaxis || {}), autorange: false, range: [minimum - padding, maximum + padding]};
}

function apply_chart_display_settings(prepared, chart_name, settings) {
  prepared.layout.xaxis = {...(prepared.layout.xaxis || {})};
  prepared.layout.yaxis = {...(prepared.layout.yaxis || {})};
  const x_title = prepared.layout.xaxis.title && typeof prepared.layout.xaxis.title === "object" ? prepared.layout.xaxis.title : {};
  const y_title = prepared.layout.yaxis.title && typeof prepared.layout.yaxis.title === "object" ? prepared.layout.yaxis.title : {};
  prepared.layout.xaxis.title = {...x_title, text: settings.x_label};
  prepared.layout.yaxis.title = {...y_title, text: settings.y_label};
  prepared.layout.xaxis.showgrid = settings.show_grid;
  prepared.layout.yaxis.showgrid = settings.show_grid;
  prepared.layout.xaxis.gridcolor = "#e7e9ed";
  prepared.layout.yaxis.gridcolor = "#e7e9ed";
  if (chart_name === "heatmap") return;
  limit_curve_snapshots(prepared, settings.max_snapshots);
  for (const trace of prepared.data || []) {
    if (!trace || !["scatter", "scattergl", undefined].includes(trace.type)) continue;
    const original_mode = String(trace.mode || "lines");
    if (!original_mode.includes("lines")) continue;
    trace.y = smoothed_values(trace.y || [], settings.smoothing);
    trace.mode = settings.chart_type === "lines_markers" ? "lines+markers" : settings.chart_type;
    trace.line = {...(trace.line || {}), width: Math.max(0.5, Number(trace.line?.width || 1) * settings.line_width)};
    if (settings.chart_type !== "lines") {
      trace.marker = {...(trace.marker || {}), size: Math.max(3, Number(trace.marker?.size || 4))};
    }
  }
  if (settings.exclude_outliers && !Number.isFinite(settings.y_min) && !Number.isFinite(settings.y_max)) {
    apply_outlier_resistant_y_range(prepared);
  }
}

function prepare_figure(figure, chart_name) {
  const prepared = clone_figure(figure);
  const override = app.chart_settings_render_override;
  const settings = normalize_chart_settings(
    chart_name,
    override?.chart_name === chart_name ? override.settings : null,
  );
  const axis_settings = chart_axis_settings(chart_name, settings);
  prepared.layout = prepared.layout || {};
  prepared.layout.uirevision = `${app.current_run_id}-${chart_name}-${JSON.stringify(settings)}`;
  prepared.layout.autosize = true;
  prepared.layout.paper_bgcolor = "white";
  prepared.layout.plot_bgcolor = "white";
  prepared.layout.hovermode = "closest";
  prepared.layout.colorway = [colour_for_run(app.current_run_id)];
  delete prepared.layout.width;
  delete prepared.layout.height;
  delete prepared.layout.title;
  prepared.layout.margin = chart_name === "heatmap"
    ? {l: 67, r: 82, t: 18, b: 104}
    : {l: 61, r: 18, t: 23, b: 50};

  if (chart_name === "heatmap") {
    transpose_heatmap(prepared);
  } else {
    const colour = colour_for_run(app.current_run_id);
    for (const trace of prepared.data || []) {
      if (trace.type && trace.type !== "scatter" && trace.type !== "scattergl") continue;
      trace.line = {...(trace.line || {}), color: colour};
      if (trace.marker) {
        trace.marker.color = colour;
        trace.marker.line = {...(trace.marker.line || {}), color: colour};
      }
    }
  }
  apply_chart_display_settings(prepared, chart_name, settings);
  apply_chart_axis_settings(prepared, chart_name, axis_settings);
  return prepared;
}

async function render_plot(mount, figure, chart_name) {
  const prepared = prepare_figure(figure, chart_name);
  if (mount.dataset.plotReady === "true") {
    await Plotly.react(mount, prepared.data, prepared.layout, plot_config);
  } else {
    mount.replaceChildren();
    await Plotly.newPlot(mount, prepared.data, prepared.layout, plot_config);
    mount.dataset.plotReady = "true";
  }
}

function add_panel_resizers(article) {
  for (const direction of ["east", "south", "both"]) {
    const handle = document.createElement("div");
    handle.className = `panel-resizer panel-resizer-${direction === "both" ? "corner" : direction}`;
    handle.dataset.resize = direction;
    handle.title = direction === "east" ? "Drag to resize chart width" : direction === "south" ? "Drag to resize chart height" : "Drag to resize chart";
    article.appendChild(handle);
  }
}

function chart_settings_button(chart_name, title) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "chart-settings-button";
  button.dataset.chartSettings = chart_name;
  button.textContent = "⚙";
  button.title = "Chart settings";
  button.setAttribute("aria-label", `Settings for ${title}`);
  button.classList.toggle("active", has_chart_axis_settings(chart_name));
  return button;
}

function depth_card(chart_name) {
  const article = document.createElement("article");
  article.className = "chart-card";
  article.dataset.chart = chart_name;
  const header = document.createElement("header");
  header.className = "chart-card-header";
  const copy = document.createElement("div");
  copy.className = "chart-heading-copy";
  const title = document.createElement("h2");
  title.textContent = normalize_chart_settings(chart_name).title;
  const detail = document.createElement("p");
  detail.id = `${chart_name}_detail`;
  detail.textContent = "Waiting for the first DEPTH weight snapshot.";
  copy.append(title, detail);
  const actions = document.createElement("div");
  actions.className = "chart-card-actions";
  const maximize_button = document.createElement("button");
  maximize_button.type = "button";
  maximize_button.className = "maximize-button";
  maximize_button.dataset.maximize = chart_name;
  maximize_button.textContent = "⛶";
  maximize_button.title = "Maximize chart";
  maximize_button.setAttribute("aria-label", `Maximize ${title.textContent}`);
  actions.append(chart_settings_button(chart_name, title.textContent), maximize_button);
  header.append(copy, actions);
  const shell = document.createElement("div");
  shell.className = "plot-shell";
  const placeholder = document.createElement("div");
  placeholder.className = "plot-placeholder";
  placeholder.id = `${chart_name}_placeholder`;
  placeholder.textContent = "Waiting for the first DEPTH weight snapshot.";
  const mount = document.createElement("div");
  mount.className = "plot-mount";
  mount.id = `${chart_name}_plot`;
  shell.append(placeholder, mount);
  article.append(header, shell);
  add_panel_resizers(article);
  return article;
}

function ensure_depth_cards() {
  const grid = by_id("chart_grid");
  for (const chart_name of Object.keys(chart_titles).filter(name => name !== "heatmap")) {
    if (!by_id(`${chart_name}_plot`)) grid.appendChild(depth_card(chart_name));
  }
  by_id("depth_group_count").textContent = String(chart_groups.depth.length);
  apply_saved_panel_sizes();
}

function reset_run_charts() {
  for (const chart_name of Object.keys(chart_titles)) {
    const mount = by_id(`${chart_name}_plot`);
    if (mount) clear_plot(mount);
    const placeholder = by_id(`${chart_name}_placeholder`);
    if (placeholder) placeholder.hidden = false;
  }
  by_id("heatmap_card_detail").textContent = "Waiting for the first layer-count probe.";
  for (const chart_name of Object.keys(chart_titles).filter(name => name !== "heatmap")) {
    const detail = by_id(`${chart_name}_detail`);
    if (detail) detail.textContent = "Waiting for the first DEPTH weight snapshot.";
  }
}

async function render_figures() {
  if (!app.figures || !app.current_run_id) return;
  if (app.figures.heatmap) {
    by_id("heatmap_placeholder").hidden = true;
    await render_plot(by_id("heatmap_plot"), app.figures.heatmap, "heatmap");
  }
  const status = current_run();
  by_id("heatmap_card_detail").textContent = status
    ? `${format_integer(status.heatmap_count)} probes · latest step ${format_integer(status.heatmap_maximum_update)} · discrete cells`
    : "Layer-count probes";
  for (const chart_name of Object.keys(chart_titles).filter(name => name !== "heatmap")) {
    const figure = app.figures.depth[chart_name];
    const detail = by_id(`${chart_name}_detail`);
    if (!figure) continue;
    by_id(`${chart_name}_placeholder`).hidden = true;
    if (detail) detail.textContent = `${format_integer(status?.depth_snapshot_count)} retained snapshots · latest step ${format_integer(status?.depth_maximum_update)}`;
    await render_plot(by_id(`${chart_name}_plot`), figure, chart_name);
  }
}

function render_run_heading() {
  const run = current_run();
  if (!run) {
    by_id("run_title").textContent = "Select a run";
    by_id("selected_run_mark").style.background = "#c7cbd1";
    by_id("run_subtitle").textContent = "The newest active W&B-linked run will be selected automatically.";
    by_id("wandb_link").hidden = true;
    return;
  }
  by_id("run_title").textContent = run.artifact_name;
  by_id("selected_run_mark").style.background = colour_for_run(app.current_run_id);
  by_id("breadcrumb_leaf").textContent = run.wandb_run_id || run.local_run_id;
  const subtitle = by_id("run_subtitle");
  subtitle.replaceChildren();
  const values = [
    {text: run.wandb_run_id ? `W&B ID ${run.wandb_run_id}` : `Local ID ${run.local_run_id}`, class_name: "identity"},
    {text: display_run_state(run)},
    {text: run.host_label ? `host ${run.host_label}` : ""},
    {text: `${format_integer(run.heatmap_count)} probes`},
    {text: `${format_integer(run.depth_snapshot_count)} curves`},
    {text: `latest step ${format_integer(run.maximum_update)}`},
    {text: format_bytes(run.database_bytes)},
  ].filter(value => value.text);
  if (run.is_legacy_layout) values.push({text: "legacy unlinked store", class_name: "stale-warning"});
  for (const value of values) {
    const span = document.createElement("span");
    span.textContent = value.text;
    if (value.class_name) span.className = value.class_name;
    subtitle.appendChild(span);
  }
  const link = by_id("wandb_link");
  link.hidden = !run.wandb_url;
  if (run.wandb_url) link.href = run.wandb_url;
}

function render_empty_state() {
  const has_run = Boolean(app.current_run_id);
  by_id("charts_empty").hidden = has_run;
  by_id("charts_scroll").hidden = !has_run;
  if (!has_run) {
    by_id("charts_empty_title").textContent = app.runs.length ? "Select a run" : "Waiting for a local run";
    by_id("charts_empty_detail").textContent = app.runs.length
      ? "Choose a run from the persistent panel on the left."
      : "This page can stay open before training starts; the active W&B-linked run will appear automatically.";
  }
}

function update_file_source_tabs() {
  document.querySelectorAll(".file-source-tab").forEach(button => {
    const selected = button.dataset.fileSource === app.file_source;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-selected", String(selected));
  });
}

function set_file_source(source) {
  if (!["instra", "wandb"].includes(source)) return;
  app.file_source = source;
  app.file_path = "";
  app.file_payload = null;
  localStorage.setItem("instra_file_source", source);
  update_file_source_tabs();
  render_files();
  refresh_files();
}

function set_file_path(path) {
  app.file_path = path || "";
  app.file_payload = null;
  render_files();
  refresh_files();
}

function file_modified_text(value) {
  if (!value) return "—";
  const timestamp = new Date(value);
  if (!Number.isFinite(timestamp.getTime())) return "—";
  const delta_seconds = Math.max(0, Math.round((Date.now() - timestamp.getTime()) / 1000));
  if (delta_seconds < 60) return "just now";
  if (delta_seconds < 3600) return `${Math.floor(delta_seconds / 60)}m ago`;
  if (delta_seconds < 86400) return `${Math.floor(delta_seconds / 3600)}h ago`;
  if (delta_seconds < 86400 * 30) return `${Math.floor(delta_seconds / 86400)}d ago`;
  return timestamp.toLocaleDateString();
}

function folder_summary(entry) {
  const folder_count = Number(entry.folder_count);
  const file_count = Number(entry.file_count ?? entry.child_count);
  if (!Number.isFinite(folder_count) && !Number.isFinite(file_count)) return "Folder";
  const parts = [];
  if (Number.isFinite(folder_count)) {
    parts.push(`${format_integer(folder_count)} subfolder${folder_count === 1 ? "" : "s"}`);
  }
  if (Number.isFinite(file_count)) {
    parts.push(`${format_integer(file_count)} file${file_count === 1 ? "" : "s"}`);
  }
  return parts.join(", ");
}

function file_breadcrumbs(payload) {
  const container = by_id("file_breadcrumbs");
  container.replaceChildren();
  const root_button = document.createElement("button");
  root_button.type = "button";
  root_button.dataset.filePath = "";
  root_button.textContent = "root";
  root_button.title = payload?.root_path || payload?.wandb_files_url || "File root";
  container.appendChild(root_button);
  const parts = String(payload?.current_path || "").split("/").filter(Boolean);
  let assembled = "";
  for (const part of parts) {
    const separator = document.createElement("span");
    separator.textContent = "›";
    separator.setAttribute("aria-hidden", "true");
    assembled = assembled ? `${assembled}/${part}` : part;
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.filePath = assembled;
    button.textContent = part;
    button.title = assembled;
    container.append(separator, button);
  }
}

function local_file_url(path, download = false) {
  const query = new URLSearchParams({run: app.current_run_id, path});
  if (download) query.set("download", "1");
  return `/api/local-file?${query}`;
}

function append_file_row(body, entry) {
  const row = document.createElement("tr");
  row.className = `file-row file-${entry.kind}`;
  const name_cell = document.createElement("td");
  const name_button = document.createElement("button");
  name_button.type = "button";
  name_button.className = "file-name-button";
  const glyph = document.createElement("span");
  glyph.className = `file-glyph ${entry.kind}`;
  glyph.setAttribute("aria-hidden", "true");
  const name = document.createElement("strong");
  name.textContent = entry.kind === "folder" ? `${entry.name} /` : entry.name;
  name.title = entry.path;
  name_button.append(glyph, name);
  if (entry.kind === "folder") {
    name_button.addEventListener("click", () => set_file_path(entry.path));
  } else if (entry.kind === "file" && app.file_source === "instra") {
    name_button.addEventListener("click", () => window.open(local_file_url(entry.path), "_blank", "noopener"));
  } else if (entry.kind === "file" && entry.download_url) {
    name_button.addEventListener("click", () => window.open(entry.download_url, "_blank", "noopener"));
  } else {
    name_button.disabled = true;
  }
  name_cell.appendChild(name_button);
  row.appendChild(name_cell);
  row.appendChild(text_cell(entry.kind === "folder" ? folder_summary(entry) : file_modified_text(entry.modified_at), "file-modified-column"));
  row.appendChild(text_cell(entry.kind === "folder" ? "0 B" : (entry.size === null || entry.size === undefined ? "—" : format_bytes(entry.size)), "file-size-column"));
  const actions = document.createElement("td");
  actions.className = "file-actions-column";
  if (entry.kind === "file" && app.file_source === "instra") {
    const download = document.createElement("a");
    download.className = "file-row-action";
    download.href = local_file_url(entry.path, true);
    download.textContent = "⇩";
    download.title = "Download file";
    download.setAttribute("aria-label", `Download ${entry.name}`);
    actions.appendChild(download);
  } else if (entry.kind === "file" && entry.download_url) {
    const download = document.createElement("a");
    download.className = "file-row-action";
    download.href = entry.download_url;
    download.target = "_blank";
    download.rel = "noopener noreferrer";
    download.textContent = "⇩";
    download.title = "Download from W&B";
    download.setAttribute("aria-label", `Download ${entry.name} from W&B`);
    actions.appendChild(download);
  }
  row.appendChild(actions);
  body.appendChild(row);
}

function render_files() {
  update_file_source_tabs();
  const body = by_id("files_body");
  body.replaceChildren();
  const empty = by_id("files_empty");
  const payload = app.file_payload;
  const run = current_run();
  const wandb_link = by_id("file_wandb_link");
  const fallback_wandb_url = run?.wandb_url ? `${String(run.wandb_url).replace(/\/$/, "")}/files` : "";
  const wandb_files_url = payload?.wandb_files_url || fallback_wandb_url;
  wandb_link.hidden = app.file_source !== "wandb" || !wandb_files_url;
  if (wandb_files_url) wandb_link.href = wandb_files_url;
  const source_label = app.file_source === "instra" ? "Local run directory" : "W&B-managed files";
  const refresh_button = by_id("refresh_files");
  refresh_button.disabled = app.file_loading;
  refresh_button.classList.toggle("loading", app.file_loading);
  file_breadcrumbs(payload);
  const query = by_id("file_search").value.trim().toLowerCase();
  const all_entries = payload?.entries || [];
  const entries = all_entries.filter(entry => !query || `${entry.name} ${entry.path} ${entry.mime_type || ""}`.toLowerCase().includes(query));
  const item_count = query
    ? `${format_integer(entries.length)} of ${format_integer(all_entries.length)} items`
    : `${format_integer(all_entries.length)} item${all_entries.length === 1 ? "" : "s"}`;
  by_id("file_source_status").textContent = app.file_loading ? `Loading · ${source_label}` : `${item_count} · ${source_label}`;
  if (!app.current_run_id) {
    empty.hidden = false;
    by_id("files_empty_title").textContent = "Select a run";
    by_id("files_empty_detail").textContent = "Choose a run from the left to inspect its files.";
    return;
  }
  if (app.file_loading) {
    empty.hidden = false;
    by_id("files_empty_title").textContent = app.file_source === "instra" ? "Reading local folder" : "Loading W&B files";
    by_id("files_empty_detail").textContent = "The selected run remains available while the manifest is loaded.";
    return;
  }
  if (payload && payload.available === false) {
    empty.hidden = false;
    by_id("files_empty_title").textContent = app.file_source === "wandb" ? "W&B files unavailable" : "Folder unavailable";
    by_id("files_empty_detail").textContent = payload.error || "This file source is unavailable for the selected run.";
    return;
  }
  if (!payload) {
    empty.hidden = false;
    by_id("files_empty_title").textContent = "Ready to browse";
    by_id("files_empty_detail").textContent = "Refresh to load this file source.";
    return;
  }
  empty.hidden = entries.length > 0;
  if (!entries.length) {
    by_id("files_empty_title").textContent = query ? "No matching files" : "This folder is empty";
    by_id("files_empty_detail").textContent = query ? "Clear the folder filter to see all items." : "There are no files at this path.";
    return;
  }
  for (const entry of entries) append_file_row(body, entry);
}

async function refresh_files(force = false) {
  if (!app.current_run_id) {
    render_files();
    return;
  }
  const request_serial = ++app.file_request_serial;
  const requested_run_id = app.current_run_id;
  const requested_source = app.file_source;
  app.file_loading = true;
  render_files();
  const endpoint = requested_source === "instra" ? "/api/local-files" : "/api/wandb-files";
  const query = new URLSearchParams({run: requested_run_id, path: app.file_path});
  if (force && requested_source === "wandb") query.set("refresh", "1");
  try {
    const payload = await fetch_json(`${endpoint}?${query}`);
    if (
      request_serial !== app.file_request_serial
      || requested_run_id !== app.current_run_id
      || requested_source !== app.file_source
    ) return;
    app.file_payload = payload;
    app.file_path = payload.current_path || "";
  } catch (error) {
    if (request_serial !== app.file_request_serial) return;
    app.file_payload = {
      source: requested_source,
      available: false,
      error: error.message,
      entries: [],
      current_path: app.file_path,
    };
  } finally {
    if (request_serial === app.file_request_serial) {
      app.file_loading = false;
      render_files();
    }
  }
}

async function refresh_current_run() {
  if (!app.current_run_id || app.refresh_in_flight) return;
  app.refresh_in_flight = true;
  const requested_id = app.current_run_id;
  try {
    const encoded = encodeURIComponent(requested_id);
    const status = await fetch_json(`/api/status?run=${encoded}`);
    if (requested_id !== app.current_run_id) return;
    app.current_status = status;
    render_run_heading();
    const revision = JSON.stringify(status.revision);
    if (revision === app.figure_revision) return;
    app.figure_revision = revision;
    const figures = await fetch_json(`/api/figures?run=${encoded}`);
    if (requested_id !== app.current_run_id) return;
    app.figures = figures;
    await render_figures();
  } catch (error) {
    show_toast(`Run refresh failed: ${error.message}`);
  } finally {
    app.refresh_in_flight = false;
  }
}

function select_run(run_id, options = {}) {
  const manual = options.manual === true;
  app.manual_selection = manual;
  if (app.current_run_id !== run_id) {
    restore_maximized_chart();
    app.file_request_serial += 1;
    app.current_run_id = run_id;
    app.current_status = app.runs.find(run => run_identifier(run) === run_id) || null;
    app.figures = null;
    app.figure_revision = null;
    app.file_path = "";
    app.file_payload = null;
    app.file_loading = false;
    by_id("file_search").value = "";
    reset_run_charts();
  }
  const route = `/runs/${encodeURIComponent(run_id)}`;
  if (window.location.pathname !== route) {
    if (options.replace_history) history.replaceState({}, "", route);
    else history.pushState({}, "", route);
  }
  render_runs();
  render_run_heading();
  render_empty_state();
  refresh_current_run();
}

function resize_plot_in_card(card) {
  const mount = card.querySelector(".plot-mount");
  if (mount?.dataset.plotReady === "true") Plotly.Plots.resize(mount);
}

function resize_visible_plots() {
  document.querySelectorAll(".chart-card").forEach(card => {
    if (card.offsetParent !== null) resize_plot_in_card(card);
  });
}

function apply_saved_panel_sizes() {
  document.querySelectorAll(".chart-card").forEach(card => {
    const size = app.panel_sizes[card.dataset.chart];
    if (!size) return;
    if (Number(size.width) > 0) card.style.flex = `0 0 ${Number(size.width)}px`;
    if (Number(size.height) > 0) card.style.height = `${Number(size.height)}px`;
  });
}

function migrate_panel_layout() {
  if (localStorage.getItem("thog2_local_panel_layout_version") === panel_layout_version) return;
  app.panel_sizes = {};
  save_json("thog2_local_panel_sizes", app.panel_sizes);
  localStorage.setItem("thog2_local_panel_layout_version", panel_layout_version);
}

function reset_panel_sizes() {
  app.panel_sizes = {};
  save_json("thog2_local_panel_sizes", app.panel_sizes);
  restore_maximized_chart();
  document.querySelectorAll(".chart-card").forEach(card => {
    card.style.removeProperty("flex");
    card.style.removeProperty("height");
  });
  requestAnimationFrame(resize_visible_plots);
}

function start_chart_resize(event, handle) {
  const card = handle.closest(".chart-card");
  if (!card || app.maximized_chart) return;
  event.preventDefault();
  event.stopPropagation();
  const direction = handle.dataset.resize;
  const start_rect = card.getBoundingClientRect();
  const start_x = event.clientX;
  const start_y = event.clientY;
  const pointer_id = event.pointerId;
  handle.setPointerCapture(pointer_id);
  document.body.classList.add("resizing-chart");

  const move = pointer_event => {
    const pane_width = Math.max(260, by_id("charts_scroll").clientWidth - 20);
    const pane_height = Math.max(260, by_id("charts_scroll").clientHeight - 20);
    const minimum_width = card.dataset.chart === "heatmap" ? 300 : 180;
    const width = Math.max(Math.min(minimum_width, pane_width), Math.min(pane_width, start_rect.width + pointer_event.clientX - start_x));
    const height = Math.max(270, Math.min(Math.max(900, pane_height), start_rect.height + pointer_event.clientY - start_y));
    if (direction === "east" || direction === "both") card.style.flex = `0 0 ${Math.round(width)}px`;
    if (direction === "south" || direction === "both") card.style.height = `${Math.round(height)}px`;
    resize_plot_in_card(card);
  };
  const finish = () => {
    handle.removeEventListener("pointermove", move);
    handle.removeEventListener("pointerup", finish);
    handle.removeEventListener("pointercancel", finish);
    document.body.classList.remove("resizing-chart");
    const rect = card.getBoundingClientRect();
    app.panel_sizes[card.dataset.chart] = {width: Math.round(rect.width), height: Math.round(rect.height)};
    save_json("thog2_local_panel_sizes", app.panel_sizes);
    resize_plot_in_card(card);
  };
  handle.addEventListener("pointermove", move);
  handle.addEventListener("pointerup", finish);
  handle.addEventListener("pointercancel", finish);
}

function toggle_maximized_chart(chart_name) {
  if (app.maximized_chart === chart_name) {
    restore_maximized_chart();
    return;
  }
  const selected_card = document.querySelector(`.chart-card[data-chart="${chart_name}"]`);
  const grid = selected_card?.closest(".chart-grid");
  if (!selected_card || !grid) return;
  app.maximized_chart = chart_name;
  grid.classList.add("is-maximized");
  selected_card?.closest(".chart-group")?.classList.add("maximized");
  by_id("charts_scroll").classList.add("maximized-mode");
  document.querySelectorAll(".chart-card").forEach(card => {
    const selected = card.dataset.chart === chart_name;
    card.classList.toggle("maximized", selected);
    const button = card.querySelector(".maximize-button");
    button.textContent = selected ? "↙" : "⛶";
    button.title = selected ? "Restore chart grid" : "Maximize chart";
    button.setAttribute("aria-label", selected ? "Restore chart grid" : `Maximize ${chart_titles[card.dataset.chart]}`);
  });
  requestAnimationFrame(() => requestAnimationFrame(() => {
    position_restore_button();
    resize_visible_plots();
  }));
}

function position_restore_button() {
  const card = document.querySelector(".chart-card.maximized");
  const button = card?.querySelector(".maximize-button");
  const settings_button = card?.querySelector(".chart-settings-button");
  const header = card?.querySelector(".chart-card-header");
  const pane = by_id("charts_pane");
  if (!button || !header || !pane) return;
  const header_rect = header.getBoundingClientRect();
  const pane_rect = pane.getBoundingClientRect();
  if (header_rect.width <= 0 || header_rect.height <= 0 || pane_rect.width <= 0) return;
  button.style.top = `${Math.round(header_rect.top + Math.max(6, (header_rect.height - button.offsetHeight) / 2))}px`;
  button.style.right = `${Math.round(Math.max(12, window.innerWidth - pane_rect.right + 12))}px`;
  if (settings_button) {
    settings_button.style.top = button.style.top;
    settings_button.style.right = `${Math.round(Math.max(49, window.innerWidth - pane_rect.right + 49))}px`;
  }
}

function restore_maximized_chart() {
  if (!app.maximized_chart) return;
  app.maximized_chart = null;
  document.querySelectorAll(".chart-grid.is-maximized").forEach(grid => grid.classList.remove("is-maximized"));
  document.querySelectorAll(".chart-group.maximized").forEach(group => group.classList.remove("maximized"));
  by_id("charts_scroll").classList.remove("maximized-mode");
  document.querySelectorAll(".chart-card").forEach(card => {
    card.classList.remove("maximized");
    const button = card.querySelector(".maximize-button");
    button.textContent = "⛶";
    button.title = "Maximize chart";
    button.setAttribute("aria-label", `Maximize ${chart_titles[card.dataset.chart]}`);
    button.style.removeProperty("top");
    button.style.removeProperty("right");
    const settings_button = card.querySelector(".chart-settings-button");
    settings_button?.style.removeProperty("top");
    settings_button?.style.removeProperty("right");
  });
  requestAnimationFrame(() => requestAnimationFrame(resize_visible_plots));
}

function toggle_chart_group(button) {
  const group = button.closest(".chart-group");
  const grid = group?.querySelector(".chart-grid");
  if (!group || !grid) return;
  const collapsed = !group.classList.contains("collapsed");
  if (collapsed && app.maximized_chart) restore_maximized_chart();
  group.classList.toggle("collapsed", collapsed);
  grid.hidden = collapsed;
  button.setAttribute("aria-expanded", String(!collapsed));
  if (!collapsed) requestAnimationFrame(() => requestAnimationFrame(resize_visible_plots));
}

function set_runs_pane_width(width) {
  const maximum = Math.max(280, window.innerWidth - 58 - 330);
  const resolved = Math.max(280, Math.min(maximum, width));
  document.documentElement.style.setProperty("--runs-width", `${resolved}px`);
  localStorage.setItem("thog2_local_runs_width", String(resolved));
}

function toggle_runs_pane(force_open = false) {
  const workspace = by_id("workspace");
  const collapsed = force_open ? false : !workspace.classList.contains("runs-collapsed");
  workspace.classList.toggle("runs-collapsed", collapsed);
  by_id("toggle_runs").textContent = collapsed ? "›" : "‹";
  localStorage.setItem("thog2_local_runs_collapsed", collapsed ? "1" : "0");
  requestAnimationFrame(() => requestAnimationFrame(resize_visible_plots));
}

function start_runs_pane_resize(event) {
  if (event.target.closest("button")) return;
  event.preventDefault();
  if (by_id("workspace").classList.contains("runs-collapsed")) toggle_runs_pane(true);
  const divider = by_id("workspace_divider");
  divider.setPointerCapture(event.pointerId);
  divider.classList.add("dragging");
  document.body.classList.add("dragging");
  const move = pointer_event => {
    set_runs_pane_width(pointer_event.clientX - by_id("workspace").getBoundingClientRect().left);
    resize_visible_plots();
  };
  const finish = () => {
    divider.removeEventListener("pointermove", move);
    divider.removeEventListener("pointerup", finish);
    divider.removeEventListener("pointercancel", finish);
    divider.classList.remove("dragging");
    document.body.classList.remove("dragging");
    resize_visible_plots();
  };
  divider.addEventListener("pointermove", move);
  divider.addEventListener("pointerup", finish);
  divider.addEventListener("pointercancel", finish);
}

function update_sort_direction_ui() {
  const button = by_id("sort_direction");
  button.textContent = app.sort_descending ? "↓" : "↑";
  button.title = app.sort_descending ? "Descending; click for ascending" : "Ascending; click for descending";
  button.setAttribute("aria-pressed", String(app.sort_descending));
}

function run_for_id(run_id) {
  return app.runs.find(run => run_identifier(run) === run_id) || null;
}

function open_run_menu(run_id, anchor) {
  app.menu_run_id = run_id;
  document.querySelectorAll(".run-menu-button").forEach(button => button.setAttribute("aria-expanded", "false"));
  anchor.setAttribute("aria-expanded", "true");
  const menu = by_id("run_menu");
  menu.hidden = false;
  const anchor_rect = anchor.getBoundingClientRect();
  const menu_width = 204;
  const menu_height = 130;
  menu.style.left = `${Math.max(8, Math.min(anchor_rect.right - menu_width, window.innerWidth - menu_width - 8))}px`;
  menu.style.top = `${Math.max(8, Math.min(anchor_rect.bottom + 4, window.innerHeight - menu_height - 8))}px`;
}

function close_run_menu() {
  by_id("run_menu").hidden = true;
  app.menu_run_id = null;
  document.querySelectorAll(".run-menu-button").forEach(button => button.setAttribute("aria-expanded", "false"));
}

async function copy_text(value, description) {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(value);
    } else {
      const textarea = document.createElement("textarea");
      textarea.value = value;
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      if (!document.execCommand("copy")) throw new Error("clipboard copy was rejected");
      textarea.remove();
    }
    show_toast(`${description} copied.`);
  } catch (error) {
    show_toast(`Copy failed: ${error.message}`);
  }
}

async function delete_menu_run() {
  const run = run_for_id(app.menu_run_id);
  if (!run) return;
  const run_id = run_identifier(run);
  const state = display_run_state(run);
  const active_warning = state === "running"
    ? "\n\nThis run still appears to be running. Its training process may recreate or continue writing the database."
    : "";
  const confirmed = window.confirm(
    `Delete instra chart data for:\n\n${run.artifact_name}\n\nPath: ${run.run_directory}${active_warning}\n\nThis does not delete checkpoints, other logs, or the W&B run.`
  );
  if (!confirmed) return;
  close_run_menu();
  try {
    await fetch_json(`/api/run?run=${encodeURIComponent(run_id)}`, {method: "DELETE"});
    delete app.colours[run_id];
    delete app.visibility[run_id];
    save_json("thog2_local_run_colours", app.colours);
    save_json("thog2_local_run_visibility", app.visibility);
    if (app.current_run_id === run_id) {
      restore_maximized_chart();
      app.current_run_id = null;
      app.current_status = null;
      app.figures = null;
      app.figure_revision = null;
      app.manual_selection = false;
      reset_run_charts();
      history.replaceState({}, "", "/runs");
    }
    show_toast("instra chart data deleted.");
    await refresh_catalog();
  } catch (error) {
    show_toast(`Delete failed: ${error.message}`);
  }
}

function update_chart_settings_buttons() {
  document.querySelectorAll(".chart-settings-button").forEach(button => {
    const chart_name = button.dataset.chartSettings;
    button.classList.toggle("active", has_chart_axis_settings(chart_name));
    const heading = button.closest(".chart-card")?.querySelector(".chart-heading-copy h2");
    const title = normalize_chart_settings(chart_name).title;
    if (heading) heading.textContent = title;
    button.setAttribute("aria-label", `Settings for ${title}`);
  });
}

function optional_chart_number(input_id) {
  const raw_value = by_id(input_id).value.trim();
  if (!raw_value) return null;
  const value = Number(raw_value);
  return Number.isFinite(value) ? value : NaN;
}

function chart_settings_form_state() {
  const chart_name = app.axis_chart_name;
  if (!chart_name) return {settings: null, error: "No chart is selected."};
  const labels = chart_default_labels(chart_name);
  const settings = {
    title: by_id("chart_title_value").value.trim() || chart_titles[chart_name],
    x_label: by_id("chart_x_label").value.trim() || labels.x_label,
    y_label: by_id("chart_y_label").value.trim() || labels.y_label,
    max_snapshots: chart_name === "heatmap" ? 0 : Number(by_id("chart_max_snapshots").value),
    exclude_outliers: chart_name !== "heatmap" && by_id("chart_exclude_outliers").checked,
    smoothing: chart_name === "heatmap" ? 0 : Number(by_id("chart_smoothing").value),
    line_width: chart_name === "heatmap" ? 1 : Number(by_id("chart_line_width").value),
    chart_type: chart_name === "heatmap"
      ? "lines"
      : (document.querySelector('input[name="chart_type"]:checked')?.value || "lines"),
    show_grid: by_id("chart_show_grid").checked,
    heatmap_row_height: chart_name === "heatmap" ? Number(by_id("chart_heatmap_row_height").value) : 12,
  };
  for (const field_name of ["x_min", "x_max", "y_min", "y_max"]) {
    const value = optional_chart_number(`chart_${field_name}`);
    if (Number.isNaN(value)) return {settings: null, error: "Axis limits must be finite numbers or blank for automatic."};
    if (value !== null) settings[field_name] = value;
  }
  if (Number.isFinite(settings.x_min) && Number.isFinite(settings.x_max) && settings.x_min >= settings.x_max) {
    return {settings: null, error: "X minimum must be less than X maximum."};
  }
  if (Number.isFinite(settings.y_min) && Number.isFinite(settings.y_max) && settings.y_min >= settings.y_max) {
    return {settings: null, error: "Y minimum must be less than Y maximum."};
  }
  if (!Number.isInteger(settings.max_snapshots) || settings.max_snapshots < 0) {
    return {settings: null, error: "Maximum snapshots must be zero for all, or a positive whole number."};
  }
  return {settings: normalize_chart_settings(chart_name, settings), error: ""};
}

function compact_chart_settings(chart_name, settings) {
  const labels = chart_default_labels(chart_name);
  const compact = {};
  for (const field_name of ["x_min", "x_max", "y_min", "y_max"]) {
    if (Number.isFinite(settings[field_name])) compact[field_name] = settings[field_name];
  }
  if (settings.title !== chart_titles[chart_name]) compact.title = settings.title;
  if (settings.x_label !== labels.x_label) compact.x_label = settings.x_label;
  if (settings.y_label !== labels.y_label) compact.y_label = settings.y_label;
  if (chart_name !== "heatmap") {
    if (settings.max_snapshots > 0) compact.max_snapshots = settings.max_snapshots;
    if (settings.exclude_outliers) compact.exclude_outliers = true;
    if (settings.smoothing > 0) compact.smoothing = settings.smoothing;
    if (settings.line_width !== 1) compact.line_width = settings.line_width;
    if (settings.chart_type !== "lines") compact.chart_type = settings.chart_type;
  }
  if (!settings.show_grid) compact.show_grid = false;
  return compact;
}

function set_chart_settings_tab(tab_name) {
  if (!["data", "display"].includes(tab_name)) return;
  document.querySelectorAll("[data-chart-settings-tab]").forEach(button => {
    const selected = button.dataset.chartSettingsTab === tab_name;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-selected", String(selected));
  });
  document.querySelectorAll("[data-chart-settings-panel]").forEach(panel => {
    panel.hidden = panel.dataset.chartSettingsPanel !== tab_name;
  });
}

function sync_chart_setting_outputs() {
  const chart_name = app.axis_chart_name;
  const maximum = Number(by_id("chart_max_snapshots").max);
  const snapshots = Number(by_id("chart_max_snapshots").value);
  by_id("chart_max_snapshots_value").textContent = snapshots > 0 ? String(snapshots) : `All${maximum > 0 ? ` ${maximum}` : ""}`;
  const smoothing = Number(by_id("chart_smoothing").value);
  by_id("chart_smoothing_value").textContent = smoothing > 0 ? smoothing.toFixed(2) : "Off";
  by_id("chart_line_width_value").textContent = `${Number(by_id("chart_line_width").value).toFixed(2)}×`;
  by_id("chart_heatmap_row_height_value").textContent = `${Number(by_id("chart_heatmap_row_height").value)} px`;
  by_id("chart_curve_display_options").hidden = chart_name === "heatmap";
  by_id("chart_heatmap_display_options").hidden = chart_name !== "heatmap";
}

function populate_chart_settings_form(chart_name, supplied = null) {
  const settings = normalize_chart_settings(chart_name, supplied);
  const labels = chart_default_labels(chart_name);
  if (chart_name === "heatmap" && !Number.isFinite(Number(supplied?.heatmap_row_height))) {
    settings.heatmap_row_height = typeof heatmap_probe_row_height_px === "function" ? heatmap_probe_row_height_px() : 12;
  }
  by_id("chart_title_value").value = settings.title;
  by_id("chart_x_source").value = labels.x_source;
  by_id("chart_y_source").value = labels.y_source;
  by_id("chart_x_label").value = settings.x_label;
  by_id("chart_y_label").value = settings.y_label;
  for (const field_name of ["x_min", "x_max", "y_min", "y_max"]) {
    by_id(`chart_${field_name}`).value = Number.isFinite(settings[field_name]) ? String(settings[field_name]) : "";
  }
  const snapshot_count = chart_name === "heatmap" ? 0 : available_snapshot_updates(figure_for_chart(chart_name)).length;
  by_id("chart_max_snapshots").max = String(Math.max(1, snapshot_count));
  by_id("chart_max_snapshots").value = String(settings.max_snapshots > 0 ? Math.min(settings.max_snapshots, Math.max(1, snapshot_count)) : 0);
  by_id("chart_exclude_outliers").checked = settings.exclude_outliers;
  by_id("chart_smoothing").value = String(settings.smoothing);
  by_id("chart_line_width").value = String(settings.line_width);
  const chart_type = document.querySelector(`input[name="chart_type"][value="${settings.chart_type}"]`);
  if (chart_type) chart_type.checked = true;
  by_id("chart_show_grid").checked = settings.show_grid;
  by_id("chart_heatmap_row_height").value = String(settings.heatmap_row_height || 12);
  sync_chart_setting_outputs();
}

function reset_chart_settings_preview_mount() {
  const shell = by_id("chart_settings_preview_shell");
  const empty = by_id("chart_settings_preview_empty");
  const mount = by_id("chart_settings_preview");
  if (mount.dataset.plotReady === "true") Plotly.purge(mount);
  mount.dataset.plotReady = "false";
  mount.removeAttribute("style");
  shell.replaceChildren(empty, mount);
}

async function render_chart_settings_preview() {
  const chart_name = app.axis_chart_name;
  if (!chart_name || by_id("chart_settings_overlay").hidden) return;
  const state = chart_settings_form_state();
  const error = by_id("chart_settings_error");
  if (state.error) {
    error.textContent = state.error;
    error.hidden = false;
    by_id("chart_settings_preview_status").textContent = "Fix the highlighted values";
    return;
  }
  error.hidden = true;
  const figure = figure_for_chart(chart_name);
  const empty = by_id("chart_settings_preview_empty");
  empty.hidden = Boolean(figure);
  by_id("chart_settings_preview_heading").textContent = state.settings.title;
  by_id("chart_settings_preview_status").textContent = "Unsaved settings";
  if (!figure) return;
  const serial = ++app.chart_settings_preview_serial;
  app.chart_settings_render_override = {chart_name, settings: state.settings};
  try {
    await render_plot(by_id("chart_settings_preview"), figure, chart_name);
  } catch (preview_error) {
    if (serial === app.chart_settings_preview_serial) {
      error.textContent = `Preview failed: ${preview_error.message}`;
      error.hidden = false;
    }
  } finally {
    if (serial === app.chart_settings_preview_serial) app.chart_settings_render_override = null;
  }
}

function schedule_chart_settings_preview() {
  sync_chart_setting_outputs();
  clearTimeout(app.chart_settings_preview_timer);
  app.chart_settings_preview_timer = setTimeout(render_chart_settings_preview, 90);
}

function open_chart_settings(chart_name) {
  if (!chart_titles[chart_name]) return;
  close_run_menu();
  close_colour_picker();
  close_settings();
  app.axis_chart_name = chart_name;
  by_id("chart_settings_title").textContent = `${normalize_chart_settings(chart_name).title} settings`;
  by_id("chart_settings_axes").textContent = chart_name === "heatmap"
    ? "Candidate-layer and optimizer-step controls apply only to this heatmap."
    : "Curve history, axes and display changes apply only to this panel.";
  populate_chart_settings_form(chart_name);
  set_chart_settings_tab("data");
  by_id("chart_settings_error").hidden = true;
  by_id("chart_settings_overlay").hidden = false;
  reset_chart_settings_preview_mount();
  requestAnimationFrame(() => {
    render_chart_settings_preview();
    by_id("chart_title_value").focus();
  });
}

function close_chart_settings() {
  const was_open = !by_id("chart_settings_overlay").hidden;
  const chart_name = app.axis_chart_name;
  clearTimeout(app.chart_settings_preview_timer);
  app.chart_settings_preview_serial += 1;
  app.chart_settings_render_override = null;
  by_id("chart_settings_overlay").hidden = true;
  reset_chart_settings_preview_mount();
  app.axis_chart_name = null;
  if (was_open) render_axis_settings_change(chart_name);
}

function render_axis_settings_change(chart_name = null) {
  update_chart_settings_buttons();
  if (!app.figures) return;
  const figure = chart_name ? figure_for_chart(chart_name) : null;
  const mount = chart_name ? by_id(`${chart_name}_plot`) : null;
  if (figure && mount) {
    render_plot(mount, figure, chart_name).catch(error => show_toast(`Chart settings failed: ${error.message}`));
    return;
  }
  render_figures().catch(error => show_toast(`Chart settings failed: ${error.message}`));
}

function save_chart_settings() {
  const chart_name = app.axis_chart_name;
  if (!chart_name) return;
  const state = chart_settings_form_state();
  if (state.error) {
    by_id("chart_settings_error").textContent = state.error;
    by_id("chart_settings_error").hidden = false;
    return;
  }
  const compact = compact_chart_settings(chart_name, state.settings);
  if (Object.keys(compact).length) app.axis_ranges[chart_name] = compact;
  else delete app.axis_ranges[chart_name];
  save_json("thog2_local_chart_axis_ranges", app.axis_ranges);
  if (chart_name === "heatmap" && typeof save_heatmap_viewer_setting === "function") {
    save_heatmap_viewer_setting("probe_row_height_px", state.settings.heatmap_row_height);
  }
  close_chart_settings();
}

function reset_chart_settings() {
  const chart_name = app.axis_chart_name;
  if (!chart_name) return;
  populate_chart_settings_form(chart_name, chart_name === "heatmap" ? {heatmap_row_height: 12} : {});
  by_id("chart_settings_error").hidden = true;
  schedule_chart_settings_preview();
}

function open_settings() {
  close_run_menu();
  close_colour_picker();
  close_chart_settings();
  by_id("crash_timeout_minutes").value = String(app.crash_timeout_minutes);
  by_id("settings_overlay").hidden = false;
  by_id("settings_nav").classList.add("selected");
  by_id("runs_nav").classList.remove("selected");
  by_id("crash_timeout_minutes").focus();
}

function close_settings() {
  by_id("settings_overlay").hidden = true;
  by_id("settings_nav").classList.remove("selected");
  by_id("runs_nav").classList.add("selected");
}

function save_settings() {
  const value = Number(by_id("crash_timeout_minutes").value);
  if (!Number.isFinite(value) || value < 1 || value > 10080) {
    show_toast("Crash timeout must be between 1 and 10,080 minutes.");
    return;
  }
  app.crash_timeout_minutes = Math.round(value);
  localStorage.setItem("thog2_local_crash_timeout_minutes", String(app.crash_timeout_minutes));
  reset_pagination();
  render_run_heading();
  close_settings();
}

function hsv_to_rgb(hue, saturation, value) {
  const chroma = value * saturation;
  const component = chroma * (1 - Math.abs((hue / 60) % 2 - 1));
  const offset = value - chroma;
  let rgb = [0, 0, 0];
  if (hue < 60) rgb = [chroma, component, 0];
  else if (hue < 120) rgb = [component, chroma, 0];
  else if (hue < 180) rgb = [0, chroma, component];
  else if (hue < 240) rgb = [0, component, chroma];
  else if (hue < 300) rgb = [component, 0, chroma];
  else rgb = [chroma, 0, component];
  return rgb.map(channel => Math.round((channel + offset) * 255));
}

function rgb_to_hsv(red, green, blue) {
  const values = [red / 255, green / 255, blue / 255];
  const maximum = Math.max(...values);
  const minimum = Math.min(...values);
  const delta = maximum - minimum;
  let hue = 0;
  if (delta) {
    if (maximum === values[0]) hue = 60 * (((values[1] - values[2]) / delta) % 6);
    else if (maximum === values[1]) hue = 60 * ((values[2] - values[0]) / delta + 2);
    else hue = 60 * ((values[0] - values[1]) / delta + 4);
  }
  if (hue < 0) hue += 360;
  return [hue, maximum ? delta / maximum : 0, maximum];
}

function rgb_to_hex(rgb) { return `#${rgb.map(value => value.toString(16).padStart(2, "0")).join("")}`.toUpperCase(); }

function hex_to_rgb(hex) {
  const match = /^#?([0-9a-f]{6})$/i.exec(String(hex).trim());
  if (!match) return null;
  return [0, 2, 4].map(index => parseInt(match[1].slice(index, index + 2), 16));
}

function draw_colour_plane() {
  const canvas = by_id("colour_plane");
  const context = canvas.getContext("2d");
  const base = hsv_to_rgb(app.picker_hue, 1, 1);
  context.fillStyle = `rgb(${base.join(",")})`;
  context.fillRect(0, 0, canvas.width, canvas.height);
  const white = context.createLinearGradient(0, 0, canvas.width, 0);
  white.addColorStop(0, "rgba(255,255,255,1)");
  white.addColorStop(1, "rgba(255,255,255,0)");
  context.fillStyle = white;
  context.fillRect(0, 0, canvas.width, canvas.height);
  const black = context.createLinearGradient(0, 0, 0, canvas.height);
  black.addColorStop(0, "rgba(0,0,0,0)");
  black.addColorStop(1, "rgba(0,0,0,1)");
  context.fillStyle = black;
  context.fillRect(0, 0, canvas.width, canvas.height);
  const x = app.picker_saturation * canvas.width;
  const y = (1 - app.picker_value) * canvas.height;
  context.beginPath();
  context.arc(x, y, 5, 0, Math.PI * 2);
  context.strokeStyle = "white";
  context.lineWidth = 2;
  context.stroke();
  context.beginPath();
  context.arc(x, y, 6.5, 0, Math.PI * 2);
  context.strokeStyle = "rgba(0,0,0,.65)";
  context.lineWidth = 1;
  context.stroke();
}

function queue_current_recolour() {
  clearTimeout(queue_current_recolour.timer);
  queue_current_recolour.timer = setTimeout(() => {
    if (app.figures && app.current_run_id === app.colour_run_id) render_figures();
  }, 120);
}

function set_picker_colour(rgb, persist = true) {
  const [hue, saturation, value] = rgb_to_hsv(...rgb);
  app.picker_hue = hue;
  app.picker_saturation = saturation;
  app.picker_value = value;
  by_id("colour_hue").value = String(Math.round(hue));
  by_id("colour_hex").value = rgb_to_hex(rgb);
  by_id("colour_r").value = String(rgb[0]);
  by_id("colour_g").value = String(rgb[1]);
  by_id("colour_b").value = String(rgb[2]);
  draw_colour_plane();
  if (persist && app.colour_run_id) {
    const colour = rgb_to_hex(rgb);
    app.colours[app.colour_run_id] = colour;
    save_json("thog2_local_run_colours", app.colours);
    document.querySelectorAll(`[data-run-id="${CSS.escape(app.colour_run_id)}"]`).forEach(row => {
      row.style.setProperty("--run-colour", colour);
      const dot = row.querySelector(".colour-dot");
      if (dot) dot.style.background = colour;
    });
    if (app.current_run_id === app.colour_run_id) by_id("selected_run_mark").style.background = colour;
    queue_current_recolour();
  }
}

function open_colour_picker(run_id, anchor) {
  app.colour_run_id = run_id;
  set_picker_colour(hex_to_rgb(colour_for_run(run_id)), false);
  const popover = by_id("colour_popover");
  popover.hidden = false;
  const anchor_rect = anchor.getBoundingClientRect();
  const width = 282;
  const height = 390;
  popover.style.left = `${Math.max(8, Math.min(anchor_rect.left - 12, window.innerWidth - width - 8))}px`;
  popover.style.top = `${Math.max(8, Math.min(anchor_rect.bottom + 8, window.innerHeight - height - 8))}px`;
}

function close_colour_picker() {
  by_id("colour_popover").hidden = true;
  app.colour_run_id = null;
}

function build_swatches() {
  const container = by_id("colour_swatches");
  for (const colour of default_palette) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "colour-swatch";
    button.style.background = colour;
    button.title = colour;
    button.addEventListener("click", () => set_picker_colour(hex_to_rgb(colour)));
    container.appendChild(button);
  }
}

function show_toast(message) {
  const toast = by_id("toast");
  toast.textContent = message;
  toast.hidden = false;
  clearTimeout(show_toast.timer);
  show_toast.timer = setTimeout(() => { toast.hidden = true; }, 4500);
}

function bind_events() {
  document.querySelectorAll(".file-source-tab").forEach(button => {
    button.addEventListener("click", () => set_file_source(button.dataset.fileSource));
  });
  by_id("file_breadcrumbs").addEventListener("click", event => {
    const button = event.target.closest("[data-file-path]");
    if (button) set_file_path(button.dataset.filePath);
  });
  by_id("file_search").addEventListener("input", render_files);
  by_id("refresh_files").addEventListener("click", () => refresh_files(true));
  by_id("charts_scroll").addEventListener("click", event => {
    const group_button = event.target.closest(".chart-group-toggle");
    if (group_button) {
      toggle_chart_group(group_button);
      return;
    }
    const chart_settings = event.target.closest(".chart-settings-button");
    if (chart_settings) {
      open_chart_settings(chart_settings.dataset.chartSettings);
      return;
    }
    const maximize_button = event.target.closest(".maximize-button");
    if (maximize_button) toggle_maximized_chart(maximize_button.dataset.maximize);
  });
  by_id("charts_scroll").addEventListener("pointerdown", event => {
    const handle = event.target.closest(".panel-resizer");
    if (handle) start_chart_resize(event, handle);
  });
  by_id("charts_scroll").addEventListener("dblclick", event => {
    const handle = event.target.closest(".panel-resizer");
    if (!handle) return;
    const card = handle.closest(".chart-card");
    delete app.panel_sizes[card.dataset.chart];
    save_json("thog2_local_panel_sizes", app.panel_sizes);
    card.style.removeProperty("flex");
    card.style.removeProperty("height");
    requestAnimationFrame(() => resize_plot_in_card(card));
  });
  by_id("run_search").addEventListener("input", reset_pagination);
  by_id("state_filter").addEventListener("change", reset_pagination);
  by_id("run_sort").addEventListener("change", reset_pagination);
  by_id("sort_direction").addEventListener("click", () => {
    app.sort_descending = !app.sort_descending;
    localStorage.setItem("thog2_local_sort_descending", app.sort_descending ? "1" : "0");
    update_sort_direction_ui();
    reset_pagination();
  });
  by_id("group_button").addEventListener("click", () => {
    app.group_by_host = !app.group_by_host;
    by_id("group_button").setAttribute("aria-pressed", String(app.group_by_host));
    reset_pagination();
  });
  by_id("page_size").addEventListener("change", event => {
    app.page_size = Number(event.target.value);
    localStorage.setItem("thog2_local_page_size", String(app.page_size));
    reset_pagination();
  });
  by_id("previous_page").addEventListener("click", () => {
    app.current_page = Math.max(1, app.current_page - 1);
    render_runs();
  });
  by_id("next_page").addEventListener("click", () => {
    app.current_page += 1;
    render_runs();
  });
  by_id("select_all").addEventListener("change", event => {
    for (const run of filtered_runs()) {
      const run_id = run_identifier(run);
      if (event.target.checked) app.selected.add(run_id); else app.selected.delete(run_id);
    }
    render_runs();
  });
  by_id("runs_nav").addEventListener("click", () => toggle_runs_pane(true));
  by_id("settings_nav").addEventListener("click", open_settings);
  by_id("close_settings").addEventListener("click", close_settings);
  by_id("cancel_settings").addEventListener("click", close_settings);
  by_id("save_settings").addEventListener("click", save_settings);
  by_id("settings_overlay").addEventListener("pointerdown", event => {
    if (event.target === by_id("settings_overlay")) close_settings();
  });
  by_id("close_chart_settings").addEventListener("click", close_chart_settings);
  by_id("cancel_chart_settings").addEventListener("click", close_chart_settings);
  by_id("save_chart_settings").addEventListener("click", save_chart_settings);
  by_id("reset_chart_settings").addEventListener("click", reset_chart_settings);
  document.querySelectorAll("[data-chart-settings-tab]").forEach(button => {
    button.addEventListener("click", () => set_chart_settings_tab(button.dataset.chartSettingsTab));
  });
  document.querySelectorAll(".chart-setting-input").forEach(input => {
    input.addEventListener("input", schedule_chart_settings_preview);
    input.addEventListener("change", schedule_chart_settings_preview);
  });
  by_id("chart_settings_overlay").addEventListener("pointerdown", event => {
    if (event.target === by_id("chart_settings_overlay")) close_chart_settings();
  });
  by_id("copy_run_name").addEventListener("click", () => {
    const run = run_for_id(app.menu_run_id);
    if (!run) return;
    close_run_menu();
    copy_text(run.artifact_name, "Run name");
  });
  by_id("copy_run_path").addEventListener("click", () => {
    const run = run_for_id(app.menu_run_id);
    if (!run) return;
    close_run_menu();
    copy_text(run.run_directory, "Run path");
  });
  by_id("delete_run").addEventListener("click", delete_menu_run);
  by_id("toggle_runs").addEventListener("click", () => toggle_runs_pane());
  by_id("toggle_runs_top").addEventListener("click", () => toggle_runs_pane());
  by_id("workspace_divider").addEventListener("pointerdown", start_runs_pane_resize);
  by_id("workspace_divider").addEventListener("keydown", event => {
    if (!event.key.startsWith("Arrow")) return;
    event.preventDefault();
    const current = by_id("runs_pane").getBoundingClientRect().width;
    set_runs_pane_width(current + (event.key === "ArrowRight" ? 24 : -24));
    requestAnimationFrame(resize_visible_plots);
  });
  by_id("restore_panels").addEventListener("click", reset_panel_sizes);
  by_id("colour_hue").addEventListener("input", event => {
    app.picker_hue = Number(event.target.value);
    set_picker_colour(hsv_to_rgb(app.picker_hue, app.picker_saturation, app.picker_value));
  });
  by_id("colour_plane").addEventListener("pointerdown", event => {
    const canvas = by_id("colour_plane");
    const update = pointer_event => {
      const rect = canvas.getBoundingClientRect();
      app.picker_saturation = Math.max(0, Math.min(1, (pointer_event.clientX - rect.left) / rect.width));
      app.picker_value = 1 - Math.max(0, Math.min(1, (pointer_event.clientY - rect.top) / rect.height));
      set_picker_colour(hsv_to_rgb(app.picker_hue, app.picker_saturation, app.picker_value));
    };
    update(event);
    canvas.setPointerCapture(event.pointerId);
    canvas.onpointermove = update;
    canvas.onpointerup = () => { canvas.onpointermove = null; canvas.onpointerup = null; };
  });
  by_id("colour_hex").addEventListener("change", event => {
    const rgb = hex_to_rgb(event.target.value);
    if (rgb) set_picker_colour(rgb); else event.target.value = colour_for_run(app.colour_run_id);
  });
  by_id("reset_colour").addEventListener("click", () => {
    if (!app.colour_run_id) return;
    const run_id = app.colour_run_id;
    delete app.colours[run_id];
    save_json("thog2_local_run_colours", app.colours);
    set_picker_colour(hex_to_rgb(colour_for_run(run_id)), false);
    render_runs();
    render_run_heading();
    if (app.current_run_id === run_id && app.figures) render_figures();
  });
  document.addEventListener("pointerdown", event => {
    const popover = by_id("colour_popover");
    if (!popover.hidden && !popover.contains(event.target) && !event.target.classList.contains("colour-dot")) close_colour_picker();
    const menu = by_id("run_menu");
    if (!menu.hidden && !menu.contains(event.target) && !event.target.classList.contains("run-menu-button")) close_run_menu();
  });
  document.addEventListener("keydown", event => {
    if ((event.ctrlKey || event.metaKey) && event.key === ".") {
      event.preventDefault();
      toggle_runs_pane();
      return;
    }
    if (event.key === "Escape") {
      if (!by_id("chart_settings_overlay").hidden) close_chart_settings();
      else if (!by_id("settings_overlay").hidden) close_settings();
      else if (!by_id("run_menu").hidden) close_run_menu();
      else if (app.maximized_chart) restore_maximized_chart();
      else close_colour_picker();
    }
  });
  window.addEventListener("resize", () => requestAnimationFrame(() => {
    position_restore_button();
    resize_visible_plots();
  }));
  window.addEventListener("popstate", () => {
    const run_id = route_run_id();
    if (run_id && app.runs.some(run => run_identifier(run) === run_id)) select_run(run_id, {manual: true, replace_history: true});
  });
}

async function start() {
  migrate_panel_layout();
  ensure_depth_cards();
  update_chart_settings_buttons();
  build_swatches();
  bind_events();
  by_id("page_size").value = String(app.page_size);
  by_id("crash_timeout_minutes").value = String(app.crash_timeout_minutes);
  update_sort_direction_ui();
  update_file_source_tabs();
  const saved_width = Number(localStorage.getItem("thog2_local_runs_width"));
  if (Number.isFinite(saved_width) && saved_width > 0) set_runs_pane_width(saved_width);
  if (localStorage.getItem("thog2_local_runs_collapsed") === "1") toggle_runs_pane();
  render_empty_state();
  await refresh_catalog();
  setInterval(refresh_catalog, 2500);
  setInterval(refresh_current_run, 2000);
}

start();
// ^^^ THOG
