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
  selected: new Set(),
  group_by_host: false,
  maximized_chart: null,
  colour_run_id: null,
  picker_hue: 250,
  picker_saturation: 56,
  picker_value: 84,
};

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

function route_run_id() {
  const match = /^\/runs\/([^/]+)$/.exec(window.location.pathname);
  return match ? decodeURIComponent(match[1]) : null;
}

async function fetch_json(url) {
  const response = await fetch(url, {cache: "no-store"});
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

function filtered_runs() {
  const query = by_id("run_search").value.trim().toLowerCase();
  const filter = by_id("state_filter").value;
  const sort = by_id("run_sort").value;
  const runs = app.runs.filter(run => {
    const searchable = `${run.artifact_name} ${run.wandb_run_id} ${run.local_run_id} ${run.host_label}`.toLowerCase();
    return (!query || searchable.includes(query)) && (filter === "all" || run.run_state === filter);
  });
  runs.sort((left, right) => {
    if (sort === "name") return String(left.artifact_name).localeCompare(String(right.artifact_name));
    if (sort === "heatmap") return Number(right.heatmap_count) - Number(left.heatmap_count);
    if (sort === "depth") return Number(right.depth_snapshot_count) - Number(left.depth_snapshot_count);
    return String(right.updated_at).localeCompare(String(left.updated_at));
  });
  return runs;
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
  badge.className = `state-badge ${run.run_state}`;
  badge.textContent = run.run_state;
  state_cell.appendChild(badge);
  row.appendChild(state_cell);
  row.appendChild(text_cell(run.host_label || "—"));
  row.appendChild(text_cell(format_integer(run.heatmap_count), "numeric-column"));
  row.appendChild(text_cell(format_integer(run.depth_snapshot_count), "numeric-column"));
  row.appendChild(text_cell(format_integer(run.maximum_update), "numeric-column"));
  row.appendChild(text_cell(format_time(run.updated_at)));
  row.addEventListener("click", event => {
    if (!event.target.closest("button, input, a")) select_run(run_id, {manual: true});
  });
  body.appendChild(row);
}

function render_runs() {
  const body = by_id("runs_body");
  body.replaceChildren();
  const runs = filtered_runs();
  by_id("run_count").textContent = String(app.runs.length);
  by_id("listed_count").textContent = `${runs.length} listed`;
  const empty = by_id("empty_runs");
  empty.hidden = runs.length !== 0;
  if (!runs.length) {
    by_id("empty_runs_title").textContent = app.runs.length ? "No matching runs" : "Waiting for a local run";
    by_id("empty_runs_detail").textContent = app.runs.length
      ? "Change the search or filter to see runs."
      : "The viewer is ready. Start training and the W&B-linked run will appear here automatically.";
    return;
  }
  let previous_group = null;
  for (const run of runs) {
    const group = run.host_label || "Unlabelled host";
    if (app.group_by_host && group !== previous_group) {
      const group_row = document.createElement("tr");
      group_row.className = "group-row";
      const group_cell = document.createElement("td");
      group_cell.colSpan = 10;
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
      app.current_run_id = null;
      app.manual_selection = false;
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

function prepare_figure(figure, chart_name) {
  const prepared = clone_figure(figure);
  prepared.layout = prepared.layout || {};
  prepared.layout.uirevision = `${app.current_run_id}-${chart_name}`;
  prepared.layout.autosize = true;
  prepared.layout.paper_bgcolor = "white";
  prepared.layout.plot_bgcolor = "white";
  prepared.layout.hovermode = "closest";
  prepared.layout.colorway = [colour_for_run(app.current_run_id)];
  delete prepared.layout.width;
  delete prepared.layout.height;
  delete prepared.layout.title;
  prepared.layout.margin = chart_name === "heatmap"
    ? {l: 67, r: 82, t: 18, b: 54}
    : {l: 61, r: 18, t: 23, b: 50};

  if (chart_name === "heatmap") {
    prepared.layout.yaxis = prepared.layout.yaxis || {};
    prepared.layout.xaxis = prepared.layout.xaxis || {};
    delete prepared.layout.yaxis.scaleanchor;
    delete prepared.layout.yaxis.scaleratio;
    delete prepared.layout.yaxis.constrain;
    delete prepared.layout.xaxis.constrain;
    if (prepared.data[0]) {
      prepared.data[0].zsmooth = false;
      prepared.data[0].xgap = 0;
      prepared.data[0].ygap = 0;
      prepared.data[0].colorbar = prepared.data[0].colorbar || {};
      prepared.data[0].colorbar.thickness = 12;
      prepared.data[0].colorbar.len = 0.82;
    }
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

function depth_card(chart_name) {
  const article = document.createElement("article");
  article.className = "chart-card";
  article.dataset.chart = chart_name;
  const header = document.createElement("header");
  header.className = "chart-card-header";
  const copy = document.createElement("div");
  copy.className = "chart-heading-copy";
  const title = document.createElement("h2");
  title.textContent = chart_titles[chart_name] || chart_name;
  const detail = document.createElement("p");
  detail.id = `${chart_name}_detail`;
  detail.textContent = "Waiting for the first DEPTH weight snapshot.";
  copy.append(title, detail);
  const button = document.createElement("button");
  button.type = "button";
  button.className = "maximize-button";
  button.dataset.maximize = chart_name;
  button.textContent = "⛶";
  button.title = "Maximize chart";
  button.setAttribute("aria-label", `Maximize ${title.textContent}`);
  header.append(copy, button);
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
    {text: run.run_state},
    {text: run.host_label ? `host ${run.host_label}` : ""},
    {text: `${format_integer(run.heatmap_count)} probes`},
    {text: `${format_integer(run.depth_snapshot_count)} curve snapshots`},
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
    app.current_run_id = run_id;
    app.current_status = app.runs.find(run => run_identifier(run) === run_id) || null;
    app.figures = null;
    app.figure_revision = null;
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
    const width = Math.max(Math.min(300, pane_width), Math.min(pane_width, start_rect.width + pointer_event.clientX - start_x));
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
  app.maximized_chart = chart_name;
  const grid = by_id("chart_grid");
  grid.classList.add("is-maximized");
  by_id("charts_scroll").classList.add("maximized-mode");
  document.querySelectorAll(".chart-card").forEach(card => {
    const selected = card.dataset.chart === chart_name;
    card.classList.toggle("maximized", selected);
    const button = card.querySelector(".maximize-button");
    button.textContent = selected ? "↙" : "⛶";
    button.title = selected ? "Restore chart grid" : "Maximize chart";
    button.setAttribute("aria-label", selected ? "Restore chart grid" : `Maximize ${chart_titles[card.dataset.chart]}`);
  });
  requestAnimationFrame(() => requestAnimationFrame(resize_visible_plots));
}

function restore_maximized_chart() {
  if (!app.maximized_chart) return;
  app.maximized_chart = null;
  by_id("chart_grid").classList.remove("is-maximized");
  by_id("charts_scroll").classList.remove("maximized-mode");
  document.querySelectorAll(".chart-card").forEach(card => {
    card.classList.remove("maximized");
    const button = card.querySelector(".maximize-button");
    button.textContent = "⛶";
    button.title = "Maximize chart";
    button.setAttribute("aria-label", `Maximize ${chart_titles[card.dataset.chart]}`);
  });
  requestAnimationFrame(() => requestAnimationFrame(resize_visible_plots));
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
  by_id("run_search").addEventListener("input", render_runs);
  by_id("state_filter").addEventListener("change", render_runs);
  by_id("run_sort").addEventListener("change", render_runs);
  by_id("group_button").addEventListener("click", () => {
    app.group_by_host = !app.group_by_host;
    by_id("group_button").setAttribute("aria-pressed", String(app.group_by_host));
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
  by_id("chart_grid").addEventListener("click", event => {
    const button = event.target.closest(".maximize-button");
    if (button) toggle_maximized_chart(button.dataset.maximize);
  });
  by_id("chart_grid").addEventListener("pointerdown", event => {
    const handle = event.target.closest(".panel-resizer");
    if (handle) start_chart_resize(event, handle);
  });
  by_id("chart_grid").addEventListener("dblclick", event => {
    const handle = event.target.closest(".panel-resizer");
    if (!handle) return;
    const card = handle.closest(".chart-card");
    delete app.panel_sizes[card.dataset.chart];
    save_json("thog2_local_panel_sizes", app.panel_sizes);
    card.style.removeProperty("flex");
    card.style.removeProperty("height");
    requestAnimationFrame(() => resize_plot_in_card(card));
  });
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
  });
  document.addEventListener("keydown", event => {
    if ((event.ctrlKey || event.metaKey) && event.key === ".") {
      event.preventDefault();
      toggle_runs_pane();
      return;
    }
    if (event.key === "Escape") {
      if (app.maximized_chart) restore_maximized_chart(); else close_colour_picker();
    }
  });
  window.addEventListener("resize", () => requestAnimationFrame(resize_visible_plots));
  window.addEventListener("popstate", () => {
    const run_id = route_run_id();
    if (run_id && app.runs.some(run => run_identifier(run) === run_id)) select_run(run_id, {manual: true, replace_history: true});
  });
}

async function start() {
  ensure_depth_cards();
  build_swatches();
  bind_events();
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
