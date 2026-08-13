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
  toImageButtonOptions: {format: "png", scale: 2},
};

const app = {
  runs: [],
  requested_run: null,
  root: "",
  current_run_id: null,
  current_status: null,
  figures: null,
  figure_revision: null,
  colours: load_json("thog2_local_run_colours", {}),
  visibility: load_json("thog2_local_run_visibility", {}),
  selected: new Set(),
  group_by_host: false,
  modal_chart: null,
  modal_square_heatmap: true,
  colour_run_id: null,
  picker_hue: 250,
  picker_saturation: 56,
  picker_value: 84,
};

function by_id(id) { return document.getElementById(id); }

function load_json(key, fallback) {
  try { return JSON.parse(localStorage.getItem(key)) || fallback; }
  catch (_error) { return fallback; }
}

function save_json(key, value) {
  localStorage.setItem(key, JSON.stringify(value));
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
  const delta_seconds = Math.round((Date.now() - timestamp.getTime()) / 1000);
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

function run_identifier(run) { return String(run.local_run_id || run.wandb_run_id || run.run_name); }

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
  if (!is_visible(run_id)) row.classList.add("run-hidden");
  row.dataset.runId = run_id;

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
  name.addEventListener("click", () => open_run(run_id));
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
    wandb_cell.textContent = run.wandb_run_id || (run.local_run_id.startsWith("local-") ? run.local_run_id : "legacy");
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
  row.addEventListener("dblclick", () => open_run(run_id));
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
      : "The viewer is ready. Start training and the run will appear here automatically.";
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

async function refresh_catalog() {
  try {
    const catalog = await fetch_json("/api/runs");
    app.runs = catalog.runs;
    app.requested_run = catalog.requested_run;
    app.root = catalog.root;
    by_id("watch_status").textContent = catalog.waiting
      ? `Waiting · ${catalog.root}`
      : `Watching ${catalog.runs.length} run${catalog.runs.length === 1 ? "" : "s"} · ${catalog.root}`;
    render_runs();
    if (app.current_run_id && !app.runs.some(run => run_identifier(run) === app.current_run_id)) {
      show_runs();
    }
    if (app.requested_run && !app.current_run_id && app.runs.length === 1) {
      open_run(run_identifier(app.runs[0]), {replace_history: true});
    }
  } catch (error) {
    by_id("watch_status").textContent = `Viewer error: ${error.message}`;
  }
}

function clear_plot(mount) {
  if (mount.dataset.plotReady === "true") Plotly.purge(mount);
  mount.replaceChildren();
  delete mount.dataset.plotReady;
}

function clone_figure(figure) { return JSON.parse(JSON.stringify(figure)); }

function prepare_figure(figure, chart_name, options = {}) {
  const prepared = clone_figure(figure);
  prepared.layout = prepared.layout || {};
  prepared.layout.uirevision = prepared.layout.uirevision || `${app.current_run_id}-${chart_name}`;
  prepared.layout.autosize = true;
  delete prepared.layout.width;
  delete prepared.layout.height;
  prepared.layout.margin = chart_name === "heatmap"
    ? {l: 80, r: 105, t: 72, b: 72}
    : {l: 74, r: 28, t: 85, b: 64};

  if (chart_name === "heatmap") {
    prepared.layout.yaxis = prepared.layout.yaxis || {};
    prepared.layout.xaxis = prepared.layout.xaxis || {};
    if (!options.square_cells) {
      delete prepared.layout.yaxis.scaleanchor;
      delete prepared.layout.yaxis.scaleratio;
      delete prepared.layout.yaxis.constrain;
      delete prepared.layout.xaxis.constrain;
    } else {
      const dimensions = app.figures.heatmap_dimensions;
      const cell_size = 16;
      const left = 105;
      const right = 275;
      const top = 82;
      const bottom = 92;
      const plot_width = Math.max(cell_size, dimensions.probes * cell_size);
      const plot_height = Math.max(cell_size, dimensions.layers * cell_size);
      const width = left + plot_width + right;
      const height = top + plot_height + bottom;
      prepared.layout.autosize = false;
      prepared.layout.width = width;
      prepared.layout.height = height;
      prepared.layout.margin = {l: 0, r: 0, t: 0, b: 0};
      prepared.layout.xaxis.domain = [left / width, (left + plot_width) / width];
      prepared.layout.yaxis.domain = [bottom / height, (bottom + plot_height) / height];
      prepared.layout.yaxis.scaleanchor = "x";
      prepared.layout.yaxis.scaleratio = 1;
      prepared.layout.xaxis.constrain = "domain";
      prepared.layout.yaxis.constrain = "domain";
      if (prepared.data[0] && prepared.data[0].colorbar) {
        prepared.data[0].colorbar.x = (left + plot_width + 28) / width;
        prepared.data[0].colorbar.xanchor = "left";
        prepared.data[0].colorbar.len = Math.min(0.82, plot_height / height);
      }
    }
  }
  return prepared;
}

async function render_plot(mount, figure, chart_name, options = {}) {
  const prepared = prepare_figure(figure, chart_name, options);
  if (mount.dataset.plotReady === "true") {
    await Plotly.react(mount, prepared.data, prepared.layout, plot_config);
  } else {
    // The mount is deliberately separate from its placeholder. Clearing it here
    // prevents stale waiting content from participating in Plotly's layout.
    mount.replaceChildren();
    await Plotly.newPlot(mount, prepared.data, prepared.layout, plot_config);
    mount.dataset.plotReady = "true";
  }
}

function depth_card(chart_name) {
  const article = document.createElement("article");
  article.className = "chart-card";
  article.dataset.chart = chart_name;
  const header = document.createElement("header");
  header.className = "chart-card-header";
  const copy = document.createElement("div");
  const title = document.createElement("h2");
  title.textContent = chart_titles[chart_name] || chart_name;
  const detail = document.createElement("p");
  detail.textContent = "Interactive history; scroll to zoom and double-click to reset.";
  copy.append(title, detail);
  const button = document.createElement("button");
  button.type = "button";
  button.className = "maximize-button";
  button.dataset.maximize = chart_name;
  button.textContent = "↗ Enlarge";
  button.addEventListener("click", () => open_modal(chart_name));
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
  return article;
}

function ensure_depth_cards() {
  const grid = by_id("depth_grid");
  for (const chart_name of Object.keys(chart_titles).filter(name => name !== "heatmap")) {
    if (!by_id(`${chart_name}_plot`)) grid.appendChild(depth_card(chart_name));
  }
}

function reset_run_charts() {
  clear_plot(by_id("heatmap_plot"));
  by_id("heatmap_placeholder").hidden = false;
  for (const chart_name of Object.keys(chart_titles).filter(name => name !== "heatmap")) {
    const mount = by_id(`${chart_name}_plot`);
    if (mount) clear_plot(mount);
    const placeholder = by_id(`${chart_name}_placeholder`);
    if (placeholder) placeholder.hidden = false;
  }
}

async function render_figures() {
  if (!app.figures) return;
  if (app.figures.heatmap) {
    by_id("heatmap_placeholder").hidden = true;
    await render_plot(by_id("heatmap_plot"), app.figures.heatmap, "heatmap", {square_cells: false});
  }
  ensure_depth_cards();
  for (const chart_name of Object.keys(chart_titles).filter(name => name !== "heatmap")) {
    const figure = app.figures.depth[chart_name];
    if (!figure) continue;
    by_id(`${chart_name}_placeholder`).hidden = true;
    await render_plot(by_id(`${chart_name}_plot`), figure, chart_name);
  }
}

function current_run() {
  return app.runs.find(run => run_identifier(run) === app.current_run_id) || app.current_status;
}

function render_run_heading() {
  const run = current_run();
  if (!run) return;
  by_id("run_title").textContent = run.artifact_name;
  by_id("run_colour_large").style.background = colour_for_run(app.current_run_id);
  const subtitle = by_id("run_subtitle");
  subtitle.replaceChildren();
  const values = [
    `${run.run_state}`,
    run.wandb_run_id ? `W&B ID ${run.wandb_run_id}` : `Local ID ${run.local_run_id}`,
    run.host_label ? `host ${run.host_label}` : "",
    `${format_integer(run.heatmap_count)} probes`,
    `${format_integer(run.depth_snapshot_count)} curve snapshots`,
    format_bytes(run.database_bytes),
  ].filter(Boolean);
  for (const value of values) {
    const span = document.createElement("span");
    span.textContent = value;
    subtitle.appendChild(span);
  }
  if (run.wandb_url) {
    const link = document.createElement("a");
    link.href = run.wandb_url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = "Open in W&B ↗";
    subtitle.appendChild(link);
  }
}

async function refresh_current_run() {
  if (!app.current_run_id) return;
  try {
    const encoded = encodeURIComponent(app.current_run_id);
    const status = await fetch_json(`/api/status?run=${encoded}`);
    app.current_status = status;
    render_run_heading();
    const revision = JSON.stringify(status.revision);
    if (revision === app.figure_revision) return;
    app.figure_revision = revision;
    app.figures = await fetch_json(`/api/figures?run=${encoded}`);
    await render_figures();
    if (app.modal_chart) await render_modal();
  } catch (error) {
    show_toast(`Run refresh failed: ${error.message}`);
  }
}

function open_run(run_id, options = {}) {
  if (app.current_run_id !== run_id) {
    app.current_run_id = run_id;
    app.current_status = null;
    app.figures = null;
    app.figure_revision = null;
    ensure_depth_cards();
    reset_run_charts();
  }
  by_id("runs_view").hidden = true;
  by_id("run_view").hidden = false;
  by_id("breadcrumb_leaf").textContent = "Run";
  if (options.replace_history) history.replaceState({}, "", `/runs/${encodeURIComponent(run_id)}`);
  else history.pushState({}, "", `/runs/${encodeURIComponent(run_id)}`);
  render_run_heading();
  refresh_current_run();
}

function show_runs(options = {}) {
  close_modal();
  app.current_run_id = null;
  app.current_status = null;
  app.figures = null;
  app.figure_revision = null;
  by_id("runs_view").hidden = false;
  by_id("run_view").hidden = true;
  by_id("breadcrumb_leaf").textContent = "Runs";
  if (!options.from_history) history.pushState({}, "", "/runs");
}

function open_modal(chart_name) {
  if (!app.figures) return;
  const figure = chart_name === "heatmap" ? app.figures.heatmap : app.figures.depth[chart_name];
  if (!figure) {
    show_toast("This chart has no data yet.");
    return;
  }
  app.modal_chart = chart_name;
  app.modal_square_heatmap = chart_name === "heatmap";
  by_id("modal_title").textContent = chart_titles[chart_name] || chart_name;
  by_id("modal_detail").textContent = current_run()?.artifact_name || "";
  by_id("heatmap_aspect_toggle").hidden = chart_name !== "heatmap";
  by_id("chart_modal").hidden = false;
  document.body.classList.add("modal-open");
  render_modal();
}

async function render_modal() {
  if (!app.modal_chart || !app.figures) return;
  const chart_name = app.modal_chart;
  const figure = chart_name === "heatmap" ? app.figures.heatmap : app.figures.depth[chart_name];
  if (!figure) return;
  const mount = by_id("modal_plot");
  mount.style.width = "100%";
  mount.style.height = `${Math.max(520, by_id("modal_scroll").clientHeight)}px`;
  if (chart_name === "heatmap" && app.modal_square_heatmap) {
    const dimensions = app.figures.heatmap_dimensions;
    mount.style.width = `${105 + Math.max(16, dimensions.probes * 16) + 275}px`;
    mount.style.height = `${82 + Math.max(16, dimensions.layers * 16) + 92}px`;
  }
  by_id("heatmap_aspect_toggle").textContent = app.modal_square_heatmap ? "Fit width" : "Square cells";
  clear_plot(mount);
  await render_plot(mount, figure, chart_name, {square_cells: app.modal_square_heatmap});
}

function close_modal() {
  const modal = by_id("chart_modal");
  if (modal.hidden) return;
  clear_plot(by_id("modal_plot"));
  modal.hidden = true;
  app.modal_chart = null;
  document.body.classList.remove("modal-open");
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
  const match = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
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
    app.colours[app.colour_run_id] = rgb_to_hex(rgb);
    save_json("thog2_local_run_colours", app.colours);
    document.querySelectorAll(`[data-run-id="${CSS.escape(app.colour_run_id)}"] .colour-dot`).forEach(dot => dot.style.background = rgb_to_hex(rgb));
    if (app.current_run_id === app.colour_run_id) by_id("run_colour_large").style.background = rgb_to_hex(rgb);
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
  popover.style.left = `${Math.min(anchor_rect.left - 12, window.innerWidth - width - 12)}px`;
  popover.style.top = `${Math.min(anchor_rect.bottom + 8, window.innerHeight - height - 12)}px`;
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
  by_id("runs_nav").addEventListener("click", show_runs);
  by_id("back_to_runs").addEventListener("click", show_runs);
  document.querySelector('[data-maximize="heatmap"]').addEventListener("click", () => open_modal("heatmap"));
  by_id("close_modal").addEventListener("click", close_modal);
  by_id("heatmap_aspect_toggle").addEventListener("click", () => {
    app.modal_square_heatmap = !app.modal_square_heatmap;
    render_modal();
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
    delete app.colours[app.colour_run_id];
    save_json("thog2_local_run_colours", app.colours);
    set_picker_colour(hex_to_rgb(colour_for_run(app.colour_run_id)), false);
    render_runs();
  });
  document.addEventListener("pointerdown", event => {
    const popover = by_id("colour_popover");
    if (!popover.hidden && !popover.contains(event.target) && !event.target.classList.contains("colour-dot")) close_colour_picker();
  });
  document.addEventListener("keydown", event => {
    if (event.key === "Escape") {
      if (!by_id("chart_modal").hidden) close_modal(); else close_colour_picker();
    }
  });
  window.addEventListener("popstate", route_from_location);
}

function route_from_location() {
  const match = /^\/runs\/([^/]+)$/.exec(window.location.pathname);
  if (match) open_run(decodeURIComponent(match[1]), {replace_history: true});
  else show_runs({from_history: true});
}

async function start() {
  ensure_depth_cards();
  build_swatches();
  bind_events();
  await refresh_catalog();
  const match = /^\/runs\/([^/]+)$/.exec(window.location.pathname);
  if (match) open_run(decodeURIComponent(match[1]), {replace_history: true});
  setInterval(refresh_catalog, 3000);
  setInterval(refresh_current_run, 3000);
}

start();
// ^^^ THOG
