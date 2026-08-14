// vvv THOG
"use strict";

// The base dashboard remains deliberately generic. This patch owns the local
// heatmap coordinate remap, viewer-only heatmap display controls, and fixed
// inner canvases used by each chart-card scroll viewport.

const heatmap_viewer_settings_key = "thog2_local_heatmap_viewer_settings";
const heatmap_default_probe_row_height_px = 12;
const heatmap_minimum_probe_row_height_px = 1;
const heatmap_maximum_probe_row_height_px = 24;
const heatmap_plot_chrome_height_px = 132;

function signed_layer_offset(value) {
  const offset = Number(value);
  if (!Number.isFinite(offset) || offset === 0) return "0";
  return offset > 0 ? `+${offset}` : String(offset);
}

function populated_heatmap_cell(value) {
  return value !== null && value !== undefined && Number.isFinite(Number(value));
}

function latest_finite_value(values) {
  for (let index = values.length - 1; index >= 0; index -= 1) {
    if (Number.isFinite(values[index])) return values[index];
  }
  return null;
}

function heatmap_viewer_settings() {
  return load_json(heatmap_viewer_settings_key, {});
}

function heatmap_run_settings() {
  return current_run()?.heatmap_settings || {};
}

function heatmap_settings_for_current_run() {
  if (!app.current_run_id) return {};
  return heatmap_viewer_settings()[app.current_run_id] || {};
}

function heatmap_abs_limit(default_limit) {
  const override = Number(heatmap_settings_for_current_run().abs_limit);
  if (Number.isFinite(override) && override > 0) return override;
  const run_limit = Number(heatmap_run_settings().abs_limit);
  if (Number.isFinite(run_limit) && run_limit > 0) return run_limit;
  return Number.isFinite(default_limit) && default_limit > 0 ? default_limit : 0.05;
}

function heatmap_probe_row_height_px() {
  const override = Number(heatmap_settings_for_current_run().probe_row_height_px);
  if (Number.isFinite(override)) {
    return Math.min(
      heatmap_maximum_probe_row_height_px,
      Math.max(heatmap_minimum_probe_row_height_px, override),
    );
  }
  return heatmap_default_probe_row_height_px;
}

function save_heatmap_viewer_setting(name, value) {
  if (!app.current_run_id) return;
  const settings = heatmap_viewer_settings();
  settings[app.current_run_id] = {
    ...(settings[app.current_run_id] || {}),
    [name]: value,
  };
  save_json(heatmap_viewer_settings_key, settings);
}

function reset_heatmap_viewer_setting(name) {
  if (!app.current_run_id) return;
  const settings = heatmap_viewer_settings();
  if (!settings[app.current_run_id]) return;
  delete settings[app.current_run_id][name];
  if (!Object.keys(settings[app.current_run_id]).length) delete settings[app.current_run_id];
  save_json(heatmap_viewer_settings_key, settings);
}

function relative_heatmap_bounds(figure) {
  const heatmap_trace = (figure?.data || []).find(trace => trace.type === "heatmap");
  const active_layer_trace = (figure?.data || []).find(
    trace => trace !== heatmap_trace && Array.isArray(trace.x) && Array.isArray(trace.y)
  );
  const candidate_layers = Array.isArray(heatmap_trace?.y) ? heatmap_trace.y.map(Number) : [];
  const active_layers = Array.isArray(active_layer_trace?.y) ? active_layer_trace.y.map(Number) : [];
  const source_z = Array.isArray(heatmap_trace?.z) ? heatmap_trace.z : [];
  let minimum = Infinity;
  let maximum = -Infinity;
  for (let probe_index = 0; probe_index < active_layers.length; probe_index += 1) {
    const active_layers_at_probe = active_layers[probe_index];
    if (!Number.isFinite(active_layers_at_probe)) continue;
    for (let candidate_index = 0; candidate_index < candidate_layers.length; candidate_index += 1) {
      const candidate_layers_at_probe = candidate_layers[candidate_index];
      const cell = Array.isArray(source_z[candidate_index])
        ? source_z[candidate_index][probe_index]
        : null;
      if (!Number.isFinite(candidate_layers_at_probe) || !populated_heatmap_cell(cell)) continue;
      const offset = candidate_layers_at_probe - active_layers_at_probe;
      minimum = Math.min(minimum, offset);
      maximum = Math.max(maximum, offset);
    }
  }
  if (!Number.isFinite(minimum) || !Number.isFinite(maximum)) return {minimum: 0, maximum: 0};
  return {minimum: Math.floor(minimum), maximum: Math.ceil(maximum)};
}

function heatmap_probe_count(figure) {
  const heatmap_trace = (figure?.data || []).find(trace => trace.type === "heatmap");
  return Array.isArray(heatmap_trace?.x) ? heatmap_trace.x.length : 0;
}

function transpose_heatmap_relative(prepared) {
  const original_xaxis = {...(prepared.layout.xaxis || {})};
  const original_yaxis = {...(prepared.layout.yaxis || {})};
  const heatmap_trace = (prepared.data || []).find(trace => trace.type === "heatmap");
  const active_layer_trace = (prepared.data || []).find(
    trace => trace !== heatmap_trace && Array.isArray(trace.x) && Array.isArray(trace.y)
  );
  if (!heatmap_trace) return;

  const probe_coordinates = Array.isArray(heatmap_trace.x) ? [...heatmap_trace.x] : [];
  const candidate_layers = Array.isArray(heatmap_trace.y) ? heatmap_trace.y.map(Number) : [];
  const active_layers = Array.isArray(active_layer_trace?.y) ? active_layer_trace.y.map(Number) : [];
  const latest_active_layer_count = latest_finite_value(active_layers);
  const original_z = Array.isArray(heatmap_trace.z) ? heatmap_trace.z : [];
  const original_customdata = Array.isArray(heatmap_trace.customdata) ? heatmap_trace.customdata : [];
  const configured_colour_limit = Math.max(
    Math.abs(Number(heatmap_trace.zmin) || 0),
    Math.abs(Number(heatmap_trace.zmax) || 0),
  );
  const colour_limit = heatmap_abs_limit(configured_colour_limit);
  const step_values = probe_coordinates.map((_coordinate, probe_index) => {
    if (
      Array.isArray(active_layer_trace?.customdata)
      && active_layer_trace.customdata[probe_index] !== undefined
    ) {
      return active_layer_trace.customdata[probe_index];
    }
    if (
      Array.isArray(original_customdata[0])
      && original_customdata[0][probe_index] !== undefined
    ) {
      return original_customdata[0][probe_index];
    }
    return probe_coordinates[probe_index];
  });

  let minimum_offset = Infinity;
  let maximum_offset = -Infinity;
  for (let probe_index = 0; probe_index < probe_coordinates.length; probe_index += 1) {
    const active_layers_at_probe = active_layers[probe_index];
    if (!Number.isFinite(active_layers_at_probe)) continue;
    for (let candidate_index = 0; candidate_index < candidate_layers.length; candidate_index += 1) {
      const candidate_layers_at_probe = candidate_layers[candidate_index];
      const cell = Array.isArray(original_z[candidate_index])
        ? original_z[candidate_index][probe_index]
        : null;
      if (!Number.isFinite(candidate_layers_at_probe) || !populated_heatmap_cell(cell)) continue;
      const offset = candidate_layers_at_probe - active_layers_at_probe;
      minimum_offset = Math.min(minimum_offset, offset);
      maximum_offset = Math.max(maximum_offset, offset);
    }
  }
  if (!Number.isFinite(minimum_offset) || !Number.isFinite(maximum_offset)) {
    minimum_offset = 0;
    maximum_offset = 0;
  }
  minimum_offset = Math.floor(minimum_offset);
  maximum_offset = Math.ceil(maximum_offset);

  const offsets = Array.from(
    {length: maximum_offset - minimum_offset + 1},
    (_unused, index) => minimum_offset + index,
  );
  const offset_index = new Map(offsets.map((offset, index) => [offset, index]));
  const relative_z = probe_coordinates.map(() => Array(offsets.length).fill(null));
  const relative_customdata = probe_coordinates.map(() => Array(offsets.length).fill(null));

  for (let probe_index = 0; probe_index < probe_coordinates.length; probe_index += 1) {
    const active_layers_at_probe = active_layers[probe_index];
    if (!Number.isFinite(active_layers_at_probe)) continue;
    for (let candidate_index = 0; candidate_index < candidate_layers.length; candidate_index += 1) {
      const candidate_layers_at_probe = candidate_layers[candidate_index];
      const cell = Array.isArray(original_z[candidate_index])
        ? original_z[candidate_index][probe_index]
        : null;
      if (!Number.isFinite(candidate_layers_at_probe) || !populated_heatmap_cell(cell)) continue;
      const offset = candidate_layers_at_probe - active_layers_at_probe;
      const destination = offset_index.get(offset);
      if (destination === undefined) continue;
      relative_z[probe_index][destination] = Number(cell);
      relative_customdata[probe_index][destination] = [
        step_values[probe_index],
        candidate_layers_at_probe,
        signed_layer_offset(offset),
      ];
    }
  }

  heatmap_trace.x = offsets;
  heatmap_trace.y = probe_coordinates;
  heatmap_trace.z = relative_z;
  heatmap_trace.customdata = relative_customdata;
  heatmap_trace.zmin = -colour_limit;
  heatmap_trace.zmax = colour_limit;
  heatmap_trace.zmid = 0;
  heatmap_trace.colorscale = [
    [0.00, "rgb(0,255,0)"],
    [0.08, "rgb(0,220,0)"],
    [0.16, "rgb(0,182,0)"],
    [0.24, "rgb(0,150,0)"],
    [0.32, "rgb(28,122,28)"],
    [0.40, "rgb(58,103,58)"],
    [0.50, "rgb(88,88,88)"],
    [0.60, "rgb(112,76,76)"],
    [0.68, "rgb(138,64,64)"],
    [0.76, "rgb(168,52,52)"],
    [0.84, "rgb(198,40,40)"],
    [0.92, "rgb(228,24,24)"],
    [1.00, "rgb(255,0,0)"],
  ];
  heatmap_trace.zsmooth = false;
  heatmap_trace.xgap = 0;
  heatmap_trace.ygap = 0;
  heatmap_trace.hovertemplate = (
    "step=%{customdata[0]}<br>candidate offset=%{customdata[2]}<br>"
    + "candidate layers=%{customdata[1]}<br>Δloss=%{z:.8f}<extra></extra>"
  );
  heatmap_trace.colorbar = heatmap_trace.colorbar || {};
  heatmap_trace.colorbar.thickness = 12;
  heatmap_trace.colorbar.len = 0.82;
  heatmap_trace.colorbar.title = `candidate loss − current loss · ±${colour_limit.toPrecision(3)}`;

  if (active_layer_trace) {
    active_layer_trace.x = probe_coordinates.map(() => 0);
    active_layer_trace.y = [...probe_coordinates];
    active_layer_trace.customdata = [...step_values];
    active_layer_trace.line = {...(active_layer_trace.line || {}), color: "white", width: 2};
    active_layer_trace.hovertemplate = (
      "step=%{customdata}<br>active-layer offset=0<extra></extra>"
    );
  }

  prepared.layout.xaxis = {
    ...original_yaxis,
    title: {text: "candidate layer-count offset from active layer count", standoff: 16},
    range: [minimum_offset - 0.5, maximum_offset + 0.5],
    tickmode: "array",
    tickvals: offsets,
    ticktext: offsets.map(offset => (
      offset === 0 && Number.isFinite(latest_active_layer_count)
        ? `0<br><b>L=${latest_active_layer_count}</b>`
        : signed_layer_offset(offset)
    )),
    anchor: "y",
    side: "bottom",
    automargin: false,
  };
  prepared.layout.yaxis = {
    ...original_xaxis,
    title: {text: "step"},
    range: [0.5, Math.max(1, probe_coordinates.length) + 0.5],
    domain: [0, 1],
    anchor: "x",
    automargin: false,
  };
  for (const axis of [prepared.layout.xaxis, prepared.layout.yaxis]) {
    delete axis.scaleanchor;
    delete axis.scaleratio;
    delete axis.constrain;
    delete axis.constraintoward;
  }

  // Pin the heatmap cell domain directly to the bottom x-axis. Resizing changes
  // cell height, not the amount of unused domain between the body and the axis.
  prepared.layout.margin = {...(prepared.layout.margin || {}), t: 18, b: 76};

  const tick_indices = evenly_spaced_indices(probe_coordinates.length, 20);
  prepared.layout.yaxis.tickmode = "array";
  prepared.layout.yaxis.tickvals = tick_indices.map(index => probe_coordinates[index]);
  prepared.layout.yaxis.ticktext = tick_indices.map(index => String(step_values[index]));
  prepared.layout.annotations = [];
}

function plot_mount_dimensions(mount, chart_name, figure) {
  const shell = mount.closest(".plot-shell");
  const shell_width = Math.max(1, shell?.clientWidth || 0);
  const shell_height = Math.max(1, shell?.clientHeight || 0);
  if (chart_name === "heatmap") {
    const bounds = relative_heatmap_bounds(figure);
    const column_count = Math.max(1, bounds.maximum - bounds.minimum + 1);
    const probe_count = heatmap_probe_count(figure);
    const natural_height = (
      heatmap_plot_chrome_height_px
      + Math.max(1, probe_count) * heatmap_probe_row_height_px()
    );
    return {
      width: Math.max(shell_width, 720, 190 + column_count * 34),
      height: Math.max(shell_height, 320, natural_height),
    };
  }
  return {
    width: Math.max(shell_width, 620),
    height: Math.max(shell_height, 320),
  };
}

function figure_for_chart(chart_name) {
  if (!app.figures) return null;
  return chart_name === "heatmap" ? app.figures.heatmap : app.figures.depth?.[chart_name];
}

function ensure_plot_scroll_canvas(mount) {
  const shell = mount.closest(".plot-shell");
  if (!shell) return null;
  if (mount.parentElement?.classList.contains("plot-scroll-canvas")) return mount.parentElement;
  const canvas = document.createElement("div");
  canvas.className = "plot-scroll-canvas";
  shell.insertBefore(canvas, mount);
  canvas.appendChild(mount);
  return canvas;
}

function set_plot_dimensions(mount, dimensions) {
  const canvas = ensure_plot_scroll_canvas(mount);
  if (canvas) {
    canvas.style.width = `${Math.round(dimensions.width)}px`;
    canvas.style.height = `${Math.round(dimensions.height)}px`;
  }
  mount.style.width = "100%";
  mount.style.height = "100%";
}

const fixed_plot_config = {...plot_config, responsive: false};

transpose_heatmap = transpose_heatmap_relative;

render_plot = async function(mount, figure, chart_name) {
  const prepared = prepare_figure(figure, chart_name);
  const dimensions = plot_mount_dimensions(mount, chart_name, figure);
  set_plot_dimensions(mount, dimensions);
  prepared.layout.autosize = false;
  prepared.layout.width = Math.round(dimensions.width);
  prepared.layout.height = Math.round(dimensions.height);
  if (mount.dataset.plotReady === "true") {
    await Plotly.react(mount, prepared.data, prepared.layout, fixed_plot_config);
  } else {
    mount.replaceChildren();
    await Plotly.newPlot(mount, prepared.data, prepared.layout, fixed_plot_config);
    mount.dataset.plotReady = "true";
  }
  if (chart_name === "heatmap") sync_heatmap_scale_control();
};

resize_plot_in_card = function(card) {
  const mount = card.querySelector(".plot-mount");
  if (!mount || mount.dataset.plotReady !== "true") return;
  const chart_name = card.dataset.chart;
  const figure = figure_for_chart(chart_name);
  if (!figure) return;
  const dimensions = plot_mount_dimensions(mount, chart_name, figure);
  set_plot_dimensions(mount, dimensions);
  Plotly.relayout(mount, {
    width: Math.round(dimensions.width),
    height: Math.round(dimensions.height),
  });
  if (chart_name === "heatmap") sync_heatmap_scale_control();
};

function make_heatmap_settings_row(grid, label_text, id, options = {}) {
  const label = document.createElement("label");
  label.htmlFor = id;
  label.textContent = label_text;
  const input = document.createElement("input");
  input.id = id;
  input.type = options.type || "text";
  if (options.readonly) input.readOnly = true;
  if (options.step !== undefined) input.step = String(options.step);
  if (options.min !== undefined) input.min = String(options.min);
  if (options.max !== undefined) input.max = String(options.max);
  grid.append(label, input);
  return input;
}

function sync_heatmap_scale_control() {
  const range = by_id("heatmap_vertical_scale");
  const value = by_id("heatmap_vertical_scale_value");
  if (!range || !value) return;
  const row_height = heatmap_probe_row_height_px();
  range.value = String(row_height);
  range.disabled = !app.current_run_id;
  value.textContent = `${row_height}px/step`;
}

function install_heatmap_scale_control() {
  const header = document.querySelector('.chart-card[data-chart="heatmap"] .chart-card-header');
  const maximize = header?.querySelector(".maximize-button");
  if (!header || !maximize || by_id("heatmap_vertical_scale")) return;

  const control = document.createElement("label");
  control.className = "heatmap-vertical-scale-control";
  control.title = "Scale the heatmap vertically without changing its width";
  const glyph = document.createElement("span");
  glyph.textContent = "↕";
  glyph.setAttribute("aria-hidden", "true");
  const range = document.createElement("input");
  range.id = "heatmap_vertical_scale";
  range.type = "range";
  range.min = String(heatmap_minimum_probe_row_height_px);
  range.max = String(heatmap_maximum_probe_row_height_px);
  range.step = "1";
  range.setAttribute("aria-label", "Heatmap vertical pixels per step");
  const value = document.createElement("span");
  value.id = "heatmap_vertical_scale_value";
  control.append(glyph, range, value);
  header.insertBefore(control, maximize);

  range.addEventListener("input", () => {
    const numeric = Number(range.value);
    if (!Number.isFinite(numeric)) return;
    save_heatmap_viewer_setting("probe_row_height_px", numeric);
    sync_heatmap_scale_control();
    const card = range.closest(".chart-card");
    if (card) resize_plot_in_card(card);
    const settings_input = by_id("heatmap_setting_row_height");
    if (settings_input) settings_input.value = String(numeric);
  });
  sync_heatmap_scale_control();
}

function sync_heatmap_settings_panel() {
  const mode = by_id("heatmap_setting_mode");
  if (!mode) return;
  const settings = heatmap_run_settings();
  mode.value = settings.mode ?? "—";
  by_id("heatmap_setting_destination").value = settings.destination ?? "—";
  by_id("heatmap_setting_linear").value = settings.linear_max_step ?? "—";
  const run_limit = Number(settings.abs_limit);
  by_id("heatmap_setting_abs_limit").value = String(
    heatmap_abs_limit(Number.isFinite(run_limit) ? run_limit : 0.05)
  );
  by_id("heatmap_setting_row_height").value = String(heatmap_probe_row_height_px());
  const active = Boolean(app.current_run_id);
  for (const id of [
    "heatmap_setting_abs_limit",
    "heatmap_setting_row_height",
    "heatmap_setting_reset",
    "heatmap_setting_row_height_reset",
  ]) {
    const element = by_id(id);
    if (element) element.disabled = !active;
  }
}

function install_heatmap_settings_panel() {
  const settings_content = document.querySelector(".settings-content");
  if (!settings_content || by_id("heatmap_settings_section")) return;

  const section = document.createElement("section");
  section.id = "heatmap_settings_section";
  section.className = "heatmap-settings-section";
  const heading = document.createElement("h3");
  heading.textContent = "Layer-count Δloss heatmap";
  const grid = document.createElement("div");
  grid.className = "heatmap-settings-grid";

  make_heatmap_settings_row(grid, "instrumentation__delta_loss_v_layer_heatmap", "heatmap_setting_mode", {readonly: true});
  make_heatmap_settings_row(grid, "instrumentation__delta_loss_v_layer_heatmap__destination", "heatmap_setting_destination", {readonly: true});
  make_heatmap_settings_row(grid, "instrumentation__delta_loss_v_layer_heatmap_linear", "heatmap_setting_linear", {readonly: true});
  const abs_limit = make_heatmap_settings_row(
    grid,
    "instrumentation__delta_loss_v_layer_heatmap_abs_limit",
    "heatmap_setting_abs_limit",
    {type: "number", min: 0.000000001, step: 0.01},
  );
  const row_height = make_heatmap_settings_row(
    grid,
    "viewer vertical pixels / step",
    "heatmap_setting_row_height",
    {
      type: "number",
      min: heatmap_minimum_probe_row_height_px,
      max: heatmap_maximum_probe_row_height_px,
      step: 1,
    },
  );

  const note = document.createElement("p");
  note.className = "heatmap-settings-note";
  note.textContent = (
    "abs_limit and vertical pixels/step are live viewer overrides. Mode, destination and linear max-step are capture/routing controls from the selected run; the viewer cannot recreate probes that were not recorded."
  );

  const actions = document.createElement("div");
  actions.className = "heatmap-settings-actions";
  const reset_abs = document.createElement("button");
  reset_abs.id = "heatmap_setting_reset";
  reset_abs.type = "button";
  reset_abs.textContent = "Reset abs limit to run value";
  const reset_height = document.createElement("button");
  reset_height.id = "heatmap_setting_row_height_reset";
  reset_height.type = "button";
  reset_height.textContent = "Reset vertical scale to 12 px/step";
  actions.append(reset_abs, reset_height);
  section.append(heading, grid, note, actions);
  settings_content.appendChild(section);

  let render_timer = null;
  const rerender_heatmap = () => {
    clearTimeout(render_timer);
    render_timer = setTimeout(() => {
      if (app.figures && app.current_run_id) render_figures();
    }, 100);
  };

  abs_limit.addEventListener("input", () => {
    const numeric = Number(abs_limit.value);
    if (!Number.isFinite(numeric) || numeric <= 0) return;
    save_heatmap_viewer_setting("abs_limit", numeric);
    rerender_heatmap();
  });
  row_height.addEventListener("input", () => {
    const numeric = Number(row_height.value);
    if (!Number.isFinite(numeric)) return;
    save_heatmap_viewer_setting(
      "probe_row_height_px",
      Math.min(heatmap_maximum_probe_row_height_px, Math.max(heatmap_minimum_probe_row_height_px, numeric)),
    );
    sync_heatmap_scale_control();
    const card = document.querySelector('.chart-card[data-chart="heatmap"]');
    if (card) resize_plot_in_card(card);
  });
  reset_abs.addEventListener("click", () => {
    reset_heatmap_viewer_setting("abs_limit");
    sync_heatmap_settings_panel();
    rerender_heatmap();
  });
  reset_height.addEventListener("click", () => {
    reset_heatmap_viewer_setting("probe_row_height_px");
    sync_heatmap_settings_panel();
    sync_heatmap_scale_control();
    const card = document.querySelector('.chart-card[data-chart="heatmap"]');
    if (card) resize_plot_in_card(card);
  });
  by_id("settings_nav")?.addEventListener("click", sync_heatmap_settings_panel);
  sync_heatmap_settings_panel();
}

function install_dashboard_ui_patch() {
  if (!document.querySelector('link[href="/assets/dashboard_ui_patch.css"]')) {
    const stylesheet = document.createElement("link");
    stylesheet.rel = "stylesheet";
    stylesheet.href = "/assets/dashboard_ui_patch.css";
    document.head.appendChild(stylesheet);
  }

  const header_eye = document.querySelector(".runs-table thead .visibility-column .eye-icon");
  if (header_eye) {
    header_eye.className = "visibility-header-eye";
    header_eye.replaceChildren(icon_svg("eye_open"));
    header_eye.title = "Run visibility";
  }

  install_heatmap_scale_control();
  install_heatmap_settings_panel();
}

install_dashboard_ui_patch();

if (app.figures && app.current_run_id) {
  queueMicrotask(() => render_figures());
}
// ^^^ THOG

// vvv THOG
// W&B-like per-artifact navigation and Overview. Charts remains the default;
// Logs, Files and Artifacts are intentionally blank placeholders for now.
const local_detail_tabs = Object.freeze(["charts", "overview", "logs", "files", "artifacts"]);
let local_active_detail_tab = "charts";

function local_first_present(object, keys, fallback = "—") {
  for (const key of keys) {
    const value = object?.[key];
    if (value !== null && value !== undefined && value !== "") return value;
  }
  return fallback;
}

function local_overview_value(value) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") return Number.isInteger(value) ? value.toLocaleString() : String(value);
  if (Array.isArray(value)) return value.length ? value.join(", ") : "—";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function local_overview_timestamp(value) {
  if (!value) return "—";
  const timestamp = new Date(value);
  if (!Number.isFinite(timestamp.getTime())) return String(value);
  return timestamp.toLocaleString(undefined, {
    year: "numeric", month: "long", day: "numeric",
    hour: "numeric", minute: "2-digit", second: "2-digit",
  });
}

function local_overview_duration(run) {
  const start = Date.parse(run?.created_at || "");
  const end = display_run_state(run) === "running" ? Date.now() : Date.parse(run?.updated_at || "");
  if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) return "—";
  let seconds = Math.floor((end - start) / 1000);
  const days = Math.floor(seconds / 86400); seconds -= days * 86400;
  const hours = Math.floor(seconds / 3600); seconds -= hours * 3600;
  const minutes = Math.floor(seconds / 60); seconds -= minutes * 60;
  return [days ? `${days}d` : "", (days || hours) ? `${hours}h` : "", (days || hours || minutes) ? `${minutes}m` : "", `${seconds}s`].filter(Boolean).join(" ");
}

function local_wandb_owner(run) {
  try { return new URL(run?.wandb_url || "").pathname.split("/").filter(Boolean)[0] || "—"; }
  catch (_error) { return "—"; }
}

function local_run_path(run) {
  try {
    const parts = new URL(run?.wandb_url || "").pathname.split("/").filter(Boolean);
    if (parts.length >= 4 && parts[2] === "runs") return `${parts[0]}/${parts[1]}/${parts[3]}`;
  } catch (_error) {}
  return run?.run_directory || "—";
}

function local_state_badge(run) {
  const state = display_run_state(run);
  const badge = document.createElement("span");
  badge.className = `state-badge ${state}`;
  const icon = document.createElement("span");
  icon.className = "state-icon";
  icon.appendChild(icon_svg(["running", "finished", "crashed"].includes(state) ? state : "unknown"));
  badge.append(icon, document.createTextNode(state));
  return badge;
}

function local_append_meta(container, label_text, value) {
  const label = document.createElement("div");
  label.className = "overview-meta-label";
  label.textContent = label_text;
  const body = document.createElement("div");
  body.className = "overview-meta-value";
  if (value instanceof Node) body.appendChild(value); else body.textContent = local_overview_value(value);
  container.append(label, body);
}

function local_hardware(configuration) {
  const block = document.createElement("div");
  block.className = "overview-hardware-grid";
  for (const [label_text, keys] of [
    ["CPU count", ["cpu_count", "physical_cpu_count"]],
    ["Logical CPU count", ["logical_cpu_count", "cpu_logical_count"]],
    ["GPU count", ["gpu_count"]],
    ["GPU type", ["gpu_type", "gpu_name"]],
  ]) {
    const label = document.createElement("span"); label.textContent = label_text;
    const value = document.createElement("span"); value.textContent = local_overview_value(local_first_present(configuration, keys));
    block.append(label, value);
  }
  return block;
}

function local_value_node(value) {
  if (!value || typeof value !== "object") {
    const span = document.createElement("span"); span.textContent = local_overview_value(value); return span;
  }
  const details = document.createElement("details"); details.className = "overview-object-details";
  const summary = document.createElement("summary");
  summary.textContent = Array.isArray(value) ? `[ ${value.length} items ]` : `{ ${Object.keys(value).length} keys }`;
  const pre = document.createElement("pre"); pre.textContent = JSON.stringify(value, null, 2);
  details.append(summary, pre); return details;
}

function local_filter(query) {
  const text = query.trim();
  if (!text) return () => true;
  try { const regex = new RegExp(text, "i"); return value => regex.test(value); }
  catch (_error) { const lowered = text.toLowerCase(); return value => value.toLowerCase().includes(lowered); }
}

function local_render_key_panel(container, title_text, values) {
  container.replaceChildren();
  const heading = document.createElement("div"); heading.className = "overview-panel-heading";
  const title = document.createElement("h3"); title.textContent = title_text;
  const count = document.createElement("span"); count.textContent = `${Object.keys(values).length} keys`;
  heading.append(title, count);
  const search = document.createElement("label"); search.className = "overview-search";
  const glyph = document.createElement("span"); glyph.textContent = "⌕";
  const input = document.createElement("input"); input.type = "search"; input.placeholder = "Search keys with regex";
  search.append(glyph, input);
  const rows = document.createElement("div"); rows.className = "overview-key-rows";
  for (const [key, value] of Object.entries(values).sort(([a], [b]) => a.localeCompare(b))) {
    const row = document.createElement("div"); row.className = "overview-key-row";
    const name = document.createElement("div"); name.className = "overview-key-name"; name.textContent = key;
    const shown = document.createElement("div"); shown.className = "overview-key-value"; shown.appendChild(local_value_node(value));
    row.dataset.searchText = `${key} ${local_overview_value(value)}`; row.append(name, shown); rows.appendChild(row);
  }
  input.addEventListener("input", () => { const accepts = local_filter(input.value); for (const row of rows.children) row.hidden = !accepts(row.dataset.searchText || ""); });
  container.append(heading, search, rows);
}

function local_summary(run, configuration) {
  return {
    artifact_name: run?.artifact_name ?? "—",
    artifact_prefix: local_first_present(configuration, ["artifact_prefix"]),
    comparison_group: local_first_present(configuration, ["comparison_group", "experiment_prefix", "group"]),
    dense_equivalent_parameters: local_first_present(configuration, ["dense_equivalent_parameters", "dense_equivalent_total_parameters"]),
    model_type: local_first_present(configuration, ["model_type"], run?.model_type || "—"),
    persistent_parameters: local_first_present(configuration, ["persistent_parameters"]),
    "train/loss": local_first_present(configuration, ["train/loss", "train_loss", "training_loss"]),
    "val/val_loss": local_first_present(configuration, ["val/val_loss", "validation_loss", "val_loss"]),
  };
}

function local_render_artifacts(container, run) {
  container.replaceChildren();
  const title = document.createElement("h3"); title.textContent = "Artifact Outputs";
  const note = document.createElement("p"); note.className = "overview-artifact-note"; note.textContent = "This run produced these artifacts as outputs. Total: 1.";
  const table = document.createElement("table"); table.className = "overview-artifact-table";
  table.innerHTML = "<thead><tr><th>Type</th><th>Name</th><th>Size</th><th>Consumer count</th></tr></thead>";
  const body = document.createElement("tbody"); const row = document.createElement("tr");
  for (const value of ["local-charts", "charts.sqlite3", format_bytes(run?.database_bytes), "—"]) { const cell = document.createElement("td"); cell.textContent = value; row.appendChild(cell); }
  row.title = run?.run_directory || ""; body.appendChild(row); table.appendChild(body); container.append(title, note, table);
}

function local_render_overview() {
  const run = current_run(); if (!run || !by_id("run_overview_pane")) return;
  const configuration = run.configuration || {};
  const metadata = by_id("overview_metadata"); metadata.replaceChildren();
  local_append_meta(metadata, "Notes", local_first_present(configuration, ["notes", "note"]));
  local_append_meta(metadata, "Tags", local_first_present(configuration, ["tags"]));
  local_append_meta(metadata, "Author", local_first_present(configuration, ["author", "wandb_entity"], local_wandb_owner(run)));
  local_append_meta(metadata, "State", local_state_badge(run));
  local_append_meta(metadata, "Start time", local_overview_timestamp(run.created_at));
  local_append_meta(metadata, "Runtime", local_overview_duration(run));
  local_append_meta(metadata, "Run path", local_run_path(run));
  local_append_meta(metadata, "Hostname", local_first_present(configuration, ["hostname", "host"], run.host_label || "—"));
  local_append_meta(metadata, "OS", local_first_present(configuration, ["os", "platform", "platform_string"]));
  local_append_meta(metadata, "Python version", local_first_present(configuration, ["python_version"]));
  local_append_meta(metadata, "Git repository", local_first_present(configuration, ["git_repository", "repository", "repo"]));
  local_append_meta(metadata, "Git state", local_first_present(configuration, ["git_state", "git_commit", "git_hash", "commit_hash"]));
  local_append_meta(metadata, "Python executable", local_first_present(configuration, ["python_executable"]));
  local_append_meta(metadata, "Command", local_first_present(configuration, ["command", "run_command"]));
  local_append_meta(metadata, "System Hardware", local_hardware(configuration));
  local_append_meta(metadata, "W&B CLI Version", local_first_present(configuration, ["wandb_version", "wandb_cli_version"]));
  local_append_meta(metadata, "Group", local_first_present(configuration, ["comparison_group", "experiment_prefix", "group"]));
  local_append_meta(metadata, "Job Type", local_first_present(configuration, ["job_type"], run.model_type || "—"));
  local_render_key_panel(by_id("overview_config_panel"), "Config", configuration);
  local_render_key_panel(by_id("overview_summary_panel"), "Summary", local_summary(run, configuration));
  local_render_artifacts(by_id("overview_artifact_outputs"), run);
}

function local_apply_detail_tab() {
  const has_run = Boolean(app.current_run_id);
  by_id("run_detail_tabs").hidden = !has_run;
  document.querySelectorAll(".run-detail-tab").forEach(button => {
    const active = button.dataset.detailTab === local_active_detail_tab;
    button.classList.toggle("active", active); button.setAttribute("aria-selected", String(active));
  });
  const charts = local_active_detail_tab === "charts";
  const overview = local_active_detail_tab === "overview";
  by_id("charts_empty").hidden = has_run || !charts;
  by_id("charts_scroll").hidden = !has_run || !charts;
  by_id("run_overview_pane").hidden = !has_run || !overview;
  by_id("run_blank_detail_pane").hidden = !has_run || charts || overview;
  if (overview && has_run) local_render_overview();
  if (charts && has_run) requestAnimationFrame(() => requestAnimationFrame(resize_visible_plots));
}

function local_set_detail_tab(tab_name) {
  if (!local_detail_tabs.includes(tab_name)) return;
  if (app.maximized_chart) restore_maximized_chart();
  local_active_detail_tab = tab_name; local_apply_detail_tab();
}

function local_install_detail_tabs() {
  const toolbar = document.querySelector(".charts-toolbar");
  if (!toolbar || by_id("run_detail_tabs")) return;
  const tabs = document.createElement("nav"); tabs.id = "run_detail_tabs"; tabs.className = "run-detail-tabs"; tabs.setAttribute("role", "tablist");
  for (const name of local_detail_tabs) {
    const button = document.createElement("button"); button.type = "button"; button.className = "run-detail-tab"; button.dataset.detailTab = name; button.setAttribute("role", "tab"); button.textContent = name[0].toUpperCase() + name.slice(1); button.addEventListener("click", () => local_set_detail_tab(name)); tabs.appendChild(button);
  }
  toolbar.insertAdjacentElement("afterend", tabs);
  const overview = document.createElement("section"); overview.id = "run_overview_pane"; overview.className = "run-overview-pane"; overview.hidden = true;
  overview.innerHTML = '<div class="overview-metadata" id="overview_metadata"></div><div class="overview-data-grid"><section class="overview-key-panel" id="overview_config_panel"></section><section class="overview-key-panel" id="overview_summary_panel"></section></div><section class="overview-artifact-outputs" id="overview_artifact_outputs"></section>';
  const blank = document.createElement("section"); blank.id = "run_blank_detail_pane"; blank.className = "run-blank-detail-pane"; blank.hidden = true;
  by_id("charts_pane").append(overview, blank); local_apply_detail_tab();
}

const local_base_render_run_heading = render_run_heading;
render_run_heading = function() { local_base_render_run_heading(); if (by_id("run_detail_tabs")) local_apply_detail_tab(); };
const local_base_render_empty_state = render_empty_state;
render_empty_state = function() { local_base_render_empty_state(); if (by_id("run_detail_tabs")) local_apply_detail_tab(); };
const local_base_select_run = select_run;
select_run = function(run_id, options = {}) { local_active_detail_tab = "charts"; return local_base_select_run(run_id, options); };
local_install_detail_tabs();
// ^^^ THOG
