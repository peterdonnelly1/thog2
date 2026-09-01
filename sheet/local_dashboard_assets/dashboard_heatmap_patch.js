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
  const preview = Number(
    app.chart_settings_render_override?.chart_name === "heatmap"
      ? app.chart_settings_render_override.settings?.heatmap_row_height
      : NaN,
  );
  if (Number.isFinite(preview)) {
    return Math.min(
      heatmap_maximum_probe_row_height_px,
      Math.max(heatmap_minimum_probe_row_height_px, preview),
    );
  }
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
        Number(cell),
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
        ? `<b>L=${latest_active_layer_count}</b>`
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
  if (app.dynamic_chart_figures?.[chart_name]) return app.dynamic_chart_figures[chart_name];
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
  const actions = maximize?.parentElement;
  if (!header || !maximize || !actions || by_id("heatmap_vertical_scale")) return;

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
  actions.insertBefore(control, maximize);

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
  heading.textContent = "Heatmap - Loss vs Counterfactual Layer Count";
  const grid = document.createElement("div");
  grid.className = "heatmap-settings-grid";

  make_heatmap_settings_row(grid, "instrumentation__delta_loss_v_layer_heatmap", "heatmap_setting_mode", {readonly: true});
  make_heatmap_settings_row(grid, "instrumentation__delta_loss_v_layer_heatmap__destination", "heatmap_setting_destination", {readonly: true});
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
    "abs_limit and vertical pixels/step are live viewer overrides. Mode and destination are capture/routing controls from the selected run."
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
// Files is the navigable instra/W&B browser; Logs and Artifacts remain placeholders.
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
  const end = is_active_run_state(run?.run_state) && display_run_state(run) !== "timed_out"
    ? Date.now()
    : Date.parse(run?.updated_at || "");
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
  const icon_state = ["preparing", "recording", "monitoring"].includes(state)
    ? "running"
    : state === "stopped"
      ? "finished"
      : state === "data_lost"
        ? "data_lost"
        : ["running", "finished", "timed_out"].includes(state) ? state : "unknown";
  icon.appendChild(icon_svg(icon_state));
  badge.append(icon, document.createTextNode(format_run_state(state)));
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

function local_dense_snapshot_metadata(configuration) {
  return configuration?.lifecycle?.dense_snapshot_baselining
    || configuration?.parameter_report?.dense_snapshot_baselining
    || configuration?.dense_snapshot_baselining || {};
}

function local_dense_snapshot_details(configuration) {
  const metadata = local_dense_snapshot_metadata(configuration);
  const used = metadata.effective_initialisation === "dense_snapshot"
    || Boolean(configuration.initialise_from_dense_snapshot);
  if (!used) return {filename: "-", parameters: null};
  const path = String(metadata.snapshot_path || configuration.initialise_from_dense_snapshot || "");
  let source = metadata.source_hyperparameters;
  if (!source) {
    // Old source runs already retain their complete configuration in chart storage.
    const source_run = (app.runs || []).find(candidate => {
      const candidate_metadata = local_dense_snapshot_metadata(candidate.configuration || {});
      return candidate_metadata.effective_initialisation === "ordinary_dense_initialisation"
        && metadata.tensor_payload_hash && metadata.compatibility_hash
        && candidate_metadata.tensor_payload_hash === metadata.tensor_payload_hash
        && candidate_metadata.compatibility_hash === metadata.compatibility_hash;
    });
    if (source_run) {
      source = {...source_run.configuration};
      delete source.parameter_report;
      delete source.lifecycle;
    }
  }
  const physical = metadata.snapshot_hyperparameters || {};
  const parameters = {...(source || {})};
  for (const [key, value] of Object.entries(physical)) parameters[`snapshot.${key}`] = value;
  if (!source) parameters.source_hyperparameters = "Not recorded with this snapshot; source run configuration is unavailable.";
  return {filename: path.split(/[\\/]/).pop() || "-", parameters};
}

function local_summary(run, configuration) {
  return {
    dense_baseline_snapshot: local_dense_snapshot_details(configuration).filename,
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
  const snapshot = local_dense_snapshot_details(configuration);
  const snapshot_panel = by_id("overview_snapshot_panel");
  snapshot_panel.hidden = snapshot.parameters === null;
  local_render_key_panel(snapshot_panel, "Dense baseline snapshot hyperparameters", snapshot.parameters || {});
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
  const files = local_active_detail_tab === "files";
  by_id("charts_empty").hidden = has_run || !charts;
  by_id("charts_scroll").hidden = !has_run || !charts;
  by_id("run_overview_pane").hidden = !has_run || !overview;
  by_id("files_workspace").hidden = !has_run || !files;
  by_id("run_blank_detail_pane").hidden = !has_run || charts || overview || files;
  if (overview && has_run) local_render_overview();
  if (files && has_run) render_files();
  if (charts && has_run) requestAnimationFrame(() => requestAnimationFrame(resize_visible_plots));
}

function local_set_detail_tab(tab_name) {
  if (!local_detail_tabs.includes(tab_name)) return;
  if (app.maximized_chart) restore_maximized_chart();
  local_active_detail_tab = tab_name; local_apply_detail_tab();
  if (tab_name === "files") refresh_files();
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
  overview.innerHTML = '<div class="overview-metadata" id="overview_metadata"></div><div class="overview-data-grid"><section class="overview-key-panel" id="overview_config_panel"></section><section class="overview-key-panel" id="overview_summary_panel"></section><section class="overview-key-panel" id="overview_snapshot_panel"></section></div>';
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

// vvv THOG literal heatmap row pitch/fitted width plus a robust signed-log trajectory view biased toward small coefficients
window.addEventListener("load", () => {
  const trajectory_chart_names_refined = new Set([
    "attn_q_head_N",
    "attn_k_head_N",
    "attn_v_head_N",
    "attn_out_head_N",
    "mlp_up",
    "mlp_down",
  ]);
  const trajectory_scale_settings_key_refined = "thog2_local_trajectory_scale_modes";
  const trajectory_scale_mode_refined = chart_name => (
    load_json(trajectory_scale_settings_key_refined, {})[chart_name] === "log" ? "log" : "linear"
  );

  const quantile = (sorted_values, fraction) => {
    if (!sorted_values.length) return 0;
    const position = Math.max(0, Math.min(sorted_values.length - 1, fraction * (sorted_values.length - 1)));
    const lower = Math.floor(position);
    const upper = Math.ceil(position);
    if (lower === upper) return sorted_values[lower];
    const interpolation = position - lower;
    return sorted_values[lower] * (1 - interpolation) + sorted_values[upper] * interpolation;
  };

  const signed_log_refined = (value, linear_threshold) => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric) || numeric === 0) return numeric;
    return Math.sign(numeric) * Math.log10(1 + Math.abs(numeric) / linear_threshold);
  };

  const trajectory_tick_label_refined = value => {
    const numeric = Number(value);
    const magnitude = Math.abs(numeric);
    if (!Number.isFinite(magnitude)) return "";
    if (magnitude === 0) return "0";
    if (magnitude >= 1000 || magnitude < 0.001) return numeric.toExponential(0);
    return String(Number(numeric.toPrecision(3)));
  };

  const trajectory_log_figure_refined = figure => {
    const transformed = clone_figure(figure);
    const finite_values = [];
    for (const trace of transformed.data || []) {
      if (!Array.isArray(trace.y)) continue;
      for (const value of trace.y) {
        const numeric = Number(value);
        if (Number.isFinite(numeric)) finite_values.push(numeric);
      }
    }
    const magnitudes = finite_values
      .map(value => Math.abs(value))
      .filter(value => value > 0)
      .sort((left, right) => left - right);
    if (!magnitudes.length) return transformed;

    // The lower tail, not the largest outlier, defines the symlog knee. Using a quarter
    // of P10 deliberately gives very small coefficients appreciable visual room while
    // still retaining arbitrarily large excursions on the same axis.
    const lower_tail = quantile(magnitudes, 0.10);
    const linear_threshold = Math.max(1e-15, lower_tail * 0.25);
    const minimum_exponent = Math.floor(Math.log10(magnitudes[0]));
    const maximum_exponent = Math.ceil(Math.log10(magnitudes[magnitudes.length - 1]));

    for (const trace of transformed.data || []) {
      if (!Array.isArray(trace.y)) continue;
      const original_y = trace.y.map(value => Number(value));
      trace.customdata = original_y;
      trace.y = original_y.map(value => signed_log_refined(value, linear_threshold));
      if (typeof trace.hovertemplate === "string") {
        trace.hovertemplate = trace.hovertemplate
          .replaceAll("%{y:.7g}", "%{customdata:.7g}")
          .replaceAll("%{y}", "%{customdata}");
      }
    }

    const exponent_span = Math.max(0, maximum_exponent - minimum_exponent);
    const exponent_step = Math.max(1, Math.ceil((exponent_span + 1) / 7));
    const tick_magnitudes = [];
    for (let exponent = minimum_exponent; exponent <= maximum_exponent; exponent += exponent_step) {
      tick_magnitudes.push(10 ** exponent);
    }
    if (tick_magnitudes[tick_magnitudes.length - 1] < 10 ** maximum_exponent) {
      tick_magnitudes.push(10 ** maximum_exponent);
    }
    const negative_magnitudes = [...tick_magnitudes].reverse().map(value => -value);
    transformed.layout = transformed.layout || {};
    transformed.layout.yaxis = {
      ...(transformed.layout.yaxis || {}),
      type: "linear",
      tickmode: "array",
      tickvals: [
        ...negative_magnitudes.map(value => signed_log_refined(value, linear_threshold)),
        0,
        ...tick_magnitudes.map(value => signed_log_refined(value, linear_threshold)),
      ],
      ticktext: [
        ...negative_magnitudes.map(trajectory_tick_label_refined),
        "0",
        ...tick_magnitudes.map(trajectory_tick_label_refined),
      ],
      title: {text: "weight value · signed log"},
      automargin: true,
    };
    return transformed;
  };

  const base_transpose_heatmap_refined = transpose_heatmap;
  transpose_heatmap = function(prepared) {
    base_transpose_heatmap_refined(prepared);
    const heatmap_trace = (prepared.data || []).find(trace => trace.type === "heatmap");
    if (!heatmap_trace) return;
    heatmap_trace.hovertemplate = (
      "step=%{customdata[0]}<br>"
      + "layer count (abs) = %{customdata[1]}<br>"
      + "layer count (rel) = %{customdata[2]}<br>"
      + "Δloss=%{customdata[3]:.8f}<extra></extra>"
    );
  };

  const base_plot_mount_dimensions_refined = plot_mount_dimensions;
  plot_mount_dimensions = function(mount, chart_name, figure) {
    if (chart_name !== "heatmap") return base_plot_mount_dimensions_refined(mount, chart_name, figure);
    const shell = mount.closest(".plot-shell");
    const viewport = shell?.querySelector(":scope > .heatmap-inner-viewport");
    const viewport_width = Math.max(1, viewport?.clientWidth || shell?.clientWidth || 1);
    const probe_count = Math.max(1, heatmap_probe_count(figure));
    const row_height = heatmap_probe_row_height_px();
    return {
      // Fit every candidate-offset column to the current heatmap viewport. This removes
      // horizontal travel in the initial/restored view while retaining the H scroll rail.
      width: Math.max(1, viewport_width - 1),
      // Plotly's heatmap domain is the total figure height minus the fixed 18px/76px
      // margins below. Therefore every probe row is literally row_height CSS pixels.
      height: 18 + 76 + probe_count * row_height,
    };
  };

  render_plot = async function(mount, figure, chart_name) {
    if (chart_name === "heatmap") ensure_plot_scroll_canvas(mount);
    const shown_figure = (
      trajectory_chart_names_refined.has(chart_name)
      && trajectory_scale_mode_refined(chart_name) === "log"
    ) ? trajectory_log_figure_refined(figure) : figure;
    const prepared = prepare_figure(shown_figure, chart_name);
    const dimensions = plot_mount_dimensions(mount, chart_name, figure);
    set_plot_dimensions(mount, dimensions);
    prepared.layout.autosize = false;
    prepared.layout.width = Math.round(dimensions.width);
    prepared.layout.height = Math.round(dimensions.height);
    if (chart_name === "heatmap") {
      // Keep enough real top margin for the mirrored axis title and its tick row.
      // This is the last layout write before Plotly receives the figure.
      prepared.layout.margin = {...(prepared.layout.margin || {}), t: 104, b: 76};
      prepared.layout.yaxis = {...(prepared.layout.yaxis || {}), domain: [0, 1], automargin: false};
    }
    if (mount.dataset.plotReady === "true") {
      await Plotly.react(mount, prepared.data, prepared.layout, fixed_plot_config);
    } else {
      mount.replaceChildren();
      await Plotly.newPlot(mount, prepared.data, prepared.layout, fixed_plot_config);
      mount.dataset.plotReady = "true";
    }
    if (chart_name === "heatmap") sync_heatmap_scale_control();
  };

  const base_restore_maximized_chart_refined = restore_maximized_chart;
  restore_maximized_chart = function() {
    base_restore_maximized_chart_refined();
    requestAnimationFrame(() => requestAnimationFrame(() => {
      const viewport = document.querySelector('.chart-card[data-chart="heatmap"] .heatmap-inner-viewport');
      if (viewport) viewport.scrollLeft = 0;
      const card = document.querySelector('.chart-card[data-chart="heatmap"]');
      if (card && card.offsetParent !== null) resize_plot_in_card(card);
    }));
  };

  if (app.figures && app.current_run_id) queueMicrotask(() => render_figures());
});
// ^^^ THOG

// vvv THOG avoid expensive trajectory redraws on heatmap-only revisions and use WebGL for dense trajectory history without dropping any points
window.addEventListener("load", () => {
  const trajectory_chart_names_fast = [
    "attn_q_head_N",
    "attn_k_head_N",
    "attn_v_head_N",
    "attn_out_head_N",
    "mlp_up",
    "mlp_down",
  ];
  const trajectory_scale_settings_key_fast = "thog2_local_trajectory_scale_modes";
  let rendered_run_id = null;
  let rendered_heatmap_revision = null;
  let rendered_depth_revision = null;
  let rendered_heatmap_viewer_signature = null;
  let rendered_depth_viewer_signature = null;

  const base_render_plot_fast = render_plot;
  render_plot = async function(mount, figure, chart_name) {
    if (!trajectory_chart_names_fast.includes(chart_name)) {
      return base_render_plot_fast(mount, figure, chart_name);
    }
    const webgl_figure = {
      ...figure,
      data: (figure?.data || []).map(trace => (
        trace?.type === "scatter" ? {...trace, type: "scattergl"} : trace
      )),
    };
    return base_render_plot_fast(mount, webgl_figure, chart_name);
  };

  const heatmap_viewer_signature = () => JSON.stringify({
    colour: colour_for_run(app.current_run_id),
    settings: heatmap_settings_for_current_run(),
    chart: stored_chart_settings("heatmap"),
  });
  const depth_viewer_signature = () => JSON.stringify({
    colour: colour_for_run(app.current_run_id),
    scales: load_json(trajectory_scale_settings_key_fast, {}),
    charts: Object.fromEntries(
      trajectory_chart_names_fast.map(chart_name => [chart_name, stored_chart_settings(chart_name)]),
    ),
  });

  render_figures = async function() {
    if (!app.figures || !app.current_run_id) return;
    const status = app.current_status || current_run();
    const heatmap_revision = JSON.stringify([
      status?.heatmap_count ?? null,
      status?.heatmap_maximum_update ?? null,
    ]);
    const depth_revision = JSON.stringify([
      status?.depth_snapshot_count ?? null,
      status?.depth_maximum_update ?? null,
    ]);
    const heatmap_signature = heatmap_viewer_signature();
    const depth_signature = depth_viewer_signature();

    if (rendered_run_id !== app.current_run_id) {
      rendered_run_id = app.current_run_id;
      rendered_heatmap_revision = null;
      rendered_depth_revision = null;
      rendered_heatmap_viewer_signature = null;
      rendered_depth_viewer_signature = null;
    }

    const heatmap_changed = (
      rendered_heatmap_revision !== heatmap_revision
      || rendered_heatmap_viewer_signature !== heatmap_signature
    );
    const depth_changed = (
      rendered_depth_revision !== depth_revision
      || rendered_depth_viewer_signature !== depth_signature
    );

    if (app.figures.heatmap && heatmap_changed) {
      by_id("heatmap_placeholder").hidden = true;
      await render_plot(by_id("heatmap_plot"), app.figures.heatmap, "heatmap");
      rendered_heatmap_revision = heatmap_revision;
      rendered_heatmap_viewer_signature = heatmap_signature;
    }
    by_id("heatmap_card_detail").textContent = status
      ? `${format_integer(status.heatmap_count)} probes · latest step ${format_integer(status.heatmap_maximum_update)}`
      : "Layer-count probes";

    if (depth_changed) {
      for (const chart_name of trajectory_chart_names_fast) {
        const figure = app.figures.depth?.[chart_name];
        const detail = by_id(`${chart_name}_detail`);
        if (!figure) continue;
        by_id(`${chart_name}_placeholder`).hidden = true;
        if (detail) {
          detail.textContent = `${format_integer(status?.depth_snapshot_count)} retained snapshots · latest step ${format_integer(status?.depth_maximum_update)}`;
        }
        await render_plot(by_id(`${chart_name}_plot`), figure, chart_name);
      }
      rendered_depth_revision = depth_revision;
      rendered_depth_viewer_signature = depth_signature;
    }
  };

  if (app.figures && app.current_run_id) queueMicrotask(() => render_figures());
});
// ^^^ THOG

// vvv THOG auto-scale each heatmap colour band from the strongest currently retained value while preserving manual limits for later restoration
window.addEventListener("load", () => {
  setTimeout(() => {
    const auto_setting_name = "auto_colour_saturation";
    const auto_enabled = () => heatmap_settings_for_current_run()[auto_setting_name] === true;
    const clamp_01 = value => Math.max(0, Math.min(1, value));
    const manual_limit_ids = [
      "heatmap_setting_negative_limit",
      "heatmap_setting_blue_limit",
      "heatmap_setting_yellow_limit",
      "heatmap_setting_positive_limit",
    ];

    const auto_limits = heatmap_trace => {
      const limits = {green: 0, blue: 0, yellow: 0, red: 0};
      for (const row of heatmap_trace.customdata || []) {
        if (!Array.isArray(row)) continue;
        for (const cell of row) {
          const delta = Number(Array.isArray(cell) ? cell[3] : NaN);
          if (!Number.isFinite(delta)) continue;
          if (delta <= -1.0) limits.yellow = Math.max(limits.yellow, Math.abs(delta));
          else if (delta <= -0.1) limits.blue = Math.max(limits.blue, Math.abs(delta));
          else if (delta < 0) limits.green = Math.max(limits.green, Math.abs(delta));
          else if (delta > 0) limits.red = Math.max(limits.red, delta);
        }
      }
      return limits;
    };

    const auto_band_value = (delta, limits) => {
      if (delta <= -1.0) {
        const intensity = limits.yellow > 0 ? clamp_01(Math.abs(delta) / limits.yellow) : 1;
        return -0.76 - 0.24 * intensity;
      }
      if (delta <= -0.1) {
        const intensity = limits.blue > 0 ? clamp_01(Math.abs(delta) / limits.blue) : 1;
        return -0.51 - 0.23 * intensity;
      }
      if (delta < 0) {
        const intensity = limits.green > 0 ? clamp_01(Math.abs(delta) / limits.green) : 1;
        return -0.01 - 0.48 * intensity;
      }
      if (delta > 0) {
        const intensity = limits.red > 0 ? clamp_01(delta / limits.red) : 1;
        return 0.01 + 0.99 * intensity;
      }
      return 0;
    };

    const format_auto_limit = (value, sign) => (
      value > 0 ? `${sign}${Number(value).toPrecision(3)}` : "—"
    );

    const base_transpose_heatmap_auto = transpose_heatmap;
    transpose_heatmap = function(prepared) {
      base_transpose_heatmap_auto(prepared);
      if (!auto_enabled()) return;
      const heatmap_trace = (prepared.data || []).find(trace => trace.type === "heatmap");
      if (!heatmap_trace) return;
      const limits = auto_limits(heatmap_trace);
      const customdata = Array.isArray(heatmap_trace.customdata) ? heatmap_trace.customdata : [];
      heatmap_trace.z = (heatmap_trace.z || []).map((row, row_index) => (
        Array.isArray(row)
          ? row.map((_value, column_index) => {
              const delta = Number(customdata[row_index]?.[column_index]?.[3]);
              return Number.isFinite(delta) ? auto_band_value(delta, limits) : null;
            })
          : row
      ));
      heatmap_trace.zmin = -1;
      heatmap_trace.zmax = 1;
      heatmap_trace.zmid = 0;
      heatmap_trace.colorbar = {
        ...(heatmap_trace.colorbar || {}),
        tickmode: "array",
        tickvals: [-1, -0.76, -0.74, -0.51, -0.49, 0, 1],
        ticktext: [
          `auto yellow ${format_auto_limit(limits.yellow, "−")}`,
          "yellow ≤ −1.0",
          `auto blue ${format_auto_limit(limits.blue, "−")}`,
          "blue ≤ −0.1",
          `auto green ${format_auto_limit(limits.green, "−")}`,
          "0",
          `auto red ${format_auto_limit(limits.red, "+")}`,
        ],
        title: "Δloss bands · auto",
      };
    };

    const style = document.createElement("style");
    style.textContent = `
      .chart-card[data-chart="heatmap"] .ytick:last-of-type text {
        transform: translate(-5px, 4px) !important;
      }
      #heatmap_setting_auto_colour {
        width: 18px !important;
        height: 18px !important;
        min-width: 18px !important;
        justify-self: start;
        accent-color: #1995ad;
      }
    `;
    document.head.appendChild(style);

    const settings_grid = document.querySelector("#heatmap_settings_section .heatmap-settings-grid");
    let checkbox = by_id("heatmap_setting_auto_colour");
    if (settings_grid && !checkbox) {
      const label = document.createElement("label");
      label.htmlFor = "heatmap_setting_auto_colour";
      label.textContent = "viewer auto colour saturation";
      checkbox = document.createElement("input");
      checkbox.id = "heatmap_setting_auto_colour";
      checkbox.type = "checkbox";
      checkbox.title = "Independently scale red, green, blue and yellow so each band's strongest retained value is fully saturated";
      settings_grid.append(label, checkbox);
    }

    const sync_auto_controls = () => {
      if (!checkbox) return;
      const active = Boolean(app.current_run_id);
      checkbox.disabled = !active;
      checkbox.checked = active && auto_enabled();
      for (const id of manual_limit_ids) {
        const input = by_id(id);
        if (input) input.disabled = !active || checkbox.checked;
      }
    };

    const rerender_heatmap_auto = async () => {
      const figure = figure_for_chart("heatmap");
      const mount = by_id("heatmap_plot");
      if (figure && mount && app.current_run_id) await render_plot(mount, figure, "heatmap");
    };

    checkbox?.addEventListener("change", async () => {
      save_heatmap_viewer_setting(auto_setting_name, checkbox.checked);
      sync_auto_controls();
      await rerender_heatmap_auto();
    });
    by_id("settings_nav")?.addEventListener("click", sync_auto_controls);

    const settings_note = document.querySelector("#heatmap_settings_section .heatmap-settings-note");
    if (settings_note && !settings_note.dataset.autoColourNote) {
      settings_note.dataset.autoColourNote = "true";
      settings_note.textContent += " Auto colour saturation independently rescales each colour band from the strongest currently retained value on every heatmap update; manual limits are preserved while Auto is enabled.";
    }

    sync_auto_controls();
    if (auto_enabled()) rerender_heatmap_auto();
  }, 0);
});
// ^^^ THOG
