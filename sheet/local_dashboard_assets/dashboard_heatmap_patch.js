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
