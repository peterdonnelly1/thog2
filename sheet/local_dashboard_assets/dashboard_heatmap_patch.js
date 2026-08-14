// vvv THOG
"use strict";

// The base dashboard remains deliberately generic. This patch owns the local
// heatmap coordinate remap, fixed per-probe row height, and viewer-only heatmap
// display overrides.

const heatmap_viewer_settings_key = "thog2_local_heatmap_viewer_settings";
const heatmap_probe_row_height_px = 12;
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

function heatmap_abs_limit(default_limit) {
  const settings = heatmap_viewer_settings();
  const override = Number(settings[app.current_run_id]?.abs_limit);
  if (Number.isFinite(override) && override > 0) return override;
  const run_limit = Number(heatmap_run_settings().abs_limit);
  if (Number.isFinite(run_limit) && run_limit > 0) return run_limit;
  return Number.isFinite(default_limit) && default_limit > 0 ? default_limit : 0.05;
}

function save_heatmap_abs_limit(value) {
  if (!app.current_run_id) return;
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) return;
  const settings = heatmap_viewer_settings();
  settings[app.current_run_id] = {
    ...(settings[app.current_run_id] || {}),
    abs_limit: numeric,
  };
  save_json(heatmap_viewer_settings_key, settings);
}

function reset_heatmap_abs_limit() {
  if (!app.current_run_id) return;
  const settings = heatmap_viewer_settings();
  if (settings[app.current_run_id]) {
    delete settings[app.current_run_id].abs_limit;
    if (!Object.keys(settings[app.current_run_id]).length) delete settings[app.current_run_id];
    save_json(heatmap_viewer_settings_key, settings);
  }
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
    const y_min = Number(original_xaxis.range?.[0] ?? 0.5);
    if (Number.isFinite(y_min) && active_layer_trace.y.length) {
      active_layer_trace.x = [0, ...active_layer_trace.x];
      active_layer_trace.y = [y_min, ...active_layer_trace.y];
      active_layer_trace.customdata = [step_values[0], ...active_layer_trace.customdata];
    }
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
    automargin: false,
  };
  prepared.layout.yaxis = {
    ...original_xaxis,
    title: {text: "step"},
  };
  for (const axis of [prepared.layout.xaxis, prepared.layout.yaxis]) {
    delete axis.scaleanchor;
    delete axis.scaleratio;
    delete axis.constrain;
    delete axis.constraintoward;
  }

  prepared.layout.margin = {...(prepared.layout.margin || {}), b: 76};

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
    return {
      width: Math.max(shell_width + 1, 190 + column_count * 34),
      height: Math.max(
        shell_height + 1,
        heatmap_plot_chrome_height_px + Math.max(1, probe_count) * heatmap_probe_row_height_px,
      ),
    };
  }
  return {
    width: Math.max(shell_width + 1, 620),
    height: Math.max(shell_height + 1, 320),
  };
}

function figure_for_chart(chart_name) {
  if (!app.figures) return null;
  return chart_name === "heatmap" ? app.figures.heatmap : app.figures.depth?.[chart_name];
}

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
  grid.append(label, input);
  return input;
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
  const active = Boolean(app.current_run_id);
  by_id("heatmap_setting_abs_limit").disabled = !active;
  by_id("heatmap_setting_reset").disabled = !active;
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

  make_heatmap_settings_row(
    grid,
    "instrumentation__delta_loss_v_layer_heatmap",
    "heatmap_setting_mode",
    {readonly: true},
  );
  make_heatmap_settings_row(
    grid,
    "instrumentation__delta_loss_v_layer_heatmap__destination",
    "heatmap_setting_destination",
    {readonly: true},
  );
  make_heatmap_settings_row(
    grid,
    "instrumentation__delta_loss_v_layer_heatmap_linear",
    "heatmap_setting_linear",
    {readonly: true},
  );
  const abs_limit = make_heatmap_settings_row(
    grid,
    "instrumentation__delta_loss_v_layer_heatmap_abs_limit",
    "heatmap_setting_abs_limit",
    {type: "number", min: 0.000000001, step: 0.01},
  );

  const note = document.createElement("p");
  note.className = "heatmap-settings-note";
  note.textContent = (
    "abs_limit is a live viewer override. Mode, destination and linear max-step are capture/routing controls from the selected run; the viewer cannot recreate probes that were not recorded."
  );

  const actions = document.createElement("div");
  actions.className = "heatmap-settings-actions";
  const reset = document.createElement("button");
  reset.id = "heatmap_setting_reset";
  reset.type = "button";
  reset.textContent = "Reset abs limit to run value";
  actions.appendChild(reset);
  section.append(heading, grid, note, actions);
  settings_content.appendChild(section);

  let render_timer = null;
  const apply_abs_limit = () => {
    const numeric = Number(abs_limit.value);
    if (!Number.isFinite(numeric) || numeric <= 0) return;
    save_heatmap_abs_limit(numeric);
    clearTimeout(render_timer);
    render_timer = setTimeout(() => {
      if (app.figures && app.current_run_id) render_figures();
    }, 120);
  };
  abs_limit.addEventListener("input", apply_abs_limit);
  reset.addEventListener("click", () => {
    reset_heatmap_abs_limit();
    sync_heatmap_settings_panel();
    if (app.figures && app.current_run_id) render_figures();
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

  install_heatmap_settings_panel();
}

transpose_heatmap = transpose_heatmap_relative;

render_plot = async function(mount, figure, chart_name) {
  const prepared = prepare_figure(figure, chart_name);
  const dimensions = plot_mount_dimensions(mount, chart_name, figure);
  mount.style.width = `${Math.round(dimensions.width)}px`;
  mount.style.height = `${Math.round(dimensions.height)}px`;
  prepared.layout.autosize = false;
  prepared.layout.width = Math.round(dimensions.width);
  prepared.layout.height = Math.round(dimensions.height);
  if (mount.dataset.plotReady === "true") {
    await Plotly.react(mount, prepared.data, prepared.layout, plot_config);
  } else {
    mount.replaceChildren();
    await Plotly.newPlot(mount, prepared.data, prepared.layout, plot_config);
    mount.dataset.plotReady = "true";
  }
};

resize_plot_in_card = function(card) {
  const mount = card.querySelector(".plot-mount");
  if (!mount || mount.dataset.plotReady !== "true") return;
  const chart_name = card.dataset.chart;
  const figure = figure_for_chart(chart_name);
  if (!figure) return;
  const dimensions = plot_mount_dimensions(mount, chart_name, figure);
  mount.style.width = `${Math.round(dimensions.width)}px`;
  mount.style.height = `${Math.round(dimensions.height)}px`;
  Plotly.relayout(mount, {
    width: Math.round(dimensions.width),
    height: Math.round(dimensions.height),
  });
};

install_dashboard_ui_patch();

if (app.figures && app.current_run_id) {
  queueMicrotask(() => render_figures());
}
// ^^^ THOG
