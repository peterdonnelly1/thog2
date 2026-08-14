// vvv THOG
"use strict";

// The base dashboard remains deliberately generic.  This patch owns the local
// heatmap coordinate remap and the native-size Plotly canvases that make each
// chart card a genuine scroll viewport.

function signed_layer_offset(value) {
  const offset = Number(value);
  if (!Number.isFinite(offset) || offset === 0) return "0";
  return offset > 0 ? `+${offset}` : String(offset);
}

function relative_heatmap_bounds(figure) {
  const heatmap_trace = (figure?.data || []).find(trace => trace.type === "heatmap");
  const active_layer_trace = (figure?.data || []).find(
    trace => trace !== heatmap_trace && Array.isArray(trace.x) && Array.isArray(trace.y)
  );
  const candidate_layers = (heatmap_trace?.y || []).map(Number).filter(Number.isFinite);
  const active_layers = (active_layer_trace?.y || []).map(Number).filter(Number.isFinite);
  if (!candidate_layers.length || !active_layers.length) return {minimum: 0, maximum: 0};
  return {
    minimum: Math.min(...candidate_layers) - Math.max(...active_layers),
    maximum: Math.max(...candidate_layers) - Math.min(...active_layers),
  };
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
  const original_z = Array.isArray(heatmap_trace.z) ? heatmap_trace.z : [];
  const original_customdata = Array.isArray(heatmap_trace.customdata) ? heatmap_trace.customdata : [];
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
    for (const candidate_layers_at_probe of candidate_layers) {
      if (!Number.isFinite(candidate_layers_at_probe)) continue;
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
      if (!Number.isFinite(candidate_layers_at_probe)) continue;
      const offset = candidate_layers_at_probe - active_layers_at_probe;
      const destination = offset_index.get(offset);
      if (destination === undefined) continue;
      relative_z[probe_index][destination] = Array.isArray(original_z[candidate_index])
        ? original_z[candidate_index][probe_index]
        : null;
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
    title: {text: "candidate layer-count offset from active layer count", standoff: 46},
    range: [minimum_offset - 0.5, maximum_offset + 0.5],
    tickmode: "array",
    tickvals: offsets,
    ticktext: offsets.map(signed_layer_offset),
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
    return {
      width: Math.max(shell_width, 190 + column_count * 34),
      height: Math.max(shell_height, 760),
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

if (app.figures && app.current_run_id) {
  queueMicrotask(() => render_figures());
}
// ^^^ THOG
