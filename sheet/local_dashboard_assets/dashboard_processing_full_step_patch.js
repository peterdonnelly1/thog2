// vvv THOG full-step Processing throughput range and lightweight PREMAT outcome overlays
"use strict";

const processing_render_throughput_before_full_step_patch = processing_render_throughput;
processing_render_throughput = async function(payload) {
  const runs = processing_throughput_workspace_runs();
  const resolved = await Promise.all(runs.map(async run => ({
    run,
    run_id: String(run_identifier(run)),
    rows: await processing_throughput_rows_for_run(run, payload),
  })));
  const populated = resolved.filter(entry => entry.rows.length);
  const traces = populated.map(entry => {
    const name = String(entry.run.artifact_name || entry.run.run_name || entry.run_id);
    const colour = colour_for_run(entry.run_id);
    return {
      type: "scatter",
      mode: entry.rows.length === 1 ? "markers" : "lines",
      name,
      meta: {instra_workspace_run_id: entry.run_id},
      x: entry.rows.map(row => Number(row.optimizer_update)),
      y: entry.rows.map(row => Number(row.tokens_per_second)),
      line: {color: colour, width: 2.4},
      marker: {color: colour},
      hovertemplate: "update %{x}<br>%{y:,.0f} tok/s<extra>%{fullData.name}</extra>",
    };
  });
  const maximum_points = Math.max(0, ...populated.map(entry => entry.rows.length));
  const updates = populated.flatMap(entry => entry.rows.map(row => Number(row.optimizer_update))).filter(Number.isFinite);
  const maximum_update = updates.length ? Math.max(...updates) : null;
  const workspace = app.workspace_mode === true;
  await processing_plot("processing_throughput_plot", traces, {
    margin: {l: 72, r: 24, t: 12, b: 54},
    xaxis: {
      title: "optimizer update",
      dtick: maximum_points <= 20 ? 1 : undefined,
      range: Number.isFinite(maximum_update) && maximum_update > 0 ? [0, maximum_update] : undefined,
    },
    yaxis: {title: "tokens / second", rangemode: "tozero", separatethousands: true},
    showlegend: traces.length > 1,
    legend: {orientation: "h", y: 1.10},
    annotations: traces.length ? [] : [{
      text: workspace ? "No retained tok/s samples for visible Workspace runs" : "No retained tok/s samples for this run",
      showarrow: false,
      xref: "paper", yref: "paper", x: 0.5, y: 0.5,
    }],
  }, plot_config);
};

const processing_update_timing_phase_traces_before_full_step_patch = processing_update_timing_phase_traces;

function processing_premat_outcome_rows(entry) {
  return (entry.timing.premat_microstep_outcomes || []).filter(row => (
    Number.isFinite(Number(row.micro_step))
    && Number.isFinite(Number(row.full_hits))
    && Number.isFinite(Number(row.partial_hits))
    && Number.isFinite(Number(row.misses))
  ));
}

function processing_microstep_right_edge_ms(entry, micro_step) {
  const ends = (entry.timing.timeline || [])
    .filter(row => Number(row.micro_step) === Number(micro_step))
    .map(row => Number(row.host_end_ms))
    .filter(Number.isFinite);
  return ends.length ? Math.max(...ends) : null;
}

function processing_premat_outcome_traces(entries) {
  const maximum_host_ms = Math.max(
    1,
    ...entries.map(entry => Number(entry.timing.official_update_ms ?? entry.timing.host_update_ms)).filter(Number.isFinite),
  );
  const gap_ms = Math.max(24, maximum_host_ms * 0.025);
  const specifications = [
    {key: "full_hits", name: "Full hit", colour: "#238451", offset: 2},
    {key: "partial_hits", name: "Partial hit", colour: "#ee9b31", offset: 1},
    {key: "misses", name: "Miss", colour: "#ef3340", offset: 0},
  ];
  const traces = specifications.map(specification => ({
    type: "scatter",
    mode: "markers+text",
    name: specification.name,
    legendgroup: "premat_outcomes",
    x: [],
    y: [],
    text: [],
    customdata: [],
    textposition: "middle center",
    textfont: {color: "#ffffff", size: 10},
    marker: {
      symbol: "circle",
      size: 23,
      color: specification.colour,
      line: {color: "#ffffff", width: 1},
    },
    hovertemplate: "%{customdata[0]}<br>microstep %{customdata[1]}<br>%{fullData.name}: %{text}<extra></extra>",
    cliponaxis: false,
  }));

  for (let lane = 0; lane < entries.length; lane += 1) {
    const entry = entries[lane];
    const rows = processing_premat_outcome_rows(entry);
    if (!rows.length) continue;
    for (const row of rows) {
      const right_edge = processing_microstep_right_edge_ms(entry, row.micro_step);
      if (!Number.isFinite(right_edge)) continue;
      specifications.forEach((specification, index) => {
        traces[index].x.push(Math.max(0, right_edge - specification.offset * gap_ms));
        traces[index].y.push(lane + 0.39);
        traces[index].text.push(String(Number(row[specification.key])));
        traces[index].customdata.push([
          processing_update_timing_experiment_label(entry),
          Number(row.micro_step),
        ]);
      });
    }
  }

  const populated = traces.filter(trace => trace.x.length);
  if (!populated.length) return [];
  const spacer = {
    type: "scatter",
    mode: "markers",
    name: "\u00a0\u00a0\u00a0\u00a0",
    legendgroup: "premat_outcome_gap",
    x: [null],
    y: [null],
    marker: {size: 1, color: "rgba(0,0,0,0)"},
    hoverinfo: "skip",
    showlegend: true,
  };
  return [spacer, ...populated];
}

processing_update_timing_phase_traces = function(entries) {
  return [
    ...processing_update_timing_phase_traces_before_full_step_patch(entries),
    ...processing_premat_outcome_traces(entries),
  ];
};
// ^^^ THOG
