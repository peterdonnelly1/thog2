"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const visible = new Set(["run-a", "run-b"]);
const colours = {"run-a": "#ef4444", "run-b": "#2563eb"};
const saved_heatmap_settings = {};
const save_button_listeners = [];
let gradient_enabled = false;
const elements = {
  heatmap_chart_group: {hidden: false, setAttribute() {}},
  save_chart_settings: {
    addEventListener: (_name, callback, options) => save_button_listeners.push({callback, capture: options === true}),
  },
  chart_settings_overlay: {hidden: false},
  chart_heatmap_probe_count: {value: "37"},
  chart_heatmap_window_mode: {value: "rolling"},
  chart_heatmap_y_display_mode: {value: "steps"},
  chart_heatmap_delta_mode: {value: "percent"},
  chart_heatmap_auto_colour: {checked: true},
  chart_heatmap_abs_limit: {value: "0.05"},
  chart_heatmap_green_limit: {value: "0.05"},
  chart_heatmap_blue_limit: {value: "1"},
  chart_heatmap_yellow_limit: {value: "2"},
  chart_heatmap_red_limit: {value: "0.05"},
};
const context = {
  console,
  JSON,
  Math,
  Number,
  Object,
  Array,
  Set,
  Map,
  Promise,
  String,
  Boolean,
  app: {
    runs: [
      {local_run_id: "run-a", artifact_name: "alpha", model_type: "sheet", depth_snapshot_count: 2},
      {local_run_id: "run-b", artifact_name: "beta", model_type: "dense", depth_snapshot_count: 2},
      {local_run_id: "run-c", artifact_name: "hidden", model_type: "sheet", depth_snapshot_count: 2},
    ],
    current_run_id: "run-a",
    current_status: null,
    axis_ranges: {},
    figures: null,
    maximized_chart: null,
  },
  chart_titles: {
    heatmap: "Heatmap - Loss vs Counterfactual Layer Count",
    attn_q_head_N: "Q weights",
    attn_k_head_N: "K weights",
    attn_v_head_N: "V weights",
    attn_out_head_N: "Output weights",
    mlp_up: "MLP up weights",
    mlp_down: "MLP down weights",
  },
  window: {
    addEventListener: (_name, callback) => callback(),
    __instra_weight_stability_final: {
      gradient_enabled: () => gradient_enabled,
    },
  },
  document: {
    body: {classList: {add() {}, remove() {}}},
    head: {appendChild() {}},
    createElement: () => ({classList: {add() {}, toggle() {}}, dataset: {}, style: {}, append() {}, addEventListener() {}, setAttribute() {}}),
    querySelector: () => null,
    querySelectorAll: () => [],
  },
  setTimeout: callback => { callback(); return 1; },
  setInterval: () => 1,
  clearTimeout() {},
  queueMicrotask: callback => callback(),
  fetch: async url => {
    const run = new URL(`http://instra.local${url}`).searchParams.get("run");
    if (url.startsWith("/api/chart-groups")) {
      return {ok: true, json: async () => ({available: true, groups: [{name: "train", chart_count: 1, revision: run === "run-a" ? 2 : 3}]})};
    }
    if (url.startsWith("/api/chart-group")) {
      return {ok: true, json: async () => ({
        available: true,
        group: {
          name: "train",
          revision: run === "run-a" ? 2 : 3,
          charts: [{
            id: "loss",
            title: "loss",
            x_title: "step",
            default_x_axis_mode: "step",
            available_x_axis_modes: ["step", "relative_wall"],
            series: [{name: "loss", x: [1], y: [run === "run-a" ? 8 : 7]}],
          }],
        },
      })};
    }
    throw new Error(`unexpected fetch ${url}`);
  },
  URL,
  by_id: id => elements[id] || null,
  run_identifier: run => run.local_run_id,
  is_visible: run_id => visible.has(run_id),
  colour_for_run: run_id => colours[run_id],
  current_run() { return context.app.runs.find(run => run.local_run_id === context.app.current_run_id); },
  select_run() {},
  render_run_heading() {},
  render_runs() {},
  render_figures: async () => {},
  reset_run_charts() {},
  refresh_current_run() {},
  restore_maximized_chart() {},
  local_set_detail_tab() {},
  load_json: () => ({}),
  save_json() {},
  render_plot: async () => {},
  heatmap_settings_for_current_run: () => ({}),
  save_heatmap_viewer_setting(name, value) { saved_heatmap_settings[name] = value; },
  prepare_figure: figure => JSON.parse(JSON.stringify(figure)),
  transpose_heatmap() {},
};

vm.createContext(context);
const source = fs.readFileSync(
  "sheet/local_dashboard_assets/dashboard_v058_repair_workspace_patch.js",
  "utf8",
);
vm.runInContext(source, context, {filename: "dashboard_v058_repair_workspace_patch.js"});

async function main() {
  context.app.axis_chart_name = "heatmap";
  save_button_listeners.find(listener => listener.capture).callback();
  // The dashboard's original save handler runs between capture and bubble. It
  // closes the valid dialog, which is the signal used by the repair layer.
  elements.chart_settings_overlay.hidden = true;
  save_button_listeners.find(listener => !listener.capture).callback();
  assert.equal(saved_heatmap_settings.probe_count, 37);
  assert.equal(saved_heatmap_settings.y_display_mode, "steps");

  context.app.current_run_id = "run-b";
  context.render_run_heading();
  assert.equal(elements.heatmap_chart_group.hidden, true);
  context.app.runs[1].heatmap_count = 1;
  context.app.runs[1].heatmap_settings = {mode: true};
  context.render_run_heading();
  assert.equal(elements.heatmap_chart_group.hidden, false, "available heatmap data was hidden by model type");
  context.app.runs[1].heatmap_count = 0;
  context.app.runs[1].heatmap_settings = {mode: false};
  context.app.current_run_id = "run-a";
  context.render_run_heading();
  assert.equal(elements.heatmap_chart_group.hidden, false);

  const workspace = context.window.__instra_workspace;
  assert.equal(workspace.active(), false);
  assert.deepEqual(Array.from(workspace.visible_runs(), run => run.local_run_id), ["run-a", "run-b"]);

  const groups = await workspace.fetch_metric_groups();
  assert.deepEqual(Array.from(groups.groups, group => group.name), ["train"]);

  const train = await workspace.fetch_metric_group("train");
  assert.equal(train.group.charts.length, 1);
  assert.deepEqual(Array.from(train.group.charts[0].series, series => series.name), ["alpha", "beta"]);
  assert.deepEqual(Array.from(train.group.charts[0].series, series => series.color), ["#ef4444", "#2563eb"]);

  const depth = await workspace.fetch_depth_payload(async url => {
    const run = new URL(`http://instra.local${url}`).searchParams.get("run");
    return {
      depth: {
        mlp_down: {
          data: [10, 20].map((step, index) => ({
            name: `step ${step}`,
            mode: "lines+markers",
            x: [1, 2],
            y: run === "run-a" ? [1 + index, 1.5 + index] : [2 + index, 2.5 + index],
            line: {width: 0.45, shape: "linear"},
            marker: {symbol: "x", size: 9, line: {width: 4}},
            meta: {
              instra_dense_weight: true,
              ...(index === 0 ? {instra_dense_optimizer_update: step} : {}),
              instra_dense_step_legend: true,
              instra_dense_scalar_id: "r1_c2",
            },
          })),
          layout: {title: {text: "DENSE learned scalar weights"}, xaxis: {title: {text: "layer index"}}},
        },
      },
    };
  });
  assert.equal(depth.depth.mlp_down.data.length, 4);
  assert.deepEqual(
    Array.from(depth.depth.mlp_down.data, trace => trace.meta.instra_workspace_run_id),
    ["run-a", "run-a", "run-b", "run-b"],
  );
  assert.deepEqual(
    Array.from(depth.depth.mlp_down.data, trace => trace.name),
    ["alpha · step 10", "alpha · step 20", "beta · step 10", "beta · step 20"],
  );
  assert.equal(depth.depth.mlp_down.layout.showlegend, false);
  assert.equal(depth.depth.mlp_down.layout.legend, undefined);
  assert.ok(depth.depth.mlp_down.data.every(trace => trace.showlegend === false));
  assert.deepEqual(
    Array.from(depth.depth.mlp_down.data, trace => trace.meta.instra_workspace_optimizer_update),
    [10, 20, 10, 20],
  );
  assert.deepEqual(
    Array.from(depth.depth.mlp_down.data, trace => trace.meta.instra_workspace_artifact_name),
    ["alpha", "alpha", "beta", "beta"],
  );
  assert.deepEqual(
    Array.from(depth.depth.mlp_down.data, trace => trace.meta.instra_workspace_run_datetime),
    ["alpha", "alpha", "beta", "beta"],
  );
  assert.ok(depth.depth.mlp_down.data[0].hovertemplate.startsWith("<b>alpha</b><br>"));
  assert.ok(depth.depth.mlp_down.data[2].hovertemplate.startsWith("<b>beta</b><br>"));

  context.app.workspace_mode = true;
  const prepared_weight = context.prepare_figure(depth.depth.mlp_down, "mlp_down");
  assert.equal(prepared_weight.layout.title, undefined);
  assert.equal(prepared_weight.layout.xaxis.side, "bottom");
  assert.equal(prepared_weight.layout.xaxis2.side, "top");
  assert.equal(prepared_weight.data[0].marker.line.width, 0.35);
  assert.equal(prepared_weight.data[0].marker.size, 6);
  assert.match(prepared_weight.data[0].marker.color, /^hsl\(/);

  // The v0.58 compatibility layer installs after the final range owner in the
  // browser. When gradient mode is active it must preserve the light-to-base
  // colours already assigned to ordinary Workspace curves.
  gradient_enabled = true;
  const gradient_colours = ["#f0d8d0", "#d6a295", colours["run-a"]];
  const gradient_figure = {
    data: [10, 20, 30].map((step, index) => ({
      mode: "lines+markers",
      line: {color: gradient_colours[index]},
      marker: {symbol: "x", color: gradient_colours[index], line: {color: gradient_colours[index]}},
      meta: {
        instra_workspace_run_id: "run-a",
        instra_workspace_optimizer_update: step,
        instra_dense_weight: true,
        instra_dense_optimizer_update: step,
      },
    })),
    layout: {},
  };
  const preserved_gradient = context.prepare_figure(gradient_figure, "mlp_down");
  assert.deepEqual(
    Array.from(preserved_gradient.data, trace => trace.line.color),
    gradient_colours,
    "late Workspace repair replaced the enabled step gradient",
  );
  assert.deepEqual(
    Array.from(preserved_gradient.data, trace => trace.marker.line.color),
    gradient_colours,
    "late Workspace repair replaced gradient marker outlines",
  );

  const single_run_gradient = {
    data: gradient_figure.data.map(trace => ({
      ...trace,
      line: {...trace.line},
      marker: {...trace.marker, line: {...trace.marker.line}},
      meta: {
        instra_dense_weight: true,
        instra_dense_optimizer_update: trace.meta.instra_dense_optimizer_update,
      },
    })),
    layout: {},
  };
  const preserved_single_run_gradient = context.prepare_figure(single_run_gradient, "mlp_down");
  assert.deepEqual(
    Array.from(preserved_single_run_gradient.data, trace => trace.line.color),
    gradient_colours,
    "late DENSE repair replaced an enabled gradient in Runs",
  );
  assert.deepEqual(
    Array.from(preserved_single_run_gradient.data, trace => trace.marker.line.color),
    gradient_colours,
    "late DENSE repair replaced enabled gradient marker outlines in Runs",
  );
  gradient_enabled = false;
  assert.equal(prepared_weight.data[0].marker.color, prepared_weight.data[2].marker.color);
  assert.notEqual(prepared_weight.data[0].marker.color, prepared_weight.data[1].marker.color);
  assert.equal(prepared_weight.data[0].mode, "lines+markers");
  assert.equal(prepared_weight.data[0].line.width, 0.45);
  assert.equal(prepared_weight.data[0].line.shape, "linear");
  assert.equal(prepared_weight.data[0].line.color, prepared_weight.data[0].marker.color);
  assert.ok(prepared_weight.data.slice(0, 4).every(trace => !/oldest|newest/i.test(trace.name)));
  const weight_top_axis_anchor = prepared_weight.data.find(trace => trace.meta?.instra_top_axis_anchor);
  assert.equal(weight_top_axis_anchor.xaxis, "x2");
  assert.equal(weight_top_axis_anchor.opacity, 0);
  assert.equal(weight_top_axis_anchor.showlegend, false);
  context.app.workspace_mode = false;
  const prepared_single_weight = context.prepare_figure(depth.depth.mlp_down, "mlp_down");
  assert.equal(prepared_single_weight.layout.title, undefined);
  assert.equal(prepared_single_weight.layout.showlegend, false);
  assert.equal(prepared_single_weight.layout.legend, undefined);
  assert.equal(prepared_single_weight.layout.xaxis2.title, undefined);
  assert.ok(prepared_single_weight.data.every(trace => trace.showlegend === false));

  // The weight-history display window counts recorded optimizer steps. It
  // deliberately has no relationship to PLASTIC probe rows.
  const dashboard_source = fs.readFileSync("sheet/local_dashboard_assets/dashboard.js", "utf8");
  const function_source_from = (source_text, name) => {
    const start = source_text.indexOf(`function ${name}(`);
    assert.notEqual(start, -1, `missing ${name}`);
    const body_start = source_text.indexOf("{", start);
    let depth = 0;
    for (let index = body_start; index < source_text.length; index += 1) {
      if (source_text[index] === "{") depth += 1;
      else if (source_text[index] === "}") {
        depth -= 1;
        if (depth === 0) return source_text.slice(start, index + 1);
      }
    }
    throw new Error(`unterminated ${name}`);
  };
  const function_source = name => function_source_from(dashboard_source, name);
  const run_list_context = {
    Date,
    Number,
    String,
    app: {timeout_minutes: 15},
  };
  vm.createContext(run_list_context);
  vm.runInContext(
    ["is_active_run_state", "display_run_state", "format_run_duration"]
      .map(function_source)
      .join("\n"),
    run_list_context,
  );
  const terminal_run = (seconds, state = "finished") => ({
    run_state: state,
    created_at: "2026-08-01T00:00:00.000Z",
    heartbeat_at: new Date(Date.parse("2026-08-01T00:00:00.000Z") + seconds * 1000).toISOString(),
  });
  assert.equal(run_list_context.format_run_duration(terminal_run(34)), "34s");
  assert.equal(run_list_context.format_run_duration(terminal_run(52.25 * 60)), "52.3m");
  assert.equal(run_list_context.format_run_duration(terminal_run(17.94 * 3600)), "17.9h");
  assert.equal(run_list_context.format_run_duration(terminal_run(8 * 24 * 3600)), "192.0h");

  const palette_start = dashboard_source.indexOf("const default_palette = [");
  const palette_end = dashboard_source.indexOf("];", palette_start) + 2;
  const palette_context = {};
  vm.createContext(palette_context);
  vm.runInContext(
    `${dashboard_source.slice(palette_start, palette_end)} this.palette = default_palette;`,
    palette_context,
  );
  assert.equal(palette_context.palette.length, 64, "run colour picker does not expose 64 colours");
  assert.equal(new Set(palette_context.palette).size, 64, "run colour palette contains duplicates");
  const palette_brightness = colour => [1, 3, 5]
    .map(index => parseInt(colour.slice(index, index + 2), 16))
    .reduce((total, value) => total + value, 0);
  for (let index = 0; index < 32; index += 1) {
    assert.ok(
      palette_brightness(palette_context.palette[index + 32])
        > palette_brightness(palette_context.palette[index]),
      `lighter palette entry ${index + 33} is not lighter than its base entry`,
    );
  }

  const step_window_context = {Number, Set};
  vm.createContext(step_window_context);
  vm.runInContext(
    ["trace_optimizer_update", "available_snapshot_updates", "limit_curve_snapshots", "retain_latest_weight_snapshots", "apply_thog_line_segments"]
      .map(function_source)
      .join("\n"),
    step_window_context,
  );
  const step_figure = {
    data: [1, 3, 5].map(step => ({
      name: `step ${step}`,
      meta: {instra_dense_optimizer_update: step},
    })).concat([{name: "top-axis anchor", meta: {instra_top_axis_anchor: true}}]),
  };
  const from_zero = JSON.parse(JSON.stringify(step_figure));
  step_window_context.limit_curve_snapshots(from_zero, 2, "from_zero");
  assert.deepEqual(
    Array.from(from_zero.data, trace => trace.meta?.instra_dense_optimizer_update ?? null),
    [1, 3, null],
  );
  const rolling = JSON.parse(JSON.stringify(step_figure));
  step_window_context.limit_curve_snapshots(rolling, 2, "rolling");
  assert.deepEqual(
    Array.from(rolling.data, trace => trace.meta?.instra_dense_optimizer_update ?? null),
    [3, 5, null],
  );
  assert.equal(step_window_context.trace_optimizer_update({name: "step 9"}), 9);
  assert.equal(step_window_context.trace_optimizer_update({name: "curve U12"}), 12);
  assert.equal(step_window_context.trace_optimizer_update({meta: {instra_thog_optimizer_update: 13}}), 13);
  assert.equal(step_window_context.trace_optimizer_update({meta: {instra_workspace_optimizer_update: 14}}), 14);
  assert.equal(step_window_context.trace_optimizer_update({meta: {instra_workspace_optimizer_update: null}, name: "owner step 336"}), 336);

  const workspace_current = {
    data: [
      {name: "step 1", meta: {instra_dense_optimizer_update: 1, instra_workspace_run_id: "run-a"}},
      {name: "step 336 scalar 1", meta: {instra_dense_optimizer_update: 336, instra_workspace_run_id: "run-a"}},
      {name: "step 336 scalar 2", meta: {instra_dense_optimizer_update: 336, instra_workspace_run_id: "run-a"}},
      {name: "historical owner trace", meta: {instra_workspace_optimizer_update: null, instra_workspace_run_id: "run-a"}},
      {name: "curve U3", meta: {instra_workspace_run_id: "run-b"}},
      {name: "curve U4", meta: {instra_workspace_run_id: "run-b"}},
      {name: "top-axis anchor", meta: {instra_top_axis_anchor: true}},
    ],
  };
  step_window_context.retain_latest_weight_snapshots(workspace_current);
  assert.deepEqual(
    Array.from(workspace_current.data, trace => [
      trace.meta?.instra_workspace_run_id || null,
      step_window_context.trace_optimizer_update(trace),
    ]),
    [["run-a", 336], ["run-a", 336], ["run-b", 4], [null, null]],
  );

  // The final loaded asset repeats the invariant after every legacy render
  // wrapper. This is the path used by Workspace after its traces are merged.
  const group_patch_source = fs.readFileSync(
    "sheet/local_dashboard_assets/dashboard_weights_group_settings_patch.js",
    "utf8",
  );
  const group_patch_context = {Number, Map, Math, Object, String};
  vm.createContext(group_patch_context);
  vm.runInContext(
    [
      "instra_weight_trace_update",
      "instra_enforce_workspace_latest_weights",
      "instra_apply_weight_group_defaults",
    ]
      .map(name => function_source_from(group_patch_source, name))
      .join("\n"),
    group_patch_context,
  );
  const final_workspace_current = {
    data: [
      {name: "step 1", meta: {instra_workspace_optimizer_update: null, instra_dense_optimizer_update: 1, instra_workspace_run_id: "run-a"}},
      {name: "step 336 scalar 1", meta: {instra_workspace_optimizer_update: null, instra_dense_optimizer_update: 336, instra_workspace_run_id: "run-a"}},
      {name: "step 336 scalar 2", meta: {instra_workspace_optimizer_update: 336, instra_dense_optimizer_update: 336, instra_workspace_run_id: "run-a"}},
      {name: "curve U7", meta: {instra_workspace_run_id: "run-b"}},
      {name: "curve U9", meta: {instra_workspace_run_id: "run-b"}},
      {name: "owner without update", meta: {instra_workspace_run_id: "run-b"}},
      {name: "top-axis anchor", meta: {instra_top_axis_anchor: true}},
    ],
  };
  group_patch_context.instra_enforce_workspace_latest_weights(final_workspace_current);
  assert.deepEqual(
    Array.from(final_workspace_current.data, trace => [
      trace.meta?.instra_workspace_run_id || null,
      group_patch_context.instra_weight_trace_update(trace),
    ]),
    [["run-a", 336], ["run-a", 336], ["run-b", 9], [null, null]],
  );
  const inherited = {title: "individual title", current_weights_only: false, line_width: 1};
  group_patch_context.instra_apply_weight_group_defaults(
    inherited,
    {current_weights_only: true, line_width: 2},
    {line_width: 0.75},
    ["current_weights_only", "line_width"],
  );
  assert.deepEqual(
    {...inherited},
    {title: "individual title", current_weights_only: true, line_width: 0.75},
  );

  const joined_thog = {
    data: [
      {
        name: "r1_c2 · U8",
        mode: "lines",
        x: [1, 1.5, 2],
        y: [0.1, 0.2, 0.3],
        line: {color: "#158f80", width: 2},
        hovertemplate: "r1_c2<br>U8<br>layer=%{x:.3f}<br>weight=%{y:.7g}<extra></extra>",
        meta: {
          instra_thog_weight: true,
          instra_thog_optimizer_update: 8,
          instra_thog_integer_x: [1, 2],
          instra_thog_integer_y: [0.1, 0.3],
        },
      },
      {
        name: "r1_c2 · executed layers",
        mode: "markers",
        x: [1, 2],
        y: [0.1, 0.3],
        meta: {instra_thog_weight: true, instra_thog_executed_overlay: true},
      },
      {
        name: "step 8",
        mode: "lines+markers",
        x: [1, 2],
        y: [1, 2],
        meta: {instra_dense_weight: true, instra_dense_optimizer_update: 8},
      },
    ],
  };
  step_window_context.apply_thog_line_segments(joined_thog);
  assert.equal(joined_thog.data.length, 2);
  assert.deepEqual(Array.from(joined_thog.data[0].x), [1, 2]);
  assert.deepEqual(Array.from(joined_thog.data[0].y), [0.1, 0.3]);
  assert.equal(joined_thog.data[0].mode, "lines+markers");
  assert.equal(joined_thog.data[0].line.shape, "linear");
  assert.equal(joined_thog.data[0].marker.symbol, "circle-open");
  assert.equal(joined_thog.data[0].marker.line.width, 1.2);
  assert.ok(joined_thog.data[0].hovertemplate.includes("layer=%{x:.0f}"));
  assert.equal(joined_thog.data[1].meta.instra_dense_weight, true);

  const weight_preference_store = {};
  const weight_preference_context = {
    app: {
      axis_chart_name: null,
      axis_chart_workspace_mode: null,
      workspace_mode: false,
      current_run_id: "run-a",
      weight_current_only: {},
      weight_join_with_line_segments: {},
    },
    depth_weight_chart_set: new Set(["mlp_down"]),
    weight_current_only_storage_key: "weight-preferences",
    weight_join_with_line_segments_storage_key: "join-preferences",
    save_json(key, value) { weight_preference_store[key] = JSON.parse(JSON.stringify(value)); },
    String,
  };
  vm.createContext(weight_preference_context);
  vm.runInContext(
    [
      "weight_current_only_scope",
      "stored_weight_current_only",
      "save_weight_current_only",
      "stored_weight_join_with_line_segments",
      "save_weight_join_with_line_segments",
    ]
      .map(function_source)
      .join("\n"),
    weight_preference_context,
  );
  weight_preference_context.save_weight_current_only("mlp_down", true);
  assert.equal(weight_preference_context.stored_weight_current_only("mlp_down"), true);
  weight_preference_context.app.current_run_id = "run-b";
  assert.equal(weight_preference_context.stored_weight_current_only("mlp_down"), false);
  weight_preference_context.app.workspace_mode = true;
  weight_preference_context.save_weight_current_only("mlp_down", true);
  weight_preference_context.save_weight_join_with_line_segments("mlp_down", true);
  assert.equal(weight_preference_context.stored_weight_current_only("mlp_down"), true);
  assert.equal(weight_preference_context.stored_weight_join_with_line_segments("mlp_down"), true);
  weight_preference_context.app.workspace_mode = false;
  weight_preference_context.app.current_run_id = "run-a";
  assert.equal(weight_preference_context.stored_weight_current_only("mlp_down"), true);
  assert.equal(weight_preference_context.stored_weight_join_with_line_segments("mlp_down"), false);
  assert.deepEqual(weight_preference_store["weight-preferences"], {
    "run:run-a:mlp_down": true,
    "workspace:mlp_down": true,
  });
  assert.deepEqual(weight_preference_store["join-preferences"], {
    "workspace:mlp_down": true,
  });

  const prepared_heatmap = {
    data: [
      {type: "scatter", name: "obsolete active-layer line", x: [-1, 1], y: [10, 11]},
      {
        type: "heatmap",
        x: [-1, 0, 1],
        y: [10, 11],
        z: [[0, 0, -0.2], [0, 0, -0.1]],
        customdata: [
          [[10, 3, -1, 0.1], [10, 4, 0, 0], [10, 5, 1, -0.2]],
          [[11, 4, -1, 0.1], [11, 5, 0, 0], [11, 6, 1, -0.1]],
        ],
        colorbar: {title: {text: "Δloss (%) bands"}},
      },
    ],
    layout: {
      xaxis: {tickfont: {size: 8}},
      yaxis: {tickfont: {size: 8}},
      annotations: [{
        xref: "x", yref: "y", x: 0, y: 10,
        hovertext: "step=10", text: "L", font: {family: "DejaVu Sans Mono, monospace", size: 9},
      }],
      meta: {
        thog2_optimizer_updates: [10, 11],
        thog2_active_layers: [4, 5],
        thog2_selected_layers: [5, 5],
        thog2_current_losses: [10, 9],
        thog2_brake_active: [false, true],
        thog2_decision_committed: [true, false],
        thog2_chaos_bump: [null, {state: "active", magnitude_percent: 5, step: 1, duration: 12}],
      },
    },
  };
  context.transpose_heatmap(prepared_heatmap);
  assert.equal(prepared_heatmap.data.length, 2);
  assert.equal(prepared_heatmap.data[0].type, "heatmap");
  const heatmap_top_axis_anchor = prepared_heatmap.data.find(trace => trace.meta?.instra_top_axis_anchor);
  assert.equal(heatmap_top_axis_anchor.xaxis, "x2");
  assert.deepEqual(Array.from(heatmap_top_axis_anchor.x), [-1, 1]);
  assert.equal(heatmap_top_axis_anchor.opacity, 0);
  assert.equal(prepared_heatmap.layout.xaxis2.side, "top");
  assert.deepEqual(Array.from(prepared_heatmap.layout.xaxis2.ticktext), ["4", "5", "6"]);
  assert.equal(prepared_heatmap.layout.xaxis.tickfont.size, 14);
  assert.equal(prepared_heatmap.layout.yaxis.tickfont.size, 14);
  assert.ok(prepared_heatmap.layout.shapes.some(shape => shape.name === "thog2-centre-datum-background" && shape.fillcolor === "#000000"));
  assert.ok(prepared_heatmap.layout.shapes.some(shape => shape.name === "thog2-committed-decision-brick" && shape.fillcolor === "#ffffff"));
  const winner = prepared_heatmap.layout.annotations.find(annotation => annotation.name === "thog2-best-better-loss");
  assert.equal(winner.text, "<b>9.800 (2.00%)</b>");
  assert.equal(winner.font.size, 9);
  assert.equal(prepared_heatmap.layout.annotations.filter(annotation => annotation.name === "thog2-committed-decision-text").length, 0);
  assert.ok(prepared_heatmap.layout.annotations.some(annotation => annotation.name === "thog2-update-brake" && annotation.font.color === "#ff9696"));
  assert.ok(prepared_heatmap.layout.annotations.some(annotation => annotation.name === "thog2-chaos-bump" && annotation.text.includes("Step 1/12")));

  // prepare_figure is the final owner after older presentation wrappers have
  // run; it must restore the absolute top title as well as retain the anchor.
  prepared_heatmap.layout.xaxis2.title = {text: "candidate layer-count offset from L"};
  const finally_prepared_heatmap = context.prepare_figure(prepared_heatmap, "heatmap");
  assert.equal(finally_prepared_heatmap.layout.xaxis2.title.text, "absolute candidate layer count · latest L=5");
  assert.equal(
    finally_prepared_heatmap.data.filter(trace => trace.meta?.instra_top_axis_anchor).length,
    1,
  );
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
