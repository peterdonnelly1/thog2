// vvv THOG
"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const repository_root = path.resolve(__dirname, "..");
const source = fs.readFileSync(
  path.join(repository_root, "sheet/local_dashboard_assets/dashboard_weight_step_controls_patch.js"),
  "utf8",
);
const weight_names = ["q", "k", "v", "o", "up", "down"];

function load_context(storage, {scale_mode = "linear"} = {}) {
  const elements = new Map([
    ["chart_current_weights_only", {checked: false, disabled: false}],
    ["chart_join_with_line_segments", {checked: false, disabled: false}],
    ["chart_current_weights_only_field", {querySelector: () => ({textContent: ""})}],
    ["chart_join_with_line_segments_field", {querySelector: () => ({textContent: ""})}],
    ["chart_settings_overlay", {hidden: true}],
    ["chart_settings_title", {textContent: "Attention - Q settings"}],
  ]);
  const run = {
    dashboard_run_id: "R1",
    maximum_update: 100,
    depth_snapshot_count: 20,
    depth_minimum_update: 81,
    depth_maximum_update: 100,
    configuration: {instrumentation__depth_weight_curves__history_length: 100},
  };
  const load_json = (key, fallback) => {
    if (key === "thog2_local_trajectory_scale_modes") return {q: scale_mode};
    if (!storage.has(key)) return fallback;
    return JSON.parse(storage.get(key));
  };
  const save_json = (key, value) => storage.set(key, JSON.stringify(value));
  const context = {
    console,
    structuredClone: global.structuredClone,
    depth_weight_chart_names: weight_names,
    app: {
      workspace_mode: false,
      current_run_id: "R1",
      current_status: run,
      refresh_in_flight: false,
      figures: {heatmap: null, depth: {}},
      figure_revision: null,
      axis_chart_name: "q",
      axis_chart_workspace_mode: false,
    },
    window: {
      location: {origin: "http://127.0.0.1:6007"},
      addEventListener(event, callback) { if (event === "load") callback(); },
      removeEventListener() {},
      __instra_workspace: {visible_runs: () => []},
      __instra_workspace_depth_cache: {clear() {}},
      __thog2_dashboard_performance: {state: {}},
      __instra_matched_weight_selection: null,
      Plotly: {purge() {}},
    },
    document: {
      activeElement: null,
      querySelector() { return null; },
      querySelectorAll() { return []; },
      createElement() {
        return {
          textContent: "", style: {}, classList: {add() {}, toggle() {}},
          append() {}, appendChild() {}, setAttribute() {}, addEventListener() {},
        };
      },
      head: {appendChild() {}},
    },
    by_id: id => elements.get(id) || null,
    load_json,
    save_json,
    normalize_chart_settings: () => ({current_weights_only: false, join_with_line_segments: false}),
    retain_latest_weight_snapshots: prepared => prepared,
    instra_enforce_workspace_latest_weights: prepared => prepared,
    open_chart_settings: () => undefined,
    populate_chart_settings_form: () => undefined,
    sync_chart_setting_outputs: () => undefined,
    current_run: () => run,
    run_identifier: value => value.dashboard_run_id,
    refresh_current_run: () => undefined,
    render_figures: async () => undefined,
    prepare_figure: figure => JSON.parse(JSON.stringify(figure)),
    render_plot: async () => undefined,
    render_runs: () => undefined,
    render_run_heading: () => undefined,
    figure_for_chart: () => null,
    show_toast: message => { throw new Error(`unexpected toast: ${message}`); },
    colour_for_run: () => "#123456",
    localStorage: {
      getItem(key) { return storage.get(key) ?? null; },
      setItem(key, value) { storage.set(key, String(value)); },
    },
    setTimeout(callback) { callback(); return 1; },
    clearTimeout() {},
  };
  context.window.window = context.window;
  vm.createContext(context);
  vm.runInContext(source, context);
  return context;
}

function persistence_regression() {
  const storage = new Map();
  let context = load_context(storage);
  context.window.__instra_weight_controls_v2.set_global_flags(
    {current_weights_only: true, join_with_line_segments: true},
    {refresh: false},
  );
  assert.ok(storage.has("thog2_local_weight_global_flags_v2"));

  context = load_context(storage);
  assert.deepEqual(
    context.window.__instra_weight_controls_v2.global_flags(),
    {current_weights_only: true, join_with_line_segments: true},
    "global weight flags did not survive a fresh dashboard load",
  );
  for (const chart_name of weight_names) {
    const settings = context.normalize_chart_settings(chart_name);
    assert.equal(settings.current_weights_only, true, `${chart_name} lost current-only after reload`);
    assert.equal(settings.join_with_line_segments, true, `${chart_name} lost line-segment mode after reload`);
  }

  context.window.__instra_weight_controls_v2.set_global_flags(
    {current_weights_only: false, join_with_line_segments: false},
    {refresh: false},
  );
  context = load_context(storage);
  assert.deepEqual(
    context.window.__instra_weight_controls_v2.global_flags(),
    {current_weights_only: false, join_with_line_segments: false},
    "cleared global weight flags did not survive a fresh dashboard load",
  );
}

function signed_log_runtime_regression() {
  const context = load_context(new Map(), {scale_mode: "log"});
  const prepared = context.prepare_figure({
    data: [],
    layout: {
      yaxis: {
        tickvals: [-1, 0, 1, 3],
        ticktext: ["-0.001", "0", "0.001", "100"],
      },
    },
  }, "q");
  const labels = prepared.layout.yaxis.ticktext;
  assert.ok(labels.includes("-1e-04"), "signed-log axis did not add the next smaller negative decade");
  assert.ok(labels.includes("1e-04"), "signed-log axis did not add the next smaller positive decade");
  assert.ok(labels.includes("0e+00"), "signed-log zero is not scientific notation");
  assert.ok(labels.includes("1e+02"), "large signed-log tick is not scientific notation");
  for (const label of labels) {
    assert.match(label, /^-?\d+(?:\.\d+)?e[+-]\d{2}$/i, `non-scientific signed-log label: ${label}`);
  }
  assert.equal(prepared.layout.yaxis.tickvals.length, prepared.layout.yaxis.ticktext.length);
}

persistence_regression();
signed_log_runtime_regression();
console.log("instra weight persistence/log regression: PASS");
// ^^^ THOG
