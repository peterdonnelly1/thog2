// vvv THOG
"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const repository_root = path.resolve(__dirname, "..");
const asset = name => path.join(repository_root, "sheet/local_dashboard_assets", name);
const load_source = name => fs.readFileSync(asset(name), "utf8");

async function request_routing_regression() {
  let current_only = true;
  let selected_range = null;
  const requests = [];
  const context = {
    console,
    URL,
    depth_weight_chart_names: ["q", "k", "v", "o", "up", "down"],
    normalize_chart_settings: () => ({current_weights_only: current_only}),
    fetch_json: async url => {
      requests.push(String(url));
      return {url: String(url)};
    },
    window: {
      location: {origin: "http://127.0.0.1:6007"},
      __instra_weight_step_filter: {request_range: () => selected_range},
    },
  };
  context.window.window = context.window;
  vm.createContext(context);
  vm.runInContext(load_source("dashboard_current_weights_request_patch.js"), context);

  await context.fetch_json("/api/figure-family?run=A&family=depth");
  let parsed = new URL(requests.at(-1), "http://127.0.0.1:6007");
  assert.equal(parsed.searchParams.get("current_only"), "1");

  selected_range = {minimum: 1230, maximum: 1329};
  await context.fetch_json("/api/figure-family?run=A&family=depth&current_only=1");
  parsed = new URL(requests.at(-1), "http://127.0.0.1:6007");
  assert.equal(parsed.searchParams.get("step_min"), "1230");
  assert.equal(parsed.searchParams.get("step_max"), "1329");
  assert.equal(parsed.searchParams.has("current_only"), false);

  selected_range = null;
  current_only = false;
  await context.fetch_json("/api/figure-family?run=A&family=depth");
  parsed = new URL(requests.at(-1), "http://127.0.0.1:6007");
  assert.equal(parsed.searchParams.has("current_only"), false);
  assert.equal(parsed.searchParams.has("step_min"), false);
}

function global_flags_and_round_trip_regression() {
  const storage = new Map();
  const tabs = [{hidden: true}, {hidden: false}];
  const click_handlers = [];
  let workspace_runs = [];
  const elements = new Map([
    ["chart_current_weights_only", {checked: false, disabled: false}],
    ["chart_join_with_line_segments", {checked: false, disabled: false}],
    ["chart_current_weights_only_field", {querySelector: () => ({textContent: ""})}],
    ["chart_join_with_line_segments_field", {querySelector: () => ({textContent: ""})}],
    ["chart_settings_overlay", {hidden: true}],
    ["chart_settings_title", {textContent: "Attention - Q settings"}],
  ]);
  const weight_names = ["q", "k", "v", "o", "up", "down"];
  let refreshes = 0;
  let retain_calls = 0;
  let workspace_latest_calls = 0;
  let open_saw_tabs = null;
  const base_settings = {current_weights_only: false, join_with_line_segments: false};

  const load_json = (key, fallback) => {
    if (key === "thog2_local_trajectory_scale_modes") return {q: "linear"};
    if (!storage.has(key)) return fallback;
    return JSON.parse(storage.get(key));
  };
  const save_json = (key, value) => storage.set(key, JSON.stringify(value));

  const current_run_value = {
    dashboard_run_id: "R1",
    maximum_update: 1100,
    depth_snapshot_count: 100,
    depth_minimum_update: 1001,
    depth_maximum_update: 1100,
    configuration: {instrumentation__depth_weight_curves__history_length: 100},
  };

  const context = {
    console,
    structuredClone: global.structuredClone,
    depth_weight_chart_names: weight_names,
    app: {
      workspace_mode: false,
      current_run_id: "R1",
      current_status: current_run_value,
      refresh_in_flight: false,
      figures: {heatmap: null, depth: {q: {data: []}}},
      figure_revision: "old",
      axis_chart_name: "q",
      axis_chart_workspace_mode: false,
    },
    window: {
      location: {origin: "http://127.0.0.1:6007"},
      addEventListener(event, callback) {
        if (event === "load") callback();
        else if (event === "click") click_handlers.push(callback);
      },
      removeEventListener() {},
      __instra_workspace: {visible_runs: () => workspace_runs},
      __instra_workspace_depth_cache: {clear() {}},
      __thog2_dashboard_performance: {state: {depth_signature: "old", pending_render: {}, deferred_coefficients: false}},
      __instra_matched_weight_selection: null,
      Plotly: {purge() {}},
    },
    document: {
      activeElement: null,
      querySelector() { return null; },
      querySelectorAll(selector) {
        if (selector === "[data-chart-settings-tab]") return tabs;
        return [];
      },
      createElement() { return {textContent: "", style: {}, classList: {add() {}, toggle() {}}, append() {}, appendChild() {}}; },
      head: {appendChild() {}},
    },
    by_id: id => elements.get(id) || null,
    load_json,
    save_json,
    normalize_chart_settings: () => ({...base_settings}),
    retain_latest_weight_snapshots: () => { retain_calls += 1; },
    instra_enforce_workspace_latest_weights: prepared => { workspace_latest_calls += 1; return prepared; },
    open_chart_settings: () => { open_saw_tabs = tabs.map(tab => tab.hidden); },
    populate_chart_settings_form: () => undefined,
    sync_chart_setting_outputs: () => undefined,
    current_run: () => current_run_value,
    run_identifier: run => run.dashboard_run_id,
    refresh_current_run: () => { refreshes += 1; },
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
  vm.runInContext(load_source("dashboard_weight_step_controls_patch.js"), context);

  const api = context.window.__instra_weight_controls_v2;
  assert.ok(api, "weights-v2 API was not installed");
  assert.equal(api.common_history_capacity(), 100);
  assert.deepEqual(api.available_step_range(), {available: true, minimum: 1001, maximum: 1100});

  for (const chart_name of weight_names) {
    const settings = context.normalize_chart_settings(chart_name);
    assert.equal(settings.current_weights_only, false);
    assert.equal(settings.join_with_line_segments, false);
  }

  // The real UI save path is captured on window before the older group handler can
  // stopImmediatePropagation at the save button.
  elements.get("chart_settings_overlay").hidden = false;
  elements.get("chart_settings_title").textContent = "Run Weights settings";
  elements.get("chart_current_weights_only").checked = true;
  elements.get("chart_join_with_line_segments").checked = true;
  const save_event = {target: {closest: selector => selector === "#save_chart_settings" ? {} : null}};
  for (const handler of click_handlers) handler(save_event);
  assert.equal(api.global_flags().current_weights_only, true);
  assert.equal(api.global_flags().join_with_line_segments, true);
  elements.get("chart_settings_overlay").hidden = true;
  elements.get("chart_settings_title").textContent = "Attention - Q settings";

  const initial_refreshes = refreshes;
  api.set_global_flags({current_weights_only: false, join_with_line_segments: false});
  assert.ok(refreshes > initial_refreshes, "disabling current-only did not refetch depth data");
  assert.equal(context.app.figures.depth, null, "stale current-only payload survived history transition");

  const before_current_enable = refreshes;
  api.set_global_flags({current_weights_only: true, join_with_line_segments: true});
  assert.ok(refreshes > before_current_enable, "enabling current-only did not refetch depth data");
  for (const chart_name of weight_names) {
    const settings = context.normalize_chart_settings(chart_name);
    assert.equal(settings.current_weights_only, true);
    assert.equal(settings.join_with_line_segments, true);
  }

  const current_figure = context.prepare_figure({
    data: [{mode: "lines+markers", line: {color: "#old", width: 1}, marker: {color: "#old"}}],
    layout: {},
  }, "q");
  assert.equal(current_figure.data[0].line.color, "#123456", "current-only Runs curve did not use run colour");
  assert.equal(current_figure.data[0].marker.color, "#123456");

  // current-only -> history must invalidate again; this is the three-screenshot regression.
  context.app.figures = {heatmap: null, depth: {q: {data: [{name: "one current trace"}]}}};
  const before_history_restore = refreshes;
  api.set_global_flags({current_weights_only: false, join_with_line_segments: false});
  assert.ok(refreshes > before_history_restore, "disabling current-only did not refetch history");
  assert.equal(context.app.figures.depth, null, "one-snapshot payload survived history restore");
  assert.equal(context.normalize_chart_settings("q").current_weights_only, false);
  assert.equal(context.normalize_chart_settings("q").join_with_line_segments, false);

  api.set_step_range(1230, 1329);
  assert.deepEqual(context.window.__instra_weight_step_filter.request_range(), {minimum: 1230, maximum: 1329});
  assert.equal(context.normalize_chart_settings("q").current_weights_only, true);
  context.retain_latest_weight_snapshots({data: []});
  context.instra_enforce_workspace_latest_weights({data: []});
  assert.equal(retain_calls, 0, "explicit step window was collapsed to newest Runs snapshot");
  assert.equal(workspace_latest_calls, 0, "explicit step window was collapsed to newest Workspace snapshot");
  api.clear_step_range();
  context.retain_latest_weight_snapshots({data: []});
  assert.equal(retain_calls, 1, "default history path did not resume after clearing explicit window");

  // Workspace availability is the intersection of eye-d runs and must change as
  // runs are ablated. Capacity is likewise the common (minimum) capture history.
  context.app.workspace_mode = true;
  workspace_runs = [
    {
      dashboard_run_id: "A", maximum_update: 200, depth_snapshot_count: 100,
      depth_minimum_update: 101, depth_maximum_update: 200,
      configuration: {instrumentation__depth_weight_curves__history_length: 100},
    },
    {
      dashboard_run_id: "B", maximum_update: 180, depth_snapshot_count: 60,
      depth_minimum_update: 121, depth_maximum_update: 180,
      configuration: {instrumentation__depth_weight_curves__history_length: 80},
    },
  ];
  assert.deepEqual(api.available_step_range(), {available: true, minimum: 121, maximum: 180});
  assert.deepEqual(api.current_step_bounds(), {minimum: 180, maximum: 200});
  assert.equal(api.common_history_capacity(), 80);

  workspace_runs = [workspace_runs[0]];
  assert.deepEqual(api.available_step_range(), {available: true, minimum: 101, maximum: 200});
  assert.equal(api.common_history_capacity(), 100);

  workspace_runs.push({
    dashboard_run_id: "C", maximum_update: 300, depth_snapshot_count: 51,
    depth_minimum_update: 250, depth_maximum_update: 300,
    configuration: {instrumentation__depth_weight_curves__history_length: 100},
  });
  assert.deepEqual(api.available_step_range(), {available: false, reason: "no overlapping steps"});
  context.app.workspace_mode = false;
  workspace_runs = [];

  tabs[0].hidden = true;
  tabs[1].hidden = false;
  context.open_chart_settings("q");
  assert.deepEqual(open_saw_tabs, [false, false], "individual weight settings did not restore both tabs");
}

function structural_regression() {
  const source = load_source("dashboard_weight_step_controls_patch.js");
  assert.match(source, /show_label\.textContent = "show weights for steps"/);
  assert.match(source, /from\.placeholder = "from"/);
  assert.match(source, /to\.placeholder = "to"/);
  assert.match(source, /Curves will be displayed when step \$\{selected_step_range\.minimum\} is reached/);
  assert.match(source, /width > capacity/);
  assert.match(source, /instrumentation__depth_weight_curves__history_length/);
  assert.match(source, /Global across all six weight charts and every run/);
  assert.match(source, /font-size: 11px/);
  assert.match(source, /#weight_random_jump::after/);
  assert.match(source, /content: "RND"/);
  assert.match(source, /margin-left: 38px !important/);
  assert.match(source, /full\.slice\(0, 10\)/);
  assert.match(source, /headers\[logged_index\]\.textContent = "STEPS"/);
  assert.match(source, /new_actual = smallest\.actual_value \/ 10/);
  assert.match(source, /padStart\(2, "0"\)/);
  assert.ok(!source.includes("MutationObserver"), "weights-v2 reintroduced a persistent DOM observer");
}

(async () => {
  await request_routing_regression();
  global_flags_and_round_trip_regression();
  structural_regression();
  console.log("instra weight step/filter regression: PASS");
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
// ^^^ THOG
