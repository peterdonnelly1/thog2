// vvv THOG
"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const repository_root = path.resolve(__dirname, "..");
const source = fs.readFileSync(
  path.join(repository_root, "sheet/local_dashboard_assets/dashboard_weight_stability_final_patch.js"),
  "utf8",
);

class FakeClassList {
  add() {}
  remove() {}
  toggle() {}
  contains() { return false; }
}

class FakeElement {
  constructor() {
    this.value = "";
    this.checked = false;
    this.disabled = false;
    this.hidden = false;
    this.textContent = "";
    this.style = {};
    this.dataset = {};
    this.classList = new FakeClassList();
    this.listeners = new Map();
  }
  addEventListener(name, callback) { this.listeners.set(name, callback); }
  querySelector(selector) { return selector === "small" ? {textContent: ""} : null; }
  matches(selector) { return selector.includes(`#${this.id}`); }
  closest(selector) { return this.matches(selector) ? this : null; }
  setAttribute(name, value) { this[name] = String(value); }
  removeAttribute(name) { delete this[name]; }
  insertAdjacentElement(_position, element) {
    if (element?.id) elements.set(element.id, element);
    return element;
  }
}

const weight_names = ["q", "k", "v", "o", "up", "down"];
const elements = new Map();
for (const name of weight_names) {
  elements.set(`${name}_placeholder`, new FakeElement());
  elements.set(`${name}_plot`, new FakeElement());
}
for (const id of [
  "weight_step_from", "weight_step_to", "weight_step_whole_range", "weight_step_current",
  "weight_step_availability", "chart_settings_overlay", "chart_current_weights_only",
  "chart_join_with_line_segments", "chart_inherit_weights_group", "weights_group_scale_field",
  "chart_current_weights_only_field", "chart_join_with_line_segments_field",
]) {
  const element = new FakeElement();
  element.id = id;
  elements.set(id, element);
}
elements.get("chart_settings_overlay").hidden = true;
elements.get("weights_group_scale_field").hidden = true;

const storage = new Map();
const save_json = (key, value) => storage.set(key, JSON.stringify(value));
const load_json = (key, fallback) => storage.has(key) ? JSON.parse(storage.get(key)) : fallback;

save_json("thog2_local_weight_group_settings_v1", {
  "run:A": {current_weights_only: true, join_with_line_segments: true},
  "run:B": {current_weights_only: false, join_with_line_segments: false},
  "run:C": {current_weights_only: false, join_with_line_segments: true},
});

const runs = {
  A: {
    dashboard_run_id: "A", run_state: "running", maximum_update: 25,
    depth_snapshot_count: 25, depth_minimum_update: 1, depth_maximum_update: 25,
    configuration: {
      instrumentation__depth_weight_curves__start_step: 20,
      instrumentation__depth_weight_curves__end_step: 30,
    },
  },
  B: {
    dashboard_run_id: "B", run_state: "finished", maximum_update: 500,
    depth_snapshot_count: 100, depth_minimum_update: 401, depth_maximum_update: 500,
    configuration: {},
  },
  C: {
    dashboard_run_id: "C", run_state: "finished", maximum_update: 500,
    depth_snapshot_count: 100, depth_minimum_update: 401, depth_maximum_update: 500,
    configuration: {
      instrumentation__depth_weight_curves__start_step: 20,
      instrumentation__depth_weight_curves__end_step: 30,
    },
  },
};
const status_responses = {...runs};

let selected_weight = false;
let refresh_resolver = null;
let refresh_calls = 0;
let cleared_step_drafts = 0;
let render_plot_calls = 0;
const click_listeners = [];
const key_listeners = [];
const emit_click = async target => {
  const event = {
    type: "click",
    target,
    preventDefault() {},
    stopImmediatePropagation() { this.stopped = true; },
  };
  for (const callback of click_listeners) {
    await callback(event);
    if (event.stopped) break;
  }
};

const context = {
  console,
  URL,
  depth_weight_chart_names: weight_names,
  app: {
    current_run_id: "A",
    current_status: runs.A,
    runs: Object.values(runs),
    workspace_mode: false,
    axis_chart_name: "q",
    axis_chart_workspace_mode: false,
    chart_settings_render_override: null,
    figures: {depth: {}},
    figure_revision: null,
    weight_current_only: {},
    weight_join_with_line_segments: {},
    refresh_in_flight: false,
  },
  window: {
    location: {origin: "http://127.0.0.1:6007"},
    addEventListener(name, callback) {
      if (name === "load") callback();
      else if (name === "click") click_listeners.push(callback);
      else if (name === "keydown") key_listeners.push(callback);
    },
    __instra_weight_controls_v2: {
      clear_step_range() {},
      selected_step_range: () => ({minimum: 999, maximum: 1000}),
      set_step_range() {},
      global_flags: () => ({current_weights_only: false, join_with_line_segments: false}),
      set_global_flags() {},
    },
    __instra_weight_group_settings: {},
    __instra_matched_weight_selection: {
      selection: () => ({user_selected: selected_weight, model_feature: 3, intermediate_feature: 4}),
      capability: () => ({available: true, maximum: 15}),
    },
    __instra_weight_presentation: {},
    __thog2_dashboard_performance: {state: {}},
    __instra_legacy_heatmap_repair: {},
    __instra_workspace_depth_cache: {clear() {}},
    __instra_clear_weight_step_input_drafts: () => { cleared_step_drafts += 1; },
  },
  document: {
    head: {appendChild() {}},
    addEventListener() {},
    createElement() { return new FakeElement(); },
  },
  by_id: id => elements.get(id) || null,
  load_json,
  save_json,
  localStorage: {
    getItem: key => storage.get(key) ?? null,
    setItem: (key, value) => storage.set(key, String(value)),
  },
  current_run: () => runs[context.app.current_run_id] || null,
  run_identifier: run => run.dashboard_run_id,
  display_run_state: run => run.run_state,
  colour_for_run: () => "#008000",
  fetch_json: async url => {
    const identifier = new URL(url, "http://127.0.0.1").searchParams.get("run");
    if (!status_responses[identifier]) throw new Error("not found");
    return status_responses[identifier];
  },
  show_toast: message => { throw new Error(`unexpected toast: ${message}`); },
  normalize_chart_settings: (chart_name, supplied = null) => ({
    current_weights_only: false,
    join_with_line_segments: false,
    // Model a persisted Instra preference which would otherwise hide all but
    // one curve from an explicitly configured instrumentation window.
    max_snapshots: 1,
    snapshot_window_mode: "from_zero",
    ...(supplied || {}),
  }),
  trace_optimizer_update: trace => trace.meta.optimizer_update,
  retain_latest_weight_snapshots(prepared) {
    const latest = Math.max(...prepared.data.map(trace => trace.meta.optimizer_update));
    prepared.data = prepared.data.filter(trace => trace.meta.optimizer_update === latest);
  },
  instra_enforce_workspace_latest_weights: prepared => prepared,
  limit_curve_snapshots() {},
  apply_thog_line_segments(prepared) {
    for (const trace of prepared.data || []) trace.instra_joined_for_test = true;
  },
  prepare_figure(figure, chart_name) {
    const prepared = JSON.parse(JSON.stringify(figure));
    const settings = context.normalize_chart_settings(chart_name);
    // Simulate the legacy matched-weight layer: current_only chose coordinate family.
    prepared.data = prepared.data.filter(trace => (
      settings.current_weights_only ? trace.meta.kind === "user" : trace.meta.kind === "random"
    ));
    if (settings.current_weights_only) context.retain_latest_weight_snapshots(prepared);
    else if (settings.max_snapshots > 0) {
      const steps = [...new Set(prepared.data.map(trace => trace.meta.optimizer_update))].sort((a, b) => a - b);
      const retained = new Set(
        settings.snapshot_window_mode === "from_zero"
          ? steps.slice(0, settings.max_snapshots)
          : steps.slice(-settings.max_snapshots),
      );
      prepared.data = prepared.data.filter(trace => retained.has(trace.meta.optimizer_update));
    }
    return prepared;
  },
  render_figures: async () => undefined,
  render_plot: async (mount, figure) => {
    render_plot_calls += 1;
    mount.data = JSON.parse(JSON.stringify(figure.data || []));
    mount.dataset.plotReady = "true";
  },
  render_run_heading: () => undefined,
  populate_chart_settings_form: () => undefined,
  sync_chart_setting_outputs: () => undefined,
  open_chart_settings: () => undefined,
  select_run(run_id) {
    context.app.current_run_id = String(run_id);
    context.app.current_status = runs[run_id];
    context.app.figures = null;
    context.refresh_current_run();
  },
  refresh_current_run() {
    refresh_calls += 1;
    if (!refresh_resolver) return Promise.resolve();
    return new Promise(resolve => {
      const prior = refresh_resolver;
      refresh_resolver = null;
      prior(resolve);
    });
  },
  queueMicrotask: callback => callback(),
  setTimeout: callback => { callback(); return 1; },
  clearTimeout() {},
  setInterval: callback => { callback(); return 1; },
  clearInterval() {},
};
context.window.window = context.window;
vm.createContext(context);
vm.runInContext(source, context, {filename: "dashboard_weight_stability_final_patch.js"});

const api = context.window.__instra_weight_stability_final;
assert.ok(api, "final weight stability owner did not install");

// Configured capture bounds are authoritative and clip the retained range. They
// also suppress a persisted current-only setting so the whole captured window is
// visible initially.
assert.deepEqual(api.selected_range(), {minimum: 20, maximum: 25});
assert.equal(api.mode(), "whole");
assert.equal(context.normalize_chart_settings("q").current_weights_only, false);
assert.equal(context.normalize_chart_settings("q").join_with_line_segments, true);
assert.equal(context.normalize_chart_settings("q").max_snapshots, 0);

context.app.current_run_id = "B";
context.app.current_status = runs.A;
assert.deepEqual(
  api.selected_range(),
  null,
  "stale status from the previous run leaked into the selected run's retained bounds",
);
context.app.current_status = runs.B;
context.app.figures = {depth: {}};
assert.equal(api.selected_range(), null);
api.sync_header();
assert.equal(elements.get("weight_step_from").value, "401");
assert.equal(elements.get("weight_step_to").value, "500");
assert.equal(context.normalize_chart_settings("q").current_weights_only, false);
assert.equal(context.normalize_chart_settings("q").join_with_line_segments, false);
assert.equal(api.placeholder_message("q"), "Loading weight curves…");

context.app.current_run_id = "C";
context.app.current_status = runs.C;
context.app.figures = {depth: {}};
assert.equal(api.selected_range(), null);
assert.equal(
  api.placeholder_message("q"),
  "Loading weight curves…",
  "snapshots outside the configured capture window leaked into the view",
);

context.app.current_run_id = "A";
context.app.current_status = runs.A;
context.app.figures = {depth: {}};
assert.deepEqual(api.selected_range(), {minimum: 20, maximum: 25});

// A configured capture window overrides current-only: random history remains
// visible for the complete retained portion of the configured window.
const figure = {
  data: [
    {name: "random-20", line: {color: "#1"}, meta: {optimizer_update: 20, kind: "random"}},
    {name: "random-25", line: {color: "#2"}, meta: {optimizer_update: 25, kind: "random"}},
    {name: "user-20", line: {color: "#3"}, meta: {optimizer_update: 20, kind: "user"}},
    {name: "user-25", line: {color: "#4"}, meta: {optimizer_update: 25, kind: "user"}},
  ],
  layout: {},
};
selected_weight = false;
let prepared = context.prepare_figure(figure, "q");
assert.deepEqual(prepared.data.map(trace => trace.name), ["random-20", "random-25"]);

// With an explicit selected coordinate, history and current-only use the same
// coordinate family; current-only only reduces it to the newest snapshot.
selected_weight = true;
save_json("thog2_local_weight_group_settings_v1", {
  ...load_json("thog2_local_weight_group_settings_v1", {}),
  "run:A": {current_weights_only: false, join_with_line_segments: true},
});
prepared = context.prepare_figure(figure, "q");
assert.deepEqual(prepared.data.map(trace => trace.name), ["user-20", "user-25"]);

save_json("thog2_local_weight_group_settings_v1", {
  ...load_json("thog2_local_weight_group_settings_v1", {}),
  "run:A": {current_weights_only: true, join_with_line_segments: true},
});
prepared = context.prepare_figure(figure, "q");
assert.deepEqual(prepared.data.map(trace => trace.name), ["user-20", "user-25"]);

// Manual range belongs to B only and survives A/B/A switching without leakage.
context.app.current_run_id = "B";
context.app.current_status = runs.B;
context.app.figures = {depth: {}};
api.set_range(450, 460);
assert.deepEqual(api.selected_range(), {minimum: 450, maximum: 460});
context.app.current_run_id = "A";
context.app.current_status = runs.A;
assert.deepEqual(api.selected_range(), {minimum: 20, maximum: 25});
context.app.current_run_id = "B";
context.app.current_status = runs.B;
assert.deepEqual(api.selected_range(), {minimum: 450, maximum: 460});
api.clear_range();
assert.deepEqual(api.selected_range(), {minimum: 401, maximum: 500});
api.show_latest();
assert.deepEqual(api.selected_range(), {minimum: 500, maximum: 500});
assert.ok(cleared_step_drafts > 0, "Whole/Latest did not release protected input drafts");

// Workspace exposes the retained intersection as an explicit final control. It
// remains clickable when no intersection exists so the established red inline
// error can explain why no range was selected.
(async () => {
context.app.workspace_mode = true;
context.window.__instra_workspace = {visible_runs: () => [runs.A, runs.B]};
selected_weight = false;
const inherited_segments = context.prepare_figure({
  data: [
    {name: "A", meta: {optimizer_update: 20, kind: "random", instra_workspace_run_id: "A"}},
    {name: "B", meta: {optimizer_update: 450, kind: "random", instra_workspace_run_id: "B"}},
  ],
  layout: {},
}, "q");
assert.equal(inherited_segments.data.find(trace => trace.name === "A").instra_joined_for_test, true);
assert.equal(inherited_segments.data.find(trace => trace.name === "B").instra_joined_for_test, undefined);
assert.equal(api.workspace_join_explicit("q"), null, "Workspace unexpectedly acquired an explicit segment preference");
save_json("thog2_local_weight_group_settings_v1", {
  ...load_json("thog2_local_weight_group_settings_v1", {}),
  workspace: {current_weights_only: false, join_with_line_segments: false},
});
assert.equal(api.workspace_join_explicit("q"), false);
const explicit_workspace_segments = context.prepare_figure({
  data: [
    {name: "A-explicit", meta: {optimizer_update: 20, kind: "random", instra_workspace_run_id: "A"}},
  ],
  layout: {},
}, "q");
assert.equal(explicit_workspace_segments.data[0].instra_joined_for_test, undefined);
const settings_without_workspace = load_json("thog2_local_weight_group_settings_v1", {});
delete settings_without_workspace.workspace;
save_json("thog2_local_weight_group_settings_v1", settings_without_workspace);
const trace_only_e = {
  dashboard_run_id: "E", run_state: "finished", depth_snapshot_count: 2,
  depth_minimum_update: null, depth_maximum_update: null, configuration: {},
};
context.window.__instra_workspace = {visible_runs: () => [trace_only_e]};
context.app.figures = {depth: {q: {data: [
  {meta: {optimizer_update: 70, instra_workspace_run_id: "E"}},
  {meta: {optimizer_update: 80, instra_workspace_run_id: "E"}},
]}}};
assert.deepEqual(api.available_range(), {minimum: 70, maximum: 80}, "rendered range fallback failed");
context.window.__instra_workspace = {visible_runs: () => [runs.A, runs.B]};
context.app.figures = {depth: {}};
api.sync_header();
const overlap_button = elements.get("weight_step_overlapping_range");
assert.ok(overlap_button, "Workspace overlap button was not installed");
assert.equal(overlap_button.hidden, false);
await emit_click(overlap_button);
assert.equal(elements.get("weight_step_range_error").hidden, false);
assert.equal(elements.get("weight_step_range_error").textContent, "No overlapping retained weight steps.");

api.show_latest();
assert.equal(api.mode(), "latest");
assert.equal(api.selected_range(), null, "Workspace latest incorrectly became one shared step");
const unequal_latest = context.prepare_figure({
  data: [
    {name: "A-20", meta: {optimizer_update: 20, kind: "random", instra_workspace_run_id: "A"}},
    {name: "A-25", meta: {optimizer_update: 25, kind: "random", instra_workspace_run_id: "A"}},
    {name: "B-450", meta: {optimizer_update: 450, kind: "random", instra_workspace_run_id: "B"}},
    {name: "B-500", meta: {optimizer_update: 500, kind: "random", instra_workspace_run_id: "B"}},
  ],
  layout: {},
}, "q");
assert.deepEqual(
  unequal_latest.data.map(trace => trace.name),
  ["A-25", "B-500"],
  "Workspace latest did not retain each run's own final trace",
);

const stale_d = {...runs.B, dashboard_run_id: "D", depth_minimum_update: null, depth_maximum_update: null};
status_responses.D = {...stale_d, depth_minimum_update: 450, depth_maximum_update: 550};
context.app.runs.push(stale_d);
context.window.__instra_workspace = {visible_runs: () => [runs.B, stale_d]};
api.sync_header();
await emit_click(overlap_button);
assert.deepEqual(api.selected_range(), {minimum: 450, maximum: 500});
assert.equal(stale_d.depth_minimum_update, 450, "overlap action did not refresh stale Workspace status");
assert.equal(elements.get("weight_step_range_error").hidden, true);
context.app.workspace_mode = false;
api.sync_header();
assert.equal(overlap_button.hidden, true, "overlap button leaked into a single-run view");
assert.equal(elements.get("weight_step_range_error").hidden, true, "Workspace range error leaked into Runs");

// A cache invalidation may clear app.figures after a valid Plotly render. Keep the
// current context's mounted curves visible, but never trust a mount from another run.
context.app.current_run_id = "B";
context.app.current_status = runs.B;
context.app.figures = {depth: {q: {data: [{meta: {optimizer_update: 500}}]}}};
elements.get("q_plot").dataset.plotReady = "true";
elements.get("q_plot").data = [{meta: {optimizer_update: 500}}];
await context.render_figures();
context.app.figures = {depth: {}};
assert.equal(api.placeholder_message("q"), null, "current mounted curves were covered by loading copy");
const gradient_button = elements.get("weight_step_gradient");
const renders_before_gradient = render_plot_calls;
await emit_click(gradient_button);
assert.equal(
  render_plot_calls,
  renders_before_gradient + 1,
  "gradient did not redraw the current mounted figure after its cache entry was cleared",
);
context.app.current_run_id = "A";
context.app.current_status = runs.A;
assert.equal(
  api.placeholder_message("q"),
  "Loading weight curves…",
  "a mounted chart from another run suppressed the loading state",
);

// A historical run switch starts with an honest loading state, never a future-step
// message inherited from the live run.
refresh_resolver = resolve => { context.__resolve_refresh = resolve; };
context.app.current_run_id = "A";
context.app.current_status = runs.A;
context.app.figures = {depth: {}};
context.select_run("B");
assert.equal(elements.get("q_placeholder").textContent, "Loading weight curves…");
assert.ok(refresh_calls > 0);
context.__resolve_refresh();

console.log("instra weight stability regression: PASS");
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
// ^^^ THOG
