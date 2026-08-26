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

let selected_weight = false;
let refresh_resolver = null;
let refresh_calls = 0;
let cleared_step_drafts = 0;
const click_listeners = [];
const key_listeners = [];
const emit_click = target => {
  const event = {
    type: "click",
    target,
    preventDefault() {},
    stopImmediatePropagation() { this.stopped = true; },
  };
  for (const callback of click_listeners) {
    callback(event);
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
assert.equal(api.placeholder_message("q"), "Weight curves unavailable.");

context.app.current_run_id = "C";
context.app.current_status = runs.C;
context.app.figures = {depth: {}};
assert.equal(api.selected_range(), null);
assert.equal(
  api.placeholder_message("q"),
  "Weight curves unavailable.",
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
context.app.workspace_mode = true;
context.window.__instra_workspace = {visible_runs: () => [runs.A, runs.B]};
api.sync_header();
const overlap_button = elements.get("weight_step_overlapping_range");
assert.ok(overlap_button, "Workspace overlap button was not installed");
assert.equal(overlap_button.hidden, false);
emit_click(overlap_button);
assert.equal(elements.get("weight_step_range_error").hidden, false);
assert.equal(elements.get("weight_step_range_error").textContent, "No overlapping retained weight steps.");

context.window.__instra_workspace = {visible_runs: () => [
  runs.B,
  {...runs.B, dashboard_run_id: "D", depth_minimum_update: 450, depth_maximum_update: 550},
]};
api.sync_header();
emit_click(overlap_button);
assert.deepEqual(api.selected_range(), {minimum: 450, maximum: 500});
assert.equal(elements.get("weight_step_range_error").hidden, true);
context.app.workspace_mode = false;
api.sync_header();
assert.equal(overlap_button.hidden, true, "overlap button leaked into a single-run view");

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
// ^^^ THOG
