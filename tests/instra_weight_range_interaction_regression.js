// vvv THOG
"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const repository_root = path.resolve(__dirname, "..");
const source = fs.readFileSync(
  path.join(repository_root, "sheet/local_dashboard_assets/dashboard_weight_range_interaction_final_patch.js"),
  "utf8",
);

class FakeElement {
  constructor(id = "") {
    this.id = id;
    this.value = "";
    this.disabled = false;
    this.hidden = false;
    this.textContent = "";
    this.title = "";
    this.dataset = {};
    this.style = {};
  }
  matches(selector) {
    return selector.split(",").some(part => part.trim() === `#${this.id}`);
  }
  closest(selector) { return this.matches(selector) ? this : null; }
  setAttribute(name, value) { this[name] = String(value); }
  insertAdjacentElement(_position, element) {
    if (element?.id) elements.set(element.id, element);
    return element;
  }
}

class FakeInputElement extends FakeElement {
  get value() { return this._value || ""; }
  set value(value) { this._value = String(value); }
}

const chart_names = [
  "attn_q_head_N", "attn_k_head_N", "attn_v_head_N",
  "attn_out_head_N", "mlp_up", "mlp_down",
];
const elements = new Map();
for (const id of [
  "weight_coupling_input", "weight_coupling_output", "weight_coupling_editor",
  "weight_residual_minus", "weight_residual_plus", "weight_branch_minus",
  "weight_branch_plus", "weight_random_jump", "chart_settings_overlay",
]) {
  const element = id.includes("coupling_input") || id.includes("coupling_output")
    ? new FakeInputElement(id)
    : new FakeElement(id);
  elements.set(id, element);
}
elements.get("chart_settings_overlay").hidden = true;

const make_trace = (chart_name, step, model_feature, intermediate_feature, value, kind) => ({
  type: "scatter",
  mode: "lines",
  name: `${chart_name}-${model_feature}-${intermediate_feature}-${step}`,
  x: [1, 2],
  y: [value, value + 0.5],
  line: {color: "#2563eb", width: 1},
  hovertemplate: "layer %{x}<br>weight %{y}<extra></extra>",
  meta: {
    instra_thog_weight: true,
    instra_thog_optimizer_update: step,
    instra_weight_selection_protocol: "matched_six_v1",
    instra_weight_selection_kind: kind,
    instra_weight_model_feature: model_feature,
    instra_weight_intermediate_feature: intermediate_feature,
    instra_weight_feature_count: 16,
  },
});

const figures = Object.fromEntries(chart_names.map((chart_name, chart_index) => [chart_name, {
  data: [
    make_trace(chart_name, 10, 2, 3, 10 + chart_index, "random"),
    make_trace(chart_name, 10, 12, 14, 110 + chart_index, "user"),
    make_trace(chart_name, 11, 2, 3, 20 + chart_index, "random"),
    make_trace(chart_name, 11, 12, 14, 120 + chart_index, "user"),
  ],
  layout: {},
}]));

const listeners = new Map();
const add_listener = (name, callback) => {
  if (!listeners.has(name)) listeners.set(name, []);
  listeners.get(name).push(callback);
};
const emit = (name, target, extra = {}) => {
  const event = {
    target,
    preventDefault() {},
    stopImmediatePropagation() { this.stopped = true; },
    ...extra,
  };
  for (const callback of listeners.get(name) || []) {
    callback(event);
    if (event.stopped) break;
  }
};

const storage = new Map();
const capture_selection = {
  protocol: "matched_six_v1",
  user_selected: true,
  model_feature: 12,
  intermediate_feature: 14,
};
const rendered = new Map();
const app = {
  current_run_id: "R1",
  workspace_mode: false,
  chart_settings_render_override: null,
};

const context = {
  console,
  app,
  HTMLInputElement: FakeInputElement,
  window: {
    addEventListener(name, callback) {
      if (name === "load") callback();
      else add_listener(name, callback);
    },
    __instra_weight_stability_final: {
      context_key: () => "run:R1",
      selected_range: () => ({minimum: 10, maximum: 11}),
    },
    __instra_matched_weight_selection: {
      selection: () => ({...capture_selection}),
    },
    __instra_weight_coupling_reliability_final: {installed: true},
  },
  document: {
    head: {appendChild() {}},
    createElement() { return new FakeElement(); },
  },
  by_id: id => elements.get(id) || null,
  figure_for_chart: chart_name => figures[chart_name] || null,
  trace_optimizer_update: trace => Number(trace?.meta?.instra_thog_optimizer_update),
  load_json: (key, fallback) => storage.has(key) ? JSON.parse(storage.get(key)) : fallback,
  save_json: (key, value) => storage.set(key, JSON.stringify(value)),
  localStorage: {
    getItem: key => storage.get(key) ?? null,
    setItem: (key, value) => storage.set(key, String(value)),
  },
  normalize_chart_settings: (_chart_name, supplied = null) => ({
    current_weights_only: false,
    ...(supplied || {}),
  }),
  colour_for_run: () => "#e8790c",
  prepare_figure(figure, chart_name) {
    const prepared = JSON.parse(JSON.stringify(figure));
    const viewer = context.window.__instra_weight_viewer_selection?.selection?.() || capture_selection;
    prepared.data = prepared.data.filter(trace => (
      trace.meta.instra_weight_model_feature === viewer.model_feature
      && trace.meta.instra_weight_intermediate_feature === viewer.intermediate_feature
    ));
    const override = app.chart_settings_render_override;
    if (override?.chart_name === chart_name && override.settings?.current_weights_only === true) {
      const latest = Math.max(...prepared.data.map(trace => trace.meta.instra_thog_optimizer_update));
      prepared.data = prepared.data.filter(trace => trace.meta.instra_thog_optimizer_update === latest);
    }
    return prepared;
  },
  async render_figures() {
    for (const chart_name of chart_names) {
      rendered.set(chart_name, context.prepare_figure(figures[chart_name], chart_name));
    }
  },
  render_run_heading() {},
  queueMicrotask(callback) { callback(); },
  setTimeout(callback) { callback(); return 1; },
  setInterval(callback) { callback(); return 1; },
  clearInterval() {},
};
context.window.window = context.window;

(async () => {
  vm.createContext(context);
  vm.runInContext(source, context, {filename: "dashboard_weight_range_interaction_final_patch.js"});
  await Promise.resolve();
  await Promise.resolve();

  const api = context.window.__instra_weight_viewer_selection;
  assert.ok(api, "viewer coupling API did not install");
  assert.deepEqual(api.recorded_pairs(), [
    {model_feature: 2, intermediate_feature: 3},
    {model_feature: 12, intermediate_feature: 14},
  ]);
  assert.deepEqual(api.pair(), {model_feature: 12, intermediate_feature: 14});
  assert.equal(elements.get("weight_coupling_input").value, "12");
  assert.equal(elements.get("weight_coupling_output").value, "14");

  const before_values = rendered.get("attn_q_head_N").data.map(trace => trace.y[0]);
  emit("click", elements.get("weight_random_jump"));
  await Promise.resolve();
  await Promise.resolve();
  assert.deepEqual(api.pair(), {model_feature: 2, intermediate_feature: 3});
  assert.deepEqual(capture_selection, {
    protocol: "matched_six_v1",
    user_selected: true,
    model_feature: 12,
    intermediate_feature: 14,
  }, "viewer RND mutated the trainer capture setting");
  const after_values = rendered.get("attn_q_head_N").data.map(trace => trace.y[0]);
  assert.notDeepEqual(after_values, before_values, "RND changed boxes without changing curve values");

  const prepared = rendered.get("attn_q_head_N");
  assert.equal(prepared.data.length, 2, "two-step viewer dropped a recorded curve");
  assert.equal(new Set(prepared.data.map(trace => trace.line.color)).size, 2, "steps reused one curve colour");
  assert.ok(prepared.data.every(trace => (
    trace.hovertemplate.includes(`step ${trace.meta.instra_thog_optimizer_update}`)
  )), "step number missing from hover");

  elements.get("weight_coupling_input").value = "7";
  elements.get("weight_coupling_output").value = "8";
  emit("change", elements.get("weight_coupling_output"));
  assert.deepEqual(api.pair(), {model_feature: 2, intermediate_feature: 3});
  assert.equal(elements.get("weight_coupling_input").value, "2");
  assert.equal(elements.get("weight_coupling_output").value, "3");
  assert.match(elements.get("weight_coupling_view_error").textContent, /was not recorded/);

  elements.get("chart_settings_overlay").hidden = false;
  app.chart_settings_render_override = {
    chart_name: "attn_q_head_N",
    settings: {current_weights_only: true},
  };
  let preview = context.prepare_figure(figures.attn_q_head_N, "attn_q_head_N");
  assert.equal(preview.data.length, 1, "Current-only preview did not show its latest curve");
  app.chart_settings_render_override.settings.current_weights_only = false;
  preview = context.prepare_figure(figures.attn_q_head_N, "attn_q_head_N");
  assert.equal(preview.data.length, 2, "history preview was blank or incomplete");

  console.log("instra weight range interaction regression: PASS");
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
// ^^^ THOG
