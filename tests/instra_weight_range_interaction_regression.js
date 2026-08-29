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
const superseded_source = fs.readFileSync(
  path.join(repository_root, "sheet/local_dashboard_assets/dashboard_weight_regression_final_patch.js"),
  "utf8",
);

assert.match(
  superseded_source,
  /if \(window\.__instra_weight_range_interaction_final\) return;[\s\S]*?event\.preventDefault\(\);/,
  "the superseded RND capture listener still blocks the run-aware final owner",
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
    instra_workspace_artifact_name: "260824-0901_scruffy_THOG_ANALOG_DENSE",
    instra_workspace_run_datetime: "260824-0901",
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
storage.set(
  "thog2_local_weight_viewer_couplings_v2",
  JSON.stringify({"run:R1": {model_feature: 10, intermediate_feature: 10}}),
);
const capture_selection = {
  protocol: "matched_six_v1",
  user_selected: true,
  model_feature: 12,
  intermediate_feature: 14,
};
let capture_save_count = 0;
let gradient_enabled = false;
const rendered = new Map();
const app = {
  current_run_id: "R1",
  current_status: {run_state: "finished"},
  workspace_mode: false,
  chart_settings_render_override: null,
  maximized_chart: null,
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
      gradient_enabled: () => gradient_enabled,
    },
    __instra_matched_weight_selection: {
      selection: () => ({...capture_selection}),
      capability: () => ({available: true, maximum: 15}),
      async save(model_feature, intermediate_feature) {
        capture_save_count += 1;
        capture_selection.user_selected = true;
        capture_selection.model_feature = model_feature;
        capture_selection.intermediate_feature = intermediate_feature;
        return {...capture_selection};
      },
    },
    __instra_weight_coupling_reliability_final: {installed: true},
  },
  document: {
    activeElement: null,
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
  colour_for_run: run_id => run_id === "R2" ? "#2563EB" : "#E8790C",
  hex_to_rgb(hex) {
    const match = /^#?([0-9a-f]{6})$/i.exec(String(hex));
    return match ? [0, 2, 4].map(index => parseInt(match[1].slice(index, index + 2), 16)) : null;
  },
  rgb_to_hex(rgb) {
    return `#${rgb.map(value => value.toString(16).padStart(2, "0")).join("")}`.toUpperCase();
  },
  current_run: () => app.current_status,
  display_run_state: run => String(run?.run_state || ""),
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
  assert.deepEqual(
    JSON.parse(storage.get("thog2_local_weight_viewer_couplings_v2"))["run:R1"],
    {model_feature: 12, intermediate_feature: 14},
    "completed view retained a stale valid-but-unrecorded coupling",
  );

  // Synchronisation must not replace either box while the user is editing the
  // pair. This was the source of the persistent 1010/old-value snapback.
  context.document.activeElement = elements.get("weight_coupling_input");
  elements.get("weight_coupling_input").value = "7";
  api.sync();
  assert.equal(elements.get("weight_coupling_input").value, "7");
  assert.equal(elements.get("weight_coupling_output").value, "14");

  // Tabbing from the first box into the second must defer the pair commit so
  // that the two fields are accepted together.
  context.document.activeElement = elements.get("weight_coupling_output");
  emit("change", elements.get("weight_coupling_input"));
  assert.equal(elements.get("weight_coupling_input").value, "7");
  assert.equal(elements.get("weight_coupling_output").value, "14");
  context.document.activeElement = null;
  api.sync();
  assert.equal(elements.get("weight_coupling_input").value, "12");

  const before_values = rendered.get("attn_q_head_N").data.map(trace => trace.y[0]);
  const random_values = [0.14, 0.21];
  context.Math = Object.create(Math);
  context.Math.random = () => random_values.shift();
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
  assert.equal(capture_save_count, 0, "recorded RND selection scheduled a redundant capture");
  const after_values = rendered.get("attn_q_head_N").data.map(trace => trace.y[0]);
  assert.notDeepEqual(after_values, before_values, "RND changed boxes without changing curve values");

  emit("click", elements.get("weight_residual_plus"));
  await Promise.resolve();
  await Promise.resolve();
  assert.deepEqual(api.pair(), {model_feature: 12, intermediate_feature: 14});
  emit("click", elements.get("weight_residual_minus"));
  await Promise.resolve();
  await Promise.resolve();
  assert.deepEqual(api.pair(), {model_feature: 2, intermediate_feature: 3});
  assert.equal(capture_save_count, 0, "recorded +/- selection scheduled a redundant capture");

  const prepared = rendered.get("attn_q_head_N");
  assert.equal(prepared.data.length, 2, "two-step viewer dropped a recorded curve");
  assert.equal(new Set(prepared.data.map(trace => trace.line.color)).size, 2, "steps reused one curve colour");
  assert.ok(prepared.data.every(trace => (
    trace.hovertemplate.includes(`step ${trace.meta.instra_thog_optimizer_update}`)
  )), "step number missing from hover");
  assert.ok(prepared.data.every(trace => {
    const rows = trace.hovertemplate.split("<extra", 1)[0].split("<br>");
    return rows[0] === "<b>260824-0901</b>"
      && rows[1] === `step ${trace.meta.instra_thog_optimizer_update}`;
  }), "compact hover did not put run datetime first and step second");

  app.maximized_chart = "attn_q_head_N";
  const maximized = context.prepare_figure(figures.attn_q_head_N, "attn_q_head_N");
  assert.ok(maximized.data.every(trace => {
    const rows = trace.hovertemplate.split("<extra", 1)[0].split("<br>");
    return rows[0] === "<b>260824-0901_scruffy_THOG_ANALOG_DENSE</b>"
      && rows[1] === `step ${trace.meta.instra_thog_optimizer_update}`;
  }), "maximized hover did not retain the full artifact with step second");
  app.maximized_chart = null;

  gradient_enabled = true;
  const gradient_figure = {
    data: [
      make_trace("attn_q_head_N", 10, 2, 3, 10, "random"),
      make_trace("attn_q_head_N", 15, 2, 3, 15, "random"),
      make_trace("attn_q_head_N", 20, 2, 3, 20, "random"),
    ],
    layout: {},
  };
  const gradient = context.prepare_figure(gradient_figure, "attn_q_head_N");
  const gradient_colours = gradient.data.map(trace => trace.line.color.toUpperCase());
  const luminance = colour => {
    const [red, green, blue] = context.hex_to_rgb(colour);
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue;
  };
  assert.equal(gradient_colours[1], "#E8790C", "gradient midpoint is not the exact run colour");
  assert.ok(luminance(gradient_colours[0]) > luminance(gradient_colours[1]), "earliest gradient curve is not lighter");
  assert.ok(luminance(gradient_colours[2]) < luminance(gradient_colours[1]), "latest gradient curve is not darker");

  app.workspace_mode = true;
  const workspace_gradient_figure = {
    data: ["R1", "R2"].flatMap(run_id => [10, 15, 20].map(step => ({
      ...make_trace("attn_q_head_N", step, 2, 3, step, "random"),
      meta: {
        ...make_trace("attn_q_head_N", step, 2, 3, step, "random").meta,
        instra_workspace_run_id: run_id,
      },
    }))),
    layout: {},
  };
  const workspace_gradient = context.prepare_figure(workspace_gradient_figure, "attn_q_head_N");
  for (const run_id of ["R1", "R2"]) {
    const traces = workspace_gradient.data.filter(trace => trace.meta.instra_workspace_run_id === run_id);
    const colours = traces.map(trace => trace.line.color.toUpperCase());
    const base = context.colour_for_run(run_id).toUpperCase();
    assert.equal(colours[1], base, `${run_id} gradient midpoint is not its exact run colour`);
    assert.equal(new Set(colours).size, 3, `${run_id} did not receive its own three-colour gradient`);
    assert.ok(luminance(colours[0]) > luminance(colours[1]), `${run_id} earliest curve is not lighter`);
    assert.ok(luminance(colours[2]) < luminance(colours[1]), `${run_id} latest curve is not darker`);
  }
  app.workspace_mode = false;
  gradient_enabled = false;

  elements.get("weight_coupling_input").value = "7";
  elements.get("weight_coupling_output").value = "8";
  emit("change", elements.get("weight_coupling_output"));
  assert.deepEqual(api.pair(), {model_feature: 2, intermediate_feature: 3});
  assert.equal(elements.get("weight_coupling_input").value, "2");
  assert.equal(elements.get("weight_coupling_output").value, "3");
  assert.match(elements.get("weight_coupling_view_error").textContent, /was not recorded/);
  await Promise.resolve();

  app.current_status.run_state = "running";
  elements.get("weight_coupling_input").value = "7";
  elements.get("weight_coupling_output").value = "8";
  emit("change", elements.get("weight_coupling_output"));
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
  assert.deepEqual(api.pair(), {model_feature: 7, intermediate_feature: 8});
  assert.equal(capture_save_count, 1, "valid active-run coupling was not scheduled");
  assert.match(elements.get("weight_coupling_view_error").textContent, /next recorded snapshot/);

  api.select_pair(2, 3);
  context.Math.random = () => 0.4;
  emit("click", elements.get("weight_random_jump"));
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
  assert.deepEqual(api.pair(), {model_feature: 7, intermediate_feature: 7});
  assert.equal(capture_save_count, 2, "active-run RND was restricted to recorded couplings");
  api.select_pair(2, 3);

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
