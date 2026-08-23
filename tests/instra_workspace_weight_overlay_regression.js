// vvv THOG
"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const chart_names = [
  "attn_q_head_N",
  "attn_k_head_N",
  "attn_v_head_N",
  "attn_out_head_N",
  "mlp_up",
  "mlp_down",
];
const protocol = "matched_six_v1";
const selected_model = 123;
const selected_intermediate = 145;
let current_only = true;
let user_selected = true;

const deep_clone = value => JSON.parse(JSON.stringify(value));
const trace_kind = trace => trace?.meta?.instra_weight_selection_kind || null;
const selected = trace => (
  ["user", "user_random"].includes(trace_kind(trace))
  && trace.meta.instra_weight_model_feature === selected_model
  && trace.meta.instra_weight_intermediate_feature === selected_intermediate
);

// Simulate the matched-weight filter immediately below the Workspace repair layer.
const base_prepare = figure => {
  const prepared = deep_clone(figure);
  if (!current_only) {
    prepared.data = prepared.data.filter(trace => trace_kind(trace) !== "user");
    return prepared;
  }
  if (!user_selected) {
    prepared.data = prepared.data.filter(trace => trace_kind(trace) !== "user");
    return prepared;
  }
  prepared.data = prepared.data.filter(selected);
  return prepared;
};

const context = {
  console,
  structuredClone: global.structuredClone,
  window: {addEventListener(event, callback) { if (event === "load") callback(); }},
  app: {workspace_mode: true, chart_settings_render_override: null, maximized_chart: null},
  depth_weight_chart_names: chart_names,
  normalize_chart_settings: () => ({current_weights_only: current_only, line_width: 1.25}),
  colour_for_run: run_id => ({A: "#a", B: "#b", C: "#c", D: "#d"})[run_id] || "#x",
  figure_for_chart: () => null,
  prepare_figure: base_prepare,
  setTimeout(callback) { callback(); return 1; },
};
vm.createContext(context);

const repository_root = path.resolve(__dirname, "..");
const repair_path = path.join(
  repository_root,
  "sheet/local_dashboard_assets/dashboard_matched_weight_workspace_repair_patch.js",
);
vm.runInContext(fs.readFileSync(repair_path, "utf8"), context, {filename: repair_path});

const compatible = ({run, kind, model, intermediate, mode = "lines", scalar = null}) => ({
  mode,
  line: mode.includes("lines") ? {width: 9, color: "#wrong"} : undefined,
  marker: mode.includes("markers") ? {color: "#wrong", line: {color: "#wrong"}} : undefined,
  meta: {
    instra_workspace_run_id: run,
    instra_weight_selection_protocol: protocol,
    instra_weight_selection_kind: kind,
    instra_weight_model_feature: model,
    instra_weight_intermediate_feature: intermediate,
    instra_weight_feature_count: 768,
    ...(scalar ? {instra_thog_scalar_id: scalar} : {}),
  },
});
const legacy = ({run, scalar, mode = "lines"}) => ({
  mode,
  line: mode.includes("lines") ? {width: 7, color: "#wrong"} : undefined,
  marker: mode.includes("markers") ? {color: "#wrong", line: {color: "#wrong"}} : undefined,
  meta: {
    instra_workspace_run_id: run,
    instra_thog_scalar_id: scalar,
  },
});

const run_ids = data => [...new Set(
  data.map(trace => trace.meta?.instra_workspace_run_id).filter(Boolean)
)];
const traces_for = (data, run) => data.filter(
  trace => trace.meta?.instra_workspace_run_id === run
);

// A selected Workspace view must never substitute a differently indexed trace.
for (const chart_name of chart_names) {
  const figure = {
    data: [
      compatible({run: "A", kind: "random", model: 1, intermediate: 2}),
      compatible({run: "A", kind: "user", model: selected_model, intermediate: selected_intermediate}),
      compatible({run: "B", kind: "random", model: 7, intermediate: 9, mode: "lines", scalar: "r7_c9"}),
      compatible({run: "B", kind: "random", model: 7, intermediate: 9, mode: "markers", scalar: "r7_c9"}),
    ],
    layout: {},
  };
  const prepared = context.prepare_figure(figure, chart_name);
  assert.deepEqual(
    run_ids(prepared.data).sort(),
    ["A"],
    `${chart_name}: a run without the selected coupling remained visible`,
  );
  assert.equal(
    traces_for(prepared.data, "A").length,
    1,
    `${chart_name}: selected run gained duplicate fallback`,
  );
  assert.equal(
    traces_for(prepared.data, "A")[0].meta.instra_weight_selection_fallback,
    undefined,
  );
  assert.equal(traces_for(prepared.data, "B").length, 0);
  assert.ok(prepared.data.every(trace => trace.meta.instra_weight_selection_fallback !== true));
}

// Reproduce the four-eyed-run/maximized screenshot, including one legacy run.
for (const chart_name of chart_names) {
  const figure = {
    data: [
      compatible({run: "A", kind: "random", model: 1, intermediate: 2}),
      compatible({run: "A", kind: "user", model: selected_model, intermediate: selected_intermediate}),
      compatible({run: "B", kind: "random", model: 7, intermediate: 9}),
      legacy({run: "C", scalar: "r4_c5", mode: "lines"}),
      legacy({run: "C", scalar: "r4_c5", mode: "markers"}),
      compatible({run: "D", kind: "random", model: 20, intermediate: 21}),
      compatible({run: "D", kind: "user", model: selected_model, intermediate: selected_intermediate}),
    ],
    layout: {},
  };
  context.app.maximized_chart = chart_name;
  const prepared = context.prepare_figure(figure, chart_name);
  assert.deepEqual(
    run_ids(prepared.data).sort(),
    ["A", "D"],
    `${chart_name}: the selected view substituted an incompatible coupling`,
  );
  assert.equal(traces_for(prepared.data, "A").length, 1);
  assert.equal(traces_for(prepared.data, "D").length, 1);
  assert.equal(traces_for(prepared.data, "B").length, 0);
  assert.equal(traces_for(prepared.data, "C").length, 0);
  assert.ok(prepared.data.every(trace => trace.meta.instra_weight_selection_fallback !== true));
}

// Current-only off: ordinary/random history passes through and gets no fallback tag.
current_only = false;
user_selected = true;
let figure = {
  data: [
    compatible({run: "A", kind: "random", model: 1, intermediate: 2}),
    compatible({run: "A", kind: "user", model: selected_model, intermediate: selected_intermediate}),
    compatible({run: "B", kind: "random", model: 7, intermediate: 9}),
  ],
  layout: {},
};
let prepared = context.prepare_figure(figure, "attn_q_head_N");
assert.deepEqual(run_ids(prepared.data).sort(), ["A", "B"]);
assert.ok(prepared.data.every(trace => trace.meta.instra_weight_selection_fallback !== true));

// Random mode: every run's recorded random trace remains visible without fallback tagging.
current_only = true;
user_selected = false;
prepared = context.prepare_figure(figure, "attn_q_head_N");
assert.deepEqual(run_ids(prepared.data).sort(), ["A", "B"]);
assert.ok(prepared.data.every(trace => trace.meta.instra_weight_selection_fallback !== true));

// Runs view remains untouched by this Workspace-only repair.
user_selected = true;
context.app.workspace_mode = false;
prepared = context.prepare_figure(
  {data: [compatible({run: "B", kind: "random", model: 7, intermediate: 9})], layout: {}},
  "attn_q_head_N",
);
assert.equal(prepared.data.length, 0, "Workspace repair leaked into Runs view");

console.log("instra workspace weight overlay regression: PASS");
// ^^^ THOG
