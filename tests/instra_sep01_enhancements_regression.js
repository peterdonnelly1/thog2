// vvv THOG
"use strict";
const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const assets = path.join(__dirname, "../sheet/local_dashboard_assets");
const further = fs.readFileSync(path.join(assets, "dashboard_instra_further_enhancements_patch.js"), "utf8");
const overview = fs.readFileSync(path.join(assets, "dashboard_heatmap_patch.js"), "utf8");
const elements = new Map();
const charts = ["attn_q_head_N", "attn_k_head_N", "attn_v_head_N", "attn_out_head_N", "mlp_up", "mlp_down"];
let visible_runs = ["a", "b", "c"].map(id => ({id}));
const app = {workspace_mode: true, figures: {depth: {}}, runs: []};
const render_calls = [];
const make_element = () => ({
  hidden: false, disabled: false, handlers: {},
  setAttribute(name, value) { this[name] = value; },
  addEventListener(name, callback) { this.handlers[name] = callback; },
  insertAdjacentElement(_position, element) { elements.set(element.id, element); },
});
elements.set("weight_step_gradient", make_element());
const context = vm.createContext({app, window: {__instra_workspace: {visible_runs: () => visible_runs}},
  run_identifier: run => run.id, by_id: id => elements.get(id), document: {createElement: make_element},
  weight_chart_names: charts, render_plot: async (mount, figure, chart_name) => {
    const prepared = JSON.parse(JSON.stringify(figure));
    context.order_workspace_weight_traces(prepared);
    mount.data = prepared.data;
    render_calls.push(chart_name);
  }, local_first_present: () => "-"});
const start = further.indexOf("  const workspace_z_order = [];");
const end = further.indexOf("  const polish_weight_header =", start);
assert.ok(start > 0 && end > start);
vm.runInContext(further.slice(start, end) + "\nthis.ensure_z_cycle = ensure_z_cycle; this.order_workspace_weight_traces = order_workspace_weight_traces;", context);
vm.runInContext(overview.slice(overview.indexOf("function local_dense_snapshot_metadata"), overview.indexOf("function local_render_artifacts")), context);
const trace = (id, step) => ({meta: {instra_workspace_run_id: id}, step});
const source = {data: [trace("a", 0), trace("a", 1), trace("b", 0), trace("b", 1), trace("c", 0), trace("c", 1)]};
for (const chart_name of charts) {
  app.figures.depth[chart_name] = source;
  elements.set(`${chart_name}_plot`, {});
}
(async () => {
  context.ensure_z_cycle();
  const button = elements.get("weight_z_cycle");
  assert.equal(button.textContent, "z");
  assert.equal(button.hidden, false);
  const original = JSON.stringify(source);
  for (const front of ["a", "b", "c", "a"]) {
    await button.handlers.click();
    for (const chart_name of charts) {
      const data = elements.get(`${chart_name}_plot`).data;
      assert.equal(data.at(-1).meta.instra_workspace_run_id, front);
      assert.deepEqual(data.filter(item => item.meta.instra_workspace_run_id === front).map(item => item.step), [0, 1]);
    }
    const refreshed = JSON.parse(original);
    context.order_workspace_weight_traces(refreshed);
    assert.equal(refreshed.data.at(-1).meta.instra_workspace_run_id, front);
  }
  assert.equal(JSON.stringify(source), original, "draw cycling mutated the stored figure");
  visible_runs.reverse();
  app.maximized_chart = "mlp_up";
  await button.handlers.click();
  assert.equal(elements.get("mlp_up_plot").data.at(-1).meta.instra_workspace_run_id, "b");
  assert.equal(elements.get("weight_z_cycle"), button, "magnification duplicated the control");
  assert.equal(render_calls.length, 30);
  app.workspace_mode = false;
  context.ensure_z_cycle();
  assert.equal(button.hidden, true);
  const single = JSON.parse(original);
  context.order_workspace_weight_traces(single);
  assert.equal(JSON.stringify(single), original);
  app.workspace_mode = true;
  visible_runs = [{id: "b"}];
  context.ensure_z_cycle();
  assert.equal(button.disabled, true);

  const metadata = {effective_initialisation: "dense_snapshot", snapshot_path: "/dense_baseline_snapshots/source.pt",
    tensor_payload_hash: "weights", compatibility_hash: "physical", snapshot_hyperparameters: {physical_layer_count: 16},
    source_hyperparameters: {learning_rate: 0.001, n_layer: 16}};
  const details = context.local_dense_snapshot_details({lifecycle: {dense_snapshot_baselining: metadata}});
  assert.equal(details.filename, "source.pt");
  assert.equal(details.parameters.learning_rate, 0.001);
  assert.equal(details.parameters["snapshot.physical_layer_count"], 16);
  assert.equal(context.local_dense_snapshot_details({}).filename, "-");
  assert.equal(context.local_dense_snapshot_details({save_dense_initialisation_snapshot: true,
    parameter_report: {dense_snapshot_baselining: {...metadata, effective_initialisation: "ordinary_dense_initialisation"}}}).filename, "-");
  const legacy = {...metadata}; delete legacy.source_hyperparameters;
  app.runs = [{configuration: {learning_rate: 0.002, parameter_report: {dense_snapshot_baselining: {
    ...legacy, effective_initialisation: "ordinary_dense_initialisation"}}}}];
  assert.equal(context.local_dense_snapshot_details({dense_snapshot_baselining: legacy}).parameters.learning_rate, 0.002);
  app.runs[0].configuration.parameter_report.dense_snapshot_baselining.tensor_payload_hash = "unrelated";
  assert.match(context.local_dense_snapshot_details({dense_snapshot_baselining: legacy}).parameters.source_hyperparameters, /Not recorded/);
  assert.doesNotMatch(overview, /local_render_artifacts\(by_id\("overview_artifact_outputs"\)/);
  assert.doesNotMatch(overview, /id="overview_artifact_outputs"/);
  assert.match(overview, /id="overview_snapshot_panel"/);
  console.log("instra September 1 behaviour regression: PASS");
})().catch(error => { console.error(error); process.exitCode = 1; });
// ^^^ THOG
