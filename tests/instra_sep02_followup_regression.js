// vvv THOG
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const assets = path.join(__dirname, "../sheet/local_dashboard_assets");
const read = name => fs.readFileSync(path.join(assets, name), "utf8");
const data = require(path.join(assets, "dashboard_weight_inspector.js"));
const compare = values => {
  const runs = values.map((_value, index) => ({id: String(index), name: String(index)}));
  const traces = values.map((value, index) => ({name: "step 1", x: [1], y: [value], meta: {instra_workspace_run_id: String(index), instra_weight_model_feature: 0, instra_weight_intermediate_feature: 0}}));
  return data.build_table({data: traces}, {data: traces}, runs, "mlp_up");
};
for (const [values, expected] of [[[1, 3], 100], [[-3, -1], 100], [[-1, 3], 400], [[0, 0], 0], [[-2, -2], 0]]) {
  const model = compare(values);
  assert.ok(Math.abs(data.value_at(model, 0, 0) - expected) < 1e-12);
  assert.equal(data.cell_text(model, 0, 0, 2), `${expected.toFixed(2)}%`);
}
for (const values of [[-1, 1], [-1, 0, 1], [1, null]]) {
  const model = compare(values);
  assert.equal(data.value_at(model, 0, 0), null);
  assert.equal(data.cell_text(model, 0, 0, 4), "—");
  assert.equal(data.tsv(model, {anchor: {row: 0, column: 0}, focus: {row: 0, column: 0}}, 4), "");
}
assert.ok(Number.isFinite(data.value_at(compare([1e308, 1.5e308]), 0, 0)), "large finite weights overflowed the mean");
assert.ok(Math.abs(Number(data.csv(compare([1, 3])).split("\r\n")[1].split(",")[3]) - 100) < 1e-12);

// Execute the actual clear, navigation capture/restore, select_run and polling
// functions. The chart renderer is stubbed; rebuilding/removing groups is real
// within this small DOM model, unlike the earlier state-map-only test.
const source = read("dashboard_wandb_groups_patch.js");
const app = {current_run_id: "a", maximized_chart: "local_metric_train_loss", dynamic_chart_figures: {}, dynamic_chart_metadata: {}};
const viewport = {hidden: false, scrollTop: 240, getBoundingClientRect: () => ({top: 100}), addEventListener() {}};
let sections = [];
let mounted = new Map();
let restore_count = 0;
const timers = [];
const frames = [];
const saved_groups = new Map([["train", false], ["val", false], ["system", true]]);
const section = (name, top) => ({
  dataset: {metricGroup: name, chartGroup: name},
  classList: {contains: () => saved_groups.get(name) ?? true},
  getBoundingClientRect: () => ({top: 100 + top - viewport.scrollTop, bottom: 100 + top + 365 - viewport.scrollTop}),
  querySelectorAll: selector => selector === ".local-metric-card" ? [{dataset: {chart: `local_metric_${name}_loss`}}] : [],
  remove() { sections = sections.filter(value => value !== this); mounted.delete(`local_metric_${name}_loss`); },
});
const context = vm.createContext({
  app, console, CSS: {escape: value => value}, chart_titles: {},
  group_revisions: new Map(), rendered_revisions: new Map(),
  by_id: id => id === "charts_scroll" ? viewport : null,
  metric_group_sections: () => sections,
  group_section: name => sections.find(value => value.dataset.metricGroup === name),
  save_group_collapsed: (name, collapsed) => saved_groups.set(name, collapsed),
  current_view_key: () => app.current_run_id, workspace_api: () => null,
  document: {
    querySelectorAll: selector => selector === ".chart-group" ? sections : [],
    querySelector: selector => [...mounted].find(([key]) => selector.includes(`"${key}"`))?.[1] || null,
  },
  restore_maximized_chart() { restore_count++; app.maximized_chart = null; },
  toggle_maximized_chart: key => { app.maximized_chart = key; },
  select_run: id => { app.current_run_id = id; viewport.scrollTop = 0; },
  fetch_json: async () => ({available: true, groups: [{name: "train"}, {name: "val"}]}),
  sync_group_order: () => { sections = [section("train", 40), section("val", 440)]; },
  refresh_group_data: async name => { mounted.set(`local_metric_${name}_loss`, {}); },
  show_toast: text => { throw new Error(text); },
  setTimeout: callback => timers.push(callback), requestAnimationFrame: callback => frames.push(callback),
});
vm.runInContext("let pending_navigation = null; let poll_in_flight = false; let last_run_id = 'a';\n" + source.slice(source.indexOf("    const clear_metric_groups ="), source.indexOf("    const sorted_group_summaries =")) + source.slice(source.indexOf("    const refresh_metric_groups ="), source.indexOf("    const base_local_apply_detail_tab_metric_groups")), context);
const install_old = () => { sections = [section("train", 80), section("val", 480)]; mounted.set("local_metric_train_loss", {}); };
(async () => {
  install_old();
  context.select_run("b");
  assert.equal(restore_count, 1, "production teardown was not exercised");
  assert.equal(app.maximized_chart, null);
  await timers.shift()();
  while (frames.length) frames.shift()();
  assert.equal(app.maximized_chart, "local_metric_train_loss", "maximized scalar chart was lost during run switch");
  assert.equal(saved_groups.get("train"), false);
  assert.equal(saved_groups.get("val"), false);
  app.maximized_chart = null;
  viewport.scrollTop = 240;
  install_old();
  const old_offset = sections[0].getBoundingClientRect().top - 100;
  context.select_run("c");
  await timers.shift()();
  while (frames.length) frames.shift()();
  assert.equal(sections[0].getBoundingClientRect().top - 100, old_offset, "run switch scrolled away from train to weights");
  app.maximized_chart = "local_metric_train_loss";
  context.select_run("d");
  await timers.shift()();
  app.current_run_id = "newer_run";
  while (frames.length) frames.shift()();
  assert.equal(app.maximized_chart, null, "obsolete frame maximized a chart in a newer run");

  const icon_context = vm.createContext({});
  const dashboard = read("dashboard.js");
  vm.runInContext(dashboard.slice(dashboard.indexOf("function chart_size_icon"), dashboard.indexOf("const chart_titles")), icon_context);
  assert.match(icon_context.chart_size_icon(), /<rect/);
  assert.match(icon_context.chart_size_icon(true), /<path.*<rect/);
  assert.notEqual(icon_context.chart_size_icon(), icon_context.chart_size_icon(true));
  assert.match(source, /header\.appendChild\(cycle\)/);
  assert.match(source, /left: 50%; top: 50%; transform: translate\(-50%, -50%\)/);
  assert.match(source, /maximized > \.chart-card-header > \.metric-z-cycle:not\(\[hidden\]\)/);
  console.log("instra follow-up percentage/navigation/icons regression: PASS");
})().catch(error => { console.error(error); process.exitCode = 1; });
// ^^^ THOG
