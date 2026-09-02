// vvv THOG
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const assets = path.join(__dirname, "../sheet/local_dashboard_assets");
const read = name => fs.readFileSync(path.join(assets, name), "utf8");
const columns_source = read("dashboard_weight_controls_run_table_patch.js");
const context = vm.createContext({});
vm.runInContext(columns_source.slice(columns_source.indexOf("    const configuration_value ="), columns_source.indexOf("    const install_run_shape_headers =")) + "\nthis.definitions = run_shape_columns;", context);
const definitions = context.definitions;
const column = key => definitions.find(item => item.key === key);
const run = {configuration: {lifecycle: {optimizer_name: "sgd", optimizer_momentum: 0.9}, learning_rate: 0.0009, min_learning_rate: 0.00009}};
assert.equal(column("optimizer").value(run), "sgd_0.9");
for (const [name, momentum, expected] of [["adamw", 0.9, "adamw"], ["sgd", 0, "sgd"], ["sgd_nesterov", 0.95, "sgd_nesterov_0.95"], ["rmsprop", 0.8, "rmsprop_0.8"], ["adafactor", 0.9, "adafactor"]]) {
  assert.equal(column("optimizer").value({configuration: {lifecycle: {optimizer_name: name, optimizer_momentum: momentum}}}), expected);
}
assert.equal(column("optimizer").value({}), "—", "unknown optimizer was invented");
assert.equal(column("learning_rate").value(run), "90");
assert.equal(column("min_learning_rate").value(run), "9");
assert.equal(column("learning_rate").value({configuration: {learning_rate: 0.01}}), "1000");
assert.equal(column("min_learning_rate").value({configuration: {min_learning_rate: 0}}), "0");
assert.equal(column("learning_rate").value({}), "—");
assert.deepEqual(Array.from(definitions.slice(-3), item => item.label), ["S", "c", "f"]);
assert.deepEqual(Array.from(definitions.slice(0, 2), item => item.label), ["p", "OPT"]);
assert.match(columns_source, /th\.run-shape-column \{ text-transform: none !important/);

// Production metric-card handlers and figure preparation, with minimal layout mocks.
class Element {
  constructor() { this.children = []; this.dataset = {}; this.attributes = {}; this.handlers = {}; this.className = ""; }
  append(...nodes) { for (const node of nodes) this.appendChild(node); }
  appendChild(node) { this.children.push(node); return node; }
  setAttribute(key, value) { this.attributes[key] = value; }
  addEventListener(name, handler) { this.handlers[name] = handler; }
  querySelector(selector) { return this.children.find(child => `.${child.className}` === selector) || this.children.map(child => child.querySelector(selector)).find(Boolean); }
}
let workspace = true;
const app = {current_run_id: "a", dynamic_chart_figures: {}, dynamic_chart_metadata: {}};
const metric = vm.createContext({
  app, chart_titles: {}, front_by_chart: new Map(), workspace_api: () => workspace,
  document: {createElement: () => new Element()},
  chart_key: (group, id) => `local_metric_${group}_${id}`,
  normalize_chart_settings: key => ({title: key}), chart_settings_button: () => new Element(),
  add_panel_resizers() {}, default_palette: ["#000"],
  prepare_figure: figure => JSON.parse(JSON.stringify(figure)),
});
metric.render_plot = async (mount, figure, key) => { mount.figure = metric.prepare_figure(figure, key); };
const metric_source = read("dashboard_wandb_groups_patch.js");
vm.runInContext(metric_source.slice(metric_source.indexOf("    const ordered_metric_figure ="), metric_source.indexOf("    const render_metric_chart =")) + "\nthis.make_card = make_metric_card; this.figure = metric_figure;", metric);
const chart = {id: "loss", title: "Loss", series: ["a", "b", "c"].map(id => ({instra_workspace_run_id: id, name: id, x: [1, 2], y: [4, 3]}))};
(async () => {
  const cards = ["train", "val"].map(group => metric.make_card(group, chart));
  for (const card of cards) app.dynamic_chart_figures[card.dataset.chart] = metric.figure(card, chart);
  const original = JSON.stringify(app.dynamic_chart_figures);
  const train_button = cards[0].querySelector(".weight-step-button metric-z-cycle");
  const val_button = cards[1].querySelector(".weight-step-button metric-z-cycle");
  assert.equal(train_button.hidden, false);
  for (const front of ["a", "b", "c", "a"]) {
    await train_button.handlers.click({stopPropagation() {}});
    assert.equal(cards[0].querySelector(".plot-mount").figure.data.at(-1).meta.instra_workspace_run_id, front);
    assert.equal(metric.prepare_figure(app.dynamic_chart_figures[cards[0].dataset.chart], cards[0].dataset.chart).data.at(-1).meta.instra_workspace_run_id, front, "refresh lost front run");
  }
  await val_button.handlers.click({stopPropagation() {}});
  assert.equal(cards[1].querySelector(".plot-mount").figure.data.at(-1).meta.instra_workspace_run_id, "a");
  assert.equal(JSON.stringify(app.dynamic_chart_figures), original, "z changed source data");
  workspace = false;
  assert.equal(metric.make_card("train", chart).querySelector(".weight-step-button metric-z-cycle").hidden, true);
  assert.equal(metric.figure(cards[0], {series: [{x: [1], y: [3]}]}).data[0].mode, "lines+markers", "single sample is invisible");
  console.log("instra September 2 columns/metric controls regression: PASS");
})().catch(error => { console.error(error); process.exitCode = 1; });
// ^^^ THOG
