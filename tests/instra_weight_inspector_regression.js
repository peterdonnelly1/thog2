// vvv THOG
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const source_path = path.join(__dirname, "../sheet/local_dashboard_assets/dashboard_weight_inspector.js");
const data = require(source_path);
const chart = "attn_q_head_N";
const runs = [{id: "a", name: "Alpha", colour: "#ff0000"}, {id: "b", name: "Beta", colour: "#008000"}];
const trace = (id, step, values, pair = [2, 3], extra = {}) => ({
  name: `step ${step}`, mode: "lines", x: [1, 2, 3], y: values,
  meta: {instra_workspace_run_id: id, instra_thog_optimizer_update: step,
    instra_weight_model_feature: pair[0], instra_weight_intermediate_feature: pair[1], ...extra},
});
const old_a = trace("a", 1, [0.123456, -0.000001, null]);
const exact_a = trace("a", 5, [999, 999, 999], [2, 3], {
  instra_thog_integer_x: [1, 2, 3], instra_thog_integer_y: [0.3333333, -0.123456, 0],
});
const final_b = trace("b", 8, [0.222222, 0.444444, 0.555555]);
const early_b = trace("b", 5, [0.666666, 0.777777, 0.888888]);
const marker = trace("a", 5, [0.3333333, -0.123456, 0], [2, 3], {instra_thog_executed_overlay: true});
const anchor = {meta: {instra_top_axis_anchor: true}};
const unknown = {x: [1], y: [99], meta: {instra_workspace_run_id: "a"}};
const input = [old_a, exact_a, {...exact_a}, marker, {...marker}, early_b, final_b, unknown, anchor];
const latest = data.latest_curves(input);
assert.deepEqual(latest, [exact_a, marker, final_b, anchor], "latest leaked duplicate, old or unidentified curves");
assert.equal(data.latest_curves([exact_a, exact_a]).length, 1);
assert.equal(data.latest_curves([trace("a", 5, [1, 2, 3], [10, 11]), exact_a]).length, 1, "legacy multi-scalar latest has multiple logical curves");
assert.equal(input.length, 9, "source figure mutated");
assert.equal(data.step({meta: {instra_workspace_optimizer_update: null, instra_dense_optimizer_update: 0}}), 0);
assert.equal(data.step({name: "r2_c3 · newest U500"}), 500);
assert.deepEqual(data.coupling(exact_a, "mlp_down"), {input: 3, output: 2});
assert.deepEqual(data.coupling({meta: {instra_dense_scalar_id: "r12_c7"}}, chart), {input: 7, output: 12});

const source = {data: [old_a, exact_a, marker, early_b, final_b, trace("a", 5, [100, 100, 100], [8, 9])]};
// Visible values have been arbitrarily transformed by chart settings. These must
// never replace the raw values. The extra coupling is not visible and is excluded.
let visible = {data: [old_a, exact_a, early_b, final_b].map(value => ({...value, y: [9999, 9999, 9999]}))};
const model = data.build_table(source, visible, runs, chart, "", false);
assert.deepEqual(model.rows.map(row => row.step), [1, 5, 8]);
assert.deepEqual(model.columns.map(col => `${col.layer}:${col.id}`), ["1:a", "1:b", "2:a", "2:b", "3:a", "3:b"]);
assert.equal(data.value_at(model, 1, 0), 0.3333333, "lost exact executed-layer data");
assert.equal(data.value_at(model, 0, 1), null, "invented a missing run/step value");
assert.equal(data.value_at(model, 0, 4), null, "null became zero");
const rectangle = {anchor: {row: 1, column: 3}, focus: {row: 0, column: 0}};
assert.equal(data.tsv(model, rectangle, 4), "0.1235\t\t0.0000\t\n0.3333\t0.6667\t-0.1235\t0.7778");
assert.equal(data.format(-0.00001, 4), "0.0000");
assert.equal(data.format(0, 0), "0");
assert.equal(data.format(1 / 3, 12), "0.333333333333");
assert.equal(data.precision(13), 4);
assert.equal(data.precision(-1), 4);
assert.equal(data.precision(null), 4);
assert.equal(data.build_table(source, {data: []}, runs, chart).rows.length, 0);
const sparse = trace("a", 1, [1, 2, 3]);
sparse.x = [1, 1.5, 3];
assert.deepEqual(data.build_table({data: [sparse]}, {data: [sparse]}, runs, chart, "", false).columns.map(col => col.layer), [1, 1, 3, 3]);
const multiple = data.build_table(source, source, runs, "mlp_down");
assert.equal(multiple.multiple_pairs, true, "distinct couplings were silently merged");
assert.equal(multiple.rows.length, 4);
const visible_window = data.window_range(280000, 700, 56, 28, 100000);
assert.ok(visible_window.end - visible_window.start < 28, "virtualization grows with total history");

const comparison = data.build_table(source, visible, runs, chart);
assert.deepEqual(comparison.columns.map(col => col.difference ? `${col.layer}:difference` : `${col.layer}:${col.id}`), ["1:difference", "1:a", "1:b", "2:difference", "2:a", "2:b", "3:difference", "3:a", "3:b"]);
assert.equal(data.value_at(comparison, 0, 0), null, "partial run coverage invented a comparison");
assert.ok(Math.abs(data.value_at(comparison, 1, 0) - 0.3333327) < 1e-15);
assert.ok(Math.abs(data.value_at(comparison, 1, 3) - 0.901233) < 1e-15);
assert.equal(data.csv(model).split("\r\n")[1], "1,2,3,0.123456,,-0.000001,,,", "CSV rounded values or invented missing entries");
const csv_model = data.build_table(source, visible, [{...runs[0], name: 'Alpha,"quoted"\nname'}, runs[1]], chart);
assert.ok(data.csv(csv_model).includes('Alpha,""quoted""\nname [a]"'), "CSV headers did not escape quotes/newlines");
assert.ok(data.csv(comparison).includes('layer_1 max_minus_min'));

const close_traces = [trace("a", 1, [0, 0, 0]), trace("b", 1, [1e-10, 0, 0]), trace("c", 1, [2e-10, 0, 0])];
const three_runs = [...runs, {id: "c", name: "Gamma"}];
const close_model = data.build_table({data: close_traces}, {data: close_traces}, three_runs, chart);
assert.equal(data.value_at(close_model, 0, 0), 2e-10);
assert.equal(data.cell_text(close_model, 0, 0, 4), "2.0000e-10", "tiny difference rounded into false equality");
assert.equal(data.value_at(close_model, 0, 4), 0, "equal weights have nonzero difference");

// Event-level DOM harness: production inspector handlers, fake layout/clipboard.
// This verifies behaviour without claiming a real-browser rendering result.
class FakeNode {
  constructor(tag = "div") {
    this.tagName = tag; this.children = []; this.dataset = {}; this.attributes = {}; this.listeners = {};
    this.style = {setProperty(name, value) { this[name] = value; }};
    this.hidden = false; this.clientWidth = 610; this.clientHeight = 250; this.scrollLeft = 0; this.scrollTop = 0;
    this.className = ""; this.textContent = "";
    this.classList = {
      contains: name => this.className.split(" ").includes(name),
      add: name => { if (!this.classList.contains(name)) this.className += ` ${name}`; },
      remove: name => { this.className = this.className.split(" ").filter(value => value !== name).join(" "); },
      toggle: (name, force) => { if (force ?? !this.classList.contains(name)) this.classList.add(name); else this.classList.remove(name); },
    };
  }
  appendChild(node) { if (node.tagName === "fragment") { for (const child of [...node.children]) this.appendChild(child); return node; } this.children.push(node); node.parentElement = this; return node; }
  append(...nodes) { nodes.forEach(node => this.appendChild(node)); }
  replaceChildren(...nodes) { this.children = []; this.append(...nodes); }
  remove() { if (this.parentElement) this.parentElement.children = this.parentElement.children.filter(node => node !== this); }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  removeAttribute(name) { delete this.attributes[name]; }
  getAttribute(name) { return this.attributes[name]; }
  addEventListener(name, handler) { (this.listeners[name] ||= []).push(handler); }
  async emit(name, extra = {}) {
    const event = {target: this, preventDefault() {}, stopPropagation() {}, stopImmediatePropagation() {}, ...extra};
    for (const handler of this.listeners[name] || []) await handler(event);
  }
  click() { if (this.tagName === "a") downloaded_link = this; }
  focus() { document.activeElement = this; }
  select() { document.activeElement = this; }
  querySelector(selector) { return find(this, node => node !== this && selector.startsWith(".") && node.classList.contains(selector.slice(1))); }
  closest(selector) { return selector.startsWith(".") && this.classList.contains(selector.slice(1)) ? this : this.parentElement?.closest(selector); }
  setPointerCapture() {}
  getBoundingClientRect() { return {left: 0, top: 0, right: this.clientWidth, bottom: this.clientHeight}; }
}
const root = new FakeNode();
const card = new FakeNode(); card.dataset.chart = chart; root.appendChild(card);
const actions = new FakeNode(); actions.className = "chart-card-actions"; card.appendChild(actions);
const controls = new FakeNode(); controls.id = "weight_step_group_controls"; root.appendChild(controls);
const overlay_node = new FakeNode(); overlay_node.id = "chart_settings_overlay"; overlay_node.hidden = true; root.appendChild(overlay_node);
const find = (node, predicate) => predicate(node) ? node : node.children.map(child => find(child, predicate)).find(Boolean);
const document = {
  head: root, createElement: tag => new FakeNode(tag), createDocumentFragment: () => new FakeNode("fragment"),
  querySelector: selector => selector.includes(`data-chart="${chart}"`) ? card : null,
  execCommand: () => true,
};
let current_precision = 4;
let current_mode = "whole";
const frames = new Map();
let next_frame = 1;
const flush_frames = () => { const jobs = [...frames.values()]; frames.clear(); jobs.forEach(job => job()); };
const window_events = {};
let copied = "";
let downloaded_blob;
let downloaded_link;
let revoked_url;
const app = {maximized_chart: null, workspace_mode: true, current_run_id: "a", figures: {depth: {[chart]: source}}};
const sandbox = {
  Blob, URL: {createObjectURL: blob => { downloaded_blob = blob; return "blob:test"; }, revokeObjectURL: url => { revoked_url = url; }},
  console, document, app, depth_weight_chart_names: [chart],
  by_id: id => find(root, node => node.id === id),
  current_run: () => ({dashboard_run_id: "a", run_name: "Alpha"}),
  run_identifier: run => run.dashboard_run_id,
  colour_for_run: id => runs.find(run => run.id === id).colour,
  resize_visible_plots() {},
  prepare_figure: () => visible,
  toggle_maximized_chart: name => { app.maximized_chart = name; },
  restore_maximized_chart: () => { app.maximized_chart = null; },
  render_run_heading() {}, render_figures: async () => {}, render_plot: async () => {},
  navigator: {clipboard: {writeText: async value => { copied = value; }}},
  requestAnimationFrame: job => { const id = next_frame++; frames.set(id, job); return id; },
  cancelAnimationFrame: id => frames.delete(id),
  setTimeout: job => job(),
  window: {
    addEventListener: (name, handler) => { (window_events[name] ||= []).push(handler); },
    __instra_further_weight_owner: {},
    __instra_weight_group_settings: {group_settings_for_scope: () => ({inspection_precision: current_precision})},
    __instra_weight_stability_final: {context_key: () => app.workspace_mode ? "workspace:a|b" : `run:${app.current_run_id}`, mode: () => current_mode},
    __instra_weight_step_filter: {signature: () => current_mode},
    __instra_workspace: {visible_runs: () => runs.map(run => ({dashboard_run_id: run.id, run_name: run.name}))},
  },
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(source_path, "utf8"), sandbox);
window_events.load.forEach(handler => handler());
const inspector = sandbox.window.__instra_weight_inspector;
assert.ok(inspector);
const button = sandbox.by_id("weight_inspect_button");
assert.equal(button.hidden, true, "inspect button leaked into tiled charts");
sandbox.toggle_maximized_chart(chart);
assert.equal(button.hidden, false);
inspector.open();
const by_class = name => find(card, node => node.classList.contains(name));
let grid = by_class("weight-inspection-grid");
let canvas = by_class("weight-inspection-canvas");
assert.equal(grid.getAttribute("aria-rowcount"), "5");
assert.equal(grid.getAttribute("aria-colcount"), "10");
assert.match(by_class("weight-inspection-metadata").textContent, /Step 1 · Layer 1 · Input coupling 2 · Output coupling 3 · max − min/);
(async () => {
  await grid.emit("keydown", {key: "ArrowRight", shiftKey: true});
  await grid.emit("keydown", {key: "ArrowDown", shiftKey: true});
  flush_frames();
  await by_class("weight-inspection-toolbar").children[2].emit("click");
  assert.equal(copied, "\t0.1235\n0.3333\t0.3333");
  let keyboard_copy;
  await by_class("weight-inspection-panel").emit("copy", {clipboardData: {setData: (_type, value) => { keyboard_copy = value; }}});
  assert.equal(keyboard_copy, copied);
  assert.match(by_class("weight-inspection-status").textContent, /Copied/);
  current_precision = 6;
  inspector.sync();
  await by_class("weight-inspection-toolbar").children[2].emit("click");
  assert.equal(copied, "\t0.123456\n0.333333\t0.333333", "precision did not refresh without losing selection");
  const first_cell = canvas.children.find(node => node.dataset.row === "0" && node.dataset.column === "0");
  await grid.emit("pointerdown", {target: first_cell, button: 0, pointerId: 1, clientX: 120, clientY: 65});
  await grid.emit("pointermove", {clientX: 365, clientY: 95});
  await grid.emit("pointerup");
  flush_frames();
  await by_class("weight-inspection-toolbar").children[2].emit("click");
  assert.equal(copied, "\t0.123456\t\n0.333333\t0.333333\t0.666666");
  await grid.emit("keydown", {key: "End", ctrlKey: true, shiftKey: true});
  flush_frames();
  assert.match(by_class("weight-inspection-status").textContent, /3 × 9 selected/);
  assert.ok(grid.scrollLeft > 0, "keyboard navigation failed to reveal a column");
  const toolbar = by_class("weight-inspection-toolbar");
  assert.equal(toolbar.children[3].hidden, false, "maximized Select all button missing");
  await toolbar.children[3].emit("click");
  flush_frames();
  assert.match(by_class("weight-inspection-status").textContent, /3 × 9 selected/);
  await toolbar.children[4].emit("click");
  assert.equal(downloaded_link.download, "instra_workspace_attn_q_head_N_weights.csv");
  assert.ok((await downloaded_blob.text()).includes("0.3333333"), "download button lost full precision");
  assert.equal(revoked_url, "blob:test", "download object URL leaked");
  await by_class("weight-inspection-toolbar").children[0].emit("click");
  assert.equal(card.classList.contains("weight-inspection-open"), false);
  assert.equal(app.maximized_chart, chart, "back arrow restored grid instead of chart");
  assert.equal(button.getAttribute("aria-pressed"), "false");
  inspector.open();
  app.current_run_id = "b"; app.workspace_mode = false;
  inspector.sync();
  assert.equal(card.classList.contains("weight-inspection-open"), false, "old run table survived context switch");
  app.current_run_id = "a";
  inspector.open();
  sandbox.restore_maximized_chart();
  assert.equal(button.hidden, true);
  assert.equal(by_class("weight-inspection-panel"), undefined);
  const standard_button = actions.children.find(node => node.classList.contains("weight-inspect-icon"));
  assert.ok(standard_button && !standard_button.hidden, "standard chart has no inspector icon");
  await standard_button.emit("click");
  assert.equal(by_class("weight-inspection-toolbar").children[3].hidden, true, "Select all leaked into standard view");
  await by_class("weight-inspection-panel").emit("keydown", {key: "a", ctrlKey: true});
  flush_frames();
  assert.match(by_class("weight-inspection-status").textContent, /2 × 3 selected/);
  copied = "";
  await by_class("weight-inspection-panel").emit("keydown", {key: "c", ctrlKey: true});
  await Promise.resolve();
  assert.ok(copied.includes("\n"), "Ctrl+C did not copy the table through the actual key handler");
  inspector.close();
  assert.equal(frames.size, 0, "animation/drag callback leaked after close");
  current_mode = "latest";
  const prepared = sandbox.prepare_figure(source, chart);
  assert.equal(prepared.data.length, 2, "final renderer did not enforce one latest curve per run");
  current_mode = "whole";
  app.workspace_mode = true;
  sandbox.toggle_maximized_chart(chart);
  const large_traces = Array.from({length: 1000}, (_, index) => runs.map(run => {
    const value = trace(run.id, index, Array.from({length: 144}, (_value, layer) => layer + index / 10000));
    value.x = Array.from({length: 144}, (_value, layer) => layer + 1);
    return value;
  })).flat();
  app.figures.depth[chart] = {data: large_traces};
  visible = app.figures.depth[chart];
  inspector.open();
  grid = by_class("weight-inspection-grid");
  canvas = by_class("weight-inspection-canvas");
  assert.equal(grid.getAttribute("aria-rowcount"), "1002");
  assert.equal(grid.getAttribute("aria-colcount"), "433");
  assert.ok(canvas.children.length < 180, "large table rendered all cells");
  await grid.emit("keydown", {key: "End", ctrlKey: true});
  flush_frames();
  assert.match(by_class("weight-inspection-metadata").textContent, /Step 999 · Layer 144.*Beta/);
  assert.ok(canvas.children.length < 180, "scrolled table rendered all cells");
  const last_cell = canvas.children.find(node => node.dataset.row === "999" && node.dataset.column === "431");
  await grid.emit("pointerdown", {target: last_cell, button: 0, pointerId: 2, clientX: 590, clientY: 230});
  const old_scroll = grid.scrollLeft;
  await grid.emit("pointermove", {clientX: 80, clientY: 30});
  flush_frames();
  assert.ok(grid.scrollLeft < old_scroll, "drag near the left edge did not autoscroll");
  await grid.emit("pointercancel");
  inspector.close();
  assert.equal(frames.size, 0, "large drag/paint callbacks survived close");
  console.log("instra weight inspector data and event regressions: PASS");
})().catch(error => { console.error(error); process.exitCode = 1; });
// ^^^ THOG
