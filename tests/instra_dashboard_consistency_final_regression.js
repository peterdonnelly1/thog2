// vvv THOG
"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const repository_root = path.resolve(__dirname, "..");
const source = fs.readFileSync(
  path.join(repository_root, "sheet/local_dashboard_assets/dashboard_consistency_final_patch.js"),
  "utf8",
);

let heatmap_repairs = 0;
let heatmap_renders = 0;
const stored = new Map([
  ["thog2_local_overview_default_font_size", "12"],
  ["thog2_local_overview_font_size", "12"],
]);
const style_values = new Map();
const mount = {dataset: {plotReady: "true", instraRenderedRunId: "R1"}};
const pane = {style: {setProperty: (name, value) => style_values.set(name, value)}};
const styles = [];
const elements = {
  heatmap_chart_group: {classList: {contains: () => false}},
  heatmap_plot: mount,
  heatmap_placeholder: {hidden: false},
  run_overview_pane: pane,
  overview_font_larger: {
    click() {
      stored.set("thog2_local_overview_font_size", "13");
      style_values.set("--thog2-overview-font-size", "13px");
    },
  },
};
const app = {
  current_run_id: "R1",
  current_status: {heatmap_count: 30},
  figures: {heatmap: {data: [{type: "heatmap"}], layout: {}}},
  workspace_mode: false,
};
const context = {
  console,
  Date,
  Math,
  app,
  window: {
    addEventListener(name, callback) { if (name === "load") callback(); },
    __thog2_synthetic_groups: {group_is_open: () => true},
    __thog2_dashboard_performance: {
      async refresh_family_if_stale() {
        heatmap_repairs += 1;
        mount.dataset.plotReady = "true";
        mount.dataset.instraRenderedRunId = app.current_run_id;
      },
    },
  },
  document: {
    head: {appendChild: element => styles.push(element)},
    createElement: () => ({id: "", textContent: ""}),
    querySelector(selector) {
      if (selector.includes("heatmap-shell")) return {clientWidth: 1000};
      if (selector === ".run-overview-pane") return pane;
      return null;
    },
  },
  localStorage: {
    getItem: key => stored.get(key) ?? null,
    setItem: (key, value) => stored.set(key, String(value)),
  },
  by_id: id => elements[id] || null,
  current_run: () => app.current_status,
  async render_plot(target) {
    heatmap_renders += 1;
    target.dataset.plotReady = "true";
  },
  prepare_figure: figure => JSON.parse(JSON.stringify(figure)),
  select_run: run_id => { app.current_run_id = run_id; },
  queueMicrotask(callback) { callback(); },
  setTimeout(callback) { callback(); return 1; },
  setInterval(callback) { callback(); return 1; },
};
context.window.window = context.window;

(async () => {
  vm.createContext(context);
  vm.runInContext(source, context, {filename: "dashboard_consistency_final_patch.js"});
  await Promise.resolve();
  await Promise.resolve();

  assert.equal(heatmap_renders, 1, "cached heatmap was not rerendered through final geometry");
  assert.equal(heatmap_repairs, 0, "geometry-only refresh issued a redundant family repair");
  assert.equal(mount.dataset.instraRenderedRunId, "R1");

  const prepared = context.prepare_figure({
    data: [{type: "heatmap", colorbar: {}}],
    layout: {margin: {l: 100, r: 100}},
  }, "heatmap");
  assert.equal(prepared.layout.margin.r, 270);
  assert.ok(prepared.data[0].colorbar.x > 1.1, "colour key lacks a body gap");
  assert.equal(prepared.data[0].colorbar.xpad, 12);

  assert.equal(stored.get("thog2_local_overview_default_font_size"), "13");
  assert.equal(stored.get("thog2_local_overview_font_size"), "13");
  assert.equal(style_values.get("--thog2-overview-font-size"), "13px");
  assert.match(styles[0].textContent, /grid-template-columns: minmax\(0, 1fr\) minmax\(0, 1fr\)/);
  assert.match(styles[0].textContent, /minmax\(120px, \.45fr\)/);

  console.log("instra dashboard consistency final regression: PASS");
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
// ^^^ THOG
