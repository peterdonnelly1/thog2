// vvv THOG
"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const repository_root = path.resolve(__dirname, "..");
const source = fs.readFileSync(
  path.join(repository_root, "sheet/local_dashboard_assets/dashboard_legacy_heatmap_repair_patch.js"),
  "utf8",
);

const button = {
  textContent: "%", dataset: {}, disabled: false, title: "",
  setAttribute(name, value) { this[name] = String(value); },
};
const context = {
  console,
  window: {addEventListener(name, callback) { if (name === "load") callback(); }},
  by_id: id => id === "heatmap_delta_loss_mode" ? button : null,
  heatmap_settings_for_current_run: () => ({
    delta_loss_display_mode: "percent",
    auto_colour_saturation: false,
    negative_abs_limit: 0.05,
    blue_abs_limit: 1,
    yellow_abs_limit: 2,
    positive_abs_limit: 0.05,
  }),
  heatmap_abs_limit: fallback => fallback,
  transpose_heatmap: prepared => prepared,
  queueMicrotask: callback => callback(),
};
context.window.window = context.window;
vm.createContext(context);
vm.runInContext(source, context);

const legacy = {
  data: [{
    type: "heatmap",
    customdata: [[
      [3100, 24, -1, -0.025, null, null],
      [3100, 25, 0, 0.0, null, null],
      [3100, 26, 1, 0.040, null, null],
    ]],
    z: [[null, null, null]],
    colorbar: {},
  }],
  layout: {meta: {thog2_current_losses: [null]}},
};
context.transpose_heatmap(legacy);
assert.equal(legacy.layout.meta.thog2_legacy_absolute_fallback, true);
assert.ok(legacy.data[0].z.flat().some(value => Number.isFinite(value) && value !== 0));
assert.equal(legacy.data[0].colorbar.title, "Δloss bands (legacy absolute fallback)");
assert.equal(button.textContent, "|abs|");
assert.equal(button.disabled, true);

const modern = {
  data: [{
    type: "heatmap",
    customdata: [[[4000, 24, -1, -0.01, 2.5, -0.4]]],
    z: [[-0.4]],
    colorbar: {},
  }],
  layout: {meta: {thog2_current_losses: [2.5]}},
};
assert.equal(context.window.__instra_legacy_heatmap_repair.heatmap_needs_absolute_fallback(modern), false);

console.log("instra legacy heatmap repair regression: PASS");
// ^^^ THOG
