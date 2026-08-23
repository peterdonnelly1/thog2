// vvv THOG
"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const repository_root = path.resolve(__dirname, "..");
const source = fs.readFileSync(
  path.join(repository_root, "sheet/local_dashboard_assets/dashboard_regression_repair_patch.js"),
  "utf8",
);

function make_context() {
  let flags = {current_weights_only: false, join_with_line_segments: false};
  let explicit_range = false;
  const click_listeners = [];
  const elements = new Map([
    ["chart_settings_overlay", {hidden: false}],
    ["chart_settings_title", {textContent: "Attention - Q settings"}],
    ["chart_current_weights_only", {checked: false, disabled: true, title: ""}],
    ["chart_join_with_line_segments", {checked: false, disabled: true, title: ""}],
    ["heatmap_delta_loss_mode", {
      textContent: "%", dataset: {}, disabled: false, title: "",
      setAttribute(name, value) { this[name] = String(value); },
    }],
  ]);
  const weight_api = {
    global_flags: () => ({...flags}),
    set_global_flags(next) {
      flags = {
        current_weights_only: next.current_weights_only === true,
        join_with_line_segments: next.join_with_line_segments === true,
      };
      return true;
    },
  };
  const context = {
    console,
    depth_weight_chart_names: ["q", "k", "v", "o", "up", "down"],
    app: {axis_chart_name: "q"},
    window: {
      __instra_weight_controls_v2: weight_api,
      __instra_weight_step_filter: {active: () => explicit_range},
      addEventListener(event, callback) {
        if (event === "load") callback();
        else if (event === "click") click_listeners.push(callback);
      },
    },
    document: {},
    by_id: id => elements.get(id) || null,
    sync_chart_setting_outputs: () => undefined,
    populate_chart_settings_form: () => undefined,
    open_chart_settings: () => undefined,
    transpose_heatmap: prepared => prepared,
    heatmap_settings_for_current_run: () => ({
      delta_loss_display_mode: "percent",
      auto_colour_saturation: false,
      negative_abs_limit: 0.05,
      blue_abs_limit: 1,
      yellow_abs_limit: 2,
      positive_abs_limit: 0.05,
    }),
    heatmap_abs_limit: fallback => fallback,
    setTimeout(callback) { callback(); return 1; },
    clearTimeout() {},
    queueMicrotask(callback) { callback(); },
  };
  context.window.window = context.window;
  vm.createContext(context);
  vm.runInContext(source, context);
  return {
    context,
    elements,
    click_listeners,
    flags: () => ({...flags}),
    set_explicit_range: value => { explicit_range = Boolean(value); },
  };
}

function click_save(click_listeners) {
  const save_button = {id: "save_chart_settings"};
  const event = {
    target: {
      closest(selector) {
        return selector === "#save_chart_settings" ? save_button : null;
      },
    },
  };
  for (const listener of click_listeners) listener(event);
}

function global_weight_control_regression() {
  const fixture = make_context();
  const {context, elements, click_listeners} = fixture;
  const current = elements.get("chart_current_weights_only");
  const join = elements.get("chart_join_with_line_segments");
  const title = elements.get("chart_settings_title");

  context.sync_chart_setting_outputs();
  assert.equal(current.disabled, false, "individual Weight editor left Current weights only disabled");
  assert.equal(join.disabled, false, "individual Weight editor left Join with line segments disabled");

  current.checked = true;
  join.checked = true;
  click_save(click_listeners);
  assert.deepEqual(
    fixture.flags(),
    {current_weights_only: true, join_with_line_segments: true},
    "individual Weight editor did not write the global flags",
  );

  title.textContent = "Completely renamed group editor";
  current.disabled = true;
  join.disabled = true;
  context.sync_chart_setting_outputs();
  assert.equal(current.disabled, false, "group Weight editor still depends on its title text");
  assert.equal(join.disabled, false, "group Join control still depends on its title text");

  fixture.set_explicit_range(true);
  context.sync_chart_setting_outputs();
  assert.equal(current.disabled, true, "explicit step range should override Current weights only");
  assert.equal(join.disabled, false, "explicit step range must not disable Join with line segments");
}

function legacy_heatmap_regression() {
  const {context, elements} = make_context();
  const prepared = {
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
  context.transpose_heatmap(prepared);
  const heatmap = prepared.data[0];
  assert.equal(prepared.layout.meta.thog2_legacy_absolute_fallback, true);
  assert.ok(heatmap.z.flat().some(value => Number.isFinite(value) && value !== 0), "legacy Δloss cells remained blank");
  assert.equal(heatmap.colorbar.title, "Δloss bands (legacy absolute fallback)");
  assert.ok(!heatmap.hovertemplate.includes("Δloss (%)"), "legacy fallback still advertises percentage Δloss");
  const button = elements.get("heatmap_delta_loss_mode");
  assert.equal(button.textContent, "|abs|");
  assert.equal(button.disabled, true);
  assert.match(button.title, /legacy run/i);

  const modern = {
    data: [{
      type: "heatmap",
      customdata: [[[4000, 24, -1, -0.01, 2.5, -0.4]]],
      z: [[-0.4]],
      colorbar: {},
    }],
    layout: {meta: {thog2_current_losses: [2.5]}},
  };
  assert.equal(context.window.__instra_regression_repair.heatmap_needs_absolute_fallback(modern), false);
}

global_weight_control_regression();
legacy_heatmap_regression();
console.log("instra final regression repair: PASS");
// ^^^ THOG
