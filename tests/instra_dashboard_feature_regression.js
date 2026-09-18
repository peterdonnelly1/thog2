"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

class FakeCard {
  constructor(chart_name) {
    this.dataset = {chart:chart_name};
    this.hidden = false;
  }
}

const swatches = [];
const heatmap = new FakeCard("heatmap");
const qkv = new FakeCard("attn_q_head_N");
const mlp = new FakeCard("mlp_down");
const count = {textContent:"7"};
const group = {
  hidden:false,
  querySelectorAll() { return [heatmap, qkv, mlp]; },
};
const container = {
  dataset:{},
  appendChild(button) { swatches.push(button); },
  querySelectorAll() { return swatches; },
};
const load_listeners = [];
const sandbox = {
  app:{figures:null},
  default_palette:Array.from({length:64}, (_value, index) => `#${(0x100000 + index).toString(16).toUpperCase()}`),
  render_figures:async function() {},
  reset_run_charts:function() {},
  open_colour_picker:function() {},
  hex_to_rgb(value) { return value; },
  set_picker_colour() {},
  by_id(id) {
    if (id === "colour_swatches") return container;
    if (id === "depth_chart_group") return group;
    if (id === "depth_group_count") return count;
    return null;
  },
  document:{
    createElement() {
      return {
        className:"",
        style:{},
        title:"",
        setAttribute() {},
        addEventListener() {},
      };
    },
  },
  window:{
    addEventListener(type, listener) { if (type === "load") load_listeners.push(listener); },
  },
  Object,
  Set,
  String,
};

const source = fs.readFileSync("sheet/local_dashboard_assets/dashboard_feature_regression_restore.js", "utf8");
vm.runInNewContext(source, sandbox);
for (const listener of load_listeners) listener();
assert.equal(swatches.length, 128, "the complete shared 128-colour palette was not restored");
assert.equal(container.dataset.instraPalette128, "true");
assert.equal(sandbox.window.instra_colour_palette.length, 128);

sandbox.app.figures = {heatmap:null, depth:{attn_q_head_N:{data:[]}}};
sandbox.window.instra_feature_regression_test_hooks.sync_depth_chart_availability();
assert.equal(heatmap.hidden, true, "an uncollected heatmap remained visible");
assert.equal(qkv.hidden, false, "a collected chart was hidden");
assert.equal(mlp.hidden, true, "an uncollected weight chart remained visible");
assert.equal(group.hidden, false);
assert.equal(count.textContent, "1");

sandbox.app.figures = {heatmap:null, depth:{}};
sandbox.window.instra_feature_regression_test_hooks.sync_depth_chart_availability();
assert.equal(group.hidden, true, "an entirely empty chart group remained visible");

const trash_source = fs.readFileSync("sheet/local_dashboard_assets/dashboard_runs_trash_restore.js", "utf8");
assert.match(trash_source, /button\.hidden = false/);
assert.match(trash_source, /toolbar\.insertBefore\(button, group\)/);
assert.match(trash_source, /min-width:30px/);

const operations_source = fs.readFileSync("sheet/local_dashboard_assets/dashboard_processing_operations_final.js", "utf8");
const throughput_source = fs.readFileSync("sheet/local_dashboard_assets/dashboard_processing_resource_attribution.js", "utf8");
assert.match(operations_source, /id="training_throughput_z_button"/);
assert.match(operations_source, /if \(reset\) reset\.disabled = false;[\s\S]*?Plotly\.relayout/);
assert.match(throughput_source, /id = "processing_throughput_z_button"/);
assert.match(throughput_source, /orientation: "v"[\s\S]*?yanchor: "bottom"/);

console.log("PASS palette, trash, no-data chart, dual throughput z and reset-zoom regressions");
