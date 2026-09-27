"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const runner = fs.readFileSync("sheet/local_dashboard_assets/dashboard_runner.js", "utf8");
const sandbox = {
  window: {},
  document: {getElementById(id) {
    if (id === "runner_view") return {};
    if (id === "runner_tabs") return {querySelectorAll() { return []; }};
    if (id === "runner_nav") return {addEventListener() {}};
    return null;
  }},
  setInterval() {},
};
vm.runInNewContext(runner, sandbox);
const {recipe_problems} = sandbox.window.instra_runner_test_hooks;
const parameters = {
  "--geometry-preset":"depth", "--optimizer":"adamw", "--n-layer":2, "DEPTH.order":1,
  "--warmup-iters":0, "--block-size":32, "--n-embd":64, "--n-head":4,
  "--gradient-accumulation-steps":1, "--checkpoint-segment-size":1,
  "--learning-rate":0.0009, "--min-lr":0.00009, "--max-iters":2, "--batch-size":1,
};
const hosts = [{local:true, thog_host_id:"thog_host.test", last_discovered:{
  execution_profiles:[{profile_key:"current"}], gpus:[{gpu_key:"GPU-0",gpu_id:"thog_host.test.gpu.GPU-0"}],
}}];
const recipe = {label:"Small", parameters, max_parallel:1, profilers:["none"]};
assert.equal(recipe_problems(recipe, hosts).length, 0,
  "the default automatic GPU selection must be valid when a discovered GPU is available");
assert.match(recipe_problems(recipe, []).join(" "), /No discovered host/);
assert.match(recipe_problems({...recipe,gpu_pool:["missing"]}, hosts).join(" "), /Selected GPU missing/);
assert.match(recipe_problems({...recipe,parameters:{...parameters,"--n-head":""}}, hosts).join(" "), /--n-head is required/);
assert.match(recipe_problems({...recipe,parameters:{...parameters,"--max-iters":NaN}}, hosts).join(" "), /valid numbers/);
assert.match(recipe_problems({...recipe,parameters:{...parameters,"DEPTH.order":""}}, hosts).join(" "), /DEPTH.order is required/);
assert.match(runner,/if \(run.state === "failed"\) row.open = true/,
  "failed attempt diagnostics must open in History without an extra click");
assert.match(runner,/Unable to retrieve attempt log: \$\{error\.message\}/,
  "log retrieval errors must be shown beside the attempt, not only in the toolbar");

const table = fs.readFileSync("sheet/local_dashboard_assets/dashboard_runs_table_restore.js", "utf8");
const segment = table.match(/activation_checkpointing: \{([\s\S]*?)\n      \},\n      learning_rate:/)?.[1];
assert.ok(segment, "active Runs table S column definition is missing");
const {activation_checkpointing} = vm.runInNewContext(`({activation_checkpointing:{${segment}}})`, {
  configured_value(run, key) { return run.configuration[key]; },
});
assert.equal(activation_checkpointing.label, "S");
assert.equal(activation_checkpointing.value({configuration:{activation_checkpointing:true,checkpoint_segment_size:4}}),4);
assert.equal(activation_checkpointing.value({configuration:{activation_checkpointing:false,checkpoint_segment_size:4}}),"off");

console.log("PASS Runner automatic placement and required-field diagnostics, History visibility, Runs S segment value");
