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
const {recipe_problems, invalid_field_value, apply_width_selection} = sandbox.window.instra_runner_test_hooks;
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
// vvv THOG width-only Recipes and sweeps must not inherit the legacy mandatory DEPTH axis
const width_parameters = {...parameters,"--geometry-preset":"width-type-I","--select-width":true,
  "WIDTH.order":16,"WIDTH.compressor":"dct"};
delete width_parameters["DEPTH.order"];
assert.equal(recipe_problems({...recipe,parameters:width_parameters},hosts).length,0,
  "a standalone width Recipe must reach Save, Preview and Launch without selecting depth");
assert.equal(recipe_problems({...recipe,parameters:{...width_parameters,"WIDTH.order":[16,32,64]}},hosts).length,0,
  "a retained-width sweep must be valid without DEPTH.order");
assert.equal(recipe_problems({...recipe,parameters:{...width_parameters,"WIDTH.order":[16,32],"DEPTH.order":[1,2]}},hosts).length,0,
  "independent width/depth grids remain valid");
assert.equal(recipe_problems({...recipe,parameters:{...width_parameters,"--geometry-preset":["dense","width-type-I"]}},hosts).length,0,
  "a dense reference can accompany width-only trials");
assert.equal(recipe_problems({...recipe,parameters:{...width_parameters,"--select-width":false}},hosts).length,0);
for (const preset of ["width", "width-type-I", ["dense", "width"]]) {
  const selected = apply_width_selection({...recipe,parameters:{...width_parameters,"--geometry-preset":preset,"--select-width":false}});
  assert.equal(selected.parameters["--select-width"],true);
  assert.equal(recipe_problems(selected,hosts).length,0);
}
assert.equal(Object.hasOwn(apply_width_selection({...recipe,parameters:{...parameters,"--select-width":true}}).parameters,"--select-width"),false);
assert.match(recipe_problems({...recipe,parameters:{...width_parameters,"--select-depth":[true]}},hosts).join(" "), /DEPTH.order is required/);
assert.equal(recipe_problems({...recipe,parameters:{...width_parameters,"--geometry-preset":"full_block","--select-width":false}},hosts).length,0);
assert.match(recipe_problems({...recipe,parameters:{...width_parameters,"WIDTH.order":""}},hosts).join(" "), /WIDTH.order.*required/);
assert.match(recipe_problems({...recipe,parameters:{...width_parameters,"--geometry-preset":["depth","width-type-I"]}},hosts).join(" "), /DEPTH.order is required/,
  "legacy depth trials must still select their depth order");
const catalogue = JSON.parse(fs.readFileSync("instra_runner_catalogue.json","utf8"));
const end_step = "--instrumentation__width_activation_curves__end_step";
assert.equal(invalid_field_value(end_step,"-1",catalogue[end_step]),false,
  "the documented unbounded capture window must be editable in Runner");
assert.equal(invalid_field_value(end_step,"-2",catalogue[end_step]),true);
assert.equal(invalid_field_value("--max-iters","-1",catalogue["--max-iters"]),true,
  "the capture sentinel must not admit negative training counts");
// ^^^ THOG
assert.match(runner,/tab === "history" && \["failed","blocked"\]\.includes\(run.state\)/,
  "failed and blocked History diagnostics open without extra clicks, while Progress details stay collapsed");
assert.match(runner,/blocked: \$\{blocked\.blocking_reason\}/,
  "Progress Grid list must show the current blocking reason");
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
