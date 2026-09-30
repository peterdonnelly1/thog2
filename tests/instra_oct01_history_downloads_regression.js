// vvv THOG timing regressions cover parallel execution, retries, legacy metadata and absent estimates
"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),vm=require("node:vm");
const sandbox={window:{},document:{getElementById(id){if(id==="runner_view")return {};if(id==="runner_tabs")return {querySelectorAll(){return [];}};if(id==="runner_nav")return {addEventListener(){}};return null;}},setInterval(){}};
vm.runInNewContext(fs.readFileSync("sheet/local_dashboard_assets/dashboard_runner.js","utf8"),sandbox);
const {grid_elapsed,estimate_range}=sandbox.window.instra_runner_test_hooks;
const stamp=seconds=>new Date(seconds*1000).toISOString();
const run=(start,end)=>({attempts:[{started_at:stamp(start),finished_at:stamp(end)}]});
assert.equal(grid_elapsed({state:"completed",runs:[run(100,160),run(110,200)]}),100,"parallel durations must not be summed");
assert.equal(grid_elapsed({state:"running",runs:[run(100,160)]},250000),150);
assert.equal(grid_elapsed({state:"cancelled",runs:[]}),0);
assert.equal(grid_elapsed({state:"failed",runs:[{attempts:[{started_at:stamp(100)}]}]}),null,"missing historical clock must remain unknown");
assert.equal(grid_elapsed({state:"completed",runs:[{duration_seconds:65,attempts:[{started_at:stamp(100)}]}]}),65);
assert.equal(grid_elapsed({state:"completed",started_at:stamp(100),finished_at:stamp(260),runs:[run(100,150),run(200,250)]}),160);
assert.equal(grid_elapsed({state:"completed",runs:[{duration_seconds:50,attempts:[{started_at:stamp(100),finished_at:stamp(120)},{started_at:stamp(200)}]}]}),150);
assert.equal(grid_elapsed({state:"completed",runs:[{duration_seconds:50,attempts:[{started_at:stamp(100)},{started_at:stamp(200)}]}]}),150,"earlier attempt end is unnecessary when the final end is known");
assert.equal(estimate_range({estimated_duration:{interval_seconds:[60,120]}}),"1m 0s – 2m 0s");
for(const estimated_duration of [undefined,{}, {seconds:90},{interval_seconds:[null,120]}])assert.equal(estimate_range({estimated_duration}),"unknown");
console.log("PASS parallel/retry/live/legacy elapsed clocks and immutable estimate ranges");
const dashboard=fs.readFileSync("sheet/local_dashboard_assets/dashboard.js","utf8");
const context={colour_for_run:()=>"hsl(207 56% 72.0%)"};
vm.createContext(context);
vm.runInContext(dashboard.slice(dashboard.indexOf("function hsv_to_rgb("),dashboard.indexOf("function draw_colour_plane(")),context);
assert.deepEqual(Array.from(context.run_colour_rgb("grid_member")),[144,188,224]);
context.colour_for_run=()=>"#CC3377";
assert.deepEqual(Array.from(context.run_colour_rgb("grid_member")),[204,51,119]);
context.colour_for_run=()=>"hsl(207 56% 0%)";
assert.deepEqual(Array.from(context.run_colour_rgb("grid_member")),[0,0,0]);
console.log("PASS HSL Grid and hex override picker initialization/reset");

// ^^^ THOG
