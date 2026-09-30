// vvv THOG focused tests for duration, validation, outcomes and persistent Grid hue allocation
"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),vm=require("node:vm");
const source=fs.readFileSync("sheet/local_dashboard_assets/dashboard_runner.js","utf8");
const sandbox={window:{},document:{getElementById(id){
  if(id==="runner_view")return {};
  if(id==="runner_tabs")return {querySelectorAll(){return [];}};
  if(id==="runner_nav")return {addEventListener(){}};
  return null;
}},setInterval(){}};
vm.runInNewContext(source,sandbox);
const hooks=sandbox.window.instra_runner_test_hooks;
assert.equal(hooks.format_duration(13687),"3h 48m 7s");
assert.equal(hooks.format_duration(65),"1m 5s");
assert.equal(hooks.format_duration(8),"8s");
assert.equal(hooks.history_outcome({runs:[{state:"completed"},{state:"failed"}]}),"partial");
assert.equal(hooks.history_outcome({runs:[{state:"completed"},{state:"completed"}]}),"complete");
assert.equal(hooks.history_outcome({runs:[{state:"failed"},{state:"cancelled"}]}),"none");
const catalogue=JSON.parse(fs.readFileSync("instra_runner_catalogue.json","utf8"));
for(const [key,spec] of Object.entries(catalogue)) {
  if(spec.ui_hidden || ["automatic","manual"].includes(spec.kind))continue;
  assert.match(hooks.field_help(spec),/Available options:/,key);
  if(spec.choices?.length)assert.equal(hooks.invalid_field_value(key,"invalid_option_xyz",spec),true,key);
}
assert.equal(hooks.invalid_field_value("--batch-size","1.5",{type:"int"}),true);
assert.equal(hooks.invalid_field_value("--batch-size","-1",{type:"int"}),true);
assert.equal(hooks.invalid_field_value("--learning-rate","NaN",{type:"float"}),true);
assert.equal(hooks.invalid_field_value("--premat","enabled, disabled",catalogue["--premat"]),false);
assert.equal(hooks.invalid_field_value("--premat","enabled, enabled",catalogue["--premat"]),true);
const dashboard=fs.readFileSync("sheet/local_dashboard_assets/dashboard.js","utf8");
const storage=new Map();
const runs=Array.from({length:12},(_,index)=>({dashboard_run_id:`run_${index}`,runner_grid_tag:`G-${String(index).padStart(5,"0")}`}));
const context={app:{runs,colours:{}},default_palette:[],run_identifier:run=>run.dashboard_run_id,
  localStorage:{getItem:key=>storage.get(key)||null,setItem:(key,value)=>storage.set(key,value)}};
vm.createContext(context);
vm.runInContext(dashboard.slice(dashboard.indexOf("function hash_text("),dashboard.indexOf("function is_visible(")),context);
const initial=runs.map(run=>context.colour_for_run(run.dashboard_run_id));
assert.equal(new Set(initial).size,12,"hash collisions must not repeat Grid tones");
const hue=value=>Number(value.match(/^hsl\((\d+)/)[1]);
const separation=(left,right)=>Math.min(Math.abs(hue(left)-hue(right)),360-Math.abs(hue(left)-hue(right)));
assert.ok(initial.every((left,index)=>initial.slice(index+1).every(right=>separation(left,right)>=22)));
context.app.runs=[...runs].reverse();
assert.deepEqual(runs.map(run=>context.colour_for_run(run.dashboard_run_id)),initial,"catalogue reorder changed existing hues");
context.app.runs=[...runs,{dashboard_run_id:"new",runner_grid_tag:"G-99999"}];
context.colour_for_run("new");
assert.deepEqual(runs.map(run=>context.colour_for_run(run.dashboard_run_id)),initial,"new Grid changed existing hues");
assert.ok(storage.get("thog2_grid_hues"),"central hues were not persisted");
console.log("PASS duration units, all field options, immediate syntax/choice validation, History outcomes and distinct persistent Grid hues");
// ^^^ THOG
// vvv THOG the live throughput formatter must accept string titles and stop redrawing unchanged layouts
const presentation=fs.readFileSync("sheet/local_dashboard_assets/dashboard_sep21_instra_repairs.js","utf8");
const first=presentation.indexOf("    function standardize_throughput_plot("),last=presentation.indexOf("\n    // vvv THOG chart discovery",first);
let relayout_calls=0;
const mount={dataset:{plotReady:"true"},layout:{xaxis:{title:"optimizer update"},margin:{},legend:{}}};
const presentation_context={Plotly:{relayout(node,updates){
  relayout_calls++;
  for(const [key,value] of Object.entries(updates)) {
    const parts=key.split(".");let target=node.layout;
    for(const part of parts.slice(0,-1))target=target[part] ||= {};
    target[parts.at(-1)]=value;
  }
  return Promise.resolve();
}}};
vm.createContext(presentation_context);
vm.runInContext(presentation.slice(first,last),presentation_context);
presentation_context.standardize_throughput_plot(mount);
assert.equal(mount.layout.xaxis.title.text,"steps");
for(let repeat=0;repeat<1000;repeat++)presentation_context.standardize_throughput_plot(mount);
assert.equal(relayout_calls,1,"unchanged throughput layout repeatedly invoked Plotly");
console.log("PASS string axis titles and 1000 unchanged layout passes without redraw");
// ^^^ THOG
