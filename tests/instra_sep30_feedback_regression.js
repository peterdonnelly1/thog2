// vvv THOG follow-up coverage for category controls and progressive/cancelled Multiview requests
"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),vm=require("node:vm");
const runner=fs.readFileSync("sheet/local_dashboard_assets/dashboard_runner.js","utf8");
const catalogue=JSON.parse(fs.readFileSync("instra_runner_catalogue.json","utf8"));
const ui={window:{},document:{getElementById(id){
  if(id==="runner_view")return {};
  if(id==="runner_tabs")return {querySelectorAll:()=>[]};
  if(id==="runner_nav")return {addEventListener(){}};
  return null;
}},setInterval(){}};
vm.runInNewContext(runner,ui);
const hooks=ui.window.instra_runner_test_hooks;
for(const [name,key,on,off] of [["Premat","--premat","enabled","disabled"],
  ["Coarse","--plastic__coarse_phase","enabled","disabled"],
  ["Variable Depth","--plastic__do_learn_layer_count",true,false],
  ["Chaos Bumps","--chaos_bump__sampling__enabled",true,false]]) {
  assert.equal(hooks.category_enabled(name,{parameters:{[key]:on}}),true,name);
  assert.equal(hooks.category_enabled(name,{parameters:{[key]:off}}),false,name);
  assert.equal(hooks.category_enabled(name,{parameters:{[key]:[off,on]}}),true,`${name} sweep`);
}
assert.equal(hooks.category_enabled("Variable Depth",{parameters:{"--plastic__do_learn_layer_count":true,"--no-plastic__do_learn_layer_count":true}}),false);
assert.equal(hooks.category_enabled("NSIGHT",{profilers:["none"]}),false);
assert.equal(hooks.category_enabled("NSIGHT",{profilers:["nsys","ncu"]}),true);
const source=fs.readFileSync("thog_grid_runner.py","utf8");
const common=source.slice(source.indexOf("COMMON ="),source.indexOf("FORBIDDEN =")).match(/"[^"]+"/g).map(value=>JSON.parse(value));
for(const key of ["--log-interval","--eval-iters","--eval-interval"]) {
  assert.ok(common.includes(key));assert.equal(catalogue[key].category,"Run Control Parameters");
  assert.deepEqual(Array.from(hooks.categories_for_field(key,catalogue,common)),["Frequently Used","Run Control Parameters"]);
  assert.equal(hooks.matches_search(key,catalogue[key],key.replaceAll("-","_")),true);
}
for(const [category,expected] of [["Coarse",["--plastic__coarse_phase"]],
  ["Variable Depth",["--plastic__do_learn_layer_count","--no-plastic__do_learn_layer_count"]],
  ["Chaos Bumps",["--chaos_bump__sampling__enabled","--no-chaos_bump__sampling__enabled"]]]) {
  const keys=Object.keys(catalogue).filter(key=>catalogue[key].category===category&&!catalogue[key].ui_hidden);
  keys.sort((left,right)=>hooks.compare_fields(left,right,category));
  assert.deepEqual(keys.slice(0,expected.length),expected);
}
const workspace=fs.readFileSync("sheet/local_dashboard_assets/dashboard_workspace_only.js","utf8");
const map_start=workspace.indexOf("    const map_with_concurrency ="),map_end=workspace.indexOf("    const source_optimizer_update",map_start);
const start=workspace.indexOf("    const merge_metric_entries ="),end=workspace.indexOf("\n\n    let refresh_timer",start);
const runs=Array.from({length:40},(_,index)=>({id:`run_${index}`}));
const payload=run=>({available:true,group:{name:"train",revision:1,charts:[{id:"train/loss",title:"Loss",
  default_x_axis_mode:"step",available_x_axis_modes:["step"],series:[{x:[1,2],y:[5,4]}]}]}});
let controller=new AbortController(),selected=runs.slice(0,3),complete_last;
const context={Date,Promise,visible_runs:()=>selected,workspace_request_signal:()=>controller.signal,
  run_identifier:run=>run.id,run_name:run=>run.id,colour_for_run:()=>"blue",clone:value=>JSON.parse(JSON.stringify(value)),
  intersect_modes:(left,right)=>left===null?[...right]:left.filter(value=>right.includes(value)),
  direct_json:async url=>url.includes("run_2&")?new Promise(resolve=>{complete_last=()=>resolve(payload());}):payload()};
const cache_start=workspace.indexOf("    const metric_revisions="),cache_end=workspace.indexOf("    const cancel_pending",cache_start);
vm.createContext(context);vm.runInContext(workspace.slice(cache_start,cache_end)+workspace.slice(map_start,map_end)+workspace.slice(start,end)+"\nthis.fetch_group=fetch_metric_group;",context);
(async()=>{
  const partial=[];
  const pending=context.fetch_group("train",value=>partial.push(value.group.charts[0]?.series.length));
  await new Promise(setImmediate);
  assert.ok(partial.some(count=>count>0),"loss did not render before the final slow run returned");
  complete_last();const final=await pending;
  assert.equal(final.group.charts[0].series.length,3);
  selected=runs;controller=new AbortController();let requests=0;
  context.direct_json=(_url,signal)=>{requests++;return new Promise((_resolve,reject)=>signal.addEventListener("abort",()=>reject(new Error("cancelled")),{once:true}));};
  const obsolete=context.fetch_group("train");
  await new Promise(setImmediate);assert.equal(requests,8);
  controller.abort();await assert.rejects(obsolete,error=>error.name==="AbortError");
  assert.equal(requests,8,"cancelled selection continued querying its remaining runs");
  console.log("PASS category enables/order/search membership, progressive loss and cancellation of obsolete 40-run selections");
})().catch(error=>{console.error(error);process.exitCode=1;});
// ^^^ THOG
