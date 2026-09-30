// vvv THOG exercise the production Grid palette and multi-Grid chart merger without a GPU
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const dashboard = fs.readFileSync("sheet/local_dashboard_assets/dashboard.js", "utf8");
const workspace = fs.readFileSync("sheet/local_dashboard_assets/dashboard_workspace_only.js", "utf8");
const groups = fs.readFileSync("sheet/local_dashboard_assets/dashboard_wandb_groups_patch.js", "utf8");
const runs = ["G-00001", "G-00002", "G-00003"].flatMap((tag, group) =>
  Array.from({length:group === 2 ? 1 : 3}, (_, index) => ({
    dashboard_run_id:`r${group}${index}`, runner_run_id:`uuid-${group}-${index}`,
    runner_grid_tag:tag, artifact_name:`${tag}_run_${index}`,
  })));
const take = (source, first, last) => source.slice(source.indexOf(first), source.indexOf(last, source.indexOf(first)));
const context = {app:{runs, colours:{}, visibility:{}},default_palette:["#aaa"],
  run_identifier:run=>run.dashboard_run_id};
vm.createContext(context);
vm.runInContext(take(dashboard,"function hash_text(","function is_visible("),context);
const colour = id => context.colour_for_run(id);
for (const tag of ["G-00001","G-00002"]) {
  const shades = runs.filter(run=>run.runner_grid_tag===tag).map(run=>colour(run.dashboard_run_id));
  assert.equal(new Set(shades).size,3);
  assert.equal(new Set(shades.map(value=>value.match(/^hsl\((\d+)/)[1])).size,1);
  assert.deepEqual(shades.map(value=>Number(value.match(/([\d.]+)%\)$/)[1])),[72,54,36]);
}
assert.notEqual(colour("r00").match(/^hsl\((\d+)/)[1],colour("r10").match(/^hsl\((\d+)/)[1]);

const member = runs.find(run=>run.dashboard_run_id==="r00");
const group_members = member.runner_grid_tag ? runs.filter(run=>run.runner_grid_tag===member.runner_grid_tag) : [member];
for (const run of group_members) context.app.visibility[run.dashboard_run_id]=false;
assert.equal(group_members.length,3);
assert.equal(runs.filter(run=>run.runner_grid_tag==="G-00002" && context.app.visibility[run.dashboard_run_id]===false).length,0);
assert.match(dashboard,/const group_members = run\.runner_grid_tag &&/,"Grid eye grouping must work in Runs and Multiview");

const start = workspace.indexOf("    const merge_metric_entries = (group_name, entries) => {");
const end = workspace.indexOf("\n\n    let refresh_timer",start);
assert.ok(start>=0 && end>start);
let selected = runs.filter(run=>run.runner_grid_tag!=="G-00003");
const merge_context = {
  workspace_request_signal:()=>undefined,
  Date,
  visible_runs:()=>selected,
  map_with_concurrency:async(values,_limit,operation)=>Promise.all(values.map(operation)),
  direct_json:async url=>{
    const id=new URL(url,"http://localhost").searchParams.get("run");
    return {available:true,group:{name:"train",revision:1,charts:[{id:"train/loss",title:"Loss",x_title:"step",
      default_x_axis_mode:"step",available_x_axis_modes:["step"],series:[{name:"Loss",x:[1,2],y:[5,Number(id.slice(-1))+2]}]}]}};
  },
  run_identifier:run=>run.dashboard_run_id,
  run_name:run=>run.artifact_name,
  colour_for_run:colour,
  clone:value=>JSON.parse(JSON.stringify(value)),
  intersect_modes:(left,right)=>left===null?[...right]:left.filter(value=>right.includes(value)),
};
vm.createContext(merge_context);
vm.runInContext(workspace.slice(start,end)+"\nthis.merge_group=fetch_metric_group;",merge_context);
(async()=>{
  let result=await merge_context.merge_group("train");
  assert.equal(result.group.charts[0].series.length,6,"two eye-selected Grids must show six loss curves");
  assert.equal(new Set(result.group.charts[0].series.map(series=>series.instra_workspace_run_id)).size,6);
  selected=runs.filter(run=>run.runner_grid_tag==="G-00002");
  result=await merge_context.merge_group("train");
  assert.equal(result.group.charts[0].series.length,3,"de-eyeing one Grid removes only its curves");
  assert.match(groups,/<b>%\{fullData\.name\}<\/b><br>step: %\{x\}<br>value: %\{y:\.6g\}<extra><\/extra>/);
  console.log("PASS Grid shades, independent eyes and multi-Grid loss merge");
})().catch(error=>{console.error(error);process.exitCode=1;});
// ^^^ THOG
