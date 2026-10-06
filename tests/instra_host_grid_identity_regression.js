// vvv THOG equal short Grid numbers on different hosts never share eyes or colours
"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),vm=require("node:vm");
const source=fs.readFileSync("sheet/local_dashboard_assets/dashboard.js","utf8");
const start=source.indexOf("function hash_text("),end=source.indexOf("function is_visible(",start);
const runs=[
  {dashboard_run_id:"s1",runner_grid_tag:"G-00021",thog_host_id:"thog_host.scruffy"},
  {dashboard_run_id:"s2",runner_grid_tag:"G-00021",thog_host_id:"thog_host.scruffy"},
  {dashboard_run_id:"d1",runner_grid_tag:"G-00021",thog_host_id:"thog_host.dreedle"},
  {dashboard_run_id:"d2",runner_grid_tag:"G-00021",thog_host_id:"thog_host.dreedle"},
];
const stored=new Map(),context={load_json:()=>({}),app:{runs,colours:{}},run_identifier:run=>run.dashboard_run_id,default_palette:["#abc"],
  localStorage:{getItem:key=>stored.get(key),setItem:(key,value)=>stored.set(key,value)}};
vm.createContext(context);vm.runInContext(source.slice(start,end),context);
assert.notEqual(context.grid_identity(runs[0]),context.grid_identity(runs[2]));
assert.deepEqual(runs.filter(run=>context.grid_identity(run)===context.grid_identity(runs[0])).map(run=>run.dashboard_run_id),["s1","s2"]);
assert.notEqual(context.colour_for_run("s1").match(/^hsl\((\d+)/)[1],context.colour_for_run("d1").match(/^hsl\((\d+)/)[1]);
const shared_uuid="a".repeat(32);
assert.equal(context.grid_identity({...runs[0],runner_grid_id:shared_uuid}),context.grid_identity({...runs[2],runner_grid_id:shared_uuid}),
  "runs dispatched by the same owning Grid may span hosts");
assert.match(source,/app\.runs\.filter\(member=>grid_identity\(member\)===grid_key\)/);
console.log("PASS host-qualified legacy eyes/colours and shared durable Grid UUIDs");
// ^^^ THOG
