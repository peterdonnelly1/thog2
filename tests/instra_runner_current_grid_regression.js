"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const runner_css = fs.readFileSync("sheet/local_dashboard_assets/dashboard_runner.css", "utf8");
assert.match(runner_css, /#runner_list \.runner-recipe-row button:nth-child\(2\)\s*\{[^}]*width:auto/,
  "Rename Grid must override the general full-width list-button rule");
assert.match(runner_css, /#runner_list \.runner-recipe-row button:first-child\s*\{[^}]*flex:1 1 0/,
  "Recipe names must receive the remaining row width");

class Element {
  constructor(tag = "div") {
    this.tagName = tag; this.children = []; this.events = {}; this.dataset = {}; this.style = {setProperty() {}};
    this.textContent = ""; this.value = "";
    const names = new Set();
    this.classList = {add: name => names.add(name), contains: name => names.has(name),
      toggle: (name, enabled) => { if (enabled) names.add(name); else names.delete(name); }};
  }
  append(child) { this.children.push(child); return child; }
  insertBefore(child, sibling) {
    this.children = this.children.filter(item => item !== child);
    const index = this.children.indexOf(sibling);
    this.children.splice(index < 0 ? this.children.length : index, 0, child);
  }
  replaceChildren() { this.children.length = 0; }
  addEventListener(name, callback) { this.events[name] = callback; }
  querySelectorAll(selector) { return selector === "button" ? this.children : []; }
  querySelector(selector) {
    const matches = child => selector.startsWith(".") ?
      String(child.className || "").split(" ").includes(selector.slice(1)) : child.tagName === selector;
    return this.children.find(matches) || this.children.map(child => child.querySelector(selector)).find(Boolean) || null;
  }
  setAttribute(name, value) { this[name] = value; }
  click() { return this.events.click?.(); }
  showModal() { this.open = true; }
  close() { this.open = false; }
  focus() {}
  select() {}
}

async function main() {
  const roots = Object.fromEntries(["runner_view","runner_list","runner_detail","runner_multiview_panel","runner_message","runner_tabs",
    "runner_nav","runs_nav","workspace_nav","networks_nav","settings_nav","runner_rename_dialog",
    "runner_rename_form","runner_rename_input","runner_rename_cancel"].map(id => [id,new Element()]));
  roots.runner_rename_dialog.append(new Element("h2"));
  for (const name of ["recipes","progress","multiview","current_scripts","history","log","files"]) {
    const tab = new Element("button"); tab.dataset.runnerTab = name; roots.runner_tabs.append(tab);
  }
  const find = (node, id) => node.id === id ? node : node.children.map(child => find(child,id)).find(Boolean);
  const document = {createElement: tag => new Element(tag), getElementById: id => roots[id] ||
    Object.values(roots).map(node => find(node,id)).find(Boolean)};
  const run = (id, state, started_at) => ({run_id:id,grid_id:"grid-1",state,host_label:"scruffy",
    gpu:{ordinal:0,model:"GPU",gpu_key:"GPU-0"},profiler:"none",required_mib:2000,parameters:{},
    attempts:[{attempt_id:id,started_at,state,exit_code:0}]});
  const active = {grid_id:"grid-1",grid_tag:"G-00002",label:"Live",state:"running",recipe_id:"latest",
    created_at:"2026-09-27T21:00:00Z",runs:[run("older","completed","2026-09-27T20:00:00Z"),
      run("newer","running","2026-09-27T21:00:00Z")]};
  const finished = {...active,grid_id:"grid-0",grid_tag:"G-00001",label:"Finished",state:"completed",runs:[]};
  const snapshot = {recipes:[{recipe_id:"old",created_at:"2026-09-26",recipe:{label:"Older",parameters:{}}},
    {recipe_id:"latest",created_at:"2026-09-27",recipe:{label:"Latest",parameters:{}}}],
    grids:[finished,active],catalogue:{},common:[]};
  const network = {hosts:[{thog_host_id:"dreedle-host",display_name:"dreedle",local:true,
    last_discovered:{execution_profiles:[{profile_key:"current"}],gpus:[
      {gpu_id:"dreedle-host.gpu.GPU-0",gpu_key:"GPU-0",ordinal:0},
      {gpu_id:"dreedle-host.gpu.GPU-1",gpu_key:"GPU-1",ordinal:1},
    ]}}]};
  const requests = [];
  let log_events = '{"time":"2026-09-27T21:00:00Z","event":"launch"}\n';
  const fetch = async (url, options) => {
    requests.push({url,options});
    const data = url === "/api/runner" ? snapshot : url === "/api/network" ? network :
      url === "/api/runs" ? {runs:[{runner_run_id:"newer",last_loss:3.14159,best_loss:2.71828}]} : {};
    return {ok:true,json:async()=>data,text:async()=>log_events};
  };
  const stored=new Map();
  const localStorage = {getItem:key=>stored.get(key)||null,setItem:(key,value)=>stored.set(key,value)};
  const context = {window:{innerWidth:1200}, document, fetch, AbortController, URLSearchParams,localStorage,
    setInterval() {},setTimeout:()=>1,clearTimeout() {},Date,JSON,Number,String,
    confirm:()=>true,prompt:()=>{throw Error("Browser prompt used for renaming");}};
  vm.runInNewContext(fs.readFileSync("sheet/local_dashboard_assets/dashboard_runner.js","utf8"),context);
  const switch_tab = name => roots.runner_tabs.children.find(tab => tab.dataset.runnerTab === name).click();
  roots.runner_nav.click();
  await new Promise(resolve => setImmediate(resolve));
  const default_recipe=context.window.instra_runner_test_hooks.current_recipe();
  assert.equal(default_recipe.power_caps["dreedle-host.gpu.GPU-0"],200);
  assert.equal(default_recipe.power_caps["dreedle-host.gpu.GPU-1"],200);
  context.window.instra_runner_test_hooks.remember_default("--n-layer",24);
  assert.equal(context.window.instra_runner_test_hooks.current_recipe().parameters["--n-layer"],24);
  context.window.instra_runner_test_hooks.remember_default("power_caps",{"dreedle-host.gpu.GPU-1":180});
  assert.equal(context.window.instra_runner_test_hooks.current_recipe().power_caps["dreedle-host.gpu.GPU-0"],200);
  assert.equal(context.window.instra_runner_test_hooks.current_recipe().power_caps["dreedle-host.gpu.GPU-1"],180);
  assert.equal(context.window.instra_runner_test_hooks.current_recipe().parameters.power_caps,undefined);
  assert.deepEqual(roots.runner_list.children.slice(1).map(row => row.children[0].textContent),["Latest","Older"]);
  assert.equal(roots.runner_list.children[1].children[1].textContent,"Rename Grid");
  assert.equal(roots.runner_list.children[1].children[2].textContent,"Delete");
  const rename_click = roots.runner_list.children[1].children[1].click();
  assert.equal(roots.runner_rename_dialog.open,true);
  roots.runner_rename_input.value="Renamed Recipe";
  roots.runner_rename_form.onsubmit({preventDefault() {}});
  await rename_click;
  const rename = requests.find(entry => entry.url === "/api/runner/action" &&
    JSON.parse(entry.options.body).recipe?.label === "Renamed Recipe");
  assert.equal(JSON.parse(rename.options.body).recipe_id,"latest","Recipe rename must preserve its ID");
  switch_tab("progress");
  assert.equal(roots.runner_list.children.length,1,"finished Grids must disappear from Progress");
  const active_row=roots.runner_list.children[0];
  assert.equal(active_row.children[1].textContent,"Rename Grid");
  assert.equal(active_row.children[2].textContent,"Kill and Flush");
  assert.match(active_row.children[2].title,/Last resort/);
  assert.equal(roots.runner_detail.children[1].children[1].textContent,"running");
  switch_tab("history");
  assert.ok(roots.runner_list.children[0].children[0].textContent.startsWith("G-00002"));
  assert.deepEqual(roots.runner_list.children[0].children.slice(1,3).map(item=>item.textContent),["Rename","Delete"]);
  assert.equal(roots.runner_list.children[0].children.at(-1).textContent,"running");
  assert.equal(roots.runner_list.children[0].children[0].textContent,"G-00002 · Live");
  const summaries = roots.runner_detail.children.filter(child => child.className === "runner-run");
  assert.equal(summaries[0].children[0].children[0].textContent,"newer", "latest attempt first");
  const header = roots.runner_detail.children.find(child => child.className === "runner-run-headings");
  assert.deepEqual(header.children.map(child => child.textContent),["Run ID","State","Step","Loss","Best loss","Host","GPU","Profiling"]);
  assert.equal(summaries[0].children[0].children[3].textContent,"3.142");
  assert.equal(summaries[0].children[0].children[4].textContent,"2.718");
  switch_tab("log");
  await new Promise(resolve => setImmediate(resolve));
  const log_viewer = roots.runner_detail.querySelector(".runner-event-log");
  assert.ok(log_viewer.textContent.includes("LAUNCH"));
  log_events += '{"time":"2026-09-27T21:00:01Z","event":"run","detail":"started"}\n';
  roots.runner_nav.click();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(roots.runner_detail.querySelector(".runner-event-log"),log_viewer,
    "live log polling must keep the viewer mounted and its scroll position");
  assert.ok(log_viewer.textContent.includes("RUN  started"),"new Grid events must appear without changing tabs");
  switch_tab("multiview");
  const frame = roots.runner_multiview_panel.children.find(child => child.tagName === "iframe");
  assert.equal(frame.src,"/?runner_grid_tag=G-00002");
  assert.equal(roots.runner_multiview_panel.hidden,false);
  const selector=roots.runner_multiview_panel.querySelector("select");
  assert.equal(selector.value,"grid-1");
  roots.runner_nav.click();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(roots.runner_multiview_panel.children.find(child => child.tagName === "iframe"),frame,
    "Runner polling must keep the chart frame mounted");
  const other={...active,grid_id:"grid-2",grid_tag:"G-00003",label:"GPU 1",runs:[{
    ...run("second-gpu","running","2026-09-27T21:30:00Z"),gpu:{ordinal:1,model:"GPU",gpu_key:"GPU-1"}
  }]};
  snapshot.grids.push(other);
  roots.runner_nav.click();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(roots.runner_multiview_panel.querySelector("select").children.length,2);
  assert.equal(roots.runner_multiview_panel.children.find(child => child.tagName === "iframe").src,
    "/?runner_grid_tag=G-00003", "automatic selection follows the newest running Grid");
  other.state="completed";
  roots.runner_nav.click();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(roots.runner_multiview_panel.children.find(child => child.tagName === "iframe").src,
    "/?runner_grid_tag=G-00002", "an automatically selected finished Grid yields to a running Grid");
  other.state="running";
  roots.runner_nav.click();
  await new Promise(resolve => setImmediate(resolve));
  const new_selector=roots.runner_multiview_panel.querySelector("select");
  new_selector.value="grid-2";new_selector.events.change();
  assert.equal(roots.runner_multiview_panel.children.find(child => child.tagName === "iframe").src,
    "/?runner_grid_tag=G-00003");
  other.state="completed";
  roots.runner_nav.click();
  await new Promise(resolve => setImmediate(resolve));
  assert.match(roots.runner_multiview_panel.querySelector(".runner-empty-state").textContent,/selected Grid has finished/);
  const finished_selector=roots.runner_multiview_panel.querySelector("select");
  roots.runner_nav.click();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(roots.runner_multiview_panel.querySelector("select"),finished_selector,
    "polling a finished selection must not repeatedly rebuild the view");
  other.state="running";
  roots.runner_nav.click();
  await new Promise(resolve => setImmediate(resolve));
  switch_tab("progress");
  assert.equal(roots.runner_list.children.length,2,"both active Grids must remain selectable");
  assert.equal(roots.runner_detail.children[0].children[0].textContent,"G-00003 · GPU 1");
  await roots.runner_list.children[0].children[2].click();
  const flush=requests.find(entry=>entry.url==="/api/runner/action" &&
    JSON.parse(entry.options.body).action==="kill_flush");
  assert.equal(JSON.parse(flush.options.body).grid_id,"grid-2");
  switch_tab("multiview");
  assert.equal(roots.runner_multiview_panel.children.find(child => child.tagName === "iframe").src,
    "/?runner_grid_tag=G-00003");
  assert.notEqual(roots.runner_multiview_panel.children.find(child => child.tagName === "iframe"),frame,
    "switching to another Grid must show that Grid's charts");
  switch_tab("progress");
  roots.runner_list.children[1].children[0].click();
  switch_tab("multiview");
  assert.equal(roots.runner_multiview_panel.children.find(child => child.tagName === "iframe").src,
    "/?runner_grid_tag=G-00002",
    "switching views must not reload a running Grid's charts");
  const html = fs.readFileSync("sheet/local_dashboard_assets/index.html","utf8");
  assert.match(html,/data-runner-tab="progress"[^]*data-runner-tab="multiview"[^]*data-runner-tab="current_scripts"/);
  assert.match(html,/data-runner-tab="history"[^]*data-runner-tab="log"/);
  assert.equal((html.match(/<path d="M9 \d+h\d+" stroke="#[0-9a-f]+"\/>/g)||[]).length,4);
  assert.match(html, /M7 5V57H59/);
  assert.equal((html.match(/M11 \d+ C\d+ \d+ \d+ \d+ 56 \d+/g)||[]).length,3);

  const base = fs.readFileSync("sheet/local_dashboard_assets/dashboard.js","utf8");
  const begin = base.indexOf("function select_run(run_id, options = {})");
  const end = base.indexOf("function resize_plot_in_card",begin);
  let resets = 0, restores = 0, refreshes = 0;
  const app = {workspace_mode:true,current_run_id:"old",runs:[{id:"new"}],file_request_serial:0};
  const select_context = {app,window:{location:{pathname:"/runs/old",search:"?runner_grid_tag=G-00002"}},
    history:{pushState(_a,_b,path){select_context.window.location.pathname=path.split("?")[0];}},
    by_id:()=>({value:""}),run_identifier:run=>run.id,
    restore_maximized_chart(){restores++;},reset_run_charts(){resets++;},
    render_runs(){},render_run_heading(){},render_empty_state(){},refresh_current_run(){refreshes++;},encodeURIComponent};
  vm.runInNewContext(base.slice(begin,end),select_context);
  select_context.select_run("new",{manual:true});
  assert.equal(restores,0,"switching a Multiview row must preserve maximized charts");
  assert.equal(resets,0,"switching a Multiview row must preserve chart mounts");
  assert.equal(refreshes,1);
  assert.equal(select_context.window.location.pathname,"/runs/new");
  const repair = fs.readFileSync("sheet/local_dashboard_assets/dashboard_sep21_instra_repairs.js","utf8");
  const count_start = repair.indexOf("    function place_training_throughput() {");
  const count_end = repair.indexOf("    // vvv THOG in Runs view",count_start);
  let has_loss = false;
  const count = {textContent:""};
  const card = {classList:{add(){}},dataset:{},parentElement:null,
    querySelector:()=>({textContent:""})};
  const grid = {appendChild(node){node.parentElement=this;},
    querySelector:selector=>selector.includes("train/loss") && has_loss ? {id:"train/loss"} : null};
  const train = {querySelector:selector=>selector.includes("count") ? count : grid};
  const count_context = {document:{querySelector:()=>train},
    by_id:id=>({training_chart_group:{hidden:false},training_throughput_card:card})[id] || null,
    standardize_throughput_plot(){},layout_train_charts(){}};
  vm.runInNewContext(repair.slice(count_start,count_end) + "\nplace_training_throughput();",count_context);
  assert.equal(count.textContent,"1","Train must not claim a missing loss chart exists");
  has_loss=true;
  count_context.place_training_throughput();
  assert.equal(count.textContent,"2","Train must count its recovered loss and throughput charts");
  const startup=fs.readFileSync("sheet/local_dashboard_assets/dashboard_sep07_fixes_and_enhancements.js","utf8");
  assert.match(startup,/startup_collapsed = \{[^}]*processing: true/s,
    "Processing must start collapsed so throughput does not displace the initial Loss view");
  const metrics=fs.readFileSync("sheet/local_dashboard_assets/dashboard_wandb_groups_patch.js","utf8");
  assert.match(metrics,/grid\.querySelectorAll\("\.local-metric-card"\)\.length/,
    "group discovery must count charts actually drawn, not hardcode two");
  console.log("PASS Runner ordering, retained History/Log, active scoped Multiview, grey icons and maximized chart selection");
}
main().catch(error=>{console.error(error);process.exitCode=1;});
