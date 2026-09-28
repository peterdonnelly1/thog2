"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

class Element {
  constructor(tag = "div") {
    this.tagName = tag; this.children = []; this.events = {}; this.dataset = {}; this.style = {setProperty() {}};
    this.textContent = ""; this.value = "";
    const names = new Set();
    this.classList = {add: name => names.add(name), contains: name => names.has(name),
      toggle: (name, enabled) => { if (enabled) names.add(name); else names.delete(name); }};
  }
  append(child) { this.children.push(child); return child; }
  replaceChildren() { this.children.length = 0; }
  addEventListener(name, callback) { this.events[name] = callback; }
  querySelectorAll(selector) { return selector === "button" ? this.children : []; }
  querySelector(selector) { return selector === ".runner-event-log" ?
    this.children.find(child => child.className === "runner-event-log") : null; }
  setAttribute(name, value) { this[name] = value; }
  click() { return this.events.click?.(); }
}

async function main() {
  const roots = Object.fromEntries(["runner_view","runner_list","runner_detail","runner_multiview_panel","runner_message","runner_tabs",
    "runner_nav","runs_nav","workspace_nav","networks_nav","settings_nav"].map(id => [id,new Element()]));
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
  const network = {hosts:[]};
  const requests = [];
  let log_events = '{"time":"2026-09-27T21:00:00Z","event":"launch"}\n';
  const fetch = async (url, options) => {
    requests.push({url,options});
    const data = url === "/api/runner" ? snapshot : url === "/api/network" ? network : {};
    return {ok:true,json:async()=>data,text:async()=>log_events};
  };
  const context = {window:{innerWidth:1200}, document, fetch, AbortController, URLSearchParams,
    setInterval() {},setTimeout:()=>1,clearTimeout() {},Date,JSON,Number,String,
    confirm:()=>true};
  vm.runInNewContext(fs.readFileSync("sheet/local_dashboard_assets/dashboard_runner.js","utf8"),context);
  const switch_tab = name => roots.runner_tabs.children.find(tab => tab.dataset.runnerTab === name).click();
  roots.runner_nav.click();
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(roots.runner_list.children.slice(1).map(row => row.children[0].textContent),["Latest","Older"]);
  assert.equal(roots.runner_list.children[1].children[1].textContent,"Delete");
  switch_tab("progress");
  assert.equal(roots.runner_list.children.length,1,"finished Grids must disappear from Progress");
  assert.equal(roots.runner_detail.children[1].children[1].textContent,"running");
  switch_tab("history");
  assert.ok(roots.runner_list.children[0].textContent.startsWith("G-00002"));
  const summaries = roots.runner_detail.children.filter(child => child.tagName === "details");
  assert.equal(summaries[0].children[0].children[0].textContent,"newer", "latest attempt first");
  const header = roots.runner_detail.children.find(child => child.className === "runner-run-headings");
  assert.deepEqual(header.children.map(child => child.textContent),["Run ID","State","Step","Loss","Host","GPU","Profiling"]);
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
  roots.runner_nav.click();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(roots.runner_multiview_panel.children.find(child => child.tagName === "iframe"),frame,
    "Runner polling must keep the chart frame mounted");
  switch_tab("progress");
  switch_tab("multiview");
  assert.equal(roots.runner_multiview_panel.children.find(child => child.tagName === "iframe"),frame,
    "switching views must not reload a running Grid's charts");
  const html = fs.readFileSync("sheet/local_dashboard_assets/index.html","utf8");
  assert.match(html,/data-runner-tab="progress"[^]*data-runner-tab="multiview"[^]*data-runner-tab="current_scripts"/);
  assert.match(html,/data-runner-tab="history"[^]*data-runner-tab="log"/);
  assert.equal((html.match(/<path d="M9 \d+h\d+" stroke="#[0-9a-f]+"\/>/g)||[]).length,4);
  assert.equal((html.match(/<path d="M9 \d+ C\d+ \d+ \d+ \d+ 56 \d+"/g)||[]).length,3);

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
  console.log("PASS Runner ordering, retained History/Log, active scoped Multiview, grey icons and maximized chart selection");
}
main().catch(error=>{console.error(error);process.exitCode=1;});
