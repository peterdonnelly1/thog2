"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const base = fs.readFileSync("sheet/local_dashboard_assets/dashboard.js", "utf8");
const table = fs.readFileSync("sheet/local_dashboard_assets/dashboard_runs_table_restore.js", "utf8");
const between = (source, start, end) => {
  const first = source.indexOf(start);
  const last = source.indexOf(end, first);
  assert.ok(first >= 0 && last > first, `missing ${start}`);
  return source.slice(first, last);
};

const app = {runs:[
  {id:"ten", maximum_update:10, last_loss:11, configuration:{n_layer:10}},
  {id:"missing", maximum_update:null, last_loss:null, configuration:{}},
  {id:"two", maximum_update:2, last_loss:2, configuration:{n_layer:2}},
], sort_descending:true, column_sort_key:null};
const controls = {run_search:{value:""}, state_filter:{value:"all"}, run_sort:{value:"updated"}};
const sandbox = {app, by_id:id => controls[id], display_run_state:() => "finished",
  is_active_run_state:() => false, run_identifier:run => run.id, Date, Number, Object, String};
vm.runInNewContext(between(table, "const finite_number =", "const order =") +
  between(base, "function filtered_runs()", "function reset_pagination()"), sandbox);
const keys = ["steps", "duration", "loss", "gpu", "gb", "layers", "depth_order", "parms",
  "equiv", "warmup", "context", "d_model", "heads", "grad_accum",
  "activation_checkpointing", "learning_rate", "min_learning_rate", "probe_start",
  "probe_end", "curve_start", "curve_end", "capture_period", "updated"];
for (const key of keys) assert.equal(typeof app.column_sort_values[key], "function", `missing ordinal sorter ${key}`);
const ids = () => Array.from(sandbox.filtered_runs(), run => run.id);
app.column_sort_key = "layers";
app.sort_descending = false;
assert.deepEqual(ids(), ["two", "ten", "missing"]);
app.sort_descending = true;
assert.deepEqual(ids(), ["ten", "two", "missing"]);
app.column_sort_key = "loss";
app.sort_descending = false;
assert.deepEqual(ids(), ["two", "ten", "missing"]);
app.column_sort_key = "steps";
assert.deepEqual(ids(), ["two", "ten", "missing"]);

let click, renders = 0;
const heading = {dataset:{instraColumnKey:"layers"}};
vm.runInNewContext(between(table,
  'document.querySelector(".runs-table thead")?.addEventListener("click", event => {',
  '\n\n    polish();'), {
  app, document:{querySelector:() => ({addEventListener:(_type, callback) => { click = callback; }})},
  localStorage:{setItem() {}}, update_sort_direction_ui() {}, reset_pagination() { renders++; },
});
app.column_sort_key = null;
click({target:{closest:() => heading}});
assert.equal(app.column_sort_key, "layers");
assert.equal(app.sort_descending, false);
click({target:{closest:() => heading}});
assert.equal(app.sort_descending, true);
assert.equal(renders, 2);

let fetches = 0, finish;
const catalogue = {runs:[], waiting:true, requested_run:null, recommended_run_id:null, root:"/logs"};
const fields = {watch_status:{}, topbar_state:{}};
const poll = {app:{runs:[], catalog_refresh_in_flight:false, current_run_id:null},
  fetch_json:() => { fetches++; return new Promise(resolve => { finish = resolve; }); },
  by_id:id => fields[id], render_runs() {}, render_run_heading() {}, render_empty_state() {},
  AbortController, setTimeout:() => 1, clearTimeout() {}};
vm.runInNewContext(between(base, "async function refresh_catalog()", "function clear_plot("), poll);
(async () => {
  const first = poll.refresh_catalog();
  await poll.refresh_catalog();
  assert.equal(fetches, 1, "overlapping catalogue polls started duplicate requests");
  finish(catalogue);
  await first;
  const next = poll.refresh_catalog();
  assert.equal(fetches, 2, "poller did not resume after a completed request");
  finish(catalogue);
  await next;
  console.log("PASS ordinal heading sorting and bounded catalogue polling");
})().catch(error => { console.error(error); process.exitCode = 1; });
