// vvv THOG
"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const root = "sheet/local_dashboard_assets/";
const source = fs.readFileSync(`${root}dashboard_processing_sep16_stability_polish.js`, "utf8");
const start = source.indexOf("  const processing_render_timeline_before_legend_polish =");
const end = source.indexOf("  const processing_render_before_stability_polish =", start);
assert.ok(start >= 0 && end > start);

(async () => {
  const traces = Array.from({length:120}, (_unused, index) => ({
    name:index % 2 ? "MAIN attention" : "PREMAT materialize", showlegend:false,
  }));
  const mount = {data:traces};
  const restyles = [];
  const context = {
    processing_render_timeline:async () => {},
    by_id:() => mount,
    Plotly:{restyle:async (_mount, update, indices) => {
      restyles.push({update, indices});
      indices.forEach((index, offset) => {
        traces[index].name = update.name[offset];
        traces[index].showlegend = update.showlegend[offset];
      });
    }},
  };
  vm.createContext(context);
  vm.runInContext(source.slice(start, end), context);
  await context.processing_render_timeline({});
  assert.equal(restyles.length, 1, "timeline redrew once per trace");
  assert.equal(restyles[0].indices.length, 2);
  assert.deepEqual(traces.filter(trace => trace.showlegend).map(trace => trace.name),
    ["PREMAT materialize", "MAIN attention"]);
  await context.processing_render_timeline({});
  assert.equal(restyles.length, 1, "unchanged legend triggered another Plotly redraw");

  const quiescence = fs.readFileSync(`${root}dashboard_processing_sep16_quiescence.js`, "utf8");
  let tick;
  let on_visibility_change;
  const calls = [];
  const state = {visibilityState:"hidden"};
  const timer_context = {
    processing_view:{timer:17},
    document:{
      get visibilityState() { return state.visibilityState; },
      addEventListener(name, listener) {
        assert.equal(name, "visibilitychange");
        on_visibility_change = listener;
      },
    },
    window:{
      clearInterval(id) { assert.equal(id, 17); },
      setInterval(listener, delay) { assert.equal(delay, 1500); tick = listener; return 18; },
    },
    current_run:() => ({run_state:"recording"}),
    is_active_run_state:() => true,
    processing_refresh(force) { calls.push(force); },
  };
  vm.runInNewContext(quiescence, timer_context);
  for (let index = 0; index < 1000; index += 1) tick();
  assert.equal(calls.length, 0, "hidden tab still polled Processing");
  state.visibilityState = "visible";
  on_visibility_change();
  assert.deepEqual(calls, [true], "returning to Instra did not refresh Processing");
  tick();
  assert.deepEqual(calls, [true, undefined], "visible active run stopped polling");
  console.log("PASS Processing redraw batching and hidden-tab quiescence");
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
// ^^^ THOG
