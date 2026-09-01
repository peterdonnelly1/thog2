// vvv THOG
"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

// Exercise the production refresh owner, including a range change during an
// outstanding HTTP request. No browser, network, or training dependencies needed.
const source = fs.readFileSync(path.join(__dirname, "../sheet/local_dashboard_assets/dashboard.js"), "utf8");
const refresh_source = source.slice(source.indexOf("async function refresh_current_run()"), source.indexOf("function select_run("));

async function main() {
  let view = "whole";
  let pending_response = null;
  let fail = false;
  let requests = 0;
  const rendered = [];
  const retries = [];
  const errors = [];
  const context = {
    app: {current_run_id: "A", workspace_mode: true, refresh_in_flight: false, figure_revision: null},
    window: {
      __instra_workspace: {selection_key: () => "A,B,C"},
      __instra_weight_step_filter: {signature: () => view},
    },
    fetch_json: async url => {
      if (url.startsWith("/api/status")) return {revision: "unchanged"};
      requests += 1;
      if (fail) throw new Error("transient failure");
      if (pending_response) return pending_response;
      return {view};
    },
    render_run_heading() {},
    render_figures: async () => { rendered.push(context.app.figures.view); },
    show_toast: message => errors.push(message),
    queueMicrotask: callback => retries.push(callback),
    encodeURIComponent,
  };
  vm.createContext(context);
  vm.runInContext(refresh_source, context);

  let resolve_old;
  pending_response = new Promise(resolve => { resolve_old = resolve; });
  const old_refresh = context.refresh_current_run();
  await new Promise(setImmediate);
  assert.equal(requests, 1);
  view = "0:0";
  await context.refresh_current_run(); // The first HTTP request still owns the lock.
  resolve_old({view: "whole"});
  await old_refresh;
  assert.deepEqual(rendered, [], "obsolete whole-range response overwrote the selected zero range");
  assert.equal(context.app.figure_revision, null);
  assert.equal(context.app.refresh_in_flight, false);
  assert.equal(retries.length, 1, "latest range was not retried after the old request finished");

  pending_response = null;
  await retries.shift()();
  assert.deepEqual(rendered, ["0:0"]);
  await context.refresh_current_run();
  assert.equal(requests, 2, "unchanged view unnecessarily fetched figures");
  view = "1:1";
  await context.refresh_current_run();
  assert.deepEqual(rendered, ["0:0", "1:1"], "unchanged server revision suppressed a new range");

  view = "whole";
  fail = true;
  await context.refresh_current_run();
  assert.equal(context.app.figure_revision, null, "failed request retained a successful revision");
  assert.equal(errors.length, 1);
  fail = false;
  await context.refresh_current_run();
  assert.equal(rendered.at(-1), "whole", "transient failure prevented a subsequent retry");
  console.log("instra September 1 refresh/range repair regression: PASS");
}

main().catch(error => { console.error(error); process.exitCode = 1; });
// ^^^ THOG
