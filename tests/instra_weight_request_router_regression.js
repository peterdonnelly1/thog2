// vvv THOG
"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const repository_root = path.resolve(__dirname, "..");
const source = fs.readFileSync(
  path.join(repository_root, "sheet/local_dashboard_assets/dashboard_weight_request_router_patch.js"),
  "utf8",
);

async function run_case(flags, range, {workspace = false, mode = "whole"} = {}) {
  const requests = [];
  const names = ["q", "k", "v", "o", "up", "down"];
  const settings = Object.fromEntries(names.map((name, index) => [name, {current_weights_only: flags[index]}]));
  const context = {
    console,
    URL,
    depth_weight_chart_names: names,
    app: {workspace_mode: workspace},
    normalize_chart_settings: name => settings[name],
    fetch_json: async url => { requests.push(String(url)); return {}; },
    window: {
      location: {origin: "http://127.0.0.1:6007"},
      __instra_weight_step_filter: {request_range: () => range},
      __instra_weight_stability_final: {mode: () => mode},
      addEventListener(name, callback) { if (name === "load") callback(); },
    },
  };
  context.window.window = context.window;
  vm.createContext(context);
  vm.runInContext(source, context);
  await context.fetch_json("/api/figure-family?run=A&family=depth");
  return new URL(requests.at(-1), "http://127.0.0.1:6007");
}

(async () => {
  // An explicit display window is authoritative regardless of the current-only flags.
  let request = await run_case([true, true, true, true, true, true], {minimum: 20, maximum: 30});
  assert.equal(request.searchParams.get("step_min"), "20");
  assert.equal(request.searchParams.get("step_max"), "30");
  assert.equal(request.searchParams.has("current_only"), false);

  request = await run_case([false, false, false, false, false, false], {minimum: 20, maximum: 30});
  assert.equal(request.searchParams.get("step_min"), "20");
  assert.equal(request.searchParams.get("step_max"), "30");
  assert.equal(request.searchParams.has("current_only"), false);

  request = await run_case([true, false, false, false, false, false], {minimum: 20, maximum: 30});
  assert.equal(request.searchParams.get("step_min"), "20");
  assert.equal(request.searchParams.get("step_max"), "30");
  assert.equal(request.searchParams.has("current_only"), false);

  // Without an explicit range the all-current case still takes the efficient latest path.
  request = await run_case([true, true, true, true, true, true], null);
  assert.equal(request.searchParams.get("current_only"), "1");
  assert.equal(request.searchParams.has("step_min"), false);
  assert.equal(request.searchParams.has("step_max"), false);

  request = await run_case([false, false, false, false, false, false], null);
  assert.equal(request.searchParams.has("current_only"), false);
  assert.equal(request.searchParams.has("step_min"), false);

  // Workspace latest is run-relative: each visible run's independent family
  // request asks the server for that run's own most recent retained snapshot.
  request = await run_case(
    [false, false, false, false, false, false],
    null,
    {workspace: true, mode: "latest"},
  );
  assert.equal(request.searchParams.get("current_only"), "1");
  assert.equal(request.searchParams.has("step_min"), false);
  assert.equal(request.searchParams.has("step_max"), false);

  console.log("instra weight request router regression: PASS");
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
// ^^^ THOG
