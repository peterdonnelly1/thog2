// vvv THOG
"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const repository_root = path.resolve(__dirname, "..");
const asset = name => path.join(repository_root, "sheet/local_dashboard_assets", name);
const load_source = name => fs.readFileSync(asset(name), "utf8");

async function request_routing_regression() {
  let current_only = true;
  let selected_range = null;
  const requests = [];
  const context = {
    console,
    URL,
    depth_weight_chart_names: ["q", "k", "v", "o", "up", "down"],
    normalize_chart_settings: () => ({current_weights_only: current_only}),
    fetch_json: async url => {
      requests.push(String(url));
      return {url: String(url)};
    },
    window: {
      location: {origin: "http://127.0.0.1:6007"},
      __instra_weight_step_filter: {
        request_range: () => selected_range,
      },
    },
  };
  context.window.window = context.window;
  vm.createContext(context);
  vm.runInContext(load_source("dashboard_current_weights_request_patch.js"), context);

  await context.fetch_json("/api/figure-family?run=A&family=depth");
  let parsed = new URL(requests.at(-1), "http://127.0.0.1:6007");
  assert.equal(parsed.searchParams.get("current_only"), "1");
  assert.equal(parsed.searchParams.has("step_min"), false);

  selected_range = {minimum: 120, maximum: 145};
  await context.fetch_json("/api/figure-family?run=A&family=depth&current_only=1");
  parsed = new URL(requests.at(-1), "http://127.0.0.1:6007");
  assert.equal(parsed.searchParams.get("step_min"), "120");
  assert.equal(parsed.searchParams.get("step_max"), "145");
  assert.equal(parsed.searchParams.has("current_only"), false, "explicit range did not override latest-only request");

  selected_range = {minimum: 333, maximum: 333};
  await context.fetch_json("/api/figure-family?run=A&family=depth");
  parsed = new URL(requests.at(-1), "http://127.0.0.1:6007");
  assert.equal(parsed.searchParams.get("step_min"), "333");
  assert.equal(parsed.searchParams.get("step_max"), "333");

  selected_range = null;
  current_only = false;
  await context.fetch_json("/api/figure-family?run=A&family=depth");
  parsed = new URL(requests.at(-1), "http://127.0.0.1:6007");
  assert.equal(parsed.searchParams.has("current_only"), false);
  assert.equal(parsed.searchParams.has("step_min"), false);
}

function signed_log_regression() {
  const style_nodes = [];
  const base_prepare = figure => JSON.parse(JSON.stringify(figure));
  const context = {
    console,
    structuredClone: global.structuredClone,
    depth_weight_chart_names: ["q"],
    app: {
      workspace_mode: false,
      current_run_id: "A",
      current_status: null,
      refresh_in_flight: false,
      figure_revision: null,
      axis_chart_name: null,
      axis_chart_workspace_mode: null,
    },
    window: {
      location: {origin: "http://127.0.0.1:6007"},
      addEventListener(event, callback) { if (event === "load") callback(); },
      __instra_workspace: {visible_runs: () => []},
      __instra_matched_weight_selection: null,
    },
    document: {
      activeElement: null,
      querySelector() { return null; },
      querySelectorAll() { return []; },
      createElement() { return {textContent: "", style: {}, classList: {toggle() {}}}; },
      head: {appendChild(node) { style_nodes.push(node); }},
    },
    by_id: () => null,
    normalize_chart_settings: () => ({current_weights_only: false}),
    retain_latest_weight_snapshots: () => undefined,
    instra_enforce_workspace_latest_weights: prepared => prepared,
    current_run: () => ({depth_snapshot_count: 10, depth_minimum_update: 10, depth_maximum_update: 100}),
    run_identifier: run => run?.dashboard_run_id || "",
    figure_for_chart: () => null,
    prepare_figure: base_prepare,
    render_runs: () => undefined,
    render_run_heading: () => undefined,
    refresh_current_run: () => undefined,
    show_toast: () => undefined,
    load_json: (key, fallback) => key === "thog2_local_trajectory_scale_modes" ? {q: "log"} : fallback,
    setTimeout(callback) { callback(); return 1; },
    setInterval() { return 1; },
    clearInterval() {},
    queueMicrotask,
  };
  context.window.window = context.window;
  vm.createContext(context);
  vm.runInContext(load_source("dashboard_weight_step_controls_patch.js"), context);

  const prepared = context.prepare_figure({
    data: [],
    layout: {
      yaxis: {
        tickvals: [-2, -1, 0, 1, 2],
        ticktext: ["-0.01", "-0.001", "0", "0.001", "0.01"],
      },
    },
  }, "q");

  const labels = prepared.layout.yaxis.ticktext;
  assert.ok(labels.includes("1e-04"), "signed-log scale did not add a smaller positive decade");
  assert.ok(labels.includes("-1e-04"), "signed-log scale did not add a smaller negative decade");
  assert.ok(labels.includes("1e-03"));
  assert.ok(labels.includes("-1e-02"));
  assert.ok(labels.includes("0e+00"));
  assert.ok(
    labels.every(label => /^-?\d+e[+-]\d{2}$/.test(label)),
    `non-scientific signed-log label remained: ${labels.join(", ")}`,
  );
  assert.ok(style_nodes.length >= 1, "final weight controls style was not installed");
}

function structural_control_regression() {
  const source = load_source("dashboard_weight_step_controls_patch.js");
  assert.match(source, /data available for steps \$\{range\.minimum\} – \$\{range\.maximum\}/);
  assert.match(source, /Math\.max\(\.\.\.ranges\.map\(range => range\.minimum\)\)/);
  assert.match(source, /Math\.min\(\.\.\.ranges\.map\(range => range\.maximum\)\)/);
  assert.match(source, /reason: "no overlapping steps"/);
  assert.match(source, /raw_maximum === "" \? minimum/,
    "blank range end no longer means one exact step");
  assert.match(source, /selected_step_range = \{minimum, maximum\}/);
  assert.match(source, /weight_step_whole_range/);
  assert.match(source, /selected_step_range = null/);

  assert.match(source, /if \(step_filter_active\(\)\) return;/,
    "Runs-view latest-only collapse is no longer bypassed for explicit step windows");
  assert.match(source, /if \(step_filter_active\(\)\) return prepared;/,
    "Workspace latest-only collapse is no longer bypassed for explicit step windows");

  assert.match(source, /label\.textContent = "weight matrix feature coupling \(i → o\):"/);
  assert.match(source, /random\.textContent = "RND"/);
  assert.match(source, /weight_coupling_input/);
  assert.match(source, /weight_coupling_output/);
  assert.match(source, /full\.slice\(0, 10\)/, "host truncation is no longer exactly 10 characters");
  assert.match(source, /headers\[logged_index\]\.textContent = "STEPS"/);
  assert.match(source, /new_actual = smallest\.actual_value \/ 10/,
    "extra smaller signed-log decade was removed");
  assert.match(source, /padStart\(2, "0"\)/,
    "scientific exponent padding was removed");
  assert.ok(!source.includes("MutationObserver"), "weight-step controls reintroduced a persistent DOM observer");
  assert.match(source, /clearInterval\(startup_timer\)/,
    "weight-step startup reconciliation is no longer bounded");
}

(async () => {
  await request_routing_regression();
  signed_log_regression();
  structural_control_regression();
  console.log("instra weight step/filter regression: PASS");
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
// ^^^ THOG
