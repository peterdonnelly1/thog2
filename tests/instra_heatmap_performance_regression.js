// vvv THOG
"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const repository_root = path.resolve(__dirname, "..");
const source = fs.readFileSync(
  path.join(repository_root, "sheet/local_dashboard_assets/dashboard_performance_patch.js"),
  "utf8",
);

async function heatmap_family_demand_regression() {
  let heatmap_open = true;
  let coefficients_open = false;
  let refresh_calls = 0;
  const requests = [];
  const click_handlers = new Map();
  const groups = {
    heatmap_chart_group: {classList: {contains: name => name === "collapsed" ? !heatmap_open : false}},
    coefficients_chart_group: {classList: {contains: name => name === "collapsed" ? !coefficients_open : false}},
    heatmap_group_toggle: {addEventListener: (_event, handler) => click_handlers.set("heatmap", handler)},
    coefficients_group_toggle: {addEventListener: (_event, handler) => click_handlers.set("coefficients", handler)},
    heatmap_card_detail: {textContent: ""},
    heatmap_placeholder: {hidden: false},
    heatmap_plot: {},
  };

  const app = {
    current_run_id: "R1",
    current_status: {
      heatmap_count: 2400,
      heatmap_maximum_update: 2400,
      depth_snapshot_count: 0,
      depth_maximum_update: null,
      heatmap_settings: {abs_limit: 0.05},
    },
    figures: {heatmap: null, heatmap_dimensions: {layers: 0, probes: 0}, depth: {}},
    figure_revision: null,
    refresh_in_flight: false,
  };

  const base_fetch = async url => {
    const text = String(url);
    requests.push(text);
    const parsed = new URL(text, "http://127.0.0.1:6007");
    if (parsed.pathname === "/api/figure-family" && parsed.searchParams.get("family") === "heatmap") {
      return {
        heatmap: {data: [{type: "heatmap", serial: requests.length}], layout: {}},
        heatmap_dimensions: {layers: 16, probes: 100},
      };
    }
    if (parsed.pathname === "/api/figure-family" && parsed.searchParams.get("family") === "depth") {
      return {depth: {}};
    }
    throw new Error(`unexpected base fetch: ${text}`);
  };

  const context = {
    console,
    URL,
    app,
    window: {
      location: {origin: "http://127.0.0.1:6007"},
      addEventListener(event, callback) { if (event === "load") callback(); },
      __thog2_synthetic_groups: {
        group_is_open: name => name === "heatmap" ? heatmap_open : coefficients_open,
      },
      __instra_workspace: null,
    },
    by_id: id => groups[id] || null,
    fetch_json: base_fetch,
    render_figures: async () => undefined,
    render_plot: async () => undefined,
    resize_plot_in_card: () => undefined,
    resize_visible_plots: () => undefined,
    select_run: run_id => { app.current_run_id = run_id; },
    current_run: () => app.current_status,
    refresh_current_run: () => { refresh_calls += 1; },
    format_integer: value => String(value),
    show_toast: message => { throw new Error(`unexpected toast: ${message}`); },
    requestAnimationFrame(callback) { callback(); return 1; },
    setTimeout(callback) { callback(); return 1; },
    queueMicrotask,
  };
  context.window.window = context.window;
  vm.createContext(context);
  vm.runInContext(source, context);

  let payload = await context.fetch_json("/api/figures?run=R1");
  assert.equal(requests.length, 1, "open heatmap did not issue exactly one family request");
  let parsed = new URL(requests.at(-1), "http://127.0.0.1:6007");
  assert.equal(parsed.pathname, "/api/figure-family");
  assert.equal(parsed.searchParams.get("family"), "heatmap");
  assert.equal(parsed.searchParams.get("probe_count"), "100");
  assert.equal(parsed.searchParams.get("window_mode"), "rolling");
  assert.ok(payload.heatmap, "heatmap family response was not merged into figure payload");
  app.figures = payload;

  payload = await context.fetch_json("/api/figures?run=R1");
  assert.equal(requests.length, 1, "unchanged heatmap revision refetched its family");
  assert.ok(payload.heatmap, "cached heatmap disappeared from empty-figures response");

  app.current_status.heatmap_count = 2401;
  app.current_status.heatmap_maximum_update = 2401;
  payload = await context.fetch_json("/api/figures?run=R1");
  assert.equal(requests.length, 2, "new heatmap probe did not invalidate the family cache");
  app.figures = payload;

  heatmap_open = false;
  app.current_status.heatmap_count = 2402;
  app.current_status.heatmap_maximum_update = 2402;
  payload = await context.fetch_json("/api/figures?run=R1");
  assert.equal(requests.length, 2, "closed heatmap group still fetched data");
  app.figures = payload;
  assert.equal(context.window.__thog2_dashboard_performance.state.deferred_heatmap, true);

  heatmap_open = true;
  context.window.__thog2_dashboard_performance.refresh_family_if_stale("heatmap");
  assert.equal(refresh_calls, 1, "reopening a stale heatmap did not wake the run refresh path");

  payload = await context.fetch_json("/api/figures?run=R1");
  assert.equal(requests.length, 3, "woken heatmap did not fetch the new revision");
  parsed = new URL(requests.at(-1), "http://127.0.0.1:6007");
  assert.equal(parsed.searchParams.get("family"), "heatmap");
  assert.equal(context.window.__thog2_dashboard_performance.state.deferred_heatmap, false);

  assert.ok(click_handlers.has("heatmap"), "heatmap group toggle wake handler was not installed");
}

(async () => {
  await heatmap_family_demand_regression();
  console.log("instra heatmap performance regression: PASS");
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
// ^^^ THOG
