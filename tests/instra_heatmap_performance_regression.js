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
  let heatmap_renders = 0;
  let depth_renders = 0;
  let cleared_plots = 0;
  let range_signature = "whole";
  const depth_payloads = [];
  const requests = [];
  const click_handlers = new Map();
  const groups = {
    heatmap_chart_group: {classList: {contains: name => name === "collapsed" ? !heatmap_open : false}},
    coefficients_chart_group: {classList: {contains: name => name === "collapsed" ? !coefficients_open : false}},
    heatmap_group_toggle: {addEventListener: (_event, handler) => click_handlers.set("heatmap", handler)},
    coefficients_group_toggle: {addEventListener: (_event, handler) => click_handlers.set("coefficients", handler)},
    heatmap_card_detail: {textContent: ""},
    heatmap_placeholder: {hidden: false},
    heatmap_plot: {id: "heatmap_plot", dataset: {}},
    local_metric_train_plot: {id: "local_metric_train_plot", dataset: {plotReady: "true"}},
    local_metric_val_plot: {id: "local_metric_val_plot", dataset: {plotReady: "true"}},
    q_detail: {textContent: ""},
    q_placeholder: {hidden: false},
    q_plot: {id: "q_plot", dataset: {}},
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
      return depth_payloads.shift() || {depth: {}};
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
      __instra_weight_step_filter: {signature: () => range_signature},
    },
    by_id: id => groups[id] || null,
    depth_weight_chart_names: ["q"],
    chart_titles: {heatmap: "Heatmap", q: "Q", local_metric_train: "Loss", local_metric_val: "Val loss"},
    fetch_json: base_fetch,
    render_figures: async () => undefined,
    clear_plot: mount => { assert.ok(!mount.id.startsWith("local_metric_"), "weight refresh erased a train/val chart"); cleared_plots += 1; mount.dataset.plotReady = "false"; },
    render_plot: async mount => {
      if (mount.id === "heatmap_plot") heatmap_renders += 1;
      else depth_renders += 1;
      mount.dataset.plotReady = "true";
    },
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
  await context.render_figures();
  assert.equal(groups.heatmap_plot.dataset.instraRenderedRunId, "R1");

  // A current cached payload with a blank/stale DOM mount must redraw locally. It
  // must not wait for a new probe or issue a redundant network refresh.
  groups.heatmap_plot.dataset.plotReady = "false";
  delete groups.heatmap_plot.dataset.instraRenderedRunId;
  const renders_before_mount_repair = heatmap_renders;
  await context.window.__thog2_dashboard_performance.refresh_family_if_stale("heatmap");
  assert.equal(refresh_calls, 0, "stale heatmap mount incorrectly refetched current payload");
  assert.equal(heatmap_renders, renders_before_mount_repair + 1, "stale heatmap mount was not redrawn");
  assert.equal(groups.heatmap_plot.dataset.instraRenderedRunId, "R1");

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
  await context.window.__thog2_dashboard_performance.refresh_family_if_stale("heatmap");
  assert.equal(refresh_calls, 0, "reopening a stale heatmap detoured through the whole-run refresh path");
  assert.equal(requests.length, 3, "reopening a stale heatmap did not fetch its family directly");

  payload = await context.fetch_json("/api/figures?run=R1");
  assert.equal(requests.length, 3, "current heatmap family was redundantly fetched again");
  parsed = new URL(requests.at(-1), "http://127.0.0.1:6007");
  assert.equal(parsed.searchParams.get("family"), "heatmap");
  assert.equal(context.window.__thog2_dashboard_performance.state.deferred_heatmap, false);

  // A family response can race a live SQLite writer. If status already reports
  // retained snapshots, an empty depth object is not a valid completed cache
  // entry: the next family wake must retry it directly.
  coefficients_open = true;
  app.current_status.depth_snapshot_count = 101;
  app.current_status.depth_maximum_update = 400;
  app.figures.depth = {};
  depth_payloads.push(
    {depth: {}},
    {depth: {q: {data: [{meta: {optimizer_update: 400}}], layout: {}}}},
  );
  payload = await context.fetch_json("/api/figures?run=R1");
  app.figures = payload;
  assert.equal(
    context.window.__thog2_dashboard_performance.state.depth_signature,
    null,
    "known retained Weights were cached as a successful blank payload",
  );
  assert.equal(app.figure_revision, null, "blank Weights did not keep the live refresh path retryable");

  await context.window.__thog2_dashboard_performance.refresh_family_if_stale("depth");
  assert.ok(app.figures.depth.q, "direct Weights family retry did not recover retained curves");
  assert.equal(depth_renders, 1, "recovered Weights family was not rendered exactly once");
  assert.notEqual(
    context.window.__thog2_dashboard_performance.state.depth_signature,
    null,
    "recovered Weights payload did not become current",
  );

  assert.ok(click_handlers.has("heatmap"), "heatmap group toggle wake handler was not installed");

  // The actual Weights toggle must refresh the depth family, retain confirmed
  // empty-range metadata, clear old curves, and avoid repeatedly fetching zero.
  range_signature = "0:0";
  depth_payloads.push({depth: {}, weight_step_range: {minimum: 0, maximum: 0, snapshot_count: 0}});
  click_handlers.get("coefficients")();
  await new Promise(setImmediate);
  assert.equal(app.figures.weight_step_range?.snapshot_count, 0, "Weights toggle dropped empty step-zero metadata");
  assert.deepEqual(Object.keys(app.figures.depth), []);
  assert.equal(cleared_plots, 1, "old curves remained on the empty initial-values plot");
  const after_initial = requests.length;
  payload = await context.fetch_json("/api/figures?run=R1");
  assert.equal(requests.length, after_initial, "confirmed empty step zero was fetched repeatedly");
  assert.equal(payload.weight_step_range?.snapshot_count, 0, "cached empty response lost its range metadata");

  range_signature = "1:1";
  depth_payloads.push({depth: {q: {data: [{meta: {optimizer_update: 1}}]}}, weight_step_range: {minimum: 1, maximum: 1, snapshot_count: 1}});
  payload = await context.fetch_json("/api/figures?run=R1");
  assert.equal(requests.length, after_initial + 1, "new range reused the empty-zero family cache");
  assert.equal(payload.weight_step_range.snapshot_count, 1);
}

(async () => {
  await heatmap_family_demand_regression();
  console.log("instra heatmap performance regression: PASS");
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
// ^^^ THOG
