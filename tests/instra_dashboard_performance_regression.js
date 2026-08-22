// vvv THOG
"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const repository_root = path.resolve(__dirname, "..");
const asset = name => path.join(repository_root, "sheet/local_dashboard_assets", name);
const load_source = name => fs.readFileSync(asset(name), "utf8");

async function workspace_depth_cache_regression() {
  let current_only = true;
  let visible_runs = [
    {dashboard_run_id: "A", depth_snapshot_count: 100, depth_maximum_update: 1000},
    {dashboard_run_id: "B", depth_snapshot_count: 100, depth_maximum_update: 900},
    {dashboard_run_id: "C", depth_snapshot_count: 40, depth_maximum_update: 400},
  ];
  const requests = [];
  let fail_run = null;

  const workspace = {
    visible_runs: () => visible_runs,
    fetch_depth_payload: async request => {
      const entries = [];
      for (const run of visible_runs) {
        const run_id = run.dashboard_run_id;
        entries.push(await request(`/api/figure-family?run=${encodeURIComponent(run_id)}&family=depth`));
      }
      return {entries};
    },
  };

  const context = {
    console,
    URL,
    app: {
      current_run_id: "A",
      current_status: {depth_snapshot_count: 100, depth_maximum_update: 1000},
    },
    depth_weight_chart_names: ["q", "k", "v", "o", "up", "down"],
    normalize_chart_settings: () => ({current_weights_only: current_only}),
    run_identifier: run => run.dashboard_run_id,
    window: {
      location: {origin: "http://127.0.0.1:6007"},
      __instra_workspace: workspace,
      addEventListener(event, callback) { if (event === "load") callback(); },
    },
    setTimeout(callback) { callback(); return 1; },
  };
  context.window.window = context.window;
  vm.createContext(context);
  vm.runInContext(load_source("dashboard_workspace_depth_cache_patch.js"), context);

  const request = async url => {
    const parsed = new URL(url, "http://127.0.0.1:6007");
    const run_id = parsed.searchParams.get("run");
    requests.push(run_id);
    if (run_id === fail_run) throw new Error(`synthetic failure for ${run_id}`);
    return {run_id, serial: requests.length};
  };

  let payload = await workspace.fetch_depth_payload(request);
  assert.equal(requests.length, 3, "initial Workspace depth load must fetch every visible run");
  assert.deepEqual(payload.entries.map(item => item.run_id), ["A", "B", "C"]);

  payload = await workspace.fetch_depth_payload(request);
  assert.equal(requests.length, 3, "unchanged visible runs were fetched again");
  assert.deepEqual(payload.entries.map(item => item.run_id), ["A", "B", "C"]);

  // /api/status is polled more frequently than /api/runs. A fresh selected-run
  // revision must invalidate the cache even while the catalog row is still stale.
  context.app.current_status.depth_maximum_update = 1001;
  await workspace.fetch_depth_payload(request);
  assert.equal(requests.length, 4, "fresh current-run status did not invalidate its cached depth payload");
  assert.equal(requests.at(-1), "A");

  visible_runs[0].depth_maximum_update = 1001;
  await workspace.fetch_depth_payload(request);
  assert.equal(requests.length, 4, "catalog catch-up refetched an already-current run");

  current_only = false;
  await workspace.fetch_depth_payload(request);
  assert.equal(requests.length, 7, "current-only/history mode transition must invalidate every visible payload");

  visible_runs = visible_runs.slice(0, 2);
  await workspace.fetch_depth_payload(request);
  assert.equal(requests.length, 7, "hiding a run should not refetch unchanged remaining runs");
  assert.equal(context.window.__instra_workspace_depth_cache.size(), 2, "hidden run cache entry was not evicted");

  visible_runs[1].depth_maximum_update = 901;
  fail_run = "B";
  await assert.rejects(() => workspace.fetch_depth_payload(request), /synthetic failure/);
  const after_failure = requests.length;
  fail_run = null;
  await workspace.fetch_depth_payload(request);
  assert.equal(
    requests.length,
    after_failure + 1,
    "failed depth response poisoned the cache instead of being retried",
  );
  assert.equal(requests.at(-1), "B");

  const before_stress = requests.length;
  for (let iteration = 0; iteration < 10000; iteration += 1) {
    await workspace.fetch_depth_payload(request);
  }
  assert.equal(
    requests.length,
    before_stress,
    "unchanged Workspace stress refreshes leaked network requests",
  );

  const stats = context.window.__instra_workspace_depth_cache.stats();
  assert.ok(stats.hits >= 20005, "Workspace depth cache recorded too few stress-test hits");
  assert.ok(stats.misses >= 8, "Workspace depth cache recorded too few misses");
  assert.equal(stats.entries, 2);
}

async function render_visibility_regression() {
  const chart_names = ["q", "k", "v", "o", "up", "down"];
  const cards = Object.fromEntries(chart_names.map(name => [name, {offsetParent: {}}]));
  const mounts = Object.fromEntries(chart_names.map(name => [name, {
    chart_name: name,
    closest(selector) { return selector === ".chart-card" ? cards[name] : null; },
  }]));
  const rendered = [];

  const context = {
    console,
    depth_weight_chart_names: chart_names,
    app: {current_run_id: "R1", maximized_chart: "q"},
    render_plot: async (mount, figure, chart_name) => {
      rendered.push({chart_name, figure, run_id: context.app.current_run_id});
    },
    toggle_maximized_chart: chart_name => {
      context.app.maximized_chart = context.app.maximized_chart === chart_name ? null : chart_name;
    },
    restore_maximized_chart: () => { context.app.maximized_chart = null; },
    toggle_chart_group: () => undefined,
    select_run: run_id => { context.app.current_run_id = run_id; },
    show_toast: message => { throw new Error(`unexpected toast: ${message}`); },
    requestAnimationFrame(callback) { callback(); return 1; },
    setTimeout(callback) { callback(); return 1; },
    window: {
      addEventListener(event, callback) { if (event === "load") callback(); },
    },
  };
  context.window.window = context.window;
  vm.createContext(context);
  vm.runInContext(load_source("dashboard_render_visibility_performance_patch.js"), context);

  await context.render_plot(mounts.q, {version: 1}, "q");
  for (const chart_name of chart_names.slice(1)) {
    await context.render_plot(mounts[chart_name], {version: 1}, chart_name);
  }
  assert.deepEqual(rendered.map(item => item.chart_name), ["q"], "hidden weight charts were still Plotly-rendered");
  assert.equal(context.window.__instra_render_visibility_performance.pending_count(), 5);

  for (let version = 2; version <= 1000; version += 1) {
    for (const chart_name of chart_names.slice(1)) {
      await context.render_plot(mounts[chart_name], {version}, chart_name);
    }
  }
  assert.equal(rendered.length, 1, "hidden fullscreen stress updates reached Plotly");
  assert.equal(
    context.window.__instra_render_visibility_performance.pending_count(),
    5,
    "hidden fullscreen stress updates accumulated instead of coalescing",
  );

  context.toggle_maximized_chart("k");
  await context.window.__instra_render_visibility_performance.flush();
  const k_renders = rendered.filter(item => item.chart_name === "k");
  assert.equal(k_renders.length, 1, "newly visible maximized chart was not flushed once");
  assert.equal(
    k_renders[0].figure.version,
    1000,
    "deferred chart did not keep only the newest hidden update",
  );

  await context.render_plot(mounts.q, {version: 1001}, "q");
  assert.equal(rendered.filter(item => item.chart_name === "q").length, 1, "newly hidden prior chart was rendered");

  context.restore_maximized_chart();
  await context.window.__instra_render_visibility_performance.flush();
  assert.equal(rendered.filter(item => item.chart_name === "q").length, 2, "restore did not flush hidden Q");
  for (const chart_name of ["v", "o", "up", "down"]) {
    const chart_renders = rendered.filter(item => item.chart_name === chart_name);
    assert.equal(chart_renders.length, 1, `restore did not flush hidden ${chart_name}`);
    assert.equal(chart_renders[0].figure.version, 1000, `${chart_name} did not flush its newest hidden update`);
  }
  assert.equal(context.window.__instra_render_visibility_performance.pending_count(), 0);

  context.app.maximized_chart = "q";
  await context.render_plot(mounts.k, {version: 2000}, "k");
  assert.equal(context.window.__instra_render_visibility_performance.pending_count(), 1);
  context.select_run("R2");
  assert.equal(context.window.__instra_render_visibility_performance.pending_count(), 0, "run switch retained stale deferred figures");
  context.restore_maximized_chart();
  await context.window.__instra_render_visibility_performance.flush();
  assert.ok(rendered.every(item => item.run_id === "R1"), "stale deferred figure leaked into the new run");
}

function preparing_observer_regression() {
  const source = load_source("dashboard_preparing_workspace_train_patch.js");
  assert.match(
    source,
    /charts_observer\.observe\(charts_scroll, \{childList: true\}\);/,
    "Workspace train observer is no longer direct-child-only",
  );
  const charts_observer_block = source.slice(
    source.indexOf("const charts_observer"),
    source.indexOf("const workspace_observer"),
  );
  assert.ok(!charts_observer_block.includes("subtree: true"), "Workspace train observer again watches Plotly descendants");
  assert.ok(!charts_observer_block.includes("attributes: true"), "Workspace train observer again watches Plotly attributes");
  assert.match(charts_observer_block, /touches_train_group/, "observer no longer filters specifically for train-group changes");
}

(async () => {
  await workspace_depth_cache_regression();
  await render_visibility_regression();
  preparing_observer_regression();
  console.log("instra dashboard performance regression: PASS");
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
// ^^^ THOG
