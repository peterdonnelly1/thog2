// vvv THOG
"use strict";

// Real Firefox + Plotly, complete production asset stack, deterministic HTTP fixtures.
// Supply NODE_PATH for playwright and INSTRA_PLOTLY_BUNDLE for plotly.min.js.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const http = require("node:http");
const {firefox} = require("playwright");
const root = path.resolve(__dirname, "..");
const assets = path.join(root, "sheet/local_dashboard_assets");
const chart_names = ["attn_q_head_N", "attn_k_head_N", "attn_v_head_N", "attn_out_head_N", "mlp_up", "mlp_down"];
const plotly_path = process.env.INSTRA_PLOTLY_BUNDLE || require.resolve("plotly.js-dist-min");
let include_zero = false;
let delay_ms = 0;
const requests = [];
const errors = [];
const runs = ["a", "b", "c"].map((id, index) => ({
  dashboard_run_id: id, local_run_id: id, run_name: id,
  artifact_name: `260901-120${index}_test_${id}`, run_state: "finished", model_type: index ? "thog2_sheet" : "dense",
  host_label: "test", created_at: "2026-09-01T12:00:00Z", maximum_update: 500,
  depth_snapshot_count: 51, depth_minimum_update: 1, depth_maximum_update: 500,
  heatmap_count: 0, heatmap_maximum_update: null, revision: [51, 500, id],
  configuration: {n_layer: 16, n_embd: 16, n_head: 1},
}));

function depth_payload(url) {
  const minimum = url.searchParams.has("step_min") ? Number(url.searchParams.get("step_min")) : null;
  const maximum = url.searchParams.has("step_max") ? Number(url.searchParams.get("step_max")) : null;
  const steps = [...(include_zero ? [0] : []), 1, ...Array.from({length: 50}, (_, index) => (index + 1) * 10)]
    .filter(step => minimum === null || (step >= minimum && step <= maximum));
  if (url.searchParams.get("current_only") === "1") steps.splice(0, steps.length - 1);
  const depth = {};
  if (steps.length) for (const chart_name of chart_names) depth[chart_name] = {
    data: steps.map(step => ({type: "scatter", mode: "lines+markers", name: `step ${step}`,
      x: Array.from({length: 16}, (_, index) => index + 1),
      y: Array.from({length: 16}, (_, index) => Math.sin(index) * 0.01 + step * 0.00001),
      meta: {instra_thog_optimizer_update: step, instra_weight_selection_protocol: "matched_six_v1",
        instra_weight_selection_kind: "random", instra_weight_model_feature: 0,
        instra_weight_intermediate_feature: 0, instra_weight_feature_count: 16},
      line: {color: "#2878a0"}, hovertemplate: `step ${step}<extra></extra>`})), layout: {},
  };
  return {depth, ...(minimum === null ? {} : {weight_step_range: {minimum, maximum, snapshot_count: steps.length}})};
}

const server = http.createServer(async (request, response) => {
  const url = new URL(request.url, "http://localhost");
  if (url.pathname.startsWith("/api/")) {
    requests.push(url.pathname + url.search);
    let result = {};
    if (url.pathname === "/api/runs") result = {runs, root: "/test/logs", recommended_run_id: "a", waiting: false};
    else if (url.pathname === "/api/status") result = runs.find(run => run.dashboard_run_id === url.searchParams.get("run")) || runs[0];
    else if (url.pathname === "/api/figure-family" && url.searchParams.get("family") === "depth") {
      result = depth_payload(url);
      if (delay_ms) await new Promise(resolve => setTimeout(resolve, delay_ms));
    } else if (url.pathname === "/api/figures") result = {heatmap: null, ...depth_payload(url)};
    else if (url.pathname.includes("weight-selection")) result = {protocol: "matched_six_v1", user_selected: false, model_feature: 0, intermediate_feature: 0};
    else if (url.pathname.includes("chart-groups")) result = {available: true, groups: []};
    response.writeHead(200, {"Content-Type": "application/json"}); response.end(JSON.stringify(result)); return;
  }
  let file = url.pathname === "/plotly.min.js" ? plotly_path : path.join(assets, path.basename(url.pathname));
  if (!url.pathname.startsWith("/assets/") && url.pathname !== "/plotly.min.js") file = path.join(assets, "index.html");
  try {
    let content = fs.readFileSync(file);
    if (file.endsWith("index.html")) {
      let html = content.toString();
      const launcher = fs.readFileSync(path.join(root, "run_thog2_dashboard.py"), "utf8");
      for (const match of launcher.matchAll(/"(dashboard_[a-z0-9_]+\.js)"/g)) {
        if (!html.includes(`/assets/${match[1]}`)) html = html.replace("</head>", `<script src="/assets/${match[1]}" defer></script></head>`);
      }
      content = html;
    }
    response.writeHead(200, {"Content-Type": file.endsWith(".js") ? "application/javascript" : file.endsWith(".css") ? "text/css" : "text/html"});
    response.end(content);
  } catch (_error) { response.writeHead(404); response.end(); }
});

(async () => {
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  const browser = await firefox.launch({headless: true});
  const watchdog = setTimeout(() => { console.error("Firefox test watchdog expired", {errors, requests: requests.slice(-8)}); void browser.close(); }, 60000);
  let page;
  try { page = await browser.newPage({viewport: {width: 1800, height: 1100}}); }
  catch (error) { clearTimeout(watchdog); await browser.close(); server.close(); throw error; }
  page.on("pageerror", error => errors.push(error.message));
  const settle = () => page.waitForFunction(() => !app.refresh_in_flight && Object.keys(app.figures?.depth || {}).length === 6
    && [...document.querySelectorAll(".plot-placeholder")].filter(node => node.id !== "heatmap_placeholder").every(node => node.hidden), {timeout: 30000});
  try {
    await page.goto(`http://127.0.0.1:${server.address().port}/runs/a`);
    console.log("Firefox: dashboard loaded");
    await page.waitForFunction(() => window.__instra_further_weight_owner && window.__instra_render_visibility_performance);
    console.log("Firefox: patch stack ready");
    await page.click("#workspace_nav");
    await page.evaluate(() => { document.querySelector("#coefficients_chart_group.collapsed .chart-group-header")?.click(); });
    await page.waitForFunction(() => document.getElementById("weight_step_whole_range"));
    await page.click("#weight_step_whole_range");
    await settle();
    console.log("Firefox: whole-range Workspace rendered");
    assert.equal(await page.evaluate(() => document.getElementById("weight_step_overlapping_range").nextElementSibling.id), "weight_z_cycle");
    assert.equal(await page.evaluate(() => getComputedStyle(document.getElementById("weight_z_cycle")).marginLeft), "12px");

    await page.click("#weight_step_initial_values");
    await page.waitForFunction(() => !app.refresh_in_flight && app.figures?.weight_step_range?.snapshot_count === 0);
    assert.equal(await page.getAttribute("#weight_step_initial_values", "aria-pressed"), "true");
    assert.equal(await page.locator("#attn_q_head_N_placeholder").textContent(), "No recorded initial weights (step 0) in this view.");
    assert.equal(await page.evaluate(() => document.getElementById("attn_q_head_N_plot").data?.length || 0), 0);
    const empty_requests = requests.filter(url => url.includes("figure-family")).length;
    await page.waitForTimeout(4500);
    assert.equal(requests.filter(url => url.includes("figure-family")).length, empty_requests, "empty range refetched indefinitely");

    include_zero = true;
    for (const run of runs) { run.depth_minimum_update = 0; run.depth_snapshot_count = 52; run.revision = [52, 500, run.dashboard_run_id]; }
    await page.click("#weight_step_whole_range"); await settle();
    await page.click("#weight_step_initial_values"); await settle();
    assert.equal(await page.evaluate(() => document.getElementById("attn_q_head_N_plot").data.every(trace => trace_optimizer_update(trace) === 0)), true);
    await page.click("#weight_step_whole_range"); await settle();
    await page.evaluate(() => toggle_maximized_chart("attn_out_head_N"));
    await page.click("#weight_z_cycle");
    await page.evaluate(() => restore_maximized_chart());
    await settle();
    await page.waitForFunction(() => window.__instra_render_visibility_performance.pending_count() === 0);
    assert.equal(await page.evaluate(() => [...document.querySelectorAll(".plot-mount")].filter(node => node.id !== "heatmap_plot")
      .every(node => node.dataset.plotReady === "true" && node.data.length > 0)), true);
    const unchanged_requests = requests.filter(url => url.includes("figure-family")).length;
    await page.waitForTimeout(4500);
    assert.equal(requests.filter(url => url.includes("figure-family")).length, unchanged_requests, "finished Workspace refetched indefinitely");

    delay_ms = 500;
    await page.click("#weight_step_initial_values");
    await page.waitForTimeout(100);
    await page.click("#weight_step_one");
    await settle();
    assert.equal(await page.evaluate(() => document.getElementById("attn_q_head_N_plot").data.every(trace => trace_optimizer_update(trace) === 1)), true);
    assert.equal(await page.getAttribute("#weight_step_one", "aria-pressed"), "true");
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({result: "PASS", browser: await browser.version(), requests: requests.length,
      depth_requests: requests.filter(url => url.includes("figure-family")).length}));
  } catch (error) {
    console.error("Firefox test failure:", error.message);
    console.error(JSON.stringify(await page.evaluate(() => ({state: app.current_run_id, workspace: app.workspace_mode,
      maximized: app.maximized_chart, in_flight: app.refresh_in_flight, figures: Object.keys(app.figures?.depth || {}),
      range: window.__instra_weight_step_filter?.request_range?.(), range_payload: app.figures?.weight_step_range,
      placeholders: [...document.querySelectorAll(".plot-placeholder")].map(node => [node.id,node.hidden,node.textContent])})), null, 2));
    console.error({errors, last_requests: requests.slice(-12)});
    throw error;
  } finally { clearTimeout(watchdog); await browser.close(); server.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; server.close(); });
// ^^^ THOG
