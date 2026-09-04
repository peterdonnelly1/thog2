// vvv THOG
"use strict";

// Manual browser fixture with the complete production asset stack.
// Set INSTRA_PLOTLY_BUNDLE to plotly.min.js, then open /runs/a on port 8765.
// No training data, model runtime, or browser automation dependencies.
const assert = require("node:assert/strict");
const {firefox} = require("playwright");
const fs = require("node:fs");
const path = require("node:path");
const http = require("node:http");
const root = path.resolve(__dirname, "..");
const assets = path.join(root, "sheet/local_dashboard_assets");
const chart_names = ["attn_q_head_N", "attn_k_head_N", "attn_v_head_N", "attn_out_head_N", "mlp_up", "mlp_down"];
const plotly_path = process.env.INSTRA_PLOTLY_BUNDLE || require.resolve("plotly.js-dist-min");
const runs = ["a", "b", "c"].map((id, index) => ({
  dashboard_run_id: id, local_run_id: id, run_name: id,
  artifact_name: `260901-120${index}_test_${id}`, run_state: "finished", model_type: index ? "thog2_sheet" : "dense",
  host_label: "test", created_at: "2026-09-01T12:00:00Z", maximum_update: 500 - index * 10,
  depth_snapshot_count: 51, depth_minimum_update: 1, depth_maximum_update: 500 - index * 10,
  heatmap_count: 0, heatmap_maximum_update: null, revision: [52 - index, 500 - index * 10, id],
  configuration: {warmup_iters: index * 10, n_layer: 16, n_embd: 16, n_head: 1, checkpoint_segment_size: 4, activation_checkpointing: true, learning_rate: 0.0009, min_learning_rate: 0.00009, lifecycle: {optimizer_name: index ? "adamw" : "sgd", optimizer_momentum: 0.9}},
}));

function depth_payload(url) {
  const minimum = url.searchParams.has("step_min") ? Number(url.searchParams.get("step_min")) : null;
  const maximum = url.searchParams.has("step_max") ? Number(url.searchParams.get("step_max")) : null;
  const run = runs.find(value => value.dashboard_run_id === url.searchParams.get("run")) || runs[0];
  const run_offset = runs.indexOf(run) * 0.001;
  const steps = [0, 1, ...Array.from({length: 50}, (_, index) => (index + 1) * 10)]
    .filter(step => step <= run.depth_maximum_update && (minimum === null || (step >= minimum && step <= maximum)));
  if (url.searchParams.get("current_only") === "1" && steps.length) steps.splice(0, steps.length - 1);
  const depth = {};
  const layer_x = Array.from({length: 16}, (_, index) => index + 1);
  if (steps.length) for (const chart_name of chart_names) depth[chart_name] = {
    data: steps.map(step => {
      const values = layer_x.map(layer => Math.sin(layer) * 0.01 + step * 0.00001 + run_offset);
      return {type: "scatter", mode: "lines+markers", name: `step ${step}`, x: layer_x, y: values,
        meta: {instra_weight_selection_protocol: "matched_six_v1", instra_weight_selection_kind: "random",
          instra_weight_model_feature: 0, instra_weight_intermediate_feature: 0, instra_weight_feature_count: 16,
          ...(run.model_type === "dense"
            ? {instra_dense_weight: true, instra_dense_optimizer_update: step, instra_dense_scalar_id: "r0_c0"}
            : {instra_thog_weight: true, instra_thog_optimizer_update: step, instra_thog_scalar_id: "r0_c0",
              instra_thog_integer_x: layer_x, instra_thog_integer_y: values})},
        line: {color: "#2878a0"}, hovertemplate: `step ${step}<extra></extra>`};
    }), layout: {},
  };
  return {depth, ...(minimum === null ? {} : {weight_step_range: {minimum, maximum, snapshot_count: steps.length}})};
}

let live_step = 0;
const server = http.createServer(async (request, response) => {
  const url = new URL(request.url, "http://localhost");
  if (url.pathname.startsWith("/api/")) {
    let result = {};
    if (url.pathname === "/api/runs") result = {runs, root: "/test/logs", recommended_run_id: "a", waiting: false};
    else if (url.pathname === "/api/status") result = runs.find(run => run.dashboard_run_id === url.searchParams.get("run")) || runs[0];
    else if (url.pathname === "/api/figure-family" && url.searchParams.get("family") === "depth") {
      result = depth_payload(url);
    } else if (url.pathname === "/api/figures") result = {heatmap: null, ...depth_payload(url)};
    else if (url.pathname.includes("weight-selection")) result = {protocol: "matched_six_v1", user_selected: false, model_feature: 0, intermediate_feature: 0};
    else if (url.pathname === "/api/optimizer-history") result = {figures: {}, errors: {}, steps: [], snapshots: []};
    else if (url.pathname === "/api/chart-groups") result = {available: true, groups:
      url.searchParams.get("run") === "c" && !live_step ? [] : ["train", "val"].map(name => ({name, revision: 118 + live_step, chart_count: 1}))};
    else if (url.pathname === "/api/chart-group") {
      const name = url.searchParams.get("group");
      const x = url.searchParams.get("run") === "c"
        ? (live_step ? [live_step] : [])
        : Array.from({length: name === "train" ? 118 : 3}, (_, index) => index * (name === "train" ? 1 : 50));
      result = {available: true, group: {name, revision: 118 + live_step, charts: [{id: "loss", title: "Loss", x_title: "step", default_x_axis_mode: "step", available_x_axis_modes: ["step"], series: [{name: "Loss", x, x_variants: {step: x}, y: x.map(step => 5 - step / 200), points: x.length}]}]}};
    }
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
  const page = await browser.newPage({viewport: {width: 1800, height: 1000}});
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  const watchdog = setTimeout(() => { console.error("Instra browser watchdog"); void browser.close(); }, 90000);
  const section = name => page.locator('.local-metric-group[data-metric-group="' + name + '"]');
  const choose = async id => page.locator('tr[data-run-id="' + id + '"] .run-link').click();
  const is_open = async name => !(await section(name).getAttribute("class")).split(" ").includes("collapsed");
  const wait_plot = async name => page.waitForFunction(name => {
    const mount = document.querySelector('.local-metric-group[data-metric-group="' + name + '"] .plot-mount');
    return mount?.data?.some(trace => trace.y?.length);
  }, name, {timeout: 15000});
  try {
    await page.goto("http://127.0.0.1:" + server.address().port + "/runs/a");
    await page.waitForFunction(() => window.__instra_further_weight_owner && window.__thog2_metric_groups);
    await section("train").waitFor();
    await section("train").locator(".chart-group-toggle").click();
    await wait_plot("train");
    console.log("Initial train renders");
    await page.locator('[data-detail-tab="overview"]').click();
    await choose("b");
    assert.equal(await page.locator("#run_overview_pane").isVisible(), true, "run switch lost Overview");
    await page.locator('[data-detail-tab="charts"]').click();
    await wait_plot("train");
    assert.equal(await is_open("train"), true, "run switch collapsed train");
    assert.equal(await is_open("val"), false, "run switch expanded val");
    await page.evaluate(() => { window.test_train_node = document.querySelector('[data-metric-group="train"]'); });
    await page.locator('tr[data-run-id="b"] .colour-dot').click();
    await page.locator("#colour_swatches .colour-swatch").first().click();
    await page.keyboard.press("Escape");
    await page.waitForTimeout(1600);
    assert.equal(await is_open("train"), true, "colour edit collapsed train");
    console.log("Runs navigation and colour state pass");

    const headers = await page.locator(".runs-table thead th").evaluateAll(nodes => nodes.filter(node => getComputedStyle(node).display !== "none").map(node => node.textContent.trim()));
    const state_index = headers.findIndex(value => value.toUpperCase() === "STATE");
    assert.match(headers[state_index + 1], /RUN NAME/);
    assert.equal(headers[headers.findIndex(value => value.toUpperCase() === "STEPS") + 1], "w");
    const swatch = await page.locator('tr[data-run-id="b"] .colour-dot').evaluate(node => {
      const style = getComputedStyle(node); return [style.width, style.height, style.borderRadius];
    });
    assert.deepEqual(swatch, ["18px", "12px", "3px"]);
    await page.locator('[data-detail-tab="overview"]').click();
    const overview = await page.locator("#overview_metadata").evaluate(node => ({
      width: node.getBoundingClientRect().width, parent_width: node.parentElement.clientWidth,
      colour: getComputedStyle(node.querySelector(".overview-meta-value")).color
    }));
    assert.equal(overview.colour, "rgb(179, 109, 22)");
    assert.ok(overview.width > overview.parent_width * .85, "Overview metadata left excessive whitespace");
    await page.locator('[data-detail-tab="charts"]').click();

    await choose("c");
    assert.equal(await is_open("train"), true);
    assert.equal(await section("train").evaluate(node => node === window.test_train_node), true, "run switch replaced train section");
    live_step = 10;
    const start = Date.now();
    await wait_plot("train");
    assert.ok(Date.now() - start < 5000, "new run first point was delayed");
    assert.equal(await section("train").locator(".plot-mount").evaluate(node => node.data[0].x[0]), 10);
    console.log("New-run empty-to-first-point transition passes");

    await page.locator("#workspace_nav").click();
    await section("train").waitFor();
    if (!(await is_open("train"))) await section("train").locator(".chart-group-toggle").click();
    await wait_plot("train");
    await choose("b");
    await wait_plot("train");
    assert.equal(await page.evaluate(() => app.workspace_mode), true, "selecting a Workspace run left Workspace");
    await page.waitForTimeout(1600);
    let widths = await section("train").locator(".plot-mount").evaluate(node => Object.fromEntries(node.data.map(trace => [trace.meta.instra_workspace_run_id, trace.line.width])));
    assert.ok(widths.b > widths.a, "selected run not emphasized");
    await choose("a");
    await wait_plot("train");
    await page.waitForTimeout(1600);
    widths = await section("train").locator(".plot-mount").evaluate(node => Object.fromEntries(node.data.map(trace => [trace.meta.instra_workspace_run_id, trace.line.width])));
    assert.ok(widths.a > widths.b && Math.abs(widths.b - 2.4) < .001, "previous run width not restored");
    const hover = await section("train").locator(".plot-mount").evaluate(node => ({mode: node.layout.hovermode, spikes: node.layout.xaxis.showspikes}));
    assert.deepEqual(hover, {mode: "closest", spikes: true});
    await page.evaluate(() => { window.test_workspace_train_node = document.querySelector('[data-metric-group="train"]'); });
    await page.locator('tr[data-run-id="a"] .colour-dot').click();
    await page.locator("#colour_swatches .colour-swatch").nth(1).click();
    await page.keyboard.press("Escape");
    await page.waitForTimeout(2200);
    assert.equal(await is_open("train"), true);
    assert.equal(await section("train").evaluate(node => node === window.test_workspace_train_node), true, "Workspace colour update destroyed train");
    console.log("Workspace emphasis, hover and colour state pass");

    const weights = page.locator("#coefficients_chart_group");
    if ((await weights.getAttribute("class")).includes("collapsed")) await weights.locator(".chart-group-toggle").click();
    await page.locator("#weight_step_overlapping_range").waitFor({state: "visible"});
    assert.equal(await page.locator("#weight_step_overlapping_range").getAttribute("aria-pressed"), "true", "initial overlap highlight wrong");
    assert.equal(await page.locator("#weight_step_whole_range").getAttribute("aria-pressed"), "false");
    assert.equal(await page.locator("#weight_step_overlapping_range").textContent(), "overlapping range");
    const order = await page.locator("#weight_step_group_controls").evaluate(node => [...node.querySelectorAll("button")].map(button => button.id));
    assert.ok(order.indexOf("weight_step_gradient") > order.indexOf("weight_step_overlapping_range"));
    assert.ok(order.indexOf("weight_z_cycle") > order.indexOf("weight_step_gradient"));
    await page.locator("#weight_step_whole_range").click();
    assert.equal(await page.locator("#weight_step_whole_range").getAttribute("aria-pressed"), "true");
    assert.equal(await page.locator("#weight_step_overlapping_range").getAttribute("aria-pressed"), "false");
    for (const kind of ["momentum", "scaling"]) {
      const history = page.locator('[data-chart-group="optimizer_' + kind + '"]');
      await history.locator(".chart-group-toggle").click();
      assert.equal(await history.locator(".thogopt-toolbar").isVisible(), false);
    }
    assert.equal(await page.locator('[data-chart-group="optimizer_momentum"] strong').textContent(), "momentum history");
    console.log("Range controls and history collapse pass");

    await page.locator("#runs_nav").click();
    await choose("a");
    await page.locator("#weights_group_settings_button").click();
    await page.locator("#chart_join_with_line_segments").check();
    await page.locator("#save_chart_settings").click();
    await page.waitForFunction(() => document.getElementById("chart_settings_overlay").hidden);
    await choose("c");
    assert.equal(await page.evaluate(() => normalize_chart_settings("mlp_up").join_with_line_segments), true, "new run did not inherit group line segments");
    assert.equal(await page.evaluate(() => JSON.parse(localStorage.getItem("thog2_local_weight_group_settings_v1")).runs.join_with_line_segments), true);
    assert.deepEqual(errors, []);
    await page.screenshot({path: "instra_sep03_browser.png"});
    console.log("Instra September 3 full-stack Firefox regression PASS");
  } catch (error) {
    console.error("Browser errors:", errors);
    console.error(await page.locator("body").innerText().catch(() => ""));
    throw error;
  } finally {
    clearTimeout(watchdog); await browser.close(); await new Promise(resolve => server.close(resolve));
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
// ^^^ THOG
