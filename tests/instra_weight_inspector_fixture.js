// vvv THOG
"use strict";

// Manual browser fixture with the complete production asset stack.
// Set INSTRA_PLOTLY_BUNDLE to plotly.min.js, then open /runs/a on port 8765.
// No training data, model runtime, or browser automation dependencies.
const fs = require("node:fs");
const path = require("node:path");
const http = require("node:http");
const root = path.resolve(__dirname, "..");
const assets = path.join(root, "sheet/local_dashboard_assets");
const chart_names = ["attn_q_head_N", "attn_k_head_N", "attn_v_head_N", "attn_out_head_N", "mlp_up", "mlp_down"];
const plotly_path = process.env.INSTRA_PLOTLY_BUNDLE || require.resolve("plotly.js-dist-min");
const runs = ["a", "b", "c", "d", "e", "f"].map((id, index) => ({
  dashboard_run_id: id, local_run_id: id, run_name: id,
  artifact_name: `260901-120${index}_test_${id}`, run_state: "finished", model_type: index ? "thog2_sheet" : "dense",
  host_label: "test", created_at: "2026-09-01T12:00:00Z", maximum_update: 500 - index * 10,
  depth_snapshot_count: 51, depth_minimum_update: 1, depth_maximum_update: 500 - index * 10,
  heatmap_count: 0, heatmap_maximum_update: null, revision: [52 - index, 500 - index * 10, id],
  configuration: {n_layer: 144, n_embd: 16, n_head: 1},
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
  const layer_x = Array.from({length: 144}, (_, index) => index + 1);
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


server.listen(Number(process.env.INSTRA_FIXTURE_PORT || 8765), "127.0.0.1", () => console.log("Instra fixture ready"));
// ^^^ THOG
