// vvv THOG
"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");

const repository_root = path.resolve(__dirname, "..");
const source = fs.readFileSync(path.join(repository_root, "run_thog2_dashboard.py"), "utf8");

const active_asset = name => new RegExp(`^\\s*"${name.replaceAll(".", "\\.")}",`, "m").test(source);
const index_of = name => source.indexOf(`"${name}"`);

assert.equal(active_asset("dashboard_weight_request_router_patch.js"), true);
assert.equal(active_asset("dashboard_legacy_heatmap_repair_patch.js"), true);
assert.equal(active_asset("dashboard_weight_stability_final_patch.js"), true);
assert.ok(index_of("dashboard_weight_request_router_patch.js") < index_of("dashboard_performance_patch.js"));
assert.ok(index_of("dashboard_weight_stability_final_patch.js") > index_of("dashboard_weight_step_controls_patch.js"));

for (const name of [
  "dashboard_weight_step_hyperparameter_patch.js",
  "dashboard_regression_repair_patch.js",
  "dashboard_weight_step_placeholder_cleanup_patch.js",
  "dashboard_weight_semantics_repair_patch.js",
  "dashboard_weight_style_semantics_repair_patch.js",
]) {
  assert.equal(active_asset(name), false, `${name} is still an active runtime owner`);
}

console.log("instra weight runtime loader regression: PASS");
// ^^^ THOG
