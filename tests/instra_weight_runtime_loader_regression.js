// vvv THOG
"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");

const repository_root = path.resolve(__dirname, "..");
const source = fs.readFileSync(path.join(repository_root, "run_thog2_dashboard.py"), "utf8");
const active_source = source.match(/_ACTIVE_EXTRA_ASSET_NAMES\s*=\s*\(([\s\S]*?)\n\)/)?.[1] || "";

const active_asset = name => active_source.includes(`"${name}"`);

assert.equal(active_asset("dashboard_wandb_groups_patch.js"), true);
assert.equal(active_asset("dashboard_group_stability_patch.js"), true);
assert.equal(active_asset("dashboard_sep20_integrated_repairs.js"), true);

for (const name of [
  "dashboard_weight_request_router_patch.js",
  "dashboard_legacy_heatmap_repair_patch.js",
  "dashboard_weight_stability_final_patch.js",
  "dashboard_weight_coupling_reliability_patch.js",
  "dashboard_consistency_final_patch.js",
  "dashboard_aug30_enhancements_patch.js",
  "dashboard_weight_inspector.js",
  "dashboard_current_weights_request_patch.js",
  "dashboard_weight_step_hyperparameter_patch.js",
  "dashboard_regression_repair_patch.js",
  "dashboard_weight_step_placeholder_cleanup_patch.js",
  "dashboard_weight_semantics_repair_patch.js",
  "dashboard_weight_style_semantics_repair_patch.js",
]) {
  assert.equal(active_asset(name), false, `${name} is still an active runtime owner`);
}

console.log("instra bounded runtime loader regression: PASS");
// ^^^ THOG
