"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");

const launcher = fs.readFileSync("run_thog2_dashboard.py", "utf8");
const repair = fs.readFileSync(
  "sheet/local_dashboard_assets/dashboard_sep20_integrated_repairs.js",
  "utf8",
);
const metric_groups = fs.readFileSync(
  "sheet/local_dashboard_assets/dashboard_wandb_groups_patch.js",
  "utf8",
);

const active_match = launcher.match(
  /_ACTIVE_EXTRA_ASSET_NAMES\s*=\s*\(([\s\S]*?)\n\)/,
);
assert.ok(active_match, "active runtime asset list is missing");

const active_assets = [...active_match[1].matchAll(/"([^"]+\.js)"/g)]
  .map(match => match[1]);
assert.deepEqual(active_assets, [
  "dashboard_wandb_groups_patch.js",
  "dashboard_group_stability_patch.js",
  "dashboard_sep20_integrated_repairs.js",
  "dashboard_sep21_instra_repairs.js",
]);

for (const obsolete_owner of [
  "dashboard_aug30_enhancements_patch.js",
  "dashboard_instra_further_enhancements_patch.js",
  "dashboard_sep07_fixes_and_enhancements.js",
  "dashboard_weight_step_controls_patch.js",
]) {
  assert.equal(
    active_assets.includes(obsolete_owner),
    false,
    `${obsolete_owner} regained runtime ownership`,
  );
}

assert.doesNotMatch(repair, /body, body \*/);
assert.doesNotMatch(
  repair,
  /observer\.observe\(document\.body,\s*\{childList:true,\s*subtree:true\}\)/,
);
assert.doesNotMatch(repair, /refresh_regular_chart_groups/);
assert.match(repair, /const render_runs_before_sep20 = render_runs/);
assert.match(repair, /ensure_runs_trash_icon\(\)/);
assert.match(metric_groups, /const order_changed = current_nodes\.length !== desired_nodes\.length/);
assert.match(metric_groups, /if \(order_changed\)/);
assert.doesNotMatch(metric_groups, /processing_anchor\.after\(processing_group\)/);

console.log("PASS bounded September runtime owners, stable redraw path and trash restoration");
