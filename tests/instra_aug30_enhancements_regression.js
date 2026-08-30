// vvv THOG
"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");

const repository_root = path.resolve(__dirname, "..");
const enhancement_source = fs.readFileSync(
  path.join(repository_root, "sheet/local_dashboard_assets/dashboard_aug30_enhancements_patch.js"),
  "utf8",
);
const dashboard_source = fs.readFileSync(
  path.join(repository_root, "sheet/local_dashboard_assets/dashboard.js"),
  "utf8",
);

// Run table: NAME has a persistent drag handle and STEPS is relocated by semantic
// column markers, rather than brittle hard-coded indices.
assert.match(enhancement_source, /thog2_local_run_name_column_width/);
assert.match(enhancement_source, /run-name-column-resizer/);
assert.match(enhancement_source, /data-instra-run-shape-header="preset"/);
assert.match(enhancement_source, /\.step-column/);
assert.match(enhancement_source, /data-instra-steps-cell/);

// Overview: all transient reader state survives the live catalog renderer, the
// requested order is explicit, and each requested panel has one disclosure owner.
for (const token of [
  "overview_summary_panel",
  "overview_config_panel",
  "overview_artifact_outputs",
  "selection_start",
  "scroll_top",
  "restore_overview_state",
  "overview-collapsible",
  "grid.append(summary, config)",
  "Not recorded (this run predates command capture)",
]) {
  assert.ok(enhancement_source.includes(token), `missing Overview contract: ${token}`);
}
assert.match(enhancement_source, /grid-template-columns: minmax\(0, 1fr\) !important/);

// Colour redraws capture the run before the popover can clear app.colour_run_id,
// and Workspace changes invalidate the merged depth payload.
assert.match(dashboard_source, /function queue_current_recolour\(run_id = app\.colour_run_id\)/);
assert.match(dashboard_source, /queue_current_recolour\(app\.colour_run_id\)/);
assert.match(dashboard_source, /__instra_workspace_depth_cache\?\.clear/);

console.log("instra August 30 enhancements regression: PASS");
// ^^^ THOG
