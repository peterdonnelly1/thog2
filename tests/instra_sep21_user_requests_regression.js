"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");

const repair = fs.readFileSync(
  "sheet/local_dashboard_assets/dashboard_sep21_instra_repairs.js",
  "utf8",
);
const table = fs.readFileSync(
  "sheet/local_dashboard_assets/dashboard_runs_table_restore.js",
  "utf8",
);
const operations = fs.readFileSync(
  "sheet/local_dashboard_assets/dashboard_processing_operations_final.js",
  "utf8",
);
const launcher = fs.readFileSync("run_thog2_dashboard.py", "utf8");

assert.match(launcher, /"dashboard_sep21_instra_repairs\.js"/);

// 1: the last runtime owner guards both expensive immutable timing renders and
// unchanged Runs-table rebuilds.
assert.match(repair, /if \(!force && fingerprint === last_timing_fingerprint\) return Promise\.resolve\(\)/);
assert.match(repair, /if \(signature === last_runs_signature\) return/);
assert.match(repair, /let timing_in_flight = null/);

// 2: opening an NSYS eye always forces a fresh pairing-capable Processing read.
assert.match(repair, /if \(nsys\)[\s\S]*if \(!is_visible\(run_id\)\) return/);
assert.match(repair, /processing_refresh\(true\)/);

// 3-5b and 20: stable title-bar geometry; only the train group is visible;
// throughput titles, axis and right-side legend are identical.
assert.match(repair, /\.charts-toolbar \{ height:62px !important/);
assert.match(repair, /#training_chart_group \{ display:none !important/);
assert.match(repair, /grid\.appendChild\(card\)/);
assert.match(repair, /title\.textContent = "tokens throughput"/);
assert.match(repair, /processing_view\.throughput_axis_mode = "step"/);
assert.match(repair, /"xaxis\.title\.text":"steps"/);
assert.match(repair, /"legend\.x":1, "legend\.xanchor":"right"/);
assert.match(repair, /data-metric-group="train"[\s\S]*flex-direction:column !important/);
assert.match(repair, /instra-train-loss-card \{ order:0/);
assert.match(repair, /#training_throughput_card \{ order:1/);

// The Processing group in Runs view requires real capture/timing/resource data;
// ordinary train throughput alone remains in the train group.
assert.match(repair, /app\.workspace_mode !== true/);
assert.match(repair, /processing_view\.trace_available[\s\S]*processing_view\.timing_available/);
assert.match(repair, /group\.hidden = !\(processing_view\.charts_tab_visible && capture_available\)/);

// 6-8: user-selectable columns, a PREMAT matrix column with fixed character
// positions, and a modestly enlarged darker trash glyph.
assert.match(repair, /button\.textContent = "Columns"/);
assert.match(repair, /thog2_local_hidden_run_columns_v1/);
assert.match(table, /premat: \{label: "premat"/);
assert.match(table, /\[1, 2, 3, 4\]\.map\(matrix => selected\.has\(matrix\) \? String\(matrix\) : " "\)\.join\(" "\)/);
assert.match(table, /white-space:pre !important/);
assert.match(repair, /width:18px !important; height:18px !important/);
assert.match(repair, /#7B1F24/);

// 9: both ordinary and maximized compatibility bars are one half of the
// immediately preceding 0.46/0.72 geometry.
assert.match(repair, /width:maximized \? 0\.36 : 0\.23/);
assert.match(repair, /"line\.width":maximized \? 17 : 8/);

// 10-13: square palette patches, pastel-only automatic colours, editable RGB
// fields in both pickers and no reset/default colour action.
assert.match(repair, /aspect-ratio:1 \/ 1 !important/);
assert.match(repair, /default_palette\.splice\(0, default_palette\.length, \.\.\.light_run_palette\)/);
assert.match(repair, /input\.readOnly = false/);
assert.match(repair, /by_id\("reset_colour"\)\?\.remove\(\)/);
assert.match(operations, /processing_operations_colour_r/);
assert.match(operations, /processing_operations_colour_b/);
assert.doesNotMatch(operations, /processing-operations-colour-reset/);

// 14-19: full-range, bounded-height timeline; ms, captured step and exact CLI
// controls are all explicit in its title-bar help text.
assert.match(repair, /configure_host_timeline\(true\)/);
assert.match(repair, /--premat_instra__full_step_timing_capture_and_chart enable/);
assert.match(repair, /--premat_instra__full_step_timing_capture_and_chart_capture step/);
assert.match(repair, /captured optimizer step/);
assert.match(repair, /elapsed host time from update entry \(ms\)/);
assert.match(repair, /"xaxis\.ticksuffix":" ms"/);
assert.match(repair, /--instra-host-timeline-height/);

// 21: Files uses the same trash SVG and is moved immediately after the source
// tabs, toward the left of the header.
assert.match(repair, /button\.innerHTML = trash_svg/);
assert.match(repair, /tabs\.insertAdjacentElement\("afterend", button\)/);

console.log("PASS September 21 Instra user-request contracts");
