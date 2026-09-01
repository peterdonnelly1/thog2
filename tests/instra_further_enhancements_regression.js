// vvv THOG
"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");

const repository_root = path.resolve(__dirname, "..");
const further = fs.readFileSync(
  path.join(repository_root, "sheet/local_dashboard_assets/dashboard_instra_further_enhancements_patch.js"),
  "utf8",
);
const coupling = fs.readFileSync(
  path.join(repository_root, "sheet/local_dashboard_assets/dashboard_weight_range_interaction_final_patch.js"),
  "utf8",
);
const base_dashboard = fs.readFileSync(
  path.join(repository_root, "sheet/local_dashboard_assets/dashboard.js"),
  "utf8",
);

const palette_match = further.match(
  /const additional_palette = Object\.freeze\((\[[\s\S]*?\])\);/,
);
assert.ok(palette_match, "the additional colour palette is not statically inspectable");
const palette = JSON.parse(palette_match[1].replace(/,\s*\]$/, "]"));
assert.equal(palette.length, 64, "the colour picker did not gain exactly 64 colours");
assert.equal(new Set(palette).size, 64, "the additional palette contains duplicates");
assert.deepEqual(palette.slice(-2), ["#000000", "#FFFFFF"]);
const base_palette_match = base_dashboard.match(/const default_palette = (\[[\s\S]*?\]);/);
assert.ok(base_palette_match, "the original palette is not statically inspectable");
const base_palette = JSON.parse(base_palette_match[1].replace(/,\s*\]$/, "]"));
const combined_palette = [...base_palette, ...palette].map(value => value.toUpperCase());
assert.equal(combined_palette.length, 128);
assert.equal(new Set(combined_palette).size, 128, "the full 128-colour palette contains duplicates");

assert.match(further, /command\.label\.textContent = "Runstring"/);
assert.match(further, /local_state_badge\(run\)/);
assert.match(further, /label\.textContent = "W&B ID"/);
assert.match(further, /\/api\/run-notes\?run=/);
assert.match(further, /text_node\.nodeValue = "RUN NAME "/);
assert.match(further, /overlapping steps \$\{available\.minimum\}–\$\{available\.maximum\}/);
assert.match(further, /step_one\.textContent = "step 1"/);
assert.match(further, /chart_titles\.mlp_up = "MLP - expansion"/);
assert.match(further, /chart_titles\.mlp_down = "MLP - contraction"/);
assert.match(further, /stability\.mode\?\.\(\) === "latest" \? 2\.4 : 1\.0/);
assert.match(further, /schedule_maximized_restore\(prior_maximized, run_id\)/);
assert.match(further, /live_payload_is_stale\(\)/);
assert.doesNotMatch(further, /const live_refresh_timer = setInterval/);
assert.match(base_dashboard, /setInterval\(\(\) => refresh_current_run\(\), 2000\)/);
assert.match(further, /\.state-badge\.finished[\s\S]*?color: #2f7d32/);
assert.match(further, /explicit-trajectory-modes[\s\S]*?display: none !important/);

assert.doesNotMatch(coupling, /localStorage|save_json|load_json/);
assert.match(coupling, /const zero_pair = \{model_feature: 0, intermediate_feature: 0\}/);
assert.match(coupling, /const pairs_by_run = new Map\(\)/);
assert.match(coupling, /sets\.every\(unit => unit\.has\(key\)\)/);
assert.match(coupling, /if \(app\.workspace_mode === true\)/);

console.log("instra further enhancements regression: PASS");
// ^^^ THOG
