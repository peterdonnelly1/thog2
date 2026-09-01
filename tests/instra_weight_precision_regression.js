// vvv THOG
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const source = fs.readFileSync(path.join(__dirname, "../sheet/local_dashboard_assets/dashboard_weights_group_settings_patch.js"), "utf8");
const normalizers = source.slice(source.indexOf("    const default_common_settings ="), source.indexOf("    const group_settings_for_scope ="));
const apply = source.slice(source.indexOf("    const apply_group_settings = async"), source.indexOf("    const install_group_button ="));
const elements = new Map([
  ["weights_group_precision", {value: "4"}], ["weights_group_scale_mode", {value: "linear"}],
  ["chart_settings_error", {hidden: true, textContent: ""}], ["save_chart_settings", {}],
]);
const store = new Map();
let renders = 0;
const sandbox = {
  console, by_id: id => elements.get(id), weight_group_settings: {}, group_editor_scope: "run:a",
  scale_storage_key: "scales", weight_chart_names: ["q", "k", "v", "o", "up", "down"],
  save_group_store: () => store.set("groups", JSON.stringify(sandbox.weight_group_settings)),
  save_json: (key, value) => store.set(key, JSON.stringify(value)),
  load_json: (key, fallback) => store.has(key) ? JSON.parse(store.get(key)) : fallback,
  chart_settings_form_state: () => ({settings: {line_width: 1.5}, error: null}),
  app: {chart_settings_preview_serial: 0}, clearTimeout() {},
  render_figures: async () => { renders += 1; }, cleanup_group_editor() {},
  render_axis_settings_change() {}, close_chart_settings() {}, update_group_button() {}, show_toast() {},
};
vm.createContext(sandbox);
vm.runInContext(`${normalizers}\nconst common_settings_from_state = normalize_common_settings;\n${apply}\nthis.normalize = normalize_common_settings; this.apply = apply_group_settings;`, sandbox);
(async () => {
  assert.equal(sandbox.normalize({}).inspection_precision, 4);
  assert.equal(sandbox.normalize({inspection_precision: null}).inspection_precision, 4);
  assert.equal(sandbox.normalize({inspection_precision: 0}).inspection_precision, 0);
  assert.equal(sandbox.normalize({inspection_precision: 12}).inspection_precision, 12);
  assert.equal(sandbox.normalize({inspection_precision: 99}).inspection_precision, 4);
  elements.get("weights_group_precision").value = "7";
  await sandbox.apply();
  assert.equal(JSON.parse(store.get("groups"))["run:a"].inspection_precision, 7);
  sandbox.group_editor_scope = "workspace";
  elements.get("weights_group_precision").value = "0";
  await sandbox.apply();
  const saved = JSON.parse(store.get("groups"));
  assert.equal(saved.workspace.inspection_precision, 0);
  assert.equal(saved["run:a"].inspection_precision, 7, "Workspace precision overwrote run precision");
  assert.equal(sandbox.normalize(saved["run:a"]).inspection_precision, 7, "saved precision did not roundtrip");
  for (const invalid of ["", " ", "-1", "13", "1.5", "NaN"]) {
    const before = store.get("groups");
    elements.get("weights_group_precision").value = invalid;
    await sandbox.apply();
    assert.equal(store.get("groups"), before, `invalid precision ${invalid} was saved`);
    assert.match(elements.get("chart_settings_error").textContent, /whole number from 0 to 12/);
  }
  assert.equal(renders, 2, "invalid precision triggered redraw");
  console.log("instra weight inspection precision persistence regression: PASS");
})().catch(error => { console.error(error); process.exitCode = 1; });
// ^^^ THOG
