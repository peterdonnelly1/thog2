// vvv THOG
"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const source = fs.readFileSync(
  "sheet/local_dashboard_assets/dashboard_wandb_groups_patch.js",
  "utf8",
);

const workspace_runs = [
  {local_run_id: "R1"},
  {local_run_id: "R2"},
];
const app = {
  current_run_id: "R1",
  workspace_mode: false,
  dynamic_chart_figures: {},
  dynamic_chart_metadata: {},
};
const context = {
  console,
  app,
  chart_titles: {},
  window: {
    addEventListener(name, callback) { if (name === "load") callback(); },
    __instra_workspace: {
      active: () => app.workspace_mode,
      visible_runs: () => workspace_runs,
      selection_key: () => workspace_runs.map(run => run.local_run_id).join("|"),
    },
  },
  document: {
    head: {appendChild() {}},
    createElement: () => ({style: {}, textContent: ""}),
    querySelectorAll: () => [],
  },
  by_id: id => id === "charts_scroll" ? {hidden: true} : null,
  run_identifier: run => run.local_run_id,
  prepare_figure: figure => JSON.parse(JSON.stringify(figure)),
  load_json: (_key, fallback) => fallback,
  save_json() {},
  select_run(run_id) { app.current_run_id = String(run_id); },
  setTimeout(callback) { callback(); return 1; },
  setInterval() { return 1; },
  clearInterval() {},
};
context.window.window = context.window;

vm.createContext(context);
vm.runInContext(source, context, {filename: "dashboard_wandb_groups_patch.js"});

const groups = context.window.__thog2_metric_groups;
assert.ok(groups, "metric-group controller did not install");
assert.equal(groups.context_key(), "run:R1");
assert.equal(groups.group_is_collapsed("train"), true);
assert.equal(groups.group_is_collapsed("system"), true);

groups.set_group_collapsed("train", false);
groups.set_group_collapsed("system", false);
assert.equal(groups.group_is_collapsed("train"), false);
assert.equal(groups.group_is_collapsed("system"), false);

app.current_run_id = "R2";
assert.equal(groups.context_key(), "run:R2");
assert.equal(groups.group_is_collapsed("train"), false, "train did not stay open after a run change");
assert.equal(groups.group_is_collapsed("system"), false, "system did not stay open after a run change");

app.workspace_mode = true;
assert.equal(groups.context_key(), "workspace:R1|R2");
assert.equal(groups.group_is_collapsed("train"), true, "train auto-opened on entering Workspace");
assert.equal(groups.group_is_collapsed("system"), true, "system auto-opened on entering Workspace");
groups.set_group_collapsed("train", false);
assert.equal(groups.group_is_collapsed("train"), false, "explicit train opening was not retained");
groups.set_group_collapsed("system", false);
assert.equal(groups.group_is_collapsed("system"), false, "multiple explicit group openings were not retained");

workspace_runs.pop();
assert.equal(groups.context_key(), "workspace:R1");
assert.equal(groups.group_is_collapsed("train"), false, "train did not stay open after Workspace membership changed");
assert.equal(groups.group_is_collapsed("system"), false, "system did not stay open after Workspace membership changed");

app.workspace_mode = false;
app.current_run_id = "R1";
assert.equal(groups.group_is_collapsed("train"), false, "returning to Runs lost train expansion");
assert.equal(groups.group_is_collapsed("system"), false, "returning to Runs lost system expansion");

console.log("instra metric group state regression: PASS");
// ^^^ THOG
