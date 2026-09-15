"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const source = fs.readFileSync("sheet/local_dashboard_assets/dashboard_processing.js", "utf8");

function extracted_function(name) {
  const function_start = source.indexOf(`function ${name}(`);
  assert.notEqual(function_start, -1, `${name} is missing`);
  const async_start = source.lastIndexOf("async ", function_start);
  const start = async_start >= 0 && async_start + 6 === function_start ? async_start : function_start;
  const body_start = source.indexOf("{", start);
  let depth = 0;
  for (let index = body_start; index < source.length; index += 1) {
    if (source[index] === "{") depth += 1;
    if (source[index] === "}") depth -= 1;
    if (depth === 0) return source.slice(start, index + 1);
  }
  throw new Error(`${name} is incomplete`);
}

const sandbox = {
  app: {workspace_mode: true},
  processing_update_timing_cache: new Map(),
  processing_escape: value => String(value),
  run_identifier: run => run.id,
  fetch_calls: 0,
  fetch: async () => {
    sandbox.fetch_calls += 1;
    return {ok: true, json: async () => ({host_update_ms: 100})};
  },
  encodeURIComponent,
  console,
  Map,
  Date,
  JSON,
  Number,
  Object,
  String,
};

for (const name of [
  "processing_update_timing_for_run",
  "processing_update_timing_run_name",
  "processing_update_timing_experiment_label",
  "processing_update_timing_premat_enabled",
  "processing_update_timing_phase_label",
  "processing_update_timing_phase_traces",
  "processing_update_timing_lane_labels",
  "processing_update_timing_match_signature",
  "processing_update_timing_pair_assessment",
]) {
  vm.runInNewContext(`${extracted_function(name)}; this.${name} = ${name};`, sandbox);
}

const nomat = {
  run_id: "n",
  run: {artifact_name: "260914-1909_scruffy_CONTROL_NOMAT___G0_test", created_at: "2026-09-14T09:09:00Z"},
  timing: {
    optimizer_update: 50,
    capture: {
      run_label: "CONTROL_NOMAT",
      premat_mode: "disabled",
      premat: true,
      gradient_accumulation_steps: 6,
      batch_size: 16,
      block_size: 1024,
      match_signature: {n_layer: 16, depth_order: 12},
    },
  },
};
const premat = {
  run_id: "p",
  run: {artifact_name: "260914-1919_scruffy_O_PREMAT___G0_test", created_at: "2026-09-14T09:19:00Z"},
  timing: {
    optimizer_update: 50,
    capture: {
      run_label: "O_PREMAT",
      premat_mode: "enabled",
      premat: true,
      gradient_accumulation_steps: 6,
      batch_size: 16,
      block_size: 1024,
      match_signature: {n_layer: 16, depth_order: 12},
    },
  },
};

assert.equal(sandbox.processing_update_timing_experiment_label(nomat), "CONTROL_NOMAT");
assert.equal(sandbox.processing_update_timing_premat_enabled(nomat), false);
const legacy_nomat = structuredClone(nomat);
delete legacy_nomat.timing.capture.premat_mode;
legacy_nomat.run.artifact_name = "260914-1909_scruffy_NOMAT___G0_test__PM__D_F";
assert.equal(sandbox.processing_update_timing_premat_enabled(legacy_nomat), false);
assert.deepEqual(
  Array.from(sandbox.processing_update_timing_lane_labels([nomat, premat])),
  ["CONTROL_NOMAT", "O_PREMAT"],
);
assert.equal(sandbox.processing_update_timing_pair_assessment([nomat, premat]).level, "ok");
assert.equal(sandbox.processing_update_timing_pair_assessment([premat, nomat]).level, "warning");

const mismatched = structuredClone(premat);
mismatched.timing.capture.match_signature.depth_order = 8;
assert.equal(sandbox.processing_update_timing_pair_assessment([nomat, mismatched]).level, "error");

nomat.timing.timeline = [
  {phase: "setup", micro_step: 1, host_start_ms: 0, host_end_ms: 2},
  {phase: "gpu_completion_drain", micro_step: null, host_start_ms: 98, host_end_ms: 100},
];
premat.timing.timeline = [
  {phase: "setup", micro_step: 1, host_start_ms: 0, host_end_ms: 3},
  {phase: "gpu_completion_drain", micro_step: null, host_start_ms: 97, host_end_ms: 100},
];
const phase_traces = sandbox.processing_update_timing_phase_traces([nomat, premat]);
assert.ok(phase_traces.every(trace => trace.type === "bar" && trace.orientation === "h"));
assert.deepEqual(Array.from(phase_traces.find(trace => trace.name === "GPU completion / drain").x), [2, 3]);

(async () => {
  const run = {id: "cache", revision: [1]};
  await sandbox.processing_update_timing_for_run(run);
  await sandbox.processing_update_timing_for_run(run);
  assert.equal(sandbox.fetch_calls, 1, "unchanged revision refetched timing JSON");
  run.revision = [2];
  await sandbox.processing_update_timing_for_run(run);
  assert.equal(sandbox.fetch_calls, 2, "changed revision retained stale timing JSON");
  console.log("PASS complete-update pair labels, ordering, matching, PREMAT parsing and cache invalidation");
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
