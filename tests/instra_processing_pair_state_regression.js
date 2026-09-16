"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const pair_source = fs.readFileSync(
  "sheet/local_dashboard_assets/dashboard_processing_pair_state_final.js",
  "utf8",
);
const eye_source = fs.readFileSync(
  "sheet/local_dashboard_assets/dashboard_processing_nsys_eye_select.js",
  "utf8",
);
const stability_source = fs.readFileSync(
  "sheet/local_dashboard_assets/dashboard_processing_sep16_stability_polish.js",
  "utf8",
);

function run(id, artifact_name, created_at) {
  return {dashboard_run_id:id, artifact_name, created_at, run_state:"finished"};
}

function make_sandbox({
  runs,
  current_run_id = "ordinary",
  visible_run_ids = [],
  stored_auto_run_ids = [],
  catalog_ready = true,
  companion_run_id = "ncu",
}) {
  const listeners = new Map();
  const storage = new Map([
    ["thog2_processing_auto_opened_run_ids", JSON.stringify(stored_auto_run_ids)],
  ]);
  const visible = new Set(visible_run_ids);
  const visibility = Object.fromEntries(runs.map(item => [item.dashboard_run_id, visible.has(item.dashboard_run_id)]));
  const runs_body = {
    dataset:{},
    addEventListener(type, listener) { listeners.set(`runs_body:${type}`, listener); },
  };
  const sandbox = {
    app:{
      runs,
      current_run_id,
      visibility,
      instra_catalog_ready:catalog_ready,
    },
    processing_view:{
      run_id:null,
      revision:null,
      companion_enriched_payload:null,
      render_request_run_id:null,
      render_request_epoch:0,
    },
    fetch_calls:0,
    render_runs_calls:0,
    listeners,
    console,
    Date,
    JSON,
    Map,
    Number,
    Object,
    Promise,
    Set,
    String,
    CustomEvent:class CustomEvent {
      constructor(type, options = {}) { this.type = type; this.detail = options.detail; }
    },
    queueMicrotask,
    setTimeout,
    clearTimeout,
    encodeURIComponent,
    load_json(key, fallback) {
      const raw = storage.get(key);
      return raw === undefined ? fallback : JSON.parse(raw);
    },
    save_json(key, value) { storage.set(key, JSON.stringify(value)); },
    run_identifier(item) { return item.dashboard_run_id; },
    is_visible(run_id) { return sandbox.app.visibility[run_id] !== false; },
    render_runs() { sandbox.render_runs_calls += 1; },
    processing_sync_visibility() {},
    processing_current_run() { return sandbox.app.current_run_id; },
    processing_render:async function() {},
    processing_render_timeline:async function() {},
    processing_render_update_timing:async function() {},
    processing_update_timing_ordered_runs() { return []; },
    processing_refresh:async function() {},
    is_active_run_state() { return false; },
    chart_titles:{},
    Plotly:{
      relayout:async function() {},
      restyle:async function() {},
    },
    select_run(run_id) {
      sandbox.app.current_run_id = String(run_id);
      return run_id;
    },
    async fetch_json() {
      sandbox.fetch_calls += 1;
      return {
        available:true,
        trace_available:true,
        revision:`revision-${sandbox.app.current_run_id}`,
        data:{
          premat_compatibility_source:{dashboard_run_id:companion_run_id},
        },
      };
    },
    by_id(id) {
      if (id === "runs_body") return runs_body;
      return null;
    },
    document:{
      visibilityState:"visible",
      head:{appendChild() {}},
      getElementById() { return null; },
      createElement() { return {id:"", textContent:""}; },
      querySelectorAll() { return []; },
    },
    window:{
      location:{pathname:`/runs/${current_run_id}`},
      addEventListener(type, listener) {
        const registered = listeners.get(type) || [];
        registered.push(listener);
        listeners.set(type, registered);
      },
      dispatchEvent(event) {
        for (const listener of listeners.get(event.type) || []) listener(event);
      },
      setInterval() { return 1; },
      clearInterval() {},
    },
  };
  sandbox.storage = storage;
  return sandbox;
}

async function settle() {
  await Promise.resolve();
  await Promise.resolve();
  await new Promise(resolve => setImmediate(resolve));
}

async function install_pair_state(options) {
  const sandbox = make_sandbox(options);
  vm.runInNewContext(pair_source, sandbox);
  await settle();
  return sandbox;
}

(async () => {
  const ordinary = run("ordinary", "260916-0915_scruffy_PREMAT_SMOKES_THOG", "2026-09-16T09:15:00Z");
  const nsys = run("nsys", "260916-1405_scruffy_NSYS_PREMAT___MATCH", "2026-09-16T14:05:00Z");
  const newer_nsys = run("newer_nsys", "260916-1612_scruffy_NSYS_PREMAT___MATCH", "2026-09-16T16:12:00Z");
  const ncu = run("ncu", "260916-1807_scruffy_NCU_PREMAT___MATCH", "2026-09-16T18:07:00Z");

  const restored_pair = await install_pair_state({
    runs:[ordinary, nsys, ncu],
    visible_run_ids:[nsys.dashboard_run_id, ncu.dashboard_run_id],
  });
  assert.equal(restored_pair.app.current_run_id, "nsys", "a restored companion eye vetoed its NSYS source");
  assert.equal(restored_pair.fetch_calls, 1, "startup pairing should issue one Processing request");
  assert.deepEqual(
    JSON.parse(restored_pair.storage.get("thog2_processing_auto_opened_run_ids")),
    ["ncu"],
    "legacy restored companion ownership was not migrated",
  );
  for (const listener of restored_pair.listeners.get("load") || []) listener();
  await new Promise(resolve => setTimeout(resolve, 0));
  await settle();
  assert.equal(restored_pair.fetch_calls, 1, "duplicate startup signals repeated pairing work");

  const ncu_only = await install_pair_state({
    runs:[ordinary, nsys, ncu],
    visible_run_ids:[ncu.dashboard_run_id],
  });
  assert.equal(ncu_only.app.current_run_id, "ordinary", "an NCU-only eye must not promote an NSYS run");
  assert.equal(ncu_only.fetch_calls, 0);

  const orphaned_ncu = await install_pair_state({
    runs:[ordinary, nsys, ncu],
    visible_run_ids:[ncu.dashboard_run_id],
    stored_auto_run_ids:[ncu.dashboard_run_id],
  });
  assert.equal(orphaned_ncu.app.visibility.ncu, false, "an orphaned auto-opened NCU eye survived reload");
  assert.deepEqual(JSON.parse(orphaned_ncu.storage.get("thog2_processing_auto_opened_run_ids")), []);

  const latest_nsys = await install_pair_state({
    runs:[ordinary, nsys, newer_nsys, ncu],
    visible_run_ids:[nsys.dashboard_run_id, newer_nsys.dashboard_run_id],
  });
  assert.equal(latest_nsys.app.current_run_id, "newer_nsys", "latest restored NSYS was not selected");
  assert.equal(latest_nsys.fetch_calls, 1);

  const persisted_pair = await install_pair_state({
    runs:[ordinary, nsys, ncu],
    visible_run_ids:[nsys.dashboard_run_id, ncu.dashboard_run_id],
    stored_auto_run_ids:[ncu.dashboard_run_id],
  });
  assert.equal(persisted_pair.app.visibility.ncu, true, "persisted companion was not reopened after reconciliation");
  assert.deepEqual(JSON.parse(persisted_pair.storage.get("thog2_processing_auto_opened_run_ids")), ["ncu"]);

  const one_request = make_sandbox({
    runs:[ordinary, nsys, ncu],
    visible_run_ids:[],
    catalog_ready:false,
  });
  vm.runInNewContext(pair_source, one_request);
  vm.runInNewContext(eye_source, one_request);
  one_request.select_run("nsys", {manual:true});
  await settle();
  assert.equal(one_request.fetch_calls, 1, "stacked NSYS selection wrappers issued duplicate Processing requests");

  const coalesced_force = make_sandbox({
    runs:[ordinary, nsys, ncu],
    current_run_id:"nsys",
    visible_run_ids:[nsys.dashboard_run_id],
    catalog_ready:false,
  });
  let release_fetch;
  coalesced_force.fetch_json = async function() {
    coalesced_force.fetch_calls += 1;
    await new Promise(resolve => { release_fetch = resolve; });
    return {
      available:true,
      trace_available:true,
      revision:"coalesced",
      data:{premat_compatibility_source:{dashboard_run_id:"ncu"}},
    };
  };
  vm.runInNewContext(pair_source, coalesced_force);
  const first_refresh = coalesced_force.processing_refresh(true);
  const second_refresh = coalesced_force.processing_refresh(true);
  await Promise.resolve();
  release_fetch();
  await Promise.all([first_refresh, second_refresh]);
  assert.equal(coalesced_force.fetch_calls, 1, "identical forced refreshes were not coalesced");

  const single_pair_owner = make_sandbox({
    runs:[ordinary, nsys, ncu],
    current_run_id:"nsys",
    visible_run_ids:[nsys.dashboard_run_id],
    catalog_ready:false,
  });
  vm.runInNewContext(stability_source, single_pair_owner);
  vm.runInNewContext(pair_source, single_pair_owner);
  await single_pair_owner.processing_refresh(true);
  assert.equal(single_pair_owner.render_runs_calls, 1, "legacy and final pair owners both rendered the Runs table");

  console.log("PASS startup profiler restoration, ownership migration and single-request NSYS selection");
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
