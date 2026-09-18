"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const pair_source = fs.readFileSync("sheet/local_dashboard_assets/dashboard_processing_pair_state_final.js", "utf8");
const eye_source = fs.readFileSync("sheet/local_dashboard_assets/dashboard_processing_nsys_eye_select.js", "utf8");

function run(id, artifact_name, created_at) {
  return {dashboard_run_id:id, artifact_name, created_at, run_state:"finished"};
}

function make_sandbox({
  runs,
  current_run_id = "ordinary",
  visible_run_ids = [],
  stored_pairs = [],
  catalog_ready = false,
  processing_responses = {},
  throughput_responses = {},
}) {
  const listeners = new Map();
  const storage = new Map([
    ["thog2_processing_pairs_v2", JSON.stringify(stored_pairs)],
    ["thog2_processing_auto_opened_run_ids", "[]"],
    ["thog2_processing_unmatched_nsys_run_ids", "[]"],
  ]);
  const visible = new Set(visible_run_ids);
  const runs_body = {
    dataset:{},
    addEventListener(type, listener) {
      const key = `runs_body:${type}`;
      const registered = listeners.get(key) || [];
      registered.push(listener);
      listeners.set(key, registered);
    },
  };
  const sandbox = {
    app:{
      runs,
      current_run_id,
      visibility:Object.fromEntries(runs.map(item => [item.dashboard_run_id, visible.has(item.dashboard_run_id)])),
      instra_catalog_ready:catalog_ready,
      instra_catalog_generation:1,
    },
    processing_view:{run_id:null, revision:null, companion_enriched_payload:null, render_request_run_id:null, render_request_epoch:0},
    fetch_calls:0,
    fetch_urls:[],
    throughput_renders:[],
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
    processing_render_throughput:async function(payload) {
      sandbox.throughput_renders.push(payload);
    },
    processing_refresh:async function() {},
    select_run(run_id) {
      sandbox.app.current_run_id = String(run_id);
      return run_id;
    },
    async fetch_json(url) {
      sandbox.fetch_calls += 1;
      sandbox.fetch_urls.push(String(url));
      const run_id = String(sandbox.app.current_run_id || "");
      if (String(url).startsWith("/api/processing-throughput")) {
        return throughput_responses[run_id] || {throughput:[]};
      }
      const response = processing_responses[run_id] || {available:false, trace_available:false};
      return typeof response === "function" ? response(String(url)) : response;
    },
    by_id(id) { return id === "runs_body" ? runs_body : null; },
    document:{head:{appendChild() {}}, getElementById() { return null; }, createElement() { return {id:"", textContent:""}; }},
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

(async () => {
  const ordinary = run("ordinary", "260918-1000_scruffy_NOMAT", "2026-09-18T10:00:00Z");
  const nsys_a = run("nsys_a", "260918-1010_scruffy_PAIR_A_NSYS_PREMAT___MATCH_A", "2026-09-18T10:10:00Z");
  const ncu_a = run("ncu_a", "260918-1011_scruffy_PAIR_A_NCU_PREMAT___MATCH_A", "2026-09-18T10:11:00Z");
  const nsys_b = run("nsys_b", "260918-1020_scruffy_PAIR_B_NSYS_PREMAT___MATCH_B", "2026-09-18T10:20:00Z");
  const ncu_b = run("ncu_b", "260918-1021_scruffy_PAIR_B_NCU_PREMAT___MATCH_B", "2026-09-18T10:21:00Z");
  const runs = [ordinary, nsys_a, ncu_a, nsys_b, ncu_b];

  const sandbox = make_sandbox({runs});
  vm.runInNewContext(pair_source, sandbox);
  const hooks = sandbox.window.processing_pair_state_test_hooks;
  assert.equal(hooks.add_pair("nsys_a", "ncu_a"), true);
  assert.equal(hooks.add_pair("nsys_b", "ncu_b"), true);
  assert.equal(Object.keys(sandbox.app.processing_pairs).length, 2, "two profiler pairs were not retained");
  assert.notEqual(sandbox.app.processing_pairs.nsys_a.colour, sandbox.app.processing_pairs.nsys_b.colour, "separate pairs received the same eye colour");
  assert.deepEqual([...sandbox.app.processing_paired_run_ids].sort(), ["ncu_a", "ncu_b", "nsys_a", "nsys_b"]);

  sandbox.select_run("ordinary", {manual:true});
  await settle();
  sandbox.select_run("ncu_a", {manual:true});
  await settle();
  assert.equal(Object.keys(sandbox.app.processing_pairs).length, 2, "ordinary run navigation broke a profiler pair");

  assert.equal(hooks.unpair_run("ncu_a"), true);
  assert.equal(sandbox.app.visibility.nsys_a, false);
  assert.equal(sandbox.app.visibility.ncu_a, false);
  assert.equal(sandbox.app.visibility.nsys_b, true, "unpairing Pair A hid Pair B's NSYS run");
  assert.equal(sandbox.app.visibility.ncu_b, true, "unpairing Pair A hid Pair B's NCU run");
  assert.deepEqual([...sandbox.app.processing_paired_run_ids].sort(), ["ncu_b", "nsys_b"]);

  const restored = make_sandbox({
    runs,
    stored_pairs:[
      {nsys_run_id:"nsys_a", ncu_run_id:"ncu_a", colour:"#24527A"},
      {nsys_run_id:"nsys_b", ncu_run_id:"ncu_b", colour:"#6B3F8C"},
    ],
  });
  vm.runInNewContext(pair_source, restored);
  restored.window.processing_pair_state_test_hooks.restore_persisted_pairs();
  for (const id of ["nsys_a", "ncu_a", "nsys_b", "ncu_b"]) assert.equal(restored.app.visibility[id], true);

  const unmatched = make_sandbox({
    runs,
    current_run_id:"nsys_a",
    visible_run_ids:["nsys_a"],
    processing_responses:{nsys_a:{available:true, trace_available:true, revision:"unmatched", data:{}}},
  });
  vm.runInNewContext(pair_source, unmatched);
  await unmatched.processing_refresh(true);
  assert.equal(unmatched.app.processing_unmatched_nsys_run_ids.has("nsys_a"), true, "NSYS without a qualifying NCU was not marked unmatched");

  const stable_pair = make_sandbox({
    runs,
    current_run_id:"nsys_a",
    visible_run_ids:["nsys_a", "ncu_a"],
    stored_pairs:[{nsys_run_id:"nsys_a", ncu_run_id:"ncu_a", colour:"#24527A"}],
    processing_responses:{
      nsys_a:{available:true, trace_available:true, revision:"changed-choice", data:{premat_compatibility_source:{nsys_dashboard_run_id:"nsys_a", dashboard_run_id:"ncu_b"}}},
    },
  });
  vm.runInNewContext(pair_source, stable_pair);
  await stable_pair.processing_refresh(true);
  assert.equal(stable_pair.app.processing_pairs.nsys_a.ncu_run_id, "ncu_a", "refreshing a pair silently changed its companion");
  assert.match(
    stable_pair.fetch_urls.find(url => url.startsWith("/api/processing?")),
    /preferred_ncu=ncu_a/,
    "an established pair did not request its persisted NCU companion",
  );

  const first_claim_wins = make_sandbox({
    runs,
    current_run_id:"nsys_b",
    visible_run_ids:["nsys_a", "ncu_a", "nsys_b"],
    stored_pairs:[{nsys_run_id:"nsys_a", ncu_run_id:"ncu_a", colour:"#24527A"}],
    processing_responses:{
      nsys_b:url => ({
        available:true,
        trace_available:true,
        revision:"fallback",
        data:{premat_compatibility_source:{
          nsys_dashboard_run_id:"nsys_b",
          dashboard_run_id:url.includes("exclude_ncu=ncu_a") ? "ncu_b" : "ncu_a",
        }},
      }),
    },
  });
  vm.runInNewContext(pair_source, first_claim_wins);
  await first_claim_wins.processing_refresh(true);
  assert.equal(first_claim_wins.app.processing_pairs.nsys_a.ncu_run_id, "ncu_a", "the first pair lost its claimed NCU");
  assert.equal(first_claim_wins.app.processing_pairs.nsys_b.ncu_run_id, "ncu_b", "the next NSYS did not claim the next eligible NCU");
  assert.match(
    first_claim_wins.fetch_urls.find(url => url.startsWith("/api/processing?")),
    /exclude_ncu=ncu_a/,
    "claimed NCU identifiers were not sent to companion selection",
  );

  const throughput_only = make_sandbox({
    runs,
    current_run_id:"ordinary",
    visible_run_ids:["ordinary"],
    throughput_responses:{
      ordinary:{throughput:[{optimizer_update:5, tokens_per_second:12345}]},
    },
  });
  vm.runInNewContext(pair_source, throughput_only);
  await throughput_only.processing_refresh(true);
  assert.equal(throughput_only.throughput_renders.length, 1, "a run without Processing evidence did not render retained Training throughput");
  assert.equal(throughput_only.throughput_renders[0].throughput[0].tokens_per_second, 12345);

  const eye_selection = make_sandbox({
    runs,
    processing_responses:{
      nsys_a:{available:true, trace_available:true, revision:"eye", data:{premat_compatibility_source:{nsys_dashboard_run_id:"nsys_a", dashboard_run_id:"ncu_a"}}},
    },
  });
  vm.runInNewContext(pair_source, eye_selection);
  vm.runInNewContext(eye_source, eye_selection);
  eye_selection.select_run("nsys_a", {manual:true});
  await settle();
  assert.equal(eye_selection.fetch_calls, 1, "stacked selection wrappers issued duplicate Processing requests");

  console.log("PASS sticky first-claim pairs, fallback companions, throughput-only runs and explicit unpairing");
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
