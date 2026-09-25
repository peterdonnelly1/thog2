// vvv THOG
"use strict";

(function install_processing_pair_state_final() {
  const pairs_storage_key = "thog2_processing_pairs_v2";
  const auto_opened_storage_key = "thog2_processing_auto_opened_run_ids";
  const unmatched_storage_key = "thog2_processing_unmatched_nsys_run_ids";
  const pair_palette = Object.freeze([
    "#0057B8", "#A000C8", "#007A3D", "#C45100",
    "#B0003A", "#006B75", "#6A4C00", "#7A2E00",
    "#0047AB", "#7B1FA2", "#008060", "#C00000",
  ]);

  function stored_string_set(key) {
    const stored = load_json(key, []);
    return new Set(
      (Array.isArray(stored) ? stored : [])
        .map(value => String(value || ""))
        .filter(Boolean),
    );
  }

  function normalise_pairs(value) {
    const source = Array.isArray(value) ? value : Object.values(value || {});
    const pairs = {};
    for (const candidate of source) {
      const nsys_run_id = String(candidate?.nsys_run_id || "");
      const ncu_run_id = String(candidate?.ncu_run_id || "");
      if (!nsys_run_id || !ncu_run_id || nsys_run_id === ncu_run_id) continue;
      pairs[nsys_run_id] = {
        nsys_run_id,
        ncu_run_id,
        colour:String(candidate?.colour || ""),
      };
    }
    return pairs;
  }

  app.processing_pairs = normalise_pairs(load_json(pairs_storage_key, []));
  app.processing_auto_opened_run_ids = app.processing_auto_opened_run_ids instanceof Set
    ? app.processing_auto_opened_run_ids
    : stored_string_set(auto_opened_storage_key);
  app.processing_unmatched_nsys_run_ids = stored_string_set(unmatched_storage_key);
  app.processing_paired_run_ids = new Set();
  app.processing_pair_roles = {};
  app.processing_pair_colours = {};
  app.processing_pair_state_final_installed = true;

  let navigation_epoch = 0;
  let refresh_serial = 0;
  let observed_current_run_id = String(app.current_run_id || "");
  let processing_refresh_promise = null;
  let processing_refresh_force_queued = false;
  let processing_refresh_in_flight_force = false;
  let processing_refresh_in_flight_run_id = "";
  let startup_reconciled_generation = -1;
  const pair_probes_in_flight = new Map();

  function run_for_id(run_id) {
    return (app.runs || []).find(candidate => String(run_identifier(candidate)) === String(run_id)) || null;
  }

  function is_nsys_run_id(run_id) {
    const run = run_for_id(run_id);
    const artifact = String(run?.artifact_name || run?.run_name || "").toUpperCase();
    return artifact.includes("_NSYS_") || artifact.includes("NSYS_PREMAT");
  }

  function is_ncu_run_id(run_id) {
    const run = run_for_id(run_id);
    const artifact = String(run?.artifact_name || run?.run_name || "").toUpperCase();
    return artifact.includes("_NCU_") || artifact.includes("NCU_PREMAT");
  }

  function run_timestamp(run_id) {
    const run = run_for_id(run_id);
    const created = Date.parse(run?.created_at || "");
    if (Number.isFinite(created)) return created;
    const artifact = String(run?.artifact_name || run?.run_name || "");
    const match = /^(\d{6})-(\d{4})/.exec(artifact);
    if (!match) return 0;
    return Date.UTC(
      2000 + Number(match[1].slice(0, 2)),
      Number(match[1].slice(2, 4)) - 1,
      Number(match[1].slice(4, 6)),
      Number(match[2].slice(0, 2)),
      Number(match[2].slice(2, 4)),
    );
  }

  function save_pairs() {
    save_json(pairs_storage_key, Object.values(app.processing_pairs || {}));
  }

  function save_auto_opened_run_ids() {
    save_json(auto_opened_storage_key, [...(app.processing_auto_opened_run_ids || [])]);
  }

  function save_unmatched_run_ids() {
    save_json(unmatched_storage_key, [...(app.processing_unmatched_nsys_run_ids || [])]);
  }

  function rebuild_pair_indexes() {
    const paired = new Set();
    const roles = {};
    const colours = {};
    for (const pair of Object.values(app.processing_pairs || {})) {
      paired.add(pair.nsys_run_id);
      paired.add(pair.ncu_run_id);
      roles[pair.nsys_run_id] = "NSYS source · paired Processing evidence; close either eye to unpair";
      roles[pair.ncu_run_id] = "NCU companion · paired Processing evidence; close either eye to unpair";
      colours[pair.nsys_run_id] = pair.colour;
      colours[pair.ncu_run_id] = pair.colour;
    }
    app.processing_paired_run_ids = paired;
    app.processing_pair_roles = roles;
    app.processing_pair_colours = colours;
  }

  function next_pair_colour() {
    const used = new Set(Object.values(app.processing_pairs || {}).map(pair => pair.colour));
    return pair_palette.find(colour => !used.has(colour))
      || pair_palette[Object.keys(app.processing_pairs || {}).length % pair_palette.length];
  }

  function pair_for_run(run_id) {
    const id = String(run_id || "");
    return Object.values(app.processing_pairs || {}).find(
      pair => pair.nsys_run_id === id || pair.ncu_run_id === id,
    ) || null;
  }

  function set_unmatched(run_id, unmatched) {
    const id = String(run_id || "");
    if (!id) return false;
    const before = app.processing_unmatched_nsys_run_ids.has(id);
    if (unmatched) app.processing_unmatched_nsys_run_ids.add(id);
    else app.processing_unmatched_nsys_run_ids.delete(id);
    if (before === unmatched) return false;
    save_unmatched_run_ids();
    return true;
  }

  function unpair_run(run_id, {close_both = true, render = true} = {}) {
    const pair = pair_for_run(run_id);
    if (!pair) return false;
    delete app.processing_pairs[pair.nsys_run_id];
    let visibility_changed = false;
    if (close_both) {
      for (const id of [pair.nsys_run_id, pair.ncu_run_id]) {
        if (is_visible(id)) {
          app.visibility[id] = false;
          visibility_changed = true;
        }
      }
    }
    app.processing_auto_opened_run_ids.delete(pair.ncu_run_id);
    set_unmatched(pair.nsys_run_id, false);
    rebuild_pair_indexes();
    save_pairs();
    save_auto_opened_run_ids();
    if (visibility_changed) save_json("thog2_local_run_visibility", app.visibility);
    if (render) render_runs();
    return true;
  }

  function add_pair(nsys_run_id, ncu_run_id, {render = true} = {}) {
    const nsys_id = String(nsys_run_id || "");
    const ncu_id = String(ncu_run_id || "");
    if (!nsys_id || !ncu_id || !is_nsys_run_id(nsys_id) || !is_ncu_run_id(ncu_id)) return false;

    const existing = pair_for_run(nsys_id);
    if (existing) return existing.nsys_run_id === nsys_id && existing.ncu_run_id === ncu_id;
    if (pair_for_run(ncu_id)) {
      set_unmatched(nsys_id, true);
      if (render) render_runs();
      return false;
    }

    app.processing_pairs[nsys_id] = {
      nsys_run_id:nsys_id,
      ncu_run_id:ncu_id,
      colour:next_pair_colour(),
    };
    let visibility_changed = false;
    for (const id of [nsys_id, ncu_id]) {
      if (!is_visible(id)) {
        app.visibility[id] = true;
        visibility_changed = true;
      }
    }
    app.processing_auto_opened_run_ids.add(ncu_id);
    set_unmatched(nsys_id, false);
    rebuild_pair_indexes();
    save_pairs();
    save_auto_opened_run_ids();
    if (visibility_changed) save_json("thog2_local_run_visibility", app.visibility);
    if (render) render_runs();
    return true;
  }

  rebuild_pair_indexes();

  function begin_navigation(next_run_id) {
    navigation_epoch += 1;
    observed_current_run_id = String(next_run_id || "");
    processing_view.run_id = null;
    processing_view.revision = null;
    processing_view.companion_enriched_payload = null;
    processing_view.render_request_run_id = null;
    processing_view.render_request_epoch = navigation_epoch;
    processing_view.training_throughput_available = false;
    if (typeof processing_sync_visibility === "function") processing_sync_visibility();
  }

  function ensure_navigation_matches_current() {
    const current = String(app.current_run_id || "");
    if (current !== observed_current_run_id) begin_navigation(current);
    return current;
  }

  const select_run_before_pair_navigation = select_run;
  select_run = function(run_id, options = {}) {
    const next = String(run_id || "");
    if (next !== String(app.current_run_id || "")) begin_navigation(next);
    const result = select_run_before_pair_navigation(run_id, options);
    if (next) {
      queueMicrotask(() => {
        if (String(app.current_run_id || "") !== next) return;
        if (is_nsys_run_id(next)) void probe_pair(next);
        processing_refresh(true);
      });
    }
    return result;
  };

  function set_processing_unavailable(run_id = null) {
    processing_view.available = false;
    processing_view.trace_available = false;
    processing_sync_visibility();
    processing_view.run_id = run_id;
    processing_view.revision = null;
    processing_view.companion_enriched_payload = null;
  }

  function pairing_request_parameters(run_id) {
    const existing = pair_for_run(run_id);
    const claimed = Object.values(app.processing_pairs || {})
      .map(pair => String(pair.ncu_run_id || ""))
      .filter(id => id && id !== existing?.ncu_run_id);
    return {
      excluded:claimed,
      preferred:String(existing?.ncu_run_id || ""),
    };
  }

  // vvv THOG update paired eyes independently of costly trace rendering and ZIP generation
  function probe_pair(run_id) {
    if (pair_for_run(run_id) || !is_visible(run_id) || !is_nsys_run_id(run_id)) return Promise.resolve(false);
    if (pair_probes_in_flight.has(run_id)) return pair_probes_in_flight.get(run_id);
    const request = (async () => {
      for (let attempt = 0; attempt < 3; attempt += 1) {
        const pairing = pairing_request_parameters(run_id);
        const query = pairing.excluded.length
          ? `&exclude_ncu=${encodeURIComponent(pairing.excluded.join(","))}`
          : "";
        const response = await fetch_json(`/api/processing-pair?run=${encodeURIComponent(run_id)}${query}`);
        if (!is_visible(run_id) || !run_for_id(run_id) || pair_for_run(run_id)) return false;
        if (!response.available || response.trace_available !== true) return false;
        const companion_id = String(response.data?.premat_compatibility_source?.dashboard_run_id || "");
        if (companion_id && pair_for_run(companion_id)) continue; // another visible source claimed it during this request
        if (companion_id) return add_pair(run_id, companion_id);
        set_unmatched(run_id, true);
        render_runs();
        return true;
      }
      return false;
    })().catch(error => {
      console.warn(`Processing pair lookup failed for ${run_id}`, error);
      return false;
    }).finally(() => pair_probes_in_flight.delete(run_id));
    pair_probes_in_flight.set(run_id, request);
    return request;
  }
  // ^^^ THOG

  async function refresh_training_throughput_only(run_id, request_epoch, request_serial) {
    const response = await fetch_json(
      `/api/processing-throughput?run=${encodeURIComponent(run_id)}`,
    );
    if (
      request_serial !== refresh_serial
      || request_epoch !== navigation_epoch
      || run_id !== String(app.current_run_id || "")
    ) return;
    await processing_render_throughput({throughput:response.throughput || []});
    if (typeof processing_sync_visibility === "function") processing_sync_visibility();
  }

  async function processing_refresh_once(force = false) {
    const run_id = ensure_navigation_matches_current();
    const request_epoch = navigation_epoch;
    const request_serial = ++refresh_serial;
    if (!run_id) {
      set_processing_unavailable(null);
      return;
    }

    try {
      const existing_pair = pair_for_run(run_id);
      const request_run_id = existing_pair && is_ncu_run_id(run_id)
        ? existing_pair.nsys_run_id
        : run_id;
      const pairing = pairing_request_parameters(run_id);
      const pairing_query = [
        pairing.excluded.length
          ? `exclude_ncu=${encodeURIComponent(pairing.excluded.join(","))}`
          : "",
        pairing.preferred
          ? `preferred_ncu=${encodeURIComponent(pairing.preferred)}`
          : "",
      ].filter(Boolean).join("&");
      const response = await fetch_json(
        `/api/processing?run=${encodeURIComponent(request_run_id)}&pair_epoch=${request_epoch}&request=${request_serial}${pairing_query ? `&${pairing_query}` : ""}`,
      );
      if (
        request_serial !== refresh_serial
        || request_epoch !== navigation_epoch
        || run_id !== String(app.current_run_id || "")
      ) return;
      if (!response.available) {
        set_processing_unavailable(run_id);
        await refresh_training_throughput_only(run_id, request_epoch, request_serial);
        return;
      }
      if (!force && processing_view.run_id === run_id && processing_view.revision === response.revision) return;
      // vvv THOG commit the revision only after the charts render successfully, so a failed render retries
      processing_view.companion_enriched_payload = null;
      processing_view.render_request_run_id = run_id;
      processing_view.render_request_epoch = request_epoch;
      await processing_render(response.data, response.trace_available === true);
      if (request_serial === refresh_serial && request_epoch === navigation_epoch && run_id === String(app.current_run_id || "")) {
        processing_view.run_id = run_id;
        processing_view.revision = response.revision;
      }
      // ^^^ THOG
    } catch (error) {
      if (
        request_serial === refresh_serial
        && request_epoch === navigation_epoch
        && run_id === String(app.current_run_id || "")
      ) console.warn("Processing data refresh failed", error);
    }
  }

  processing_refresh = function(force = false) {
    if (processing_refresh_promise) {
      const current_run_id = String(app.current_run_id || "");
      if (force && (!processing_refresh_in_flight_force || processing_refresh_in_flight_run_id !== current_run_id)) {
        processing_refresh_force_queued = true;
      }
      return processing_refresh_promise;
    }
    processing_refresh_promise = (async () => {
      let next_force = Boolean(force);
      do {
        processing_refresh_force_queued = false;
        processing_refresh_in_flight_force = next_force;
        processing_refresh_in_flight_run_id = String(app.current_run_id || "");
        await processing_refresh_once(next_force);
        next_force = processing_refresh_force_queued;
      } while (next_force);
    })().finally(() => {
      processing_refresh_promise = null;
      processing_refresh_in_flight_force = false;
      processing_refresh_in_flight_run_id = "";
    });
    return processing_refresh_promise;
  };

  function companion_provenance(payload) {
    const source = payload?.premat_compatibility_source;
    if (!source?.dashboard_run_id) return;
    const element = by_id("processing_compatibility_source");
    if (!element) return;
    const distance = Number(source.distance_minutes);
    const skipped = Number(source.skipped_closer_invalid_count || 0);
    const distance_text = Number.isFinite(distance)
      ? `${distance < 10 ? distance.toFixed(1) : Math.round(distance)} min from NSYS`
      : "nearest viable matching NCU";
    const skipped_text = skipped > 0
      ? ` · ${skipped} closer matching NCU capture${skipped === 1 ? "" : "s"} skipped (no usable compatibility data)`
      : "";
    const artifact = String(source.artifact_name || source.dashboard_run_id);
    element.textContent = `NCU companion: ${artifact} · ${distance_text}${skipped_text}`;
    element.title = [artifact, ...(source.skipped_closer_invalid_artifacts || [])].join("\n");
  }

  function apply_pair_state(payload, trace_available, render_run_id) {
    if (render_run_id !== String(app.current_run_id || "") || !is_nsys_run_id(render_run_id)) return;
    const source = payload?.premat_compatibility_source;
    const source_nsys_run_id = String(source?.nsys_dashboard_run_id || "");
    if (source_nsys_run_id && source_nsys_run_id !== render_run_id) return;

    const existing = pair_for_run(render_run_id);
    if (existing) {
      set_unmatched(render_run_id, false);
      companion_provenance(payload);
      render_runs();
      return;
    }

    const companion_id = String(source?.dashboard_run_id || "");
    if (trace_available && companion_id) add_pair(render_run_id, companion_id);
    else if (trace_available) {
      set_unmatched(render_run_id, true);
      render_runs();
    }
    companion_provenance(payload);
  }

  const processing_render_before_pair_state_final = processing_render;
  processing_render = async function(payload, trace_available) {
    const render_run_id = String(processing_view.render_request_run_id || processing_current_run() || "");
    const render_epoch = Number(processing_view.render_request_epoch ?? navigation_epoch);
    await processing_render_before_pair_state_final(payload, trace_available);
    if (render_epoch !== navigation_epoch || render_run_id !== String(app.current_run_id || "")) return;
    apply_pair_state(processing_view.companion_enriched_payload || payload, Boolean(trace_available), render_run_id);
  };

  function restore_persisted_pairs() {
    let pairs_changed = false;
    let visibility_changed = false;
    const seen_ncu = new Set();
    for (const [nsys_run_id, pair] of Object.entries({...app.processing_pairs})) {
      if (
        !run_for_id(nsys_run_id)
        || !run_for_id(pair.ncu_run_id)
        || !is_nsys_run_id(nsys_run_id)
        || !is_ncu_run_id(pair.ncu_run_id)
        || run_timestamp(pair.ncu_run_id) <= run_timestamp(nsys_run_id)
        || seen_ncu.has(pair.ncu_run_id)
      ) {
        delete app.processing_pairs[nsys_run_id];
        pairs_changed = true;
        continue;
      }
      seen_ncu.add(pair.ncu_run_id);
      if (!pair.colour) {
        pair.colour = next_pair_colour();
        pairs_changed = true;
      }
      for (const id of [pair.nsys_run_id, pair.ncu_run_id]) {
        if (!is_visible(id)) {
          app.visibility[id] = true;
          visibility_changed = true;
        }
      }
    }
    rebuild_pair_indexes();
    if (pairs_changed) save_pairs();
    if (visibility_changed) save_json("thog2_local_run_visibility", app.visibility);
    if (pairs_changed || visibility_changed || app.processing_paired_run_ids.size) render_runs();
  }

  async function reconcile_startup_pairing() {
    if (app.instra_catalog_ready !== true || !(app.runs || []).length) return;
    const generation = Number(app.instra_catalog_generation || 0);
    if (startup_reconciled_generation === generation) return;
    startup_reconciled_generation = generation;
    restore_persisted_pairs();
    const visible_nsys = (app.runs || [])
      .map(run => String(run_identifier(run)))
      .filter(run_id => is_visible(run_id) && is_nsys_run_id(run_id))
      .sort((left, right) => run_timestamp(left) - run_timestamp(right));
    let state_changed = false;
    for (const nsys_run_id of visible_nsys) {
      if (pair_for_run(nsys_run_id)) continue;
      state_changed = (await probe_pair(nsys_run_id)) || state_changed;
    }
    if (state_changed) render_runs();
  }

  const runs_body = by_id("runs_body");
  if (runs_body && runs_body.dataset.instraPairUnpairInstalled !== "true") {
    runs_body.dataset.instraPairUnpairInstalled = "true";
    runs_body.addEventListener("click", event => {
      const eye = event.target.closest?.(".eye-button");
      const row = eye?.closest?.("tr[data-run-id]");
      const run_id = String(row?.dataset?.runId || "");
      if (!run_id) return;
      if (is_visible(run_id)) {
        if (is_nsys_run_id(run_id)) void probe_pair(run_id);
        return;
      }
      if (!pair_for_run(run_id)) return;
      unpair_run(run_id, {close_both:true, render:true});
    });
  }

  window.addEventListener("load", () => setTimeout(reconcile_startup_pairing, 0));
  window.addEventListener("instra:catalog-ready", () => queueMicrotask(reconcile_startup_pairing));
  if (app.instra_catalog_ready === true) queueMicrotask(reconcile_startup_pairing);
  window.addEventListener("popstate", () => {
    queueMicrotask(() => {
      const current = ensure_navigation_matches_current();
      if (current && is_nsys_run_id(current)) processing_refresh(true);
    });
  });

  window.setInterval(() => {
    const current = String(app.current_run_id || "");
    if (current === observed_current_run_id) return;
    begin_navigation(current);
    if (current && is_nsys_run_id(current)) processing_refresh(true);
  }, 250);

  window.processing_pair_unpair_run = unpair_run;
  window.processing_pair_state_test_hooks = Object.freeze({
    add_pair,
    unpair_run,
    pair_for_run,
    rebuild_pair_indexes,
    restore_persisted_pairs,
    reconcile_startup_pairing,
    pair_palette,
    pairing_request_parameters,
    probe_pair,
  });
})();
// ^^^ THOG
