// vvv THOG
"use strict";

(function install_processing_pair_state_final() {
  app.processing_auto_opened_run_ids = app.processing_auto_opened_run_ids || new Set();
  app.processing_paired_run_ids = app.processing_paired_run_ids || new Set();
  app.processing_pair_roles = app.processing_pair_roles || {};

  let navigation_epoch = 0;
  let refresh_serial = 0;
  let observed_current_run_id = String(app.current_run_id || "");

  function same_set(left, right) {
    if (left.size !== right.size) return false;
    for (const value of left) if (!right.has(value)) return false;
    return true;
  }

  function visibility_snapshot() {
    const snapshot = new Map();
    for (const run of app.runs || []) {
      const run_id = String(run_identifier(run));
      snapshot.set(run_id, is_visible(run_id));
    }
    return snapshot;
  }

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

  function save_visibility_if(changed) {
    if (changed) save_json("thog2_local_run_visibility", app.visibility);
  }

  function clear_managed_pair(next_run_id = "") {
    let visibility_changed = false;
    for (const run_id of new Set(app.processing_auto_opened_run_ids || [])) {
      if (run_id === next_run_id) continue;
      if (is_visible(run_id)) {
        app.visibility[run_id] = false;
        visibility_changed = true;
      }
    }
    save_visibility_if(visibility_changed);
    const had_pair = (app.processing_paired_run_ids?.size || 0) > 0
      || (app.processing_auto_opened_run_ids?.size || 0) > 0;
    app.processing_paired_run_ids = new Set();
    app.processing_auto_opened_run_ids = new Set();
    app.processing_pair_roles = {};
    if (visibility_changed || had_pair) render_runs();
  }

  function begin_navigation(next_run_id) {
    navigation_epoch += 1;
    observed_current_run_id = String(next_run_id || "");
    clear_managed_pair(observed_current_run_id);
    processing_view.run_id = null;
    processing_view.revision = null;
    processing_view.companion_enriched_payload = null;
    processing_view.render_request_run_id = null;
    processing_view.render_request_epoch = navigation_epoch;
  }

  function ensure_navigation_matches_current() {
    const current = String(app.current_run_id || "");
    if (current !== observed_current_run_id) begin_navigation(current);
    return current;
  }

  const select_run_before_pair_navigation = select_run;
  select_run = function(run_id, options = {}) {
    const next = String(run_id || "");
    const changed = next !== String(app.current_run_id || "");
    if (changed) begin_navigation(next);
    const result = select_run_before_pair_navigation(run_id, options);
    if (next) {
      queueMicrotask(() => {
        if (String(app.current_run_id || "") !== next) return;
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

  processing_refresh = async function(force = false) {
    const run_id = ensure_navigation_matches_current();
    const request_epoch = navigation_epoch;
    const request_serial = ++refresh_serial;

    if (!run_id) {
      set_processing_unavailable(null);
      clear_managed_pair("");
      return;
    }

    try {
      const response = await fetch_json(`/api/processing?run=${encodeURIComponent(run_id)}`);
      if (
        request_serial !== refresh_serial
        || request_epoch !== navigation_epoch
        || run_id !== String(app.current_run_id || "")
      ) return;

      if (!response.available) {
        set_processing_unavailable(run_id);
        clear_managed_pair(run_id);
        return;
      }

      if (!force && processing_view.run_id === run_id && processing_view.revision === response.revision) return;

      processing_view.run_id = run_id;
      processing_view.revision = response.revision;
      processing_view.companion_enriched_payload = null;
      processing_view.render_request_run_id = run_id;
      processing_view.render_request_epoch = request_epoch;
      processing_view.pair_visibility_before_render = visibility_snapshot();

      await processing_render(response.data, response.trace_available === true);

      if (
        request_serial !== refresh_serial
        || request_epoch !== navigation_epoch
        || run_id !== String(app.current_run_id || "")
      ) {
        queueMicrotask(() => processing_refresh(true));
      }
    } catch (error) {
      if (
        request_serial === refresh_serial
        && request_epoch === navigation_epoch
        && run_id === String(app.current_run_id || "")
      ) console.warn("Processing data refresh failed", error);
    }
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

  function apply_pair_state(payload, trace_available, before_visibility, render_run_id) {
    if (
      render_run_id !== String(app.current_run_id || "")
      || !is_nsys_run_id(render_run_id)
    ) return;

    const source = payload?.premat_compatibility_source;
    const companion_id = String(source?.dashboard_run_id || "");
    const next = new Set();
    const roles = {};
    if (trace_available && render_run_id && companion_id) {
      next.add(render_run_id);
      next.add(companion_id);
      roles[render_run_id] = "NSYS source · paired Processing evidence";
      roles[companion_id] = "NCU companion · nearest viable matching capture automatically paired with selected NSYS run";
    }

    let visibility_changed = false;
    const previous_auto = new Set(app.processing_auto_opened_run_ids || []);
    const next_auto = new Set();

    for (const run_id of previous_auto) {
      if (next.has(run_id)) continue;
      if (is_visible(run_id)) {
        app.visibility[run_id] = false;
        visibility_changed = true;
      }
    }

    for (const run_id of next) {
      const was_visible_before_render = before_visibility.get(run_id) === true;
      if (!is_visible(run_id)) {
        app.visibility[run_id] = true;
        visibility_changed = true;
      }
      if (!was_visible_before_render || previous_auto.has(run_id)) next_auto.add(run_id);
    }

    save_visibility_if(visibility_changed);
    const changed = !same_set(app.processing_paired_run_ids || new Set(), next)
      || !same_set(app.processing_auto_opened_run_ids || new Set(), next_auto);
    app.processing_paired_run_ids = next;
    app.processing_auto_opened_run_ids = next_auto;
    app.processing_pair_roles = roles;
    if (changed || visibility_changed) render_runs();
    companion_provenance(payload);
  }

  function undo_stale_render_visibility(payload, before_visibility, render_run_id) {
    const ids = new Set([
      render_run_id,
      String(payload?.premat_compatibility_source?.dashboard_run_id || ""),
    ]);
    let changed = false;
    const current = String(app.current_run_id || "");
    for (const run_id of ids) {
      if (!run_id || run_id === current) continue;
      if (before_visibility.get(run_id) === false && is_visible(run_id)) {
        app.visibility[run_id] = false;
        changed = true;
      }
    }
    save_visibility_if(changed);
    if (changed) render_runs();
  }

  const processing_render_before_pair_state_final = processing_render;
  processing_render = async function(payload, trace_available) {
    const render_run_id = String(processing_view.render_request_run_id || processing_current_run() || "");
    const render_epoch = Number(processing_view.render_request_epoch ?? navigation_epoch);
    const before_visibility = processing_view.pair_visibility_before_render || visibility_snapshot();

    await processing_render_before_pair_state_final(payload, trace_available);
    const effective = processing_view.companion_enriched_payload || payload;

    if (
      render_epoch !== navigation_epoch
      || render_run_id !== String(app.current_run_id || "")
    ) {
      undo_stale_render_visibility(effective, before_visibility, render_run_id);
      return;
    }

    apply_pair_state(effective, Boolean(trace_available), before_visibility, render_run_id);
  };

  function visible_run_ids(predicate) {
    return (app.runs || [])
      .map(run => String(run_identifier(run)))
      .filter(run_id => is_visible(run_id) && predicate(run_id));
  }

  function reconcile_startup_pairing() {
    const current = ensure_navigation_matches_current();
    if (current && is_nsys_run_id(current)) {
      processing_refresh(true);
      return;
    }

    const visible_nsys = visible_run_ids(is_nsys_run_id);
    const visible_ncu = visible_run_ids(is_ncu_run_id);
    if (visible_nsys.length !== 1 || visible_ncu.length !== 0) return;

    // Persisted visibility can restore an NSYS eye before any current-run
    // selection exists. In that unambiguous case, promote the visible NSYS to
    // the Processing source and let normal pairing open its NCU companion.
    const source_run_id = visible_nsys[0];
    if (source_run_id !== String(app.current_run_id || "")) {
      select_run(source_run_id, {manual:true, replace_history:true});
    } else {
      processing_refresh(true);
    }
  }

  function refresh_selected_nsys_now() {
    const current = ensure_navigation_matches_current();
    if (!current || !is_nsys_run_id(current)) return;
    processing_refresh(true);
  }

  // window.load can precede the first catalogue result. Keep this fast-path for
  // warm starts, and reconcile again when dashboard.js reports that the first
  // run-database read has actually completed.
  window.addEventListener("load", () => {
    setTimeout(reconcile_startup_pairing, 0);
  });
  window.addEventListener("instra:catalog-ready", () => {
    queueMicrotask(reconcile_startup_pairing);
  });

  window.addEventListener("popstate", () => {
    queueMicrotask(refresh_selected_nsys_now);
  });

  // Defensive navigation watcher: cheap string comparison only. This catches any
  // future code path that changes app.current_run_id without going through
  // select_run(), while avoiding periodic Processing fetches when nothing moved.
  window.setInterval(() => {
    const current = String(app.current_run_id || "");
    if (current === observed_current_run_id) return;
    begin_navigation(current);
    if (current && is_nsys_run_id(current)) processing_refresh(true);
  }, 250);
})();
// ^^^ THOG
