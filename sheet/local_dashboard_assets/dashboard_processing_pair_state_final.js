// vvv THOG
"use strict";

(function install_processing_pair_state_final() {
  app.processing_auto_opened_run_ids = app.processing_auto_opened_run_ids || new Set();

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

  function apply_pair_state(payload, trace_available, before_visibility) {
    const source = payload?.premat_compatibility_source;
    const selected_id = String(processing_current_run() || "");
    const companion_id = String(source?.dashboard_run_id || "");
    const next = new Set();
    const roles = {};
    if (trace_available && selected_id && companion_id) {
      next.add(selected_id);
      next.add(companion_id);
      roles[selected_id] = "NSYS source · paired Processing evidence";
      roles[companion_id] = "NCU companion · nearest matching capture automatically paired with selected NSYS run";
    }

    let visibility_changed = false;
    const previous_auto = new Set(app.processing_auto_opened_run_ids || []);
    const next_auto = new Set();

    // Close only companions that pairing opened itself. Ordinary comparison eyes
    // that the user opened manually remain untouched when the profiler pair moves.
    for (const run_id of previous_auto) {
      if (next.has(run_id)) {
        next_auto.add(run_id);
        continue;
      }
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
      if (run_id !== selected_id && (!was_visible_before_render || previous_auto.has(run_id))) {
        // The older pairing layer may already have opened this eye during the
        // inner render. The pre-render snapshot lets us still record ownership.
        next_auto.add(run_id);
      }
    }

    if (visibility_changed) save_json("thog2_local_run_visibility", app.visibility);

    const changed = !same_set(app.processing_paired_run_ids || new Set(), next)
      || !same_set(app.processing_auto_opened_run_ids || new Set(), next_auto);
    app.processing_paired_run_ids = next;
    app.processing_auto_opened_run_ids = next_auto;
    app.processing_pair_roles = roles;

    if (changed || visibility_changed) render_runs();
  }

  const processing_render_before_pair_state_final = processing_render;
  processing_render = async function(payload, trace_available) {
    const before_visibility = visibility_snapshot();
    await processing_render_before_pair_state_final(payload, trace_available);
    const effective = processing_view.companion_enriched_payload || payload;
    apply_pair_state(effective, Boolean(trace_available), before_visibility);
  };
})();
// ^^^ THOG
