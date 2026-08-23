// vvv THOG
"use strict";

// Final owner for Weights display semantics.
// Restores the intended precedence:
// chart override > Weights-group setting > default.
// "Current weights only" always means the latest optimizer step for that chart;
// an explicit step window applies only to charts that are not current-only.
window.addEventListener("load", () => {
  setTimeout(() => {
    const weight_chart_names = [...depth_weight_chart_names];
    const weight_chart_set = new Set(weight_chart_names);
    const group_storage_key = "thog2_local_weight_group_settings_v1";
    const override_storage_key = "thog2_local_weight_chart_overrides_v1";

    const own = (object, key) => (
      Boolean(object)
      && typeof object === "object"
      && !Array.isArray(object)
      && Object.prototype.hasOwnProperty.call(object, key)
    );

    const json_store = key => {
      const value = load_json(key, {});
      return value && typeof value === "object" && !Array.isArray(value) ? value : {};
    };

    const editor_workspace_mode = chart_name => (
      app.axis_chart_name === chart_name
      && typeof app.axis_chart_workspace_mode === "boolean"
        ? app.axis_chart_workspace_mode
        : app.workspace_mode === true
    );

    const group_scope = chart_name => (
      editor_workspace_mode(chart_name)
        ? "workspace"
        : `run:${String(app.current_run_id || "unselected")}`
    );

    const chart_scope = chart_name => `${group_scope(chart_name)}:${chart_name}`;

    const group_settings = chart_name => {
      const store = json_store(group_storage_key);
      const value = store[group_scope(chart_name)];
      return value && typeof value === "object" && !Array.isArray(value) ? value : {};
    };

    const modern_override = chart_name => {
      const store = json_store(override_storage_key);
      const value = store[chart_scope(chart_name)];
      return value && typeof value === "object" && !Array.isArray(value) ? value : {};
    };

    const legacy_override = (chart_name, field) => {
      const scope = chart_scope(chart_name);
      if (field === "current_weights_only" && own(app.weight_current_only, scope)) {
        return app.weight_current_only[scope] === true;
      }
      if (field === "join_with_line_segments" && own(app.weight_join_with_line_segments, scope)) {
        return app.weight_join_with_line_segments[scope] === true;
      }
      return null;
    };

    const stored_flag = (chart_name, field) => {
      const override = modern_override(chart_name);
      if (own(override, field)) return override[field] === true;
      const legacy = legacy_override(chart_name, field);
      if (legacy !== null) return legacy;
      const group = group_settings(chart_name);
      if (own(group, field)) return group[field] === true;
      return false;
    };

    const supplied_flag = (supplied, field) => (
      supplied && typeof supplied === "object" && !Array.isArray(supplied) && own(supplied, field)
        ? supplied[field] === true
        : null
    );

    const base_normalize_chart_settings_weight_semantics = normalize_chart_settings;
    normalize_chart_settings = function(chart_name, supplied = null) {
      const normalized = base_normalize_chart_settings_weight_semantics(chart_name, supplied);
      if (!weight_chart_set.has(chart_name)) return normalized;

      const supplied_current = supplied_flag(supplied, "current_weights_only");
      const supplied_join = supplied_flag(supplied, "join_with_line_segments");
      normalized.current_weights_only = supplied_current !== null
        ? supplied_current
        : stored_flag(chart_name, "current_weights_only");
      normalized.join_with_line_segments = supplied_join !== null
        ? supplied_join
        : stored_flag(chart_name, "join_with_line_segments");
      return normalized;
    };

    const trace_update = trace => {
      try {
        const value = trace_optimizer_update(trace);
        return Number.isFinite(value) ? Number(value) : null;
      } catch (_error) {
        return null;
      }
    };

    const trace_run_id = trace => {
      const meta = trace?.meta;
      if (meta && typeof meta === "object" && !Array.isArray(meta) && meta.instra_workspace_run_id) {
        return String(meta.instra_workspace_run_id);
      }
      return "__instra_single_run__";
    };

    const retain_latest = prepared => {
      const latest_by_run = new Map();
      for (const trace of prepared?.data || []) {
        const update = trace_update(trace);
        if (update === null) continue;
        const run_id = trace_run_id(trace);
        latest_by_run.set(run_id, Math.max(update, latest_by_run.get(run_id) ?? -Infinity));
      }
      prepared.data = (prepared?.data || []).filter(trace => {
        const update = trace_update(trace);
        if (update === null) return true;
        const latest = latest_by_run.get(trace_run_id(trace));
        return Number.isFinite(latest) && update === latest;
      });
      return prepared;
    };

    // Replace the late step-window bypass. Current-only must remain current-only
    // even when a start/end display window exists.
    retain_latest_weight_snapshots = retain_latest;
    if (typeof instra_enforce_workspace_latest_weights === "function") {
      instra_enforce_workspace_latest_weights = retain_latest;
    }

    const selected_range = () => {
      const value = window.__instra_weight_step_filter?.request_range?.();
      const minimum = Number(value?.minimum);
      const maximum = Number(value?.maximum);
      if (!Number.isInteger(minimum) || !Number.isInteger(maximum)) return null;
      if (minimum < 0 || maximum < minimum) return null;
      return {minimum, maximum};
    };

    const current_only_flags = () => (
      weight_chart_names.map(chart_name => normalize_chart_settings(chart_name)?.current_weights_only === true)
    );

    // Bypass the older request wrapper for depth figures so a mixed view can satisfy
    // current-only charts and historical-window charts simultaneously.
    const base_fetch_json_weight_semantics = fetch_json;
    fetch_json = async function(url, options = {}) {
      let parsed = null;
      try {
        parsed = new URL(url, window.location.origin);
      } catch (_error) {
        return base_fetch_json_weight_semantics(url, options);
      }
      if (
        options?.method
        || parsed.pathname !== "/api/figure-family"
        || parsed.searchParams.get("family") !== "depth"
      ) {
        return base_fetch_json_weight_semantics(url, options);
      }

      const range = selected_range();
      const flags = current_only_flags();
      const any_current_only = flags.some(Boolean);
      const all_current_only = flags.length > 0 && flags.every(Boolean);

      parsed.searchParams.delete("current_only");
      parsed.searchParams.delete("step_min");
      parsed.searchParams.delete("step_max");

      if (all_current_only) {
        parsed.searchParams.set("current_only", "1");
      } else if (range && !any_current_only) {
        parsed.searchParams.set("step_min", String(range.minimum));
        parsed.searchParams.set("step_max", String(range.maximum));
      }
      // Mixed mode intentionally requests the retained family: current-only charts
      // need the actual newest snapshot, while historical charts are filtered below
      // to the selected range.

      const response = await fetch(
        `${parsed.pathname}${parsed.search}`,
        {cache: "no-store", ...options},
      );
      const value = await response.json();
      if (!response.ok) throw new Error(value.error || `${response.status} ${response.statusText}`);
      return value;
    };

    const base_prepare_figure_weight_semantics = prepare_figure;
    prepare_figure = function(figure, chart_name) {
      const prepared = base_prepare_figure_weight_semantics(figure, chart_name);
      if (!weight_chart_set.has(chart_name)) return prepared;

      const render_override = app.chart_settings_render_override;
      const supplied = render_override?.chart_name === chart_name ? render_override.settings : null;
      const settings = normalize_chart_settings(chart_name, supplied);
      const range = selected_range();

      if (settings.current_weights_only === true) {
        retain_latest(prepared);
      } else if (range) {
        prepared.data = (prepared.data || []).filter(trace => {
          const update = trace_update(trace);
          return update === null || (update >= range.minimum && update <= range.maximum);
        });
      }
      return prepared;
    };

    const weight_editor_open = () => (
      !by_id("chart_settings_overlay")?.hidden
      && weight_chart_set.has(app.axis_chart_name)
    );

    const group_editor_open = () => (
      weight_editor_open()
      && Boolean(by_id("weights_group_scale_field"))
      && by_id("weights_group_scale_field").hidden === false
    );

    const sync_editor_controls = ({load_values = false} = {}) => {
      if (!weight_editor_open()) return;
      const chart_name = app.axis_chart_name;
      const current = by_id("chart_current_weights_only");
      const join = by_id("chart_join_with_line_segments");
      const current_note = by_id("chart_current_weights_only_field")?.querySelector("small");
      const join_note = by_id("chart_join_with_line_segments_field")?.querySelector("small");

      if (group_editor_open()) {
        const group = group_settings(chart_name);
        if (load_values && current) current.checked = group.current_weights_only === true;
        if (load_values && join) join.checked = group.join_with_line_segments === true;
        if (current) current.disabled = false;
        if (join) join.disabled = false;
        if (current_note) current_note.textContent = "Applies to all six weight charts in this view unless a chart overrides it.";
        if (join_note) join_note.textContent = "Applies to all six weight charts in this view unless a chart overrides it.";
        return;
      }

      const inherit = by_id("chart_inherit_weights_group")?.checked === true;
      if (load_values) {
        if (current) current.checked = stored_flag(chart_name, "current_weights_only");
        if (join) join.checked = stored_flag(chart_name, "join_with_line_segments");
      }
      if (current) current.disabled = inherit;
      if (join) join.disabled = inherit;
      const note = inherit
        ? "Inherited from Weights group settings."
        : "Overrides Weights group settings for this chart.";
      if (current_note) current_note.textContent = note;
      if (join_note) join_note.textContent = note;
    };

    const base_populate_chart_settings_form_weight_semantics = populate_chart_settings_form;
    populate_chart_settings_form = function(chart_name, supplied = null) {
      const result = base_populate_chart_settings_form_weight_semantics(chart_name, supplied);
      if (weight_chart_set.has(chart_name)) sync_editor_controls({load_values: true});
      return result;
    };

    const base_sync_chart_setting_outputs_weight_semantics = sync_chart_setting_outputs;
    sync_chart_setting_outputs = function() {
      const editing = weight_editor_open();
      const current = by_id("chart_current_weights_only");
      const join = by_id("chart_join_with_line_segments");
      const current_checked = editing && current ? current.checked : null;
      const join_checked = editing && join ? join.checked : null;
      const result = base_sync_chart_setting_outputs_weight_semantics();
      if (editing) {
        if (current && current_checked !== null) current.checked = current_checked;
        if (join && join_checked !== null) join.checked = join_checked;
        sync_editor_controls({load_values: false});
      }
      return result;
    };

    const base_open_chart_settings_weight_semantics = open_chart_settings;
    open_chart_settings = function(chart_name) {
      const result = base_open_chart_settings_weight_semantics(chart_name);
      if (weight_chart_set.has(chart_name)) {
        queueMicrotask(() => sync_editor_controls({load_values: true}));
      }
      return result;
    };

    by_id("chart_inherit_weights_group")?.addEventListener("change", () => {
      queueMicrotask(() => sync_editor_controls({load_values: true}));
    });

    // The obsolete global store is deliberately ignored. Expose the repaired
    // effective state for diagnostics without changing the established step API.
    window.__instra_weight_semantics_repair = Object.freeze({
      effective: chart_name => ({
        current_weights_only: stored_flag(chart_name, "current_weights_only"),
        join_with_line_segments: stored_flag(chart_name, "join_with_line_segments"),
      }),
      selected_range,
    });

    sync_editor_controls({load_values: true});
  }, 5);
});
// ^^^ THOG
