// vvv THOG
"use strict";

// Stable final owner for Weights time/history semantics. Installation is dependency-
// gated rather than timer-ordered: legacy layers may initialise later, but once this
// owner installs they cannot subsequently overwrite its state or rendering contract.
window.addEventListener("load", () => {
  const install = () => {
    if (window.__instra_weight_stability_final) return true;
    if (!window.__instra_weight_controls_v2) return false;
    if (!window.__instra_weight_group_settings) return false;
    if (!window.__instra_matched_weight_selection) return false;
    if (!window.__instra_weight_presentation) return false;
    if (!window.__thog2_dashboard_performance) return false;
    if (!window.__instra_legacy_heatmap_repair) return false;
    if (typeof normalize_chart_settings !== "function") return false;
    if (typeof prepare_figure !== "function") return false;
    if (typeof refresh_current_run !== "function") return false;
    if (typeof select_run !== "function") return false;

    const weight_chart_names = [...depth_weight_chart_names];
    const weight_chart_set = new Set(weight_chart_names);
    const group_storage_key = "thog2_local_weight_group_settings_v1";
    const override_storage_key = "thog2_local_weight_chart_overrides_v1";
    const start_key = "instrumentation__depth_weight_curves__start_step";
    const end_key = "instrumentation__depth_weight_curves__end_step";
    const legacy_step_api = window.__instra_weight_controls_v2;
    const legacy_clear_step_range = typeof legacy_step_api.clear_step_range === "function"
      ? legacy_step_api.clear_step_range.bind(legacy_step_api)
      : null;
    const selection_api = window.__instra_matched_weight_selection;
    const range_by_context = new Map();
    const loading_contexts = new Set();
    let refresh_suppression = 0;

    const own = (object, key) => (
      Boolean(object)
      && typeof object === "object"
      && !Array.isArray(object)
      && Object.prototype.hasOwnProperty.call(object, key)
    );

    const finite_step = value => {
      if (value === null || value === undefined || value === "") return null;
      const numeric = Number(value);
      return Number.isInteger(numeric) && numeric >= 0 ? numeric : null;
    };

    const json_store = key => {
      const value = load_json(key, {});
      return value && typeof value === "object" && !Array.isArray(value) ? value : {};
    };

    const run_id = run => {
      if (!run) return "";
      try { return String(run_identifier(run)); }
      catch (_error) {
        return String(run.dashboard_run_id || run.local_run_id || run.wandb_run_id || run.run_name || "");
      }
    };

    const visible_workspace_runs = () => (
      app.workspace_mode === true && typeof window.__instra_workspace?.visible_runs === "function"
        ? window.__instra_workspace.visible_runs()
        : []
    );

    const context_key = () => {
      if (app.workspace_mode === true) {
        const identifiers = visible_workspace_runs().map(run_id).filter(Boolean).sort();
        return `workspace:${identifiers.join("|")}`;
      }
      return app.current_run_id ? `run:${String(app.current_run_id)}` : "";
    };

    const current_context_runs = () => {
      if (app.workspace_mode === true) return visible_workspace_runs();
      const run = typeof current_run === "function" ? current_run() : app.current_status;
      return run ? [run] : [];
    };

    const merged_configuration = run => {
      const base = run?.configuration && typeof run.configuration === "object"
        ? run.configuration
        : {};
      const current = (
        run_id(run)
        && run_id(run) === String(app.current_run_id || "")
        && app.current_status?.configuration
        && typeof app.current_status.configuration === "object"
      ) ? app.current_status.configuration : {};
      return {...base, ...current};
    };

    const configured_range = () => {
      const runs = current_context_runs();
      if (!runs.length) return null;
      const ranges = [];
      for (const run of runs) {
        const configuration = merged_configuration(run);
        if (!own(configuration, start_key) || !own(configuration, end_key)) return null;
        const minimum = finite_step(configuration[start_key]);
        const maximum = finite_step(configuration[end_key]);
        if (minimum === null || maximum === null || maximum < minimum) return null;
        ranges.push({minimum, maximum});
      }
      const first = ranges[0];
      return ranges.every(range => (
        range.minimum === first.minimum && range.maximum === first.maximum
      )) ? first : null;
    };

    const state_for_context = (key = context_key()) => {
      if (!key) return null;
      if (!range_by_context.has(key)) {
        range_by_context.set(key, {range: null, user_overridden: false});
      }
      return range_by_context.get(key);
    };

    const seed_configured_range = () => {
      const state = state_for_context();
      if (!state || state.user_overridden || state.range) return false;
      const configured = configured_range();
      if (!configured) return false;
      state.range = configured;
      return true;
    };

    const selected_range = () => {
      seed_configured_range();
      const range = state_for_context()?.range;
      if (!range) return null;
      return {minimum: range.minimum, maximum: range.maximum};
    };

    const group_scope = chart_name => {
      if (app.workspace_mode === true) return "workspace";
      return `run:${String(app.current_run_id || "unselected")}`;
    };

    const chart_scope = chart_name => `${group_scope(chart_name)}:${chart_name}`;

    const effective_flag = (chart_name, field) => {
      const overrides = json_store(override_storage_key);
      const override = overrides[chart_scope(chart_name)];
      if (override && typeof override === "object" && own(override, field)) {
        return override[field] === true;
      }
      const legacy_store = field === "current_weights_only"
        ? app.weight_current_only
        : app.weight_join_with_line_segments;
      if (own(legacy_store, chart_scope(chart_name))) {
        return legacy_store[chart_scope(chart_name)] === true;
      }
      const groups = json_store(group_storage_key);
      const group = groups[group_scope(chart_name)];
      if (group && typeof group === "object" && own(group, field)) return group[field] === true;
      return false;
    };

    const supplied_flag = (supplied, field) => (
      supplied && typeof supplied === "object" && !Array.isArray(supplied) && own(supplied, field)
        ? supplied[field] === true
        : null
    );

    const base_normalize_chart_settings = normalize_chart_settings;
    const final_normalize_chart_settings = function(chart_name, supplied = null) {
      const normalized = base_normalize_chart_settings(chart_name, supplied);
      if (!weight_chart_set.has(chart_name)) return normalized;
      const current = supplied_flag(supplied, "current_weights_only");
      const join = supplied_flag(supplied, "join_with_line_segments");
      normalized.current_weights_only = current !== null
        ? current
        : effective_flag(chart_name, "current_weights_only");
      normalized.join_with_line_segments = join !== null
        ? join
        : effective_flag(chart_name, "join_with_line_segments");
      return normalized;
    };
    normalize_chart_settings = final_normalize_chart_settings;

    const trace_update = trace => {
      try {
        const value = trace_optimizer_update(trace);
        return Number.isFinite(value) ? Number(value) : null;
      } catch (_error) {
        return null;
      }
    };

    const trace_run = trace => {
      const meta = trace?.meta;
      if (meta && typeof meta === "object" && !Array.isArray(meta) && meta.instra_workspace_run_id) {
        return String(meta.instra_workspace_run_id);
      }
      return "__instra_single_run__";
    };

    const retain_latest = prepared => {
      const latest = new Map();
      for (const trace of prepared?.data || []) {
        const update = trace_update(trace);
        if (update === null) continue;
        const key = trace_run(trace);
        latest.set(key, Math.max(update, latest.get(key) ?? -Infinity));
      }
      prepared.data = (prepared?.data || []).filter(trace => {
        const update = trace_update(trace);
        if (update === null) return true;
        return update === latest.get(trace_run(trace));
      });
      return prepared;
    };

    retain_latest_weight_snapshots = retain_latest;
    if (typeof instra_enforce_workspace_latest_weights === "function") {
      instra_enforce_workspace_latest_weights = retain_latest;
    }

    const selected_coordinate_enabled = chart_name => {
      const selection = selection_api.selection?.() || {};
      if (selection.user_selected !== true) return false;
      const capability = selection_api.capability?.(chart_name);
      return capability?.available !== false;
    };

    const trace_key = trace => {
      const meta = trace?.meta && typeof trace.meta === "object" && !Array.isArray(trace.meta)
        ? trace.meta
        : {};
      return JSON.stringify([
        trace_update(trace),
        String(meta.instra_workspace_run_id || ""),
        String(meta.instra_weight_selection_kind || ""),
        finite_step(meta.instra_weight_model_feature),
        finite_step(meta.instra_weight_intermediate_feature),
      ]);
    };

    const original_colours = figure => {
      const colours = new Map();
      for (const trace of figure?.data || []) {
        colours.set(trace_key(trace), {
          line: trace?.line?.color,
          marker: trace?.marker?.color,
          marker_line: trace?.marker?.line?.color,
        });
      }
      return colours;
    };

    const restore_colours = (prepared, colours) => {
      for (const trace of prepared?.data || []) {
        const colour = colours.get(trace_key(trace));
        if (!colour) continue;
        if (trace?.line && colour.line !== undefined) trace.line = {...trace.line, color: colour.line};
        if (trace?.marker && colour.marker !== undefined) trace.marker = {...trace.marker, color: colour.marker};
        if (trace?.marker?.line && colour.marker_line !== undefined) {
          trace.marker.line = {...trace.marker.line, color: colour.marker_line};
        }
      }
    };

    const apply_run_colour = prepared => {
      if (app.workspace_mode === true) return;
      const colour = colour_for_run(String(app.current_run_id || ""));
      for (const trace of prepared?.data || []) {
        if (trace?.meta?.instra_top_axis_anchor === true) continue;
        const mode = String(trace?.mode || "");
        if (mode.includes("lines") && trace.line) trace.line = {...trace.line, color: colour};
        if (mode.includes("markers") || trace.marker) {
          trace.marker = {...(trace.marker || {}), color: colour};
          trace.marker.line = {...(trace.marker?.line || {}), color: colour};
        }
      }
    };

    const base_prepare_figure = prepare_figure;
    prepare_figure = function(figure, chart_name) {
      if (!weight_chart_set.has(chart_name)) return base_prepare_figure(figure, chart_name);

      const render_override = app.chart_settings_render_override;
      const supplied = render_override?.chart_name === chart_name ? render_override.settings : null;
      const settings = final_normalize_chart_settings(chart_name, supplied);
      const coordinate_selected = selected_coordinate_enabled(chart_name);
      const colours = app.workspace_mode === true ? null : original_colours(figure);
      const saved_normalize = normalize_chart_settings;
      const saved_retain = retain_latest_weight_snapshots;
      const saved_workspace_latest = typeof instra_enforce_workspace_latest_weights === "function"
        ? instra_enforce_workspace_latest_weights
        : null;
      const editor_current = by_id("chart_current_weights_only");
      const editor_controls_chart = (
        app.axis_chart_name === chart_name
        && !by_id("chart_settings_overlay")?.hidden
        && editor_current
      );
      const saved_editor_current = editor_controls_chart ? editor_current.checked : null;

      // The legacy matched-weight layer treats current-only as a coordinate selector.
      // Feed that one inner layer the coordinate choice while suppressing its time
      // collapse. The actual time/history filter is applied once, below.
      normalize_chart_settings = function(candidate, inner_supplied = null) {
        const normalized = saved_normalize(candidate, inner_supplied);
        if (candidate === chart_name) normalized.current_weights_only = coordinate_selected;
        return normalized;
      };
      retain_latest_weight_snapshots = () => undefined;
      if (saved_workspace_latest) instra_enforce_workspace_latest_weights = prepared => prepared;
      if (editor_controls_chart) editor_current.checked = coordinate_selected;

      let prepared;
      try {
        prepared = base_prepare_figure(figure, chart_name);
      } finally {
        normalize_chart_settings = saved_normalize;
        retain_latest_weight_snapshots = saved_retain;
        if (saved_workspace_latest) instra_enforce_workspace_latest_weights = saved_workspace_latest;
        if (editor_controls_chart) editor_current.checked = saved_editor_current;
      }

      const range = selected_range();
      if (settings.current_weights_only === true) {
        retain_latest(prepared);
        apply_run_colour(prepared);
      } else if (range) {
        prepared.data = (prepared.data || []).filter(trace => {
          const update = trace_update(trace);
          return update === null || (update >= range.minimum && update <= range.maximum);
        });
        if (colours) restore_colours(prepared, colours);
      } else {
        if (typeof limit_curve_snapshots === "function") {
          limit_curve_snapshots(prepared, settings.max_snapshots, settings.snapshot_window_mode);
        }
        if (colours) restore_colours(prepared, colours);
      }
      return prepared;
    };

    const invalidate_depth = () => {
      window.__instra_workspace_depth_cache?.clear?.();
      const performance = window.__thog2_dashboard_performance?.state;
      if (performance) {
        performance.depth_signature = null;
        performance.pending_render = null;
        performance.deferred_coefficients = true;
      }
      if (app.figures && typeof app.figures === "object") {
        app.figures = {...app.figures, depth: {}};
      }
      app.figure_revision = null;
    };

    const sync_header = () => {
      const range = selected_range();
      const from = by_id("weight_step_from");
      const to = by_id("weight_step_to");
      const whole = by_id("weight_step_whole_range");
      const current = by_id("weight_step_current");
      const availability = by_id("weight_step_availability");
      if (from) from.value = range ? String(range.minimum) : "";
      if (to) to.value = range ? String(range.maximum) : "";
      if (whole) whole.disabled = !range;

      const runs = current_context_runs();
      const current_steps = runs.map(run => (
        finite_step(run?.maximum_update)
        ?? finite_step(run?.depth_maximum_update)
        ?? finite_step(run?.heatmap_maximum_update)
      )).filter(value => value !== null);
      if (current) {
        current.textContent = current_steps.length
          ? `(current step ${Math.max(...current_steps)})`
          : "";
        current.style.fontWeight = "400";
      }

      const retained = runs.map(run => {
        const minimum = finite_step(run?.depth_minimum_update);
        const maximum = finite_step(run?.depth_maximum_update);
        return minimum !== null && maximum !== null ? {minimum, maximum} : null;
      }).filter(Boolean);
      if (availability) {
        if (!retained.length) availability.textContent = "data available —";
        else if (app.workspace_mode === true) {
          const minimum = Math.max(...retained.map(value => value.minimum));
          const maximum = Math.min(...retained.map(value => value.maximum));
          availability.textContent = minimum <= maximum
            ? `data available ${minimum}–${maximum}`
            : "data available —";
        } else {
          availability.textContent = `data available ${retained[0].minimum}–${retained[0].maximum}`;
        }
      }
    };

    const selected_run_state = () => {
      const run = current_context_runs()[0] || app.current_status;
      if (!run) return "unknown";
      try { return display_run_state(run); }
      catch (_error) { return String(run.run_state || "unknown"); }
    };

    const placeholder_message = chart_name => {
      const key = context_key();
      if (key && loading_contexts.has(key)) return "Loading weight curves…";
      const figure = app.figures?.depth?.[chart_name];
      if (figure) return null;
      const runs = current_context_runs();
      const range = selected_range();
      const state = selected_run_state();
      const running = state === "running" || state === "preparing";

      if (range && runs.length) {
        const current_steps = runs.map(run => (
          finite_step(run?.maximum_update)
          ?? finite_step(run?.depth_maximum_update)
        )).filter(value => value !== null);
        const retained = runs.map(run => ({
          minimum: finite_step(run?.depth_minimum_update),
          maximum: finite_step(run?.depth_maximum_update),
        })).filter(value => value.minimum !== null && value.maximum !== null);

        if (retained.length && retained.every(value => range.maximum < value.minimum)) {
          return `Selected steps ${range.minimum}–${range.maximum} are no longer retained.`;
        }
        if (running && current_steps.length && Math.max(...current_steps) < range.minimum) {
          return `Curves will be displayed when step ${range.minimum} is reached.`;
        }
        return running
          ? `Waiting for a recorded weight snapshot in steps ${range.minimum}–${range.maximum}.`
          : `No recorded weight snapshots in steps ${range.minimum}–${range.maximum}.`;
      }

      const snapshot_count = Math.max(0, ...runs.map(run => Number(run?.depth_snapshot_count || 0)));
      if (snapshot_count > 0) return "Weight curves unavailable.";
      return running ? "Waiting for the first weight snapshot." : "No recorded weight snapshots.";
    };

    const reconcile_placeholders = () => {
      for (const chart_name of weight_chart_names) {
        const placeholder = by_id(`${chart_name}_placeholder`);
        if (!placeholder) continue;
        const message = placeholder_message(chart_name);
        if (message === null) {
          placeholder.hidden = true;
          placeholder.classList?.remove?.("instra-step-window-placeholder");
        } else {
          placeholder.textContent = message;
          placeholder.hidden = false;
          placeholder.classList?.remove?.("instra-step-window-placeholder");
        }
      }
    };

    const sync_editor_controls = ({load_values = false} = {}) => {
      if (by_id("chart_settings_overlay")?.hidden) return;
      const chart_name = app.axis_chart_name;
      if (!weight_chart_set.has(chart_name)) return;
      const current = by_id("chart_current_weights_only");
      const join = by_id("chart_join_with_line_segments");
      const group_editor = by_id("weights_group_scale_field")?.hidden === false;
      const inherit = !group_editor && by_id("chart_inherit_weights_group")?.checked === true;
      if (load_values) {
        if (current) current.checked = effective_flag(chart_name, "current_weights_only");
        if (join) join.checked = effective_flag(chart_name, "join_with_line_segments");
      }
      if (current) current.disabled = false;
      if (join) join.disabled = false;
      const current_note = by_id("chart_current_weights_only_field")?.querySelector?.("small");
      const join_note = by_id("chart_join_with_line_segments_field")?.querySelector?.("small");
      const note = group_editor
        ? "Applies to all six weight charts in this view unless a chart overrides it."
        : inherit
          ? "Inherited from Weights group settings; changing this setting creates a chart override."
          : "Overrides Weights group settings for this chart.";
      if (current_note) current_note.textContent = note;
      if (join_note) join_note.textContent = note;
    };

    // Neutralise the legacy controller's one global range. The visible/public range
    // below is context-scoped; leaving the old closure populated is what caused one
    // live run's future window to blank every historical run.
    if (legacy_clear_step_range) {
      refresh_suppression += 1;
      const saved_refresh = refresh_current_run;
      refresh_current_run = () => undefined;
      try { legacy_clear_step_range(); }
      finally {
        refresh_current_run = saved_refresh;
        refresh_suppression -= 1;
      }
    }

    const public_step_filter = {
      active: () => selected_range() !== null,
      signature: () => {
        const range = selected_range();
        return range ? `${context_key()}:${range.minimum}:${range.maximum}` : `${context_key()}:default`;
      },
      request_range: selected_range,
    };
    window.__instra_weight_step_filter = public_step_filter;

    const set_context_range = (minimum, maximum, {user = true, refresh = true} = {}) => {
      const resolved_minimum = finite_step(minimum);
      const resolved_maximum = finite_step(maximum);
      if (resolved_minimum === null || resolved_maximum === null || resolved_maximum < resolved_minimum) {
        throw new Error("weight step range must contain non-negative whole steps with end >= start");
      }
      const state = state_for_context();
      if (!state) return false;
      const changed = !state.range
        || state.range.minimum !== resolved_minimum
        || state.range.maximum !== resolved_maximum;
      state.range = {minimum: resolved_minimum, maximum: resolved_maximum};
      if (user) state.user_overridden = true;
      sync_header();
      reconcile_placeholders();
      if (changed && refresh) {
        invalidate_depth();
        refresh_current_run();
      }
      return changed;
    };

    const clear_context_range = ({user = true, refresh = true} = {}) => {
      const state = state_for_context();
      if (!state) return false;
      const changed = state.range !== null;
      state.range = null;
      if (user) state.user_overridden = true;
      sync_header();
      reconcile_placeholders();
      if (changed && refresh) {
        invalidate_depth();
        refresh_current_run();
      }
      return changed;
    };

    legacy_step_api.selected_step_range = selected_range;
    legacy_step_api.set_step_range = (minimum, maximum) => (
      set_context_range(minimum, maximum, {user: true, refresh: true})
    );
    legacy_step_api.clear_step_range = () => clear_context_range({user: true, refresh: true});
    legacy_step_api.global_flags = () => ({
      current_weights_only: effective_flag(app.axis_chart_name || weight_chart_names[0], "current_weights_only"),
      join_with_line_segments: effective_flag(app.axis_chart_name || weight_chart_names[0], "join_with_line_segments"),
    });
    legacy_step_api.set_global_flags = () => false;

    const apply_range_from_header = event => {
      const target = event.target;
      const apply = event.type === "click" && target?.closest?.("#weight_step_apply");
      const enter = (
        event.type === "keydown"
        && event.key === "Enter"
        && target?.matches?.("#weight_step_from, #weight_step_to")
      );
      const whole = event.type === "click" && target?.closest?.("#weight_step_whole_range");
      if (!apply && !enter && !whole) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      if (whole) {
        clear_context_range({user: true, refresh: true});
        return;
      }
      const minimum = finite_step(by_id("weight_step_from")?.value);
      const maximum = finite_step(by_id("weight_step_to")?.value);
      if (minimum === null || maximum === null || maximum < minimum) {
        show_toast("Enter a valid whole-number start and end step.");
        return;
      }
      set_context_range(minimum, maximum, {user: true, refresh: true});
    };
    window.addEventListener("click", apply_range_from_header, true);
    window.addEventListener("keydown", apply_range_from_header, true);

    // The two key Weights toggles are always directly editable. If an individual
    // chart was inheriting its group defaults, the first edit explicitly turns
    // inheritance off so the canonical group-settings owner persists an override.
    window.addEventListener("click", event => {
      const target = event.target;
      if (!target?.matches?.("#chart_current_weights_only, #chart_join_with_line_segments")) return;
      if (by_id("chart_settings_overlay")?.hidden || !weight_chart_set.has(app.axis_chart_name)) return;
      const desired = target.checked === true;
      const group_editor = by_id("weights_group_scale_field")?.hidden === false;
      if (!group_editor) {
        const inherit = by_id("chart_inherit_weights_group");
        if (inherit?.checked) inherit.checked = false;
      }
      queueMicrotask(() => {
        target.checked = desired;
        sync_editor_controls({load_values: false});
        if (typeof schedule_chart_settings_preview === "function") schedule_chart_settings_preview();
      });
    }, true);

    const base_sync_chart_setting_outputs = sync_chart_setting_outputs;
    sync_chart_setting_outputs = function() {
      const editing = !by_id("chart_settings_overlay")?.hidden && weight_chart_set.has(app.axis_chart_name);
      const current = by_id("chart_current_weights_only");
      const join = by_id("chart_join_with_line_segments");
      const current_checked = editing && current ? current.checked : null;
      const join_checked = editing && join ? join.checked : null;
      const result = base_sync_chart_setting_outputs();
      if (editing) {
        if (current && current_checked !== null) current.checked = current_checked;
        if (join && join_checked !== null) join.checked = join_checked;
        sync_editor_controls({load_values: false});
      }
      return result;
    };

    const base_populate_chart_settings_form = populate_chart_settings_form;
    populate_chart_settings_form = function(chart_name, supplied = null) {
      const result = base_populate_chart_settings_form(chart_name, supplied);
      if (weight_chart_set.has(chart_name)) sync_editor_controls({load_values: true});
      return result;
    };

    const base_open_chart_settings = open_chart_settings;
    open_chart_settings = function(chart_name) {
      const result = base_open_chart_settings(chart_name);
      if (weight_chart_set.has(chart_name)) queueMicrotask(() => sync_editor_controls({load_values: true}));
      return result;
    };

    by_id("chart_inherit_weights_group")?.addEventListener("change", () => {
      queueMicrotask(() => sync_editor_controls({load_values: true}));
    });

    const base_render_figures = render_figures;
    render_figures = async function() {
      const result = await base_render_figures();
      sync_header();
      reconcile_placeholders();
      return result;
    };

    const base_render_run_heading = render_run_heading;
    render_run_heading = function() {
      const result = base_render_run_heading();
      sync_header();
      return result;
    };

    const base_refresh_current_run = refresh_current_run;
    refresh_current_run = async function() {
      if (refresh_suppression > 0) return;
      const key = context_key();
      const need_loading = !app.figures?.depth || Object.keys(app.figures.depth || {}).length === 0;
      if (key && need_loading && !app.refresh_in_flight) {
        loading_contexts.add(key);
        reconcile_placeholders();
      }
      try {
        return await base_refresh_current_run();
      } finally {
        if (key) loading_contexts.delete(key);
        const seeded = seed_configured_range();
        if (seeded) {
          invalidate_depth();
          if (!app.refresh_in_flight) queueMicrotask(() => refresh_current_run());
        }
        sync_header();
        reconcile_placeholders();
      }
    };

    const base_select_run = select_run;
    select_run = function(run_identifier_value, options = {}) {
      if (String(run_identifier_value) === String(app.current_run_id || "")) {
        return base_select_run(run_identifier_value, options);
      }
      const saved_refresh = refresh_current_run;
      let refresh_requested = false;
      refresh_current_run = () => { refresh_requested = true; };
      let result;
      try {
        result = base_select_run(run_identifier_value, options);
      } finally {
        refresh_current_run = saved_refresh;
      }
      seed_configured_range();
      sync_header();
      const key = context_key();
      if (key) loading_contexts.add(key);
      reconcile_placeholders();
      if (refresh_requested || app.current_run_id) saved_refresh();
      return result;
    };

    // A save may change current-only and therefore the optimal server request. The
    // older global writers are ignored; after canonical group/override persistence
    // has completed, invalidate exactly the depth family and refresh once.
    window.addEventListener("click", event => {
      if (!event.target.closest?.("#save_chart_settings")) return;
      if (!weight_chart_set.has(app.axis_chart_name)) return;
      setTimeout(() => {
        sync_editor_controls({load_values: true});
        invalidate_depth();
        refresh_current_run();
      }, 0);
    }, true);

    window.__instra_weight_stability_final = Object.freeze({
      context_key,
      selected_range,
      set_range: (minimum, maximum) => set_context_range(minimum, maximum, {user: true, refresh: true}),
      clear_range: () => clear_context_range({user: true, refresh: true}),
      effective: chart_name => ({
        current_weights_only: effective_flag(chart_name, "current_weights_only"),
        join_with_line_segments: effective_flag(chart_name, "join_with_line_segments"),
      }),
      placeholder_message,
      reconcile_placeholders,
      seed_configured_range,
    });

    sync_header();
    reconcile_placeholders();

    // Any fetch that began before this dependency-gated owner installed may have
    // used the superseded global range/flags. Let it finish, then replace that one
    // payload with a correctly scoped request. This is bounded and runs once.
    let initial_refresh_attempts = 0;
    const refresh_after_legacy_settles = () => {
      initial_refresh_attempts += 1;
      if (app.refresh_in_flight && initial_refresh_attempts < 80) {
        setTimeout(refresh_after_legacy_settles, 50);
        return;
      }
      if (!app.current_run_id) return;
      invalidate_depth();
      const key = context_key();
      if (key) loading_contexts.add(key);
      reconcile_placeholders();
      refresh_current_run();
    };
    setTimeout(refresh_after_legacy_settles, 0);
    return true;
  };

  if (install()) return;
  let attempts = 0;
  const timer = setInterval(() => {
    attempts += 1;
    if (install() || attempts >= 240) clearInterval(timer);
  }, 25);
});
// ^^^ THOG
