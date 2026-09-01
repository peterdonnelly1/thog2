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
    const gradient_storage_key = "thog2_local_weight_gradient_v1";
    const legacy_step_api = window.__instra_weight_controls_v2;
    const legacy_clear_step_range = typeof legacy_step_api.clear_step_range === "function"
      ? legacy_step_api.clear_step_range.bind(legacy_step_api)
      : null;
    const selection_api = window.__instra_matched_weight_selection;
    const range_by_context = new Map();
    const loading_contexts = new Set();
    let gradient_enabled = localStorage.getItem(gradient_storage_key) === "true";
    let refresh_suppression = 0;
    let editor_draft = null;
    let last_header_context = "";

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
      if (app.workspace_mode === true) {
        const status = app.current_status && typeof app.current_status === "object"
          ? app.current_status
          : null;
        const selected = String(app.current_run_id || "");
        return visible_workspace_runs().map(run => {
          if (!status || !selected || run_id(run) !== selected || run_id(status) !== selected) {
            return run;
          }
          return {
            ...run,
            ...status,
            configuration: {
              ...(run.configuration || {}),
              ...(status.configuration || {}),
            },
          };
        });
      }
      const run = typeof current_run === "function" ? current_run() : null;
      const status = app.current_status && typeof app.current_status === "object"
        ? app.current_status
        : null;
      const selected = String(app.current_run_id || "");
      const status_matches = Boolean(status && selected && run_id(status) === selected);
      if (!run) return status_matches ? [status] : [];
      if (!status_matches || run_id(run) !== selected) return [run];
      return [{
        ...run,
        ...status,
        configuration: {
          ...(run.configuration || {}),
          ...(status.configuration || {}),
        },
      }];
    };

    const capture_bounds_for_run = run => {
      const configuration = run?.configuration || {};
      const minimum = finite_step(
        configuration.instrumentation__depth_weight_curves__start_step
      );
      const maximum = finite_step(
        configuration.instrumentation__depth_weight_curves__end_step
      );
      return {
        minimum,
        maximum,
        present: minimum !== null || maximum !== null,
      };
    };

    const configured_capture_range = () => {
      const runs = current_context_runs();
      if (!runs.length) return null;
      const bounds = runs.map(capture_bounds_for_run);
      if (!bounds.some(value => value.present)) return null;
      const minima = bounds.map(value => value.minimum).filter(value => value !== null);
      const maxima = bounds.map(value => value.maximum).filter(value => value !== null);
      return {
        minimum: minima.length ? Math.max(...minima) : null,
        maximum: maxima.length ? Math.min(...maxima) : null,
        present: true,
      };
    };

    const available_range = () => {
      const runs = current_context_runs();
      if (!runs.length) return null;
      let rendered_ranges = null;
      const rendered_range_for_run = identifier => {
        if (rendered_ranges === null) {
          rendered_ranges = new Map();
          const single_identifier = runs.length === 1 ? run_id(runs[0]) : "";
          for (const chart_name of weight_chart_names) {
            let figure = null;
            try { figure = figure_for_chart(chart_name); }
            catch (_error) { figure = app.figures?.depth?.[chart_name] || null; }
            for (const trace of figure?.data || []) {
              const trace_identifier = String(
                trace?.meta?.instra_workspace_run_id || single_identifier
              );
              if (!trace_identifier) continue;
              let update = null;
              try { update = finite_step(trace_optimizer_update(trace)); }
              catch (_error) { update = null; }
              if (update === null) continue;
              const prior = rendered_ranges.get(trace_identifier);
              rendered_ranges.set(trace_identifier, prior
                ? {
                    minimum: Math.min(prior.minimum, update),
                    maximum: Math.max(prior.maximum, update),
                  }
                : {minimum: update, maximum: update});
            }
          }
        }
        return rendered_ranges.get(identifier) || null;
      };
      const ranges = [];
      for (const run of runs) {
        const identifier = run_id(run);
        const stored_minimum = finite_step(run?.depth_minimum_update);
        const stored_maximum = finite_step(run?.depth_maximum_update);
        const rendered = stored_minimum === null || stored_maximum === null
          ? rendered_range_for_run(identifier)
          : null;
        const retained_minimum = stored_minimum
          ?? rendered?.minimum
          ?? null;
        const retained_maximum = stored_maximum
          ?? rendered?.maximum
          ?? null;
        const configured = capture_bounds_for_run(run);
        const minimum = retained_minimum === null
          ? null
          : Math.max(retained_minimum, configured.minimum ?? retained_minimum);
        const maximum = retained_maximum === null
          ? null
          : Math.min(retained_maximum, configured.maximum ?? retained_maximum);
        if (minimum === null || maximum === null || maximum < minimum) return null;
        ranges.push({minimum, maximum});
      }
      const minimum = Math.max(...ranges.map(range => range.minimum));
      const maximum = Math.min(...ranges.map(range => range.maximum));
      return minimum <= maximum ? {minimum, maximum} : null;
    };

    const state_for_context = (key = context_key()) => {
      if (!key) return null;
      if (!range_by_context.has(key)) {
        range_by_context.set(key, {mode: "settings", range: null, user_selected: false});
      }
      return range_by_context.get(key);
    };

    const seed_configured_range = () => {
      const state = state_for_context();
      if (!state || state.user_selected) return false;
      const mode = configured_capture_range()?.present ? "whole" : "settings";
      const changed = state.mode !== mode || state.range !== null;
      state.mode = mode;
      state.range = null;
      return changed;
    };

    const selected_range = () => {
      const state = state_for_context();
      if (state?.mode === "custom" && state.range) return {...state.range};
      const available = available_range();
      if (!state || !available || state.mode === "settings") return null;
      if (state.mode === "latest") {
        // Workspace latest is deliberately run-relative. Each per-run family
        // request uses current_only=1 and the merged payload is reduced again by
        // run identity below, so unequal final capture steps remain visible.
        if (app.workspace_mode === true) return null;
        return {minimum: available.maximum, maximum: available.maximum};
      }
      if (state.mode === "custom" && state.range) {
        return {minimum: state.range.minimum, maximum: state.range.maximum};
      }
      return available;
    };

    const selected_range_mode = () => state_for_context()?.mode || "whole";

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

    const explicit_flag_for_scope = (scope, chart_name, field) => {
      const overrides = json_store(override_storage_key);
      const override_scope = `${scope}:${chart_name}`;
      const override = overrides[override_scope];
      if (override && typeof override === "object" && own(override, field)) {
        return override[field] === true;
      }
      const legacy_store = field === "current_weights_only"
        ? app.weight_current_only
        : app.weight_join_with_line_segments;
      if (own(legacy_store, override_scope)) return legacy_store[override_scope] === true;
      const group = json_store(group_storage_key)[scope];
      if (group && typeof group === "object" && own(group, field)) return group[field] === true;
      return null;
    };

    const inherited_workspace_join = (chart_name, identifier) => (
      explicit_flag_for_scope(`run:${identifier}`, chart_name, "join_with_line_segments") === true
    );

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
      if (configured_capture_range()?.present === true) {
        normalized.current_weights_only = false;
        // Instrumentation owns the recorded step window.  A persisted Instra
        // history count/window may narrow runs without capture bounds, but it
        // must never silently discard part of an explicit capture window.
        normalized.max_snapshots = 0;
        normalized.snapshot_window_mode = "rolling";
      }
      return normalized;
    };
    normalize_chart_settings = final_normalize_chart_settings;

    // Settings mode is not an explicit request range, but the header should still
    // describe the curves those settings display. Derive that descriptive range
    // from actual retained trace updates so sparse capture cadences stay honest.
    const settings_display_range = () => {
      const available = available_range();
      if (!available) return null;
      const ranges = [];
      for (const chart_name of weight_chart_names) {
        let figure = null;
        try { figure = figure_for_chart(chart_name); }
        catch (_error) { figure = app.figures?.depth?.[chart_name] || null; }
        const updates = [...new Set((figure?.data || []).map(trace => {
          try { return finite_step(trace_optimizer_update(trace)); }
          catch (_error) { return null; }
        }).filter(value => value !== null))].sort((left, right) => left - right);
        const settings = final_normalize_chart_settings(chart_name);
        if (settings.current_weights_only === true) {
          ranges.push({minimum: available.maximum, maximum: available.maximum});
          continue;
        }
        const count = Math.max(0, finite_step(settings.max_snapshots) ?? 0);
        if (!updates.length || count < 1 || count >= updates.length) {
          ranges.push(available);
          continue;
        }
        const selected = settings.snapshot_window_mode === "from_zero"
          ? updates.slice(0, count)
          : updates.slice(-count);
        ranges.push({minimum: selected[0], maximum: selected[selected.length - 1]});
      }
      if (!ranges.length) return available;
      return {
        minimum: Math.min(...ranges.map(range => range.minimum)),
        maximum: Math.max(...ranges.map(range => range.maximum)),
      };
    };

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
      const viewer = window.__instra_weight_viewer_selection;
      const selection = viewer?.selection?.() || selection_api.selection?.() || {};
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
      const capture_limited = configured_capture_range()?.present === true;
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
        if (candidate === chart_name) {
          normalized.current_weights_only = coordinate_selected;
          if (capture_limited) {
            normalized.max_snapshots = 0;
            normalized.snapshot_window_mode = "rolling";
          }
        }
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
      const run_relative_latest = (
        app.workspace_mode === true && selected_range_mode() === "latest"
      );
      if (run_relative_latest || (!capture_limited && settings.current_weights_only === true)) {
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
      if (
        app.workspace_mode === true
        && explicit_flag_for_scope("workspace", chart_name, "join_with_line_segments") === null
        && typeof apply_thog_line_segments === "function"
      ) {
        prepared.data = (prepared.data || []).flatMap(trace => {
          const identifier = String(trace?.meta?.instra_workspace_run_id || "");
          if (!identifier || !inherited_workspace_join(chart_name, identifier)) return [trace];
          const one = {data: [trace]};
          apply_thog_line_segments(one);
          return one.data || [];
        });
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

    const ensure_final_step_controls = () => {
      const whole = by_id("weight_step_whole_range");
      if (!whole) return false;
      let latest = by_id("weight_step_latest");
      if (!latest) {
        latest = document.createElement("button");
        latest.id = "weight_step_latest";
        latest.type = "button";
        latest.className = "weight-step-button";
        latest.textContent = "latest step";
        latest.title = "Show the latest retained weight snapshot";
        whole.insertAdjacentElement("afterend", latest);
      }
      let gradient = by_id("weight_step_gradient");
      if (!gradient) {
        gradient = document.createElement("button");
        gradient.id = "weight_step_gradient";
        gradient.type = "button";
        gradient.className = "weight-step-button";
        gradient.textContent = "gradient";
        gradient.title = "Colour retained curves from lightest at the earliest step to darker than the run colour at the latest step";
        latest.insertAdjacentElement("afterend", gradient);
      }
      let overlap = by_id("weight_step_overlapping_range");
      if (!overlap) {
        overlap = document.createElement("button");
        overlap.id = "weight_step_overlapping_range";
        overlap.type = "button";
        overlap.className = "weight-step-button";
        overlap.textContent = "show overlapping range";
        overlap.title = "Show the optimizer-step range retained by every visible Workspace run";
        gradient.insertAdjacentElement("afterend", overlap);
      }
      let error = by_id("weight_step_range_error");
      if (!error) {
        error = document.createElement("span");
        error.id = "weight_step_range_error";
        error.className = "weight-step-range-error";
        error.setAttribute("role", "status");
        error.hidden = true;
        overlap.insertAdjacentElement("afterend", error);
      }
      return true;
    };

    const show_range_error = message => {
      ensure_final_step_controls();
      const error = by_id("weight_step_range_error");
      if (!error) return;
      error.textContent = String(message || "");
      error.hidden = !message;
    };

    const write_step_input = (input, value) => {
      if (!input) return;
      const descriptor = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value");
      if (descriptor?.set) descriptor.set.call(input, String(value));
      else input.value = String(value);
    };

    const redraw_weight_figures = async () => {
      const final_redraw = window.__instra_weight_range_interaction_final?.redraw_mounted;
      if (typeof final_redraw === "function") {
        await final_redraw();
        return;
      }
      await render_figures();
      const rendered_context = context_key();
      if (!rendered_context || typeof render_plot !== "function") return;
      const fallback_jobs = [];
      for (const chart_name of weight_chart_names) {
        if (app.figures?.depth?.[chart_name]) continue;
        const mount = by_id(`${chart_name}_plot`);
        if (
          mount?.dataset?.plotReady !== "true"
          || mount.dataset.instraWeightContext !== rendered_context
          || !mount.__instraWeightFigure
        ) continue;
        const placeholder = by_id(`${chart_name}_placeholder`);
        if (placeholder) placeholder.hidden = true;
        fallback_jobs.push(render_plot(mount, mount.__instraWeightFigure, chart_name));
      }
      await Promise.all(fallback_jobs);
    };

    const sync_header = () => {
      ensure_final_step_controls();
      const next_context = context_key();
      if (next_context !== last_header_context) {
        last_header_context = next_context;
        show_range_error("");
      }
      const range = selected_range();
      const mode = selected_range_mode();
      const displayed_range = range || (mode === "settings" ? settings_display_range() : null);
      const available = available_range();
      const from = by_id("weight_step_from");
      const to = by_id("weight_step_to");
      const whole = by_id("weight_step_whole_range");
      const latest = by_id("weight_step_latest");
      const gradient = by_id("weight_step_gradient");
      const overlap = by_id("weight_step_overlapping_range");
      const current = by_id("weight_step_current");
      const availability = by_id("weight_step_availability");
      const controls = by_id("weight_step_group_controls");
      const configured = configured_capture_range();
      if (from) {
        from.value = displayed_range
          ? String(displayed_range.minimum)
          : (configured?.minimum !== null && configured?.minimum !== undefined
              ? String(configured.minimum)
              : "");
      }
      if (to) {
        to.value = displayed_range
          ? String(displayed_range.maximum)
          : (configured?.maximum !== null && configured?.maximum !== undefined
              ? String(configured.maximum)
              : "");
      }
      if (from && available) {
        from.min = String(available.minimum);
        from.max = String(available.maximum);
      }
      if (to && available) {
        to.min = String(available.minimum);
        to.max = String(available.maximum);
      }
      if (whole) whole.disabled = !available;
      if (latest) latest.disabled = !available;
      if (overlap) {
        overlap.hidden = app.workspace_mode !== true;
        overlap.disabled = false;
      }
      whole?.setAttribute("aria-pressed", String(mode === "whole"));
      latest?.setAttribute("aria-pressed", String(mode === "latest"));
      gradient?.setAttribute("aria-pressed", String(gradient_enabled));
      by_id("weight_step_initial_values")?.setAttribute("aria-pressed", String(
        mode === "custom" && range?.minimum === 0 && range?.maximum === 0
      ));
      by_id("weight_step_one")?.setAttribute("aria-pressed", String(
        mode === "custom" && range?.minimum === 1 && range?.maximum === 1
      ));
      controls?.classList?.toggle?.("active", range !== null);
      if (whole) whole.title = "Show every retained weight snapshot in this view";

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

      if (availability) {
        const configured_text = configured
          ? `capture window ${configured.minimum ?? "first"}–${configured.maximum ?? "latest"}`
          : "";
        availability.textContent = configured_text;
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
      const figure = app.figures?.depth?.[chart_name];
      if (figure) return null;
      const mount = by_id(`${chart_name}_plot`);
      const mounted_for_context = Boolean(
        key
        && mount?.dataset?.plotReady === "true"
        && mount.dataset.instraWeightContext === key
        && mount.dataset.instraWeightView === window.__instra_weight_step_filter?.signature?.()
        && Array.isArray(mount.data)
        && mount.data.length > 0
      );
      if (mounted_for_context) return null;
      if (key && loading_contexts.has(key)) return "Loading weight curves…";
      const runs = current_context_runs();
      const range = selected_range();
      const state = selected_run_state();
      const running = ["preparing", "recording", "monitoring"].includes(state);
      const configured = configured_capture_range();

      if (!range && configured?.present && running && configured.minimum !== null) {
        return `Curves will be displayed when capture step ${configured.minimum} is reached.`;
      }

      if (range && runs.length) {
        if (range.minimum === 0 && range.maximum === 0) {
          return "No recorded initial weights (step 0) in this view.";
        }
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
      if (snapshot_count > 0) return "Loading weight curves…";
      if (app.refresh_in_flight) return "Loading weight curves…";
      return running ? "Loading weight curves…" : "No recorded weight snapshots.";
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

    const editor_is_group = () => by_id("weights_group_scale_field")?.hidden === false;

    const draft_matches_editor = () => Boolean(
      editor_draft
      && editor_draft.chart_name === app.axis_chart_name
      && editor_draft.group_editor === editor_is_group()
    );

    const apply_editor_draft = () => {
      if (!draft_matches_editor()) return;
      const current = by_id("chart_current_weights_only");
      const join = by_id("chart_join_with_line_segments");
      if (editor_draft.current_weights_only !== null && current) {
        current.checked = editor_draft.current_weights_only;
      }
      if (editor_draft.join_with_line_segments !== null && join) {
        join.checked = editor_draft.join_with_line_segments;
      }
      if (editor_draft.force_override) {
        const inherit = by_id("chart_inherit_weights_group");
        if (inherit) inherit.checked = false;
      }
    };

    const sync_editor_controls = ({load_values = false} = {}) => {
      if (by_id("chart_settings_overlay")?.hidden) return;
      const chart_name = app.axis_chart_name;
      if (!weight_chart_set.has(chart_name)) return;
      const current = by_id("chart_current_weights_only");
      const join = by_id("chart_join_with_line_segments");
      const group_editor = editor_is_group();
      const inherit = !group_editor && by_id("chart_inherit_weights_group")?.checked === true;
      const capture_limited = configured_capture_range()?.present === true;
      if (load_values && !group_editor && !draft_matches_editor()) {
        if (current) current.checked = effective_flag(chart_name, "current_weights_only");
        if (join) join.checked = effective_flag(chart_name, "join_with_line_segments");
      }
      apply_editor_draft();
      if (current && capture_limited) current.checked = false;
      if (current) current.disabled = capture_limited;
      if (join) join.disabled = false;
      const maximum = by_id("chart_max_snapshots");
      const window_mode = by_id("chart_snapshot_window_mode");
      if (maximum) maximum.disabled = capture_limited || current?.checked === true;
      if (window_mode) {
        const all_snapshots = Number(maximum?.value || 0) <= 0;
        window_mode.disabled = capture_limited || current?.checked === true || all_snapshots;
      }
      const current_note = by_id("chart_current_weights_only_field")?.querySelector?.("small");
      const join_note = by_id("chart_join_with_line_segments_field")?.querySelector?.("small");
      const note = capture_limited
        ? "The run's configured capture window is authoritative; use the header controls to narrow it."
        : group_editor
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
        return range
          ? `${context_key()}:${selected_range_mode()}:${range.minimum}:${range.maximum}`
          : `${context_key()}:settings`;
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
      const changed = state.mode !== "custom"
        || !state.range
        || state.range.minimum !== resolved_minimum
        || state.range.maximum !== resolved_maximum;
      state.mode = "custom";
      state.range = {minimum: resolved_minimum, maximum: resolved_maximum};
      state.user_selected = user;
      sync_header();
      reconcile_placeholders();
      if (changed && refresh) {
        invalidate_depth();
        refresh_current_run();
      }
      return changed;
    };

    const set_context_mode = (mode, {refresh = true, user = true} = {}) => {
      const state = state_for_context();
      if (!state) return false;
      const normalized = ["whole", "latest", "settings"].includes(mode) ? mode : "whole";
      const changed = state.mode !== normalized || state.range !== null;
      window.__instra_clear_weight_step_input_drafts?.();
      state.mode = normalized;
      state.range = null;
      state.user_selected = user;
      sync_header();
      reconcile_placeholders();
      if (changed && refresh) {
        invalidate_depth();
        refresh_current_run();
      }
      return changed;
    };

    const clear_context_range = ({refresh = true} = {}) => (
      set_context_mode("whole", {refresh, user: true})
    );

    legacy_step_api.selected_step_range = selected_range;
    legacy_step_api.set_step_range = (minimum, maximum) => (
      set_context_range(minimum, maximum, {user: true, refresh: true})
    );
    legacy_step_api.clear_step_range = () => clear_context_range({refresh: true});
    legacy_step_api.global_flags = () => ({
      current_weights_only: effective_flag(app.axis_chart_name || weight_chart_names[0], "current_weights_only"),
      join_with_line_segments: effective_flag(app.axis_chart_name || weight_chart_names[0], "join_with_line_segments"),
    });
    legacy_step_api.set_global_flags = () => false;

    let last_edited_step_input = "";
    window.addEventListener("input", event => {
      if (!event.target?.matches?.("#weight_step_from, #weight_step_to")) return;
      last_edited_step_input = event.target.id;
      show_range_error("");
    }, true);

    const refresh_workspace_statuses = async () => {
      if (app.workspace_mode !== true) return;
      const results = await Promise.all(visible_workspace_runs().map(async run => {
        const identifier = run_id(run);
        if (!identifier) return null;
        try {
          return {
            identifier,
            status: await fetch_json(`/api/status?run=${encodeURIComponent(identifier)}`),
          };
        } catch (_error) {
          return null;
        }
      }));
      for (const result of results.filter(Boolean)) {
        const run = app.runs?.find(candidate => run_id(candidate) === result.identifier);
        if (run) {
          Object.assign(run, result.status, {
            configuration: {
              ...(run.configuration || {}),
              ...(result.status.configuration || {}),
            },
          });
        }
        if (String(app.current_run_id || "") === result.identifier) {
          app.current_status = {
            ...(app.current_status || {}),
            ...result.status,
            configuration: {
              ...(app.current_status?.configuration || {}),
              ...(result.status.configuration || {}),
            },
          };
        }
      }
    };

    const corrected_header_range = () => {
      const available = available_range();
      if (!available) return {error: "No retained weight steps are available."};
      const from = by_id("weight_step_from");
      const to = by_id("weight_step_to");
      let minimum = finite_step(from?.value);
      let maximum = finite_step(to?.value);
      const errors = [];

      if (minimum === null) {
        minimum = available.minimum;
        errors.push("'from' value must be a non-negative whole number");
      } else if (minimum < available.minimum) {
        minimum = available.minimum;
        errors.push(`'from' value cannot be less than ${available.minimum}`);
      } else if (minimum > available.maximum) {
        minimum = available.maximum;
        errors.push(`'from' value cannot be greater than ${available.maximum}`);
      }

      if (maximum === null) {
        maximum = available.maximum;
        errors.push("'to' value must be a non-negative whole number");
      } else if (maximum > available.maximum) {
        maximum = available.maximum;
        errors.push(`'to' value cannot be greater than ${available.maximum}`);
      } else if (maximum < available.minimum) {
        maximum = available.minimum;
        errors.push(`'to' value cannot be less than ${available.minimum}`);
      }

      if (minimum > maximum) {
        if (last_edited_step_input === "weight_step_from") {
          minimum = maximum;
          errors.push("'from' value cannot be greater than 'to'");
        } else {
          maximum = minimum;
          errors.push("'to' value cannot be less than 'from'");
        }
      }
      write_step_input(from, minimum);
      write_step_input(to, maximum);
      return {minimum, maximum, error: errors.join("; ")};
    };

    const apply_range_from_header = async event => {
      const target = event.target;
      const apply = event.type === "click" && target?.closest?.("#weight_step_apply");
      const enter = (
        event.type === "keydown"
        && event.key === "Enter"
        && target?.matches?.("#weight_step_from, #weight_step_to")
      );
      const whole = event.type === "click" && target?.closest?.("#weight_step_whole_range");
      const latest = event.type === "click" && target?.closest?.("#weight_step_latest");
      const gradient = event.type === "click" && target?.closest?.("#weight_step_gradient");
      const overlap = event.type === "click" && target?.closest?.("#weight_step_overlapping_range");
      if (!apply && !enter && !whole && !latest && !gradient && !overlap) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      if (gradient) {
        gradient_enabled = !gradient_enabled;
        localStorage.setItem(gradient_storage_key, String(gradient_enabled));
        sync_header();
        try { await redraw_weight_figures(); }
        catch (error) { show_range_error(`Weight gradient redraw failed: ${error.message}`); }
        return;
      }
      if (overlap) {
        overlap.disabled = true;
        overlap.setAttribute("aria-busy", "true");
        await refresh_workspace_statuses();
        const available = available_range();
        overlap.disabled = false;
        overlap.removeAttribute("aria-busy");
        if (app.workspace_mode !== true || !available) {
          show_range_error("No overlapping retained weight steps.");
          return;
        }
        show_range_error("");
        set_context_range(available.minimum, available.maximum, {user: true, refresh: true});
        return;
      }
      if (whole) {
        show_range_error("");
        set_context_mode("whole", {refresh: true});
        return;
      }
      if (latest) {
        show_range_error("");
        set_context_mode("latest", {refresh: true});
        return;
      }
      const corrected = corrected_header_range();
      if (corrected.error) {
        show_range_error(corrected.error);
        return;
      }
      show_range_error("");
      set_context_range(corrected.minimum, corrected.maximum, {user: true, refresh: true});
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
      const group_editor = editor_is_group();
      if (!draft_matches_editor()) {
        editor_draft = {
          chart_name: app.axis_chart_name,
          group_editor,
          current_weights_only: null,
          join_with_line_segments: null,
          force_override: false,
        };
      }
      if (target.id === "chart_current_weights_only") {
        editor_draft.current_weights_only = target.checked === true;
      } else {
        editor_draft.join_with_line_segments = target.checked === true;
      }
      if (!group_editor) editor_draft.force_override = true;
      apply_editor_draft();
      queueMicrotask(() => {
        apply_editor_draft();
        sync_editor_controls({load_values: false});
        if (typeof schedule_chart_settings_preview === "function") schedule_chart_settings_preview();
      });
      setTimeout(() => {
        apply_editor_draft();
        sync_editor_controls({load_values: false});
      }, 0);
    }, true);

    // The group editor repopulates its shared form after open_chart_settings has
    // scheduled the base preview. Re-render once from the settled form so both the
    // group and individual editors immediately show the selected retained coupling.
    window.addEventListener("click", event => {
      const opener = event.target?.closest?.(
        "#weights_group_settings_button, .chart-card[data-chart] .chart-settings-button"
      );
      if (!opener) return;
      setTimeout(() => {
        if (by_id("chart_settings_overlay")?.hidden) return;
        if (!weight_chart_set.has(app.axis_chart_name)) return;
        if (typeof schedule_chart_settings_preview === "function") {
          schedule_chart_settings_preview();
        }
      }, 0);
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
      editor_draft = null;
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
      const rendered_context = context_key();
      if (rendered_context) {
        for (const chart_name of weight_chart_names) {
          const mount = by_id(`${chart_name}_plot`);
          if (
            app.figures?.depth?.[chart_name]
            && mount?.dataset?.plotReady === "true"
            && Array.isArray(mount.data)
            && mount.data.length > 0
          ) {
            mount.dataset.instraWeightContext = rendered_context;
            mount.dataset.instraWeightView = window.__instra_weight_step_filter?.signature?.() || "";
            mount.__instraWeightFigure = app.figures.depth[chart_name];
          }
        }
      }
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
      if (seed_configured_range()) invalidate_depth();
      const key = context_key();
      const need_loading = !app.figures?.depth || Object.keys(app.figures.depth || {}).length === 0;
      if (key && need_loading && !app.refresh_in_flight) {
        loading_contexts.add(key);
        reconcile_placeholders();
      }
      try {
        return await base_refresh_current_run();
      } finally {
        const has_depth = Boolean(
          app.figures?.depth && Object.keys(app.figures.depth).length > 0
        );
        const known_snapshot_count = Math.max(
          0,
          ...current_context_runs().map(run => Number(run?.depth_snapshot_count || 0)),
        );
        const requested_snapshot_count = finite_step(
          app.figures?.weight_step_range?.snapshot_count
        );
        if (
          key
          && (has_depth || known_snapshot_count === 0 || requested_snapshot_count === 0)
        ) loading_contexts.delete(key);
        const reseeded = seed_configured_range();
        sync_header();
        reconcile_placeholders();
        if (reseeded) {
          invalidate_depth();
          queueMicrotask(() => refresh_current_run());
        }
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
      state_for_context();
      seed_configured_range();
      sync_header();
      const key = context_key();
      if (key) loading_contexts.add(key);
      reconcile_placeholders();
      if (refresh_requested || app.current_run_id) saved_refresh();
      return result;
    };

    // A save may change current-only and therefore the optimal server request. Make
    // the editor draft authoritative at capture time, before the canonical group/
    // override handler reads the form, then refresh once after persistence completes.
    window.addEventListener("click", event => {
      if (!event.target.closest?.("#save_chart_settings")) return;
      if (!weight_chart_set.has(app.axis_chart_name)) return;
      if (by_id("chart_settings_overlay")?.hidden) return;
      apply_editor_draft();
      const saved_current_only = by_id("chart_current_weights_only")?.checked === true;
      setTimeout(() => {
        editor_draft = null;
        sync_editor_controls({load_values: true});
        if (saved_current_only) {
          set_context_mode("settings", {refresh: false, user: false});
        } else if (selected_range_mode() === "settings") {
          set_context_mode("whole", {refresh: false, user: false});
        }
        invalidate_depth();
        refresh_current_run();
      }, 0);
    }, true);

    const style = document.createElement("style");
    style.id = "thog2_weight_stability_final_style";
    style.textContent = `
      .weight-step-range-error {
        color: #b42318;
        font-size: 10px;
        font-weight: 400;
        line-height: 1.2;
        margin-left: 2px;
        white-space: nowrap;
      }
      #weight_step_gradient { margin-left: 6px; }
      .weight-step-button[aria-busy="true"] {
        opacity: .62;
        cursor: progress;
      }
      .weight-step-button[aria-pressed="true"] {
        border-color: #1590a8;
        color: #0b6577;
        background: #edfafd;
      }
    `;
    document.head.appendChild(style);

    window.__instra_weight_stability_final = Object.freeze({
      context_key,
      available_range,
      selected_range,
      mode: selected_range_mode,
      sync_header,
      set_range: (minimum, maximum) => set_context_range(minimum, maximum, {user: true, refresh: true}),
      clear_range: () => clear_context_range({refresh: true}),
      show_whole: () => set_context_mode("whole", {refresh: true}),
      show_latest: () => set_context_mode("latest", {refresh: true}),
      show_settings: () => set_context_mode("settings", {refresh: true}),
      effective: chart_name => ({
        current_weights_only: effective_flag(chart_name, "current_weights_only"),
        join_with_line_segments: effective_flag(chart_name, "join_with_line_segments"),
      }),
      gradient_enabled: () => gradient_enabled,
      workspace_join_explicit: chart_name => (
        explicit_flag_for_scope("workspace", chart_name, "join_with_line_segments")
      ),
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
  const retry = () => {
    attempts += 1;
    if (install() || attempts >= 240) return;
    setTimeout(retry, 25);
  };
  setTimeout(retry, 0);
});
// ^^^ THOG
