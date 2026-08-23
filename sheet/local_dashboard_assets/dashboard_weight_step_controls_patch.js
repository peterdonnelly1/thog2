// vvv THOG
"use strict";

// Final owner for Weights header controls. The explicit step window is display-only:
// it requests only retained snapshots in [start, end], never changes trainer capture.
// Current-only and integer-line-segment mode are global across every weight chart,
// run and Workspace; individual chart editors cannot silently override those flags.
window.addEventListener("load", () => {
  setTimeout(() => {
    const weight_chart_names = [...depth_weight_chart_names];
    const weight_chart_set = new Set(weight_chart_names);
    const trajectory_scale_settings_key = "thog2_local_trajectory_scale_modes";
    const global_flags_storage_key = "thog2_local_weight_global_flags_v2";
    const legacy_group_storage_key = "thog2_local_weight_group_settings_v1";
    const default_history_capacity = 100;
    let selected_step_range = null;
    let coupling_save_in_flight = false;
    let refresh_serial = 0;

    const finite_integer = value => {
      if (value === null || value === undefined || value === "") return null;
      const numeric = Number(value);
      return Number.isInteger(numeric) ? numeric : null;
    };

    const finite_positive_integer = value => {
      const numeric = finite_integer(value);
      return numeric !== null && numeric > 0 ? numeric : null;
    };

    const read_global_flags = () => {
      const stored = load_json(global_flags_storage_key, null);
      if (stored && typeof stored === "object" && !Array.isArray(stored)) {
        return {
          current_weights_only: stored.current_weights_only === true,
          join_with_line_segments: stored.join_with_line_segments === true,
        };
      }
      const legacy = load_json(legacy_group_storage_key, {});
      const candidates = [];
      if (legacy && typeof legacy === "object" && !Array.isArray(legacy)) {
        const current_scope = `run:${String(app.current_run_id || "")}`;
        if (legacy[current_scope]) candidates.push(legacy[current_scope]);
        if (legacy.workspace) candidates.push(legacy.workspace);
        for (const value of Object.values(legacy)) {
          if (value && typeof value === "object" && !Array.isArray(value)) candidates.push(value);
        }
      }
      const candidate = candidates[0] || {};
      const migrated = {
        current_weights_only: candidate.current_weights_only === true,
        join_with_line_segments: candidate.join_with_line_segments === true,
      };
      save_json(global_flags_storage_key, migrated);
      return migrated;
    };

    let global_flags = read_global_flags();

    const step_filter_active = () => (
      finite_integer(selected_step_range?.minimum) !== null
      && finite_integer(selected_step_range?.maximum) !== null
    );

    const step_filter_signature = () => step_filter_active()
      ? `${selected_step_range.minimum}:${selected_step_range.maximum}`
      : "default";

    window.__instra_weight_step_filter = {
      active: step_filter_active,
      signature: step_filter_signature,
      request_range: () => step_filter_active() ? {...selected_step_range} : null,
    };

    const visible_workspace_runs = () => (
      app.workspace_mode === true && typeof window.__instra_workspace?.visible_runs === "function"
        ? window.__instra_workspace.visible_runs()
        : []
    );

    const run_id = run => {
      try { return String(run_identifier(run)); }
      catch (_error) { return String(run?.dashboard_run_id || run?.local_run_id || ""); }
    };

    const fresh_run = run => {
      if (!run) return null;
      const identifier = run_id(run);
      if (
        identifier
        && identifier === String(app.current_run_id || "")
        && app.current_status
        && typeof app.current_status === "object"
      ) {
        return {...run, ...app.current_status, configuration: run.configuration || app.current_status.configuration || {}};
      }
      return run;
    };

    const context_runs = () => {
      if (app.workspace_mode === true) return visible_workspace_runs().map(fresh_run).filter(Boolean);
      const run = fresh_run(current_run());
      return run ? [run] : [];
    };

    const configuration_value = (run, ...names) => {
      const configuration = run?.configuration && typeof run.configuration === "object"
        ? run.configuration
        : {};
      for (const name of names) {
        if (Object.prototype.hasOwnProperty.call(configuration, name)) return configuration[name];
      }
      return null;
    };

    const history_capacity_for_run = run => (
      finite_positive_integer(configuration_value(
        run,
        "instrumentation__depth_weight_curves__history_length",
        "depth_weight_curves__history_length",
      )) || default_history_capacity
    );

    const common_history_capacity = () => {
      const runs = context_runs();
      if (!runs.length) return default_history_capacity;
      return Math.min(...runs.map(history_capacity_for_run));
    };

    const current_step_for_run = run => (
      finite_integer(run?.maximum_update)
      ?? finite_integer(run?.depth_maximum_update)
      ?? finite_integer(run?.heatmap_maximum_update)
    );

    const current_step_bounds = () => {
      const values = context_runs().map(current_step_for_run).filter(value => value !== null);
      if (!values.length) return null;
      return {minimum: Math.min(...values), maximum: Math.max(...values)};
    };

    const retained_range_for_run = run => {
      if (!run || Number(run.depth_snapshot_count || 0) <= 0) return null;
      const minimum = finite_integer(run.depth_minimum_update);
      const maximum = finite_integer(run.depth_maximum_update);
      if (minimum === null || maximum === null || maximum < minimum) return null;
      return {minimum, maximum};
    };

    const available_step_range = () => {
      const runs = context_runs();
      if (!runs.length) return {available: false, reason: "no visible runs"};
      const ranges = runs.map(retained_range_for_run);
      if (ranges.some(range => range === null)) {
        return {available: false, reason: app.workspace_mode === true ? "no overlapping steps" : "no stored weight steps"};
      }
      const minimum = Math.max(...ranges.map(range => range.minimum));
      const maximum = Math.min(...ranges.map(range => range.maximum));
      return minimum <= maximum
        ? {available: true, minimum, maximum}
        : {available: false, reason: "no overlapping steps"};
    };

    const all_runs_before_selected_start = () => {
      if (!step_filter_active()) return false;
      const bounds = current_step_bounds();
      return Boolean(bounds && bounds.maximum < selected_step_range.minimum);
    };

    const selected_range_has_any_figure = () => weight_chart_names.some(chart_name => (
      Boolean(app.figures?.depth?.[chart_name])
    ));

    const invalidate_depth_view = () => {
      refresh_serial += 1;
      const serial = refresh_serial;
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
      apply_selected_range_placeholders();

      const attempt = () => {
        if (serial !== refresh_serial || !app.current_run_id) return;
        if (app.refresh_in_flight) {
          setTimeout(attempt, 75);
          return;
        }
        refresh_current_run();
      };
      attempt();
    };

    const set_global_flags = (next, {refresh = true} = {}) => {
      const normalized = {
        current_weights_only: next?.current_weights_only === true,
        join_with_line_segments: next?.join_with_line_segments === true,
      };
      const changed = (
        normalized.current_weights_only !== global_flags.current_weights_only
        || normalized.join_with_line_segments !== global_flags.join_with_line_segments
      );
      global_flags = normalized;
      save_json(global_flags_storage_key, global_flags);
      sync_global_weight_fields();
      if (changed && refresh) invalidate_depth_view();
      return changed;
    };

    // Final global semantics. A selected explicit step window temporarily behaves
    // like current-only for matched-coupling filtering, but the latest-only collapse
    // itself is bypassed below so every retained snapshot in [start,end] survives.
    const base_normalize_chart_settings_weight_v2 = normalize_chart_settings;
    normalize_chart_settings = function(chart_name, supplied = null) {
      const normalized = base_normalize_chart_settings_weight_v2(chart_name, supplied);
      if (!weight_chart_set.has(chart_name)) return normalized;
      normalized.current_weights_only = step_filter_active() ? true : global_flags.current_weights_only;
      normalized.join_with_line_segments = global_flags.join_with_line_segments;
      return normalized;
    };

    const base_retain_latest_weight_snapshots_weight_v2 = retain_latest_weight_snapshots;
    retain_latest_weight_snapshots = function(prepared) {
      if (step_filter_active()) return;
      return base_retain_latest_weight_snapshots_weight_v2(prepared);
    };

    if (typeof instra_enforce_workspace_latest_weights === "function") {
      const base_workspace_latest_weight_v2 = instra_enforce_workspace_latest_weights;
      instra_enforce_workspace_latest_weights = function(prepared) {
        if (step_filter_active()) return prepared;
        return base_workspace_latest_weight_v2(prepared);
      };
    }

    const sync_global_weight_fields = () => {
      const current = by_id("chart_current_weights_only");
      const join = by_id("chart_join_with_line_segments");
      if (current) current.checked = global_flags.current_weights_only;
      if (join) join.checked = global_flags.join_with_line_segments;
      const current_small = by_id("chart_current_weights_only_field")?.querySelector("small");
      const join_small = by_id("chart_join_with_line_segments_field")?.querySelector("small");
      if (current_small) current_small.textContent = "Global across all six weight charts and every run.";
      if (join_small) join_small.textContent = "Global across all six weight charts and every run.";
    };

    const chart_settings_is_weight_editor = () => (
      !by_id("chart_settings_overlay")?.hidden
      && weight_chart_set.has(app.axis_chart_name)
    );

    const base_open_chart_settings_weight_v2 = open_chart_settings;
    open_chart_settings = function(chart_name) {
      // A prior group-editor session may have left the Data tab hidden. Always
      // restore both tabs before opening an individual chart; the established group
      // editor deliberately hides Data again after this function returns.
      for (const button of document.querySelectorAll("[data-chart-settings-tab]")) button.hidden = false;
      return base_open_chart_settings_weight_v2(chart_name);
    };

    const base_populate_chart_settings_form_weight_v2 = populate_chart_settings_form;
    populate_chart_settings_form = function(chart_name, supplied = null) {
      const result = base_populate_chart_settings_form_weight_v2(chart_name, supplied);
      if (weight_chart_set.has(chart_name)) sync_global_weight_fields();
      return result;
    };

    const base_sync_chart_setting_outputs_weight_v2 = sync_chart_setting_outputs;
    sync_chart_setting_outputs = function() {
      const result = base_sync_chart_setting_outputs_weight_v2();
      if (!weight_chart_set.has(app.axis_chart_name)) return result;
      sync_global_weight_fields();
      const current = by_id("chart_current_weights_only");
      const join = by_id("chart_join_with_line_segments");
      if (current) current.disabled = false;
      if (join) join.disabled = false;
      return result;
    };

    // Capture global flags at the window before the older group-save handler uses
    // stopImmediatePropagation on the button itself. Synthetic hidden matched-weight
    // saves are ignored because their settings overlay is not open.
    window.addEventListener("click", event => {
      const button = event.target.closest?.("#save_chart_settings");
      if (!button || !chart_settings_is_weight_editor()) return;
      set_global_flags({
        current_weights_only: by_id("chart_current_weights_only")?.checked === true,
        join_with_line_segments: by_id("chart_join_with_line_segments")?.checked === true,
      });
    }, true);

    const install_step_controls = () => {
      const header = by_id("coefficients_chart_group")?.querySelector(":scope > .chart-group-header");
      const coupling_controls = by_id("weight_index_group_controls");
      if (!header || !coupling_controls) return false;
      let controls = by_id("weight_step_group_controls");
      if (controls) return true;

      controls = document.createElement("div");
      controls.id = "weight_step_group_controls";
      controls.className = "weight-step-group-controls";
      controls.setAttribute("role", "group");
      controls.setAttribute("aria-label", "Weight curve optimizer-step window");

      const current = document.createElement("span");
      current.id = "weight_step_current";
      current.className = "weight-step-current";

      const availability = document.createElement("span");
      availability.id = "weight_step_availability";
      availability.className = "weight-step-availability";

      const show_label = document.createElement("span");
      show_label.className = "weight-step-show-label";
      show_label.textContent = "show weights for steps";

      const from = document.createElement("input");
      from.id = "weight_step_from";
      from.type = "number";
      from.min = "0";
      from.step = "1";
      from.inputMode = "numeric";
      from.placeholder = "from";
      from.title = "First optimizer step in the explicit display window";

      const separator = document.createElement("span");
      separator.className = "weight-step-range-separator";
      separator.textContent = "–";

      const to = document.createElement("input");
      to.id = "weight_step_to";
      to.type = "number";
      to.min = "0";
      to.step = "1";
      to.inputMode = "numeric";
      to.placeholder = "to";
      to.title = "Last optimizer step in the explicit display window";

      const apply = document.createElement("button");
      apply.id = "weight_step_apply";
      apply.type = "button";
      apply.className = "weight-step-button";
      apply.textContent = "show";
      apply.title = "Show only retained weight snapshots in this inclusive step window";

      const whole = document.createElement("button");
      whole.id = "weight_step_whole_range";
      whole.type = "button";
      whole.className = "weight-step-button";
      whole.textContent = "whole range";
      whole.title = "Clear the explicit window and restore the normal Weights settings";

      controls.append(current, availability, show_label, from, separator, to, apply, whole);
      header.insertBefore(controls, coupling_controls);

      const apply_from_inputs = () => {
        const minimum = finite_integer(from.value);
        const maximum = finite_integer(to.value);
        if (minimum === null || maximum === null) {
          show_toast("Enter whole-number start and end steps.");
          return;
        }
        if (minimum < 0 || maximum < minimum) {
          show_toast("The weight-step end must be greater than or equal to the start.");
          return;
        }
        const capacity = common_history_capacity();
        const width = maximum - minimum + 1;
        if (width > capacity) {
          show_toast(`Selected window is ${width} steps; the common retained-history maximum is ${capacity}.`);
          return;
        }

        // A future window is valid. For any run that has already reached its start,
        // reject the request only if the beginning has already fallen out of that
        // run's retained store; otherwise the range can fill progressively.
        for (const run of context_runs()) {
          const current_step = current_step_for_run(run);
          const retained = retained_range_for_run(run);
          if (current_step === null || current_step < minimum) continue;
          if (!retained || minimum < retained.minimum) {
            show_toast(`Step ${minimum} is no longer retained for ${run.artifact_name || run_id(run) || "a visible run"}.`);
            return;
          }
        }

        selected_step_range = {minimum, maximum};
        sync_step_controls();
        invalidate_depth_view();
      };

      apply.addEventListener("click", apply_from_inputs);
      for (const input of [from, to]) {
        input.addEventListener("keydown", event => {
          if (event.key !== "Enter") return;
          event.preventDefault();
          apply_from_inputs();
        });
      }
      whole.addEventListener("click", () => {
        if (!step_filter_active()) return;
        selected_step_range = null;
        from.value = "";
        to.value = "";
        sync_step_controls();
        invalidate_depth_view();
      });
      return true;
    };

    function sync_step_controls() {
      install_step_controls();
      const controls = by_id("weight_step_group_controls");
      if (!controls) return;
      const current = by_id("weight_step_current");
      const availability = by_id("weight_step_availability");
      const from = by_id("weight_step_from");
      const to = by_id("weight_step_to");
      const apply = by_id("weight_step_apply");
      const whole = by_id("weight_step_whole_range");
      const bounds = current_step_bounds();
      const retained = available_step_range();
      const capacity = common_history_capacity();

      if (!bounds) current.textContent = "current step —";
      else if (bounds.minimum === bounds.maximum) current.textContent = `current step ${bounds.maximum}`;
      else current.textContent = `current steps ${bounds.minimum}–${bounds.maximum}`;

      if (retained.available) {
        availability.textContent = `data available ${retained.minimum}–${retained.maximum}`;
        availability.title = `Intersection of retained weight snapshots; maximum selectable window ${capacity} steps.`;
      } else {
        availability.textContent = `data available: ${retained.reason}`;
        availability.title = `Maximum selectable window ${capacity} steps.`;
      }

      if (step_filter_active()) {
        if (document.activeElement !== from) from.value = String(selected_step_range.minimum);
        if (document.activeElement !== to) to.value = String(selected_step_range.maximum);
      }
      from.title = `First step; selected window may contain at most ${capacity} steps.`;
      to.title = `Last step; selected window may contain at most ${capacity} steps.`;
      apply.disabled = !context_runs().length;
      whole.disabled = !step_filter_active();
      controls.classList.toggle("active", step_filter_active());
      controls.classList.toggle("unavailable", !retained.available);
      apply_selected_range_placeholders();
    }

    const placeholder_message = () => {
      if (!step_filter_active()) return null;
      if (all_runs_before_selected_start()) {
        return `Curves will be displayed when step ${selected_step_range.minimum} is reached`;
      }
      if (!selected_range_has_any_figure()) {
        return `Waiting for a recorded weight snapshot in steps ${selected_step_range.minimum}–${selected_step_range.maximum}`;
      }
      return null;
    };

    function apply_selected_range_placeholders() {
      const message = placeholder_message();
      if (!message) return;
      for (const chart_name of weight_chart_names) {
        const placeholder = by_id(`${chart_name}_placeholder`);
        const mount = by_id(`${chart_name}_plot`);
        if (placeholder) {
          placeholder.textContent = message;
          placeholder.hidden = false;
          placeholder.classList.add("instra-step-window-placeholder");
        }
        if (mount && mount.dataset.instraStepWindowPlaceholder !== step_filter_signature()) {
          try { window.Plotly?.purge?.(mount); } catch (_error) { /* no-op */ }
          mount.replaceChildren?.();
          mount.dataset.instraStepWindowPlaceholder = step_filter_signature();
        }
      }
    }

    const selected_coupling = () => {
      const api = window.__instra_matched_weight_selection;
      const selection = typeof api?.selection === "function" ? api.selection() : null;
      return selection && typeof selection === "object" ? selection : null;
    };

    const coupling_capability = () => {
      const api = window.__instra_matched_weight_selection;
      if (!api || typeof api.capability !== "function") return null;
      const chart_name = weight_chart_names.find(name => {
        try { return Boolean(figure_for_chart(name)); }
        catch (_error) { return false; }
      }) || weight_chart_names[0];
      try { return api.capability(chart_name); }
      catch (_error) { return null; }
    };

    const control_chart_name = () => weight_chart_names.find(name => {
      try { return Boolean(figure_for_chart(name)); }
      catch (_error) { return false; }
    }) || weight_chart_names[0];

    const same_coupling = (selection, input_feature, output_feature) => (
      selection?.user_selected === true
      && finite_integer(selection?.model_feature) === input_feature
      && finite_integer(selection?.intermediate_feature) === output_feature
    );

    const wait_for_coupling = async (input_feature, output_feature, timeout_ms = 5000) => {
      const deadline = Date.now() + timeout_ms;
      while (Date.now() < deadline) {
        if (same_coupling(selected_coupling(), input_feature, output_feature)) return true;
        await new Promise(resolve => setTimeout(resolve, 25));
      }
      return false;
    };

    const save_coupling = async (input_feature, output_feature) => {
      if (coupling_save_in_flight) return false;
      const capability = coupling_capability();
      const maximum = finite_integer(capability?.maximum);
      if (!capability?.available || maximum === null) {
        show_toast(capability?.reason || "Weight feature bounds are not available yet.");
        return false;
      }
      if (
        !Number.isInteger(input_feature) || !Number.isInteger(output_feature)
        || input_feature < 0 || output_feature < 0
        || input_feature > maximum || output_feature > maximum
      ) {
        show_toast(`Input and output features must both be between 0 and ${maximum}.`);
        return false;
      }
      if (same_coupling(selected_coupling(), input_feature, output_feature)) return true;
      if (!by_id("chart_settings_overlay")?.hidden) {
        show_toast("Close chart settings before changing the feature coupling.");
        return false;
      }

      const save_button = by_id("save_chart_settings");
      const user_toggle = by_id("chart_user_selected_weight");
      const input = by_id("chart_weight_model_feature");
      const output = by_id("chart_weight_intermediate_feature");
      if (!save_button || !user_toggle || !input || !output) {
        show_toast("Matched-weight controls are not ready yet.");
        return false;
      }

      coupling_save_in_flight = true;
      sync_coupling_editor();
      const previous_axis_chart_name = app.axis_chart_name;
      const previous_axis_workspace_mode = app.axis_chart_workspace_mode;
      const previous_show_toast = show_toast;
      let first_click_seen = false;

      const stop_completion_click = event => {
        const button = event.target.closest?.("#save_chart_settings");
        if (!button) return;
        if (!first_click_seen) {
          first_click_seen = true;
          return;
        }
        event.preventDefault();
        event.stopImmediatePropagation();
      };

      show_toast = message => {
        const text = String(message || "");
        if (
          text.startsWith("Matched weight set to model ")
          || text.startsWith("Weight matrix feature coupling set to input ")
        ) return;
        previous_show_toast(message);
      };
      window.addEventListener("click", stop_completion_click, true);

      try {
        app.axis_chart_name = control_chart_name();
        app.axis_chart_workspace_mode = app.workspace_mode === true;
        user_toggle.checked = true;
        input.value = String(input_feature);
        output.value = String(output_feature);
        save_button.disabled = false;
        save_button.click();
        const saved = await wait_for_coupling(input_feature, output_feature);
        await new Promise(resolve => setTimeout(resolve, 60));
        if (!saved) throw new Error("matched-weight save did not complete");
        previous_show_toast(
          `Weight matrix feature coupling set to ${input_feature} → ${output_feature}; active runs use it at the next weight snapshot.`
        );
        return true;
      } catch (error) {
        previous_show_toast(`Feature-coupling change failed: ${error.message}`);
        return false;
      } finally {
        window.removeEventListener("click", stop_completion_click, true);
        show_toast = previous_show_toast;
        app.axis_chart_name = previous_axis_chart_name;
        app.axis_chart_workspace_mode = previous_axis_workspace_mode;
        coupling_save_in_flight = false;
        sync_coupling_editor();
      }
    };

    const random_other_index = (current, maximum) => {
      if (maximum < 1) return current;
      const candidate = Math.floor(Math.random() * maximum);
      return candidate >= current ? candidate + 1 : candidate;
    };

    const install_coupling_editor = () => {
      const controls = by_id("weight_index_group_controls");
      if (!controls) return false;
      const old_summary = by_id("weight_index_group_summary");
      if (old_summary) old_summary.hidden = true;
      if (by_id("weight_coupling_editor")) return true;

      const editor = document.createElement("div");
      editor.id = "weight_coupling_editor";
      editor.className = "weight-coupling-editor";

      const label = document.createElement("span");
      label.className = "weight-coupling-label";
      label.textContent = "weight matrix feature coupling (i → o):";

      const input = document.createElement("input");
      input.id = "weight_coupling_input";
      input.type = "number";
      input.min = "0";
      input.step = "1";
      input.inputMode = "numeric";
      input.setAttribute("aria-label", "Input feature index");
      input.title = "Input feature index";

      const arrow = document.createElement("span");
      arrow.textContent = "→";
      arrow.setAttribute("aria-hidden", "true");

      const output = document.createElement("input");
      output.id = "weight_coupling_output";
      output.type = "number";
      output.min = "0";
      output.step = "1";
      output.inputMode = "numeric";
      output.setAttribute("aria-label", "Output feature index");
      output.title = "Output feature index";

      editor.append(label, input, arrow, output);
      controls.insertBefore(editor, controls.firstChild);

      const commit = () => {
        const input_feature = finite_integer(input.value);
        const output_feature = finite_integer(output.value);
        if (input_feature === null || output_feature === null) {
          show_toast("Input and output feature indices must be whole numbers.");
          sync_coupling_editor();
          return;
        }
        save_coupling(input_feature, output_feature);
      };
      for (const control of [input, output]) {
        control.addEventListener("change", commit);
        control.addEventListener("keydown", event => {
          if (event.key !== "Enter") return;
          event.preventDefault();
          commit();
        });
      }

      controls.addEventListener("click", event => {
        const button = event.target.closest?.("button");
        if (!button || !controls.contains(button)) return;
        const ids = new Set([
          "weight_residual_minus",
          "weight_residual_plus",
          "weight_branch_minus",
          "weight_branch_plus",
          "weight_random_jump",
        ]);
        if (!ids.has(button.id)) return;
        event.preventDefault();
        event.stopImmediatePropagation();
        if (coupling_save_in_flight) return;
        const capability = coupling_capability();
        const maximum = finite_integer(capability?.maximum);
        const selection = selected_coupling();
        if (!capability?.available || maximum === null || !selection) return;
        let next_input = Math.min(maximum, Math.max(0, finite_integer(selection.model_feature) ?? 0));
        let next_output = Math.min(maximum, Math.max(0, finite_integer(selection.intermediate_feature) ?? 0));
        if (button.id === "weight_residual_minus") next_input = Math.max(0, next_input - 1);
        else if (button.id === "weight_residual_plus") next_input = Math.min(maximum, next_input + 1);
        else if (button.id === "weight_branch_minus") next_output = Math.max(0, next_output - 1);
        else if (button.id === "weight_branch_plus") next_output = Math.min(maximum, next_output + 1);
        else {
          next_input = random_other_index(next_input, maximum);
          next_output = random_other_index(next_output, maximum);
        }
        save_coupling(next_input, next_output);
      }, true);
      return true;
    };

    function sync_coupling_editor() {
      install_coupling_editor();
      const input = by_id("weight_coupling_input");
      const output = by_id("weight_coupling_output");
      if (!input || !output) return;
      const capability = coupling_capability();
      const maximum = finite_integer(capability?.maximum);
      const selection = selected_coupling();
      const input_feature = finite_integer(selection?.model_feature) ?? 0;
      const output_feature = finite_integer(selection?.intermediate_feature) ?? 0;
      if (document.activeElement !== input) input.value = String(input_feature);
      if (document.activeElement !== output) output.value = String(output_feature);
      if (maximum !== null) {
        input.max = String(maximum);
        output.max = String(maximum);
      } else {
        input.removeAttribute("max");
        output.removeAttribute("max");
      }
      const disabled = coupling_save_in_flight || !capability?.available || maximum === null;
      input.disabled = disabled;
      output.disabled = disabled;
      for (const id of [
        "weight_residual_minus",
        "weight_residual_plus",
        "weight_branch_minus",
        "weight_branch_plus",
        "weight_random_jump",
      ]) {
        const button = by_id(id);
        if (button) button.disabled = disabled || (id === "weight_random_jump" && maximum < 1);
      }
      const button_labels = new Map([
        ["weight_residual_minus", "i−"],
        ["weight_residual_plus", "i+"],
        ["weight_branch_minus", "o−"],
        ["weight_branch_plus", "o+"],
      ]);
      for (const [id, label] of button_labels) {
        const button = by_id(id);
        if (button && button.textContent !== label) button.textContent = label;
      }
      const random = by_id("weight_random_jump");
      if (random) {
        random.hidden = false;
        random.removeAttribute("hidden");
        random.textContent = "RND";
        random.title = "Let INSTRA choose a new random feature coupling";
        random.setAttribute("aria-label", random.title);
      }
    }

    const scientific_tick_label = value => {
      const numeric = Number(value);
      if (!Number.isFinite(numeric)) return "";
      if (numeric === 0) return "0e+00";
      const [mantissa, exponent_text] = numeric.toExponential(0).split("e");
      const exponent = Number(exponent_text);
      const sign = exponent >= 0 ? "+" : "-";
      return `${mantissa}e${sign}${String(Math.abs(exponent)).padStart(2, "0")}`;
    };

    const refine_signed_log_axis = (prepared, chart_name) => {
      if (!weight_chart_set.has(chart_name)) return;
      if (load_json(trajectory_scale_settings_key, {})[chart_name] !== "log") return;
      const axis = prepared.layout?.yaxis;
      const tickvals = Array.isArray(axis?.tickvals) ? axis.tickvals.map(Number) : [];
      const ticktext = Array.isArray(axis?.ticktext) ? axis.ticktext : [];
      if (!tickvals.length || tickvals.length !== ticktext.length) return;

      const pairs = tickvals.map((axis_value, index) => ({
        axis_value,
        actual_value: Number(ticktext[index]),
      })).filter(pair => Number.isFinite(pair.axis_value) && Number.isFinite(pair.actual_value));
      const positive = pairs
        .filter(pair => pair.actual_value > 0 && pair.axis_value > 0)
        .sort((left, right) => left.actual_value - right.actual_value);
      if (positive.length) {
        const smallest = positive[0];
        const denominator = (10 ** smallest.axis_value) - 1;
        const threshold = denominator > 0 ? smallest.actual_value / denominator : null;
        if (Number.isFinite(threshold) && threshold > 0) {
          const new_actual = smallest.actual_value / 10;
          const new_axis = Math.log10(1 + new_actual / threshold);
          if (Number.isFinite(new_axis) && new_axis > 0) {
            pairs.push({axis_value: new_axis, actual_value: new_actual});
            pairs.push({axis_value: -new_axis, actual_value: -new_actual});
          }
        }
      }
      const unique = new Map();
      for (const pair of pairs) unique.set(pair.axis_value.toPrecision(15), pair);
      const ordered = [...unique.values()].sort((left, right) => left.axis_value - right.axis_value);
      prepared.layout.yaxis = {
        ...axis,
        tickmode: "array",
        tickvals: ordered.map(pair => pair.axis_value),
        ticktext: ordered.map(pair => scientific_tick_label(pair.actual_value)),
      };
    };

    const base_prepare_figure_weight_v2 = prepare_figure;
    prepare_figure = function(figure, chart_name) {
      const prepared = base_prepare_figure_weight_v2(figure, chart_name);
      refine_signed_log_axis(prepared, chart_name);
      if (!weight_chart_set.has(chart_name)) return prepared;

      // In Runs view a current-only curve should identify its run visually, not
      // inherit the THOG age ramp / DENSE source styling that is useful for history.
      if (
        app.workspace_mode !== true
        && global_flags.current_weights_only
        && !step_filter_active()
      ) {
        const run_colour = colour_for_run(String(app.current_run_id || ""));
        for (const trace of prepared.data || []) {
          if (trace?.meta?.instra_top_axis_anchor === true) continue;
          const mode = String(trace?.mode || "");
          if (mode.includes("lines") && trace.line) trace.line = {...trace.line, color: run_colour};
          if (mode.includes("markers") || trace.marker) {
            trace.marker = {...(trace.marker || {}), color: run_colour};
            trace.marker.line = {...(trace.marker?.line || {}), color: run_colour};
          }
        }
      }
      return prepared;
    };

    const base_render_plot_weight_v2 = render_plot;
    render_plot = async function(mount, figure, chart_name) {
      if (weight_chart_set.has(chart_name) && all_runs_before_selected_start()) {
        apply_selected_range_placeholders();
        return;
      }
      if (mount?.dataset) delete mount.dataset.instraStepWindowPlaceholder;
      return base_render_plot_weight_v2(mount, figure, chart_name);
    };

    const base_render_figures_weight_v2 = render_figures;
    render_figures = async function() {
      const result = await base_render_figures_weight_v2();
      apply_selected_range_placeholders();
      sync_step_controls();
      sync_coupling_editor();
      return result;
    };

    const polish_runs_table = () => {
      const header_row = document.querySelector(".runs-table thead tr");
      if (!header_row) return;
      const headers = [...header_row.children];
      const host_index = headers.findIndex(cell => cell.textContent.trim().toLowerCase() === "host");
      const logged_index = headers.findIndex(cell => ["logged", "steps"].includes(cell.textContent.trim().toLowerCase()));
      if (logged_index >= 0) {
        headers[logged_index].textContent = "STEPS";
        headers[logged_index].title = "Latest optimizer step recorded by INSTRA";
      }
      if (host_index >= 0) {
        headers[host_index].title = "Host (first 10 characters)";
        for (const row of document.querySelectorAll(".runs-table tbody tr[data-run-id]")) {
          const cell = row.children[host_index];
          if (!cell) continue;
          const full = cell.dataset.instraFullHost || String(cell.textContent || "");
          cell.dataset.instraFullHost = full;
          cell.textContent = full.slice(0, 10);
          cell.title = full;
        }
      }
    };

    const base_render_runs_weight_v2 = render_runs;
    render_runs = function() {
      const result = base_render_runs_weight_v2();
      polish_runs_table();
      sync_step_controls();
      sync_coupling_editor();
      return result;
    };

    const base_render_run_heading_weight_v2 = render_run_heading;
    render_run_heading = function() {
      const result = base_render_run_heading_weight_v2();
      sync_step_controls();
      sync_coupling_editor();
      return result;
    };

    window.__instra_weight_controls_v2 = {
      available_step_range,
      common_history_capacity,
      current_step_bounds,
      global_flags: () => ({...global_flags}),
      selected_step_range: () => step_filter_active() ? {...selected_step_range} : null,
      set_global_flags,
      set_step_range: (minimum, maximum) => {
        selected_step_range = {minimum: Number(minimum), maximum: Number(maximum)};
        sync_step_controls();
        invalidate_depth_view();
      },
      clear_step_range: () => {
        selected_step_range = null;
        sync_step_controls();
        invalidate_depth_view();
      },
    };

    const style = document.createElement("style");
    style.textContent = `
      #coefficients_chart_group > .chart-group-header {
        overflow: visible !important;
      }
      .weight-step-group-controls {
        flex: 0 1 auto; min-width: 0; margin-left: 40px; display: inline-flex;
        align-items: center; gap: 6px; white-space: nowrap; color: #414a55;
        font-size: 11px; font-weight: 500; font-variant-numeric: tabular-nums;
      }
      .weight-step-current { color: #2f3740; font-weight: 650; }
      .weight-step-availability { color: #59636e; margin-right: 8px; }
      .weight-step-group-controls.unavailable .weight-step-availability { color: #8d4a4a; }
      .weight-step-group-controls.active { color: #155f70; }
      .weight-step-show-label { color: #3d4650; }
      .weight-step-group-controls input,
      .weight-coupling-editor input {
        height: 23px; box-sizing: border-box; padding: 0 4px; border: 1px solid #c8ced5;
        border-radius: 3px; background: #fff; color: #252b32; font-size: 11px;
        font-variant-numeric: tabular-nums;
      }
      .weight-step-group-controls input { width: 58px; }
      .weight-step-button {
        height: 23px; padding: 0 7px; border: 1px solid #cbd1d8; border-radius: 3px;
        background: #f7f8fa; color: #46505b; font-size: 10px; cursor: pointer;
      }
      .weight-step-button:hover:not(:disabled) { border-color: #8fc8d2; background: #eaf7f9; color: #006f83; }
      .weight-step-button:disabled { opacity: .42; cursor: default; }
      .instra-step-window-placeholder {
        display: flex !important; align-items: center; justify-content: center;
        color: #66717c !important; font-size: 13px !important; font-weight: 500 !important;
        text-align: center; padding: 24px;
      }

      #weight_index_group_summary { display: none !important; }
      #weight_index_group_controls {
        flex: 0 0 auto !important; min-width: max-content !important; margin-left: auto !important;
        gap: 4px !important; overflow: visible !important; color: #414a55 !important;
        font-size: 11px !important; font-weight: 500 !important;
      }
      .weight-coupling-editor {
        flex: 0 0 auto; min-width: max-content; display: inline-flex; align-items: center;
        gap: 4px; margin-right: 5px; overflow: visible;
      }
      .weight-coupling-label { white-space: nowrap; color: #3d4650; }
      .weight-coupling-editor input { width: 48px; text-align: center; }
      #weight_index_group_controls .weight-index-step-button {
        min-width: 27px !important; width: 27px !important; height: 23px !important;
        padding: 0 2px !important; display: inline-flex !important;
        align-items: center !important; justify-content: center !important; line-height: 1 !important;
      }
      #weight_random_jump,
      #weight_random_jump[hidden] {
        display: inline-flex !important; align-items: center !important; justify-content: center !important;
        min-width: 35px !important; width: 35px !important; height: 23px !important;
        padding: 0 !important; line-height: 1 !important; text-align: center !important;
        font-size: 0 !important;
      }
      #weight_random_jump::after {
        content: "RND"; font-size: 10px; line-height: 1; display: block;
      }
      #coefficients_chart_group .weights-group-settings-button {
        margin-left: 38px !important; margin-right: 8px !important;
      }
      .runs-table th:nth-child(6), .runs-table td:nth-child(6) {
        width: 10ch; max-width: 10ch; overflow: hidden; text-overflow: clip; white-space: nowrap;
      }
    `;
    document.head.appendChild(style);

    install_step_controls();
    install_coupling_editor();
    polish_runs_table();
    sync_step_controls();
    sync_coupling_editor();
    sync_global_weight_fields();
  }, 0);
});
// ^^^ THOG
