// vvv THOG
"use strict";

// Retained-step replay and final Weights-header controls. A selected step/window
// overrides "current weights only" only for snapshot retention: matched-coupling
// semantics stay active, while the server reads only the requested retained rows.
window.addEventListener("load", () => {
  setTimeout(() => {
    const weight_chart_names = new Set([...depth_weight_chart_names]);
    const trajectory_scale_settings_key = "thog2_local_trajectory_scale_modes";
    let selected_step_range = null;
    let coupling_save_in_flight = false;
    let refresh_serial = 0;
    let reconcile_in_flight = false;

    const finite_integer = value => {
      if (value === null || value === undefined || value === "") return null;
      const numeric = Number(value);
      return Number.isInteger(numeric) ? numeric : null;
    };

    const step_filter_active = () => (
      finite_integer(selected_step_range?.minimum) !== null
      && finite_integer(selected_step_range?.maximum) !== null
    );

    const step_filter_signature = () => step_filter_active()
      ? `${selected_step_range.minimum}:${selected_step_range.maximum}`
      : "whole-range";

    window.__instra_weight_step_filter = {
      active: step_filter_active,
      signature: step_filter_signature,
      request_range: () => step_filter_active() ? {...selected_step_range} : null,
    };

    // Keep matched selection active for an explicit historical window, but do not
    // collapse that window back to its newest snapshot.
    const base_normalize_chart_settings_step_filter = normalize_chart_settings;
    normalize_chart_settings = function(chart_name, supplied = null) {
      const normalized = base_normalize_chart_settings_step_filter(chart_name, supplied);
      if (!step_filter_active() || !weight_chart_names.has(chart_name)) return normalized;
      return {...normalized, current_weights_only: true};
    };

    const base_retain_latest_weight_snapshots_step_filter = retain_latest_weight_snapshots;
    retain_latest_weight_snapshots = function(prepared) {
      if (step_filter_active()) return;
      return base_retain_latest_weight_snapshots_step_filter(prepared);
    };

    if (typeof instra_enforce_workspace_latest_weights === "function") {
      const base_workspace_latest_step_filter = instra_enforce_workspace_latest_weights;
      instra_enforce_workspace_latest_weights = function(prepared) {
        if (step_filter_active()) return prepared;
        return base_workspace_latest_step_filter(prepared);
      };
    }

    const visible_workspace_runs = () => (
      app.workspace_mode === true && typeof window.__instra_workspace?.visible_runs === "function"
        ? window.__instra_workspace.visible_runs()
        : []
    );

    const retained_range_for_run = run => {
      if (!run || Number(run.depth_snapshot_count || 0) <= 0) return null;
      const minimum = finite_integer(run.depth_minimum_update);
      const maximum = finite_integer(run.depth_maximum_update);
      if (minimum === null || maximum === null || maximum < minimum) return null;
      return {minimum, maximum};
    };

    const available_step_range = () => {
      if (app.workspace_mode === true) {
        const runs = visible_workspace_runs();
        if (!runs.length) return {available: false, reason: "no visible runs"};
        const ranges = runs.map(retained_range_for_run);
        if (ranges.some(range => range === null)) {
          return {available: false, reason: "no overlapping steps"};
        }
        const minimum = Math.max(...ranges.map(range => range.minimum));
        const maximum = Math.min(...ranges.map(range => range.maximum));
        return minimum <= maximum
          ? {available: true, minimum, maximum}
          : {available: false, reason: "no overlapping steps"};
      }
      const range = retained_range_for_run(current_run());
      return range
        ? {available: true, ...range}
        : {available: false, reason: "no stored weight steps"};
    };

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
      app.figure_revision = null;

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

    const install_step_controls = () => {
      const header = by_id("coefficients_chart_group")?.querySelector(":scope > .chart-group-header");
      const coupling_controls = by_id("weight_index_group_controls");
      if (!header || !coupling_controls || by_id("weight_step_group_controls")) return false;

      const controls = document.createElement("div");
      controls.id = "weight_step_group_controls";
      controls.className = "weight-step-group-controls";
      controls.setAttribute("role", "group");
      controls.setAttribute("aria-label", "Retained optimizer-step window");

      const availability = document.createElement("span");
      availability.id = "weight_step_availability";
      availability.className = "weight-step-availability";

      const show_label = document.createElement("span");
      show_label.className = "weight-step-show-label";
      show_label.textContent = "show";

      const from = document.createElement("input");
      from.id = "weight_step_from";
      from.type = "number";
      from.min = "0";
      from.step = "1";
      from.inputMode = "numeric";
      from.placeholder = "step / from";
      from.title = "Exact step, or first step of a range";

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
      to.title = "Last step of range; leave blank for one exact step";

      const apply = document.createElement("button");
      apply.id = "weight_step_apply";
      apply.type = "button";
      apply.className = "weight-step-button";
      apply.textContent = "show";
      apply.title = "Show the exact retained step or inclusive retained-step range";

      const whole = document.createElement("button");
      whole.id = "weight_step_whole_range";
      whole.type = "button";
      whole.className = "weight-step-button";
      whole.textContent = "whole range";
      whole.title = "Clear the explicit step selection and restore the normal Weights view";

      controls.append(availability, show_label, from, separator, to, apply, whole);
      header.insertBefore(controls, coupling_controls);

      const apply_from_inputs = () => {
        const range = available_step_range();
        if (!range.available) {
          show_toast(`Weight-step selection unavailable: ${range.reason}.`);
          return;
        }
        const minimum = finite_integer(from.value);
        const raw_maximum = to.value.trim();
        const maximum = raw_maximum === "" ? minimum : finite_integer(raw_maximum);
        if (minimum === null || maximum === null) {
          show_toast("Enter a whole-number step, or an inclusive from–to range.");
          return;
        }
        if (minimum > maximum) {
          show_toast("Weight-step range end must be greater than or equal to its start.");
          return;
        }
        if (minimum < range.minimum || maximum > range.maximum) {
          show_toast(`Weight steps must lie within ${range.minimum}–${range.maximum}.`);
          return;
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
      const range = available_step_range();
      const availability = by_id("weight_step_availability");
      const from = by_id("weight_step_from");
      const to = by_id("weight_step_to");
      const apply = by_id("weight_step_apply");
      const whole = by_id("weight_step_whole_range");

      if (range.available) {
        availability.textContent = `data available for steps ${range.minimum} – ${range.maximum}`;
        availability.title = "Retained local weight snapshots; older discarded snapshots cannot be replayed.";
        from.min = String(range.minimum);
        from.max = String(range.maximum);
        to.min = String(range.minimum);
        to.max = String(range.maximum);
      } else {
        availability.textContent = `data available: ${range.reason}`;
        from.removeAttribute("max");
        to.removeAttribute("max");
      }
      const disabled = !range.available;
      from.disabled = disabled;
      to.disabled = disabled;
      apply.disabled = disabled;
      whole.disabled = !step_filter_active();
      controls.classList.toggle("active", step_filter_active());
      controls.classList.toggle("unavailable", disabled);
    }

    const selected_coupling = () => {
      const api = window.__instra_matched_weight_selection;
      const selection = typeof api?.selection === "function" ? api.selection() : null;
      return selection && typeof selection === "object" ? selection : null;
    };

    const coupling_capability = () => {
      const api = window.__instra_matched_weight_selection;
      if (!api || typeof api.capability !== "function") return null;
      const chart_name = [...weight_chart_names].find(name => {
        try { return Boolean(figure_for_chart(name)); }
        catch (_error) { return false; }
      }) || [...weight_chart_names][0];
      try { return api.capability(chart_name); }
      catch (_error) { return null; }
    };

    const control_chart_name = () => [...weight_chart_names].find(name => {
      try { return Boolean(figure_for_chart(name)); }
      catch (_error) { return false; }
    }) || [...weight_chart_names][0];

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
          control.blur();
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
      if (!weight_chart_names.has(chart_name)) return;
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

    const base_prepare_figure_step_controls = prepare_figure;
    prepare_figure = function(figure, chart_name) {
      const prepared = base_prepare_figure_step_controls(figure, chart_name);
      refine_signed_log_axis(prepared, chart_name);
      return prepared;
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

    const reconcile_selected_range = () => {
      if (reconcile_in_flight || !step_filter_active()) return;
      const available = available_step_range();
      if (
        available.available
        && selected_step_range.minimum >= available.minimum
        && selected_step_range.maximum <= available.maximum
      ) return;
      reconcile_in_flight = true;
      selected_step_range = null;
      const from = by_id("weight_step_from");
      const to = by_id("weight_step_to");
      if (from) from.value = "";
      if (to) to.value = "";
      sync_step_controls();
      queueMicrotask(() => {
        reconcile_in_flight = false;
        invalidate_depth_view();
      });
    };

    const base_render_runs_step_controls = render_runs;
    render_runs = function() {
      const result = base_render_runs_step_controls();
      polish_runs_table();
      sync_step_controls();
      sync_coupling_editor();
      reconcile_selected_range();
      return result;
    };

    const base_render_run_heading_step_controls = render_run_heading;
    render_run_heading = function() {
      const result = base_render_run_heading_step_controls();
      sync_step_controls();
      sync_coupling_editor();
      reconcile_selected_range();
      return result;
    };

    const style = document.createElement("style");
    style.textContent = `
      .weight-step-group-controls {
        flex: 0 1 auto; min-width: 0; margin-left: 32px; display: inline-flex;
        align-items: center; gap: 5px; white-space: nowrap; color: #5b6470;
        font-size: 9px; font-variant-numeric: tabular-nums;
      }
      .weight-step-availability { margin-right: 8px; }
      .weight-step-group-controls.unavailable .weight-step-availability { color: #9b2c2c; }
      .weight-step-group-controls.active { color: #1d6572; }
      .weight-step-group-controls input,
      .weight-coupling-editor input {
        height: 23px; box-sizing: border-box; padding: 0 4px; border: 1px solid #d4d8de;
        border-radius: 3px; background: #fff; color: #303841; font-size: 10px;
        font-variant-numeric: tabular-nums;
      }
      .weight-step-group-controls input { width: 66px; }
      .weight-step-button {
        height: 23px; padding: 0 6px; border: 1px solid #d4d8de; border-radius: 3px;
        background: #f7f8fa; color: #4f5863; font-size: 9px; cursor: pointer;
      }
      .weight-step-button:hover:not(:disabled) { border-color: #9bcfd8; background: #e8f7f9; color: #007f98; }
      .weight-step-button:disabled { opacity: .42; cursor: default; }

      #weight_index_group_summary { display: none !important; }
      #weight_index_group_controls { margin-left: auto !important; gap: 4px !important; }
      .weight-coupling-editor { display: inline-flex; align-items: center; gap: 4px; margin-right: 4px; }
      .weight-coupling-editor input { width: 48px; }
      #weight_index_group_controls .weight-index-step-button {
        min-width: 27px !important; width: 27px !important; height: 23px !important;
        padding-left: 2px !important; padding-right: 2px !important;
      }
      #weight_random_jump,
      #weight_random_jump[hidden] {
        display: inline-flex !important; min-width: 35px !important; width: 35px !important;
        height: 23px !important; padding-left: 3px !important; padding-right: 3px !important;
      }
      #coefficients_chart_group .weights-group-settings-button {
        margin-left: 26px !important; margin-right: 8px !important;
      }
      .runs-table th:nth-child(6), .runs-table td:nth-child(6) {
        max-width: 10ch; overflow: hidden; text-overflow: clip; white-space: nowrap;
      }
    `;
    document.head.appendChild(style);

    install_step_controls();
    install_coupling_editor();
    polish_runs_table();
    sync_step_controls();
    sync_coupling_editor();

    // Earlier compatibility layers have bounded startup passes of their own. Keep
    // this final labelling/layout pass bounded as well; never observe Plotly DOM.
    let startup_passes = 0;
    const startup_timer = setInterval(() => {
      startup_passes += 1;
      install_step_controls();
      sync_step_controls();
      sync_coupling_editor();
      polish_runs_table();
      if (startup_passes >= 40 || (
        by_id("weight_step_group_controls")
        && by_id("weight_coupling_editor")
        && by_id("weight_random_jump")?.textContent === "RND"
      )) clearInterval(startup_timer);
    }, 100);
  }, 650);
});
// ^^^ THOG
