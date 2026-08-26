// vvv THOG
"use strict";

// Final weight-comparison/run-table interaction pass.  The two logical weight
// coordinates are residual-stream feature and branch feature: Q/K/V and MLP
// expansion map branch <- residual, while attention output and MLP contraction
// map residual <- branch.  The persisted storage/API field names remain unchanged
// for compatibility with already-running trainers and stored snapshots.
window.addEventListener("load", () => {
  setTimeout(() => {
    const weight_chart_names = [...depth_weight_chart_names];
    const weight_chart_set = new Set(weight_chart_names);
    const workspace_weight_baseline_px = (3.6 + 0.45) / 2;
    let coordinate_save_in_flight = false;

    const finite_integer = value => {
      if (value === null || value === undefined || value === "") return null;
      const numeric = Number(value);
      return Number.isInteger(numeric) ? numeric : null;
    };

    const matched_api = () => window.__instra_matched_weight_selection || null;
    const control_chart_name = () => (
      weight_chart_names.find(chart_name => {
        try { return Boolean(figure_for_chart(chart_name)); }
        catch (_error) { return false; }
      }) || weight_chart_names[0]
    );
    const matched_capability = () => {
      const api = matched_api();
      if (!api || typeof api.capability !== "function") {
        return {available: false, maximum: null, reason: "Waiting for matched-weight controls."};
      }
      return api.capability(control_chart_name());
    };
    const matched_selection = () => {
      const api = matched_api();
      const selection = typeof api?.selection === "function" ? api.selection() : null;
      return selection && typeof selection === "object"
        ? selection
        : {user_selected: false, model_feature: 0, intermediate_feature: 0};
    };

    const same_coordinate = (left, right) => (
      left?.user_selected === right?.user_selected
      && finite_integer(left?.model_feature) === finite_integer(right?.model_feature)
      && finite_integer(left?.intermediate_feature) === finite_integer(right?.intermediate_feature)
    );

    const wait_for_selection = async (expected, timeout_ms = 5000) => {
      const deadline = Date.now() + timeout_ms;
      while (Date.now() < deadline) {
        if (same_coordinate(matched_selection(), expected)) return true;
        await new Promise(resolve => setTimeout(resolve, 25));
      }
      return false;
    };

    // Reuse the established matched-selection save handler so its private
    // in-memory selection stays coherent without duplicating selection state.
    const save_weight_coordinate = async next => {
      if (coordinate_save_in_flight) return false;
      if (!by_id("chart_settings_overlay")?.hidden) {
        show_toast("Close chart settings before changing the group weight indices.");
        return false;
      }
      const capability = matched_capability();
      const maximum = finite_integer(capability?.maximum);
      const residual = finite_integer(next.model_feature);
      const branch = finite_integer(next.intermediate_feature);
      if (!capability?.available || maximum === null) {
        show_toast(capability?.reason || "Weight-index bounds are not available yet.");
        return false;
      }
      if (
        residual === null || branch === null
        || residual < 0 || branch < 0
        || residual > maximum || branch > maximum
      ) {
        show_toast(`Weight indices must both be between 0 and ${maximum}.`);
        return false;
      }

      const save_button = by_id("save_chart_settings");
      const user_toggle = by_id("chart_user_selected_weight");
      const residual_input = by_id("chart_weight_model_feature");
      const branch_input = by_id("chart_weight_intermediate_feature");
      if (!save_button || !user_toggle || !residual_input || !branch_input) {
        show_toast("Matched-weight editor controls are not ready yet.");
        return false;
      }

      coordinate_save_in_flight = true;
      update_weight_index_controls();
      const previous_axis_chart_name = app.axis_chart_name;
      const previous_axis_workspace_mode = app.axis_chart_workspace_mode;
      const previous_show_toast = show_toast;
      let first_click_seen = false;

      // The first synthetic click is consumed by the matched-selection handler.
      // Its own completion click would otherwise save unrelated chart settings;
      // stop that second click at window capture after the private selection has
      // already been updated.
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
        if (text.startsWith("Matched weight set to model ")) return;
        previous_show_toast(message);
      };
      window.addEventListener("click", stop_completion_click, true);

      try {
        app.axis_chart_name = control_chart_name();
        app.axis_chart_workspace_mode = app.workspace_mode === true;
        user_toggle.checked = true;
        residual_input.value = String(residual);
        branch_input.value = String(branch);
        save_button.disabled = false;
        save_button.click();
        const saved = await wait_for_selection({
          user_selected: true,
          model_feature: residual,
          intermediate_feature: branch,
        });
        await new Promise(resolve => setTimeout(resolve, 60));
        if (!saved) throw new Error("matched-weight save did not complete");
        previous_show_toast(
          `Weight indices set to residual ${residual}, branch ${branch}; active runs use them at the next weight snapshot.`
        );
        return true;
      } catch (error) {
        previous_show_toast(`Weight-index change failed: ${error.message}`);
        return false;
      } finally {
        window.removeEventListener("click", stop_completion_click, true);
        show_toast = previous_show_toast;
        app.axis_chart_name = previous_axis_chart_name;
        app.axis_chart_workspace_mode = previous_axis_workspace_mode;
        coordinate_save_in_flight = false;
        update_weight_index_controls();
      }
    };

    const random_other_index = (current, maximum) => {
      if (maximum < 1) return current;
      const candidate = Math.floor(Math.random() * maximum);
      return candidate >= current ? candidate + 1 : candidate;
    };

    const install_weight_index_controls = () => {
      const header = by_id("coefficients_chart_group")?.querySelector(":scope > .chart-group-header");
      if (!header || by_id("weight_index_group_controls")) return;

      const controls = document.createElement("div");
      controls.id = "weight_index_group_controls";
      controls.className = "weight-index-group-controls";
      controls.setAttribute("role", "group");
      controls.setAttribute(
        "aria-label",
        "Matched weight indices. Residual feature is the d_model coordinate; branch feature is the attention or MLP-internal coordinate."
      );

      const summary = document.createElement("span");
      summary.id = "weight_index_group_summary";
      summary.className = "weight-index-group-summary";
      summary.title = (
        "Residual feature is the d_model/residual-stream coordinate. Branch feature is the attention feature for Q/K/V/output "
        + "and the MLP hidden feature for expansion/contraction."
      );
      controls.appendChild(summary);

      const button_specs = [
        ["weight_residual_minus", "res−", "Decrement residual feature"],
        ["weight_residual_plus", "res+", "Increment residual feature"],
        ["weight_branch_minus", "branch−", "Decrement branch feature"],
        ["weight_branch_plus", "branch+", "Increment branch feature"],
        ["weight_random_jump", "random", "Choose new random residual and branch features within bounds"],
      ];
      for (const [id, label, title] of button_specs) {
        const button = document.createElement("button");
        button.id = id;
        button.type = "button";
        button.className = "weight-index-step-button";
        button.textContent = label;
        button.title = title;
        controls.appendChild(button);
      }

      const gear = header.querySelector(".weights-group-settings-button");
      header.insertBefore(controls, gear || null);

      by_id("weight_residual_minus").addEventListener("click", () => {
        const selection = matched_selection();
        save_weight_coordinate({
          ...selection,
          user_selected: true,
          model_feature: Math.max(0, Number(selection.model_feature || 0) - 1),
        });
      });
      by_id("weight_residual_plus").addEventListener("click", () => {
        const selection = matched_selection();
        const maximum = finite_integer(matched_capability()?.maximum);
        if (maximum === null) return;
        save_weight_coordinate({
          ...selection,
          user_selected: true,
          model_feature: Math.min(maximum, Number(selection.model_feature || 0) + 1),
        });
      });
      by_id("weight_branch_minus").addEventListener("click", () => {
        const selection = matched_selection();
        save_weight_coordinate({
          ...selection,
          user_selected: true,
          intermediate_feature: Math.max(0, Number(selection.intermediate_feature || 0) - 1),
        });
      });
      by_id("weight_branch_plus").addEventListener("click", () => {
        const selection = matched_selection();
        const maximum = finite_integer(matched_capability()?.maximum);
        if (maximum === null) return;
        save_weight_coordinate({
          ...selection,
          user_selected: true,
          intermediate_feature: Math.min(maximum, Number(selection.intermediate_feature || 0) + 1),
        });
      });
      by_id("weight_random_jump").addEventListener("click", () => {
        const selection = matched_selection();
        const maximum = finite_integer(matched_capability()?.maximum);
        if (maximum === null || maximum < 1) return;
        const residual = Math.min(maximum, Math.max(0, Number(selection.model_feature || 0)));
        const branch = Math.min(maximum, Math.max(0, Number(selection.intermediate_feature || 0)));
        save_weight_coordinate({
          ...selection,
          user_selected: true,
          model_feature: random_other_index(residual, maximum),
          intermediate_feature: random_other_index(branch, maximum),
        });
      });
    };

    function update_weight_index_controls() {
      install_weight_index_controls();
      const controls = by_id("weight_index_group_controls");
      if (!controls) return;
      const summary = by_id("weight_index_group_summary");
      const selection = matched_selection();
      const capability = matched_capability();
      const maximum = finite_integer(capability?.maximum);
      const residual = finite_integer(selection.model_feature) ?? 0;
      const branch = finite_integer(selection.intermediate_feature) ?? 0;

      if (!capability?.available || maximum === null) {
        summary.textContent = `weight indices: ${capability?.reason || "waiting…"}`;
      } else if (selection.user_selected === true) {
        summary.textContent = `weight indices: residual ${residual}, branch ${branch}`;
      } else {
        summary.textContent = `weight indices: random · stored residual ${residual}, branch ${branch}`;
      }

      const unavailable = coordinate_save_in_flight || !capability?.available || maximum === null;
      by_id("weight_residual_minus").disabled = unavailable || residual <= 0;
      by_id("weight_residual_plus").disabled = unavailable || residual >= maximum;
      by_id("weight_branch_minus").disabled = unavailable || branch <= 0;
      by_id("weight_branch_plus").disabled = unavailable || branch >= maximum;
      by_id("weight_random_jump").disabled = unavailable || maximum < 1;
      controls.classList.toggle("saving", coordinate_save_in_flight);
    }

    const configuration_value = (run, ...names) => {
      const configuration = run?.configuration && typeof run.configuration === "object"
        ? run.configuration
        : {};
      for (const name of names) {
        if (Object.prototype.hasOwnProperty.call(configuration, name)) return configuration[name];
      }
      return null;
    };
    const display_config_value = value => (
      value === null || value === undefined || value === "" ? "—" : String(value)
    );
    const display_boolean = value => {
      if (value === true || value === 1 || String(value).toLowerCase() === "true") return "Y";
      if (value === false || value === 0 || String(value).toLowerCase() === "false") return "N";
      return "—";
    };

    const run_shape_columns = Object.freeze([
      {
        key: "preset", label: "p", title: "preset", numeric: false,
        value: run => run?.preset || configuration_value(run, "geometry_preset", "model_type"),
      },
      {key: "layers", label: "L", title: "layers", value: run => configuration_value(run, "n_layer")},
      {
        key: "depth_order", label: "P", title: "depth order (not applicable to DENSE)",
        value: run => String(run?.model_type || configuration_value(run, "model_type") || "").toLowerCase() === "dense"
          ? "—"
          : configuration_value(run, "o_depth"),
      },
      {key: "context", label: "C", title: "context length", value: run => configuration_value(run, "block_size")},
      {key: "d_model", label: "D", title: "d_model", value: run => configuration_value(run, "n_embd")},
      {key: "heads", label: "H", title: "attention heads", value: run => configuration_value(run, "n_head")},
      {
        key: "grad_accum", label: "A", title: "gradient accumulation steps",
        value: run => configuration_value(run, "gradient_accumulation_steps"),
      },
      {
        key: "activation_checkpointing", label: "S", title: "activation checkpointing",
        value: run => display_boolean(configuration_value(run, "activation_checkpointing")),
      },
    ]);

    const install_run_shape_headers = () => {
      const header_row = document.querySelector(".runs-table thead tr");
      if (!header_row || header_row.querySelector("[data-instra-run-shape-header]")) return;
      const host_header = [...header_row.children].find(cell => cell.textContent.trim().toLowerCase() === "host");
      if (!host_header) return;
      let marker = host_header;
      for (const definition of run_shape_columns) {
        const header = document.createElement("th");
        header.dataset.instraRunShapeHeader = definition.key;
        header.className = definition.numeric === false
          ? "run-shape-column run-preset-column"
          : "numeric-column run-shape-column";
        header.textContent = definition.label;
        header.title = definition.title;
        marker.insertAdjacentElement("afterend", header);
        marker = header;
      }
    };

    const base_append_run_row_weight_controls = append_run_row;
    append_run_row = function(body, run) {
      const result = base_append_run_row_weight_controls(body, run);
      const row = body.lastElementChild;
      if (!row || row.classList.contains("group-row")) return result;
      const host_cell = row.children[5];
      if (!host_cell || row.querySelector("[data-instra-run-shape-cell]")) return result;
      let marker = host_cell;
      for (const definition of run_shape_columns) {
        const cell = document.createElement("td");
        cell.dataset.instraRunShapeCell = definition.key;
        cell.className = definition.numeric === false
          ? "run-shape-column run-preset-column"
          : "numeric-column run-shape-column";
        const raw = definition.value(run);
        const full_text = definition.key === "activation_checkpointing"
          ? raw
          : display_config_value(raw);
        cell.textContent = definition.key === "preset"
          ? full_text.slice(0, 9)
          : full_text;
        cell.title = `${definition.title}: ${full_text}`;
        marker.insertAdjacentElement("afterend", cell);
        marker = cell;
      }
      return result;
    };

    const install_visibility_header_toggle = () => {
      const header = document.querySelector(".runs-table thead .visibility-column");
      if (!header || by_id("workspace_visibility_all")) return;
      header.replaceChildren();
      const button = document.createElement("button");
      button.id = "workspace_visibility_all";
      button.type = "button";
      button.className = "eye-button visibility-header-toggle";
      button.setAttribute("aria-label", "Show or hide all runs in Workspace");
      button.addEventListener("click", () => {
        const should_show = app.runs.some(run => !is_visible(run_identifier(run)));
        for (const run of app.runs) app.visibility[run_identifier(run)] = should_show;
        save_json("thog2_local_run_visibility", app.visibility);
        render_runs();
      });
      header.appendChild(button);
    };

    const update_visibility_header_toggle = () => {
      install_visibility_header_toggle();
      const button = by_id("workspace_visibility_all");
      if (!button) return;
      const all_visible = app.runs.length > 0 && app.runs.every(run => is_visible(run_identifier(run)));
      button.replaceChildren(icon_svg(all_visible ? "eye_open" : "eye_closed"));
      button.title = all_visible ? "Hide all runs from Workspace" : "Show all runs in Workspace";
      button.setAttribute("aria-label", button.title);
    };

    const base_render_runs_weight_controls = render_runs;
    render_runs = function() {
      install_run_shape_headers();
      const result = base_render_runs_weight_controls();
      update_visibility_header_toggle();
      const column_count = document.querySelectorAll(".runs-table thead th").length;
      document.querySelectorAll(".runs-table .group-row td").forEach(cell => {
        cell.colSpan = column_count;
      });
      return result;
    };

    const rename_weight_label = trace => {
      const meta = trace?.meta;
      if (!meta || typeof meta !== "object" || Array.isArray(meta)) return;
      const residual = finite_integer(meta.instra_weight_model_feature);
      const branch = finite_integer(meta.instra_weight_intermediate_feature);
      if (residual === null || branch === null) return;
      const replacement = `residual feature ${residual} · branch feature ${branch}`;
      const patterns = [
        /model\s+\d+\s+·\s+attention feature\s+\d+/g,
        /model\s+\d+\s+·\s+MLP feature\s+\d+/g,
        /model feature\s+\d+\s*[,·]\s*intermediate feature\s+\d+/gi,
      ];
      for (const field of ["name", "hovertemplate"]) {
        let text = String(trace?.[field] || "");
        for (const pattern of patterns) text = text.replace(pattern, replacement);
        trace[field] = text;
      }
    };

    const base_prepare_figure_weight_controls = prepare_figure;
    prepare_figure = function(figure, chart_name) {
      const prepared = base_prepare_figure_weight_controls(figure, chart_name);
      if (!weight_chart_set.has(chart_name)) return prepared;
      for (const trace of prepared.data || []) rename_weight_label(trace);
      if (app.workspace_mode !== true) return prepared;

      const override = app.chart_settings_render_override;
      const settings = normalize_chart_settings(
        chart_name,
        override?.chart_name === chart_name ? override.settings : null,
      );
      const multiplier = Number(settings?.line_width);
      const width = workspace_weight_baseline_px * (
        Number.isFinite(multiplier) ? Math.min(3, Math.max(0.5, multiplier)) : 1
      );
      for (const trace of prepared.data || []) {
        const meta = trace?.meta;
        if (meta?.instra_top_axis_anchor === true) continue;
        const mode = String(trace?.mode || "");
        if (!mode.includes("lines")) continue;
        trace.line = {...(trace.line || {}), width};
      }
      return prepared;
    };

    // Matched coordinate selection is a Weights-group concern now; retain the
    // established hidden controls only as the compatibility save mechanism.
    const style = document.createElement("style");
    style.textContent = `
      #chart_matched_weight_section { display: none !important; }
      #coefficients_chart_group > .chart-group-header { min-width: 0; }
      .weight-index-group-controls {
        min-width: 0; height: 100%; margin-left: auto; display: inline-flex;
        align-items: center; gap: 5px; color: #5b6470; font-size: 10px; white-space: nowrap;
      }
      .weight-index-group-summary {
        max-width: 330px; overflow: hidden; text-overflow: ellipsis; font-variant-numeric: tabular-nums;
      }
      .weight-index-step-button {
        height: 25px; min-width: 36px; padding: 0 7px; border: 1px solid #d4d8de;
        border-radius: 4px; background: #f7f8fa; color: #4f5863; font-size: 10px; cursor: pointer;
      }
      .weight-index-step-button:hover:not(:disabled) { border-color: #9bcfd8; background: #e8f7f9; color: #007f98; }
      .weight-index-step-button:disabled { opacity: .42; cursor: default; }
      .weight-index-group-controls.saving { opacity: .65; }
      #coefficients_chart_group .weights-group-settings-button {
        margin-left: 18px !important; margin-right: 8px !important;
      }
      .chart-card .explicit-trajectory-modes { margin-right: 10px; }
      .chart-card.maximized .explicit-trajectory-modes {
        position: fixed !important; z-index: 70 !important;
        top: calc(var(--topbar-height) + 115px) !important; right: 104px !important;
        display: inline-flex !important; visibility: visible !important; opacity: 1 !important;
        pointer-events: auto !important; margin-right: 0 !important;
      }
      .chart-card.maximized .explicit-trajectory-modes .explicit-mode-button {
        display: inline-flex !important; visibility: visible !important;
      }
      .runs-table { min-width: 1560px; }
      .run-shape-column { width: 52px !important; text-align: right !important; font-variant-numeric: tabular-nums; }
      .run-preset-column { width: 76px !important; text-align: left !important; text-transform: none !important; }
      [data-instra-run-shape-header="grad_accum"], [data-instra-run-shape-cell="grad_accum"] { width: 64px !important; }
      .visibility-header-toggle { margin: 0 auto; }
    `;
    document.head.appendChild(style);

    install_run_shape_headers();
    install_visibility_header_toggle();
    install_weight_index_controls();
    update_weight_index_controls();
    update_visibility_header_toggle();
    render_runs();

    // First snapshots and mixed-generation Workspace capability can arrive after
    // page load; this refresh only updates tiny header controls and does no Plotly work.
    setInterval(update_weight_index_controls, 750);
  }, 1550);
});
// ^^^ THOG
