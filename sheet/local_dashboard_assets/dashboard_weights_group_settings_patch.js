// vvv THOG
"use strict";

function instra_weight_trace_update(trace) {
  const meta = trace?.meta;
  for (const value of [
    meta?.instra_workspace_optimizer_update,
    meta?.instra_dense_optimizer_update,
    meta?.instra_thog_optimizer_update,
  ]) {
    if (value === null || value === undefined || value === "") continue;
    const numeric = Number(value);
    if (Number.isFinite(numeric)) return numeric;
  }
  const description = `${trace?.name || ""} ${trace?.hovertemplate || ""}`;
  const step_match = description.match(/(?:^|[^A-Za-z0-9])step\s+(\d+)(?:\D|$)/i);
  if (step_match) return Number(step_match[1]);
  const update_match = description.match(/(?:^|[^A-Za-z0-9])U(\d+)(?:\D|$)/);
  return update_match ? Number(update_match[1]) : null;
}

function instra_enforce_workspace_latest_weights(prepared) {
  const latest_by_run = new Map();
  for (const trace of prepared.data || []) {
    const run_id = trace?.meta?.instra_workspace_run_id;
    const update = instra_weight_trace_update(trace);
    if (!run_id || !Number.isFinite(update)) continue;
    const key = String(run_id);
    latest_by_run.set(key, Math.max(update, latest_by_run.get(key) ?? -Infinity));
  }
  prepared.data = (prepared.data || []).filter(trace => {
    const run_id = trace?.meta?.instra_workspace_run_id;
    if (!run_id) return true;
    const key = String(run_id);
    if (!latest_by_run.has(key)) return true;
    const update = instra_weight_trace_update(trace);
    return Number.isFinite(update) && update === latest_by_run.get(key);
  });
  return prepared;
}

function instra_apply_weight_group_defaults(normalized, group, explicit, fields) {
  for (const field of fields) {
    if (Object.prototype.hasOwnProperty.call(explicit || {}, field)) {
      normalized[field] = explicit[field];
    } else if (Object.prototype.hasOwnProperty.call(group || {}, field)) {
      normalized[field] = group[field];
    }
  }
  return normalized;
}

// Final owner for Weights-group defaults and Workspace newest-step enforcement.
// Group defaults are scoped independently to one run or to Workspace. Explicit
// chart settings win; the chart editor can opt back into group inheritance.
window.addEventListener("load", () => {
  setTimeout(() => {
    const weight_chart_names = [...depth_weight_chart_names];
    const weight_chart_set = new Set(weight_chart_names);
    const group_storage_key = "thog2_local_weight_group_settings_v1";
    const override_storage_key = "thog2_local_weight_chart_overrides_v1";
    const scale_storage_key = "thog2_local_trajectory_scale_modes";
    const common_axis_fields = Object.freeze([
      "max_snapshots",
      "snapshot_window_mode",
      "exclude_outliers",
      "smoothing",
      "line_width",
      "chart_type",
      "show_grid",
    ]);
    const common_control_ids = Object.freeze([
      "chart_current_weights_only",
      "chart_join_with_line_segments",
      "chart_max_snapshots",
      "chart_snapshot_window_mode",
      "chart_exclude_outliers",
      "chart_smoothing",
      "chart_line_width",
      "chart_show_grid",
    ]);

    let weight_group_settings = load_json(group_storage_key, {});
    let weight_chart_overrides = load_json(override_storage_key, {});
    let group_editor_active = false;
    let group_editor_scope = null;
    let pending_chart_save = null;

    if (!weight_group_settings || typeof weight_group_settings !== "object" || Array.isArray(weight_group_settings)) {
      weight_group_settings = {};
    }
    if (!weight_chart_overrides || typeof weight_chart_overrides !== "object" || Array.isArray(weight_chart_overrides)) {
      weight_chart_overrides = {};
    }

    const own = (object, key) => (
      Boolean(object)
      && typeof object === "object"
      && !Array.isArray(object)
      && Object.prototype.hasOwnProperty.call(object, key)
    );

    const editor_workspace_mode = chart_name => (
      app.axis_chart_name === chart_name
      && typeof app.axis_chart_workspace_mode === "boolean"
        ? app.axis_chart_workspace_mode
        : app.workspace_mode === true
    );

    const weight_group_scope = (chart_name = app.axis_chart_name) => (
      editor_workspace_mode(chart_name)
        ? "workspace"
        : `run:${String(app.current_run_id || "unselected")}`
    );

    const weight_chart_scope = (chart_name, group_scope = weight_group_scope(chart_name)) => (
      `${group_scope}:${chart_name}`
    );

    const default_common_settings = () => ({
      current_weights_only: false,
      join_with_line_segments: false,
      max_snapshots: 0,
      snapshot_window_mode: "rolling",
      exclude_outliers: false,
      smoothing: 0,
      line_width: 1,
      chart_type: "lines",
      show_grid: true,
      scale_mode: "linear",
      inspection_precision: 4,
    });

    const normalize_common_settings = supplied => {
      const source = supplied && typeof supplied === "object" && !Array.isArray(supplied)
        ? supplied
        : {};
      const normalized = default_common_settings();
      normalized.current_weights_only = source.current_weights_only === true;
      normalized.join_with_line_segments = source.join_with_line_segments === true;
      const maximum = Number(source.max_snapshots);
      normalized.max_snapshots = Number.isInteger(maximum) && maximum > 0 ? maximum : 0;
      normalized.snapshot_window_mode = source.snapshot_window_mode === "from_zero" ? "from_zero" : "rolling";
      normalized.exclude_outliers = source.exclude_outliers === true;
      const smoothing = Number(source.smoothing);
      normalized.smoothing = Number.isFinite(smoothing) ? Math.min(0.95, Math.max(0, smoothing)) : 0;
      const line_width = Number(source.line_width);
      normalized.line_width = Number.isFinite(line_width) ? Math.min(3, Math.max(0.5, line_width)) : 1;
      normalized.chart_type = ["lines", "lines_markers", "markers"].includes(source.chart_type)
        ? source.chart_type
        : "lines";
      normalized.show_grid = source.show_grid !== false;
      normalized.scale_mode = source.scale_mode === "log" ? "log" : "linear";
      const precision = source.inspection_precision;
      normalized.inspection_precision = Number.isInteger(precision) && precision >= 0 && precision <= 12 ? precision : 4;
      return normalized;
    };

    const group_settings_for_scope = group_scope => {
      if (!own(weight_group_settings, group_scope)) return null;
      return normalize_common_settings(weight_group_settings[group_scope]);
    };

    const group_settings_for_chart = chart_name => (
      group_settings_for_scope(weight_group_scope(chart_name))
    );

    const legacy_chart_settings = (chart_name, group_scope) => {
      const settings = {};
      const raw_axis = stored_chart_settings(chart_name);
      for (const field of common_axis_fields) {
        if (own(raw_axis, field)) settings[field] = raw_axis[field];
      }
      const chart_scope = weight_chart_scope(chart_name, group_scope);
      if (own(app.weight_current_only, chart_scope)) {
        settings.current_weights_only = app.weight_current_only[chart_scope] === true;
      }
      if (own(app.weight_join_with_line_segments, chart_scope)) {
        settings.join_with_line_segments = app.weight_join_with_line_segments[chart_scope] === true;
      }
      return settings;
    };

    const explicit_chart_settings = (chart_name, group_scope) => {
      const legacy = legacy_chart_settings(chart_name, group_scope);
      const chart_scope = weight_chart_scope(chart_name, group_scope);
      const modern = own(weight_chart_overrides, chart_scope)
        ? weight_chart_overrides[chart_scope]
        : {};
      return {...legacy, ...(modern || {})};
    };

    const chart_has_common_override = (chart_name, group_scope = weight_group_scope(chart_name)) => (
      Object.keys(explicit_chart_settings(chart_name, group_scope)).length > 0
    );

    const common_settings_from_state = settings => normalize_common_settings(settings);

    const save_group_store = () => save_json(group_storage_key, weight_group_settings);
    const save_override_store = () => save_json(override_storage_key, weight_chart_overrides);

    const base_normalize_chart_settings = normalize_chart_settings;
    normalize_chart_settings = function(chart_name, supplied = null) {
      const normalized = base_normalize_chart_settings(chart_name, supplied);
      if (supplied !== null || !weight_chart_set.has(chart_name)) return normalized;
      const group_scope = weight_group_scope(chart_name);
      const group = group_settings_for_scope(group_scope);
      if (!group) return normalized;
      const explicit = explicit_chart_settings(chart_name, group_scope);
      instra_apply_weight_group_defaults(
        normalized,
        group,
        explicit,
        Object.keys(default_common_settings()).filter(field => field !== "scale_mode"),
      );
      const common = normalize_common_settings({...normalized, scale_mode: group.scale_mode});
      for (const field of Object.keys(common)) {
        if (field !== "scale_mode") normalized[field] = common[field];
      }
      return normalized;
    };

    const base_prepare_figure_weights_group = prepare_figure;
    prepare_figure = function(figure, chart_name) {
      const prepared = base_prepare_figure_weights_group(figure, chart_name);
      if (!app.workspace_mode || !weight_chart_set.has(chart_name)) return prepared;
      const render_override = app.chart_settings_render_override;
      const settings = render_override?.chart_name === chart_name
        ? render_override.settings
        : normalize_chart_settings(chart_name);
      if (settings?.current_weights_only === true) instra_enforce_workspace_latest_weights(prepared);
      return prepared;
    };

    const common_controls = () => [
      ...common_control_ids.map(by_id).filter(Boolean),
      ...document.querySelectorAll('input[name="chart_type"]'),
    ];

    const write_common_controls = settings => {
      const normalized = normalize_common_settings(settings);
      by_id("chart_current_weights_only").checked = normalized.current_weights_only;
      by_id("chart_join_with_line_segments").checked = normalized.join_with_line_segments;
      const maximum = Math.max(1, Number(by_id("chart_max_snapshots").max || 1));
      by_id("chart_max_snapshots").value = String(
        normalized.max_snapshots > 0 ? Math.min(normalized.max_snapshots, maximum) : 0
      );
      by_id("chart_snapshot_window_mode").value = normalized.snapshot_window_mode;
      by_id("chart_exclude_outliers").checked = normalized.exclude_outliers;
      by_id("chart_smoothing").value = String(normalized.smoothing);
      by_id("chart_line_width").value = String(normalized.line_width);
      const chart_type = document.querySelector(`input[name="chart_type"][value="${normalized.chart_type}"]`);
      if (chart_type) chart_type.checked = true;
      by_id("chart_show_grid").checked = normalized.show_grid;
      if (group_editor_active && by_id("weights_group_precision")) {
        by_id("weights_group_precision").value = String(normalized.inspection_precision);
      }
    };

    const install_group_form_fields = () => {
      const curve_section = by_id("chart_curve_display_options");
      const current_field = by_id("chart_current_weights_only_field");
      if (!curve_section || !current_field) return;

      if (!by_id("chart_inherit_weights_group_field")) {
        const inherit_field = document.createElement("label");
        inherit_field.id = "chart_inherit_weights_group_field";
        inherit_field.className = "chart-toggle-row";
        inherit_field.htmlFor = "chart_inherit_weights_group";
        inherit_field.innerHTML = (
          "<span><strong>Use Weights group settings</strong>"
          + "<small>Inherited common display settings; this chart's title and axes remain individual.</small></span>"
        );
        const checkbox = document.createElement("input");
        checkbox.id = "chart_inherit_weights_group";
        checkbox.type = "checkbox";
        inherit_field.appendChild(checkbox);
        current_field.insertAdjacentElement("beforebegin", inherit_field);
        checkbox.addEventListener("change", () => {
          if (checkbox.checked) {
            write_common_controls(group_settings_for_chart(app.axis_chart_name) || default_common_settings());
          }
          sync_chart_setting_outputs();
          schedule_chart_settings_preview();
        });
      }

      if (!by_id("weights_group_scale_field")) {
        const scale_field = document.createElement("label");
        scale_field.id = "weights_group_scale_field";
        scale_field.className = "chart-editor-field chart-editor-field-wide";
        scale_field.hidden = true;
        scale_field.innerHTML = "<span>Y scale</span>";
        const select = document.createElement("select");
        select.id = "weights_group_scale_mode";
        select.innerHTML = '<option value="linear">Linear</option><option value="log">Signed log</option>';
        scale_field.appendChild(select);
        current_field.insertAdjacentElement("beforebegin", scale_field);
      }
    };

    install_group_form_fields();

    const inheritance_checkbox = () => by_id("chart_inherit_weights_group");

    // vvv THOG canonical group/override owner captures these toggles before legacy preview listeners can rewrite them
    const preserve_weight_toggle = event => {
      if (!weight_chart_set.has(app.axis_chart_name)) return;
      const target = event.currentTarget;
      if (!target) return;
      if (!group_editor_active && inheritance_checkbox()?.checked === true) {
        inheritance_checkbox().checked = false;
      }
      event.stopImmediatePropagation();
    };
    for (const id of ["chart_current_weights_only", "chart_join_with_line_segments"]) {
      const control = by_id(id);
      control?.addEventListener("input", preserve_weight_toggle, true);
      control?.addEventListener("change", preserve_weight_toggle, true);
    }
    // ^^^ THOG

    const base_sync_chart_setting_outputs = sync_chart_setting_outputs;
    sync_chart_setting_outputs = function() {
      base_sync_chart_setting_outputs();
      const chart_name = app.axis_chart_name;
      const weight_chart = weight_chart_set.has(chart_name);
      const inherit_field = by_id("chart_inherit_weights_group_field");
      const inherit = weight_chart && !group_editor_active && inheritance_checkbox()?.checked === true;
      if (inherit_field) inherit_field.hidden = !weight_chart || group_editor_active;
      by_id("weights_group_scale_field").hidden = !group_editor_active;
      if (by_id("weights_group_precision_field")) by_id("weights_group_precision_field").hidden = !group_editor_active;
      for (const control of common_controls()) control.disabled = inherit;
      if (!inherit) {
        const current_only = weight_chart && by_id("chart_current_weights_only").checked;
        by_id("chart_max_snapshots").disabled = current_only;
        by_id("chart_snapshot_window_mode").disabled = current_only;
      }
    };

    const base_populate_chart_settings_form = populate_chart_settings_form;
    populate_chart_settings_form = function(chart_name, supplied = null) {
      base_populate_chart_settings_form(chart_name, supplied);
      if (!weight_chart_set.has(chart_name)) return;
      const group_scope = group_editor_active ? group_editor_scope : weight_group_scope(chart_name);
      const inherit = !group_editor_active && !chart_has_common_override(chart_name, group_scope);
      inheritance_checkbox().checked = inherit;
      if (group_editor_active) {
        write_common_controls(group_settings_for_scope(group_scope) || default_common_settings());
      } else if (inherit) {
        write_common_controls(group_settings_for_scope(group_scope) || default_common_settings());
      }
      sync_chart_setting_outputs();
      if (group_editor_active) {
        write_common_controls(group_settings_for_scope(group_scope) || default_common_settings());
      }
    };

    const clear_legacy_common_settings = (chart_name, group_scope) => {
      const raw_axis = stored_chart_settings(chart_name);
      for (const field of common_axis_fields) delete raw_axis[field];
      if (!Object.keys(raw_axis).length) delete app.axis_ranges[chart_name];
      const chart_scope = weight_chart_scope(chart_name, group_scope);
      delete app.weight_current_only[chart_scope];
      delete app.weight_join_with_line_segments[chart_scope];
      save_json("thog2_local_chart_axis_ranges", app.axis_ranges);
      save_json(weight_current_only_storage_key, app.weight_current_only);
      save_json(weight_join_with_line_segments_storage_key, app.weight_join_with_line_segments);
    };

    const save_chart_override = pending => {
      const chart_scope = weight_chart_scope(pending.chart_name, pending.group_scope);
      if (pending.inherit) {
        delete weight_chart_overrides[chart_scope];
        clear_legacy_common_settings(pending.chart_name, pending.group_scope);
      } else {
        weight_chart_overrides[chart_scope] = common_settings_from_state(pending.settings);
      }
      save_override_store();
    };

    const update_group_button = () => {
      const button = by_id("weights_group_settings_button");
      if (!button) return;
      const group_scope = weight_group_scope();
      button.classList.toggle("active", own(weight_group_settings, group_scope));
      button.title = editor_workspace_mode() ? "Workspace Weights settings" : "Run Weights settings";
    };

    const cleanup_group_editor = () => {
      group_editor_active = false;
      group_editor_scope = null;
      const data_tab = document.querySelector('[data-chart-settings-tab="data"]');
      if (data_tab) data_tab.hidden = false;
      by_id("weights_group_scale_field").hidden = true;
      if (by_id("weights_group_precision_field")) by_id("weights_group_precision_field").hidden = true;
      update_group_button();
    };

    const first_weight_chart = () => (
      weight_chart_names.find(chart_name => figure_for_chart(chart_name)) || weight_chart_names[0]
    );

    const open_group_editor = () => {
      const chart_name = first_weight_chart();
      group_editor_active = true;
      group_editor_scope = weight_group_scope(chart_name);
      open_chart_settings(chart_name);
      const group = group_settings_for_scope(group_editor_scope) || default_common_settings();
      const base = base_normalize_chart_settings(chart_name, {});
      populate_chart_settings_form(chart_name, {...base, ...group});
      by_id("weights_group_scale_mode").value = group.scale_mode;
      by_id("chart_settings_title").textContent = editor_workspace_mode(chart_name)
        ? "Workspace Weights settings"
        : "Run Weights settings";
      by_id("chart_settings_axes").textContent = (
        "Common display settings apply to all six weight charts. Individual chart overrides take precedence."
      );
      const data_tab = document.querySelector('[data-chart-settings-tab="data"]');
      if (data_tab) data_tab.hidden = true;
      set_chart_settings_tab("display");
      sync_chart_setting_outputs();
      // vvv THOG reassert canonical group values after every legacy sync wrapper has run
      write_common_controls(group);
      by_id("weights_group_scale_mode").value = group.scale_mode;
      queueMicrotask(() => {
        if (!group_editor_active) return;
        const current_group = group_settings_for_scope(group_editor_scope) || default_common_settings();
        write_common_controls(current_group);
        by_id("weights_group_scale_mode").value = current_group.scale_mode;
      });
      // ^^^ THOG
    };

    // const apply_group_settings = () => {
    //   const state = chart_settings_form_state();
    //   if (state.error) {
    //     by_id("chart_settings_error").textContent = state.error;
    //     by_id("chart_settings_error").hidden = false;
    //     return;
    //   }
    //   const group = common_settings_from_state(state.settings);
    //   group.scale_mode = by_id("weights_group_scale_mode").value === "log" ? "log" : "linear";
    //   weight_group_settings[group_editor_scope] = group;
    //   save_group_store();
    //   const scales = load_json(scale_storage_key, {});
    //   for (const chart_name of weight_chart_names) scales[chart_name] = group.scale_mode;
    //   save_json(scale_storage_key, scales);
    //   cleanup_group_editor();
    //   close_chart_settings();
    //   update_group_button();
    //   render_figures().catch(error => show_toast(`Weights settings failed: ${error.message}`));
    //   show_toast("Weights group settings applied.");
    // };
    // vvv THOG make group Apply transactional with respect to its redraw: clear preview ownership, await one six-chart render, then close without launching the legacy duplicate redraw
    const apply_group_settings = async () => {
      const state = chart_settings_form_state();
      const precision_text = by_id("weights_group_precision")?.value ?? "4";
      const precision = Number(precision_text);
      if (precision_text.trim() === "" || !Number.isInteger(precision) || precision < 0 || precision > 12) {
        state.error = "Weight inspection precision must be a whole number from 0 to 12.";
      }
      if (state.error) {
        by_id("chart_settings_error").textContent = state.error;
        by_id("chart_settings_error").hidden = false;
        return;
      }
      const group = common_settings_from_state(state.settings);
      group.inspection_precision = precision;
      group.scale_mode = by_id("weights_group_scale_mode").value === "log" ? "log" : "linear";
      weight_group_settings[group_editor_scope] = group;
      save_group_store();
      const scales = load_json(scale_storage_key, {});
      for (const chart_name of weight_chart_names) scales[chart_name] = group.scale_mode;
      save_json(scale_storage_key, scales);

      const save_button = by_id("save_chart_settings");
      if (save_button) save_button.disabled = true;
      clearTimeout(app.chart_settings_preview_timer);
      app.chart_settings_preview_serial += 1;
      app.chart_settings_render_override = null;

      try {
        await render_figures();
      } catch (error) {
        show_toast(`Weights settings failed: ${error.message}`);
        return;
      } finally {
        if (save_button) save_button.disabled = false;
      }

      cleanup_group_editor();
      const saved_render_axis_settings_change = render_axis_settings_change;
      render_axis_settings_change = () => undefined;
      try {
        close_chart_settings();
      } finally {
        render_axis_settings_change = saved_render_axis_settings_change;
      }
      update_group_button();
      show_toast("Weights group settings applied.");
    };
    // ^^^ THOG

    const install_group_button = () => {
      const header = by_id("coefficients_chart_group")?.querySelector(".chart-group-header");
      if (!header || by_id("weights_group_settings_button")) return;
      const button = document.createElement("button");
      button.id = "weights_group_settings_button";
      button.type = "button";
      button.className = "weights-group-settings-button";
      button.appendChild(chart_settings_icon());
      button.setAttribute("aria-label", "Weights group settings");
      button.addEventListener("click", event => {
        event.preventDefault();
        event.stopPropagation();
        open_group_editor();
      });
      header.appendChild(button);
      update_group_button();
    };

    const style = document.createElement("style");
    style.textContent = `
      .weights-group-settings-button {
        width: 28px; height: 28px; margin: 0 7px 0 auto; padding: 5px;
        display: inline-flex; align-items: center; justify-content: center;
        border: 1px solid transparent; border-radius: 4px; background: transparent;
        color: #59636f; cursor: pointer;
      }
      .weights-group-settings-button:hover { background: #f2f4f6; border-color: #cfd5dc; }
      .weights-group-settings-button.active { color: #008da5; background: #e8f8fb; border-color: #a8dae2; }
      .weights-group-settings-button svg { width: 16px; height: 16px; fill: none; stroke: currentColor; stroke-width: 1.8; }
    `;
    document.head.appendChild(style);
    install_group_button();

    by_id("save_chart_settings")?.addEventListener("click", event => {
      if (group_editor_active) {
        event.preventDefault();
        event.stopImmediatePropagation();
        apply_group_settings();
        return;
      }
      const chart_name = app.axis_chart_name;
      if (!weight_chart_set.has(chart_name)) return;
      const state = chart_settings_form_state();
      if (state.error) return;
      pending_chart_save = {
        chart_name,
        group_scope: weight_group_scope(chart_name),
        inherit: inheritance_checkbox()?.checked === true,
        settings: state.settings,
      };
    }, true);

    by_id("save_chart_settings")?.addEventListener("click", () => {
      if (!pending_chart_save) return;
      const pending = pending_chart_save;
      pending_chart_save = null;
      save_chart_override(pending);
      render_figures().catch(error => show_toast(`Chart settings failed: ${error.message}`));
    });

    for (const id of ["close_chart_settings", "cancel_chart_settings"]) {
      by_id(id)?.addEventListener("click", cleanup_group_editor);
    }
    by_id("chart_settings_overlay")?.addEventListener("pointerdown", event => {
      if (event.target === by_id("chart_settings_overlay")) cleanup_group_editor();
    });

    const base_render_run_heading_weights_group = render_run_heading;
    render_run_heading = function() {
      const result = base_render_run_heading_weights_group();
      update_group_button();
      return result;
    };

    window.__instra_weight_group_settings = {
      enforce_workspace_latest_weights: instra_enforce_workspace_latest_weights,
      group_settings_for_scope,
      normalize_common_settings,
      weight_group_scope,
    };
  }, 0);
});
// ^^^ THOG
