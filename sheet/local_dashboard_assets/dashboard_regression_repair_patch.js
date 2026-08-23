// vvv THOG
"use strict";

// Final regression repair for two late-dashboard interactions:
// 1. Current-only / join-segments are global Weights settings and must be editable
//    from either the group editor or any individual weight-chart editor.
// 2. Legacy heatmaps without centre-loss metadata cannot support percentage Δloss;
//    render their retained raw Δloss in absolute mode instead of blanking every cell.
window.addEventListener("load", () => {
  setTimeout(() => {
    const weight_chart_set = new Set(depth_weight_chart_names || []);

    const finite_number = value => {
      if (value === null || value === undefined || value === "") return null;
      const numeric = Number(value);
      return Number.isFinite(numeric) ? numeric : null;
    };

    const weight_editor_open = () => (
      !by_id("chart_settings_overlay")?.hidden
      && weight_chart_set.has(app.axis_chart_name)
    );

    const step_range_active = () => (
      typeof window.__instra_weight_step_filter?.active === "function"
      && window.__instra_weight_step_filter.active()
    );

    const weight_api = () => window.__instra_weight_controls_v2 || window.__instra_weight_step_controls || null;

    const repair_depth_payload_shape = () => {
      if (!app.figures || typeof app.figures !== "object" || app.figures.depth !== null) return;
      app.figures = {...app.figures, depth: {}};
    };

    const install_safe_global_flag_writer = () => {
      const api = window.__instra_weight_controls_v2;
      if (!api || typeof api.set_global_flags !== "function" || api.__instra_safe_flag_writer === true) return;
      const base_set_global_flags = api.set_global_flags.bind(api);
      api.set_global_flags = function(...args) {
        const result = base_set_global_flags(...args);
        repair_depth_payload_shape();
        return result;
      };
      api.__instra_safe_flag_writer = true;
    };

    const sync_global_weight_controls = () => {
      install_safe_global_flag_writer();
      if (!weight_editor_open()) return;
      const api = weight_api();
      const flags = typeof api?.global_flags === "function" ? api.global_flags() : null;
      const current = by_id("chart_current_weights_only");
      const join = by_id("chart_join_with_line_segments");
      if (current) {
        if (flags) current.checked = flags.current_weights_only === true;
        current.disabled = step_range_active();
        current.title = current.disabled
          ? "Current weights only is overridden while an explicit weight-step range is active."
          : "Global across all six weight charts and every run.";
      }
      if (join) {
        if (flags) join.checked = flags.join_with_line_segments === true;
        join.disabled = false;
        join.title = "Global across all six weight charts and every run.";
      }
    };

    const base_sync_chart_setting_outputs_regression = sync_chart_setting_outputs;
    sync_chart_setting_outputs = function() {
      const result = base_sync_chart_setting_outputs_regression();
      sync_global_weight_controls();
      return result;
    };

    const base_populate_chart_settings_form_regression = populate_chart_settings_form;
    populate_chart_settings_form = function(chart_name, supplied = null) {
      const result = base_populate_chart_settings_form_regression(chart_name, supplied);
      if (weight_chart_set.has(chart_name)) sync_global_weight_controls();
      return result;
    };

    const base_open_chart_settings_regression = open_chart_settings;
    open_chart_settings = function(chart_name) {
      const result = base_open_chart_settings_regression(chart_name);
      if (weight_chart_set.has(chart_name)) queueMicrotask(sync_global_weight_controls);
      return result;
    };

    // The older group-save handler is allowed to run first. This capture listener
    // then updates the same global store when an individual editor is used; when
    // the group handler already wrote identical values, set_global_flags is a no-op.
    window.addEventListener("click", event => {
      const button = event.target.closest?.("#save_chart_settings");
      if (!button || !weight_editor_open()) return;
      install_safe_global_flag_writer();
      const api = weight_api();
      if (typeof api?.set_global_flags !== "function") return;
      const current = by_id("chart_current_weights_only");
      const join = by_id("chart_join_with_line_segments");
      const before = typeof api.global_flags === "function" ? api.global_flags() : {};
      const next = {
        current_weights_only: current?.checked === true,
        join_with_line_segments: join?.checked === true,
      };
      if (
        before.current_weights_only !== next.current_weights_only
        || before.join_with_line_segments !== next.join_with_line_segments
      ) api.set_global_flags(next);
      repair_depth_payload_shape();
    }, true);

    const clamp_01 = value => Math.max(0, Math.min(1, Number(value)));
    const viewer_limit = (name, fallback) => {
      const value = finite_number(heatmap_settings_for_current_run()?.[name]);
      return value !== null && value > 0 ? value : fallback;
    };
    const manual_band_limits = () => {
      const base_limit = heatmap_abs_limit(0.05);
      return {
        green: Math.min(0.1, viewer_limit("negative_abs_limit", Math.min(0.1, base_limit))),
        blue: Math.min(1.0, Math.max(0.100000001, viewer_limit("blue_abs_limit", 1.0))),
        yellow: Math.max(1.000000001, viewer_limit("yellow_abs_limit", 2.0)),
        red: viewer_limit("positive_abs_limit", base_limit),
      };
    };
    const limits_from_values = values => {
      const limits = {green: 0, blue: 0, yellow: 0, red: 0};
      for (const value of values) {
        if (!Number.isFinite(value)) continue;
        if (value <= -1.0) limits.yellow = Math.max(limits.yellow, Math.abs(value));
        else if (value <= -0.1) limits.blue = Math.max(limits.blue, Math.abs(value));
        else if (value < 0) limits.green = Math.max(limits.green, Math.abs(value));
        else if (value > 0) limits.red = Math.max(limits.red, value);
      }
      return limits;
    };
    const band_value = (value, limits) => {
      if (!Number.isFinite(value)) return null;
      if (value <= -1.0) {
        const denominator = Math.max(1e-12, limits.yellow - 1.0);
        return -0.76 - 0.24 * clamp_01((Math.abs(value) - 1.0) / denominator);
      }
      if (value <= -0.1) {
        const denominator = Math.max(1e-12, limits.blue - 0.1);
        return -0.51 - 0.23 * clamp_01((Math.abs(value) - 0.1) / denominator);
      }
      if (value < 0) {
        const intensity = limits.green > 0 ? clamp_01(Math.abs(value) / limits.green) : 1;
        return -0.01 - 0.48 * intensity;
      }
      if (value > 0) {
        const intensity = limits.red > 0 ? clamp_01(value / limits.red) : 1;
        return 0.01 + 0.99 * intensity;
      }
      return 0;
    };
    const format_limit = (value, sign) => (
      Number(value) > 0 ? `${sign}${Number(value).toPrecision(3)}` : "—"
    );

    const percent_requested = () => (
      heatmap_settings_for_current_run()?.delta_loss_display_mode !== "absolute"
    );

    const heatmap_needs_absolute_fallback = prepared => {
      if (!percent_requested()) return false;
      const heatmap = (prepared.data || []).find(trace => trace.type === "heatmap");
      if (!heatmap || !Array.isArray(heatmap.customdata)) return false;
      const current_losses = Array.isArray(prepared.layout?.meta?.thog2_current_losses)
        ? prepared.layout.meta.thog2_current_losses
        : [];
      let saw_data = false;
      for (let row_index = 0; row_index < heatmap.customdata.length; row_index += 1) {
        const row = heatmap.customdata[row_index];
        if (!Array.isArray(row)) continue;
        const row_has_delta = row.some(cell => (
          Array.isArray(cell) && finite_number(cell[3]) !== null
        ));
        if (!row_has_delta) continue;
        saw_data = true;
        const current_loss = finite_number(current_losses[row_index]);
        if (current_loss === null || current_loss === 0) return true;
      }
      return saw_data && current_losses.length === 0;
    };

    const apply_legacy_absolute_heatmap = prepared => {
      const heatmap = (prepared.data || []).find(trace => trace.type === "heatmap");
      if (!heatmap || !Array.isArray(heatmap.customdata)) return false;
      const raw_values = [];
      for (const row of heatmap.customdata) {
        if (!Array.isArray(row)) continue;
        for (const cell of row) {
          if (!Array.isArray(cell)) continue;
          const raw_delta = finite_number(cell[3]);
          if (raw_delta === null) continue;
          cell[5] = raw_delta;
          raw_values.push(raw_delta);
        }
      }
      const settings = heatmap_settings_for_current_run() || {};
      const limits = settings.auto_colour_saturation === true
        ? limits_from_values(raw_values)
        : manual_band_limits();
      heatmap.z = heatmap.customdata.map(row => (
        Array.isArray(row)
          ? row.map(cell => {
              const raw_delta = Array.isArray(cell) ? finite_number(cell[3]) : null;
              return raw_delta === null ? null : band_value(raw_delta, limits);
            })
          : row
      ));
      heatmap.zmin = -1;
      heatmap.zmax = 1;
      heatmap.zmid = 0;
      heatmap.colorscale = [
        [0.000, "rgb(255,226,0)"],
        [0.120, "rgb(108,96,43)"],
        [0.130, "rgb(0,126,255)"],
        [0.245, "rgb(48,72,104)"],
        [0.255, "rgb(0,255,0)"],
        [0.495, "rgb(72,96,72)"],
        [0.500, "rgb(88,88,88)"],
        [0.505, "rgb(112,76,76)"],
        [1.000, "rgb(255,0,0)"],
      ];
      heatmap.hovertemplate = (
        "step=%{customdata[0]}<br>"
        + "layer count (abs) = %{customdata[1]}<br>"
        + "layer count (rel) = %{customdata[2]}<br>"
        + "Δloss=%{customdata[3]:.8f}<extra></extra>"
      );
      heatmap.colorbar = {
        ...(heatmap.colorbar || {}),
        tickmode: "array",
        tickvals: [-1, -0.76, -0.74, -0.51, -0.49, 0, 1],
        ticktext: [
          `yellow ${format_limit(limits.yellow, "−")}`,
          "yellow ≤ −1",
          `blue ${format_limit(limits.blue, "−")}`,
          "blue ≤ −0.1",
          `green ${format_limit(limits.green, "−")}`,
          "0",
          `red ${format_limit(limits.red, "+")}`,
        ],
        title: "Δloss bands (legacy absolute fallback)",
      };
      prepared.layout = prepared.layout || {};
      prepared.layout.meta = {
        ...(prepared.layout.meta || {}),
        thog2_legacy_absolute_fallback: true,
      };
      return true;
    };

    const sync_legacy_heatmap_button = fallback => {
      const button = by_id("heatmap_delta_loss_mode");
      if (!button) return;
      if (!fallback) {
        button.disabled = false;
        return;
      }
      button.textContent = "|abs|";
      button.dataset.mode = "absolute-legacy";
      button.disabled = true;
      button.setAttribute("aria-pressed", "false");
      button.title = "Percentage Δloss is unavailable for this legacy run because centre-loss metadata was not recorded; showing absolute Δloss.";
      button.setAttribute("aria-label", button.title);
    };

    const base_transpose_heatmap_regression = transpose_heatmap;
    transpose_heatmap = function(prepared) {
      const result = base_transpose_heatmap_regression(prepared);
      const fallback = heatmap_needs_absolute_fallback(prepared);
      if (fallback) apply_legacy_absolute_heatmap(prepared);
      else if (prepared.layout?.meta) delete prepared.layout.meta.thog2_legacy_absolute_fallback;
      queueMicrotask(() => sync_legacy_heatmap_button(fallback));
      return result;
    };

    window.__instra_regression_repair = Object.freeze({
      weight_editor_open,
      sync_global_weight_controls,
      repair_depth_payload_shape,
      heatmap_needs_absolute_fallback,
      apply_legacy_absolute_heatmap,
    });

    install_safe_global_flag_writer();
    sync_global_weight_controls();
  }, 2200);
});
// ^^^ THOG
