// vvv THOG
"use strict";

// Keep Runs-view trace colours aligned with the repaired per-chart current-only
// semantics.  A late legacy layer can otherwise recolour a historical chart from
// its obsolete global current-only flag even though the data filtering is correct.
window.addEventListener("load", () => {
  setTimeout(() => {
    const weight_chart_set = new Set(depth_weight_chart_names || []);

    const trace_update = trace => {
      try {
        const value = trace_optimizer_update(trace);
        return Number.isFinite(value) ? Number(value) : null;
      } catch (_error) {
        return null;
      }
    };

    const trace_key = trace => {
      const meta = trace?.meta && typeof trace.meta === "object" && !Array.isArray(trace.meta)
        ? trace.meta
        : {};
      return JSON.stringify([
        trace_update(trace),
        String(trace?.name || ""),
        String(meta.instra_workspace_run_id || ""),
        String(meta.instra_weight_selection_kind || ""),
        Number(meta.instra_weight_model_feature),
        Number(meta.instra_weight_intermediate_feature),
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

    const restore_historical_colours = (prepared, colours) => {
      for (const trace of prepared?.data || []) {
        const original = colours.get(trace_key(trace));
        if (!original) continue;
        if (trace?.line && original.line !== undefined) trace.line = {...trace.line, color: original.line};
        if (trace?.marker && original.marker !== undefined) {
          trace.marker = {...trace.marker, color: original.marker};
        }
        if (trace?.marker?.line && original.marker_line !== undefined) {
          trace.marker.line = {...trace.marker.line, color: original.marker_line};
        }
      }
    };

    const apply_current_run_colour = prepared => {
      const run_colour = colour_for_run(String(app.current_run_id || ""));
      for (const trace of prepared?.data || []) {
        if (trace?.meta?.instra_top_axis_anchor === true) continue;
        const mode = String(trace?.mode || "");
        if (mode.includes("lines") && trace.line) trace.line = {...trace.line, color: run_colour};
        if (mode.includes("markers") || trace.marker) {
          trace.marker = {...(trace.marker || {}), color: run_colour};
          trace.marker.line = {...(trace.marker?.line || {}), color: run_colour};
        }
      }
    };

    const base_prepare_figure_weight_style_semantics = prepare_figure;
    prepare_figure = function(figure, chart_name) {
      const colours = weight_chart_set.has(chart_name) && app.workspace_mode !== true
        ? original_colours(figure)
        : null;
      const prepared = base_prepare_figure_weight_style_semantics(figure, chart_name);
      if (!weight_chart_set.has(chart_name) || app.workspace_mode === true) return prepared;

      const render_override = app.chart_settings_render_override;
      const supplied = render_override?.chart_name === chart_name ? render_override.settings : null;
      const settings = normalize_chart_settings(chart_name, supplied);
      const range_active = window.__instra_weight_step_filter?.active?.() === true;

      if (settings.current_weights_only === true) apply_current_run_colour(prepared);
      else if (!range_active && colours) restore_historical_colours(prepared, colours);
      return prepared;
    };
  }, 10);
});
// ^^^ THOG
