// vvv THOG
"use strict";

// Keep explicit matched-weight selection stable in mixed-generation Workspaces.
// Older runs can remain visible for ordinary/random weight views, but when a
// user-selected coordinate is active they are treated as incompatible rather
// than disabling matched selection for every newer run. If a visible run did not
// record the currently selected coupling, retain that run's actually recorded
// random/legacy coupling rather than rendering an empty Workspace chart. Workspace
// line widths are also normalised so THOG age styling and DENSE source styling do
// not imply a false difference between runs. Workspace colours are always the
// owning run colour, including DENSE traces and THOG executed-layer overlays.
window.addEventListener("load", () => {
  setTimeout(() => {
    const protocol = "matched_six_v1";
    const weight_chart_set = new Set([...depth_weight_chart_names]);

    const finite_integer = value => {
      if (value === null || value === undefined || value === "") return null;
      const numeric = Number(value);
      return Number.isInteger(numeric) ? numeric : null;
    };

    const compatible_feature_count = figure => {
      const counts = [];
      for (const trace of figure?.data || []) {
        const meta = trace?.meta;
        if (!meta || typeof meta !== "object" || Array.isArray(meta)) continue;
        if (meta.instra_weight_selection_protocol !== protocol) continue;
        const count = finite_integer(meta.instra_weight_feature_count);
        if (count !== null && count > 0) counts.push(count);
      }
      return counts.length ? Math.min(...counts) : null;
    };

    const mark_incompatible_workspace_traces = (figure, chart_name) => {
      if (
        app.workspace_mode !== true
        || !weight_chart_set.has(chart_name)
        || !figure
      ) return figure;

      const feature_count = compatible_feature_count(figure);
      if (feature_count === null) return figure;

      let changed = false;
      const data = (figure.data || []).map(trace => {
        const meta = trace?.meta;
        if (!meta || typeof meta !== "object" || Array.isArray(meta)) return trace;
        if (!meta.instra_workspace_run_id) return trace;
        if (meta.instra_weight_selection_protocol === protocol) return trace;
        changed = true;
        return {
          ...trace,
          meta: {
            ...meta,
            instra_weight_selection_protocol: protocol,
            instra_weight_selection_kind: "incompatible",
            instra_weight_feature_count: feature_count,
          },
        };
      });
      return changed ? {...figure, data} : figure;
    };

    const workspace_run_id = trace => {
      const value = trace?.meta?.instra_workspace_run_id;
      return value === null || value === undefined || value === "" ? null : String(value);
    };

    const selection_kind = trace => {
      const meta = trace?.meta;
      if (!meta || typeof meta !== "object" || Array.isArray(meta)) return null;
      if (meta.instra_weight_selection_protocol !== protocol) return null;
      return String(meta.instra_weight_selection_kind || "random");
    };

    const fallback_pair_key = trace => {
      const meta = trace?.meta;
      if (!meta || typeof meta !== "object" || Array.isArray(meta)) return null;
      const model_feature = finite_integer(meta.instra_weight_model_feature);
      const intermediate_feature = finite_integer(meta.instra_weight_intermediate_feature);
      if (model_feature !== null && intermediate_feature !== null) {
        return `logical:${model_feature}:${intermediate_feature}`;
      }
      const scalar_id = meta.instra_thog_scalar_id || meta.instra_dense_scalar_id;
      return scalar_id ? `scalar:${String(scalar_id)}` : null;
    };

    const clone_trace = trace => {
      if (typeof structuredClone === "function") {
        try { return structuredClone(trace); }
        catch (_error) { /* fall through to JSON clone */ }
      }
      return JSON.parse(JSON.stringify(trace));
    };

    const current_only = chart_name => {
      const override = app.chart_settings_render_override;
      const settings = normalize_chart_settings(
        chart_name,
        override?.chart_name === chart_name ? override.settings : null,
      );
      return settings?.current_weights_only === true;
    };

    const restore_missing_workspace_runs = (source, prepared, chart_name) => {
      if (
        app.workspace_mode !== true
        || !weight_chart_set.has(chart_name)
        || !current_only(chart_name)
        || !source
        || !prepared
      ) return prepared;

      const source_data = source.data || [];
      const prepared_data = prepared.data || [];
      const source_run_ids = new Set(source_data.map(workspace_run_id).filter(Boolean));
      const prepared_run_ids = new Set(prepared_data.map(workspace_run_id).filter(Boolean));

      for (const run_id of source_run_ids) {
        if (prepared_run_ids.has(run_id)) continue;
        const candidates = source_data.filter(trace => {
          if (workspace_run_id(trace) !== run_id) return false;
          const kind = selection_kind(trace);
          return kind === "random" || kind === "user_random" || kind === "incompatible";
        });
        if (!candidates.length) continue;

        const preferred_key = candidates.map(fallback_pair_key).find(Boolean);
        const chosen = preferred_key
          ? candidates.filter(trace => fallback_pair_key(trace) === preferred_key)
          : [candidates[0]];
        for (const trace of chosen) {
          const cloned = clone_trace(trace);
          cloned.meta = {
            ...(cloned.meta || {}),
            instra_weight_selection_fallback: true,
          };
          prepared_data.push(cloned);
        }
        prepared_run_ids.add(run_id);
      }

      prepared.data = prepared_data;
      return prepared;
    };

    const base_figure_for_chart_matched_workspace_repair = figure_for_chart;
    figure_for_chart = function(chart_name) {
      return mark_incompatible_workspace_traces(
        base_figure_for_chart_matched_workspace_repair(chart_name),
        chart_name,
      );
    };

    const base_prepare_figure_matched_workspace_repair = prepare_figure;
    prepare_figure = function(figure, chart_name) {
      const source = mark_incompatible_workspace_traces(figure, chart_name);
      const prepared = restore_missing_workspace_runs(
        source,
        base_prepare_figure_matched_workspace_repair(source, chart_name),
        chart_name,
      );
      if (app.workspace_mode !== true || !weight_chart_set.has(chart_name)) return prepared;

      const override = app.chart_settings_render_override;
      const settings = normalize_chart_settings(
        chart_name,
        override?.chart_name === chart_name ? override.settings : null,
      );
      const requested_width = Number(settings?.line_width);
      const line_width = Number.isFinite(requested_width)
        ? Math.min(3, Math.max(0.5, requested_width))
        : 1;

      for (const trace of prepared.data || []) {
        const meta = trace?.meta;
        if (!meta || typeof meta !== "object" || Array.isArray(meta)) continue;
        const run_id = meta.instra_workspace_run_id;
        if (!run_id || meta.instra_top_axis_anchor === true) continue;
        const run_colour = colour_for_run(String(run_id));
        const mode = String(trace.mode || "");
        if (mode.includes("lines")) {
          trace.line = {...(trace.line || {}), color: run_colour, width: line_width};
        }
        if (mode.includes("markers") || trace.marker) {
          trace.marker = {...(trace.marker || {}), color: run_colour};
          trace.marker.line = {...(trace.marker.line || {}), color: run_colour};
        }
      }
      return prepared;
    };
  }, 0);
});
// ^^^ THOG
