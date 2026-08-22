// vvv THOG
"use strict";

// Keep current-weight charts useful when a finished/older run never recorded the
// presently requested user coupling.  In that case, Runs view falls back to the
// single random coupling that was actually recorded in the latest snapshot.  This
// pass also owns the final six-chart order and keeps the explicit random control
// visible after the earlier compatibility layers have installed their controls.
window.addEventListener("load", () => {
  setTimeout(() => {
    const weight_chart_names = new Set([
      "attn_q_head_N",
      "attn_k_head_N",
      "attn_v_head_N",
      "attn_out_head_N",
      "mlp_up",
      "mlp_down",
    ]);
    const reverse_coupling_charts = new Set(["attn_out_head_N", "mlp_down"]);
    const desired_weight_order = Object.freeze([
      "attn_q_head_N",
      "attn_k_head_N",
      "mlp_up",
      "attn_v_head_N",
      "attn_out_head_N",
      "mlp_down",
    ]);
    const attention_letters = Object.freeze({
      attn_q_head_N: "Q",
      attn_k_head_N: "K",
      attn_v_head_N: "V",
      attn_out_head_N: "O",
    });
    const line_width_scale = 0.80;

    const finite_integer = value => {
      if (value === null || value === undefined || value === "") return null;
      const numeric = Number(value);
      return Number.isInteger(numeric) ? numeric : null;
    };

    const trace_kind = trace => {
      const meta = trace?.meta;
      if (!meta || typeof meta !== "object" || Array.isArray(meta)) return null;
      if (meta.instra_weight_selection_protocol !== "matched_six_v1") return null;
      return String(meta.instra_weight_selection_kind || "random");
    };

    const trace_coupling_key = trace => {
      const model_feature = finite_integer(trace?.meta?.instra_weight_model_feature);
      const intermediate_feature = finite_integer(trace?.meta?.instra_weight_intermediate_feature);
      if (model_feature === null || intermediate_feature === null) return null;
      return `${model_feature}:${intermediate_feature}`;
    };

    const actual_coupling = (trace, chart_name) => {
      const model_feature = finite_integer(trace?.meta?.instra_weight_model_feature);
      const intermediate_feature = finite_integer(trace?.meta?.instra_weight_intermediate_feature);
      if (model_feature === null || intermediate_feature === null) return null;
      return reverse_coupling_charts.has(chart_name)
        ? {input_feature: intermediate_feature, output_feature: model_feature}
        : {input_feature: model_feature, output_feature: intermediate_feature};
    };

    const replace_coupling_label = (value, trace, chart_name) => {
      if (typeof value !== "string") return value;
      const coupling = actual_coupling(trace, chart_name);
      if (!coupling) return value;
      const replacement = `input feature ${coupling.input_feature} → output feature ${coupling.output_feature}`;
      return value
        .replace(/residual feature \d+\s*[·•]\s*branch feature \d+/g, replacement)
        .replace(/model \d+\s*[·•]\s*(?:attention|MLP) feature \d+/g, replacement);
    };

    const clone_trace = trace => {
      if (typeof structuredClone === "function") {
        try { return structuredClone(trace); }
        catch (_error) { /* fall through */ }
      }
      return JSON.parse(JSON.stringify(trace));
    };

    const resolved_current_only = chart_name => {
      const override = app.chart_settings_render_override;
      const settings = normalize_chart_settings(
        chart_name,
        override?.chart_name === chart_name ? override.settings : null,
      );
      return settings?.current_weights_only === true;
    };

    const has_rendered_weight_trace = prepared => (prepared.data || []).some(trace => {
      if (trace_kind(trace) === null) return false;
      const mode = String(trace?.mode || "");
      return mode.includes("lines") || mode.includes("markers");
    });

    const recorded_random_fallback = (figure, chart_name) => {
      const candidates = (figure?.data || []).filter(trace => {
        const kind = trace_kind(trace);
        return kind === "random" || kind === "user_random";
      });
      const first_key = candidates.map(trace_coupling_key).find(Boolean);
      if (!first_key) return [];
      return candidates
        .filter(trace => trace_coupling_key(trace) === first_key)
        .map(trace => {
          const cloned = clone_trace(trace);
          cloned.meta = {...(cloned.meta || {}), instra_weight_selection_fallback: true};
          cloned.name = replace_coupling_label(cloned.name, cloned, chart_name);
          cloned.hovertemplate = replace_coupling_label(cloned.hovertemplate, cloned, chart_name);
          const width = Number(cloned?.line?.width);
          if (String(cloned?.mode || "").includes("lines") && Number.isFinite(width) && width > 0) {
            cloned.line = {...(cloned.line || {}), width: width * line_width_scale};
          }
          return cloned;
        });
    };

    const attention_plot_title = (value, chart_name) => {
      if (typeof value !== "string" || !attention_letters[chart_name]) return value;
      const replacements = [
        [/attention query/gi, "Attention - <b>Q</b>"],
        [/attention key/gi, "Attention - <b>K</b>"],
        [/attention value/gi, "Attention - <b>V</b>"],
        [/attention output/gi, "Attention - <b>O</b>"],
        [/attn_q_head_\d+/gi, "Attention - <b>Q</b>"],
        [/attn_k_head_\d+/gi, "Attention - <b>K</b>"],
        [/attn_v_head_\d+/gi, "Attention - <b>V</b>"],
        [/attn_out_head_\d+/gi, "Attention - <b>O</b>"],
      ];
      let updated = value;
      for (const [pattern, replacement] of replacements) updated = updated.replace(pattern, replacement);
      return updated;
    };

    const base_prepare_figure_weight_reliability = prepare_figure;
    prepare_figure = function(figure, chart_name) {
      const prepared = base_prepare_figure_weight_reliability(figure, chart_name);
      if (!weight_chart_names.has(chart_name)) return prepared;

      const title = prepared.layout?.title;
      if (title) {
        if (typeof title === "string") {
          prepared.layout.title = attention_plot_title(title, chart_name);
        } else {
          prepared.layout.title = {
            ...title,
            text: attention_plot_title(String(title.text || ""), chart_name),
          };
        }
      }

      // A user-selected coupling is only present in snapshots recorded after that
      // selection was made.  Finished/older runs must not turn into six empty axes:
      // use their recorded random pair when the requested pair is absent.
      if (
        app.workspace_mode !== true
        && resolved_current_only(chart_name)
        && !has_rendered_weight_trace(prepared)
      ) {
        const fallback = recorded_random_fallback(figure, chart_name);
        if (fallback.length) prepared.data = [...(prepared.data || []), ...fallback];
      }
      return prepared;
    };

    const reorder_weight_cards = () => {
      const cards = desired_weight_order.map(chart_name =>
        document.querySelector(`.chart-card[data-chart="${chart_name}"]`)
      );
      if (cards.some(card => !card)) return;
      const parent = cards[0].parentElement;
      if (!parent || cards.some(card => card.parentElement !== parent)) return;
      const current = [...parent.children]
        .filter(card => card.matches?.(".chart-card") && weight_chart_names.has(card.dataset.chart))
        .map(card => card.dataset.chart);
      if (current.join("|") === desired_weight_order.join("|")) return;
      for (const card of cards) parent.appendChild(card);
    };

    const show_random_control = () => {
      const button = by_id("weight_random_jump");
      if (!button) return;
      button.hidden = false;
      button.removeAttribute("hidden");
      button.textContent = "random";
      button.title = "Let INSTRA choose a new random feature coupling";
      button.setAttribute("aria-label", button.title);
    };

    const enforce_static_state = () => {
      reorder_weight_cards();
      show_random_control();
    };

    enforce_static_state();
    const weight_group = by_id("coefficients_chart_group") || by_id("depth_chart_group");
    if (weight_group) {
      const observer = new MutationObserver(enforce_static_state);
      observer.observe(weight_group, {childList: true, subtree: true, attributes: true, attributeFilter: ["hidden"]});
    }

    const style = document.createElement("style");
    style.textContent = `
      #weight_random_jump,
      #weight_random_jump[hidden] {
        display: inline-flex !important;
        width: auto !important;
        min-width: 48px !important;
        padding-left: 6px !important;
        padding-right: 6px !important;
      }
      .chart-card[data-chart="attn_q_head_N"] { order: 1 !important; }
      .chart-card[data-chart="attn_k_head_N"] { order: 2 !important; }
      .chart-card[data-chart="mlp_up"] { order: 3 !important; }
      .chart-card[data-chart="attn_v_head_N"] { order: 4 !important; }
      .chart-card[data-chart="attn_out_head_N"] { order: 5 !important; }
      .chart-card[data-chart="mlp_down"] { order: 6 !important; }
    `;
    document.head.appendChild(style);

    requestAnimationFrame(() => requestAnimationFrame(async () => {
      enforce_static_state();
      for (const chart_name of weight_chart_names) {
        const mount = by_id(`${chart_name}_plot`);
        let figure = null;
        try { figure = figure_for_chart(chart_name); }
        catch (_error) { figure = null; }
        if (mount && figure) await render_plot(mount, figure, chart_name);
      }
    }));
  }, 420);
});
// ^^^ THOG
