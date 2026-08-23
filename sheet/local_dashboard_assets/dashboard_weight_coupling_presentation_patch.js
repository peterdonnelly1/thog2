// vvv THOG
"use strict";

// Final Weights presentation pass: use ML-facing feature-coupling terminology,
// keep the group index controls compact, put the two MLP charts in the right
// column, simplify attention labels, and keep identity/chrome in the card/group
// headers rather than repeating it inside every Plotly chart.
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
    const attention_letters = Object.freeze({
      attn_q_head_N: "Q",
      attn_k_head_N: "K",
      attn_v_head_N: "V",
      attn_out_head_N: "O",
    });
    const attention_legacy_titles = new Set([
      "Attention query scalar trajectories",
      "Attention key scalar trajectories",
      "Attention value scalar trajectories",
      "Attention output scalar trajectories",
      "Attention query",
      "Attention key",
      "Attention value",
      "Attention output",
    ]);
    const desired_weight_order = Object.freeze([
      "attn_q_head_N",
      "attn_k_head_N",
      "mlp_up",
      "attn_v_head_N",
      "attn_out_head_N",
      "mlp_down",
    ]);
    const line_width_scale = 0.80;

    const finite_integer = value => {
      if (value === null || value === undefined || value === "") return null;
      const numeric = Number(value);
      return Number.isInteger(numeric) ? numeric : null;
    };

    const matched_selection = () => {
      const api = window.__instra_matched_weight_selection;
      const selection = typeof api?.selection === "function" ? api.selection() : null;
      return selection && typeof selection === "object" ? selection : null;
    };

    const matched_capability = () => {
      const api = window.__instra_matched_weight_selection;
      if (!api || typeof api.capability !== "function") return null;
      const chart_name = [...weight_chart_names].find(name => {
        try { return Boolean(figure_for_chart(name)); }
        catch (_error) { return false; }
      }) || "attn_q_head_N";
      try { return api.capability(chart_name); }
      catch (_error) { return null; }
    };

    const apply_weight_control_labels = () => {
      const controls = by_id("weight_index_group_controls");
      if (!controls) return;
      const summary = by_id("weight_index_group_summary");
      const selection = matched_selection();
      const capability = matched_capability();
      const input_feature = finite_integer(selection?.model_feature) ?? 0;
      const output_feature = finite_integer(selection?.intermediate_feature) ?? 0;

      if (summary) {
        let summary_text = "weight matrix feature coupling (input → output): waiting…";
        if (capability?.available !== false && selection?.user_selected === true) {
          summary_text = `weight matrix feature coupling (input → output): ${input_feature} → ${output_feature}`;
        } else if (capability?.available !== false && selection) {
          summary_text = `weight matrix feature coupling (input → output): random · ${input_feature} → ${output_feature}`;
        }
        if (summary.textContent !== summary_text) summary.textContent = summary_text;
        summary.title = (
          "A scalar weight couples one input feature to one output feature for every abstract-representation row. "
          + "Reverse-direction matrices use the transposed matrix element so the same paired features remain comparable."
        );
      }

      const button_specs = [
        ["weight_residual_minus", "i−", "Decrement input feature"],
        ["weight_residual_plus", "i+", "Increment input feature"],
        ["weight_branch_minus", "o−", "Decrement output feature"],
        ["weight_branch_plus", "o+", "Increment output feature"],
      ];
      for (const [id, label, title] of button_specs) {
        const button = by_id(id);
        if (!button) continue;
        if (button.textContent !== label) button.textContent = label;
        button.title = title;
        button.setAttribute("aria-label", title);
      }
      const random_button = by_id("weight_random_jump");
      if (random_button) {
        if (random_button.hidden) random_button.hidden = false;
        if (random_button.textContent !== "random") random_button.textContent = "random";
        random_button.title = "Let INSTRA choose a new random feature coupling";
        random_button.setAttribute("aria-label", random_button.title);
      }
      controls.setAttribute(
        "aria-label",
        "Weight matrix feature coupling. Input and output feature index controls."
      );
    };

    const reorder_weight_cards = () => {
      const grid = by_id("chart_grid");
      if (!grid) return;
      const cards = desired_weight_order
        .map(chart_name => grid.querySelector(`:scope > .chart-card[data-chart="${chart_name}"]`))
        .filter(Boolean);
      if (cards.length !== desired_weight_order.length) return;
      const current = [...grid.querySelectorAll(":scope > .chart-card[data-chart]")]
        .filter(card => weight_chart_names.has(card.dataset.chart))
        .map(card => card.dataset.chart);
      if (current.join("|") === desired_weight_order.join("|")) return;
      for (const card of cards) grid.appendChild(card);
    };

    const apply_attention_headings = () => {
      for (const [chart_name, letter] of Object.entries(attention_letters)) {
        const heading = document.querySelector(
          `.chart-card[data-chart="${chart_name}"] > .chart-card-header .chart-heading-copy h2`
        );
        if (!heading) continue;
        const expected_text = `Attention - ${letter}`;
        const configured_title = String(normalize_chart_settings(chart_name).title || expected_text);
        if (configured_title !== expected_text) {
          if (heading.textContent !== configured_title) heading.textContent = configured_title;
          continue;
        }
        const strong_present = heading.querySelector("strong")?.textContent === letter;
        if (heading.textContent === expected_text && strong_present) continue;
        heading.replaceChildren(document.createTextNode("Attention - "));
        const strong = document.createElement("strong");
        strong.textContent = letter;
        heading.appendChild(strong);
      }
    };

    for (const [chart_name, letter] of Object.entries(attention_letters)) {
      chart_titles[chart_name] = `Attention - ${letter}`;
    }

    const base_normalize_chart_settings_coupling = normalize_chart_settings;
    normalize_chart_settings = function(chart_name, supplied_settings = null) {
      const normalized = base_normalize_chart_settings_coupling(chart_name, supplied_settings);
      const letter = attention_letters[chart_name];
      if (!letter) return normalized;
      if (attention_legacy_titles.has(String(normalized.title || "").trim())) {
        return {...normalized, title: `Attention - ${letter}`};
      }
      return normalized;
    };

    const base_ensure_depth_cards_coupling = ensure_depth_cards;
    ensure_depth_cards = function() {
      const result = base_ensure_depth_cards_coupling();
      reorder_weight_cards();
      apply_attention_headings();
      apply_weight_control_labels();
      return result;
    };

    const base_show_toast_coupling = show_toast;
    show_toast = function(message) {
      let text = String(message || "");
      const saved = /^Weight indices set to residual (\d+), branch (\d+);(.*)$/.exec(text);
      if (saved) {
        text = `Weight matrix feature coupling set to input ${saved[1]} → output ${saved[2]};${saved[3]}`;
      } else if (text.startsWith("Weight indices must both be between ")) {
        text = text.replace("Weight indices", "Input and output features");
      }
      return base_show_toast_coupling(text);
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

    const current_only = chart_name => {
      const override = app.chart_settings_render_override;
      const settings = normalize_chart_settings(
        chart_name,
        override?.chart_name === chart_name ? override.settings : null,
      );
      return settings?.current_weights_only === true;
    };

    const selected_coupling = () => {
      const api = window.__instra_matched_weight_selection;
      const selection = typeof api?.selection === "function" ? api.selection() : null;
      if (!selection || selection.user_selected !== true) return null;
      const model_feature = finite_integer(selection.model_feature);
      const intermediate_feature = finite_integer(selection.intermediate_feature);
      if (model_feature === null || intermediate_feature === null) return null;
      return {model_feature, intermediate_feature};
    };

    const has_weight_trace = prepared => (prepared.data || []).some(trace => {
      if (trace?.meta?.instra_top_axis_anchor === true) return false;
      const meta = trace?.meta;
      if (!meta || typeof meta !== "object" || Array.isArray(meta)) return false;
      if (meta.instra_weight_selection_protocol !== "matched_six_v1") return false;
      const mode = String(trace?.mode || "");
      return mode.includes("lines") || mode.includes("markers");
    });

    const apply_unavailable_annotation = (prepared, chart_name) => {
      const annotation_name = "instra-selected-coupling-unavailable";
      const annotations = (prepared.layout.annotations || []).filter(
        annotation => annotation?.name !== annotation_name
      );
      const selection = selected_coupling();
      if (selection && current_only(chart_name) && !has_weight_trace(prepared)) {
        annotations.push({
          name: annotation_name,
          xref: "paper",
          yref: "paper",
          x: 0.5,
          y: 0.5,
          xanchor: "center",
          yanchor: "middle",
          align: "center",
          showarrow: false,
          font: {size: 12, color: "#667085"},
          text: (
            `Selected coupling ${selection.model_feature} → ${selection.intermediate_feature} `
            + "was not recorded for this view."
          ),
        });
      }
      if (annotations.length) prepared.layout.annotations = annotations;
      else delete prepared.layout.annotations;
    };

    const apply_weight_plot_chrome = (prepared, chart_name) => {
      if (!weight_chart_names.has(chart_name)) return prepared;
      prepared.layout = prepared.layout || {};
      delete prepared.layout.title;
      prepared.layout.showlegend = false;
      delete prepared.layout.legend;
      if (prepared.layout.xaxis2) delete prepared.layout.xaxis2.title;
      for (const trace of prepared.data || []) trace.showlegend = false;
      apply_unavailable_annotation(prepared, chart_name);
      return prepared;
    };

    const base_prepare_figure_coupling = prepare_figure;
    prepare_figure = function(figure, chart_name) {
      const prepared = base_prepare_figure_coupling(figure, chart_name);
      if (!weight_chart_names.has(chart_name)) return prepared;

      for (const trace of prepared.data || []) {
        trace.name = replace_coupling_label(trace.name, trace, chart_name);
        trace.hovertemplate = replace_coupling_label(trace.hovertemplate, trace, chart_name);
        const mode = String(trace?.mode || "");
        const width = Number(trace?.line?.width);
        if (mode.includes("lines") && Number.isFinite(width) && width > 0) {
          trace.line = {...trace.line, width: width * line_width_scale};
        }
      }
      return apply_weight_plot_chrome(prepared, chart_name);
    };

    window.__instra_weight_presentation = {
      apply_plot_chrome: apply_weight_plot_chrome,
    };

    const enforce_static_presentation = () => {
      apply_weight_control_labels();
      reorder_weight_cards();
      apply_attention_headings();
    };

    // Plotly mutates the chart subtree heavily during redraws. A broad
    // MutationObserver here amplifies every redraw and, when another patch toggles
    // a control attribute, can become a self-sustaining browser loop. Use bounded
    // startup passes plus the card-construction wrapper instead.
    enforce_static_presentation();
    let startup_passes = 0;
    const startup_timer = setInterval(() => {
      startup_passes += 1;
      enforce_static_presentation();
      if (startup_passes >= 20 || (
        by_id("weight_index_group_controls")
        && desired_weight_order.every(chart_name => document.querySelector(`.chart-card[data-chart="${chart_name}"]`))
      )) {
        clearInterval(startup_timer);
      }
    }, 100);

    const style = document.createElement("style");
    style.textContent = `
      #weight_index_group_controls .weight-index-step-button {
        min-width: 27px !important;
        width: 27px !important;
        padding-left: 3px !important;
        padding-right: 3px !important;
      }
      #weight_random_jump {
        display: inline-flex !important;
        width: auto !important;
        min-width: 48px !important;
        padding-left: 6px !important;
        padding-right: 6px !important;
      }
      #weight_index_group_summary { margin-right: 7px; }
      .chart-card[data-chart="attn_q_head_N"] .chart-heading-copy h2 strong,
      .chart-card[data-chart="attn_k_head_N"] .chart-heading-copy h2 strong,
      .chart-card[data-chart="attn_v_head_N"] .chart-heading-copy h2 strong,
      .chart-card[data-chart="attn_out_head_N"] .chart-heading-copy h2 strong {
        font-weight: 800;
      }
    `;
    document.head.appendChild(style);
  }, 320);
});
// ^^^ THOG
