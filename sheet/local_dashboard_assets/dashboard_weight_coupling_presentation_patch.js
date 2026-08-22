// vvv THOG
"use strict";

// Final Weights presentation pass: use ML-facing feature-coupling terminology,
// keep the four group index controls compact, put the two MLP charts in the
// right column, simplify attention labels, separate Plotly title/subtitle text,
// and slightly reduce trajectory line weight without changing chart settings.
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
      if (random_button) random_button.hidden = true;
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

    const attention_plot_title = (text, chart_name) => {
      const letter = attention_letters[chart_name];
      if (!letter || typeof text !== "string") return text;
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
      let updated = text;
      for (const [pattern, replacement] of replacements) updated = updated.replace(pattern, replacement);
      return updated;
    };

    const separate_plot_title_lines = prepared => {
      const current = prepared.layout?.title;
      if (!current) return;
      const title = typeof current === "string" ? {text: current} : {...current};
      let text = String(title.text || "");
      text = text.replace(
        /<br><sup>(.*?)<\/sup>/gi,
        '<br><span style="font-size:10px">$1</span>',
      );
      title.text = text;
      title.pad = {...(title.pad || {}), b: Math.max(10, Number(title.pad?.b || 0))};
      prepared.layout.title = title;
      prepared.layout.margin = {
        ...(prepared.layout.margin || {}),
        t: Math.max(90, Number(prepared.layout.margin?.t || 0)),
      };
    };

    const base_prepare_figure_coupling = prepare_figure;
    prepare_figure = function(figure, chart_name) {
      const prepared = base_prepare_figure_coupling(figure, chart_name);
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
      separate_plot_title_lines(prepared);

      for (const trace of prepared.data || []) {
        trace.name = replace_coupling_label(trace.name, trace, chart_name);
        trace.hovertemplate = replace_coupling_label(trace.hovertemplate, trace, chart_name);
        const mode = String(trace?.mode || "");
        const width = Number(trace?.line?.width);
        if (mode.includes("lines") && Number.isFinite(width) && width > 0) {
          trace.line = {...trace.line, width: width * line_width_scale};
        }
      }
      return prepared;
    };

    const enforce_static_presentation = () => {
      apply_weight_control_labels();
      reorder_weight_cards();
      apply_attention_headings();
    };

    enforce_static_presentation();
    const group = by_id("coefficients_chart_group");
    if (group) {
      const observer = new MutationObserver(enforce_static_presentation);
      observer.observe(group, {childList: true, subtree: true, characterData: true});
    }

    const style = document.createElement("style");
    style.textContent = `
      #weight_random_jump { display: none !important; }
      #weight_index_group_controls .weight-index-step-button {
        min-width: 27px !important;
        width: 27px !important;
        padding-left: 3px !important;
        padding-right: 3px !important;
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

    requestAnimationFrame(() => requestAnimationFrame(async () => {
      for (const chart_name of weight_chart_names) {
        const mount = by_id(`${chart_name}_plot`);
        let figure = null;
        try { figure = figure_for_chart(chart_name); }
        catch (_error) { figure = null; }
        if (mount && figure) await render_plot(mount, figure, chart_name);
      }
    }));
  }, 320);
});
// ^^^ THOG
