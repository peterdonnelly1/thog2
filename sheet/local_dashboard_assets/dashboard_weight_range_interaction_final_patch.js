// vvv THOG
"use strict";

// Narrow final owner for explicit-range rendering and the top-level coupling controls.
// An explicit step window is a history request and therefore wins over Current weights
// only; the matched-coordinate selector still runs inside the established render stack.
window.addEventListener("load", () => {
  const weight_chart_set = new Set([
    "attn_q_head_N",
    "attn_k_head_N",
    "attn_v_head_N",
    "attn_out_head_N",
    "mlp_up",
    "mlp_down",
  ]);

  const finite_integer = value => {
    if (value === null || value === undefined || value === "") return null;
    const numeric = Number(value);
    return Number.isInteger(numeric) ? numeric : null;
  };

  const random_other_index = (current, maximum) => {
    if (maximum < 1) return current;
    const candidate = Math.floor(Math.random() * maximum);
    return candidate >= current ? candidate + 1 : candidate;
  };

  // Install this capture listener before the delayed regression guard installs its
  // retained-only RND handler. With one retained scalar that older policy can only
  // select the same pair again, making RND visibly inert.
  window.addEventListener("click", event => {
    const button = event.target.closest?.("#weight_random_jump");
    if (!button) return;
    event.preventDefault();
    event.stopImmediatePropagation();

    const api = window.__instra_matched_weight_selection;
    const selection = typeof api?.selection === "function" ? api.selection() : null;
    const chart_name = [...weight_chart_set].find(name => {
      try { return Boolean(figure_for_chart(name)); }
      catch (_error) { return false; }
    }) || "attn_q_head_N";
    let capability = null;
    try { capability = typeof api?.capability === "function" ? api.capability(chart_name) : null; }
    catch (_error) { capability = null; }
    const maximum = finite_integer(capability?.maximum);
    if (!capability?.available || maximum === null || maximum < 1) {
      show_toast(capability?.reason || "Weight feature bounds are not available yet.");
      return;
    }

    const current_input = Math.min(maximum, Math.max(0, finite_integer(selection?.model_feature) ?? 0));
    const current_output = Math.min(maximum, Math.max(0, finite_integer(selection?.intermediate_feature) ?? 0));
    const next_input = random_other_index(current_input, maximum);
    const next_output = random_other_index(current_output, maximum);
    const input = by_id("weight_coupling_input");
    const output = by_id("weight_coupling_output");
    if (!input || !output || input.disabled || output.disabled) {
      show_toast("Weight coupling controls are not ready yet.");
      return;
    }
    input.value = String(next_input);
    output.value = String(next_output);
    input.dispatchEvent(new Event("change", {bubbles: true}));
  }, true);

  const install_final_render_owner = () => {
    if (window.__instra_weight_range_interaction_final) return true;
    if (!window.__instra_weight_stability_final) return false;
    if (!window.__instra_weight_coupling_reliability_final) return false;
    if (typeof prepare_figure !== "function") return false;

    const base_prepare_figure_weight_range_final = prepare_figure;
    prepare_figure = function(figure, chart_name) {
      if (!weight_chart_set.has(chart_name)) return base_prepare_figure_weight_range_final(figure, chart_name);

      const range = window.__instra_weight_stability_final?.selected_range?.() || null;
      const saved_override = app.chart_settings_render_override;
      if (range) {
        const supplied = saved_override?.chart_name === chart_name && saved_override.settings
          ? saved_override.settings
          : normalize_chart_settings(chart_name);
        app.chart_settings_render_override = {
          chart_name,
          settings: {...supplied, current_weights_only: false},
        };
      }

      let prepared;
      try {
        prepared = base_prepare_figure_weight_range_final(figure, chart_name);
      } finally {
        app.chart_settings_render_override = saved_override;
      }

      for (const trace of prepared?.data || []) {
        if (trace?.meta?.instra_top_axis_anchor === true) continue;
        let update = null;
        try { update = Number(trace_optimizer_update(trace)); }
        catch (_error) { update = null; }
        if (!Number.isFinite(update)) continue;
        const marker = `step ${update}`;
        const existing = typeof trace.hovertemplate === "string" ? trace.hovertemplate : "";
        if (existing.includes(marker)) continue;
        if (!existing) {
          trace.hovertemplate = `${marker}<br>layer %{x}<br>weight %{y}<extra></extra>`;
          continue;
        }
        const extra_index = existing.indexOf("<extra");
        trace.hovertemplate = extra_index >= 0
          ? `${existing.slice(0, extra_index)}<br>${marker}${existing.slice(extra_index)}`
          : `${existing}<br>${marker}`;
      }
      return prepared;
    };

    const style = document.createElement("style");
    style.id = "thog2_weight_range_interaction_final_style";
    style.textContent = `
      .weight-coupling-editor input {
        width: 62px !important;
        min-width: 62px !important;
      }
    `;
    document.head.appendChild(style);

    window.__instra_weight_range_interaction_final = Object.freeze({installed: true});
    if (app.current_run_id && typeof render_figures === "function") {
      queueMicrotask(() => {
        try {
          const result = render_figures();
          if (result && typeof result.catch === "function") result.catch(() => undefined);
        } catch (_error) {}
      });
    }
    return true;
  };

  let attempts = 0;
  const retry = () => {
    attempts += 1;
    if (install_final_render_owner() || attempts >= 240) return;
    setTimeout(retry, 25);
  };
  // Delay the first render-owner attempt until the intentionally delayed regression
  // patch has wrapped prepare_figure; the RND capture listener above is already live.
  setTimeout(retry, 400);
});
// ^^^ THOG
