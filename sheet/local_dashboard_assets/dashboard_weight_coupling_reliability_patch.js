// vvv THOG
"use strict";

// Keep the final six-chart order stable and the explicit random control visible.
// Missing selected couplings are deliberately not replaced here: the presentation
// owner renders an honest unavailable state instead of a differently indexed line.
window.addEventListener("load", () => {
  // vvv THOG install only after both consolidated Weights semantics and the delayed presentation owner are live
  const install = () => {
    if (window.__instra_weight_coupling_reliability_final) return true;
    if (!window.__instra_weight_stability_final) return false;
    if (!window.__instra_weight_presentation) return false;

    const weight_chart_names = new Set([
      "attn_q_head_N",
      "attn_k_head_N",
      "attn_v_head_N",
      "attn_out_head_N",
      "mlp_up",
      "mlp_down",
    ]);
    const desired_weight_order = Object.freeze([
      "attn_q_head_N",
      "attn_k_head_N",
      "mlp_up",
      "attn_v_head_N",
      "attn_out_head_N",
      "mlp_down",
    ]);
    const reorder_weight_cards = () => {
      const cards = desired_weight_order.map(chart_name =>
        document.querySelector(`.chart-card[data-chart="${chart_name}"]`)
      );
      if (cards.some(card => !card)) return false;
      const parent = cards[0].parentElement;
      if (!parent || cards.some(card => card.parentElement !== parent)) return false;
      const current = [...parent.children]
        .filter(card => card.matches?.(".chart-card") && weight_chart_names.has(card.dataset.chart))
        .map(card => card.dataset.chart);
      if (current.join("|") !== desired_weight_order.join("|")) {
        for (const card of cards) parent.appendChild(card);
      }
      return true;
    };

    const show_random_control = () => {
      const button = by_id("weight_random_jump");
      if (!button) return false;
      if (button.hidden) button.hidden = false;
      if (button.hasAttribute("hidden")) button.removeAttribute("hidden");
      if (button.textContent !== "random") button.textContent = "random";
      const title = "Let INSTRA choose a new random feature coupling";
      if (button.title !== title) button.title = title;
      if (button.getAttribute("aria-label") !== title) button.setAttribute("aria-label", title);
      return true;
    };

    const enforce_static_state = () => ({
      cards_ready: reorder_weight_cards(),
      random_ready: show_random_control(),
    });

    const weight_flag_control_ids = new Set([
      "chart_current_weights_only",
      "chart_join_with_line_segments",
    ]);
    let group_flag_draft = null;
    const group_editor_open = () => (
      by_id("chart_settings_overlay")?.hidden === false
      && by_id("weights_group_scale_field")?.hidden === false
    );
    const remember_group_flag = control => {
      if (!group_editor_open() || !control || !weight_flag_control_ids.has(control.id)) return;
      if (!group_flag_draft) {
        group_flag_draft = {
          current_weights_only: by_id("chart_current_weights_only")?.checked === true,
          join_with_line_segments: by_id("chart_join_with_line_segments")?.checked === true,
        };
      }
      if (control.id === "chart_current_weights_only") {
        group_flag_draft.current_weights_only = control.checked === true;
      } else {
        group_flag_draft.join_with_line_segments = control.checked === true;
      }
    };
    const preserve_weight_flag_draft = event => {
      const control = event.target;
      if (!control || !weight_flag_control_ids.has(control.id)) return;
      if (by_id("chart_settings_overlay")?.hidden) return;
      remember_group_flag(control);
      const group_editor = by_id("weights_group_scale_field")?.hidden === false;
      const inherit = by_id("chart_inherit_weights_group");
      if (!group_editor && inherit?.checked === true) inherit.checked = false;
      event.stopImmediatePropagation();
    };
    window.addEventListener("input", preserve_weight_flag_draft, true);
    window.addEventListener("change", preserve_weight_flag_draft, true);
    window.addEventListener("click", event => {
      const target = event.target;
      if (target && weight_flag_control_ids.has(target.id)) {
        remember_group_flag(target);
        return;
      }
      if (!target?.closest?.("#save_chart_settings") || !group_editor_open() || !group_flag_draft) return;
      const current = by_id("chart_current_weights_only");
      const join = by_id("chart_join_with_line_segments");
      if (current) current.checked = group_flag_draft.current_weights_only;
      if (join) join.checked = group_flag_draft.join_with_line_segments;
      queueMicrotask(() => { group_flag_draft = null; });
    }, true);

    const base_prepare_figure_weight_reliability = prepare_figure;
    prepare_figure = function(figure, chart_name) {
      const prepared = base_prepare_figure_weight_reliability(figure, chart_name);
      if (!weight_chart_names.has(chart_name)) return prepared;
      const render_override = app.chart_settings_render_override;
      const editor_open = by_id("chart_settings_overlay")?.hidden === false;
      const preview_settings = (
        editor_open && render_override?.chart_name === chart_name
          ? render_override.settings
          : null
      );
      const persisted = window.__instra_weight_stability_final?.effective?.(chart_name) || {};
      const join_with_line_segments = (
        preview_settings
        && Object.prototype.hasOwnProperty.call(preview_settings, "join_with_line_segments")
          ? preview_settings.join_with_line_segments === true
          : persisted.join_with_line_segments === true
      );
      if (!join_with_line_segments) {
        // vvv THOG hidden Firefox-only CI diagnostic: compactly expose final setting sources without changing visible Plotly chrome
        const firefox_runtime = (
          typeof navigator !== "undefined"
          && /Firefox/i.test(String(navigator.userAgent || ""))
        );
        if (firefox_runtime) {
          const group_scope = app.workspace_mode === true
            ? "workspace"
            : `run:${String(app.current_run_id || "unselected")}`;
          const chart_scope = `${group_scope}:${chart_name}`;
          const group = window.__instra_weight_group_settings?.group_settings_for_scope?.(group_scope) || null;
          const encode = value => value === true ? "1" : value === false ? "0" : "n";
          prepared.layout = prepared.layout || {};
          prepared.layout.legend = {
            instra_join_diagnostic: (
              `e${encode(editor_open)}`
              + `p${encode(preview_settings?.join_with_line_segments)}`
              + `s${encode(persisted.join_with_line_segments)}`
              + `g${encode(group?.join_with_line_segments)}`
              + `l${encode(app.weight_join_with_line_segments?.[chart_scope])}`
            ),
          };
        }
        // ^^^ THOG
        return prepared;
      }
      prepared.data = (prepared.data || []).filter(
        trace => trace?.meta?.instra_thog_executed_overlay !== true
      );
      for (const trace of prepared.data || []) {
        if (trace?.meta?.instra_thog_weight !== true) continue;
        const mode = String(trace.mode || "");
        if (!mode.includes("lines")) continue;
        trace.line = {...(trace.line || {}), shape: "linear"};
      }
      return prepared;
    };

    // Never observe the Plotly/chart subtree continuously. A previous version had
    // two observers disagreeing over the random button's hidden state; one observer
    // also rewrote textContent, generating another child-list mutation. That formed
    // an unbounded callback loop and pegged Firefox. Bounded startup retries plus
    // the card-construction wrapper are sufficient and cannot self-trigger forever.
    const base_ensure_depth_cards_weight_reliability = ensure_depth_cards;
    ensure_depth_cards = function() {
      const result = base_ensure_depth_cards_weight_reliability();
      enforce_static_state();
      return result;
    };

    enforce_static_state();
    let startup_passes = 0;
    const startup_timer = setInterval(() => {
      startup_passes += 1;
      const ready = enforce_static_state();
      if ((ready.cards_ready && ready.random_ready) || startup_passes >= 30) {
        clearInterval(startup_timer);
      }
    }, 100);

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

    window.__instra_weight_coupling_reliability_final = Object.freeze({
      installed: true,
    });

    // The presentation owner may have rendered once before this final wrapper was
    // installed. Re-render exactly once so existing Plotly mounts immediately obey
    // the no-executed-overlay / integer-line-segment contract.
    if (app.current_run_id && typeof render_figures === "function") {
      setTimeout(() => {
        try {
          const result = render_figures();
          if (result && typeof result.catch === "function") result.catch(() => undefined);
        } catch (_error) {
          // A later normal refresh will retry; installation itself must stay stable.
        }
      }, 0);
    }
    return true;
  };

  if (install()) return;
  let attempts = 0;
  const retry = () => {
    attempts += 1;
    if (install() || attempts >= 240) return;
    setTimeout(retry, 0);
  };
  setTimeout(retry, 0);
  // ^^^ THOG
});
// ^^^ THOG
