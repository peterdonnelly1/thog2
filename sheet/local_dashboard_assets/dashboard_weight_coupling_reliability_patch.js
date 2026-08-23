// vvv THOG
"use strict";

// Keep the final six-chart order stable and the explicit random control visible.
// Missing selected couplings are deliberately not replaced here: the presentation
// owner renders an honest unavailable state instead of a differently indexed line.
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
  }, 420);
});
// ^^^ THOG
