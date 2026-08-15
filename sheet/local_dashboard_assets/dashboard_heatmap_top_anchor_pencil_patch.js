// vvv THOG
"use strict";

// Top-anchor the newest-first heatmap as one visual unit: x-axis/ticks/L datum
// at the top with the newest probe immediately below. Flatten chart-card gutters
// into one true one-pixel separator instead of adjacent card borders plus gaps.
window.addEventListener("load", () => {
  setTimeout(() => {
    const base_transpose_heatmap_top_anchor = transpose_heatmap;
    transpose_heatmap = function(prepared) {
      base_transpose_heatmap_top_anchor(prepared);
      if (!prepared?.layout?.xaxis) return;

      prepared.layout.xaxis = {
        ...prepared.layout.xaxis,
        side: "top",
        anchor: "y",
      };

      // The established heatmap chrome budget is 94 px. Swap the old bottom-heavy
      // margins so the x ticks, L= datum and axis title occupy that chrome above
      // the heatmap rather than below it.
      prepared.layout.margin = {
        ...(prepared.layout.margin || {}),
        t: 76,
        b: 18,
      };
    };

    const keep_heatmap_viewport_at_top = () => {
      const viewport = document.querySelector(
        '.chart-card[data-chart="heatmap"] .heatmap-inner-viewport'
      );
      if (!viewport) return;
      // Only establish the natural initial/top-anchor position. Do not continuously
      // fight a user who deliberately scrolls down through a tall heatmap.
      if (!viewport.dataset.thog2TopAnchorInitialised) {
        viewport.scrollTop = 0;
        viewport.dataset.thog2TopAnchorInitialised = "true";
      }
    };

    const heatmap_mount = by_id("heatmap_plot");
    if (heatmap_mount) {
      const observer = new MutationObserver(keep_heatmap_viewport_at_top);
      observer.observe(heatmap_mount, {childList: true, subtree: true});
      keep_heatmap_viewport_at_top();
    }

    const base_select_run_top_anchor = select_run;
    select_run = function(run_id, options = {}) {
      const changing_run = String(run_id || "") !== String(app.current_run_id || "");
      if (changing_run) {
        const viewport = document.querySelector(
          '.chart-card[data-chart="heatmap"] .heatmap-inner-viewport'
        );
        if (viewport) delete viewport.dataset.thog2TopAnchorInitialised;
      }
      return base_select_run_top_anchor(run_id, options);
    };

    const style = document.createElement("style");
    style.textContent = `
      /* Top anchoring replaces the earlier bottom-anchor rule. */
      .heatmap-inner-content {
        align-items: flex-start !important;
      }

      /* A real pencil-line chart grid: one 1px separator, not 10px whitespace
         plus two independent card borders. */
      .chart-grid,
      .local-metric-group .local-metric-grid {
        gap: 1px !important;
        padding: 1px !important;
        background: #d7dbe0 !important;
      }
      .chart-card,
      .local-metric-card {
        border: 0 !important;
        border-radius: 0 !important;
        box-shadow: none !important;
      }
      .chart-card:hover,
      .local-metric-card:hover {
        border: 0 !important;
        box-shadow: none !important;
      }
      .chart-card-header {
        border-bottom: 1px solid #dfe2e6 !important;
      }
      .chart-card:not([data-chart="heatmap"]),
      .local-metric-card {
        flex-basis: calc(33.333% - 1px) !important;
      }
      @media (max-width: 1100px) {
        .local-metric-card { flex-basis: calc(50% - 1px) !important; }
      }
      @media (max-width: 760px) {
        .chart-card:not([data-chart="heatmap"]),
        .local-metric-card { flex-basis: 100% !important; }
      }
    `;
    document.head.appendChild(style);

    if (app.figures?.heatmap && app.current_run_id) {
      queueMicrotask(async () => {
        const mount = by_id("heatmap_plot");
        if (!mount) return;
        await render_plot(mount, app.figures.heatmap, "heatmap");
        keep_heatmap_viewport_at_top();
      });
    }
  }, 0);
});
// ^^^ THOG
