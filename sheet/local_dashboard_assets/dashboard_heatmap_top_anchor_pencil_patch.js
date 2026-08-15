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

// vvv THOG pin the natural heatmap canvas itself to the viewport top and make ordinary chart scrollbars contingency UI rather than permanent fat separators
window.addEventListener("load", () => {
  setTimeout(() => {
    const pin_heatmap_canvas_to_top = () => {
      const mount = by_id("heatmap_plot");
      const canvas = mount?.closest(".plot-scroll-canvas");
      const content = mount?.closest(".heatmap-inner-content");
      const viewport = mount?.closest(".heatmap-inner-viewport");
      if (!mount || !canvas || !content || !viewport) return;

      const canvas_height = Math.max(
        1,
        Math.round(Number.parseFloat(canvas.style.height) || canvas.getBoundingClientRect().height || 1),
      );
      const canvas_width = Math.max(
        1,
        Math.round(Number.parseFloat(canvas.style.width) || canvas.getBoundingClientRect().width || 1),
      );

      content.style.setProperty("display", "block", "important");
      content.style.setProperty("position", "relative", "important");
      content.style.setProperty("min-height", "0", "important");
      content.style.setProperty("height", `${canvas_height}px`, "important");
      content.style.setProperty("min-width", "0", "important");
      content.style.setProperty("width", `${Math.max(canvas_width, viewport.clientWidth)}px`, "important");

      canvas.style.setProperty("position", "absolute", "important");
      canvas.style.setProperty("top", "0", "important");
      canvas.style.setProperty("left", "0", "important");
      canvas.style.setProperty("margin", "0", "important");

      if (!viewport.dataset.thog2NaturalTopPinned) {
        viewport.scrollTop = 0;
        viewport.dataset.thog2NaturalTopPinned = "true";
      }
    };

    const base_render_plot_natural_top = render_plot;
    render_plot = async function(mount, figure, chart_name) {
      const result = await base_render_plot_natural_top(mount, figure, chart_name);
      if (chart_name === "heatmap") pin_heatmap_canvas_to_top();
      return result;
    };

    const base_select_run_natural_top = select_run;
    select_run = function(run_id, options = {}) {
      const changing_run = String(run_id || "") !== String(app.current_run_id || "");
      if (changing_run) {
        const viewport = document.querySelector(
          '.chart-card[data-chart="heatmap"] .heatmap-inner-viewport'
        );
        if (viewport) delete viewport.dataset.thog2NaturalTopPinned;
      }
      return base_select_run_natural_top(run_id, options);
    };

    const style = document.createElement("style");
    style.textContent = `
      .heatmap-inner-content {
        min-height: 0 !important;
      }

      /* Ordinary chart scrollbars appear only when a resized plot genuinely
         overflows its card. This removes the dark Firefox rails that looked
         like thick borders between adjacent charts. */
      .plot-shell:not(.heatmap-shell) {
        overflow: auto !important;
        scrollbar-width: thin !important;
      }
      .plot-shell:not(.heatmap-shell)::-webkit-scrollbar {
        width: 7px !important;
        height: 7px !important;
      }
      .plot-shell:not(.heatmap-shell)::-webkit-scrollbar-thumb {
        min-width: 20px !important;
        min-height: 20px !important;
        border: 1px solid #e6e9ed !important;
      }

      /* Preserve resize hit areas while drawing only a pencil line. */
      .panel-resizer-east {
        width: 5px !important;
        border-right: 1px solid #aab3be !important;
      }
      .panel-resizer-south {
        height: 5px !important;
        border-bottom: 1px solid #aab3be !important;
      }
      .panel-resizer-corner {
        width: 10px !important;
        height: 10px !important;
      }
    `;
    document.head.appendChild(style);

    pin_heatmap_canvas_to_top();
    if (app.figures?.heatmap && app.current_run_id) {
      queueMicrotask(async () => {
        const mount = by_id("heatmap_plot");
        if (!mount) return;
        await render_plot(mount, app.figures.heatmap, "heatmap");
        pin_heatmap_canvas_to_top();
      });
    }
  }, 0);
});
// ^^^ THOG
