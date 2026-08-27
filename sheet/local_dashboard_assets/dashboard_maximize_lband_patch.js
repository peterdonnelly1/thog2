// vvv THOG
"use strict";

// Define chart maximise as occupying the complete Charts-tab content area while
// retaining the run list/top run chrome. Heatmap datum geometry is deliberately
// left unchanged: the centre/L background must remain exactly one cell wide.
window.addEventListener("load", () => {
  setTimeout(() => {
    const sync_tab_maximize_surface = () => {
      const charts_scroll = by_id("charts_scroll");
      if (!charts_scroll) return;
      const maximized = Boolean(app.maximized_chart);
      charts_scroll.classList.toggle("thog2-tab-maximized", maximized);

      if (!maximized) {
        for (const actions of document.querySelectorAll(".thog2-maximized-weight-controls")) {
          actions.classList.remove("thog2-maximized-weight-controls");
          for (const property of ["top", "left", "right"]) actions.style.removeProperty(property);
        }
        return;
      }
      const selected_card = document.querySelector(
        `.chart-card[data-chart="${CSS.escape(String(app.maximized_chart))}"]`
      );
      const selected_group = selected_card?.closest(".chart-group");
      if (!selected_card || !selected_group) return;
      for (const group of charts_scroll.querySelectorAll(":scope > .chart-group")) {
        group.classList.toggle("thog2-tab-maximized-group", group === selected_group);
      }
      selected_group.classList.add("thog2-tab-maximized-group");
      const weight_chart = Array.isArray(depth_weight_chart_names)
        && depth_weight_chart_names.includes(String(app.maximized_chart));
      if (!weight_chart) return;
      const actions = selected_card.querySelector(".chart-card-actions");
      const header = selected_group.querySelector(":scope > .chart-group-header");
      if (!actions || !header) return;
      actions.classList.add("thog2-maximized-weight-controls");
      requestAnimationFrame(() => {
        const header_rect = header.getBoundingClientRect();
        const anchor = by_id("weight_index_group_controls")
          || header.querySelector(".weights-group-settings-button");
        const anchor_rect = anchor?.getBoundingClientRect?.();
        const left = anchor_rect && anchor_rect.width > 0
          ? Math.max(header_rect.left + 90, anchor_rect.left - actions.offsetWidth - 12)
          : header_rect.right - actions.offsetWidth - 12;
        actions.style.setProperty("top", `${Math.round(header_rect.top + Math.max(4, (header_rect.height - actions.offsetHeight) / 2))}px`, "important");
        actions.style.setProperty("left", `${Math.round(left)}px`, "important");
        actions.style.setProperty("right", "auto", "important");
      });
    };

    const base_toggle_maximized_tab = toggle_maximized_chart;
    toggle_maximized_chart = function(chart_name) {
      const result = base_toggle_maximized_tab(chart_name);
      sync_tab_maximize_surface();
      requestAnimationFrame(() => requestAnimationFrame(resize_visible_plots));
      return result;
    };

    const base_restore_maximized_tab = restore_maximized_chart;
    restore_maximized_chart = function() {
      const result = base_restore_maximized_tab();
      const charts_scroll = by_id("charts_scroll");
      charts_scroll?.classList.remove("thog2-tab-maximized");
      for (const group of charts_scroll?.querySelectorAll(":scope > .chart-group") || []) {
        group.classList.remove("thog2-tab-maximized-group");
      }
      sync_tab_maximize_surface();
      requestAnimationFrame(() => requestAnimationFrame(resize_visible_plots));
      return result;
    };

    const style = document.createElement("style");
    style.textContent = `
      /* Maximise means the complete Charts-tab content area: all sibling groups
         disappear and the selected group/grid/card consume the available tab. */
      .charts-scroll.thog2-tab-maximized {
        position: relative !important;
        overflow: hidden !important;
      }
      .charts-scroll.thog2-tab-maximized > .chart-group:not(.thog2-tab-maximized-group) {
        display: none !important;
      }
      .charts-scroll.thog2-tab-maximized > .chart-group.thog2-tab-maximized-group {
        display: flex !important;
        flex-direction: column !important;
        position: absolute !important;
        inset: 0 !important;
        width: 100% !important;
        height: 100% !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        background: #fff !important;
      }
      .charts-scroll.thog2-tab-maximized > .chart-group.thog2-tab-maximized-group > .chart-group-header {
        display: flex !important;
        flex: 0 0 auto !important;
        min-height: 35px !important;
      }
      .charts-scroll.thog2-tab-maximized > .chart-group.thog2-tab-maximized-group > .chart-grid {
        flex: 1 1 auto !important;
        width: 100% !important;
        height: 100% !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        gap: 0 !important;
        overflow: hidden !important;
        background: #fff !important;
      }
      .charts-scroll.thog2-tab-maximized .chart-card.maximized {
        display: block !important;
        flex: 1 1 100% !important;
        width: 100% !important;
        height: 100% !important;
        min-width: 0 !important;
        min-height: 0 !important;
        max-width: none !important;
        margin: 0 !important;
      }
      .chart-card.maximized .chart-card-actions.thog2-maximized-weight-controls {
        position: fixed !important;
        z-index: 75 !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 3px !important;
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: auto !important;
      }
      .chart-card.maximized .chart-card-actions.thog2-maximized-weight-controls > *,
      .chart-card.maximized .chart-card-actions.thog2-maximized-weight-controls .explicit-trajectory-modes {
        position: static !important;
        inset: auto !important;
        display: inline-flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        margin: 0 !important;
        pointer-events: auto !important;
      }
    `;
    document.head.appendChild(style);

    sync_tab_maximize_surface();
    window.addEventListener("resize", sync_tab_maximize_surface);
  }, 140);
});
// ^^^ THOG
