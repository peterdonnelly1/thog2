// vvv THOG
"use strict";

// Define chart maximise as occupying the complete Charts-tab content area, while
// retaining the run list/top run chrome. Also widen the heatmap's centre/L datum
// band when dense x geometry would otherwise force its CLI-style text unreadably
// small.
window.addEventListener("load", () => {
  setTimeout(() => {
    const sync_tab_maximize_surface = () => {
      const charts_scroll = by_id("charts_scroll");
      if (!charts_scroll) return;
      const maximized = Boolean(app.maximized_chart);
      charts_scroll.classList.toggle("thog2-tab-maximized", maximized);

      if (!maximized) return;
      const selected_card = document.querySelector(
        `.chart-card[data-chart="${CSS.escape(String(app.maximized_chart))}"]`
      );
      const selected_group = selected_card?.closest(".chart-group");
      if (!selected_card || !selected_group) return;
      for (const group of charts_scroll.querySelectorAll(":scope > .chart-group")) {
        group.classList.toggle("thog2-tab-maximized-group", group === selected_group);
      }
      selected_group.classList.add("thog2-tab-maximized-group");
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
      requestAnimationFrame(() => requestAnimationFrame(resize_visible_plots));
      return result;
    };

    const finite_number = value => {
      const numeric = Number(value);
      return Number.isFinite(numeric) ? numeric : null;
    };

    const centre_datum_annotation = annotation => (
      annotation
      && annotation.xref === "x"
      && annotation.yref === "y"
      && annotation.showarrow === false
      && typeof annotation.hovertext === "string"
      && annotation.hovertext.startsWith("step=")
      && String(annotation.font?.family || "").includes("Mono")
    );

    const widen_centre_datum_band = (prepared, heatmap_trace) => {
      const shape = (prepared.layout?.shapes || []).find(
        candidate => candidate?.name === "thog2-centre-datum-background"
      );
      if (!shape) return;

      const shell = document.querySelector('.chart-card[data-chart="heatmap"] .heatmap-shell');
      const shell_width = Math.max(1, Number(shell?.clientWidth || prepared.layout?.width || 1));
      const margin = prepared.layout?.margin || {};
      const plot_width = Math.max(
        1,
        shell_width - Number(margin.l || 0) - Number(margin.r || 0),
      );
      const column_count = Math.max(
        1,
        Array.isArray(heatmap_trace.x) ? heatmap_trace.x.length : 1,
      );
      const cell_width = Math.max(1, plot_width / column_count);
      const row_height = Math.max(1, Number(heatmap_probe_row_height_px()));

      // Above 10 px/step every probe row may carry text, so respect row pitch.
      // Below that the established renderer already samples annotations to y ticks,
      // which lets those retained rows use a useful 10 px font despite tiny cells.
      const target_font_size = row_height >= 10
        ? Math.max(8, Math.min(13, Math.floor(row_height * 0.78)))
        : 10;
      const required_text_width = 17 * target_font_size * 0.61 + 6;
      const band_width_in_cells = Math.max(
        1,
        Math.min(7, required_text_width / cell_width),
      );
      const half_band = Math.max(0.5, band_width_in_cells / 2);
      const available_band_width = Math.max(cell_width, band_width_in_cells * cell_width);
      const width_limited_font = Math.max(
        7,
        Math.floor((available_band_width - 4) / (17 * 0.61)),
      );
      const font_size = Math.min(target_font_size, width_limited_font);

      shape.x0 = -half_band;
      shape.x1 = half_band;
      shape.fillcolor = "#000000";
      shape.line = {width: 0};

      for (const annotation of prepared.layout?.annotations || []) {
        if (!centre_datum_annotation(annotation)) continue;
        annotation.x = -half_band;
        annotation.xanchor = "left";
        annotation.xshift = 2;
        annotation.align = "left";
        annotation.font = {
          ...(annotation.font || {}),
          size: font_size,
        };
      }

      prepared.layout.meta = {
        ...(prepared.layout.meta || {}),
        thog2_centre_band_half_width: half_band,
        thog2_centre_band_font_size: font_size,
      };
    };

    const base_transpose_heatmap_lband = transpose_heatmap;
    transpose_heatmap = function(prepared) {
      base_transpose_heatmap_lband(prepared);
      const heatmap_trace = (prepared.data || []).find(trace => trace.type === "heatmap");
      if (!heatmap_trace) return;
      widen_centre_datum_band(prepared, heatmap_trace);
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
        display: none !important;
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
    `;
    document.head.appendChild(style);

    sync_tab_maximize_surface();
  }, 140);
});
// ^^^ THOG
