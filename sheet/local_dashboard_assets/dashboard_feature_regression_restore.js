// vvv THOG
"use strict";

(function install_dashboard_feature_regression_restore() {
  const additional_palette = Object.freeze([
    "#3B82F6", "#1D4ED8", "#0EA5E9", "#0369A1", "#06B6D4", "#0E7490", "#14B8A6", "#0F766E",
    "#10B981", "#047857", "#22C55E", "#15803D", "#84CC16", "#4D7C0F", "#A3E635", "#65A30D",
    "#EAB308", "#A16207", "#F59E0B", "#B45309", "#F97316", "#C2410C", "#EF4444", "#B91C1C",
    "#F43F5E", "#BE123C", "#EC4899", "#BE185D", "#D946EF", "#A21CAF", "#A855F7", "#7E22CE",
    "#8B5CF6", "#6D28D9", "#6366F1", "#4338CA", "#4F46E5", "#312E81", "#7C3AED", "#5B21B6",
    "#0D9488", "#115E59", "#0891B2", "#155E75", "#0284C7", "#075985", "#2563EB", "#1E40AF",
    "#92400E", "#78350F", "#A0522D", "#6B4423", "#708090", "#475569", "#334155", "#1E293B",
    "#FF0000", "#00FF00", "#0000FF", "#00FFFF", "#FF00FF", "#FFFF00", "#000000", "#FFFFFF",
  ]);
  const combined_palette = Object.freeze(
    [...new Set([...default_palette, ...additional_palette].map(colour => String(colour).toUpperCase()))],
  );
  window.instra_colour_palette = combined_palette;

  function install_palette() {
    const container = by_id("colour_swatches");
    if (!container) return;
    const installed = new Set(
      [...container.querySelectorAll(".colour-swatch")]
        .map(button => String(button.title || "").toUpperCase()),
    );
    for (const colour of combined_palette) {
      if (installed.has(colour)) continue;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "colour-swatch";
      button.style.background = colour;
      button.title = colour;
      button.setAttribute("aria-label", `Choose run colour ${colour}`);
      button.addEventListener("click", () => set_picker_colour(hex_to_rgb(colour)));
      container.appendChild(button);
    }
    container.dataset.instraPalette128 = "true";
  }

  const open_colour_picker_before_palette_restore = open_colour_picker;
  open_colour_picker = function(run_id, anchor) {
    install_palette();
    return open_colour_picker_before_palette_restore(run_id, anchor);
  };

  function chart_has_data(chart_name) {
    if (!app.figures) return false;
    if (chart_name === "heatmap") return Boolean(app.figures.heatmap);
    return Boolean(app.figures.depth?.[chart_name]);
  }

  function sync_depth_chart_availability() {
    const group = by_id("depth_chart_group");
    if (!group) return;
    let visible_count = 0;
    for (const card of group.querySelectorAll(".chart-card[data-chart]")) {
      const has_data = chart_has_data(String(card.dataset.chart || ""));
      card.hidden = !has_data;
      if (has_data) visible_count += 1;
    }
    group.hidden = visible_count === 0;
    const count = by_id("depth_group_count");
    if (count) count.textContent = String(visible_count);
  }

  const render_figures_before_feature_restore = render_figures;
  render_figures = async function() {
    await render_figures_before_feature_restore();
    sync_depth_chart_availability();
  };

  const reset_run_charts_before_feature_restore = reset_run_charts;
  reset_run_charts = function() {
    const result = reset_run_charts_before_feature_restore();
    sync_depth_chart_availability();
    return result;
  };

  window.addEventListener("load", () => {
    install_palette();
    sync_depth_chart_availability();
  });

  window.instra_feature_regression_test_hooks = Object.freeze({
    additional_palette,
    combined_palette,
    install_palette,
    sync_depth_chart_availability,
  });
})();
// ^^^ THOG
