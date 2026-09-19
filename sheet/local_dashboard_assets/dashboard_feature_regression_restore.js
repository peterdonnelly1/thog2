// vvv THOG
"use strict";

(function install_dashboard_feature_regression_restore() {
  // One deliberately categorical palette for both run and operation pickers.
  // It replaces the former light/dark shade families, whose adjacent patches
  // were effectively duplicates at the size used by the picker.
  const combined_palette = Object.freeze([
    "#0000FF", "#FF0000", "#00C800", "#000033", "#FF00B6", "#005300",
    "#FFD300", "#009FFF", "#9A4D42", "#00DDA3", "#783FC1", "#1F9698",
    "#FFACFD", "#8EAD3A", "#F1085C", "#FE8F42", "#B900D6", "#201A01",
    "#720055", "#766C95", "#02AD24", "#B5D900", "#886C00", "#FFB79F",
    "#858567", "#A10300", "#14DDE5", "#00479E", "#DC5E93", "#93D4FF",
    "#004CFF", "#E6E600", "#D000D0", "#007D16", "#D6005E", "#00A7FF",
    "#00A86B", "#B36B00", "#6840E0", "#008A8A", "#A58F00", "#FF6E9C",
    "#3B5B00", "#6FA8DC", "#6B2D5C", "#008000", "#B78AD6", "#000000",
  ]);
  window.instra_colour_palette = combined_palette;

  function install_palette() {
    const container = by_id("colour_swatches");
    if (!container) return;
    container.replaceChildren();
    for (const colour of combined_palette) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "colour-swatch";
      button.style.background = colour;
      button.title = colour;
      button.setAttribute("aria-label", `Choose run colour ${colour}`);
      button.addEventListener("click", () => set_picker_colour(hex_to_rgb(colour)));
      container.appendChild(button);
    }
    container.dataset.instraPaletteVersion = "categorical-v2";
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
    combined_palette,
    install_palette,
    sync_depth_chart_availability,
  });
})();
// ^^^ THOG
