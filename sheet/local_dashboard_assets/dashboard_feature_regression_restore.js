// vvv THOG
"use strict";

(function install_dashboard_feature_regression_restore() {
  function hsl_hex(hue, saturation, lightness) {
    const s = saturation / 100;
    const l = lightness / 100;
    const chroma = (1 - Math.abs(2 * l - 1)) * s;
    const section = ((hue % 360) + 360) % 360 / 60;
    const second = chroma * (1 - Math.abs(section % 2 - 1));
    const pairs = [
      [chroma, second, 0], [second, chroma, 0], [0, chroma, second],
      [0, second, chroma], [second, 0, chroma], [chroma, 0, second],
    ];
    const [red, green, blue] = pairs[Math.floor(section) % 6];
    const match = l - chroma / 2;
    return `#${[red, green, blue].map(value => (
      Math.round((value + match) * 255).toString(16).padStart(2, "0")
    )).join("").toUpperCase()}`;
  }

  // Step seven traverses every one of the 24 hue positions while keeping
  // neighbouring picker patches far apart.  Each tonal band occupies six
  // complete rows of eight; the exact primaries/secondaries/white/black form
  // the nineteenth and final row.
  const hue_order = Object.freeze(Array.from({length:24}, (_value, index) => (index * 7) % 24));
  function tonal_band(first_saturation, first_lightness, second_saturation, second_lightness) {
    return [
      ...hue_order.map(index => hsl_hex(index * 15, first_saturation, first_lightness)),
      ...hue_order.map(index => hsl_hex(index * 15 + 7.5, second_saturation, second_lightness)),
    ];
  }
  const combined_palette = Object.freeze([
    ...tonal_band(78, 29, 66, 43),
    ...tonal_band(72, 61, 60, 72),
    ...tonal_band(68, 84, 55, 92),
    "#FF0000", "#00FF00", "#0000FF", "#00FFFF",
    "#FF00FF", "#FFFF00", "#FFFFFF", "#000000",
  ]);
  window.instra_colour_palette = combined_palette;

  const palette_style = document.createElement("style");
  palette_style.id = "instra-shared-colour-palette-style";
  palette_style.textContent = `
    #colour_swatches {
      grid-template-columns:repeat(8,minmax(0,1fr)) !important;
    }
    #colour_popover {
      width:282px !important;
      max-height:min(590px,calc(100vh - 16px)) !important;
      overflow:auto !important;
    }
  `;
  document.head.appendChild(palette_style);

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
    container.dataset.instraPaletteVersion = "categorical-v3-152";
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
