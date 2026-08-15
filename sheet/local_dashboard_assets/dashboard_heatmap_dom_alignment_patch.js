// vvv THOG
"use strict";

// Final DOM-measured alignment for heatmap labels. Plotly's SVG placement can
// override/neutralise nominal yshift/transform values as the heatmap is rebuilt,
// so align against the rendered heatmap/tick bounding boxes after every render.
window.addEventListener("load", () => {
  setTimeout(() => {
    const x_title_text = "candidate layer-count offset from active layer count";
    const x_title_gap_px = 8;
    const newest_y_left_shift_px = -7;

    const rendered_heatmap_image = mount => (
      mount?.querySelector(".heatmaplayer image")
      || mount?.querySelector("g.hm image")
      || null
    );

    const heatmap_x_title_node = mount => {
      if (!mount) return null;
      return [...mount.querySelectorAll("svg text")].find(
        node => String(node.textContent || "").trim() === x_title_text
      ) || null;
    };

    const align_heatmap_dom_labels = () => {
      const mount = by_id("heatmap_plot");
      if (!mount || !app.current_run_id) return;

      const image = rendered_heatmap_image(mount);
      const y_ticks = [...mount.querySelectorAll(".ytick text")];
      if (image && y_ticks.length) {
        const newest = y_ticks.reduce((best, candidate) => (
          candidate.getBoundingClientRect().top < best.getBoundingClientRect().top
            ? candidate
            : best
        ));

        // Remove every previous guessed transform, measure the native Plotly
        // position, then centre the label on the first literal heatmap row.
        newest.style.setProperty("transform", "none", "important");
        newest.style.setProperty("transform-box", "fill-box", "important");
        newest.style.setProperty("transform-origin", "center", "important");
        const image_rect = image.getBoundingClientRect();
        const newest_rect = newest.getBoundingClientRect();
        const row_height = Math.max(1, Number(heatmap_probe_row_height_px()));
        const target_center_y = image_rect.top + row_height / 2;
        const current_center_y = newest_rect.top + newest_rect.height / 2;
        const delta_y = target_center_y - current_center_y;
        newest.style.setProperty(
          "transform",
          `translate(${newest_y_left_shift_px}px, ${delta_y.toFixed(2)}px)`,
          "important",
        );
      }

      const title = heatmap_x_title_node(mount);
      const x_ticks = [...mount.querySelectorAll(".xtick text")];
      if (title && x_ticks.length) {
        // Measure from a neutral transform so repeated renders/resize events do
        // not accumulate translation. Enforce a real white-space gap.
        title.style.setProperty("transform", "none", "important");
        title.style.setProperty("transform-box", "fill-box", "important");
        title.style.setProperty("transform-origin", "center", "important");
        const tick_top = Math.min(...x_ticks.map(node => node.getBoundingClientRect().top));
        const title_rect = title.getBoundingClientRect();
        const desired_bottom = tick_top - x_title_gap_px;
        const delta_y = Math.min(0, desired_bottom - title_rect.bottom);
        title.style.setProperty(
          "transform",
          `translateY(${delta_y.toFixed(2)}px)`,
          "important",
        );
      }
    };

    const schedule_alignment = () => {
      requestAnimationFrame(() => requestAnimationFrame(align_heatmap_dom_labels));
    };

    const base_render_plot_dom_alignment = render_plot;
    render_plot = async function(mount, figure, chart_name) {
      const result = await base_render_plot_dom_alignment(mount, figure, chart_name);
      if (chart_name === "heatmap") schedule_alignment();
      return result;
    };

    window.addEventListener("resize", schedule_alignment);
    by_id("heatmap_vertical_scale")?.addEventListener("input", schedule_alignment);
    document.querySelector('.chart-card[data-chart="heatmap"] .maximize-button')
      ?.addEventListener("click", schedule_alignment);

    schedule_alignment();
  }, 320);
});
// ^^^ THOG
