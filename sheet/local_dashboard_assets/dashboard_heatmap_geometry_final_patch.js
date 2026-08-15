// vvv THOG
"use strict";

// Final geometry pass: keep literal px/step independent of card height, keep the
// heatmap bottom-anchored to its x axis, put the colour key in a real right-side
// gutter, and keep centre text/newest-step labels stable across maximize/restore.
window.addEventListener("load", () => {
  setTimeout(() => {
    const heatmap_vertical_chrome_px = 94; // 18 top + 76 bottom; cell body is probes * px/step.
    const heatmap_right_key_margin_px = 220;
    const heatmap_colour_key_min_height_px = 90;
    const heatmap_colour_key_max_height_px = 220;

    const base_plot_mount_dimensions_geometry = plot_mount_dimensions;
    plot_mount_dimensions = function(mount, chart_name, figure) {
      if (chart_name !== "heatmap") {
        return base_plot_mount_dimensions_geometry(mount, chart_name, figure);
      }
      const shell = mount.closest(".plot-shell");
      const shell_width = Math.max(1, Number(shell?.clientWidth || 0));
      const probe_count = Math.max(1, heatmap_probe_count(figure));
      return {
        // The restored/default view must show the full heatmap width with no H scroll.
        width: shell_width,
        // Do not stretch to card/full-screen height: literal row pitch is authoritative.
        height: heatmap_vertical_chrome_px + probe_count * heatmap_probe_row_height_px(),
      };
    };

    const centre_annotation = annotation => (
      annotation
      && annotation.xref === "x"
      && annotation.yref === "y"
      && annotation.showarrow === false
      && typeof annotation.hovertext === "string"
      && annotation.hovertext.startsWith("step=")
      && String(annotation.font?.family || "").includes("DejaVu Sans Mono")
    );

    const floating_latest_annotation = annotation => (
      annotation
      && annotation.xref === "paper"
      && annotation.yref === "y"
      && annotation.showarrow === false
      && Number(annotation.font?.size) >= 15
      && /^<b>.*<\/b>$/.test(String(annotation.text || ""))
    );

    const first_step_from_row = row => {
      if (!Array.isArray(row)) return null;
      for (const cell of row) {
        if (!Array.isArray(cell) || cell.length < 1) continue;
        if (cell[0] !== null && cell[0] !== undefined && cell[0] !== "") return cell[0];
      }
      return null;
    };

    const ensure_latest_real_y_tick = (prepared, heatmap_trace) => {
      const coordinates = Array.isArray(heatmap_trace.y) ? heatmap_trace.y : [];
      const customdata = Array.isArray(heatmap_trace.customdata) ? heatmap_trace.customdata : [];
      if (!coordinates.length) return;
      const latest_coordinate = Number(coordinates[coordinates.length - 1]);
      const latest_step = first_step_from_row(customdata[customdata.length - 1]);
      if (!Number.isFinite(latest_coordinate) || latest_step === null) return;

      const tickvals = Array.isArray(prepared.layout?.yaxis?.tickvals)
        ? [...prepared.layout.yaxis.tickvals]
        : [];
      const ticktext = Array.isArray(prepared.layout?.yaxis?.ticktext)
        ? [...prepared.layout.yaxis.ticktext]
        : tickvals.map(String);
      const index = tickvals.findIndex(value => Number(value) === latest_coordinate);
      if (index >= 0) {
        ticktext[index] = String(latest_step);
      } else {
        tickvals.push(latest_coordinate);
        ticktext.push(String(latest_step));
        const paired = tickvals.map((value, pair_index) => ({
          value,
          text: ticktext[pair_index],
        })).sort((left, right) => Number(left.value) - Number(right.value));
        prepared.layout.yaxis.tickvals = paired.map(item => item.value);
        prepared.layout.yaxis.ticktext = paired.map(item => item.text);
        return;
      }
      prepared.layout.yaxis.tickvals = tickvals;
      prepared.layout.yaxis.ticktext = ticktext;
    };

    const fit_centre_annotations = (prepared, heatmap_trace) => {
      const shell = document.querySelector('.chart-card[data-chart="heatmap"] .heatmap-shell');
      const shell_width = Math.max(1, Number(shell?.clientWidth || 0));
      const margin = prepared.layout?.margin || {};
      const plot_width = Math.max(
        1,
        shell_width - Number(margin.l || 0) - heatmap_right_key_margin_px,
      );
      const column_count = Math.max(1, Array.isArray(heatmap_trace.x) ? heatmap_trace.x.length : 1);
      const cell_width = plot_width / column_count;
      const row_height = heatmap_probe_row_height_px();
      const row_limited = Math.max(4, Math.min(13, Math.floor(row_height * 0.82)));
      // "11.000  Δ= -0.303" is 17 monospace glyphs. No artificial 6px floor:
      // when the cell is narrow, shrinking is preferable to spilling into neighbours.
      const width_limited = Math.max(3, Math.floor((cell_width - 2) / (17 * 0.61)));
      const font_size = Math.min(row_limited, width_limited);
      for (const annotation of prepared.layout.annotations || []) {
        if (!centre_annotation(annotation)) continue;
        annotation.x = -0.5;
        annotation.xanchor = "left";
        annotation.xshift = 1;
        annotation.align = "left";
        annotation.font = {
          ...(annotation.font || {}),
          size: font_size,
        };
      }
    };

    const move_colour_key_off_plot = (prepared, heatmap_trace) => {
      prepared.layout.margin = {
        ...(prepared.layout.margin || {}),
        r: Math.max(heatmap_right_key_margin_px, Number(prepared.layout?.margin?.r || 0)),
      };
      const body_height_px = Math.max(
        1,
        (Array.isArray(heatmap_trace.y) ? heatmap_trace.y.length : 1)
          * heatmap_probe_row_height_px(),
      );
      // At very early row counts a vertical key is intrinsically illegible; hide it
      // until there is enough actual heatmap body to carry one cleanly.
      if (body_height_px < 80) {
        heatmap_trace.showscale = false;
        return;
      }
      heatmap_trace.showscale = true;
      const colour_key_height_px = Math.min(
        heatmap_colour_key_max_height_px,
        Math.max(heatmap_colour_key_min_height_px, Math.round(body_height_px * 0.55)),
      );
      heatmap_trace.colorbar = {
        ...(heatmap_trace.colorbar || {}),
        x: 1.025,
        xanchor: "left",
        xpad: 8,
        y: 0.5,
        yanchor: "middle",
        // A pixel-bounded key scales with the chart without ever becoming a
        // full-height bar on tall or maximized heatmaps.
        len: colour_key_height_px,
        lenmode: "pixels",
        thickness: 13,
        thicknessmode: "pixels",
      };
    };

    const base_transpose_heatmap_geometry = transpose_heatmap;
    transpose_heatmap = function(prepared) {
      base_transpose_heatmap_geometry(prepared);
      const heatmap_trace = (prepared.data || []).find(trace => trace.type === "heatmap");
      if (!heatmap_trace) return;

      prepared.layout.annotations = (prepared.layout.annotations || []).filter(
        annotation => !floating_latest_annotation(annotation)
      );
      ensure_latest_real_y_tick(prepared, heatmap_trace);
      move_colour_key_off_plot(prepared, heatmap_trace);
      fit_centre_annotations(prepared, heatmap_trace);
    };

    const style = document.createElement("style");
    style.textContent = `
      .chart-card[data-chart="heatmap"] .ytick:last-of-type text {
        font-size: 15px !important;
        font-weight: 800 !important;
        fill: #171a1f !important;
        transform: translate(-7px, 3px) !important;
      }
      .heatmap-inner-content {
        align-items: flex-end !important;
      }
      .heatmap-shell g.colorbar,
      .heatmap-shell .colorbar {
        transform: none !important;
      }
    `;
    document.head.appendChild(style);

    const rerender_heatmap_after_layout_change = () => {
      requestAnimationFrame(() => requestAnimationFrame(async () => {
        const mount = by_id("heatmap_plot");
        const figure = app.figures?.heatmap;
        if (!mount || !figure || !app.current_run_id) return;
        await render_plot(mount, figure, "heatmap");
      }));
    };

    // Maximize/restore changes cell width but must not stretch row height. Rebuild the
    // heatmap after the card has its final width so centre text is fitted again.
    document.querySelector('.chart-card[data-chart="heatmap"] .maximize-button')
      ?.addEventListener("click", rerender_heatmap_after_layout_change);
    window.addEventListener("resize", rerender_heatmap_after_layout_change);

    if (app.figures && app.current_run_id) rerender_heatmap_after_layout_change();
  }, 10);
});
// ^^^ THOG
