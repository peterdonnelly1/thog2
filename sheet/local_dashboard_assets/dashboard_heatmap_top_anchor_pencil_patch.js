// vvv THOG
"use strict";

// Final authoritative local-dashboard geometry. Install after the earlier heatmap
// overlays have finished composing so no later minimum-size/alignment rule can
// drag the newest-first heatmap back down or manufacture scrollbars between
// otherwise correctly-sized trajectory cards.
window.addEventListener("load", () => {
  setTimeout(() => {
    // The mirrored x axes use 104 px above and 76 px below the body. The larger
    // top allocation keeps the title on its own line above the ordinates.
    const heatmap_chrome_height_px = 180;
    const ordinary_plot_minimum_width_px = 360;
    const ordinary_plot_minimum_height_px = 240;

    const first_step_from_row = row => {
      if (!Array.isArray(row)) return null;
      for (const cell of row) {
        if (!Array.isArray(cell) || cell.length < 1) continue;
        const value = Number(cell[0]);
        if (Number.isFinite(value)) return value;
      }
      return null;
    };

    const natural_heatmap_dimensions = (mount, figure) => {
      const shell = mount.closest(".plot-shell");
      const width = Math.max(1, Number(shell?.clientWidth || mount.clientWidth || 1));
      const probes = Math.max(1, heatmap_probe_count(figure));
      return {
        width,
        height: heatmap_chrome_height_px + probes * heatmap_probe_row_height_px(),
      };
    };

    const base_plot_mount_dimensions_final = plot_mount_dimensions;
    plot_mount_dimensions = function(mount, chart_name, figure) {
      if (chart_name === "heatmap") {
        return natural_heatmap_dimensions(mount, figure);
      }

      const shell = mount.closest(".plot-shell");
      if (!shell) return base_plot_mount_dimensions_final(mount, chart_name, figure);
      return {
        // Normal chart cards should fit their viewport exactly. Only genuinely
        // tiny user-resized cards fall back to an inner scrollbar.
        width: Math.max(ordinary_plot_minimum_width_px, Number(shell.clientWidth || 1)),
        height: Math.max(ordinary_plot_minimum_height_px, Number(shell.clientHeight || 1)),
      };
    };

    const orient_heatmap_newest_at_top = prepared => {
      const heatmap_trace = (prepared.data || []).find(trace => trace.type === "heatmap");
      if (!heatmap_trace || !prepared.layout?.yaxis) return;
      const coordinates = Array.isArray(heatmap_trace.y)
        ? heatmap_trace.y.map(Number)
        : [];
      const customdata = Array.isArray(heatmap_trace.customdata)
        ? heatmap_trace.customdata
        : [];
      const finite_coordinates = coordinates.filter(Number.isFinite);
      if (!finite_coordinates.length) return;

      let newest_index = -1;
      let newest_step = -Infinity;
      for (let index = 0; index < customdata.length; index += 1) {
        const step = first_step_from_row(customdata[index]);
        if (step !== null && step >= newest_step) {
          newest_step = step;
          newest_index = index;
        }
      }
      if (newest_index < 0) newest_index = coordinates.length - 1;
      const newest_coordinate = Number(coordinates[newest_index]);
      if (!Number.isFinite(newest_coordinate)) return;

      const sorted_unique = [...new Set(finite_coordinates)].sort((left, right) => left - right);
      let pitch = 1;
      for (let index = 1; index < sorted_unique.length; index += 1) {
        const difference = sorted_unique[index] - sorted_unique[index - 1];
        if (difference > 0) {
          pitch = difference;
          break;
        }
      }
      const minimum = Math.min(...finite_coordinates);
      const maximum = Math.max(...finite_coordinates);
      const lower_edge = minimum - pitch / 2;
      const upper_edge = maximum + pitch / 2;
      const newest_on_high_side = (
        Math.abs(newest_coordinate - maximum) <= Math.abs(newest_coordinate - minimum)
      );

      prepared.layout.yaxis.range = newest_on_high_side
        ? [lower_edge, upper_edge]
        : [upper_edge, lower_edge];
      prepared.layout.yaxis.autorange = false;
    };

    const finite_number = value => {
      if (value === null || value === undefined || value === "") return null;
      const numeric = Number(value);
      return Number.isFinite(numeric) ? numeric : null;
    };

    const heatmap_x_axis_typeface = prepared => {
      const body_typeface = document.body
        ? window.getComputedStyle(document.body).fontFamily
        : "";
      return String(
        prepared.layout?.xaxis?.tickfont?.family
        || prepared.layout?.font?.family
        || body_typeface
        || "Inter, ui-sans-serif, system-ui, sans-serif"
      );
    };

    const centre_stat_font = prepared => {
      const centre_stat = (prepared.layout?.annotations || []).find(annotation => (
        annotation?.xref === "x"
        && annotation?.yref === "y"
        && annotation?.showarrow === false
        && typeof annotation?.hovertext === "string"
        && annotation.hovertext.startsWith("step=")
        && String(annotation?.font?.family || "").includes("Mono")
      ));
      return {
        family: String(centre_stat?.font?.family || "DejaVu Sans Mono, monospace"),
        size: Math.max(1, Number(centre_stat?.font?.size || 10)),
      };
    };

    const best_better_loss_annotations = (prepared, heatmap_trace) => {
      const x_coordinates = Array.isArray(heatmap_trace.x) ? heatmap_trace.x : [];
      const y_coordinates = Array.isArray(heatmap_trace.y) ? heatmap_trace.y : [];
      const customdata = Array.isArray(heatmap_trace.customdata) ? heatmap_trace.customdata : [];
      const current_losses = Array.isArray(prepared.layout?.meta?.thog2_current_losses)
        ? prepared.layout.meta.thog2_current_losses
        : [];
      const resolved_centre_font = centre_stat_font(prepared);
      const annotations = [];

      for (let row_index = 0; row_index < customdata.length; row_index += 1) {
        const current_loss = finite_number(current_losses[row_index]);
        const y_coordinate = finite_number(y_coordinates[row_index]);
        const row = customdata[row_index];
        if (current_loss === null || y_coordinate === null || !Array.isArray(row)) continue;

        let best = null;
        for (let column_index = 0; column_index < row.length; column_index += 1) {
          const cell = row[column_index];
          const candidate_delta = Array.isArray(cell) ? finite_number(cell[3]) : null;
          if (candidate_delta === null || !(candidate_delta < 0)) continue;
          const candidate_loss = current_loss + candidate_delta;
          const improvement_percent = current_loss === 0
            ? null
            : 100 * (current_loss - candidate_loss) / Math.abs(current_loss);
          if (best === null || candidate_loss < best.loss) {
            best = {column_index, loss: candidate_loss, improvement_percent};
          }
        }
        if (best === null || best.column_index >= x_coordinates.length) continue;
        const percent_suffix = Number.isFinite(best.improvement_percent)
          ? ` (${best.improvement_percent.toFixed(2)}%)`
          : "";

        annotations.push({
          name: "thog2-best-better-loss",
          x: x_coordinates[best.column_index],
          y: y_coordinate,
          xref: "x",
          yref: "y",
          text: `<b>${best.loss.toFixed(4)}${percent_suffix}</b>`,
          showarrow: false,
          xanchor: "center",
          yanchor: "middle",
          font: {
            family: resolved_centre_font.family,
            size: resolved_centre_font.size,
            color: "#000000",
          },
          align: "center",
          captureevents: false,
        });
      }
      return annotations;
    };

    const base_transpose_heatmap_final = transpose_heatmap;
    transpose_heatmap = function(prepared) {
      base_transpose_heatmap_final(prepared);
      if (!prepared?.layout?.xaxis) return;

      orient_heatmap_newest_at_top(prepared);
      const axis_typeface = heatmap_x_axis_typeface(prepared);
      prepared.layout.xaxis = {
        ...prepared.layout.xaxis,
        side: "top",
        anchor: "y",
        tickfont: {
          ...(prepared.layout.xaxis.tickfont || {}),
          family: axis_typeface,
        },
      };
      prepared.layout.xaxis2 = {
        ...prepared.layout.xaxis,
        side: "bottom",
        anchor: "y",
        overlaying: "x",
        matches: "x",
        showgrid: false,
        zeroline: false,
      };
      const heatmap_trace = (prepared.data || []).find(trace => trace.type === "heatmap");
      if (heatmap_trace) {
        const existing_annotations = Array.isArray(prepared.layout.annotations)
          ? prepared.layout.annotations.filter(
              annotation => annotation?.name !== "thog2-best-better-loss"
            )
          : [];
        prepared.layout.annotations = [
          ...existing_annotations,
          ...best_better_loss_annotations(prepared, heatmap_trace),
        ];
      }
      prepared.layout.margin = {
        ...(prepared.layout.margin || {}),
        t: 104,
        b: 76,
      };
    };

    const pin_heatmap_canvas = (mount, figure) => {
      const viewport = mount?.closest(".heatmap-inner-viewport");
      const content = mount?.closest(".heatmap-inner-content");
      const canvas = mount?.closest(".plot-scroll-canvas");
      if (!viewport || !content || !canvas) return;

      const dimensions = natural_heatmap_dimensions(mount, figure);
      canvas.style.setProperty("position", "absolute", "important");
      canvas.style.setProperty("top", "0", "important");
      canvas.style.setProperty("left", "0", "important");
      canvas.style.setProperty("width", `${Math.round(dimensions.width)}px`, "important");
      canvas.style.setProperty("height", `${Math.round(dimensions.height)}px`, "important");
      canvas.style.setProperty("margin", "0", "important");

      content.style.setProperty("display", "block", "important");
      content.style.setProperty("position", "relative", "important");
      content.style.setProperty("align-items", "initial", "important");
      content.style.setProperty("justify-content", "initial", "important");
      content.style.setProperty("min-height", "0", "important");
      content.style.setProperty("height", `${Math.round(dimensions.height)}px`, "important");
      content.style.setProperty("min-width", "0", "important");
      content.style.setProperty("width", `${Math.max(Math.round(dimensions.width), viewport.clientWidth)}px`, "important");

      // Newest data are the top of the heatmap, so every refresh follows them in
      // exactly the same way the newest-first Logs tab follows its top line.
      viewport.scrollTop = 0;
    };

    const base_render_plot_final_geometry = render_plot;
    render_plot = async function(mount, figure, chart_name) {
      const result = await base_render_plot_final_geometry(mount, figure, chart_name);
      if (chart_name === "heatmap") {
        pin_heatmap_canvas(mount, figure);
      }
      return result;
    };

    const style = document.createElement("style");
    style.textContent = `
      /* One true pencil-line separator. */
      .chart-grid,
      .local-metric-group .local-metric-grid {
        gap: 1px !important;
        padding: 1px !important;
        background: #d7dbe0 !important;
      }
      .chart-card,
      .local-metric-card,
      .chart-card:hover,
      .local-metric-card:hover {
        border: 0 !important;
        border-radius: 0 !important;
        box-shadow: none !important;
      }
      .chart-card-header {
        border-bottom: 1px solid #dfe2e6 !important;
      }

      /* The old 320px plot minimum made every normal ~313px chart body overflow
         by a few pixels, creating a permanent dark Firefox scrollbar that looked
         like a fat inter-panel border. Scrollbars are now contingency UI only. */
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

      .heatmap-inner-content {
        min-height: 0 !important;
        align-items: initial !important;
      }

      /* Keep resize hit zones useful while drawing only a pencil line. */
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

    // Install this after all earlier heatmap overlays, then rebuild once using the
    // final authoritative dimensions/orientation.
    if (app.figures?.heatmap && app.current_run_id) {
      queueMicrotask(async () => {
        const mount = by_id("heatmap_plot");
        if (!mount) return;
        await render_plot(mount, app.figures.heatmap, "heatmap");
        pin_heatmap_canvas(mount, app.figures.heatmap);
      });
    }
  }, 100);
});
// ^^^ THOG
