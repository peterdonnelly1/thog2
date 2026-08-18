// vvv THOG
"use strict";

// Final presentation correction for newest-step emphasis and a colour key that
// remains outside the heatmap body at very small row counts. Retain the useful
// centre/L text over a pure-black background exactly one heatmap cell wide.
window.addEventListener("load", () => {
  setTimeout(() => {
    const strict_finite_number = value => {
      if (value === null || value === undefined || value === "") return null;
      const numeric = Number(value);
      return Number.isFinite(numeric) ? numeric : null;
    };

    const signed_fixed_3 = value => {
      const numeric = strict_finite_number(value);
      if (numeric === null) return "—";
      return `${numeric >= 0 ? "+" : "-"}${Math.abs(numeric).toFixed(3)}`;
    };

    const delta_colour = value => {
      if (!Number.isFinite(value) || value === 0) return "#dcdcdc";
      return value < 0 ? "#66ff00" : "#ff0000";
    };

    const step_from_custom_row = row => {
      if (!Array.isArray(row)) return undefined;
      const populated_cell = row.find(cell => Array.isArray(cell));
      return populated_cell?.[0];
    };

    const is_old_centre_annotation = annotation => (
      annotation
      && annotation.xref === "x"
      && annotation.yref === "y"
      && annotation.showarrow === false
      && typeof annotation.hovertext === "string"
      && annotation.hovertext.startsWith("step=")
      && String(annotation.font?.family || "").includes("DejaVu Sans Mono")
    );

    const centre_cell_width_px = (prepared, heatmap_trace) => {
      const shell = document.querySelector('.chart-card[data-chart="heatmap"] .heatmap-shell');
      const shell_width = Number(shell?.clientWidth || prepared.layout?.width || 0);
      const margin = prepared.layout?.margin || {};
      const plot_width = Math.max(
        1,
        shell_width - Number(margin.l || 0) - Number(margin.r || 0),
      );
      const column_count = Math.max(1, Array.isArray(heatmap_trace.x) ? heatmap_trace.x.length : 1);
      return plot_width / column_count;
    };

    const centre_font_size_px = (prepared, heatmap_trace, row_height) => {
      const row_limited = Math.max(7, Math.min(13, Math.round(Number(row_height) * 0.85)));
      const cell_width = centre_cell_width_px(prepared, heatmap_trace);
      // "11.000  Δ= -0.303" is the widest normal datum: 17 monospace glyphs.
      const width_limited = Math.max(6, Math.floor((cell_width - 3) / (17 * 0.60)));
      return Math.min(row_limited, width_limited);
    };

    const centre_annotations = (prepared, heatmap_trace, current_losses) => {
      const coordinates = Array.isArray(heatmap_trace.y) ? heatmap_trace.y : [];
      const customdata = Array.isArray(heatmap_trace.customdata) ? heatmap_trace.customdata : [];
      const row_height = heatmap_probe_row_height_px();
      const font_size = centre_font_size_px(prepared, heatmap_trace, row_height);
      const retained_indices = new Set();

      if (row_height >= 10) {
        for (let index = 0; index < coordinates.length; index += 1) retained_indices.add(index);
      } else {
        const tickvals = Array.isArray(prepared.layout?.yaxis?.tickvals)
          ? prepared.layout.yaxis.tickvals
          : [];
        const coordinate_to_index = new Map(
          coordinates.map((coordinate, index) => [Number(coordinate), index])
        );
        for (const tick of tickvals) {
          const index = coordinate_to_index.get(Number(tick));
          if (index !== undefined) retained_indices.add(index);
        }
        if (coordinates.length) retained_indices.add(coordinates.length - 1);
      }

      const annotations = [];
      for (const index of [...retained_indices].sort((left, right) => left - right)) {
        const loss = strict_finite_number(current_losses[index]);
        if (loss === null) continue;

        const previous_loss = index > 0
          ? strict_finite_number(current_losses[index - 1])
          : null;
        const delta = previous_loss === null ? null : loss - previous_loss;
        const step = step_from_custom_row(customdata[index]);
        const loss_text = loss.toFixed(3).padStart(6, " ").replace(/^ /, "&nbsp;");
        const delta_text = delta === null ? "Δ=      —" : `Δ= ${signed_fixed_3(delta)}`;
        annotations.push({
          x: -0.5,
          y: coordinates[index],
          xref: "x",
          yref: "y",
          text: `${loss_text}&nbsp;&nbsp;<span style="color:${delta_colour(delta)};font-weight:700">${delta_text}</span>`,
          showarrow: false,
          xanchor: "left",
          yanchor: "middle",
          xshift: 1,
          font: {
            family: "DejaVu Sans Mono, monospace",
            size: font_size,
            color: "#ffffff",
          },
          align: "left",
          captureevents: false,
          hovertext: step === undefined ? undefined : `step=${step}`,
        });
      }
      return annotations;
    };

    const centre_background_shape = prepared => {
      const range = Array.isArray(prepared.layout?.yaxis?.range)
        ? prepared.layout.yaxis.range.map(Number)
        : [0.5, 1.5];
      const finite = range.filter(Number.isFinite);
      return {
        type: "rect",
        xref: "x",
        yref: "y",
        x0: -0.5,
        x1: 0.5,
        y0: finite.length ? Math.min(...finite) : 0.5,
        y1: finite.length ? Math.max(...finite) : 1.5,
        line: {width: 0},
        fillcolor: "#000000",
        layer: "above",
        name: "thog2-centre-datum-background",
      };
    };

    const emphasize_latest_step = (prepared, heatmap_trace, annotations) => {
      const coordinates = Array.isArray(heatmap_trace.y) ? heatmap_trace.y : [];
      const customdata = Array.isArray(heatmap_trace.customdata) ? heatmap_trace.customdata : [];
      if (!coordinates.length) return;
      const latest_coordinate = coordinates[coordinates.length - 1];
      const latest_step = step_from_custom_row(customdata[customdata.length - 1]);
      if (latest_step === undefined || latest_step === null) return;

      const tickvals = Array.isArray(prepared.layout?.yaxis?.tickvals)
        ? [...prepared.layout.yaxis.tickvals]
        : [];
      const ticktext = Array.isArray(prepared.layout?.yaxis?.ticktext)
        ? [...prepared.layout.yaxis.ticktext]
        : tickvals.map(String);
      const match = tickvals.findIndex(value => Number(value) === Number(latest_coordinate));
      if (match >= 0) {
        ticktext[match] = "";
        prepared.layout.yaxis.ticktext = ticktext;
      }
      annotations.push({
        x: 0,
        y: latest_coordinate,
        xref: "paper",
        yref: "y",
        text: `<b>${latest_step}</b>`,
        showarrow: false,
        xanchor: "right",
        yanchor: "middle",
        xshift: -9,
        yshift: -2,
        font: {size: 15, color: "#20252c"},
        captureevents: false,
      });
    };

    const compact_colourbar = (prepared, heatmap_trace) => {
      const colourbar = {...(heatmap_trace.colorbar || {})};
      const row_count = Math.max(1, Array.isArray(heatmap_trace.y) ? heatmap_trace.y.length : 1);
      const body_height_px = row_count * heatmap_probe_row_height_px();
      const percent_mode = heatmap_settings_for_current_run().delta_loss_display_mode === "percent";
      const text = Array.isArray(colourbar.ticktext) ? colourbar.ticktext.map(String) : [];
      const yellow = text.find(value => value.includes("yellow") && !value.includes("≤")) || "yellow";
      const blue = text.find(value => value.includes("blue") && !value.includes("≤")) || "blue";
      const green = text.find(value => value.includes("green") && !value.includes("start")) || "green";
      const red = text.find(value => value.includes("red")) || "red";

      // At one or two literal pixel rows there is no honest way to draw a legible
      // vertical key inside the heatmap body. Hide it until enough body exists.
      if (body_height_px < 36) {
        heatmap_trace.showscale = false;
        return;
      }
      heatmap_trace.showscale = true;

      const domain = Array.isArray(prepared.layout?.yaxis?.domain)
        ? prepared.layout.yaxis.domain.map(Number)
        : [0, 1];
      const lower = Number.isFinite(domain[0]) ? domain[0] : 0;
      const upper = Number.isFinite(domain[1]) ? domain[1] : 1;
      const span = Math.max(0.02, Math.abs(upper - lower));

      colourbar.x = 1.01;
      colourbar.xanchor = "left";
      colourbar.xpad = 5;
      colourbar.y = (lower + upper) / 2;
      colourbar.yanchor = "middle";
      colourbar.len = span;
      colourbar.lenmode = "fraction";
      colourbar.thickness = 13;
      colourbar.thicknessmode = "pixels";
      colourbar.tickfont = {size: body_height_px < 110 ? 8 : 9};
      colourbar.title = {text: `${percent_mode ? "Δloss (%)" : "Δloss"} bands`, side: "top", font: {size: 9}};

      if (body_height_px < 110) {
        colourbar.tickmode = "array";
        colourbar.tickvals = [-0.72, 0, 1];
        colourbar.ticktext = [
          percent_mode ? "Y/B/G negative (%)" : "Y/B/G negative",
          "0",
          red,
        ];
      } else {
        colourbar.tickmode = "array";
        colourbar.tickvals = [-0.88, -0.625, -0.25, 0, 1];
        colourbar.ticktext = [yellow, blue, green, "0", red];
      }
      heatmap_trace.colorbar = colourbar;
    };

    const base_transpose_heatmap_centre_format = transpose_heatmap;
    transpose_heatmap = function(prepared) {
      base_transpose_heatmap_centre_format(prepared);
      const heatmap_trace = (prepared.data || []).find(trace => trace.type === "heatmap");
      if (!heatmap_trace) return;

      // Reserve real room for the colour key instead of dragging it left over the
      // rightmost heatmap cells with a CSS transform.
      prepared.layout.margin = {
        ...(prepared.layout.margin || {}),
        r: Math.max(150, Number(prepared.layout?.margin?.r || 0)),
      };

      const current_losses = Array.isArray(prepared.layout?.meta?.thog2_current_losses)
        ? prepared.layout.meta.thog2_current_losses
        : [];
      const existing_annotations = Array.isArray(prepared.layout.annotations)
        ? prepared.layout.annotations
        : [];
      const annotations = existing_annotations.filter(annotation => !is_old_centre_annotation(annotation));
      // Retain the useful L-column loss/delta text over a one-cell black datum.
      annotations.push(...centre_annotations(prepared, heatmap_trace, current_losses));
      emphasize_latest_step(prepared, heatmap_trace, annotations);
      prepared.layout.annotations = annotations;

      const existing_shapes = Array.isArray(prepared.layout.shapes)
        ? prepared.layout.shapes.filter(shape => shape?.name !== "thog2-centre-datum-background")
        : [];
      prepared.layout.shapes = [...existing_shapes, centre_background_shape(prepared)];
      compact_colourbar(prepared, heatmap_trace);
    };

    const style = document.createElement("style");
    style.textContent = `
      .bulk-delete-runs-button svg {
        width: 19px !important;
        height: 19px !important;
        stroke-width: 1.9 !important;
      }
      /* The Plotly layout now reserves a real right margin for the key. */
      .heatmap-shell g.colorbar,
      .heatmap-shell .colorbar {
        transform: none !important;
      }
    `;
    document.head.appendChild(style);

    // The original px/step slider only changes the Plotly canvas dimensions via
    // relayout(). Re-render just the heatmap after each slider move so annotation
    // font size/density and the compact colour key are recalculated without
    // touching the six trajectory charts.
    const vertical_scale = by_id("heatmap_vertical_scale");
    let scale_render_frame = null;
    vertical_scale?.addEventListener("input", () => {
      if (scale_render_frame !== null) cancelAnimationFrame(scale_render_frame);
      scale_render_frame = requestAnimationFrame(async () => {
        scale_render_frame = null;
        const mount = by_id("heatmap_plot");
        const figure = app.figures?.heatmap;
        if (!mount || !figure || !app.current_run_id) return;
        await render_plot(mount, figure, "heatmap");
      });
    });

    if (app.figures && app.current_run_id) render_figures();
  }, 5);
});
// ^^^ THOG
