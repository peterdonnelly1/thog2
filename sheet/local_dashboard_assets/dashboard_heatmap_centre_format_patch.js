// vvv THOG
"use strict";

// Final presentation correction for the centre/L datum.  Legacy rows have no
// stored centre loss and must remain blank rather than allowing Number(null)
// to fabricate 0.00.  Genuine rows mirror the compact CLI loss/delta layout.
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
      if (!Number.isFinite(value) || value === 0) return "rgb(220,220,220)";
      return value < 0 ? "rgb(102,255,0)" : "rgb(255,0,0)";
    };

    const is_old_centre_annotation = annotation => (
      annotation
      && Number(annotation.x) === 0
      && annotation.xref === "x"
      && annotation.yref === "y"
      && annotation.showarrow === false
      && typeof annotation.hovertext === "string"
      && annotation.hovertext.startsWith("step=")
      && String(annotation.font?.family || "").includes("DejaVu Sans Mono")
    );

    const centre_annotations = (prepared, heatmap_trace, current_losses) => {
      const coordinates = Array.isArray(heatmap_trace.y) ? heatmap_trace.y : [];
      const customdata = Array.isArray(heatmap_trace.customdata) ? heatmap_trace.customdata : [];
      const row_height = heatmap_probe_row_height_px();
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
        const step = Array.isArray(customdata[index]?.[0])
          ? customdata[index][0][0]
          : customdata[index]?.[0]?.[0];
        const common = {
          x: 0,
          y: coordinates[index],
          xref: "x",
          yref: "y",
          showarrow: false,
          yanchor: "middle",
          font: {family: "DejaVu Sans Mono, monospace", size: 10},
          captureevents: false,
          hovertext: step === undefined ? undefined : `step=${step}`,
        };

        annotations.push({
          ...common,
          text: loss.toFixed(4),
          xanchor: "right",
          xshift: -20,
          font: {...common.font, color: "rgb(255,255,255)"},
        });
        annotations.push({
          ...common,
          text: delta === null ? "Δ=      —" : `Δ= ${signed_fixed_3(delta)}`,
          xanchor: "left",
          xshift: -12,
          font: {...common.font, color: delta_colour(delta)},
        });
      }
      return annotations;
    };

    const base_transpose_heatmap_centre_format = transpose_heatmap;
    transpose_heatmap = function(prepared) {
      base_transpose_heatmap_centre_format(prepared);
      const heatmap_trace = (prepared.data || []).find(trace => trace.type === "heatmap");
      if (!heatmap_trace) return;

      const current_losses = Array.isArray(prepared.layout?.meta?.thog2_current_losses)
        ? prepared.layout.meta.thog2_current_losses
        : [];
      const existing_annotations = Array.isArray(prepared.layout.annotations)
        ? prepared.layout.annotations
        : [];
      prepared.layout.annotations = [
        ...existing_annotations.filter(annotation => !is_old_centre_annotation(annotation)),
        ...centre_annotations(prepared, heatmap_trace, current_losses),
      ];
    };

    const style = document.createElement("style");
    style.textContent = `
      .bulk-delete-runs-button svg {
        width: 19px !important;
        height: 19px !important;
        stroke-width: 1.9 !important;
      }
    `;
    document.head.appendChild(style);

    if (app.figures && app.current_run_id) render_figures();
  }, 5);
});
// ^^^ THOG
