// vvv THOG
"use strict";

// Make heatmap wheel zoom a centred, geometry-aware viewport operation. Plotly
// normally preserves shapes/annotations in data units, which makes the centre L
// band balloon during x zoom and leaves centre text at its pre-zoom density.
window.addEventListener("load", () => {
  setTimeout(() => {
    const state_by_mount = new WeakMap();
    const minimum_font_px = 8;
    const maximum_font_px = 16;
    const centre_text_glyph_count = 17;
    const centre_text_width_factor = 0.61;
    const centre_band_padding_px = 10;
    const top_tick_padding_px = 2;

    const finite_number = value => {
      if (value === null || value === undefined || value === "") return null;
      const numeric = Number(value);
      return Number.isFinite(numeric) ? numeric : null;
    };

    const axis_range = (mount, axis_name) => {
      const source = mount?._fullLayout?.[axis_name]?.range || mount?.layout?.[axis_name]?.range;
      if (!Array.isArray(source) || source.length < 2) return null;
      const first = Number(source[0]);
      const second = Number(source[1]);
      return Number.isFinite(first) && Number.isFinite(second) && first !== second
        ? [first, second]
        : null;
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

    const heatmap_trace_for_mount = mount => (
      (mount?.data || []).find(trace => trace?.type === "heatmap") || null
    );

    const current_losses_for_mount = mount => {
      const values = (
        mount?.layout?.meta?.thog2_current_losses
        || mount?._fullLayout?.meta?.thog2_current_losses
        || []
      );
      return Array.isArray(values) ? values : [];
    };

    const step_from_custom_row = row => {
      if (!Array.isArray(row)) return null;
      for (const cell of row) {
        if (!Array.isArray(cell) || !cell.length) continue;
        const step = finite_number(cell[0]);
        if (step !== null) return step;
      }
      return null;
    };

    const delta_colour = value => {
      const numeric = finite_number(value);
      if (numeric === null || numeric === 0) return "#dcdcdc";
      return numeric < 0 ? "#66ff00" : "#ff0000";
    };

    const signed_fixed_3 = value => {
      const numeric = finite_number(value);
      if (numeric === null) return "—";
      return `${numeric >= 0 ? "+" : "-"}${Math.abs(numeric).toFixed(3)}`;
    };

    const plot_size = mount => {
      const size = mount?._fullLayout?._size || {};
      return {
        width: Math.max(1, Number(size.w || mount?.clientWidth || 1)),
        height: Math.max(1, Number(size.h || mount?.clientHeight || 1)),
        top: Math.max(0, Number(size.t || 0)),
      };
    };

    const symmetric_x_range = range => {
      if (!range) return null;
      const span = Math.abs(range[1] - range[0]);
      if (!(span > 0)) return null;
      const radius = Math.max(0.5, span / 2);
      return range[1] >= range[0] ? [-radius, radius] : [radius, -radius];
    };

    const ranges_differ = (left, right, tolerance = 1e-6) => (
      !left
      || !right
      || Math.abs(Number(left[0]) - Number(right[0])) > tolerance
      || Math.abs(Number(left[1]) - Number(right[1])) > tolerance
    );

    const visible_row_geometry = mount => {
      const trace = heatmap_trace_for_mount(mount);
      const y_range = axis_range(mount, "yaxis");
      if (!trace || !y_range) return null;
      const coordinates = Array.isArray(trace.y) ? trace.y.map(Number) : [];
      const losses = current_losses_for_mount(mount);
      const customdata = Array.isArray(trace.customdata) ? trace.customdata : [];
      const lower = Math.min(...y_range);
      const upper = Math.max(...y_range);
      const top_value = y_range[1]; // Plotly's second y-range endpoint is the plot top.
      const bottom_value = y_range[0];
      const denominator = bottom_value - top_value;
      const rows = [];

      for (let index = 0; index < coordinates.length; index += 1) {
        const coordinate = coordinates[index];
        const loss = finite_number(losses[index]);
        if (!Number.isFinite(coordinate) || loss === null) continue;
        if (coordinate < lower - 0.5 || coordinate > upper + 0.5) continue;
        const previous_loss = index > 0 ? finite_number(losses[index - 1]) : null;
        const delta = previous_loss === null ? null : loss - previous_loss;
        const fraction_from_top = denominator === 0
          ? 0
          : (coordinate - top_value) / denominator;
        rows.push({
          index,
          coordinate,
          loss,
          delta,
          step: step_from_custom_row(customdata[index]),
          fraction_from_top,
        });
      }
      rows.sort((left, right) => left.fraction_from_top - right.fraction_from_top);

      const size = plot_size(mount);
      const visible_span = Math.max(1e-6, Math.abs(y_range[1] - y_range[0]));
      const row_pitch_px = size.height / visible_span;
      const font_size = Math.max(
        minimum_font_px,
        Math.min(maximum_font_px, Math.floor(row_pitch_px * 0.90)),
      );
      const minimum_label_gap_px = font_size + 3;
      const label_stride = Math.max(1, Math.ceil(minimum_label_gap_px / Math.max(1, row_pitch_px)));

      return {rows, row_pitch_px, font_size, label_stride, y_range, size};
    };

    const retained_visible_rows = geometry => {
      if (!geometry?.rows?.length) return [];
      const retained = [];
      // Start at the top visible row so the newest/top row wins whenever labels
      // must be thinned. Then retain only rows with adequate rendered separation.
      let last_fraction = null;
      const minimum_fraction_gap = (
        (geometry.font_size + 3) / Math.max(1, geometry.size.height)
      );
      for (const row of geometry.rows) {
        if (last_fraction === null || row.fraction_from_top - last_fraction >= minimum_fraction_gap) {
          retained.push(row);
          last_fraction = row.fraction_from_top;
        }
      }
      return retained;
    };

    const dynamic_centre_annotations = (mount, half_band, geometry) => {
      const rows = retained_visible_rows(geometry);
      return rows.map(row => {
        const loss_text = row.loss.toFixed(3).padStart(6, " ").replace(/^ /, "&nbsp;");
        const delta_text = row.delta === null ? "Δ=      —" : `Δ= ${signed_fixed_3(row.delta)}`;
        return {
          x: -half_band,
          y: row.coordinate,
          xref: "x",
          yref: "y",
          text: `${loss_text}&nbsp;&nbsp;<span style="color:${delta_colour(row.delta)};font-weight:700">${delta_text}</span>`,
          showarrow: false,
          xanchor: "left",
          yanchor: "middle",
          xshift: 2,
          font: {
            family: "DejaVu Sans Mono, monospace",
            size: geometry.font_size,
            color: "#ffffff",
          },
          align: "left",
          captureevents: false,
          hovertext: row.step === null ? "step=—" : `step=${row.step}`,
        };
      });
    };

    const dynamic_centre_band = (mount, x_range, geometry) => {
      const size = geometry?.size || plot_size(mount);
      const x_span = Math.max(1e-6, Math.abs(x_range[1] - x_range[0]));
      const pixels_per_x_unit = size.width / x_span;
      const desired_text_width_px = (
        centre_text_glyph_count
        * geometry.font_size
        * centre_text_width_factor
        + centre_band_padding_px
      );
      return Math.max(0.08, desired_text_width_px / (2 * Math.max(1e-6, pixels_per_x_unit)));
    };

    const align_top_visible_y_tick = mount => {
      if (!mount?._fullLayout?._size) return;
      const mount_rect = mount.getBoundingClientRect();
      const plot_top = mount_rect.top + Number(mount._fullLayout._size.t || 0);
      const plot_bottom = plot_top + Number(mount._fullLayout._size.h || 0);
      const ticks = [...mount.querySelectorAll(".ytick text")];

      for (const tick of ticks) {
        if (!tick.classList.contains("thog2-zoom-top-y-tick")) continue;
        tick.classList.remove("thog2-zoom-top-y-tick");
        tick.style.removeProperty("transform");
        tick.style.removeProperty("transform-box");
        tick.style.removeProperty("transform-origin");
      }

      const visible = ticks.filter(tick => {
        const rect = tick.getBoundingClientRect();
        return rect.bottom >= plot_top - rect.height && rect.top <= plot_bottom;
      });
      if (!visible.length) return;
      const top_tick = visible.reduce((best, candidate) => (
        candidate.getBoundingClientRect().top < best.getBoundingClientRect().top
          ? candidate
          : best
      ));
      top_tick.classList.add("thog2-zoom-top-y-tick");
      top_tick.style.setProperty("transform", "none", "important");
      top_tick.style.setProperty("transform-box", "fill-box", "important");
      top_tick.style.setProperty("transform-origin", "center", "important");
      const rect = top_tick.getBoundingClientRect();
      const minimum_center = plot_top + rect.height / 2 + top_tick_padding_px;
      const current_center = rect.top + rect.height / 2;
      const shift_y = Math.max(0, minimum_center - current_center);
      const shift_x = top_tick.classList.contains("thog2-latest-y-tick") ? -7 : 0;
      top_tick.style.setProperty(
        "transform",
        `translate(${shift_x}px, ${shift_y.toFixed(2)}px)`,
        "important",
      );
    };

    const reflow_heatmap_viewport = async (mount, normalise_x) => {
      if (!mount || mount.dataset.plotReady !== "true") return;
      const state = state_by_mount.get(mount);
      if (!state || state.applying) return;

      const current_x = axis_range(mount, "xaxis");
      const current_y = axis_range(mount, "yaxis");
      if (!current_x || !current_y) return;
      const target_x = normalise_x ? symmetric_x_range(current_x) : current_x;
      const geometry = visible_row_geometry(mount);
      if (!target_x || !geometry) return;

      const half_band = dynamic_centre_band(mount, target_x, geometry);
      const existing_annotations = Array.isArray(mount.layout?.annotations)
        ? mount.layout.annotations
        : [];
      const annotations = existing_annotations.filter(annotation => !centre_datum_annotation(annotation));
      annotations.push(...dynamic_centre_annotations(mount, half_band, geometry));

      const shapes = (Array.isArray(mount.layout?.shapes) ? mount.layout.shapes : []).map(shape => {
        if (shape?.name !== "thog2-centre-datum-background") return shape;
        return {
          ...shape,
          x0: -half_band,
          x1: half_band,
          fillcolor: "#000000",
          line: {width: 0},
        };
      });

      const update = {
        annotations,
        shapes,
      };
      if (normalise_x && ranges_differ(current_x, target_x)) {
        update["xaxis.range"] = target_x;
        update["xaxis.autorange"] = false;
      }

      state.applying = true;
      try {
        await Plotly.relayout(mount, update);
      } finally {
        state.applying = false;
      }
      requestAnimationFrame(() => requestAnimationFrame(() => align_top_visible_y_tick(mount)));
    };

    const schedule_reflow = (mount, normalise_x = false) => {
      let state = state_by_mount.get(mount);
      if (!state) {
        state = {applying: false, frame: null, normalise_x: false, handler: null};
        state_by_mount.set(mount, state);
      }
      state.normalise_x = state.normalise_x || normalise_x;
      if (state.frame !== null) return;
      state.frame = requestAnimationFrame(() => {
        state.frame = null;
        const should_normalise_x = state.normalise_x;
        state.normalise_x = false;
        reflow_heatmap_viewport(mount, should_normalise_x);
      });
    };

    const bind_heatmap_relayout = mount => {
      if (!mount || typeof mount.on !== "function") return;
      let state = state_by_mount.get(mount);
      if (!state) {
        state = {applying: false, frame: null, normalise_x: false, handler: null};
        state_by_mount.set(mount, state);
      }
      if (!state.handler) {
        state.handler = event => {
          if (state.applying) return;
          const keys = Object.keys(event || {});
          const x_changed = keys.some(key => key.startsWith("xaxis.range") || key === "xaxis.autorange");
          const y_changed = keys.some(key => key.startsWith("yaxis.range") || key === "yaxis.autorange");
          if (!x_changed && !y_changed) return;
          schedule_reflow(mount, x_changed);
        };
      }
      if (typeof mount.removeListener === "function") {
        mount.removeListener("plotly_relayout", state.handler);
      }
      mount.on("plotly_relayout", state.handler);
    };

    const base_render_plot_heatmap_zoom = render_plot;
    render_plot = async function(mount, figure, chart_name) {
      const result = await base_render_plot_heatmap_zoom(mount, figure, chart_name);
      if (chart_name === "heatmap") {
        bind_heatmap_relayout(mount);
        schedule_reflow(mount, false);
      }
      return result;
    };

    const heatmap_mount = by_id("heatmap_plot");
    if (heatmap_mount?.dataset.plotReady === "true") {
      bind_heatmap_relayout(heatmap_mount);
      schedule_reflow(heatmap_mount, false);
    }
  }, 560);
});
// ^^^ THOG
