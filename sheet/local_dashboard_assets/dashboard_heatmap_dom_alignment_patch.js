// vvv THOG
"use strict";

// Final DOM-measured alignment for heatmap labels and authoritative signed-log
// ordinate rendering. Plotly can neutralise nominal SVG shifts and preserve source
// axis flags which hide log tick labels, so resolve both from rendered/data state.
window.addEventListener("load", () => {
  setTimeout(() => {
    const newest_y_left_shift_px = -7;
    const trajectory_scale_settings_key = "thog2_local_trajectory_scale_modes";
    const trajectory_chart_names = new Set([
      "attn_q_head_N",
      "attn_k_head_N",
      "attn_v_head_N",
      "attn_out_head_N",
      "mlp_up",
      "mlp_down",
    ]);

    const rendered_heatmap_image = mount => (
      mount?.querySelector(".heatmaplayer image")
      || mount?.querySelector("g.hm image")
      || null
    );

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

    };

    const trajectory_mode = chart_name => (
      load_json(trajectory_scale_settings_key, {})[chart_name] === "log" ? "log" : "linear"
    );

    const signed_log_value = (value, linear_threshold) => {
      const numeric = Number(value);
      if (!Number.isFinite(numeric) || numeric === 0) return numeric;
      return Math.sign(numeric) * Math.log10(1 + Math.abs(numeric) / linear_threshold);
    };

    const signed_log_tick_label = value => {
      const numeric = Number(value);
      const magnitude = Math.abs(numeric);
      if (!Number.isFinite(magnitude)) return "";
      if (magnitude === 0) return "0";
      if (magnitude >= 1000 || magnitude < 0.001) return numeric.toExponential(0);
      return String(Number(numeric.toPrecision(3)));
    };

    const original_trajectory_values = mount => {
      const values = [];
      for (const trace of mount?.data || []) {
        if (!Array.isArray(trace.customdata)) continue;
        for (const value of trace.customdata) {
          const numeric = Number(value);
          if (Number.isFinite(numeric)) values.push(numeric);
        }
      }
      return values;
    };

    const force_signed_log_ordinates = async (mount, chart_name) => {
      if (
        !mount
        || !trajectory_chart_names.has(chart_name)
        || trajectory_mode(chart_name) !== "log"
      ) return;

      const values = original_trajectory_values(mount);
      const maximum_abs = Math.max(...values.map(value => Math.abs(value)), 0);
      if (!(maximum_abs > 0)) return;

      // Match the existing signed-log transform exactly: a four-decade dynamic
      // span above the linear threshold, with explicit symmetric decade ticks.
      const maximum_exponent = Math.ceil(Math.log10(maximum_abs));
      const linear_exponent = maximum_exponent - 4;
      const linear_threshold = Math.max(1e-12, 10 ** linear_exponent);
      const magnitudes = [];
      for (let exponent = linear_exponent; exponent <= maximum_exponent; exponent += 1) {
        magnitudes.push(10 ** exponent);
      }
      const negative_magnitudes = [...magnitudes].reverse().map(value => -value);
      const tick_values = [
        ...negative_magnitudes.map(value => signed_log_value(value, linear_threshold)),
        0,
        ...magnitudes.map(value => signed_log_value(value, linear_threshold)),
      ];
      const tick_text = [
        ...negative_magnitudes.map(signed_log_tick_label),
        "0",
        ...magnitudes.map(signed_log_tick_label),
      ];
      const left_margin = Math.max(108, Number(mount.layout?.margin?.l || 0));

      await Plotly.relayout(mount, {
        "margin.l": left_margin,
        "yaxis.type": "linear",
        "yaxis.tickmode": "array",
        "yaxis.tickvals": tick_values,
        "yaxis.ticktext": tick_text,
        "yaxis.showticklabels": true,
        "yaxis.ticks": "outside",
        "yaxis.ticklen": 4,
        "yaxis.tickwidth": 1,
        "yaxis.tickcolor": "#6d7680",
        "yaxis.tickfont": {size: 10, color: "#30343b"},
        "yaxis.automargin": true,
      });
    };

    const schedule_alignment = () => {
      requestAnimationFrame(() => requestAnimationFrame(align_heatmap_dom_labels));
    };

    const base_render_plot_dom_alignment = render_plot;
    render_plot = async function(mount, figure, chart_name) {
      const result = await base_render_plot_dom_alignment(mount, figure, chart_name);
      if (chart_name === "heatmap") schedule_alignment();
      await force_signed_log_ordinates(mount, chart_name);
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
