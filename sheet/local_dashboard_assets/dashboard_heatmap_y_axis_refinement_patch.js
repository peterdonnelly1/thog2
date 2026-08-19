// vvv THOG
"use strict";

// Keep heatmap y-axis numbering independent of centre-loss metadata. Older rows
// may legitimately predate L_loss capture, but they still own optimiser-step
// coordinates and must participate in axis tick selection.
window.addEventListener("load", () => {
  setTimeout(() => {
    const state_by_mount = new WeakMap();
    const maximum_tick_count = 14;
    const minimum_tick_font_px = 12;
    const maximum_tick_font_px = 17;

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

    const heatmap_trace = mount => (
      (mount?.data || []).find(trace => trace?.type === "heatmap") || null
    );

    const step_from_custom_row = row => {
      if (!Array.isArray(row)) return null;
      for (const cell of row) {
        if (!Array.isArray(cell) || !cell.length) continue;
        const step = finite_number(cell[0]);
        if (step !== null) return step;
      }
      return null;
    };

    const visible_step_pairs = mount => {
      const trace = heatmap_trace(mount);
      const range = axis_range(mount, "yaxis");
      if (!trace || !range) return [];
      const coordinates = Array.isArray(trace.y) ? trace.y.map(Number) : [];
      const customdata = Array.isArray(trace.customdata) ? trace.customdata : [];
      const lower = Math.min(...range);
      const upper = Math.max(...range);
      const pairs = [];
      for (let index = 0; index < coordinates.length; index += 1) {
        const coordinate = Number(coordinates[index]);
        const step = step_from_custom_row(customdata[index]);
        if (!Number.isFinite(coordinate) || step === null) continue;
        if (coordinate < lower - 0.5 || coordinate > upper + 0.5) continue;
        pairs.push({coordinate, step: Number(step)});
      }
      pairs.sort((left, right) => left.step - right.step);
      return pairs;
    };

    const nice_intervals_around = span => {
      const exponent = Math.floor(Math.log10(Math.max(1e-12, span))) - 2;
      const intervals = [];
      for (let power = exponent - 1; power <= exponent + 4; power += 1) {
        const scale = 10 ** power;
        for (const multiplier of [1, 2, 5]) intervals.push(multiplier * scale);
      }
      return [...new Set(intervals.filter(value => Number.isFinite(value) && value > 0))]
        .sort((left, right) => left - right);
    };

    const tick_count_for_interval = (minimum_step, maximum_step, interval) => {
      const first = Math.ceil(minimum_step / interval) * interval;
      const last = Math.floor(maximum_step / interval) * interval;
      if (last < first) return 0;
      return Math.floor((last - first) / interval + 1 + 1e-9);
    };

    const preferred_interval = (minimum_step, maximum_step) => {
      const span = maximum_step - minimum_step;
      if (!(span > 0)) return 1;
      const candidates = nice_intervals_around(span);
      for (const interval of candidates) {
        const count = tick_count_for_interval(minimum_step, maximum_step, interval);
        if (count >= 3 && count <= maximum_tick_count) return interval;
      }
      return candidates[candidates.length - 1] || 1;
    };

    const interpolate_coordinate = (pairs, target_step) => {
      if (!pairs.length) return null;
      if (target_step <= pairs[0].step) return pairs[0].coordinate;
      if (target_step >= pairs[pairs.length - 1].step) return pairs[pairs.length - 1].coordinate;
      for (let index = 1; index < pairs.length; index += 1) {
        const left = pairs[index - 1];
        const right = pairs[index];
        if (target_step < left.step || target_step > right.step) continue;
        const span = right.step - left.step;
        if (!(span > 0)) return left.coordinate;
        const fraction = (target_step - left.step) / span;
        return left.coordinate + fraction * (right.coordinate - left.coordinate);
      }
      return null;
    };

    const ticks_for_mount = mount => {
      const pairs = visible_step_pairs(mount);
      if (pairs.length < 2) return null;
      const minimum_step = pairs[0].step;
      const maximum_step = pairs[pairs.length - 1].step;
      const span = maximum_step - minimum_step;
      if (!(span > 0)) return null;

      if (span <= 20 && pairs.length <= 20) {
        return {
          tickvals: pairs.map(pair => pair.coordinate),
          ticktext: pairs.map(pair => String(pair.step)),
        };
      }

      const interval = preferred_interval(minimum_step, maximum_step);
      const first_tick = Math.ceil(minimum_step / interval) * interval;
      const last_tick = Math.floor(maximum_step / interval) * interval;
      const tickvals = [];
      const ticktext = [];
      for (let step = first_tick; step <= last_tick + interval * 1e-9; step += interval) {
        const coordinate = interpolate_coordinate(pairs, step);
        if (!Number.isFinite(coordinate)) continue;
        tickvals.push(coordinate);
        ticktext.push(String(Number(step.toPrecision(12))));
      }
      return tickvals.length >= 2 ? {tickvals, ticktext} : null;
    };

    const tick_font_size = (mount, tick_count) => {
      const plot_height = Math.max(1, Number(mount?._fullLayout?._size?.h || 1));
      const spacing = plot_height / Math.max(1, Number(tick_count) - 1);
      return Math.max(
        minimum_tick_font_px,
        Math.min(maximum_tick_font_px, Math.round(spacing * 0.30)),
      );
    };

    const ensure_state = mount => {
      let state = state_by_mount.get(mount);
      if (!state) {
        state = {applying: false, frame: null, handler: null};
        state_by_mount.set(mount, state);
      }
      return state;
    };

    const apply_axis_refinement = async mount => {
      if (!mount || mount.dataset.plotReady !== "true") return;
      const state = ensure_state(mount);
      if (state.applying) return;
      const ticks = ticks_for_mount(mount);
      if (!ticks) return;

      const existing_font = mount.layout?.yaxis?.tickfont || {};
      const update = {
        "yaxis.tickmode": "array",
        "yaxis.tickvals": ticks.tickvals,
        "yaxis.ticktext": ticks.ticktext,
        "yaxis.tickfont": {
          ...existing_font,
          size: tick_font_size(mount, ticks.tickvals.length),
          color: existing_font.color || "#30343b",
        },
      };

      state.applying = true;
      try {
        await Plotly.relayout(mount, update);
      } finally {
        state.applying = false;
      }
    };

    const schedule_refinement = mount => {
      const state = ensure_state(mount);
      if (state.frame !== null) return;
      state.frame = requestAnimationFrame(() => {
        state.frame = requestAnimationFrame(() => {
          state.frame = null;
          apply_axis_refinement(mount);
        });
      });
    };

    const bind_axis_refinement = mount => {
      if (!mount || typeof mount.on !== "function") return;
      const state = ensure_state(mount);
      if (!state.handler) {
        state.handler = event => {
          if (state.applying) return;
          const keys = Object.keys(event || {});
          const relevant = keys.some(key => (
            key.startsWith("yaxis.range")
            || key === "yaxis.autorange"
            || key.startsWith("yaxis.tick")
          ));
          if (relevant) schedule_refinement(mount);
        };
      }
      if (typeof mount.removeListener === "function") {
        mount.removeListener("plotly_relayout", state.handler);
      }
      mount.on("plotly_relayout", state.handler);
    };

    const base_render_plot_y_axis_refinement = render_plot;
    render_plot = async function(mount, figure, chart_name) {
      const result = await base_render_plot_y_axis_refinement(mount, figure, chart_name);
      if (chart_name === "heatmap") {
        bind_axis_refinement(mount);
        schedule_refinement(mount);
      }
      return result;
    };

    const mount = by_id("heatmap_plot");
    if (mount?.dataset.plotReady === "true") {
      bind_axis_refinement(mount);
      schedule_refinement(mount);
    }
  }, 640);
});
// ^^^ THOG
