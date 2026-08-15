// vvv THOG
"use strict";

// Final browser-side performance pass. Keep the established UI/Plotly behaviour,
// but avoid sending/rendering unchanged chart families and coalesce bursts of
// Plotly work from sliders, resize drags, and overlapping refreshes.
window.addEventListener("load", () => {
  setTimeout(() => {
    const performance_state = {
      run_id: null,
      heatmap_signature: null,
      depth_signature: null,
      pending_render: null,
      deferred_depth: false,
    };

    const reset_performance_state_for_run = run_id => {
      if (performance_state.run_id === run_id) return;
      performance_state.run_id = run_id;
      performance_state.heatmap_signature = null;
      performance_state.depth_signature = null;
      performance_state.pending_render = null;
      performance_state.deferred_depth = false;
    };

    const heatmap_signature = status => JSON.stringify([
      status?.heatmap_count ?? null,
      status?.heatmap_maximum_update ?? null,
      status?.heatmap_settings?.abs_limit ?? null,
    ]);

    const depth_signature = status => JSON.stringify([
      status?.depth_snapshot_count ?? null,
      status?.depth_maximum_update ?? null,
    ]);

    const base_fetch_json_performance = fetch_json;
    fetch_json = async function(url, options = {}) {
      let parsed = null;
      try {
        parsed = new URL(url, window.location.origin);
      } catch (_error) {
        return base_fetch_json_performance(url, options);
      }
      if (parsed.pathname !== "/api/figures" || options?.method) {
        return base_fetch_json_performance(url, options);
      }

      const run_id = parsed.searchParams.get("run") || app.current_run_id;
      if (!run_id || run_id !== app.current_run_id || !app.current_status) {
        return base_fetch_json_performance(url, options);
      }
      reset_performance_state_for_run(run_id);

      const next_heatmap_signature = heatmap_signature(app.current_status);
      const next_depth_signature = depth_signature(app.current_status);
      const need_heatmap = (
        !app.figures
        || performance_state.heatmap_signature !== next_heatmap_signature
      );
      const need_depth = (
        !app.figures
        || performance_state.depth_signature !== next_depth_signature
      );

      if (!need_heatmap && !need_depth && app.figures) {
        performance_state.pending_render = {heatmap: false, depth: false};
        return app.figures;
      }

      const encoded_run = encodeURIComponent(run_id);
      try {
        const [heatmap_payload, depth_payload] = await Promise.all([
          need_heatmap
            ? base_fetch_json_performance(`/api/figure-family?run=${encoded_run}&family=heatmap`)
            : Promise.resolve(null),
          need_depth
            ? base_fetch_json_performance(`/api/figure-family?run=${encoded_run}&family=depth`)
            : Promise.resolve(null),
        ]);
        if (run_id !== app.current_run_id) return app.figures || {heatmap: null, heatmap_dimensions: {}, depth: {}};

        const combined = {
          heatmap: heatmap_payload?.heatmap ?? app.figures?.heatmap ?? null,
          heatmap_dimensions: heatmap_payload?.heatmap_dimensions
            ?? app.figures?.heatmap_dimensions
            ?? {layers: 0, probes: 0},
          depth: depth_payload?.depth ?? app.figures?.depth ?? {},
        };
        if (need_heatmap) performance_state.heatmap_signature = next_heatmap_signature;
        if (need_depth) performance_state.depth_signature = next_depth_signature;
        performance_state.pending_render = {heatmap: need_heatmap, depth: need_depth};
        return combined;
      } catch (_family_error) {
        // Keep the established all-figures endpoint as a compatibility fallback.
        const payload = await base_fetch_json_performance(url, options);
        performance_state.heatmap_signature = next_heatmap_signature;
        performance_state.depth_signature = next_depth_signature;
        performance_state.pending_render = {heatmap: true, depth: true};
        return payload;
      }
    };

    const base_render_figures_performance = render_figures;
    render_figures = async function() {
      const pending = performance_state.pending_render;
      performance_state.pending_render = null;
      if (!pending) {
        // Explicit/manual rerenders (viewer setting changes, initial overlays, etc.)
        // retain their established full-render behaviour.
        return base_render_figures_performance();
      }
      if (!app.figures || !app.current_run_id) return;

      const status = app.current_status || current_run();
      const heatmap_detail = by_id("heatmap_card_detail");
      if (heatmap_detail) {
        heatmap_detail.textContent = status
          ? `${format_integer(status.heatmap_count)} probes · latest step ${format_integer(status.heatmap_maximum_update)} · discrete cells`
          : "Layer-count probes";
      }

      if (pending.heatmap) {
        const mount = by_id("heatmap_plot");
        const figure = app.figures.heatmap;
        if (mount && figure) {
          const placeholder = by_id("heatmap_placeholder");
          if (placeholder) placeholder.hidden = true;
          await render_plot(mount, figure, "heatmap");
        }
      }

      if (pending.depth) {
        const depth_group = by_id("depth_chart_group");
        const depth_hidden = Boolean(depth_group?.classList.contains("collapsed"));
        if (depth_hidden) {
          performance_state.deferred_depth = true;
        } else {
          performance_state.deferred_depth = false;
          for (const chart_name of Object.keys(chart_titles).filter(name => name !== "heatmap")) {
            const figure = app.figures.depth?.[chart_name];
            if (!figure) continue;
            const mount = by_id(`${chart_name}_plot`);
            const placeholder = by_id(`${chart_name}_placeholder`);
            const detail = by_id(`${chart_name}_detail`);
            if (placeholder) placeholder.hidden = true;
            if (detail) {
              detail.textContent = `${format_integer(status?.depth_snapshot_count)} retained snapshots · latest step ${format_integer(status?.depth_maximum_update)}`;
            }
            if (mount) await render_plot(mount, figure, chart_name);
          }
        }
      }
    };

    // A mature heatmap can take longer to render than the input/refresh cadence.
    // Never accumulate an unbounded queue: one active render plus one newest job.
    const base_render_plot_performance = render_plot;
    const render_queues = new Map();
    render_plot = function(mount, figure, chart_name) {
      let state = render_queues.get(chart_name);
      if (!state) {
        state = {running: false, queued: false, latest: null, promise: Promise.resolve()};
        render_queues.set(chart_name, state);
      }
      state.latest = {mount, figure, chart_name};
      if (state.running) {
        state.queued = true;
        return state.promise;
      }

      state.running = true;
      state.promise = (async () => {
        try {
          do {
            state.queued = false;
            const job = state.latest;
            await base_render_plot_performance(job.mount, job.figure, job.chart_name);
          } while (state.queued);
        } finally {
          state.running = false;
        }
      })();
      return state.promise;
    };

    // Plotly resize is expensive. Pointermove can fire far faster than the display
    // refresh rate, so collapse resize work to one call/card/animation frame.
    const base_resize_plot_in_card_performance = resize_plot_in_card;
    const pending_resize_cards = new Set();
    let resize_frame = null;
    resize_plot_in_card = function(card) {
      if (!card) return;
      pending_resize_cards.add(card);
      if (resize_frame !== null) return;
      resize_frame = requestAnimationFrame(() => {
        resize_frame = null;
        const cards = [...pending_resize_cards];
        pending_resize_cards.clear();
        for (const candidate of cards) base_resize_plot_in_card_performance(candidate);
      });
    };

    // If DEPTH changed while its group was collapsed, draw it once when expanded.
    by_id("depth_group_toggle")?.addEventListener("click", () => {
      queueMicrotask(async () => {
        if (!performance_state.deferred_depth) return;
        const group = by_id("depth_chart_group");
        if (group?.classList.contains("collapsed")) return;
        performance_state.pending_render = {heatmap: false, depth: true};
        await render_figures();
      });
    });

    // Switching runs must invalidate family signatures immediately; select_run's
    // established reset path still owns Plotly clearing and navigation state.
    const base_select_run_performance = select_run;
    select_run = function(run_id, options = {}) {
      if (run_id !== app.current_run_id) reset_performance_state_for_run(run_id);
      return base_select_run_performance(run_id, options);
    };

    window.__thog2_dashboard_performance = {
      state: performance_state,
      render_queues,
    };
  }, 420);
});
// ^^^ THOG
