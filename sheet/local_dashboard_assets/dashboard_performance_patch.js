// vvv THOG
"use strict";

// Browser-side performance pass. Heatmap and coefficient snapshots are distinct
// synthetic groups and therefore distinct demand boundaries: a closed group does
// not fetch, build, serialize, or Plotly-render its figure family.
window.addEventListener("load", () => {
  setTimeout(() => {
    const performance_state = {
      run_id: null,
      heatmap_signature: null,
      depth_signature: null,
      pending_render: null,
      deferred_heatmap: false,
      deferred_coefficients: false,
    };

    const empty_figures = () => ({
      heatmap: app.figures?.heatmap ?? null,
      heatmap_dimensions: app.figures?.heatmap_dimensions ?? {layers: 0, probes: 0},
      depth: app.figures?.depth ?? {},
    });

    const synthetic_group_open = name => {
      const helper = window.__thog2_synthetic_groups?.group_is_open;
      if (typeof helper === "function") return helper(name);
      const group = by_id(`${name}_chart_group`);
      return Boolean(group && !group.classList.contains("collapsed"));
    };

    const reset_performance_state_for_run = run_id => {
      if (performance_state.run_id === run_id) return;
      performance_state.run_id = run_id;
      performance_state.heatmap_signature = null;
      performance_state.depth_signature = null;
      performance_state.pending_render = null;
      performance_state.deferred_heatmap = false;
      performance_state.deferred_coefficients = false;
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

    const family_stale = family => {
      const status = app.current_status || current_run();
      if (!status) return true;
      if (family === "heatmap") {
        return (
          !app.figures?.heatmap
          || performance_state.heatmap_signature !== heatmap_signature(status)
        );
      }
      return (
        !app.figures?.depth
        || performance_state.depth_signature !== depth_signature(status)
      );
    };

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
        !app.figures?.heatmap
        || performance_state.heatmap_signature !== next_heatmap_signature
      );
      const need_depth = (
        !app.figures?.depth
        || performance_state.depth_signature !== next_depth_signature
      );
      const fetch_heatmap = need_heatmap && synthetic_group_open("heatmap");
      const fetch_depth = need_depth && synthetic_group_open("coefficients");

      if (need_heatmap && !fetch_heatmap) performance_state.deferred_heatmap = true;
      if (need_depth && !fetch_depth) performance_state.deferred_coefficients = true;

      if (!fetch_heatmap && !fetch_depth) {
        performance_state.pending_render = {heatmap: false, depth: false};
        return empty_figures();
      }

      const encoded_run = encodeURIComponent(run_id);
      try {
        const [heatmap_payload, depth_payload] = await Promise.all([
          fetch_heatmap
            ? base_fetch_json_performance(`/api/figure-family?run=${encoded_run}&family=heatmap`)
            : Promise.resolve(null),
          fetch_depth
            ? base_fetch_json_performance(`/api/figure-family?run=${encoded_run}&family=depth`)
            : Promise.resolve(null),
        ]);
        if (run_id !== app.current_run_id) return empty_figures();

        const combined = {
          heatmap: heatmap_payload?.heatmap ?? app.figures?.heatmap ?? null,
          heatmap_dimensions: heatmap_payload?.heatmap_dimensions
            ?? app.figures?.heatmap_dimensions
            ?? {layers: 0, probes: 0},
          depth: depth_payload?.depth ?? app.figures?.depth ?? {},
        };
        if (fetch_heatmap) {
          performance_state.heatmap_signature = next_heatmap_signature;
          performance_state.deferred_heatmap = false;
        }
        if (fetch_depth) {
          performance_state.depth_signature = next_depth_signature;
          performance_state.deferred_coefficients = false;
        }
        performance_state.pending_render = {
          heatmap: fetch_heatmap,
          depth: fetch_depth,
        };
        return combined;
      } catch (_family_error) {
        // Preserve the established all-figures endpoint as a compatibility
        // fallback, but only if at least one synthetic group is actually open.
        const payload = await base_fetch_json_performance(url, options);
        if (synthetic_group_open("heatmap")) {
          performance_state.heatmap_signature = next_heatmap_signature;
          performance_state.deferred_heatmap = false;
        }
        if (synthetic_group_open("coefficients")) {
          performance_state.depth_signature = next_depth_signature;
          performance_state.deferred_coefficients = false;
        }
        performance_state.pending_render = {
          heatmap: synthetic_group_open("heatmap"),
          depth: synthetic_group_open("coefficients"),
        };
        return payload;
      }
    };

    const base_render_figures_performance = render_figures;
    render_figures = async function() {
      const synthetic_groups_present = Boolean(
        by_id("heatmap_chart_group") && by_id("coefficients_chart_group")
      );
      if (!synthetic_groups_present) return base_render_figures_performance();
      if (!app.figures || !app.current_run_id) return;

      const explicit_pending = performance_state.pending_render;
      performance_state.pending_render = null;
      const pending = explicit_pending || {
        heatmap: synthetic_group_open("heatmap") && Boolean(app.figures.heatmap),
        depth: synthetic_group_open("coefficients") && Boolean(app.figures.depth),
      };

      const status = app.current_status || current_run();
      const heatmap_detail = by_id("heatmap_card_detail");
      if (heatmap_detail) {
        heatmap_detail.textContent = status
          ? `${format_integer(status.heatmap_count)} probes · latest step ${format_integer(status.heatmap_maximum_update)} · discrete cells`
          : "Layer-count probes";
      }

      if (pending.heatmap) {
        if (!synthetic_group_open("heatmap")) {
          performance_state.deferred_heatmap = true;
        } else {
          const mount = by_id("heatmap_plot");
          const figure = app.figures.heatmap;
          if (mount && figure) {
            const placeholder = by_id("heatmap_placeholder");
            if (placeholder) placeholder.hidden = true;
            await render_plot(mount, figure, "heatmap");
          }
        }
      }

      if (pending.depth) {
        if (!synthetic_group_open("coefficients")) {
          performance_state.deferred_coefficients = true;
        } else {
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

    const refresh_family_if_stale = family => {
      const group_name = family === "heatmap" ? "heatmap" : "coefficients";
      if (!synthetic_group_open(group_name) || !app.current_run_id) return;
      if (!family_stale(family)) {
        requestAnimationFrame(resize_visible_plots);
        return;
      }
      if (family === "heatmap") performance_state.deferred_heatmap = true;
      else performance_state.deferred_coefficients = true;
      app.figure_revision = null;

      const attempt = () => {
        if (!app.current_run_id || !synthetic_group_open(group_name)) return;
        if (app.refresh_in_flight) {
          setTimeout(attempt, 75);
          return;
        }
        refresh_current_run();
      };
      attempt();
    };

    const bind_synthetic_group = (button_id, family) => {
      by_id(button_id)?.addEventListener("click", () => {
        // Base charts-scroll delegation performs the actual open/close toggle.
        queueMicrotask(() => refresh_family_if_stale(family));
      });
    };
    bind_synthetic_group("heatmap_group_toggle", "heatmap");
    bind_synthetic_group("coefficients_group_toggle", "coefficients");

    // Switching runs invalidates family signatures immediately. Both synthetic
    // groups retain their current open/closed state, but closed families remain
    // completely unmaterialised until explicitly opened.
    const base_select_run_performance = select_run;
    select_run = function(run_id, options = {}) {
      if (run_id !== app.current_run_id) reset_performance_state_for_run(run_id);
      return base_select_run_performance(run_id, options);
    };

    window.__thog2_dashboard_performance = {
      state: performance_state,
      render_queues,
      refresh_family_if_stale,
    };
  }, 420);
});
// ^^^ THOG
