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
      weight_step_range: app.figures?.weight_step_range ?? null,
    });

    const synthetic_group_open = name => {
      const helper = window.__thog2_synthetic_groups?.group_is_open;
      if (typeof helper === "function") return helper(name);
      const group = by_id(`${name}_chart_group`);
      return Boolean(group && !group.classList.contains("collapsed"));
    };

    const workspace_api = () => {
      const candidate = window.__instra_workspace;
      return candidate?.active?.() === true ? candidate : null;
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

    const heatmap_window_settings = () => {
      const settings = typeof heatmap_settings_for_current_run === "function"
        ? heatmap_settings_for_current_run()
        : {};
      const count = Number(settings.probe_count);
      return {
        probe_count: Number.isInteger(count) && count > 0 ? Math.min(512, count) : 100,
        window_mode: settings.window_mode === "from_zero" ? "from_zero" : "rolling",
      };
    };

    const heatmap_signature = status => {
      const viewer = heatmap_window_settings();
      return JSON.stringify([
      status?.heatmap_count ?? null,
      status?.heatmap_maximum_update ?? null,
      status?.heatmap_settings?.abs_limit ?? null,
        viewer.probe_count,
        viewer.window_mode,
      ]);
    };

    const depth_signature = status => JSON.stringify([
      status?.depth_snapshot_count ?? null,
      status?.depth_maximum_update ?? null,
      workspace_api()?.selection_key?.() || null,
      window.__instra_weight_step_filter?.signature?.() || null,
    ]);

    const depth_figures_present = figures => Boolean(
      figures && Object.keys(figures).length > 0
    );

    const depth_payload_present = () => (
      depth_figures_present(app.figures?.depth)
      || app.figures?.weight_step_range?.snapshot_count === 0
    );

    const depth_payload_expected = status => (
      Number(status?.depth_snapshot_count || 0) > 0
    );

    const family_payload_stale = family => {
      const status = app.current_status || current_run();
      if (!status) return true;
      if (family === "heatmap") {
        return (
          !app.figures?.heatmap
          || performance_state.heatmap_signature !== heatmap_signature(status)
        );
      }
      return (
        (depth_payload_expected(status) && !depth_payload_present())
        || performance_state.depth_signature !== depth_signature(status)
      );
    };

    const heatmap_mount_stale = () => {
      const mount = by_id("heatmap_plot");
      if (!mount) return true;
      return (
        mount.dataset?.plotReady !== "true"
        || mount.dataset?.instraRenderedRunId !== String(app.current_run_id || "")
      );
    };

    const family_stale = family => (
      family_payload_stale(family)
      || (family === "heatmap" && heatmap_mount_stale())
    );

    const base_fetch_json_performance = fetch_json;

    const fetch_family_payload = (family, run_id) => {
      const encoded_run = encodeURIComponent(run_id);
      if (family === "heatmap") {
        const viewer = heatmap_window_settings();
        return base_fetch_json_performance(
          `/api/figure-family?run=${encoded_run}&family=heatmap`
          + `&probe_count=${viewer.probe_count}&window_mode=${encodeURIComponent(viewer.window_mode)}`
        );
      }
      const workspace = workspace_api();
      return workspace
        ? workspace.fetch_depth_payload(base_fetch_json_performance)
        : base_fetch_json_performance(
            `/api/figure-family?run=${encoded_run}&family=depth`
          );
    };

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
        (depth_payload_expected(app.current_status) && !depth_payload_present())
        || performance_state.depth_signature !== next_depth_signature
      );
      const workspace = workspace_api();
      const fetch_heatmap = need_heatmap && !workspace && synthetic_group_open("heatmap");
      const fetch_depth = need_depth && synthetic_group_open("coefficients");

      if (need_heatmap && !fetch_heatmap) performance_state.deferred_heatmap = true;
      if (need_depth && !fetch_depth) performance_state.deferred_coefficients = true;

      if (!fetch_heatmap && !fetch_depth) {
        performance_state.pending_render = {heatmap: false, depth: false};
        return empty_figures();
      }

      try {
        const [heatmap_payload, depth_payload] = await Promise.all([
          fetch_heatmap
            ? fetch_family_payload("heatmap", run_id)
            : Promise.resolve(null),
          fetch_depth
            ? fetch_family_payload("depth", run_id)
            : Promise.resolve(null),
        ]);
        if (run_id !== app.current_run_id) return empty_figures();

        const combined = {
          heatmap: heatmap_payload?.heatmap ?? app.figures?.heatmap ?? null,
          heatmap_dimensions: heatmap_payload?.heatmap_dimensions
            ?? app.figures?.heatmap_dimensions
            ?? {layers: 0, probes: 0},
          depth: depth_payload?.depth ?? app.figures?.depth ?? {},
          weight_step_range: fetch_depth
            ? depth_payload?.weight_step_range ?? null
            : app.figures?.weight_step_range ?? null,
        };
        if (fetch_heatmap) {
          performance_state.heatmap_signature = next_heatmap_signature;
          performance_state.deferred_heatmap = false;
        }
        if (fetch_depth) {
          if (
            depth_figures_present(combined.depth)
            || combined.weight_step_range?.snapshot_count === 0
            || !depth_payload_expected(app.current_status)
          ) {
            performance_state.depth_signature = next_depth_signature;
            performance_state.deferred_coefficients = false;
          } else {
            performance_state.depth_signature = null;
            performance_state.deferred_coefficients = true;
            app.figure_revision = null;
          }
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
          if (
            depth_figures_present(payload?.depth)
            || payload?.weight_step_range?.snapshot_count === 0
            || !depth_payload_expected(app.current_status)
          ) {
            performance_state.depth_signature = next_depth_signature;
            performance_state.deferred_coefficients = false;
          } else {
            performance_state.depth_signature = null;
            performance_state.deferred_coefficients = true;
            app.figure_revision = null;
          }
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
        heatmap: !workspace_api() && synthetic_group_open("heatmap") && Boolean(app.figures.heatmap),
        depth: synthetic_group_open("coefficients") && Boolean(app.figures.depth),
      };

      const status = app.current_status || current_run();
      const heatmap_detail = by_id("heatmap_card_detail");
      if (heatmap_detail) {
        heatmap_detail.textContent = status
          ? `${format_integer(status.heatmap_count)} probes · latest step ${format_integer(status.heatmap_maximum_update)}`
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
            mount.dataset.instraRenderedRunId = String(app.current_run_id || "");
          }
        }
      }

      if (pending.depth) {
        if (!synthetic_group_open("coefficients")) {
          performance_state.deferred_coefficients = true;
        } else {
          const render_jobs = [];
          for (const chart_name of Object.keys(chart_titles).filter(name => name !== "heatmap")) {
            const figure = app.figures.depth?.[chart_name];
            const mount = by_id(`${chart_name}_plot`);
            if (!figure) {
              // A completed empty range must replace old curves, not leave a
              // previous view visible indefinitely beneath a loading label.
              if (mount) {
                if (mount.dataset?.plotReady === "true") clear_plot(mount);
                delete mount.__instraWeightFigure;
                delete mount.dataset.instraWeightContext;
                delete mount.dataset.instraWeightView;
              }
              continue;
            }
            const placeholder = by_id(`${chart_name}_placeholder`);
            const detail = by_id(`${chart_name}_detail`);
            if (placeholder) placeholder.hidden = true;
            if (detail) {
              detail.textContent = `${format_integer(status?.depth_snapshot_count)} retained snapshots · latest step ${format_integer(status?.depth_maximum_update)}`;
            }
            if (mount) render_jobs.push(render_plot(mount, figure, chart_name));
          }
          await Promise.all(render_jobs);
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
      state.latest = {mount, figure, chart_name, run_id: String(app.current_run_id || "")};
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
            if (
              job.chart_name === "heatmap"
              && job.mount?.id === "heatmap_plot"
              && job.run_id === String(app.current_run_id || "")
            ) {
              job.mount.dataset.instraRenderedRunId = job.run_id;
            }
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

    const family_refreshes = new Map();
    const refresh_family_if_stale = family => {
      const group_name = family === "heatmap" ? "heatmap" : "coefficients";
      if (!synthetic_group_open(group_name) || !app.current_run_id) return;
      if (!family_stale(family)) {
        requestAnimationFrame(resize_visible_plots);
        return;
      }
      if (
        family === "heatmap"
        && !family_payload_stale("heatmap")
        && app.figures?.heatmap
      ) {
        performance_state.pending_render = {heatmap: true, depth: false};
        const result = render_figures();
        if (result && typeof result.catch === "function") {
          result.catch(error => show_toast(`Heatmap redraw failed: ${error.message}`));
        }
        return result;
      }
      const existing = family_refreshes.get(family);
      if (existing) return existing;

      if (family === "heatmap") performance_state.deferred_heatmap = true;
      else performance_state.deferred_coefficients = true;
      const run_id = String(app.current_run_id);
      const requested_depth_signature = depth_signature(app.current_status || current_run());
      const refresh = (async () => {
        try {
          const payload = await fetch_family_payload(family, run_id);
          if (
            run_id !== String(app.current_run_id || "")
            || !synthetic_group_open(group_name)
          ) return;
          if (family === "depth" && requested_depth_signature !== depth_signature(app.current_status || current_run())) {
            app.figure_revision = null;
            queueMicrotask(() => refresh_current_run());
            return;
          }

          app.figures = {
            heatmap: family === "heatmap"
              ? payload?.heatmap ?? null
              : app.figures?.heatmap ?? null,
            heatmap_dimensions: family === "heatmap"
              ? payload?.heatmap_dimensions ?? {layers: 0, probes: 0}
              : app.figures?.heatmap_dimensions ?? {layers: 0, probes: 0},
            depth: family === "depth"
              ? payload?.depth ?? {}
              : app.figures?.depth ?? {},
            weight_step_range: family === "depth"
              ? payload?.weight_step_range ?? null
              : app.figures?.weight_step_range ?? null,
          };

          const status = app.current_status || current_run();
          if (family === "heatmap") {
            performance_state.heatmap_signature = heatmap_signature(status);
            performance_state.deferred_heatmap = false;
          } else if (depth_payload_present() || !depth_payload_expected(status)) {
            performance_state.depth_signature = depth_signature(status);
            performance_state.deferred_coefficients = false;
          } else {
            // A known non-empty store must never become a cached successful blank.
            // Leave it stale so the ordinary live refresh retries the family.
            performance_state.depth_signature = null;
            performance_state.deferred_coefficients = true;
            app.figure_revision = null;
          }
          performance_state.pending_render = {
            heatmap: family === "heatmap",
            depth: family === "depth",
          };
          await render_figures();
        } catch (error) {
          show_toast(`${family === "heatmap" ? "Heatmap" : "Weights"} refresh failed: ${error.message}`);
        } finally {
          family_refreshes.delete(family);
        }
      })();
      family_refreshes.set(family, refresh);
      return refresh;
    };

    const bind_synthetic_group = (button_id, family) => {
      by_id(button_id)?.addEventListener("click", () => {
        // Base charts-scroll delegation performs the actual open/close toggle.
        queueMicrotask(() => refresh_family_if_stale(family));
      });
    };
    bind_synthetic_group("heatmap_group_toggle", "heatmap");
    bind_synthetic_group("coefficients_group_toggle", "depth");

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
