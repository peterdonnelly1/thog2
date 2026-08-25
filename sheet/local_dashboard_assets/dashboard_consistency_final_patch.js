// vvv THOG
"use strict";

// Last-loaded consistency guard for three presentation details that depend on the
// final composed dashboard stack: a cached heatmap must own a live Plotly mount,
// its colour key must sit clear of the matrix body, and Overview columns share the
// available width evenly with compact labels.
window.addEventListener("load", () => {
  setTimeout(() => {
    const heatmap_group_open = () => {
      const helper = window.__thog2_synthetic_groups?.group_is_open;
      if (typeof helper === "function") return helper("heatmap");
      const group = by_id("heatmap_chart_group");
      return Boolean(group && !group.classList.contains("collapsed"));
    };

    const heatmap_mount_current = () => {
      const mount = by_id("heatmap_plot");
      return Boolean(
        mount
        && mount.dataset?.plotReady === "true"
        && mount.dataset?.instraRenderedRunId === String(app.current_run_id || "")
      );
    };

    let reconcile_busy = false;
    let last_reconcile_at = 0;
    let geometry_run_id = "";
    const reconcile_heatmap = async () => {
      const run_id = String(app.current_run_id || "");
      const needs_geometry_refresh = geometry_run_id !== run_id;
      if (
        reconcile_busy
        || app.workspace_mode === true
        || !run_id
        || !heatmap_group_open()
        || (heatmap_mount_current() && !needs_geometry_refresh)
      ) return;

      let status = app.current_status;
      try { status = status || current_run(); }
      catch (_error) {}
      const has_payload = Boolean(app.figures?.heatmap);
      const has_records = Number(status?.heatmap_count || 0) > 0;
      if (!has_payload && !has_records) return;

      const now = Date.now();
      if (now - last_reconcile_at < 750) return;
      last_reconcile_at = now;
      reconcile_busy = true;
      try {
        // The final geometry wrapper can install after an otherwise successful
        // first render. Re-render the cached payload once per run so the new key
        // gutter applies without waiting for a probe or a manual run switch.
        if (has_payload && needs_geometry_refresh && typeof render_plot === "function") {
          const mount = by_id("heatmap_plot");
          const placeholder = by_id("heatmap_placeholder");
          if (placeholder) placeholder.hidden = true;
          await render_plot(mount, app.figures.heatmap, "heatmap");
          if (run_id === String(app.current_run_id || "")) {
            if (mount) mount.dataset.instraRenderedRunId = run_id;
            geometry_run_id = run_id;
          }
          return;
        }
        const performance = window.__thog2_dashboard_performance;
        if (typeof performance?.refresh_family_if_stale === "function") {
          await performance.refresh_family_if_stale("heatmap");
          return;
        }
        if (has_payload && typeof render_plot === "function") {
          const mount = by_id("heatmap_plot");
          const placeholder = by_id("heatmap_placeholder");
          if (placeholder) placeholder.hidden = true;
          await render_plot(mount, app.figures.heatmap, "heatmap");
          if (mount) mount.dataset.instraRenderedRunId = String(app.current_run_id || "");
          return;
        }
        if (!app.refresh_in_flight && typeof refresh_current_run === "function") {
          app.figure_revision = null;
          await refresh_current_run();
        }
      } catch (error) {
        if (typeof show_toast === "function") {
          show_toast(`Heatmap recovery failed: ${error.message}`);
        }
      } finally {
        reconcile_busy = false;
      }
    };

    if (typeof select_run === "function") {
      const base_select_run_consistency_final = select_run;
      select_run = function(run_id, options = {}) {
        const result = base_select_run_consistency_final(run_id, options);
        queueMicrotask(reconcile_heatmap);
        setTimeout(reconcile_heatmap, 400);
        return result;
      };
    }

    if (typeof prepare_figure === "function") {
      const base_prepare_figure_consistency_final = prepare_figure;
      prepare_figure = function(figure, chart_name) {
        const prepared = base_prepare_figure_consistency_final(figure, chart_name);
        if (chart_name !== "heatmap" || !prepared || typeof prepared !== "object") {
          return prepared;
        }
        const heatmap = (prepared.data || []).find(trace => trace?.type === "heatmap");
        if (!heatmap) return prepared;
        prepared.layout = {...(prepared.layout || {})};
        prepared.layout.margin = {
          ...(prepared.layout.margin || {}),
          r: Math.max(270, Number(prepared.layout?.margin?.r || 0)),
        };
        const shell = document.querySelector('.chart-card[data-chart="heatmap"] .heatmap-shell');
        const shell_width = Math.max(0, Number(shell?.clientWidth || 0));
        const plot_width = Math.max(
          1,
          shell_width
            - Number(prepared.layout.margin.l || 0)
            - Number(prepared.layout.margin.r || 0),
        );
        const body_to_key_gap_px = 64;
        heatmap.colorbar = {
          ...(heatmap.colorbar || {}),
          x: shell_width > 0 ? 1 + body_to_key_gap_px / plot_width : 1.1,
          xanchor: "left",
          xpad: 12,
        };
        return prepared;
      };
    }

    const style = document.createElement("style");
    style.id = "thog2_dashboard_consistency_final_style";
    style.textContent = `
      .run-overview-pane .overview-metadata {
        grid-template-columns: 104px minmax(0, 1fr) !important;
      }
      .run-overview-pane .overview-hardware-grid {
        grid-template-columns: 90px minmax(0, 1fr) !important;
      }
      .run-overview-pane .overview-data-grid {
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) !important;
        gap: 14px !important;
      }
      .run-overview-pane .overview-key-row {
        grid-template-columns: minmax(120px, .45fr) minmax(180px, 1.55fr) !important;
        gap: 10px !important;
      }
      @media (max-width: 1100px) {
        .run-overview-pane .overview-data-grid { grid-template-columns: 1fr !important; }
      }
    `;
    document.head.appendChild(style);

    // Raise the default by one pixel while preserving an explicitly non-default
    // choice. The companion default marker lets later revisions distinguish the
    // product default from a user preference.
    const overview_default_key = "thog2_local_overview_default_font_size";
    const overview_current_key = "thog2_local_overview_font_size";
    const raw_default = localStorage.getItem(overview_default_key);
    const raw_current = localStorage.getItem(overview_current_key);
    const old_default = raw_default === null ? Number.NaN : Number(raw_default);
    const old_current = raw_current === null ? Number.NaN : Number(raw_current);
    if (!Number.isFinite(old_default) || old_default === 12) {
      localStorage.setItem(overview_default_key, "13");
      if (!Number.isFinite(old_current) || old_current === old_default || old_current === 12) {
        const larger = by_id("overview_font_larger");
        if (larger && old_current === 12) larger.click();
        else {
          localStorage.setItem(overview_current_key, "13");
          const pane = by_id("run_overview_pane") || document.querySelector(".run-overview-pane");
          pane?.style.setProperty("--thog2-overview-font-size", "13px");
        }
      }
    }

    queueMicrotask(reconcile_heatmap);
    setInterval(reconcile_heatmap, 1000);
    window.__instra_dashboard_consistency_final = Object.freeze({
      installed: true,
      reconcile_heatmap,
    });
  }, 1000);
});
// ^^^ THOG
