// vvv THOG
"use strict";

// The Workspace identity is the visible-run composition, not every live data
// revision. Keep the v0.58 selection key stable across ordinary run writes so
// metric-group DOM is not purged/recreated every few seconds. Weight snapshots
// get their own lightweight revision watch and refresh in place without blanking
// the six existing charts first.
window.addEventListener("load", () => {
  setTimeout(() => {
    const workspace_active = () => app.workspace_mode === true;
    const visible_workspace_runs = () => (
      typeof window.__instra_workspace?.visible_runs === "function"
        ? window.__instra_workspace.visible_runs()
        : app.runs.filter(run => is_visible(run_identifier(run)))
    );

    const stabilise_run_revisions = runs => {
      for (const run of runs || []) {
        if (!run || typeof run !== "object") continue;
        if (!Object.prototype.hasOwnProperty.call(run, "instra_live_revision")) {
          run.instra_live_revision = run.revision;
        }
        run.revision = ["workspace", run_identifier(run)];
      }
    };

    const base_fetch_json_workspace_stability = fetch_json;
    fetch_json = async function(url, options = {}) {
      const payload = await base_fetch_json_workspace_stability(url, options);
      let parsed = null;
      try {
        parsed = new URL(url, window.location.origin);
      } catch (_error) {
        return payload;
      }
      if (
        !options?.method
        && parsed.pathname === "/api/runs"
        && workspace_active()
        && Array.isArray(payload?.runs)
      ) {
        stabilise_run_revisions(payload.runs);
      }
      return payload;
    };

    by_id("workspace_nav")?.addEventListener("click", () => {
      stabilise_run_revisions(app.runs);
    }, true);

    const workspace_view_key = () => visible_workspace_runs().map(run => {
      const id = run_identifier(run);
      return `${id}:${colour_for_run(id)}`;
    }).join("|");

    if (window.__instra_workspace) {
      window.__instra_workspace.selection_key = () => `workspace:${workspace_view_key()}`;
    }

    const composition_key = () => visible_workspace_runs()
      .map(run => run_identifier(run))
      .join("|");
    const depth_key = () => visible_workspace_runs().map(run => (
      `${run_identifier(run)}:${Number(run.depth_snapshot_count || 0)}:${String(run.depth_minimum_update ?? "")}:${String(run.depth_maximum_update ?? "")}`
    )).join("|");

    let last_composition_key = null;
    let last_depth_key = null;
    let smooth_refresh_queued = false;

    const queue_smooth_depth_refresh = () => {
      if (smooth_refresh_queued || !workspace_active()) return;
      smooth_refresh_queued = true;
      const attempt = () => {
        if (!workspace_active()) {
          smooth_refresh_queued = false;
          return;
        }
        if (app.refresh_in_flight) {
          setTimeout(attempt, 80);
          return;
        }
        const performance = window.__thog2_dashboard_performance?.state;
        if (performance) {
          performance.depth_signature = null;
          performance.pending_render = null;
          performance.deferred_coefficients = true;
        }
        app.figure_revision = null;
        smooth_refresh_queued = false;
        refresh_current_run();
      };
      attempt();
    };

    setInterval(() => {
      if (!workspace_active()) {
        last_composition_key = null;
        last_depth_key = null;
        return;
      }
      stabilise_run_revisions(app.runs);
      const next_composition = composition_key();
      const next_depth = depth_key();
      if (last_composition_key === null) {
        last_composition_key = next_composition;
        last_depth_key = next_depth;
        return;
      }
      if (next_composition !== last_composition_key) {
        // The established eye-toggle path owns composition changes.
        last_composition_key = next_composition;
        last_depth_key = next_depth;
        return;
      }
      if (next_depth !== last_depth_key) {
        last_depth_key = next_depth;
        queue_smooth_depth_refresh();
      }
    }, 500);
  }, 0);
});
// ^^^ THOG
