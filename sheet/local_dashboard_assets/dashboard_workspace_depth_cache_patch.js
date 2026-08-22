// vvv THOG
"use strict";

// Workspace weight refreshes previously re-requested every visible run whenever
// any one visible run advanced. Cache each run's raw depth payload by its own
// depth revision so an active run refresh does not repeatedly serialize/download
// unchanged finished runs. The existing Workspace merger still owns all trace
// cloning, colours, filtering and layout composition.
window.addEventListener("load", () => {
  setTimeout(() => {
    const workspace = window.__instra_workspace;
    if (!workspace || typeof workspace.fetch_depth_payload !== "function") return;

    const base_fetch_depth_payload = workspace.fetch_depth_payload;
    const payload_cache = new Map();
    const stats = {hits: 0, misses: 0, evictions: 0};

    const current_only_variant = () => (
      Array.isArray(depth_weight_chart_names)
      && depth_weight_chart_names.length > 0
      && depth_weight_chart_names.every(chart_name => (
        normalize_chart_settings(chart_name)?.current_weights_only === true
      ))
    );

    const visible_runs_by_id = () => new Map(
      (typeof workspace.visible_runs === "function" ? workspace.visible_runs() : [])
        .map(run => [String(run_identifier(run)), run])
    );

    const depth_revision = (run, variant) => JSON.stringify([
      Number(run?.depth_snapshot_count || 0),
      run?.depth_maximum_update ?? null,
      variant,
    ]);

    const prune_hidden_runs = visible_by_id => {
      for (const run_id of [...payload_cache.keys()]) {
        if (visible_by_id.has(run_id)) continue;
        payload_cache.delete(run_id);
        stats.evictions += 1;
      }
    };

    workspace.fetch_depth_payload = async function(request) {
      const visible_by_id = visible_runs_by_id();
      prune_hidden_runs(visible_by_id);
      const variant = current_only_variant() ? "current_only" : "history";

      const cached_request = async url => {
        let parsed = null;
        try {
          parsed = new URL(url, window.location.origin);
        } catch (_error) {
          return request(url);
        }
        if (
          parsed.pathname !== "/api/figure-family"
          || parsed.searchParams.get("family") !== "depth"
        ) {
          return request(url);
        }

        const run_id = String(parsed.searchParams.get("run") || "");
        const run = visible_by_id.get(run_id);
        if (!run) return request(url);
        const revision = depth_revision(run, variant);
        const cached = payload_cache.get(run_id);
        if (cached?.revision === revision) {
          stats.hits += 1;
          return cached.payload;
        }

        stats.misses += 1;
        const payload = await request(url);
        payload_cache.set(run_id, {revision, payload});
        return payload;
      };

      return base_fetch_depth_payload(cached_request);
    };

    window.__instra_workspace_depth_cache = {
      clear: () => payload_cache.clear(),
      size: () => payload_cache.size,
      stats: () => ({...stats, entries: payload_cache.size}),
    };
  }, 1700);
});
// ^^^ THOG
