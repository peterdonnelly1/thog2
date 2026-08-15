// vvv THOG
"use strict";

// Start every chart group closed. Closed W&B groups already avoid payload/render
// work; depth needs an explicit gate because its heatmap and six coefficient
// figures otherwise continue to be fetched/rendered behind the collapsed header.
(() => {
  const group_state_key = "thog2_local_metric_group_collapsed";

  // Every browser load starts from a cheap, closed navigation state. Group choices
  // made after startup still behave normally for the remainder of that page load.
  try {
    const saved = JSON.parse(localStorage.getItem(group_state_key) || "{}");
    for (const name of Object.keys(saved)) saved[name] = true;
    saved.train = true;
    saved.system = true;
    localStorage.setItem(group_state_key, JSON.stringify(saved));
  } catch (_error) {
    localStorage.setItem(group_state_key, JSON.stringify({train: true, system: true}));
  }

  const depth_group = by_id("depth_chart_group");
  const depth_grid = by_id("chart_grid");
  const depth_toggle = by_id("depth_group_toggle");
  if (depth_group && depth_grid && depth_toggle) {
    depth_group.classList.add("collapsed");
    depth_grid.hidden = true;
    depth_toggle.setAttribute("aria-expanded", "false");
  }

  window.__thog2_depth_group_deferred = true;

  const depth_is_collapsed = () => Boolean(
    by_id("depth_chart_group")?.classList.contains("collapsed")
  );

  const empty_figures = () => ({
    heatmap: app.figures?.heatmap ?? null,
    heatmap_dimensions: app.figures?.heatmap_dimensions ?? {layers: 0, probes: 0},
    depth: app.figures?.depth ?? {},
  });

  const install_closed_depth_gate = () => {
    // fetch_json is wrapped again later by the performance pass. Re-installing
    // after load places this cheap visibility gate outside that wrapper too.
    if (typeof fetch_json === "function" && !fetch_json.__thog2_closed_depth_gate) {
      const base_fetch_json_closed_depth = fetch_json;
      const gated_fetch_json = async function(url, options = {}) {
        let path = "";
        try {
          path = new URL(url, window.location.origin).pathname;
        } catch (_error) {
          return base_fetch_json_closed_depth(url, options);
        }
        if (path === "/api/figures" && !options?.method && depth_is_collapsed()) {
          window.__thog2_depth_group_deferred = true;
          return empty_figures();
        }
        return base_fetch_json_closed_depth(url, options);
      };
      gated_fetch_json.__thog2_closed_depth_gate = true;
      fetch_json = gated_fetch_json;
    }

    if (typeof render_figures === "function" && !render_figures.__thog2_closed_depth_gate) {
      const base_render_figures_closed_depth = render_figures;
      const gated_render_figures = async function(...args) {
        if (depth_is_collapsed()) {
          window.__thog2_depth_group_deferred = true;
          return;
        }
        return base_render_figures_closed_depth(...args);
      };
      gated_render_figures.__thog2_closed_depth_gate = true;
      render_figures = gated_render_figures;
    }
  };

  // Install immediately to catch the first asynchronous catalog refresh, then
  // again after the late performance overlay has composed its wrappers.
  install_closed_depth_gate();
  window.addEventListener("load", () => setTimeout(install_closed_depth_gate, 520));

  depth_toggle?.addEventListener("click", () => {
    queueMicrotask(() => {
      if (depth_is_collapsed()) return;
      if (!window.__thog2_depth_group_deferred && app.figures) return;
      window.__thog2_depth_group_deferred = false;
      // Status may already have advanced while the group was closed. Force one
      // family refresh now that opening the group has made Plotly work useful.
      app.figure_revision = null;
      refresh_current_run();
    });
  });

  const style = document.createElement("style");
  style.textContent = `
    .chart-card {
      border: 1px solid #d7dbe0 !important;
      border-radius: 2px !important;
      box-shadow: none !important;
    }
    .chart-card:hover {
      border-color: #c9ced4 !important;
      box-shadow: none !important;
    }
    .chart-card-header {
      border-bottom: 1px solid #e1e4e8 !important;
    }
  `;
  document.head.appendChild(style);
})();
// ^^^ THOG
