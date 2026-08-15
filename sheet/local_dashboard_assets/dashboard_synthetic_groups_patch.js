// vvv THOG
"use strict";

// Split the old dashboard-only "depth" presentation bucket into two honest
// synthetic groups. The underlying storage names remain unchanged: heatmap data
// come from probe history and coefficient data come from DEPTH weight snapshots.
(() => {
  const source_group = by_id("depth_chart_group");
  const source_grid = by_id("chart_grid");
  const source_toggle = by_id("depth_group_toggle");
  if (!source_group || !source_grid || !source_toggle) return;

  const coefficient_chart_names = Object.keys(chart_titles).filter(name => name !== "heatmap");

  // Keep a non-visible compatibility anchor where the old depth group began.
  // Existing W&B navigation code inserts real metric groups before this anchor.
  const legacy_anchor = document.createElement("span");
  legacy_anchor.id = "depth_chart_group";
  legacy_anchor.hidden = true;
  legacy_anchor.setAttribute("aria-hidden", "true");
  source_group.insertAdjacentElement("beforebegin", legacy_anchor);

  source_group.id = "heatmap_chart_group";
  source_group.dataset.chartGroup = "heatmap";
  source_grid.id = "heatmap_chart_grid";
  source_toggle.id = "heatmap_group_toggle";
  source_toggle.setAttribute("aria-controls", source_grid.id);
  source_toggle.setAttribute("aria-expanded", "false");
  const heatmap_group_name = source_toggle.querySelector("strong");
  if (heatmap_group_name) heatmap_group_name.textContent = "heatmap";
  const heatmap_group_count = source_toggle.querySelector(".group-count");
  if (heatmap_group_count) heatmap_group_count.textContent = "1";
  source_group.classList.add("collapsed");
  source_grid.hidden = true;

  const coefficients_group = document.createElement("section");
  coefficients_group.id = "coefficients_chart_group";
  coefficients_group.className = "chart-group collapsed";
  coefficients_group.dataset.chartGroup = "coefficients";

  const coefficients_header = document.createElement("header");
  coefficients_header.className = "chart-group-header";
  const coefficients_toggle = document.createElement("button");
  coefficients_toggle.id = "coefficients_group_toggle";
  coefficients_toggle.type = "button";
  coefficients_toggle.className = "chart-group-toggle";
  coefficients_toggle.setAttribute("aria-expanded", "false");
  coefficients_toggle.setAttribute("aria-controls", "coefficients_chart_grid");
  coefficients_toggle.innerHTML = (
    '<span class="group-grip" aria-hidden="true">⠿</span>'
    + '<span class="group-caret" aria-hidden="true">⌄</span>'
    + '<strong>coefficients</strong>'
    + `<span class="group-count">${coefficient_chart_names.length}</span>`
  );
  coefficients_header.appendChild(coefficients_toggle);

  const coefficients_grid = document.createElement("div");
  coefficients_grid.id = "coefficients_chart_grid";
  coefficients_grid.className = "chart-grid";
  coefficients_grid.hidden = true;

  for (const card of [...source_grid.querySelectorAll(".chart-card")]) {
    if (card.dataset.chart !== "heatmap") coefficients_grid.appendChild(card);
  }
  coefficients_group.append(coefficients_header, coefficients_grid);
  source_group.insertAdjacentElement("afterend", coefficients_group);

  // Keep the public chart-group description aligned with the UI while retaining
  // app.figures.depth as the compatibility/storage name for coefficient figures.
  chart_groups.heatmap = ["heatmap"];
  chart_groups.coefficients = [...coefficient_chart_names];

  const group_is_open = name => {
    const group = by_id(`${name}_chart_group`);
    return Boolean(group && !group.classList.contains("collapsed"));
  };

  window.__thog2_synthetic_groups = {
    coefficient_chart_names,
    group_is_open,
  };

  // The base dashboard may still be waiting on its first /api/runs request. Gate
  // the old all-figures endpoint now, before that request can lead to a hidden
  // seven-chart materialisation. The later performance overlay supersedes this
  // with independent family fetching once the page has fully loaded.
  if (typeof fetch_json === "function" && !fetch_json.__thog2_synthetic_startup_gate) {
    const base_fetch_json_synthetic_startup = fetch_json;
    const startup_gated_fetch_json = async function(url, options = {}) {
      let parsed = null;
      try {
        parsed = new URL(url, window.location.origin);
      } catch (_error) {
        return base_fetch_json_synthetic_startup(url, options);
      }
      if (
        parsed.pathname === "/api/figures"
        && !options?.method
        && !group_is_open("heatmap")
        && !group_is_open("coefficients")
      ) {
        return app.figures || {
          heatmap: null,
          heatmap_dimensions: {layers: 0, probes: 0},
          depth: {},
        };
      }
      return base_fetch_json_synthetic_startup(url, options);
    };
    startup_gated_fetch_json.__thog2_synthetic_startup_gate = true;
    fetch_json = startup_gated_fetch_json;
  }

  // Cover the short interval before the late performance overlay attaches its
  // family-specific wake handlers. Once that overlay exists, these listeners are
  // intentionally inert and the optimised stale-family logic owns the refresh.
  const early_wake = name => {
    queueMicrotask(() => {
      if (!group_is_open(name) || window.__thog2_dashboard_performance) return;
      if (!app.current_run_id) return;
      app.figure_revision = null;
      refresh_current_run();
    });
  };
  source_toggle.addEventListener("click", () => early_wake("heatmap"));
  coefficients_toggle.addEventListener("click", () => early_wake("coefficients"));
})();
// ^^^ THOG
