// vvv THOG
"use strict";

// Start every real W&B metric group closed. Synthetic heatmap/coefficients groups
// own their state independently in dashboard_synthetic_groups_patch.js.
(() => {
  const group_state_key = "thog2_local_metric_group_collapsed";

  // Every browser load starts from a cheap, closed navigation state. Choices made
  // after startup remain in force for the rest of that page load.
  try {
    const saved = JSON.parse(localStorage.getItem(group_state_key) || "{}");
    for (const name of Object.keys(saved)) saved[name] = true;
    saved.train = true;
    saved.system = true;
    localStorage.setItem(group_state_key, JSON.stringify(saved));
  } catch (_error) {
    localStorage.setItem(group_state_key, JSON.stringify({train: true, system: true}));
  }

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
