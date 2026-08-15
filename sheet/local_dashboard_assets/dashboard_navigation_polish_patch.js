// vvv THOG
"use strict";

// Establish W&B-like initial group state before the metric-group overlay runs,
// while preserving any group state the user has explicitly changed already.
(() => {
  const group_state_key = "thog2_local_metric_group_collapsed";
  try {
    const saved = JSON.parse(localStorage.getItem(group_state_key) || "{}");
    let changed = false;
    if (!Object.prototype.hasOwnProperty.call(saved, "train")) {
      saved.train = false;
      changed = true;
    }
    if (!Object.prototype.hasOwnProperty.call(saved, "system")) {
      saved.system = true;
      changed = true;
    }
    if (changed) localStorage.setItem(group_state_key, JSON.stringify(saved));
  } catch (_error) {
    localStorage.setItem(group_state_key, JSON.stringify({train: false, system: true}));
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
