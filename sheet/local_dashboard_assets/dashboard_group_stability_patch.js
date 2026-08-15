// vvv THOG
"use strict";

// Keep W&B-like group navigation structurally stable while a mature run's local
// .wandb file is still catching up. Train is always present first, but remains
// collapsed so startup does not imply chart materialisation/render work.
window.addEventListener("load", () => {
  const placeholder_id = "thog2_pending_train_group";

  const real_train_group = () => [...document.querySelectorAll(".local-metric-group")].find(
    section => section.dataset.metricGroup === "train"
  ) || null;

  const remove_placeholder = () => by_id(placeholder_id)?.remove();

  const ensure_placeholder = () => {
    const charts_scroll = by_id("charts_scroll");
    if (!charts_scroll || charts_scroll.hidden || !app.current_run_id) {
      remove_placeholder();
      return;
    }
    if (real_train_group()) {
      remove_placeholder();
      return;
    }

    let section = by_id(placeholder_id);
    if (!section) {
      section = document.createElement("section");
      section.id = placeholder_id;
      section.className = "chart-group thog2-pending-train-group collapsed";
      section.dataset.chartGroup = "train";

      const header = document.createElement("header");
      header.className = "chart-group-header";
      const button = document.createElement("button");
      button.type = "button";
      button.className = "chart-group-toggle";
      button.disabled = true;
      button.setAttribute("aria-expanded", "false");
      button.innerHTML = (
        '<span class="group-grip" aria-hidden="true">⠿</span>'
        + '<span class="group-caret" aria-hidden="true">⌄</span>'
        + '<strong>train</strong>'
        + '<span class="group-count">…</span>'
      );
      header.appendChild(button);
      section.appendChild(header);
    }

    const first_metric_group = charts_scroll.querySelector(":scope > .local-metric-group");
    const depth_group = by_id("depth_chart_group");
    const anchor = first_metric_group || depth_group || charts_scroll.firstElementChild;
    if (section.parentElement !== charts_scroll || section.nextElementSibling !== anchor) {
      charts_scroll.insertBefore(section, anchor || null);
    }
  };

  const style = document.createElement("style");
  style.textContent = `
    .thog2-pending-train-group { min-height: 35px; background: #fff; }
    .thog2-pending-train-group .chart-group-toggle:disabled {
      opacity: 1;
      color: #49515b;
      cursor: default;
    }
  `;
  document.head.appendChild(style);

  const charts_scroll = by_id("charts_scroll");
  if (charts_scroll) {
    const observer = new MutationObserver(ensure_placeholder);
    observer.observe(charts_scroll, {childList: true});
  }

  const base_select_run_group_stability = select_run;
  select_run = function(run_id, options = {}) {
    const result = base_select_run_group_stability(run_id, options);
    queueMicrotask(ensure_placeholder);
    return result;
  };

  const base_local_apply_detail_tab_group_stability = typeof local_apply_detail_tab === "function"
    ? local_apply_detail_tab
    : null;
  if (base_local_apply_detail_tab_group_stability) {
    local_apply_detail_tab = function() {
      const result = base_local_apply_detail_tab_group_stability();
      queueMicrotask(ensure_placeholder);
      return result;
    };
  }

  ensure_placeholder();
});
// ^^^ THOG
