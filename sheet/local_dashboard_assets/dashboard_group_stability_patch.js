// vvv THOG
"use strict";

// Keep W&B-like group navigation structurally stable while a mature run's local
// .wandb file is still catching up. Train is always present first, but remains
// collapsed so startup does not imply chart materialisation/render work. The
// placeholder remains expandable: it must never look like a broken train group
// while the local W&B scanner is still catching up.
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
      button.setAttribute("aria-expanded", "false");
      button.innerHTML = (
        '<span class="group-grip" aria-hidden="true">⠿</span>'
        + '<span class="group-caret" aria-hidden="true">⌄</span>'
        + '<strong>train</strong>'
        + '<span class="group-count">…</span>'
      );
      button.addEventListener("click", () => {
        // Opening the pending train group is an explicit user preference. Carry it
        // into the real W&B-backed group when the first history record arrives.
        queueMicrotask(() => {
          window.__thog2_metric_groups?.set_group_collapsed?.(
            "train",
            section.classList.contains("collapsed"),
          );
          if (!section.classList.contains("collapsed")) {
            window.__thog2_metric_groups?.refresh?.();
          }
        });
      });
      header.appendChild(button);
      const grid = document.createElement("div");
      grid.className = "chart-grid thog2-pending-train-grid";
      grid.hidden = true;
      const message = document.createElement("p");
      message.className = "thog2-pending-train-message";
      message.textContent = "Scanning local W&B history for train charts…";
      grid.appendChild(message);
      section.append(header, grid);
    }

    const collapsed = window.__thog2_metric_groups?.group_is_collapsed?.("train") ?? true;
    section.classList.toggle("collapsed", collapsed);
    const pending_grid = section.querySelector(".chart-grid");
    const pending_toggle = section.querySelector(".chart-group-toggle");
    if (pending_grid) pending_grid.hidden = collapsed;
    pending_toggle?.setAttribute("aria-expanded", String(!collapsed));

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
    .thog2-pending-train-message { margin: 14px 18px; color: #68717c; font-size: 12px; }
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
