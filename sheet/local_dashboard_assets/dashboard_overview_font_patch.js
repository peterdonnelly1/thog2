// vvv THOG
"use strict";

// Give the W&B-like Overview pane persistent compact font-size controls without
// disturbing the established Overview layout or the independent Logs font size.
window.addEventListener("load", () => {
  setTimeout(() => {
    const storage_key = "thog2_local_overview_font_size";
    const minimum_px = 8;
    const maximum_px = 18;
    const default_px = 11;

    let overview_font_size = Number(localStorage.getItem(storage_key));
    if (!Number.isFinite(overview_font_size)) overview_font_size = default_px;
    overview_font_size = Math.max(minimum_px, Math.min(maximum_px, Math.round(overview_font_size)));

    const apply_overview_font_size = () => {
      const pane = by_id("run_overview_pane") || document.querySelector(".run-overview-pane");
      if (!pane) return;
      pane.style.setProperty("--thog2-overview-font-size", `${overview_font_size}px`);
      const smaller = by_id("overview_font_smaller");
      const larger = by_id("overview_font_larger");
      if (smaller) smaller.disabled = overview_font_size <= minimum_px;
      if (larger) larger.disabled = overview_font_size >= maximum_px;
    };

    const save_overview_font_size = value => {
      overview_font_size = Math.max(minimum_px, Math.min(maximum_px, Math.round(value)));
      localStorage.setItem(storage_key, String(overview_font_size));
      apply_overview_font_size();
    };

    const install_controls = () => {
      const pane = by_id("run_overview_pane") || document.querySelector(".run-overview-pane");
      if (!pane) return false;
      if (!by_id("overview_font_controls")) {
        const anchor = document.createElement("div");
        anchor.id = "overview_font_controls";
        anchor.className = "overview-font-controls-anchor";
        const controls = document.createElement("div");
        controls.className = "overview-font-controls";

        const smaller = document.createElement("button");
        smaller.id = "overview_font_smaller";
        smaller.type = "button";
        smaller.className = "overview-font-button";
        smaller.textContent = "A↓";
        smaller.title = "Decrease Overview font size";
        smaller.setAttribute("aria-label", smaller.title);
        smaller.addEventListener("click", event => {
          event.preventDefault();
          event.stopPropagation();
          save_overview_font_size(overview_font_size - 1);
        });

        const larger = document.createElement("button");
        larger.id = "overview_font_larger";
        larger.type = "button";
        larger.className = "overview-font-button";
        larger.textContent = "A↑";
        larger.title = "Increase Overview font size";
        larger.setAttribute("aria-label", larger.title);
        larger.addEventListener("click", event => {
          event.preventDefault();
          event.stopPropagation();
          save_overview_font_size(overview_font_size + 1);
        });

        controls.append(smaller, larger);
        anchor.appendChild(controls);
        pane.prepend(anchor);
      }
      apply_overview_font_size();
      return true;
    };

    const style = document.createElement("style");
    style.textContent = `
      .run-overview-pane {
        --thog2-overview-font-size: 11px;
      }
      .overview-font-controls-anchor {
        position: sticky;
        z-index: 20;
        top: 8px;
        height: 0;
        display: flex;
        justify-content: flex-end;
        overflow: visible;
        pointer-events: none;
      }
      .overview-font-controls {
        display: inline-flex;
        gap: 4px;
        pointer-events: auto;
      }
      .overview-font-button {
        width: 36px;
        height: 28px;
        padding: 0;
        border: 1px solid #c8cdd3;
        border-radius: 4px;
        background: #f9fafb;
        color: #3f4751;
        font-size: 14px;
        line-height: 1;
        cursor: pointer;
      }
      .overview-font-button:hover:not(:disabled) { background: #e9ecef; }
      .overview-font-button:active:not(:disabled) {
        background: #d7dbe0;
        box-shadow: inset 0 1px 2px rgba(28,34,40,.22);
      }
      .overview-font-button:disabled { opacity: .42; cursor: default; }

      .run-overview-pane .overview-meta-label,
      .run-overview-pane .overview-meta-value,
      .run-overview-pane .overview-hardware-grid,
      .run-overview-pane .overview-artifact-table {
        font-size: var(--thog2-overview-font-size) !important;
      }
      .run-overview-pane .overview-key-name,
      .run-overview-pane .overview-key-value,
      .run-overview-pane .overview-search input,
      .run-overview-pane .overview-panel-heading span,
      .run-overview-pane .overview-artifact-note,
      .run-overview-pane .overview-object-details pre {
        font-size: max(7px, calc(var(--thog2-overview-font-size) - 1px)) !important;
      }
      .run-overview-pane .overview-panel-heading h3,
      .run-overview-pane .overview-artifact-outputs h3 {
        font-size: calc(var(--thog2-overview-font-size) + 2px) !important;
      }
    `;
    document.head.appendChild(style);

    if (!install_controls()) {
      const observer = new MutationObserver(() => {
        if (install_controls()) observer.disconnect();
      });
      observer.observe(document.body, {childList: true, subtree: true});
    }
  }, 0);
});
// ^^^ THOG
