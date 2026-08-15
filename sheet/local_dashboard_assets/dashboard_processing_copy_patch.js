// vvv THOG
"use strict";

// Use neutral chart-loading copy. A chart can already have retained data while its
// group is closed or its payload is still being materialised, so "waiting for the
// first ..." is often factually wrong.
(() => {
  const processing_text = "Processing chart data…";

  const normalise_processing_copy = () => {
    const heatmap_detail = by_id("heatmap_card_detail");
    if (heatmap_detail && /Waiting for the first/i.test(heatmap_detail.textContent || "")) {
      heatmap_detail.textContent = processing_text;
    }

    for (const placeholder of document.querySelectorAll(".plot-placeholder")) {
      if (/Waiting for the first/i.test(placeholder.textContent || "")) {
        placeholder.textContent = processing_text;
      }
    }

    for (const detail of document.querySelectorAll(".chart-heading-copy p")) {
      if (/Waiting for the first/i.test(detail.textContent || "")) {
        detail.textContent = processing_text;
      }
    }
  };

  const base_reset_run_charts_processing_copy = reset_run_charts;
  reset_run_charts = function(...args) {
    const result = base_reset_run_charts_processing_copy(...args);
    normalise_processing_copy();
    return result;
  };

  normalise_processing_copy();
})();
// ^^^ THOG
