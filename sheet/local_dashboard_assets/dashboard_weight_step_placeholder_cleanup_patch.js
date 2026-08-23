// vvv THOG
"use strict";

// A future explicit weight-step window owns its placeholders.  Once the window is
// cleared, remove that owned state immediately rather than leaving stale "wait for
// step" copy visible while the normal depth payload refetch completes.
window.addEventListener("load", () => {
  const weight_chart_names = [...depth_weight_chart_names];

  const clear_stale_step_placeholders = () => {
    if (window.__instra_weight_step_filter?.active?.() === true) return;
    for (const chart_name of weight_chart_names) {
      const placeholder = by_id(`${chart_name}_placeholder`);
      const mount = by_id(`${chart_name}_plot`);
      if (placeholder?.classList.contains("instra-step-window-placeholder")) {
        placeholder.classList.remove("instra-step-window-placeholder");
        placeholder.hidden = true;
        placeholder.textContent = "Waiting for the first weight snapshot.";
      }
      if (mount?.dataset) delete mount.dataset.instraStepWindowPlaceholder;
    }
  };

  document.addEventListener("click", event => {
    if (!event.target.closest?.("#weight_step_whole_range")) return;
    queueMicrotask(clear_stale_step_placeholders);
  });

  // Keep the programmatic API consistent with the visible button as well.
  setTimeout(() => {
    const api = window.__instra_weight_controls_v2;
    if (!api || typeof api.clear_step_range !== "function" || api.__instra_placeholder_cleanup === true) return;
    const base_clear_step_range = api.clear_step_range.bind(api);
    api.clear_step_range = function(...args) {
      const result = base_clear_step_range(...args);
      clear_stale_step_placeholders();
      return result;
    };
    api.__instra_placeholder_cleanup = true;
  }, 0);

  window.__instra_weight_step_placeholder_cleanup = Object.freeze({
    clear: clear_stale_step_placeholders,
  });
});
// ^^^ THOG
