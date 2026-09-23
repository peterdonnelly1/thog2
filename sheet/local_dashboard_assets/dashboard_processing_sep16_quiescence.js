// vvv THOG
"use strict";

(function install_processing_sep16_quiescence() {
  if (processing_view.timer) {
    window.clearInterval(processing_view.timer);
    processing_view.timer = null;
  }

  let last_finished_poll_ms = 0;
  const finished_poll_interval_ms = 10000;

  processing_view.timer = window.setInterval(() => {
    // vvv THOG suspend background polling while the tab is hidden
    if (document.visibilityState !== "visible") return;
    // ^^^ THOG
    const selected = typeof current_run === "function" ? current_run() : null;
    const active = selected && is_active_run_state(selected.run_state);
    if (active) {
      processing_refresh();
      return;
    }
    const now = Date.now();
    if (now - last_finished_poll_ms < finished_poll_interval_ms) return;
    last_finished_poll_ms = now;
    processing_refresh();
  }, 1500);

  // vvv THOG refresh once on return instead of polling a hidden tab
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState !== "visible") return;
    last_finished_poll_ms = Date.now();
    processing_refresh(true);
  });
  // ^^^ THOG
})();
// ^^^ THOG
