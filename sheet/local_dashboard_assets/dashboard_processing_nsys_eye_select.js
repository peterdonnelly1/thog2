// vvv THOG
"use strict";

(function install_processing_nsys_eye_select() {
  function is_nsys_run(run) {
    const artifact = String(run?.artifact_name || run?.run_name || "").toUpperCase();
    return artifact.includes("_NSYS_") || artifact.includes("NSYS_PREMAT");
  }

  function run_for_selection(run_id) {
    return (app.runs || []).find(candidate => String(run_identifier(candidate)) === String(run_id)) || null;
  }

  function refresh_processing_now_for(run_id) {
    const run = run_for_selection(run_id);
    if (!run || !is_nsys_run(run)) return;
    queueMicrotask(() => {
      if (typeof processing_refresh !== "function") return;
      processing_view.run_id = null;
      processing_view.revision = null;
      processing_refresh();
    });
  }

  // Selecting an NSYS source by name, row, route or the eye must not wait for
  // the finished-run polling cadence before profiler pairing becomes visible.
  const select_run_before_nsys_processing_refresh = select_run;
  select_run = function(run_id, options = {}) {
    const result = select_run_before_nsys_processing_refresh(run_id, options);
    refresh_processing_now_for(run_id);
    return result;
  };

  const runs_body = by_id("runs_body");
  if (!runs_body || runs_body.dataset.instraNsysEyeSelectInstalled === "true") return;
  runs_body.dataset.instraNsysEyeSelectInstalled = "true";

  runs_body.addEventListener("click", event => {
    const eye = event.target.closest?.(".eye-button");
    if (!eye) return;
    const row = eye.closest("tr[data-run-id]");
    if (!row) return;
    const run_id = String(row.dataset.runId || "");
    const run = run_for_selection(run_id);
    if (!run || !is_nsys_run(run)) return;

    // The button's own listener runs before this delegated tbody listener, so
    // visibility already reflects the click. Only opening the eye promotes the
    // NSYS run to the active Processing source; closing it remains an ordinary
    // visibility action.
    if (!is_visible(run_id)) return;

    if (String(app.current_run_id || "") !== run_id) {
      select_run(run_id, {manual:true});
    } else {
      refresh_processing_now_for(run_id);
    }
  });
})();
// ^^^ THOG
