// vvv THOG
"use strict";

(function install_instra_startup_feedback() {
  const style = document.createElement("style");
  style.id = "instra-startup-feedback-style";
  style.textContent = `
    .runs-table th.duration-column { text-transform:none !important; }
  `;
  document.head.appendChild(style);

  const watch_status = by_id("watch_status");
  const startup_text = "Reading run databases…";
  let catalog_ready_emitted = false;

  function status_is_ready() {
    const text = String(watch_status?.textContent || "").trim();
    return Boolean(
      text
      && text !== "Connecting…"
      && text !== startup_text
      && !text.startsWith("Viewer error:")
    );
  }

  function emit_catalog_ready_once() {
    if (catalog_ready_emitted || !status_is_ready()) return;
    catalog_ready_emitted = true;
    window.dispatchEvent(new CustomEvent("instra:catalog-ready"));
  }

  // dashboard.js has already started its first asynchronous catalogue read by
  // the time this appended overlay executes. Replace the uninformative
  // "Connecting…" text while that read is outstanding; no controls are disabled.
  if (watch_status && String(watch_status.textContent || "").trim() === "Connecting…") {
    watch_status.textContent = startup_text;
    watch_status.title = "Instra is reading local run databases. Controls remain available.";
  }

  if (watch_status && typeof MutationObserver === "function") {
    const observer = new MutationObserver(() => {
      emit_catalog_ready_once();
      if (catalog_ready_emitted) observer.disconnect();
    });
    observer.observe(watch_status, {childList:true, characterData:true, subtree:true});
  }

  // Covers the fast-start case where the first catalogue read completed before
  // this overlay installed its observer.
  queueMicrotask(emit_catalog_ready_once);
})();
// ^^^ THOG
