// vvv THOG
"use strict";

// Final owner for the August 30 run-table and Overview interaction refinements.
// This asset loads after the historical patch stack and preserves live input/scroll
// state whenever the periodically refreshed run record repopulates Overview.
window.addEventListener("load", () => {
  const panel_ids = [
    "overview_summary_panel",
    "overview_config_panel",
    "overview_artifact_outputs",
  ];

  const capture_overview_state = () => {
    const state = {};
    for (const id of panel_ids) {
      const panel = by_id(id);
      const input = panel?.querySelector?.('input[type="search"]');
      const rows = panel?.querySelector?.(".overview-key-rows");
      const disclosure = panel?.querySelector?.(".overview-collapsible");
      state[id] = {
        query: input?.value || "",
        scroll_top: Number(rows?.scrollTop || 0),
        open: disclosure ? disclosure.open : true,
        focused: input === document.activeElement,
        selection_start: input?.selectionStart ?? null,
        selection_end: input?.selectionEnd ?? null,
      };
    }
    return state;
  };

  const restore_overview_state = state => {
    for (const id of panel_ids) {
      const saved = state[id];
      const panel = by_id(id);
      if (!saved || !panel) continue;
      const disclosure = panel.querySelector(".overview-collapsible");
      if (disclosure) disclosure.open = saved.open;
      const input = panel.querySelector('input[type="search"]');
      if (input) {
        input.value = saved.query;
        input.dispatchEvent(new Event("input"));
        if (saved.focused) {
          input.focus({preventScroll: true});
          if (saved.selection_start !== null && saved.selection_end !== null) {
            input.setSelectionRange(saved.selection_start, saved.selection_end);
          }
        }
      }
      const rows = panel.querySelector(".overview-key-rows");
      if (rows) rows.scrollTop = saved.scroll_top;
    }
  };

  const make_collapsible = (panel, fallback_title) => {
    if (!panel || panel.querySelector(":scope > .overview-collapsible")) return;
    const heading = panel.querySelector(":scope > .overview-panel-heading");
    const artifact_heading = panel.querySelector(":scope > h3");
    const title = String(
      heading?.querySelector("h3")?.textContent
      || artifact_heading?.textContent
      || fallback_title
    );
    const count = String(heading?.querySelector("span")?.textContent || "");
    heading?.remove();
    artifact_heading?.remove();

    const disclosure = document.createElement("details");
    disclosure.className = "overview-collapsible";
    disclosure.open = true;
    const summary = document.createElement("summary");
    summary.className = "overview-collapse-summary";
    const label = document.createElement("span");
    label.textContent = title;
    summary.appendChild(label);
    if (count) {
      const count_node = document.createElement("small");
      count_node.textContent = count;
      summary.appendChild(count_node);
    }
    const body = document.createElement("div");
    body.className = "overview-collapse-body";
    while (panel.firstChild) body.appendChild(panel.firstChild);
    disclosure.append(summary, body);
    panel.appendChild(disclosure);
  };

  const populate_historical_command_note = () => {
    const labels = document.querySelectorAll("#overview_metadata .overview-meta-label");
    for (const label of labels) {
      if (String(label.textContent || "").trim() !== "Command") continue;
      const value = label.nextElementSibling;
      if (value && String(value.textContent || "").trim() === "—") {
        value.textContent = "Not recorded (this run predates command capture)";
      }
      break;
    }
  };

  const enhance_overview = () => {
    const grid = document.querySelector("#run_overview_pane .overview-data-grid");
    const summary = by_id("overview_summary_panel");
    const config = by_id("overview_config_panel");
    if (grid && summary && config) {
      grid.dataset.instraStacked = "true";
      grid.append(summary, config);
    }
    make_collapsible(summary, "Summary");
    make_collapsible(config, "Config");
    make_collapsible(by_id("overview_artifact_outputs"), "Artifact Outputs");
    populate_historical_command_note();
  };

  if (typeof local_render_overview === "function") {
    const base_local_render_overview_aug30 = local_render_overview;
    local_render_overview = function() {
      const state = capture_overview_state();
      const result = base_local_render_overview_aug30();
      enhance_overview();
      restore_overview_state(state);
      requestAnimationFrame(() => restore_overview_state(state));
      return result;
    };
    if (!by_id("run_overview_pane")?.hidden) local_render_overview();
  }

  const name_width_storage_key = "thog2_local_run_name_column_width";
  const default_name_width = 390;
  const minimum_name_width = 180;
  const maximum_name_width = 1100;
  let table_patch_installed = false;
  let table_observer = null;
  let polish_scheduled = false;

  const stored_name_width = () => {
    const raw = localStorage.getItem(name_width_storage_key);
    if (raw === null) return default_name_width;
    const value = Number(raw);
    return Number.isFinite(value)
      ? Math.max(minimum_name_width, Math.min(maximum_name_width, Math.round(value)))
      : default_name_width;
  };

  const apply_name_width = width => {
    const table = document.querySelector(".runs-table");
    if (!table) return;
    const pixels = Math.max(minimum_name_width, Math.min(maximum_name_width, Math.round(width)));
    table.style.setProperty("--instra-run-name-width", `${pixels}px`);
    table.style.minWidth = `${1580 + pixels - default_name_width}px`;
  };

  const move_steps_after_preset = () => {
    const header_row = document.querySelector(".runs-table thead tr");
    const preset_header = header_row?.querySelector('[data-instra-run-shape-header="preset"]');
    const steps_header = header_row?.querySelector(".step-column");
    if (!preset_header || !steps_header) return false;
    if (preset_header.nextElementSibling !== steps_header) {
      preset_header.insertAdjacentElement("afterend", steps_header);
    }
    for (const row of document.querySelectorAll(".runs-table tbody tr[data-run-id]")) {
      const preset = row.querySelector('[data-instra-run-shape-cell="preset"]');
      let steps = row.querySelector('[data-instra-steps-cell="true"]');
      if (!steps) {
        steps = row.querySelector(".duration-column")?.previousElementSibling || null;
        if (steps) steps.dataset.instraStepsCell = "true";
      }
      if (preset && steps && preset.nextElementSibling !== steps) {
        preset.insertAdjacentElement("afterend", steps);
      }
    }
    return true;
  };

  const schedule_table_polish = () => {
    if (polish_scheduled) return;
    polish_scheduled = true;
    requestAnimationFrame(() => {
      polish_scheduled = false;
      move_steps_after_preset();
      apply_name_width(stored_name_width());
    });
  };

  const install_name_resizer = () => {
    const header = document.querySelector(".runs-table .name-column");
    if (!header || header.querySelector(".run-name-column-resizer")) return false;
    const handle = document.createElement("span");
    handle.className = "run-name-column-resizer";
    handle.setAttribute("role", "separator");
    handle.setAttribute("aria-orientation", "vertical");
    handle.setAttribute("aria-label", "Resize Run NAME column");
    handle.title = "Drag to resize Run NAME; double-click to reset";
    handle.addEventListener("pointerdown", event => {
      if (event.button !== 0) return;
      event.preventDefault();
      event.stopPropagation();
      const start_x = event.clientX;
      const start_width = header.getBoundingClientRect().width;
      handle.classList.add("dragging");
      handle.setPointerCapture?.(event.pointerId);
      const move = pointer_event => {
        const width = start_width + pointer_event.clientX - start_x;
        apply_name_width(width);
      };
      const finish = pointer_event => {
        handle.classList.remove("dragging");
        handle.releasePointerCapture?.(pointer_event.pointerId);
        window.removeEventListener("pointermove", move, true);
        window.removeEventListener("pointerup", finish, true);
        const table = document.querySelector(".runs-table");
        const width = Number.parseFloat(
          table?.style?.getPropertyValue("--instra-run-name-width") || String(default_name_width)
        );
        localStorage.setItem(name_width_storage_key, String(Math.round(width)));
      };
      window.addEventListener("pointermove", move, true);
      window.addEventListener("pointerup", finish, true);
    });
    handle.addEventListener("dblclick", event => {
      event.preventDefault();
      event.stopPropagation();
      localStorage.removeItem(name_width_storage_key);
      apply_name_width(default_name_width);
    });
    header.appendChild(handle);
    apply_name_width(stored_name_width());
    return true;
  };

  const install_table_patch = () => {
    if (table_patch_installed) return true;
    if (!move_steps_after_preset() || !install_name_resizer()) return false;
    table_patch_installed = true;
    const base_render_runs_aug30 = render_runs;
    render_runs = function() {
      const result = base_render_runs_aug30();
      schedule_table_polish();
      return result;
    };
    const table = document.querySelector(".runs-table");
    table_observer = new MutationObserver(schedule_table_polish);
    table_observer.observe(table, {childList: true, subtree: true});
    schedule_table_polish();
    return true;
  };

  let install_attempts = 0;
  const install_when_ready = () => {
    install_attempts += 1;
    if (install_table_patch() || install_attempts >= 120) return;
    setTimeout(install_when_ready, 25);
  };
  install_when_ready();

  const style = document.createElement("style");
  style.id = "thog2_aug30_enhancements_style";
  style.textContent = `
    .runs-table { --instra-run-name-width: 390px; }
    .runs-table .name-column {
      position: relative;
      width: var(--instra-run-name-width) !important;
      min-width: var(--instra-run-name-width) !important;
      max-width: var(--instra-run-name-width) !important;
    }
    .run-name-column-resizer {
      position: absolute;
      z-index: 4;
      top: 0;
      right: -4px;
      bottom: 0;
      width: 9px;
      cursor: col-resize;
      touch-action: none;
    }
    .run-name-column-resizer::after {
      content: "";
      position: absolute;
      top: 5px;
      right: 4px;
      bottom: 5px;
      width: 1px;
      background: #b8bec6;
    }
    .run-name-column-resizer:hover::after,
    .run-name-column-resizer.dragging::after {
      width: 2px;
      background: #1590a8;
    }
    #run_overview_pane .overview-data-grid[data-instra-stacked="true"] {
      grid-template-columns: minmax(0, 1fr) !important;
      gap: 16px !important;
    }
    #run_overview_pane .overview-collapsible {
      min-width: 0;
      border: 1px solid #d9dce1;
      border-radius: 4px;
      background: #fff;
    }
    #run_overview_pane .overview-collapse-summary {
      display: flex;
      align-items: baseline;
      gap: 8px;
      padding: 9px 11px;
      color: #4b535d;
      font-size: calc(var(--thog2-overview-font-size) + 2px);
      font-weight: 560;
      cursor: pointer;
      user-select: none;
    }
    #run_overview_pane .overview-collapse-summary small {
      color: #7b838e;
      font-size: max(7px, calc(var(--thog2-overview-font-size) - 1px));
      font-style: italic;
      font-weight: 400;
    }
    #run_overview_pane .overview-collapse-body { padding: 0 10px 10px; }
    #run_overview_pane .overview-artifact-outputs { margin-top: 16px; }
    #run_overview_pane .overview-artifact-outputs .overview-collapse-body {
      overflow-x: auto;
    }
  `;
  document.head.appendChild(style);

  window.__instra_aug30_enhancements = Object.freeze({
    enhance_overview,
    move_steps_after_preset,
    apply_name_width,
  });
});
// ^^^ THOG
