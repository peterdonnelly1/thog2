// vvv THOG
"use strict";

(function install_runs_trash_restore() {
  const style_id = "instra-runs-trash-restore-style";
  if (!document.getElementById(style_id)) {
    const style = document.createElement("style");
    style.id = style_id;
    style.textContent = `
      .runs-trash-button {
        width:30px;
        height:28px;
        padding:0;
        display:inline-flex;
        align-items:center;
        justify-content:center;
      }
      .runs-trash-button svg {
        width:15px;
        height:15px;
        fill:none;
        stroke:currentColor;
        stroke-width:1.7;
        stroke-linecap:round;
        stroke-linejoin:round;
      }
      .runs-trash-button:not(:disabled) { color:#a12a2a; }
      .runs-trash-button:disabled { opacity:.38; cursor:default; }
    `;
    document.head.appendChild(style);
  }

  function selected_existing_runs() {
    const ids = new Set((app.runs || []).map(run => String(run_identifier(run))));
    return [...(app.selected || new Set())].filter(run_id => ids.has(String(run_id)));
  }

  function update_trash_button() {
    const button = by_id("delete_selected_runs");
    if (!button) return;
    const selected = selected_existing_runs();
    button.disabled = selected.length === 0;
    button.title = selected.length
      ? `Delete instra chart data for ${selected.length} selected run${selected.length === 1 ? "" : "s"}`
      : "Select runs with the checkboxes to delete their instra chart data";
    button.setAttribute("aria-label", button.title);
  }

  async function delete_selected_runs() {
    const selected = selected_existing_runs();
    if (!selected.length) return;
    const names = selected
      .map(run_id => (app.runs || []).find(run => String(run_identifier(run)) === String(run_id))?.artifact_name || run_id)
      .slice(0, 5);
    const extra = selected.length > names.length ? `\n…and ${selected.length - names.length} more` : "";
    const confirmed = window.confirm(
      `Delete instra chart data for ${selected.length} selected run${selected.length === 1 ? "" : "s"}?\n\n`
      + `${names.join("\n")}${extra}\n\nThis does not delete checkpoints, other logs, or W&B runs.`
    );
    if (!confirmed) return;

    const button = by_id("delete_selected_runs");
    if (button) button.disabled = true;
    let deleted = 0;
    const failures = [];
    for (const run_id of selected) {
      try {
        await fetch_json(`/api/run?run=${encodeURIComponent(run_id)}`, {method:"DELETE"});
        deleted += 1;
        app.selected.delete(run_id);
        delete app.colours[run_id];
        delete app.visibility[run_id];
        app.processing_paired_run_ids?.delete?.(run_id);
        app.processing_auto_opened_run_ids?.delete?.(run_id);
      } catch (error) {
        failures.push(`${run_id}: ${error.message}`);
      }
    }
    save_json("thog2_local_run_colours", app.colours);
    save_json("thog2_local_run_visibility", app.visibility);
    await refresh_catalog();
    update_trash_button();
    if (failures.length) {
      show_toast(`Deleted ${deleted}; ${failures.length} failed.`);
    } else {
      show_toast(`Deleted instra chart data for ${deleted} run${deleted === 1 ? "" : "s"}.`);
    }
  }

  function ensure_trash_button() {
    const toolbar = document.querySelector(".runs-pane-header .toolbar");
    if (!toolbar) return null;
    let button = by_id("delete_selected_runs");
    if (!button) {
      button = document.createElement("button");
      button.id = "delete_selected_runs";
      button.type = "button";
      button.className = "toolbar-button runs-trash-button";
      button.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16"/><path d="M9 7V4h6v3"/><path d="M6.5 7l1 13h9l1-13"/><path d="M10 11v5M14 11v5"/></svg>';
      button.addEventListener("click", delete_selected_runs);
      const group = by_id("group_button");
      if (group) group.insertAdjacentElement("afterend", button);
      else toolbar.appendChild(button);
    }
    update_trash_button();
    return button;
  }

  const render_runs_before_trash_restore = render_runs;
  render_runs = function() {
    const result = render_runs_before_trash_restore();
    ensure_trash_button();
    update_trash_button();
    return result;
  };

  document.addEventListener("change", event => {
    if (event.target.matches?.('.runs-table input[type="checkbox"]')) {
      queueMicrotask(update_trash_button);
    }
  }, true);

  window.addEventListener("load", () => {
    ensure_trash_button();
    update_trash_button();
  });
})();
// ^^^ THOG
