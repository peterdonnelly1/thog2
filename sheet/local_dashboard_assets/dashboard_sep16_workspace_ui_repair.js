// vvv THOG
"use strict";

/* Restore legacy global-eye semantics without changing the Runs table itself. */
(function install_sep16_workspace_ui_repair() {
  const style_id = "instra-sep16-workspace-ui-repair-style";
  if (!document.getElementById(style_id)) {
    const style = document.createElement("style");
    style.id = style_id;
    style.textContent = `
      .runs-table thead .visibility-column .eye-button {
        display:inline-flex;
        align-items:center;
        justify-content:center;
        margin:auto;
      }
      .processing-throughput-z-button {
        display:inline-flex !important;
        align-items:center !important;
        justify-content:center !important;
        text-align:center !important;
        font-family:inherit !important;
        font-size:inherit !important;
        font-weight:400 !important;
        font-style:normal !important;
        line-height:1 !important;
        padding:0 !important;
      }
    `;
    document.head.appendChild(style);
  }

  function all_runs_visible() {
    return Array.isArray(app?.runs)
      && app.runs.length > 0
      && app.runs.every(run => is_visible(run_identifier(run)));
  }

  function sync_global_eye() {
    const cell = document.querySelector(".runs-table thead .visibility-column");
    if (!cell) return;

    let button = cell.querySelector("button.visibility-global-eye");
    if (!button) {
      cell.replaceChildren();
      button = document.createElement("button");
      button.type = "button";
      button.className = "eye-button visibility-global-eye";
      button.addEventListener("click", event => {
        event.preventDefault();
        event.stopPropagation();
        const make_visible = !all_runs_visible();
        for (const run of app.runs || []) {
          app.visibility[run_identifier(run)] = make_visible;
        }
        save_json("thog2_local_run_visibility", app.visibility);
        render_runs();
      });
      cell.appendChild(button);
    }

    const open = all_runs_visible();
    button.classList.toggle("off", !open);
    button.replaceChildren(icon_svg(open ? "eye_open" : "eye_closed"));
    button.title = open ? "Hide all runs from Workspace" : "Show all runs in Workspace";
    button.setAttribute("aria-label", button.title);
    button.setAttribute("aria-pressed", open ? "true" : "false");
  }

  window.addEventListener("load", () => {
    const base_render_runs_ui = render_runs;
    render_runs = function() {
      const result = base_render_runs_ui();
      sync_global_eye();
      return result;
    };
    sync_global_eye();
  });
})();
// ^^^ THOG
