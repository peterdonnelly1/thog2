// vvv THOG
"use strict";

/* Final Workspace/UI repair: restore the legacy global eye and normalize Z controls. */
(function install_sep16_workspace_ui_repair() {
  const style_id = "instra-sep16-workspace-ui-repair-style";
  if (!document.getElementById(style_id)) {
    const style = document.createElement("style");
    style.id = style_id;
    style.textContent = `
      .visibility-global-eye {
        appearance: none;
        border: 0;
        background: transparent;
        color: inherit;
        width: 100%;
        height: 100%;
        min-width: 22px;
        min-height: 22px;
        padding: 0;
        margin: 0;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
      }
      .visibility-global-eye svg {
        width: 18px;
        height: 15px;
        display: block;
      }
      .instra-z-icon-normalized {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        font-family: inherit !important;
        font-size: inherit !important;
        font-weight: 400 !important;
        font-style: normal !important;
        line-height: 1 !important;
      }
    `;
    document.head.appendChild(style);
  }

  function all_runs_visible() {
    return Array.isArray(app?.runs)
      && app.runs.length > 0
      && app.runs.every(run => is_visible(run_identifier(run)));
  }

  function install_global_eye() {
    const cell = document.querySelector(".runs-table thead .visibility-column");
    if (!cell) return;

    let button = cell.querySelector("button.visibility-global-eye");
    if (!button) {
      cell.replaceChildren();
      button = document.createElement("button");
      button.type = "button";
      button.className = "visibility-global-eye";
      button.addEventListener("click", event => {
        event.preventDefault();
        event.stopPropagation();
        const make_visible = !all_runs_visible();
        for (const run of app.runs || []) {
          app.visibility[run_identifier(run)] = make_visible;
        }
        save_json("thog2_local_run_visibility", app.visibility);
        render_runs();
        refresh_ui();
      });
      cell.appendChild(button);
    }

    const state = all_runs_visible() ? "open" : "closed";
    if (button.dataset.state !== state) {
      button.dataset.state = state;
      button.replaceChildren(icon_svg(state === "open" ? "eye_open" : "eye_closed"));
    }
    button.title = state === "open" ? "Hide all runs from Workspace" : "Show all runs in Workspace";
    button.setAttribute("aria-label", button.title);
    button.setAttribute("aria-pressed", state === "open" ? "true" : "false");
  }

  function normalize_z_controls() {
    for (const button of document.querySelectorAll("button")) {
      if (String(button.textContent || "").trim() !== "Z") continue;
      button.classList.add("instra-z-icon-normalized");
    }
  }

  function refresh_ui() {
    install_global_eye();
    normalize_z_controls();
  }

  window.addEventListener("load", () => {
    setTimeout(refresh_ui, 0);
    const observer = new MutationObserver(refresh_ui);
    observer.observe(document.body, {childList: true, subtree: true});
  });
})();
// ^^^ THOG
