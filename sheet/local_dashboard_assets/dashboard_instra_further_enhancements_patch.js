// vvv THOG
"use strict";

// Final owner for the INSTRA_FURTHER_ENHANCEMENTS dashboard requirements.
window.addEventListener("load", () => {
  const weight_chart_names = Object.freeze([
    "attn_q_head_N",
    "attn_k_head_N",
    "attn_v_head_N",
    "attn_out_head_N",
    "mlp_up",
    "mlp_down",
  ]);
  const weight_chart_set = new Set(weight_chart_names);
  const additional_palette = Object.freeze([
    "#3B82F6", "#1D4ED8", "#0EA5E9", "#0369A1", "#06B6D4", "#0E7490", "#14B8A6", "#0F766E",
    "#10B981", "#047857", "#22C55E", "#15803D", "#84CC16", "#4D7C0F", "#A3E635", "#65A30D",
    "#EAB308", "#A16207", "#F59E0B", "#B45309", "#F97316", "#C2410C", "#EF4444", "#B91C1C",
    "#F43F5E", "#BE123C", "#EC4899", "#BE185D", "#D946EF", "#A21CAF", "#A855F7", "#7E22CE",
    "#8B5CF6", "#6D28D9", "#6366F1", "#4338CA", "#4F46E5", "#312E81", "#7C3AED", "#5B21B6",
    "#0D9488", "#115E59", "#0891B2", "#155E75", "#0284C7", "#075985", "#2563EB", "#1E40AF",
    "#92400E", "#78350F", "#A0522D", "#6B4423", "#708090", "#475569", "#334155", "#1E293B",
    "#FF6B6B", "#FF922B", "#FCC419", "#51CF66", "#20C997", "#845EF7", "#000000", "#FFFFFF",
  ]);

  const selected_run = () => {
    let run = null;
    try { run = typeof current_run === "function" ? current_run() : null; }
    catch (_error) { run = null; }
    const status = app.current_status && typeof app.current_status === "object"
      ? app.current_status
      : null;
    const identifier = String(app.current_run_id || "");
    if (!status || !identifier) return run || status;
    let status_identifier = "";
    try { status_identifier = String(run_identifier(status)); }
    catch (_error) {
      status_identifier = String(
        status.dashboard_run_id || status.local_run_id || status.wandb_run_id || status.run_name || ""
      );
    }
    return status_identifier === identifier ? {...(run || {}), ...status} : (run || status);
  };

  const install_palette = () => {
    const container = by_id("colour_swatches");
    if (!container || container.dataset.instraPalette128 === "true") return;
    for (const colour of additional_palette) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "colour-swatch";
      button.style.background = colour;
      button.title = colour;
      button.setAttribute("aria-label", `Choose run colour ${colour}`);
      button.addEventListener("click", () => set_picker_colour(hex_to_rgb(colour)));
      container.appendChild(button);
    }
    container.dataset.instraPalette128 = "true";
  };

  const base_open_colour_picker_further = open_colour_picker;
  open_colour_picker = function(run_id, anchor) {
    const result = base_open_colour_picker_further(run_id, anchor);
    requestAnimationFrame(() => {
      const popover = by_id("colour_popover");
      if (!popover || popover.hidden) return;
      popover.style.maxHeight = `${Math.max(260, window.innerHeight - 16)}px`;
      popover.style.overflowY = "auto";
      const anchor_rect = anchor?.getBoundingClientRect?.();
      if (anchor_rect) {
        const top = Math.max(
          8,
          Math.min(anchor_rect.bottom + 8, window.innerHeight - popover.offsetHeight - 8),
        );
        popover.style.top = `${Math.round(top)}px`;
      }
    });
    return result;
  };

  let wandb_column_index = null;
  const rename_run_name_header = header => {
    if (!header) return;
    const text_node = [...header.childNodes].find(node => node.nodeType === Node.TEXT_NODE);
    if (text_node && String(text_node.nodeValue || "").trim() !== "RUN NAME") {
      text_node.nodeValue = "RUN NAME ";
    }
    header.setAttribute("aria-label", "Run name; drag the right edge to resize");
  };

  const polish_run_table = () => {
    const table = document.querySelector(".runs-table");
    const header_row = table?.querySelector("thead tr");
    if (!table || !header_row) return;
    rename_run_name_header(header_row.querySelector(".name-column"));
    if (wandb_column_index === null) {
      wandb_column_index = [...header_row.children].findIndex(header => (
        String(header.textContent || "").trim() === "W&B ID"
      ));
    }
    if (wandb_column_index >= 0) {
      header_row.children[wandb_column_index]?.classList.add("instra-hidden-wandb-column");
      for (const row of table.querySelectorAll("tbody tr:not(.group-row)")) {
        row.children[wandb_column_index]?.classList.add("instra-hidden-wandb-column");
      }
    }
    const search = by_id("run_search");
    if (search) search.placeholder = "Search runs by name";
  };

  const find_meta = label_text => {
    for (const label of document.querySelectorAll("#overview_metadata .overview-meta-label")) {
      if (String(label.textContent || "").trim() === label_text) {
        return {label, value: label.nextElementSibling};
      }
    }
    return null;
  };

  const note_drafts = new Map();
  const note_save_timers = new Map();

  const update_cached_notes = (run_id, notes) => {
    if (String(app.current_run_id || "") === run_id && app.current_status) {
      app.current_status.notes = notes;
    }
    const run = app.runs?.find(candidate => {
      try { return String(run_identifier(candidate)) === run_id; }
      catch (_error) { return false; }
    });
    if (run) run.notes = notes;
  };

  const save_notes = async run_id => {
    const draft = note_drafts.get(run_id);
    if (!draft || !draft.dirty || draft.saving) return;
    draft.saving = true;
    const editor = document.querySelector(`textarea.overview-notes-editor[data-run-id="${CSS.escape(run_id)}"]`);
    const status = editor?.parentElement?.querySelector?.(".overview-notes-status");
    const button = editor?.parentElement?.querySelector?.(".overview-notes-save");
    if (status) status.textContent = "Saving…";
    if (button) button.disabled = true;
    const submitted = draft.value;
    try {
      const response = await fetch_json(
        `/api/run-notes?run=${encodeURIComponent(run_id)}`,
        {
          method: "PUT",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({notes: submitted}),
        },
      );
      update_cached_notes(run_id, String(response.notes || ""));
      if (draft.value === submitted) draft.dirty = false;
      if (status) status.textContent = "Saved";
    } catch (error) {
      if (status) status.textContent = `Save failed: ${error.message}`;
    } finally {
      draft.saving = false;
      if (button) button.disabled = false;
    }
  };

  const schedule_notes_save = run_id => {
    clearTimeout(note_save_timers.get(run_id));
    note_save_timers.set(run_id, setTimeout(() => save_notes(run_id), 900));
  };

  const install_notes_editor = run => {
    const row = find_meta("Notes");
    const run_id = String(app.current_run_id || "");
    if (!row?.value || !run_id) return;
    const stored = String(run?.notes ?? run?.configuration?.notes ?? run?.configuration?.note ?? "");
    if (!note_drafts.has(run_id)) {
      note_drafts.set(run_id, {value: stored, dirty: false, saving: false});
    }
    const draft = note_drafts.get(run_id);
    if (!draft.dirty && !draft.saving && draft.value !== stored) draft.value = stored;

    const shell = document.createElement("div");
    shell.className = "overview-notes-shell";
    const editor = document.createElement("textarea");
    editor.className = "overview-notes-editor";
    editor.dataset.runId = run_id;
    editor.maxLength = 4000;
    editor.rows = 2;
    editor.value = draft.value;
    editor.placeholder = "Add brief notes for this run";
    editor.addEventListener("input", () => {
      draft.value = editor.value;
      draft.dirty = true;
      const status = shell.querySelector(".overview-notes-status");
      if (status) status.textContent = "Unsaved";
      schedule_notes_save(run_id);
    });
    editor.addEventListener("blur", () => save_notes(run_id));
    const footer = document.createElement("div");
    footer.className = "overview-notes-footer";
    const status = document.createElement("span");
    status.className = "overview-notes-status";
    status.textContent = draft.dirty ? "Unsaved" : "Saved";
    const save = document.createElement("button");
    save.type = "button";
    save.className = "overview-notes-save";
    save.textContent = "Save";
    save.addEventListener("click", () => save_notes(run_id));
    footer.append(status, save);
    shell.append(editor, footer);
    row.value.replaceChildren(shell);
  };

  const add_wandb_overview_field = run => {
    const metadata = by_id("overview_metadata");
    if (!metadata || find_meta("W&B ID")) return;
    const label = document.createElement("div");
    label.className = "overview-meta-label";
    label.textContent = "W&B ID";
    const value = document.createElement("div");
    value.className = "overview-meta-value";
    value.textContent = String(run?.wandb_run_id || "—");
    const state = find_meta("State")?.label;
    metadata.insertBefore(label, state || null);
    metadata.insertBefore(value, state || null);
  };

  const polish_overview = () => {
    const run = selected_run();
    if (!run) return;
    const command = find_meta("Command") || find_meta("Runstring");
    if (command) {
      command.label.textContent = "Runstring";
      if (String(command.value?.textContent || "").trim() === "—") {
        command.value.textContent = "Not recorded (this run predates runstring capture)";
      }
    }
    const state = find_meta("State");
    if (state?.value && !state.value.querySelector(".state-badge") && typeof local_state_badge === "function") {
      state.value.replaceChildren(local_state_badge(run));
    }
    add_wandb_overview_field(run);
    install_notes_editor(run);
  };

  if (typeof local_render_overview === "function") {
    const base_local_render_overview_further = local_render_overview;
    local_render_overview = function() {
      const active = document.activeElement?.matches?.("textarea.overview-notes-editor")
        ? {
            run_id: document.activeElement.dataset.runId,
            start: document.activeElement.selectionStart,
            end: document.activeElement.selectionEnd,
          }
        : null;
      const result = base_local_render_overview_further();
      polish_overview();
      if (active) {
        const replacement = document.querySelector(
          `textarea.overview-notes-editor[data-run-id="${CSS.escape(active.run_id)}"]`
        );
        replacement?.focus?.({preventScroll: true});
        replacement?.setSelectionRange?.(active.start, active.end);
      }
      return result;
    };
  }

  const ensure_step_shortcuts = () => {
    const apply = by_id("weight_step_apply");
    const whole = by_id("weight_step_whole_range");
    if ((!apply && !whole) || by_id("weight_step_initial_values")) return;
    const initial = document.createElement("button");
    initial.id = "weight_step_initial_values";
    initial.type = "button";
    initial.className = "weight-step-button";
    initial.textContent = "initial values";
    initial.title = "Show the step 0 initial weight values";
    const step_one = document.createElement("button");
    step_one.id = "weight_step_one";
    step_one.type = "button";
    step_one.className = "weight-step-button";
    step_one.textContent = "step 1";
    step_one.title = "Show the step 1 weight values";
    if (apply) {
      apply.insertAdjacentElement("afterend", initial);
      initial.insertAdjacentElement("afterend", step_one);
    } else {
      whole.insertAdjacentElement("beforebegin", initial);
      initial.insertAdjacentElement("afterend", step_one);
    }
  };

  // Keep the circular draw order independent of table sorting and plot refreshes.
  const workspace_z_order = [];
  let workspace_front_run = null;
  const sync_workspace_z_order = () => {
    const visible_ids = (window.__instra_workspace?.visible_runs?.() || [])
      .map(run => String(run_identifier(run)));
    for (const identifier of visible_ids) {
      if (!workspace_z_order.includes(identifier)) workspace_z_order.push(identifier);
    }
    const active = workspace_z_order.filter(identifier => visible_ids.includes(identifier));
    if (!active.includes(workspace_front_run)) workspace_front_run = active.at(-1) || null;
    return active;
  };

  const order_workspace_weight_traces = prepared => {
    if (app.workspace_mode !== true || !prepared?.data) return;
    const active = sync_workspace_z_order();
    if (active.length < 2) return;
    const front_index = active.indexOf(workspace_front_run);
    const rotated = [...active.slice(front_index + 1), ...active.slice(0, front_index + 1)];
    const rank = new Map(rotated.map((identifier, index) => [identifier, index]));
    prepared.data.sort((left, right) => (
      (rank.get(String(left?.meta?.instra_workspace_run_id || "")) ?? -1)
      - (rank.get(String(right?.meta?.instra_workspace_run_id || "")) ?? -1)
    ));
  };

  const ensure_z_cycle = () => {
    const overlap = by_id("weight_step_overlapping_range");
    let button = by_id("weight_z_cycle");
    if (!button && overlap) {
      button = document.createElement("button");
      button.id = "weight_z_cycle";
      button.type = "button";
      button.className = "weight-step-button";
      button.textContent = "z";
      button.title = "Bring the next Workspace run to the front";
      button.setAttribute("aria-label", button.title);
      overlap.insertAdjacentElement("afterend", button);
      button.addEventListener("click", async () => {
        const active = sync_workspace_z_order();
        if (active.length < 2) return;
        workspace_front_run = active[(active.indexOf(workspace_front_run) + 1) % active.length];
        const jobs = [];
        for (const chart_name of weight_chart_names) {
          const mount = by_id(`${chart_name}_plot`);
          const figure = app.figures?.depth?.[chart_name];
          if (mount && figure) jobs.push(render_plot(mount, figure, chart_name));
        }
        await Promise.all(jobs);
      });
    }
    if (button) {
      button.hidden = app.workspace_mode !== true;
      button.disabled = sync_workspace_z_order().length < 2;
    }
  };

  const polish_weight_header = () => {
    ensure_step_shortcuts();
    ensure_z_cycle();
    if (app.workspace_mode !== true) return;
    const current = by_id("weight_step_current");
    if (current && current.textContent !== "") current.textContent = "";
    const availability = by_id("weight_step_availability");
    const available = window.__instra_weight_stability_final?.available_range?.();
    const text = available
      ? `overlapping steps ${available.minimum}–${available.maximum}`
      : "overlapping steps —";
    if (availability && availability.textContent !== text) availability.textContent = text;
  };

  window.addEventListener("click", event => {
    const initial = event.target?.closest?.("#weight_step_initial_values");
    const step_one = event.target?.closest?.("#weight_step_one");
    if (!initial && !step_one) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    const step = initial ? 0 : 1;
    window.__instra_clear_weight_step_input_drafts?.();
    window.__instra_weight_stability_final?.set_range?.(step, step);
    queueMicrotask(polish_weight_header);
  }, true);

  const polish_weight_titles = () => {
    chart_titles.mlp_up = "MLP - expansion";
    chart_titles.mlp_down = "MLP - contraction";
    for (const [chart_name, title] of [
      ["mlp_up", "MLP - expansion"],
      ["mlp_down", "MLP - contraction"],
    ]) {
      const heading = document.querySelector(
        `.chart-card[data-chart="${chart_name}"] .chart-heading-copy h2`
      );
      if (heading && heading.textContent !== title) heading.textContent = title;
    }
  };

  const polish_heading = () => {
    const subtitle = by_id("run_subtitle");
    for (const identity of subtitle?.querySelectorAll?.(".identity") || []) {
      if (String(identity.textContent || "").startsWith("W&B ID ")) identity.remove();
    }
    polish_run_table();
    polish_weight_header();
    polish_weight_titles();
  };

  const base_render_runs_further = render_runs;
  render_runs = function() {
    const result = base_render_runs_further();
    polish_run_table();
    return result;
  };

  const base_render_run_heading_further = render_run_heading;
  render_run_heading = function() {
    const result = base_render_run_heading_further();
    polish_heading();
    return result;
  };

  const schedule_maximized_restore = (chart_name, run_id) => {
    let attempts = 0;
    const restore = () => {
      attempts += 1;
      if (String(app.current_run_id || "") !== String(run_id || "")) return;
      if (app.maximized_chart) return;
      const card = document.querySelector(
        `.chart-card[data-chart="${CSS.escape(String(chart_name))}"]`
      );
      if (card) {
        toggle_maximized_chart(chart_name);
        return;
      }
      if (attempts < 50) setTimeout(restore, 100);
    };
    queueMicrotask(restore);
  };

  const base_select_run_further = select_run;
  select_run = function(run_id, options = {}) {
    const changing = String(run_id || "") !== String(app.current_run_id || "");
    const prior_maximized = changing ? app.maximized_chart : null;
    const result = base_select_run_further(run_id, options);
    if (prior_maximized) schedule_maximized_restore(prior_maximized, run_id);
    return result;
  };

  const install_final_weight_owner = () => {
    if (window.__instra_further_weight_owner) return true;
    const stability = window.__instra_weight_stability_final;
    if (!stability || !window.__instra_weight_range_interaction_final) return false;
    if (typeof prepare_figure !== "function" || typeof refresh_current_run !== "function") return false;

    const base_prepare_figure_further = prepare_figure;
    prepare_figure = function(figure, chart_name) {
      const prepared = base_prepare_figure_further(figure, chart_name);
      if (!weight_chart_set.has(chart_name)) return prepared;
      const width = stability.mode?.() === "latest" ? 2.4 : 1.0;
      for (const trace of prepared?.data || []) {
        if (trace?.meta?.instra_top_axis_anchor === true) continue;
        const mode = String(trace?.mode || "");
        if (mode.includes("lines") || trace.line) {
          trace.line = {...(trace.line || {}), width};
        }
      }
      order_workspace_weight_traces(prepared);
      return prepared;
    };

    let maxima_figures = null;
    let maxima_cache = new Map();
    const raw_depth_maxima = () => {
      if (app.figures?.depth === maxima_figures) return maxima_cache;
      const maxima = new Map();
      for (const figure of Object.values(app.figures?.depth || {})) {
        for (const trace of figure?.data || []) {
          if (trace?.meta?.instra_top_axis_anchor === true) continue;
          let update = null;
          try { update = Number(trace_optimizer_update(trace)); }
          catch (_error) { update = null; }
          if (!Number.isInteger(update) || update < 0) continue;
          const identifier = String(
            trace?.meta?.instra_workspace_run_id || app.current_run_id || ""
          );
          maxima.set(identifier, Math.max(maxima.get(identifier) ?? -1, update));
        }
      }
      maxima_figures = app.figures?.depth;
      maxima_cache = maxima;
      return maxima;
    };

    const live_payload_is_stale = () => {
      if (!["whole", "latest"].includes(String(stability.mode?.() || ""))) return false;
      if (app.refresh_in_flight) return false;
      const group = by_id("coefficients_chart_group");
      if (!group || group.classList.contains("collapsed")) return false;
      // An overlap/custom server window need not contain every run's final step.
      // Compare with its expected window, not the unfiltered run maximum.
      const range = stability.selected_range?.();
      const maxima = raw_depth_maxima();
      const runs = app.workspace_mode === true
        ? (window.__instra_workspace?.visible_runs?.() || [])
        : [selected_run()].filter(Boolean);
      for (const run of runs) {
        if (Number(run?.depth_snapshot_count || 0) <= 0) continue;
        let identifier = "";
        try { identifier = String(run_identifier(run)); }
        catch (_error) { identifier = String(app.current_run_id || ""); }
        const maximum = Number(run?.depth_maximum_update);
        if (range && Number(range.maximum) < maximum) continue;
        const expected = maximum;
        if (Number.isInteger(expected) && (maxima.get(identifier) ?? -1) < expected) return true;
      }
      return false;
    };

    const invalidate_live_depth = () => {
      window.__instra_workspace_depth_cache?.clear?.();
      const performance = window.__thog2_dashboard_performance?.state;
      if (performance) {
        performance.depth_signature = null;
        performance.pending_render = null;
        performance.deferred_coefficients = true;
      }
      app.figure_revision = null;
    };

    let catchup_in_flight = false;
    const base_refresh_current_run_further = refresh_current_run;
    refresh_current_run = async function() {
      const result = await base_refresh_current_run_further();
      if (!catchup_in_flight && live_payload_is_stale()) {
        catchup_in_flight = true;
        try {
          invalidate_live_depth();
          await base_refresh_current_run_further();
        } finally {
          catchup_in_flight = false;
        }
      }
      polish_weight_header();
      return result;
    };

    window.__instra_further_weight_owner = Object.freeze({
      live_payload_is_stale,
      raw_depth_maxima,
    });
    return true;
  };

  const style = document.createElement("style");
  style.id = "thog2_instra_further_enhancements_style";
  style.textContent = `
    .instra-hidden-wandb-column { display: none !important; }
    .state-badge.finished {
      color: #2f7d32 !important;
      background: #e8ebef !important;
    }
    .overview-notes-shell { min-width: 0; }
    .overview-notes-editor {
      width: 100%; min-height: 48px; box-sizing: border-box; resize: vertical;
      padding: 6px 7px; border: 1px solid #cbd1d8; border-radius: 4px;
      background: #fff; color: #303842; font: inherit; line-height: 1.35;
    }
    .overview-notes-editor:focus { border-color: #1590a8; outline: 1px solid #9ad9e3; }
    .overview-notes-footer {
      display: flex; align-items: center; justify-content: flex-end; gap: 8px; margin-top: 4px;
    }
    .overview-notes-status { color: #6b7480; font-size: 10px; }
    .overview-notes-save {
      height: 24px; padding: 0 9px; border: 1px solid #cbd1d8; border-radius: 4px;
      background: #f7f8fa; color: #46505b; font-size: 10px; cursor: pointer;
    }
    .overview-notes-save:disabled { opacity: .5; cursor: default; }
    .colour-swatch[title="#FFFFFF"] { border-color: #9ca3af; }
    #weight_step_initial_values { margin-left: 6px; }
    #weight_z_cycle { margin-left: 36px; }
    #coefficients_chart_group.thog2-tab-maximized-group > .chart-group-header #weights_group_settings_button {
      order: 999 !important; margin-left: auto !important; margin-right: 8px !important;
      display: inline-flex !important; visibility: visible !important;
    }
    #coefficients_chart_group.thog2-tab-maximized-group .chart-card.maximized .explicit-trajectory-modes,
    #coefficients_chart_group.thog2-tab-maximized-group .chart-card.maximized .chart-settings-button {
      display: none !important; visibility: hidden !important;
    }
  `;
  document.head.appendChild(style);

  install_palette();
  polish_run_table();
  polish_overview();
  polish_heading();

  let install_attempts = 0;
  const install_when_ready = () => {
    install_attempts += 1;
    if (install_final_weight_owner() || install_attempts >= 240) return;
    setTimeout(install_when_ready, 25);
  };
  setTimeout(install_when_ready, 450);

  let header_observer_attempts = 0;
  const install_header_observer = () => {
    header_observer_attempts += 1;
    const controls = by_id("weight_step_group_controls");
    if (!controls) {
      if (header_observer_attempts < 120) setTimeout(install_header_observer, 25);
      return;
    }
    const observer = new MutationObserver(polish_weight_header);
    observer.observe(controls, {childList: true, subtree: true, characterData: true});
    polish_weight_header();
  };
  install_header_observer();

  window.__instra_further_enhancements = Object.freeze({
    additional_palette,
    polish_overview,
    polish_run_table,
    polish_weight_header,
  });
});
// ^^^ THOG
