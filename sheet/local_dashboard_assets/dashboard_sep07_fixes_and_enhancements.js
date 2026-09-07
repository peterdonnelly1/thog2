// vvv THOG
"use strict";

// Final owner for September Instra startup, chart, colour, memory and Runs-table behaviour.
window.addEventListener("load", () => {
  setTimeout(() => {
    const ordinary_text_storage_key = "thog2_local_ordinary_text_size";
    const updated_text_by_run = new Map();
    let table_polish_scheduled = false;
    let workspace_hover_run_id = null;

    const finite_number = value => {
      if (value === null || value === undefined || value === "") return null;
      const numeric = Number(value);
      return Number.isFinite(numeric) ? numeric : null;
    };
    const clamp_integer = (value, minimum, maximum) => Math.max(
      minimum,
      Math.min(maximum, Math.round(Number(value))),
    );
    const ordinary_text_size = () => {
      const stored = finite_number(localStorage.getItem(ordinary_text_storage_key));
      return stored === null ? 12 : clamp_integer(stored, 8, 20);
    };
    const apply_ordinary_text_size = value => {
      const size = clamp_integer(value, 8, 20);
      document.documentElement.style.setProperty("--instra-ordinary-text-size", `${size}px`);
      for (const mount of document.querySelectorAll(".plot-mount.js-plotly-plot")) {
        if (!mount.layout) continue;
        const update = {};
        for (const axis_name of Object.keys(mount.layout).filter(name => /^(xaxis|yaxis)\d*$/.test(name))) {
          update[`${axis_name}.tickfont.size`] = size;
          update[`${axis_name}.title.font.size`] = size;
        }
        if (Object.keys(update).length) Plotly.relayout(mount, update);
      }
    };

    const install_ordinary_text_setting = () => {
      if (by_id("ordinary_text_size")) return;
      const content = document.querySelector(".settings-dialog .settings-content");
      if (!content) return;
      const block = document.createElement("div");
      block.className = "ordinary-text-setting";
      const label = document.createElement("label");
      label.htmlFor = "ordinary_text_size";
      label.textContent = "Ordinary text size";
      const control = document.createElement("div");
      control.className = "ordinary-text-control";
      const input = document.createElement("input");
      input.id = "ordinary_text_size";
      input.type = "number";
      input.min = "8";
      input.max = "20";
      input.step = "1";
      input.value = String(ordinary_text_size());
      const unit = document.createElement("span");
      unit.textContent = "px";
      control.append(input, unit);
      block.append(label, control);
      content.appendChild(block);
    };
    install_ordinary_text_setting();
    apply_ordinary_text_size(ordinary_text_size());

    const base_open_settings_sep07 = open_settings;
    open_settings = function() {
      install_ordinary_text_setting();
      const result = base_open_settings_sep07();
      const input = by_id("ordinary_text_size");
      if (input) input.value = String(ordinary_text_size());
      return result;
    };
    const base_save_settings_sep07 = save_settings;
    save_settings = function() {
      const desired = finite_number(by_id("ordinary_text_size")?.value);
      if (desired === null || desired < 8 || desired > 20) {
        show_toast("Ordinary text size must be between 8 and 20 px.");
        return;
      }
      localStorage.setItem(ordinary_text_storage_key, String(Math.round(desired)));
      apply_ordinary_text_size(desired);
      return base_save_settings_sep07();
    };

    const metric_identity = chart_name => {
      const card = document.querySelector(
        `.local-metric-card[data-chart="${CSS.escape(String(chart_name))}"]`
      );
      return String(card?.dataset?.metricChartId || "");
    };
    const axis_title = axis => (
      typeof axis?.title === "string" ? axis.title : String(axis?.title?.text || "value")
    );
    const base_prepare_figure_sep07 = prepare_figure;
    prepare_figure = function(figure, chart_name) {
      const prepared = base_prepare_figure_sep07(figure, chart_name);
      prepared.layout = prepared.layout || {};
      const text_size = ordinary_text_size();
      for (const axis_name of Object.keys(prepared.layout).filter(name => /^(xaxis|yaxis)\d*$/.test(name))) {
        const axis = prepared.layout[axis_name] || {};
        const title = typeof axis.title === "string" ? {text: axis.title} : {...(axis.title || {})};
        prepared.layout[axis_name] = {
          ...axis,
          tickfont: {...(axis.tickfont || {}), size: text_size},
          title: {...title, font: {...(title.font || {}), size: text_size}},
        };
      }
      for (const trace of prepared.data || []) {
        if (!String(trace?.type || "scatter").startsWith("scatter")) continue;
        trace.line = {...(trace.line || {}), dash: "solid"};
        if (app.workspace_mode === true && trace.meta?.instra_workspace_run_id) {
          trace.line.width = String(trace.meta.instra_workspace_run_id) === String(workspace_hover_run_id || "") ? 3 : 1.25;
        } else if (["train/loss", "val/val_loss"].includes(metric_identity(chart_name))) {
          trace.line.width = 1.25;
        }
      }
      const metric_name = metric_identity(chart_name);
      if (["train/loss", "val/val_loss"].includes(metric_name)) {
        const x_name = axis_title(prepared.layout.xaxis);
        for (const trace of prepared.data || []) {
          trace.hovertemplate = `<b>%{fullData.name}</b><br>${metric_name}: %{y:.6g}<br>${x_name}: %{x}<extra></extra>`;
        }
        prepared.layout.hoverlabel = {
          ...(prepared.layout.hoverlabel || {}),
          bgcolor: "#222831",
          bordercolor: "#222831",
          font: {...(prepared.layout.hoverlabel?.font || {}), color: "#ffffff"},
          align: "left",
          namelength: -1,
        };
      }
      return prepared;
    };

    const install_colour_inputs = () => {
      const hex = by_id("colour_hex");
      const channels = [by_id("colour_r"), by_id("colour_g"), by_id("colour_b")];
      if (!hex || channels.some(input => !input) || hex.dataset.instraEditable === "true") return;
      hex.dataset.instraEditable = "true";
      hex.addEventListener("input", () => {
        const rgb = hex_to_rgb(hex.value);
        if (rgb) set_picker_colour(rgb);
      });
      hex.addEventListener("keydown", event => {
        if (event.key === "Enter") {
          event.preventDefault();
          const rgb = hex_to_rgb(hex.value);
          if (rgb) set_picker_colour(rgb);
        }
      });
      const update_rgb = event => {
        const values = channels.map(input => finite_number(input.value));
        if (values.some(value => value === null)) return;
        if (event.type === "input" && values.some(value => value < 0 || value > 255)) return;
        set_picker_colour(values.map(value => clamp_integer(value, 0, 255)));
      };
      for (const input of channels) {
        input.readOnly = false;
        input.type = "number";
        input.min = "0";
        input.max = "255";
        input.step = "1";
        input.addEventListener("input", update_rgb);
        input.addEventListener("change", update_rgb);
      }
    };
    install_colour_inputs();

    const configuration = run => (
      run?.configuration && typeof run.configuration === "object" ? run.configuration : {}
    );
    const configured_value = (run, ...names) => {
      const config = configuration(run);
      for (const name of names) {
        if (run?.[name] !== undefined && run[name] !== null) return run[name];
        if (config?.[name] !== undefined && config[name] !== null) return config[name];
      }
      return null;
    };
    const parameter_report = run => {
      const report = configuration(run).parameter_report;
      return report && typeof report === "object" ? report : {};
    };
    const million_parameters = (run, dense_equivalent) => {
      const report = parameter_report(run);
      const raw = dense_equivalent
        ? (report.dense_equivalent_total_parameters ?? report.dense_equivalent_parameters)
        : (report.persistent_parameters ?? report.trainable_parameters);
      const value = finite_number(raw);
      return value === null ? "—" : String(Math.round(value / 1_000_000));
    };
    const gpu_value = run => {
      const explicit = configured_value(run, "gpu_index");
      if (explicit !== null && explicit !== undefined && explicit !== "") return explicit;
      const config = configuration(run);
      const source = [
        config.cuda_visible_devices,
        config.command,
        run?.command,
        config.host_label,
        run?.host_label,
      ].filter(Boolean).join(" ");
      const environment_match = source.match(/CUDA_VISIBLE_DEVICES\s*=\s*["']?(\d+)/i);
      if (environment_match) return environment_match[1];
      const host_match = source.match(/(?:^|[_ -])gpu[_ -]?(\d+)(?:$|[_ -])/i);
      if (host_match) return host_match[1];
      return /(?:^|[_ -])scruffy(?:$|[_ -])/i.test(source) ? 0 : null;                                                                                     // <<< THOG the current single-GPU scruffy host is physical GPU 0
    };
    const table_definitions = Object.freeze({
      gpu: {label: "GPU", title: "physical GPU selected for this run", value: gpu_value},
      gb: {label: "GB", title: "peak process GPU memory allocated so far (GiB)", value: run => configured_value(run, "gpu_peak_memory_allocated_gb")},
      parms: {label: "PARMS", title: "persistent model parameters (millions, rounded)", value: run => million_parameters(run, false)},
      equiv: {
        label: "EQUIV", title: "dense-equivalent parameters (millions, THOG only)",
        value: run => String(configured_value(run, "model_type") || "").toLowerCase() === "dense" ? "—" : million_parameters(run, true),
      },
      capture_period: {
        label: "C_p", title: "weight-curve capture period in optimizer steps",
        value: run => configured_value(run, "instrumentation__depth_weight_curves__log_every_n_steps")
          ?? configuration(run).instrumentation_configuration?.THOG2_INSTRUMENTATION_DEPTH_WEIGHT_CURVES_LOG_EVERY_N_STEPS,
      },
    });
    const display_cell_value = value => {
      if (value === null || value === undefined || value === "") return "—";
      const numeric = finite_number(value);
      return numeric === null ? String(value) : String(Math.round(numeric * 10) / 10);
    };
    const ensure_header = (key, marker, placement = "afterend") => {
      const header_row = document.querySelector(".runs-table thead tr");
      if (!header_row || !marker) return null;
      let header = header_row.querySelector(`[data-instra-sep07-header="${key}"]`);
      if (!header) {
        const definition = table_definitions[key];
        header = document.createElement("th");
        header.dataset.instraSep07Header = key;
        header.className = "numeric-column instra-sep07-column";
        header.textContent = definition.label;
        header.title = definition.title;
      }
      if (placement !== "afterend" || marker.nextElementSibling !== header) marker.insertAdjacentElement(placement, header);
      return header;
    };
    const ensure_row_cell = (row, run, key, marker, placement = "afterend") => {
      if (!marker) return null;
      let cell = row.querySelector(`[data-instra-sep07-cell="${key}"]`);
      if (!cell) {
        cell = document.createElement("td");
        cell.dataset.instraSep07Cell = key;
        cell.className = "numeric-column instra-sep07-column";
      }
      const definition = table_definitions[key];
      const shown = display_cell_value(definition.value(run));
      if (cell.textContent !== shown) cell.textContent = shown;
      cell.title = `${definition.title}: ${shown}`;
      if (placement !== "afterend" || marker.nextElementSibling !== cell) marker.insertAdjacentElement(placement, cell);
      return cell;
    };

    const polish_run_table = () => {
      const table = document.querySelector(".runs-table");
      const header_row = table?.querySelector("thead tr");
      if (!table || !header_row) return;
      const state_header = [...header_row.children].find(cell => cell.textContent.trim().toUpperCase() === "STATE");
      const name_header = header_row.querySelector(".name-column");
      const duration_header = header_row.querySelector(".duration-column");
      const host_header = [...header_row.children].find(cell => cell.textContent.trim().toUpperCase() === "HOST");
      const steps_header = header_row.querySelector(".step-column");
      const depth_order_header = header_row.querySelector('[data-instra-run-shape-header="depth_order"]');
      const curve_end_header = header_row.querySelector(".curve-end-column");
      if (state_header && duration_header && name_header) {
        state_header.insertAdjacentElement("afterend", duration_header);
        duration_header.insertAdjacentElement("afterend", name_header);
      }
      const gpu_header = ensure_header("gpu", host_header);
      ensure_header("gb", steps_header);
      const parms_header = ensure_header("parms", depth_order_header);
      ensure_header("equiv", parms_header);
      ensure_header("capture_period", curve_end_header);

      for (const row of table.querySelectorAll("tbody tr[data-run-id]")) {
        const run = app.runs.find(candidate => String(run_identifier(candidate)) === String(row.dataset.runId));
        if (!run) continue;
        const state_cell = row.querySelector(".state-badge")?.parentElement;
        const name_cell = row.querySelector(".run-link")?.parentElement;
        const duration_cell = row.querySelector(".duration-column");
        const preset_cell = row.querySelector('[data-instra-run-shape-cell="preset"]');
        let host_cell = row.querySelector('[data-instra-sep07-anchor="host"]');
        if (!host_cell) {
          host_cell = preset_cell?.previousElementSibling;
          if (host_cell) host_cell.dataset.instraSep07Anchor = "host";
        }
        const steps_cell = row.querySelector('[data-instra-steps-cell="true"]');
        const depth_order_cell = row.querySelector('[data-instra-run-shape-cell="depth_order"]');
        const menu_cell = row.querySelector(".menu-column");
        const updated_cell = menu_cell?.previousElementSibling;
        let curve_end_cell = row.querySelector('[data-instra-sep07-anchor="curve_end"]');
        if (!curve_end_cell) {
          curve_end_cell = duration_cell?.previousElementSibling;
          if (curve_end_cell) curve_end_cell.dataset.instraSep07Anchor = "curve_end";
        }
        if (state_cell && duration_cell && name_cell) {
          state_cell.insertAdjacentElement("afterend", duration_cell);
          duration_cell.insertAdjacentElement("afterend", name_cell);
        }
        ensure_row_cell(row, run, "gpu", host_cell);
        ensure_row_cell(row, run, "gb", steps_cell);
        const parms_cell = ensure_row_cell(row, run, "parms", depth_order_cell);
        ensure_row_cell(row, run, "equiv", parms_cell);
        ensure_row_cell(row, run, "capture_period", curve_end_cell);
        if (updated_cell) {
          updated_cell.classList.add("instra-updated-cell");
          const next_text = String(updated_cell.textContent || "").trim().toLowerCase();
          const previous_text = updated_text_by_run.get(row.dataset.runId);
          updated_text_by_run.set(row.dataset.runId, next_text);
          if (next_text === "just now" && previous_text !== "just now") {
            updated_cell.classList.add("instra-just-now-flash");
            setTimeout(() => updated_cell.classList.remove("instra-just-now-flash"), 500);
          }
        }
      }
      const column_count = header_row.children.length;
      table.querySelectorAll("tbody .group-row td").forEach(cell => { cell.colSpan = column_count; });
      void gpu_header;
    };
    const schedule_table_polish = () => {
      if (table_polish_scheduled) return;
      table_polish_scheduled = true;
      requestAnimationFrame(() => {
        table_polish_scheduled = false;
        polish_run_table();
      });
    };
    const base_render_runs_sep07 = render_runs;
    render_runs = function() {
      const result = base_render_runs_sep07();
      schedule_table_polish();
      return result;
    };
    const runs_table = document.querySelector(".runs-table");
    if (runs_table) new MutationObserver(schedule_table_polish).observe(runs_table, {childList: true, subtree: true});
    schedule_table_polish();

    const trace_run_id = trace => String(trace?.meta?.instra_workspace_run_id || "");
    const restyle_workspace_curves = highlighted_run_id => {
      workspace_hover_run_id = highlighted_run_id || null;
      if (app.workspace_mode !== true) return;
      for (const mount of document.querySelectorAll(".plot-mount.js-plotly-plot")) {
        const indices = [];
        const widths = [];
        for (let index = 0; index < (mount.data || []).length; index += 1) {
          const run_id = trace_run_id(mount.data[index]);
          if (!run_id || !String(mount.data[index]?.type || "scatter").startsWith("scatter")) continue;
          indices.push(index);
          widths.push(run_id === String(workspace_hover_run_id || "") ? 3 : 1.25);
        }
        if (indices.length) Plotly.restyle(mount, {"line.width": widths, "line.dash": indices.map(() => "solid")}, indices);
      }
    };
    const clear_curve_row_highlight = () => {
      document.querySelectorAll("#runs_body tr.instra-curve-hover").forEach(row => row.classList.remove("instra-curve-hover"));
    };
    const highlight_curve_row = run_id => {
      clear_curve_row_highlight();
      if (!run_id || app.workspace_mode !== true) return;
      document.querySelector(`#runs_body tr[data-run-id="${CSS.escape(String(run_id))}"]`)?.classList.add("instra-curve-hover");
    };
    by_id("runs_body")?.addEventListener("pointerover", event => {
      const row = event.target.closest("tr[data-run-id]");
      if (!row || row.contains(event.relatedTarget)) return;
      restyle_workspace_curves(row.dataset.runId);
    });
    by_id("runs_body")?.addEventListener("pointerout", event => {
      const row = event.target.closest("tr[data-run-id]");
      if (!row || row.contains(event.relatedTarget)) return;
      restyle_workspace_curves(null);
    });
    const bind_plot_hover = mount => {
      if (!mount?.on || mount.dataset.instraRunHover === "true") return;
      mount.dataset.instraRunHover = "true";
      mount.on("plotly_hover", event => highlight_curve_row(trace_run_id(event?.points?.[0]?.data)));
      mount.on("plotly_unhover", clear_curve_row_highlight);
    };
    const base_render_plot_sep07 = render_plot;
    render_plot = async function(mount, figure, chart_name) {
      const result = await base_render_plot_sep07(mount, figure, chart_name);
      bind_plot_hover(mount);
      if (app.workspace_mode === true) restyle_workspace_curves(workspace_hover_run_id);
      return result;
    };
    document.querySelectorAll(".plot-mount.js-plotly-plot").forEach(bind_plot_hover);

    const align_chart_actions = root => {
      for (const card of root.querySelectorAll?.(".chart-card") || []) {
        const header = card.querySelector(":scope > .chart-card-header");
        const maximize = header?.querySelector(".maximize-button");
        if (!header || !maximize) continue;
        let actions = header.querySelector(":scope > .chart-card-actions");
        if (!actions) {
          actions = document.createElement("div");
          actions.className = "chart-card-actions";
          header.appendChild(actions);
        }
        for (const control of [...header.children]) {
          if (control === actions || control.classList.contains("chart-heading-copy")) continue;
          if (control.matches("button, .explicit-trajectory-modes, .weight-step-controls")) actions.appendChild(control);
        }
        if (actions.lastElementChild !== maximize) actions.appendChild(maximize);
      }
    };
    align_chart_actions(document);
    const chart_root = by_id("charts_scroll");
    if (chart_root) new MutationObserver(() => align_chart_actions(chart_root)).observe(chart_root, {childList: true, subtree: true});

    const sync_navigation_selection = () => {
      const settings_open = by_id("settings_overlay")?.hidden === false;
      const workspace_open = app.workspace_mode === true && !settings_open;
      by_id("settings_nav")?.classList.toggle("selected", settings_open);
      by_id("workspace_nav")?.classList.toggle("selected", workspace_open);
      by_id("runs_nav")?.classList.toggle("selected", !settings_open && !workspace_open);
    };
    for (const id of ["workspace_nav", "runs_nav", "settings_nav", "close_settings", "cancel_settings", "save_settings"]) {
      by_id(id)?.addEventListener("click", () => queueMicrotask(sync_navigation_selection));
    }
    const nav_observer = new MutationObserver(sync_navigation_selection);
    for (const id of ["workspace_nav", "runs_nav", "settings_nav", "settings_overlay"]) {
      const node = by_id(id);
      if (node) nav_observer.observe(node, {attributes: true, attributeFilter: ["class", "hidden"]});
    }

    const set_group_collapsed = (section, collapsed) => {
      if (!section) return;
      section.classList.toggle("collapsed", collapsed);
      const toggle = section.querySelector(":scope > .chart-group-header .chart-group-toggle");
      const controls = toggle?.getAttribute("aria-controls");
      const grid = controls ? by_id(controls) : section.querySelector(":scope > .chart-grid");
      if (grid) grid.hidden = collapsed;
      if (toggle) toggle.setAttribute("aria-expanded", String(!collapsed));
    };
    const startup_collapsed = {
      train: false,
      val: true,
      memory: true,
      system: true,
      depth: true,
      heatmap: true,
      coefficients: true,
      optimizer_momentum: true,
      optimizer_scaling: true,
    };
    save_json("thog2_local_metric_group_collapsed_v2", {
      runs: {train: false, val: true, memory: true, system: true},
      workspace: {train: false, val: true, memory: true, system: true},
    });
    let selected_latest = false;
    let requested_workspace = false;
    let startup_done = false;
    const apply_startup_policy = () => {
      if (startup_done) return;
      if (!selected_latest && app.runs?.length) {
        const latest = [...app.runs].sort((left, right) => (
          String(right.created_at || "").localeCompare(String(left.created_at || ""))
          || String(run_identifier(right)).localeCompare(String(run_identifier(left)))
        ))[0];
        if (latest) {
          selected_latest = true;
          if (String(app.current_run_id || "") !== String(run_identifier(latest))) {
            select_run(run_identifier(latest), {manual: false, replace_history: true});
          }
        }
      }
      if (selected_latest && !requested_workspace) {
        requested_workspace = true;
        by_id("workspace_nav")?.click();
      }
      for (const section of document.querySelectorAll(".chart-group")) {
        const name = String(section.dataset.metricGroup || section.dataset.chartGroup || "");
        if (Object.hasOwn(startup_collapsed, name) && section.dataset.instraStartupApplied !== "true") {
          set_group_collapsed(section, startup_collapsed[name]);
          section.dataset.instraStartupApplied = "true";
        }
      }
      const loss_card = document.querySelector('.local-metric-group[data-metric-group="train"] .local-metric-card[data-metric-chart-id="train/loss"]');
      if (app.workspace_mode === true && loss_card && !app.maximized_chart) {
        toggle_maximized_chart(loss_card.dataset.chart);
      }
      const optimizer_groups = document.querySelectorAll('.thogopt-group[data-instra-startup-applied="true"]');
      if (app.workspace_mode === true && loss_card && app.maximized_chart === loss_card.dataset.chart && optimizer_groups.length === 2) {
        startup_done = true;
        startup_observer.disconnect();
      }
      sync_navigation_selection();
    };
    const startup_observer = new MutationObserver(apply_startup_policy);
    if (chart_root) startup_observer.observe(chart_root, {childList: true, subtree: true});
    const startup_timer = setInterval(() => {
      apply_startup_policy();
      if (startup_done) clearInterval(startup_timer);
    }, 100);
    setTimeout(() => clearInterval(startup_timer), 30000);
    apply_startup_policy();

    const style = document.createElement("style");
    style.id = "thog2_sep07_fixes_and_enhancements_style";
    style.textContent = `
      :root { --instra-ordinary-text-size: 12px; }
      #settings_overlay .settings-dialog,
      #chart_settings_overlay .settings-dialog { font-size: var(--instra-ordinary-text-size) !important; }
      #settings_overlay .settings-dialog :is(label, input, select, button, p, span),
      #chart_settings_overlay .settings-dialog :is(label, input, select, button, p, span) { font-size: inherit; }
      .ordinary-text-setting {
        display: grid; grid-template-columns: minmax(210px, 1fr) 150px; gap: 12px;
        align-items: center; margin-top: 18px; padding-top: 18px; border-top: 1px solid var(--border);
      }
      .ordinary-text-control { display: flex; align-items: center; gap: 7px; }
      .ordinary-text-control input { width: 92px; height: 32px; padding: 0 8px; border: 1px solid var(--border); border-radius: 4px; }
      .ordinary-text-control span { color: var(--muted); }
      .chart-group-toggle { align-items: center !important; }
      .group-caret { display: inline-grid !important; place-items: center; width: 14px; min-width: 14px; align-self: center; }
      .chart-card-header .chart-card-actions { order: 999; margin-left: auto !important; padding-right: 2px; }
      .chart-card-header .chart-card-actions .maximize-button { order: 9999; margin-left: 5px; }
      #runs_body tr.instra-curve-hover > td { background: #f3f4f6 !important; }
      .instra-just-now-flash { font-weight: 800 !important; }
      .runs-table .instra-sep07-column { white-space: nowrap; text-align: right; }
      .plotly .hoverlayer .hovertext text { fill: #ffffff !important; }
    `;
    document.head.appendChild(style);

    window.__instra_sep07 = Object.freeze({
      apply_startup_policy,
      polish_run_table,
      restyle_workspace_curves,
      sync_navigation_selection,
    });
  }, 80);
});
// ^^^ THOG
