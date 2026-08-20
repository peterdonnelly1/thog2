// vvv THOG
"use strict";

// Instra heatmap v0.57: viewer-owned history windows, exact probe/step rows,
// decision and runtime-state annotation, and the final two-axis presentation.
window.addEventListener("load", () => {
  setTimeout(() => {
    const default_settings = Object.freeze({
      probe_count: 100,
      window_mode: "rolling",
      y_display_mode: "probes",
      delta_loss_display_mode: "percent",
      auto_colour_saturation: false,
    });
    const axis_tick_font_px = 14;
    const axis_title_font_px = 14;
    let heatmap_dialog_draft = null;

    const finite_number = value => {
      if (value === null || value === undefined || value === "") return null;
      const numeric = Number(value);
      return Number.isFinite(numeric) ? numeric : null;
    };
    const finite_positive = (value, fallback) => {
      const numeric = finite_number(value);
      return numeric !== null && numeric > 0 ? numeric : fallback;
    };
    const base_heatmap_settings_for_current_run_v057 = heatmap_settings_for_current_run;
    const persisted_settings = () => ({
      ...default_settings,
      ...base_heatmap_settings_for_current_run_v057(),
    });
    heatmap_settings_for_current_run = function() {
      const persisted = persisted_settings();
      return heatmap_dialog_draft === null ? persisted : {...persisted, ...heatmap_dialog_draft};
    };

    const make_field = (parent, label_text, id, kind, options = {}) => {
      const label = document.createElement("label");
      label.className = "heatmap-v057-field";
      label.htmlFor = id;
      const caption = document.createElement("span");
      caption.textContent = label_text;
      let input;
      if (kind === "select") {
        input = document.createElement("select");
        for (const [value, text] of options.choices || []) {
          const option = document.createElement("option");
          option.value = value;
          option.textContent = text;
          input.appendChild(option);
        }
      } else {
        input = document.createElement("input");
        input.type = kind;
        if (options.min !== undefined) input.min = String(options.min);
        if (options.max !== undefined) input.max = String(options.max);
        if (options.step !== undefined) input.step = String(options.step);
        if (options.readonly) input.readOnly = true;
      }
      input.id = id;
      input.className = "chart-setting-input";
      label.append(caption, input);
      parent.appendChild(label);
      return input;
    };

    const install_heatmap_gear_fields = () => {
      const panel = by_id("chart_heatmap_display_options");
      if (!panel || by_id("chart_heatmap_probe_count")) return;
      const section = document.createElement("section");
      section.className = "heatmap-v057-settings";
      const heading = document.createElement("h3");
      heading.textContent = "Heatmap data and colour";
      const grid = document.createElement("div");
      grid.className = "heatmap-v057-settings-grid";
      make_field(grid, "Capture mode", "chart_heatmap_capture_mode", "text", {readonly: true});
      make_field(grid, "Destination", "chart_heatmap_destination", "text", {readonly: true});
      make_field(grid, "Probe count", "chart_heatmap_probe_count", "number", {min: 1, max: 512, step: 1});
      make_field(grid, "History window", "chart_heatmap_window_mode", "select", {
        choices: [["rolling", "Rolling (latest probes)"], ["from_zero", "From zero (earliest probes)"]],
      });
      make_field(grid, "Rows", "chart_heatmap_y_display_mode", "select", {
        choices: [["probes", "Probes only"], ["steps", "Every optimizer step"]],
      });
      make_field(grid, "Δloss display", "chart_heatmap_delta_mode", "select", {
        choices: [["percent", "Percentage"], ["absolute", "|abs|"]],
      });
      make_field(grid, "Automatic colour saturation", "chart_heatmap_auto_colour", "checkbox");
      make_field(grid, "Base absolute limit", "chart_heatmap_abs_limit", "number", {min: 0.000000001, step: 0.001});
      make_field(grid, "Green saturation", "chart_heatmap_green_limit", "number", {min: 0.000000001, max: 0.1, step: 0.005});
      make_field(grid, "Blue saturation", "chart_heatmap_blue_limit", "number", {min: 0.100000001, max: 1, step: 0.05});
      make_field(grid, "Yellow saturation", "chart_heatmap_yellow_limit", "number", {min: 1.000000001, step: 0.1});
      make_field(grid, "Red saturation", "chart_heatmap_red_limit", "number", {min: 0.000000001, step: 0.005});
      const note = document.createElement("p");
      note.className = "chart-fidelity-note";
      note.textContent = (
        "Probe count is a reversible Instra view window; capture retains the complete local history. "
        + "Every-step mode inserts blank rows between actual probes and always labels optimizer steps."
      );
      section.append(heading, grid, note);
      panel.appendChild(section);

      for (const input of grid.querySelectorAll("input:not([readonly]), select")) {
        input.addEventListener("input", () => {
          read_heatmap_gear_draft();
          schedule_chart_settings_preview();
        });
        input.addEventListener("change", () => {
          read_heatmap_gear_draft();
          schedule_chart_settings_preview();
        });
      }
    };

    const populate_heatmap_gear_fields = () => {
      install_heatmap_gear_fields();
      const run_settings = heatmap_run_settings();
      const settings = heatmap_settings_for_current_run();
      by_id("chart_heatmap_capture_mode").value = String(run_settings.mode ?? "—");
      by_id("chart_heatmap_destination").value = String(run_settings.destination ?? "—");
      by_id("chart_heatmap_probe_count").value = String(
        Math.max(1, Math.min(512, Math.round(finite_positive(settings.probe_count, 100))))
      );
      by_id("chart_heatmap_window_mode").value = settings.window_mode === "from_zero" ? "from_zero" : "rolling";
      by_id("chart_heatmap_y_display_mode").value = settings.y_display_mode === "steps" ? "steps" : "probes";
      by_id("chart_heatmap_delta_mode").value = settings.delta_loss_display_mode === "absolute" ? "absolute" : "percent";
      by_id("chart_heatmap_auto_colour").checked = settings.auto_colour_saturation === true;
      by_id("chart_heatmap_abs_limit").value = String(finite_positive(settings.abs_limit, heatmap_abs_limit(0.05)));
      by_id("chart_heatmap_green_limit").value = String(finite_positive(settings.negative_abs_limit, Math.min(0.1, heatmap_abs_limit(0.05))));
      by_id("chart_heatmap_blue_limit").value = String(finite_positive(settings.blue_abs_limit, 1));
      by_id("chart_heatmap_yellow_limit").value = String(finite_positive(settings.yellow_abs_limit, 2));
      by_id("chart_heatmap_red_limit").value = String(finite_positive(settings.positive_abs_limit, heatmap_abs_limit(0.05)));
    };

    function read_heatmap_gear_draft() {
      if (!by_id("chart_heatmap_probe_count")) return;
      const probe_count = Math.round(Number(by_id("chart_heatmap_probe_count").value));
      heatmap_dialog_draft = {
        ...(heatmap_dialog_draft || persisted_settings()),
        probe_count: Math.max(1, Math.min(512, Number.isFinite(probe_count) ? probe_count : 100)),
        window_mode: by_id("chart_heatmap_window_mode").value === "from_zero" ? "from_zero" : "rolling",
        y_display_mode: by_id("chart_heatmap_y_display_mode").value === "steps" ? "steps" : "probes",
        delta_loss_display_mode: by_id("chart_heatmap_delta_mode").value === "absolute" ? "absolute" : "percent",
        auto_colour_saturation: by_id("chart_heatmap_auto_colour").checked,
        abs_limit: finite_positive(by_id("chart_heatmap_abs_limit").value, 0.05),
        negative_abs_limit: Math.min(0.1, finite_positive(by_id("chart_heatmap_green_limit").value, 0.05)),
        blue_abs_limit: Math.min(1, Math.max(0.100000001, finite_positive(by_id("chart_heatmap_blue_limit").value, 1))),
        yellow_abs_limit: Math.max(1.000000001, finite_positive(by_id("chart_heatmap_yellow_limit").value, 2)),
        positive_abs_limit: finite_positive(by_id("chart_heatmap_red_limit").value, 0.05),
      };
    }

    const persist_heatmap_draft = () => {
      if (heatmap_dialog_draft === null) return;
      for (const [name, value] of Object.entries(heatmap_dialog_draft)) {
        save_heatmap_viewer_setting(name, value);
      }
    };

    install_heatmap_gear_fields();
    const legacy_settings = by_id("heatmap_settings_section");
    if (legacy_settings) {
      legacy_settings.hidden = true;
      legacy_settings.setAttribute("aria-hidden", "true");
    }

    const base_open_chart_settings_v057 = open_chart_settings;
    open_chart_settings = function(chart_name) {
      heatmap_dialog_draft = chart_name === "heatmap" ? persisted_settings() : null;
      const result = base_open_chart_settings_v057(chart_name);
      if (chart_name === "heatmap") populate_heatmap_gear_fields();
      return result;
    };

    const base_close_chart_settings_v057 = close_chart_settings;
    close_chart_settings = function() {
      heatmap_dialog_draft = null;
      return base_close_chart_settings_v057();
    };

    const base_save_chart_settings_v057 = save_chart_settings;
    save_chart_settings = function() {
      const heatmap = app.axis_chart_name === "heatmap";
      if (heatmap) {
        read_heatmap_gear_draft();
        persist_heatmap_draft();
      }
      const result = base_save_chart_settings_v057();
      if (heatmap) {
        const performance = window.__thog2_dashboard_performance?.state;
        if (performance) performance.heatmap_signature = null;
        app.figure_revision = null;
        queueMicrotask(() => refresh_current_run());
      }
      return result;
    };

    const base_reset_chart_settings_v057 = reset_chart_settings;
    reset_chart_settings = function() {
      const heatmap = app.axis_chart_name === "heatmap";
      const result = base_reset_chart_settings_v057();
      if (heatmap) {
        heatmap_dialog_draft = {...default_settings};
        populate_heatmap_gear_fields();
      }
      return result;
    };

    const source_heatmap_trace = prepared => (
      (prepared.data || []).find(trace => trace.type === "heatmap") || null
    );
    const source_active_trace = (prepared, heatmap_trace) => (
      (prepared.data || []).find(trace => (
        trace !== heatmap_trace && Array.isArray(trace.x) && Array.isArray(trace.y)
      )) || null
    );
    const metadata_array = (meta, name, length, fallback = null) => {
      const source = Array.isArray(meta?.[name]) ? meta[name] : [];
      return Array.from({length}, (_unused, index) => (
        source[index] === undefined ? fallback : source[index]
      ));
    };

    const expand_source_to_optimizer_steps = prepared => {
      if (heatmap_settings_for_current_run().y_display_mode !== "steps") return;
      const heatmap = source_heatmap_trace(prepared);
      const active = source_active_trace(prepared, heatmap);
      if (!heatmap || !active) return;
      const meta = prepared.layout?.meta || {};
      const source_steps = metadata_array(meta, "thog2_optimizer_updates", active.y.length)
        .map(Number);
      if (!source_steps.length || source_steps.some(step => !Number.isInteger(step))) return;
      const first_step = source_steps[0];
      const last_step = source_steps[source_steps.length - 1];
      if (last_step < first_step) return;
      const expanded_steps = Array.from(
        {length: last_step - first_step + 1},
        (_unused, index) => first_step + index,
      );
      const probe_index_by_step = new Map(source_steps.map((step, index) => [step, index]));
      const source_active = metadata_array(meta, "thog2_active_layers", source_steps.length)
        .map(Number);
      const source_selected = metadata_array(meta, "thog2_selected_layers", source_steps.length)
        .map((value, index) => Number(value ?? source_active[index]));
      const source_current = metadata_array(meta, "thog2_current_losses", source_steps.length);
      const source_brake = metadata_array(meta, "thog2_brake_active", source_steps.length, false);
      const source_decision = metadata_array(meta, "thog2_decision_committed", source_steps.length, false);
      const source_chaos = metadata_array(meta, "thog2_chaos_bump", source_steps.length);
      const expanded = {
        active: [], selected: [], current: [], brake: [], decision: [], chaos: [],
      };
      let prior_probe = 0;
      for (const step of expanded_steps) {
        const probe_index = probe_index_by_step.get(step);
        if (probe_index !== undefined) prior_probe = probe_index;
        const exact = probe_index !== undefined;
        const carried_active = exact
          ? source_active[probe_index]
          : source_selected[prior_probe];
        expanded.active.push(carried_active);
        expanded.selected.push(exact ? source_selected[probe_index] : carried_active);
        expanded.current.push(exact ? source_current[probe_index] : null);
        expanded.brake.push(exact ? Boolean(source_brake[probe_index]) : false);
        expanded.decision.push(exact ? Boolean(source_decision[probe_index]) : false);
        if (exact) {
          expanded.chaos.push(source_chaos[probe_index]);
        } else {
          const marker = source_chaos[prior_probe];
          if (marker?.state === "active") {
            const elapsed = step - source_steps[prior_probe];
            const bump_step = Number(marker.step) + elapsed;
            expanded.chaos.push(bump_step <= Number(marker.duration)
              ? {...marker, step: bump_step}
              : null);
          } else {
            expanded.chaos.push(null);
          }
        }
      }

      const original_z = Array.isArray(heatmap.z) ? heatmap.z : [];
      heatmap.x = expanded_steps.map((_step, index) => index + 1);
      heatmap.z = original_z.map(row => expanded_steps.map(step => {
        const index = probe_index_by_step.get(step);
        return index === undefined ? null : row[index];
      }));
      if (Array.isArray(heatmap.customdata)) {
        heatmap.customdata = heatmap.customdata.map(row => expanded_steps.map(step => {
          const index = probe_index_by_step.get(step);
          return index === undefined ? step : row[index];
        }));
      }
      active.x = expanded_steps.map((_step, index) => index + 1);
      active.y = expanded.active;
      active.customdata = expanded_steps;
      prepared.layout.meta = {
        ...meta,
        thog2_optimizer_updates: expanded_steps,
        thog2_active_layers: expanded.active,
        thog2_selected_layers: expanded.selected,
        thog2_current_losses: expanded.current,
        thog2_brake_active: expanded.brake,
        thog2_decision_committed: expanded.decision,
        thog2_chaos_bump: expanded.chaos,
      };
    };

    const add_step_customdata = (prepared, heatmap) => {
      const steps = prepared.layout?.meta?.thog2_optimizer_updates || [];
      const active = prepared.layout?.meta?.thog2_active_layers || [];
      const offsets = Array.isArray(heatmap.x) ? heatmap.x.map(Number) : [];
      if (!Array.isArray(heatmap.customdata)) return;
      for (let row = 0; row < heatmap.customdata.length; row += 1) {
        if (!Array.isArray(heatmap.customdata[row])) heatmap.customdata[row] = [];
        for (let column = 0; column < offsets.length; column += 1) {
          if (Array.isArray(heatmap.customdata[row][column])) continue;
          heatmap.customdata[row][column] = [
            steps[row],
            Number(active[row]) + offsets[column],
            signed_layer_offset(offsets[column]),
            null,
            null,
            null,
          ];
        }
      }
    };

    const decision_overlays = (prepared, heatmap) => {
      const meta = prepared.layout?.meta || {};
      const active = meta.thog2_active_layers || [];
      const selected = meta.thog2_selected_layers || [];
      const decisions = meta.thog2_decision_committed || [];
      const current = meta.thog2_current_losses || [];
      const ys = Array.isArray(heatmap.y) ? heatmap.y : [];
      const xs = Array.isArray(heatmap.x) ? heatmap.x.map(Number) : [];
      const custom = Array.isArray(heatmap.customdata) ? heatmap.customdata : [];
      const shapes = (prepared.layout.shapes || []).filter(
        shape => shape?.name !== "thog2-committed-decision-brick"
      );
      const annotations = (prepared.layout.annotations || []).filter(
        annotation => annotation?.name !== "thog2-committed-decision-text"
      );
      for (let row = 0; row < ys.length; row += 1) {
        if (!Boolean(decisions[row]) || Number(selected[row]) === Number(active[row])) continue;
        const offset = Number(selected[row]) - Number(active[row]);
        const column = xs.findIndex(value => value === offset);
        if (column < 0) continue;
        shapes.push({
          name: "thog2-committed-decision-brick",
          type: "rect", xref: "x", yref: "y",
          x0: offset - 0.5, x1: offset + 0.5,
          y0: Number(ys[row]) - 0.5, y1: Number(ys[row]) + 0.5,
          line: {color: "#000000", width: 1}, fillcolor: "#ffffff", layer: "above",
        });
        const raw_delta = finite_number(custom[row]?.[column]?.[3]);
        const centre_loss = finite_number(current[row]);
        const loss = raw_delta === null || centre_loss === null ? null : centre_loss + raw_delta;
        annotations.push({
          name: "thog2-committed-decision-text",
          x: offset, y: ys[row], xref: "x", yref: "y",
          text: loss === null ? "<b>decision</b>" : `<b>${loss.toFixed(3)}</b>`,
          showarrow: false, xanchor: "center", yanchor: "middle",
          font: {family: "DejaVu Sans Mono, monospace", size: 11, color: "#000000"},
          captureevents: false,
        });
      }
      prepared.layout.shapes = shapes;
      prepared.layout.annotations = annotations;
    };

    const header_annotations = prepared => {
      const meta = prepared.layout?.meta || {};
      const brakes = meta.thog2_brake_active || [];
      const chaos = meta.thog2_chaos_bump || [];
      const latest_brake = Boolean(brakes[brakes.length - 1]);
      const latest_chaos = chaos[chaos.length - 1];
      const annotations = (prepared.layout.annotations || []).filter(
        annotation => !["thog2-update-brake", "thog2-chaos-bump"].includes(annotation?.name)
      );
      if (latest_brake) {
        annotations.push({
          name: "thog2-update-brake", x: 0.01, y: 1.16, xref: "paper", yref: "paper",
          text: "<b>update brake on</b>", showarrow: false,
          xanchor: "left", yanchor: "bottom", font: {size: 13, color: "#111111"},
        });
      }
      if (latest_chaos?.state === "active") {
        annotations.push({
          name: "thog2-chaos-bump", x: 0.01, y: 1.105, xref: "paper", yref: "paper",
          text: (
            `sampling chaos bump made - magnitude ${Number(latest_chaos.magnitude_percent).toFixed(1)}%. `
            + `Step ${latest_chaos.step}/${latest_chaos.duration}`
          ),
          showarrow: false, xanchor: "left", yanchor: "bottom",
          font: {size: 13, color: "#96dcff"},
        });
      } else if (latest_chaos?.state === "reverted") {
        annotations.push({
          name: "thog2-chaos-bump", x: 0.01, y: 1.105, xref: "paper", yref: "paper",
          text: "reverted to pre-chaos bump sampling", showarrow: false,
          xanchor: "left", yanchor: "bottom", font: {size: 13, color: "#96dcff"},
        });
      }
      prepared.layout.annotations = annotations;
    };

    const configure_axes = (prepared, heatmap) => {
      const meta = prepared.layout?.meta || {};
      const active = meta.thog2_active_layers || [];
      const selected = meta.thog2_selected_layers || [];
      const offsets = Array.isArray(heatmap.x) ? heatmap.x.map(Number) : [];
      const latest_l = finite_number(active[active.length - 1]);
      const prior = active.length > 1 ? active.length - 2 : -1;
      const highlight_l = prior >= 0 && (
        Number(selected[prior]) !== Number(active[prior])
        && Number(active[active.length - 1]) === Number(selected[prior])
      );
      const relative_ticktext = offsets.map(offset => (
        offset === 0 && latest_l !== null
          ? `<b style="color:${highlight_l ? "#1769d2" : "#20252c"}">L=${latest_l}</b>`
          : signed_layer_offset(offset)
      ));
      const absolute_ticktext = offsets.map(offset => (
        latest_l === null ? "—" : String(latest_l + offset)
      ));
      prepared.layout.xaxis = {
        ...(prepared.layout.xaxis || {}),
        side: "bottom", anchor: "y", tickmode: "array", tickvals: offsets,
        ticktext: relative_ticktext,
        title: {text: "candidate layer-count offset from L", standoff: 18, font: {size: axis_title_font_px}},
        tickfont: {...(prepared.layout.xaxis?.tickfont || {}), size: axis_tick_font_px},
      };
      prepared.layout.xaxis2 = {
        ...(prepared.layout.xaxis || {}),
        side: "top", anchor: "y", overlaying: "x", matches: "x",
        tickmode: "array", tickvals: offsets, ticktext: absolute_ticktext,
        showgrid: false, zeroline: false,
        title: {
          text: latest_l === null ? "absolute candidate layer count" : `absolute candidate layer count · latest L=${latest_l}`,
          standoff: 18, font: {size: axis_title_font_px},
        },
        tickfont: {...(prepared.layout.xaxis?.tickfont || {}), size: axis_tick_font_px},
      };
      const steps = meta.thog2_optimizer_updates || [];
      const ys = Array.isArray(heatmap.y) ? heatmap.y : [];
      const tick_indices = evenly_spaced_indices(ys.length, 16);
      prepared.layout.yaxis = {
        ...(prepared.layout.yaxis || {}),
        title: {text: "optimizer step", font: {size: axis_title_font_px}},
        tickmode: "array",
        tickvals: tick_indices.map(index => ys[index]),
        ticktext: tick_indices.map(index => String(steps[index] ?? "")),
        tickfont: {...(prepared.layout.yaxis?.tickfont || {}), size: axis_tick_font_px},
      };
      prepared.layout.margin = {...(prepared.layout.margin || {}), t: 126, b: 82};
    };

    const finish_colour_key = heatmap => {
      const colorbar = heatmap.colorbar || {};
      const title = typeof colorbar.title === "string"
        ? {text: colorbar.title}
        : {...(colorbar.title || {})};
      heatmap.colorbar = {
        ...colorbar,
        y: Math.min(0.54, Number(colorbar.y || 0.5) + 0.025),
        title: {...title, side: "top", font: {...(title.font || {}), size: 11}},
      };
    };

    const base_transpose_heatmap_v057 = transpose_heatmap;
    transpose_heatmap = function(prepared) {
      expand_source_to_optimizer_steps(prepared);
      base_transpose_heatmap_v057(prepared);
      const heatmap = source_heatmap_trace(prepared);
      if (!heatmap) return;
      add_step_customdata(prepared, heatmap);
      decision_overlays(prepared, heatmap);
      configure_axes(prepared, heatmap);
      header_annotations(prepared);
      finish_colour_key(heatmap);
    };

    const position_header_controls = () => {
      const header = document.querySelector('.chart-card[data-chart="heatmap"] .chart-card-header');
      const actions = header?.querySelector(".chart-card-actions");
      const mode = by_id("heatmap_delta_loss_modes") || by_id("heatmap_delta_loss_mode");
      const scale = by_id("heatmap_vertical_scale")?.closest(".heatmap-vertical-scale-control");
      if (!header || !actions || !mode || !scale) return;
      header.classList.add("heatmap-v057-header");
      mode.classList.add("heatmap-v057-mode-control");
      scale.classList.add("heatmap-v057-scale-control");
    };
    position_header_controls();

    const style = document.createElement("style");
    style.textContent = `
      .heatmap-v057-settings { margin-top: 18px; padding-top: 16px; border-top: 1px solid var(--border); }
      .heatmap-v057-settings-grid { display: grid; grid-template-columns: minmax(190px, 1fr) minmax(160px, 220px); gap: 10px 14px; align-items: center; }
      .heatmap-v057-field { display: contents; }
      .heatmap-v057-field > span { font-size: 11px; color: #313842; }
      .heatmap-v057-field input:not([type="checkbox"]), .heatmap-v057-field select { width: 100%; min-height: 32px; padding: 4px 8px; border: 1px solid #cfd4da; border-radius: 4px; background: #fff; }
      .heatmap-v057-field input[type="checkbox"] { width: 18px; height: 18px; accent-color: #1995ad; }
      .heatmap-v057-header { position: relative !important; min-height: 58px; }
      .heatmap-v057-header .chart-card-actions { position: static !important; }
      .heatmap-v057-mode-control { position: absolute !important; left: 50% !important; top: 9px !important; transform: translateX(-50%) !important; z-index: 8; }
      .heatmap-v057-scale-control { position: absolute !important; left: 73% !important; top: 9px !important; transform: translateX(-50%) !important; z-index: 8; }
      .heatmap-v057-header .chart-settings-button { margin-left: 8px !important; }
      .chart-card[data-chart="heatmap"] .xtick text,
      .chart-card[data-chart="heatmap"] .ytick text { font-size: ${axis_tick_font_px}px !important; }
    `;
    document.head.appendChild(style);

    const base_render_figures_v057 = render_figures;
    render_figures = async function() {
      const result = await base_render_figures_v057();
      const detail = by_id("heatmap_card_detail");
      const status = app.current_status || current_run();
      if (detail && status) {
        const dimensions = app.figures?.heatmap_dimensions || {};
        detail.textContent = (
          `${format_integer(status.heatmap_count)} probes · latest step ${format_integer(status.heatmap_maximum_update)}`
          + ` · showing ${format_integer(dimensions.probes || 0)} ${dimensions.window_mode === "from_zero" ? "from zero" : "rolling"}`
        );
      }
      position_header_controls();
      return result;
    };

    document.querySelector('.chart-card[data-chart="heatmap"] .chart-heading-copy h2')
      ?.replaceChildren("Heatmap - Loss vs Counterfactual Layer Count");

    if (app.figures?.heatmap && app.current_run_id) {
      queueMicrotask(() => render_figures());
    }
  }, 800);
});
// ^^^ THOG
