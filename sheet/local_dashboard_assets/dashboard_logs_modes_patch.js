// vvv THOG
"use strict";

// Final local-dashboard interaction pass: a real ANSI train.log viewer, explicit
// stateful Abs/% and Linear/Log controls, and enough top-label offset for literal
// one-pixel heatmap rows.
window.addEventListener("load", () => {
  setTimeout(() => {
    const trajectory_chart_names = [
      "attn_q_head_N",
      "attn_k_head_N",
      "attn_v_head_N",
      "attn_out_head_N",
      "mlp_up",
      "mlp_down",
    ];
    const trajectory_scale_settings_key = "thog2_local_trajectory_scale_modes";
    const heatmap_display_mode_setting = "delta_loss_display_mode";
    const log_font_storage_key = "thog2_local_log_font_size";
    const log_maximum_lines = 5000;
    const log_poll_interval_ms = 1000;

    const trajectory_mode = chart_name => (
      load_json(trajectory_scale_settings_key, {})[chart_name] === "log" ? "log" : "linear"
    );
    const save_trajectory_mode = (chart_name, mode) => {
      const settings = load_json(trajectory_scale_settings_key, {});
      settings[chart_name] = mode;
      save_json(trajectory_scale_settings_key, settings);
    };
    const heatmap_mode = () => (
      heatmap_settings_for_current_run()[heatmap_display_mode_setting] === "percent"
        ? "percent"
        : "absolute"
    );

    const make_mode_button = (label, mode, title, glyph_kind = null) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "explicit-mode-button";
      button.dataset.mode = mode;
      button.title = title;
      button.setAttribute("aria-label", title);
      if (glyph_kind) {
        const glyph = document.createElement("span");
        glyph.className = `explicit-scale-glyph ${glyph_kind}`;
        glyph.setAttribute("aria-hidden", "true");
        glyph.append(document.createElement("i"), document.createElement("i"), document.createElement("i"));
        button.appendChild(glyph);
      }
      const copy = document.createElement("span");
      copy.textContent = label;
      button.appendChild(copy);
      return button;
    };

    const install_heatmap_mode_buttons = () => {
      const old_button = by_id("heatmap_delta_loss_mode");
      if (!old_button || by_id("heatmap_delta_loss_modes")) return;
      old_button.hidden = true;
      const group = document.createElement("div");
      group.id = "heatmap_delta_loss_modes";
      group.className = "explicit-mode-group heatmap-mode-group";
      group.setAttribute("role", "group");
      group.setAttribute("aria-label", "Heatmap delta-loss display mode");
      const absolute = make_mode_button("Abs", "absolute", "Colour heatmap by absolute Δloss");
      const percent = make_mode_button("%", "percent", "Colour heatmap by percentage Δloss relative to the centre L loss");
      const choose = async mode => {
        if (!app.current_run_id || heatmap_mode() === mode) return;
        save_heatmap_viewer_setting(heatmap_display_mode_setting, mode);
        sync_mode_buttons();
        const figure = app.figures?.heatmap;
        const mount = by_id("heatmap_plot");
        if (figure && mount) await render_plot(mount, figure, "heatmap");
      };
      absolute.addEventListener("click", event => {
        event.preventDefault(); event.stopPropagation(); choose("absolute");
      });
      percent.addEventListener("click", event => {
        event.preventDefault(); event.stopPropagation(); choose("percent");
      });
      group.append(absolute, percent);
      old_button.insertAdjacentElement("beforebegin", group);
    };

    const install_trajectory_mode_buttons = () => {
      for (const chart_name of trajectory_chart_names) {
        const card = document.querySelector(`.chart-card[data-chart="${chart_name}"]`);
        const old_button = card?.querySelector(".trajectory-scale-toggle");
        if (!old_button || card.querySelector(".explicit-trajectory-modes")) continue;
        old_button.hidden = true;
        const group = document.createElement("div");
        group.className = "explicit-mode-group explicit-trajectory-modes";
        group.dataset.chartName = chart_name;
        group.setAttribute("role", "group");
        group.setAttribute("aria-label", `${chart_titles[chart_name] || chart_name} Y scale`);
        const linear = make_mode_button("Linear", "linear", "Use a linear Y scale", "linear");
        const log = make_mode_button("Log", "log", "Use the signed-log Y scale; negative and zero values remain visible", "log");
        const choose = async mode => {
          if (trajectory_mode(chart_name) === mode) return;
          save_trajectory_mode(chart_name, mode);
          sync_mode_buttons();
          const figure = app.figures?.depth?.[chart_name];
          const mount = by_id(`${chart_name}_plot`);
          if (figure && mount) await render_plot(mount, figure, chart_name);
        };
        linear.addEventListener("click", event => {
          event.preventDefault(); event.stopPropagation(); choose("linear");
        });
        log.addEventListener("click", event => {
          event.preventDefault(); event.stopPropagation(); choose("log");
        });
        group.append(linear, log);
        old_button.insertAdjacentElement("beforebegin", group);
      }
    };

    function sync_mode_buttons() {
      const heatmap_group = by_id("heatmap_delta_loss_modes");
      if (heatmap_group) {
        const mode = heatmap_mode();
        for (const button of heatmap_group.querySelectorAll(".explicit-mode-button")) {
          const active = button.dataset.mode === mode;
          button.classList.toggle("active", active);
          button.setAttribute("aria-pressed", String(active));
          button.disabled = !app.current_run_id;
        }
      }
      for (const group of document.querySelectorAll(".explicit-trajectory-modes")) {
        const mode = trajectory_mode(group.dataset.chartName);
        for (const button of group.querySelectorAll(".explicit-mode-button")) {
          const active = button.dataset.mode === mode;
          button.classList.toggle("active", active);
          button.setAttribute("aria-pressed", String(active));
        }
      }
    }

    const ansi_basic = [
      "#000000", "#cd0000", "#00cd00", "#cdcd00",
      "#0000ee", "#cd00cd", "#00cdcd", "#e5e5e5",
      "#7f7f7f", "#ff0000", "#00ff00", "#ffff00",
      "#5c5cff", "#ff00ff", "#00ffff", "#ffffff",
    ];

    const ansi_256_colour = value => {
      const index = Math.max(0, Math.min(255, Number(value) || 0));
      if (index < 16) return ansi_basic[index];
      if (index >= 232) {
        const shade = 8 + (index - 232) * 10;
        return `rgb(${shade},${shade},${shade})`;
      }
      const cube = index - 16;
      const blue = cube % 6;
      const green = Math.floor(cube / 6) % 6;
      const red = Math.floor(cube / 36) % 6;
      const component = level => level === 0 ? 0 : 55 + level * 40;
      return `rgb(${component(red)},${component(green)},${component(blue)})`;
    };

    const clean_terminal_text = value => String(value)
      .replace(/\x1b\][^\x07]*(?:\x07|\x1b\\)/g, "")
      .replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, "");

    const apply_sgr = (parameters, state) => {
      const values = parameters.length ? parameters.map(value => Number(value || 0)) : [0];
      for (let index = 0; index < values.length; index += 1) {
        const code = values[index];
        if (code === 0) {
          state.colour = null; state.bold = false; state.dim = false; state.italic = false; state.underline = false;
        } else if (code === 1) state.bold = true;
        else if (code === 2) state.dim = true;
        else if (code === 3) state.italic = true;
        else if (code === 4) state.underline = true;
        else if (code === 22) { state.bold = false; state.dim = false; }
        else if (code === 23) state.italic = false;
        else if (code === 24) state.underline = false;
        else if (code === 39) state.colour = null;
        else if (code >= 30 && code <= 37) state.colour = ansi_basic[code - 30];
        else if (code >= 90 && code <= 97) state.colour = ansi_basic[8 + code - 90];
        else if (code === 38 && values[index + 1] === 5 && Number.isFinite(values[index + 2])) {
          state.colour = ansi_256_colour(values[index + 2]); index += 2;
        } else if (code === 38 && values[index + 1] === 2 && values.slice(index + 2, index + 5).every(Number.isFinite)) {
          const [red, green, blue] = values.slice(index + 2, index + 5).map(value => Math.max(0, Math.min(255, value)));
          state.colour = `rgb(${red},${green},${blue})`; index += 4;
        }
      }
    };

    const ansi_line = value => {
      const line = document.createElement("div");
      line.className = "local-log-line";
      const state = {colour: null, bold: false, dim: false, italic: false, underline: false};
      const expression = /\x1b\[([0-9;]*)m/g;
      let cursor = 0;
      let match = null;
      const append = text => {
        const cleaned = clean_terminal_text(text);
        if (!cleaned) return;
        const span = document.createElement("span");
        span.textContent = cleaned;
        if (state.colour) span.style.color = state.colour;
        if (state.bold) span.style.fontWeight = "700";
        if (state.dim) span.style.opacity = "0.72";
        if (state.italic) span.style.fontStyle = "italic";
        if (state.underline) span.style.textDecoration = "underline";
        line.appendChild(span);
      };
      while ((match = expression.exec(value)) !== null) {
        append(value.slice(cursor, match.index));
        apply_sgr(match[1].split(";"), state);
        cursor = expression.lastIndex;
      }
      append(value.slice(cursor));
      if (!line.childNodes.length) line.appendChild(document.createTextNode(" "));
      return line;
    };

    let log_font_size = Number(localStorage.getItem(log_font_storage_key));
    if (!Number.isFinite(log_font_size)) log_font_size = 12;
    log_font_size = Math.max(8, Math.min(24, Math.round(log_font_size)));
    const log_state = {
      run_id: null,
      path: "",
      offset: null,
      partial: "",
      partial_node: null,
      loading: false,
      user_scrolled: false,
    };

    const set_log_font_size = value => {
      log_font_size = Math.max(8, Math.min(24, Math.round(value)));
      localStorage.setItem(log_font_storage_key, String(log_font_size));
      const output = by_id("local_log_output");
      if (output) output.style.fontSize = `${log_font_size}px`;
      const value_node = by_id("local_log_font_value");
      if (value_node) value_node.textContent = `${log_font_size}px`;
    };

    const reset_log_state = () => {
      log_state.run_id = app.current_run_id;
      log_state.path = "";
      log_state.offset = null;
      log_state.partial = "";
      log_state.partial_node = null;
      log_state.user_scrolled = false;
      const output = by_id("local_log_output");
      if (output) {
        output.replaceChildren();
        output.scrollTop = 0;
      }
      const status = by_id("local_log_status");
      if (status) status.textContent = "Loading train.log…";
    };

    const trim_log_lines = output => {
      while (output.childElementCount > log_maximum_lines + (log_state.partial_node ? 1 : 0)) {
        output.lastElementChild?.remove();
      }
    };

    const ingest_log_text = (text, reset) => {
      const output = by_id("local_log_output");
      if (!output) return;
      if (reset) {
        output.replaceChildren();
        log_state.partial = "";
        log_state.partial_node = null;
        log_state.user_scrolled = false;
      }
      if (log_state.partial_node) {
        log_state.partial_node.remove();
        log_state.partial_node = null;
      }
      const normalized = `${log_state.partial}${String(text || "")}`
        .replace(/\r\n/g, "\n")
        .replace(/\r/g, "\n");
      const lines = normalized.split("\n");
      log_state.partial = lines.pop() ?? "";
      for (const line_text of lines) output.prepend(ansi_line(line_text));
      if (log_state.partial) {
        log_state.partial_node = ansi_line(log_state.partial);
        log_state.partial_node.classList.add("partial");
        output.prepend(log_state.partial_node);
      }
      trim_log_lines(output);
      if (!log_state.user_scrolled) output.scrollTop = 0;
    };

    const refresh_log = async () => {
      if (local_active_detail_tab !== "logs" || !app.current_run_id || log_state.loading) return;
      if (log_state.run_id !== app.current_run_id) reset_log_state();
      log_state.loading = true;
      try {
        const offset = log_state.offset === null ? "" : `&offset=${encodeURIComponent(log_state.offset)}`;
        const payload = await fetch_json(
          `/api/log?run=${encodeURIComponent(app.current_run_id)}${offset}&max_bytes=${2 * 1024 * 1024}`
        );
        if (!payload.available) {
          const status = by_id("local_log_status");
          if (status) status.textContent = "No matching local train.log was found for this run.";
          return;
        }
        const source_changed = log_state.path && log_state.path !== payload.path;
        const reset = Boolean(payload.reset || source_changed);
        if (reset || payload.text) ingest_log_text(payload.text, reset);
        log_state.path = payload.path;
        log_state.offset = Number(payload.end);
        const status = by_id("local_log_status");
        if (status) status.textContent = `${payload.path} · latest first`;
      } catch (error) {
        const status = by_id("local_log_status");
        if (status) status.textContent = `Log read failed: ${error.message}`;
      } finally {
        log_state.loading = false;
      }
    };

    const install_logs_pane = () => {
      if (by_id("run_logs_pane")) return;
      const pane = document.createElement("section");
      pane.id = "run_logs_pane";
      pane.className = "run-logs-pane";
      pane.hidden = true;
      const toolbar = document.createElement("header");
      toolbar.className = "local-log-toolbar";
      const status = document.createElement("div");
      status.id = "local_log_status";
      status.className = "local-log-status";
      status.textContent = "Select Logs to load train.log.";
      const controls = document.createElement("div");
      controls.className = "local-log-font-controls";
      const smaller = document.createElement("button");
      smaller.type = "button";
      smaller.className = "local-log-font-button";
      smaller.textContent = "A↓";
      smaller.title = "Decrease log font size";
      smaller.setAttribute("aria-label", smaller.title);
      const larger = document.createElement("button");
      larger.type = "button";
      larger.className = "local-log-font-button";
      larger.textContent = "A↑";
      larger.title = "Increase log font size";
      larger.setAttribute("aria-label", larger.title);
      const size = document.createElement("span");
      size.id = "local_log_font_value";
      size.className = "local-log-font-value";
      controls.append(smaller, larger, size);
      toolbar.append(status, controls);
      const output = document.createElement("div");
      output.id = "local_log_output";
      output.className = "local-log-output";
      output.setAttribute("role", "log");
      output.setAttribute("aria-live", "off");
      output.addEventListener("scroll", () => {
        log_state.user_scrolled = output.scrollTop > 12;
      });
      smaller.addEventListener("click", () => set_log_font_size(log_font_size - 1));
      larger.addEventListener("click", () => set_log_font_size(log_font_size + 1));
      pane.append(toolbar, output);
      by_id("charts_pane")?.appendChild(pane);
      set_log_font_size(log_font_size);
    };

    const base_local_apply_detail_tab_logs = local_apply_detail_tab;
    local_apply_detail_tab = function() {
      base_local_apply_detail_tab_logs();
      const logs = Boolean(app.current_run_id) && local_active_detail_tab === "logs";
      const pane = by_id("run_logs_pane");
      if (pane) pane.hidden = !logs;
      if (logs) {
        const blank = by_id("run_blank_detail_pane");
        if (blank) blank.hidden = true;
        refresh_log();
      }
    };

    const base_render_run_heading_logs_modes = render_run_heading;
    render_run_heading = function() {
      base_render_run_heading_logs_modes();
      sync_mode_buttons();
      if (local_active_detail_tab === "logs" && log_state.run_id !== app.current_run_id) {
        reset_log_state();
        refresh_log();
      }
    };

    install_logs_pane();
    install_heatmap_mode_buttons();
    install_trajectory_mode_buttons();
    sync_mode_buttons();
    setInterval(refresh_log, log_poll_interval_ms);

    const style = document.createElement("style");
    style.textContent = `
      /* Make the literal 1px-row newest tick fit wholly inside the SVG. */
      .chart-card[data-chart="heatmap"] .ytick:last-of-type text {
        transform: translate(-7px, 7px) !important;
      }

      .explicit-mode-group {
        height: 29px;
        display: inline-flex;
        align-items: stretch;
        flex: 0 0 auto;
        border: 1px solid #cfd3d8;
        border-radius: 5px;
        overflow: hidden;
        background: #f6f7f8;
      }
      .explicit-mode-button {
        min-width: 38px;
        padding: 0 9px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 5px;
        border: 0;
        border-left: 1px solid #d9dde1;
        background: #f6f7f8;
        color: #505862;
        font: inherit;
        font-size: 11px;
        cursor: pointer;
      }
      .explicit-mode-button:first-child { border-left: 0; }
      .explicit-mode-button:hover:not(:disabled):not(.active) { background: #eceff1; }
      .explicit-mode-button.active {
        background: #d7dbe0;
        color: #242a31;
        font-weight: 700;
        box-shadow: inset 0 1px 3px rgba(28,34,40,.24);
      }
      .explicit-mode-button:disabled { opacity: .45; cursor: default; }
      .explicit-trajectory-modes .explicit-mode-button { min-width: 67px; padding: 0 7px; }
      .explicit-scale-glyph {
        position: relative;
        width: 14px;
        height: 13px;
        display: inline-block;
        flex: 0 0 14px;
      }
      .explicit-scale-glyph i {
        position: absolute;
        right: 1px;
        left: 1px;
        height: 1.5px;
        border-radius: 2px;
        background: currentColor;
      }
      .explicit-scale-glyph.linear i:nth-child(1) { top: 2px; }
      .explicit-scale-glyph.linear i:nth-child(2) { top: 6px; }
      .explicit-scale-glyph.linear i:nth-child(3) { top: 10px; }
      .explicit-scale-glyph.log i:nth-child(1) { top: 1px; }
      .explicit-scale-glyph.log i:nth-child(2) { top: 4px; }
      .explicit-scale-glyph.log i:nth-child(3) { top: 11px; }

      .run-logs-pane {
        flex: 1 1 auto;
        min-height: 0;
        display: flex;
        flex-direction: column;
        overflow: hidden;
        background: #000000;
      }
      .run-logs-pane[hidden] { display: none !important; }
      .local-log-toolbar {
        flex: 0 0 38px;
        min-height: 38px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding: 0 12px;
        border-bottom: 1px solid #cfd3d8;
        background: #f3f4f5;
        color: #59616b;
      }
      .local-log-status {
        min-width: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        font-size: 11px;
      }
      .local-log-font-controls {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        flex: 0 0 auto;
      }
      .local-log-font-button {
        width: 36px;
        height: 28px;
        border: 1px solid #c8cdd3;
        border-radius: 4px;
        background: #f9fafb;
        color: #3f4751;
        font-size: 14px;
        line-height: 1;
        cursor: pointer;
      }
      .local-log-font-button:hover { background: #e9ecef; }
      .local-log-font-button:active {
        background: #d7dbe0;
        box-shadow: inset 0 1px 2px rgba(28,34,40,.22);
      }
      .local-log-font-value {
        width: 35px;
        color: #6a727c;
        text-align: right;
        font-size: 10px;
      }
      .local-log-output {
        flex: 1 1 auto;
        min-height: 0;
        overflow: auto;
        padding: 9px 12px 18px;
        background: #000000 !important;
        color: #e6e6e6;
        font-family: "DejaVu Sans Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-variant-ligatures: none;
        line-height: 1.28;
        tab-size: 8;
        scrollbar-color: #747b83 #111111;
      }
      .local-log-line {
        min-height: 1.28em;
        margin: 0;
        white-space: pre;
        overflow: visible;
      }
    `;
    document.head.appendChild(style);

    local_apply_detail_tab();
  }, 25);
});
// ^^^ THOG
