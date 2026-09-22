// vvv THOG
"use strict";

window.addEventListener("load", () => {
  window.setTimeout(() => {
    const light_run_palette = Object.freeze([
      "#B9A7EA", "#F3AD8A", "#FFD68A", "#BDDB93", "#91C4A3", "#79BEB5", "#99D5E4", "#9FB9E9",
      "#EFB8EB", "#D88FBA", "#D3A798", "#C9CDD0", "#B1A9DD", "#E8A89F", "#EFC09B", "#EAD184",
      "#ACD18F", "#8FC1A0", "#86C6CF", "#93CED8", "#8FC4DF", "#9EAFE1", "#AAA0D6", "#D3A4DF",
      "#E496BD", "#C794AE", "#C5A89F", "#B8C0C6", "#C9BDEB", "#F0C2AE", "#C5DCAA", "#A9D4D0",
    ]);
    if (Array.isArray(default_palette)) default_palette.splice(0, default_palette.length, ...light_run_palette);

    const trash_svg = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16"/><path d="M9 7V4h6v3"/><path d="M6.5 7l1 13h9l1-13"/><path d="M10 11v5M14 11v5"/></svg>';
    const style = document.createElement("style");
    style.id = "instra-sep21-repairs-style";
    style.textContent = `
      .charts-toolbar { height:62px !important; min-height:62px !important; max-height:62px !important; }
      .selected-run-copy { min-height:44px; display:flex; flex-direction:column; justify-content:center; overflow:hidden; }
      .run-subtitle { height:14px; min-height:14px; flex-wrap:nowrap !important; overflow:hidden; white-space:nowrap; }
      #colour_swatches .colour-swatch, .processing-operations-colour-swatches .colour-swatch {
        width:100% !important; height:auto !important; min-height:0 !important; aspect-ratio:1 / 1 !important;
      }
      .runs-trash-button svg, .file-delete-button svg {
        width:18px !important; height:18px !important; fill:none; stroke:currentColor;
        stroke-width:1.9; stroke-linecap:round; stroke-linejoin:round;
      }
      .runs-trash-button:not(:disabled), .file-delete-button:not(:disabled) { color:#7B1F24 !important; }
      .file-delete-button {
        flex:0 0 30px; width:30px; height:28px; align-self:center; margin-left:-8px; margin-right:auto;
        display:inline-flex; align-items:center; justify-content:center; padding:0 !important;
      }
      #training_chart_group { display:none !important; }
      #training_throughput_card {
        width:auto; max-width:none; min-width:300px; height:365px; flex:1 1 calc(33.333% - 10px);
      }
      #training_throughput_card > .chart-card-header { position:relative; }
      #training_throughput_card > .chart-card-header > .chart-heading-copy { max-width:calc(50% - 24px); }
      #training_throughput_card .plot-mount { width:100%; min-width:0; height:100%; min-height:0; }
      #training_throughput_card .plot-shell { overflow:hidden; }
      /* vvv THOG Runs/Workspace train charts are one full-width vertical stack, with loss before throughput. */
      .local-metric-group[data-metric-group="train"] > .local-metric-grid {
        flex-direction:column !important; flex-wrap:nowrap !important; align-items:stretch !important;
      }
      .local-metric-group[data-metric-group="train"] > .local-metric-grid > .chart-card:not(.maximized) {
        width:100% !important; max-width:none !important; min-width:300px !important; flex:0 0 365px !important;
      }
      .local-metric-group[data-metric-group="train"] > .local-metric-grid > .chart-card { order:2; }
      .local-metric-group[data-metric-group="train"] > .local-metric-grid > .instra-train-loss-card { order:0; }
      .local-metric-group[data-metric-group="train"] > .local-metric-grid > #training_throughput_card { order:1; }
      /* ^^^ THOG */
      @media (max-width:1100px) { #training_throughput_card { flex-basis:calc(50% - 10px); } }
      @media (max-width:760px) { #training_throughput_card { flex-basis:100%; } }
      #processing_update_timing_timeline_card > .chart-card-header { min-height:78px !important; height:78px !important; }
      #processing_chart_group.maximized > #processing_grid.is-maximized > #processing_update_timing_timeline_card.maximized {
        flex:0 0 var(--instra-host-timeline-height,360px) !important;
        height:var(--instra-host-timeline-height,360px) !important;
        min-height:260px !important; max-height:min(72vh,520px) !important; align-self:flex-start;
      }
      .instra-columns-popover {
        position:fixed; z-index:180; width:250px; max-height:min(560px,calc(100vh - 20px)); overflow:auto;
        padding:10px; border:1px solid #cfd2d8; border-radius:6px; background:#fff;
        box-shadow:0 8px 28px rgba(23,25,30,.22);
      }
      .instra-columns-popover[hidden] { display:none !important; }
      .instra-columns-popover strong { display:block; margin:0 0 7px; color:#3f4650; font-size:11px; }
      .instra-columns-list { display:grid; grid-template-columns:1fr 1fr; gap:5px 9px; }
      .instra-columns-list label { display:flex; align-items:center; gap:6px; min-width:0; color:#4f5761; font-size:10px; }
      .instra-columns-list input { margin:0; }
    `;
    document.head.appendChild(style);

    function install_run_colour_inputs() {
      const hex = by_id("colour_hex");
      const channels = [by_id("colour_r"), by_id("colour_g"), by_id("colour_b")];
      if (!hex || channels.some(input => !input) || hex.dataset.instraSep21Editable === "true") return;
      hex.dataset.instraSep21Editable = "true";
      const apply_hex = () => {
        const rgb = hex_to_rgb(hex.value);
        if (rgb) set_picker_colour(rgb);
      };
      hex.addEventListener("input", apply_hex);
      hex.addEventListener("keydown", event => {
        if (event.key !== "Enter") return;
        event.preventDefault();
        apply_hex();
      });
      const apply_rgb = () => {
        const values = channels.map(input => Number(input.value));
        if (values.some(value => !Number.isInteger(value) || value < 0 || value > 255)) return;
        set_picker_colour(values);
      };
      for (const input of channels) {
        input.readOnly = false;
        input.type = "number";
        input.min = "0";
        input.max = "255";
        input.step = "1";
        input.addEventListener("input", apply_rgb);
        input.addEventListener("change", apply_rgb);
      }
      by_id("reset_colour")?.remove();
    }

    const column_storage_key = "thog2_local_hidden_run_columns_v1";
    const column_labels = Object.freeze({
      select:"select", visibility:"visible", steps:"logged", duration:"t", state:"state", name:"name",
      wandb:"W&B ID", host:"host", gpu:"GPU", preset:"p", optimizer:"OPT", gb:"GB", layers:"L",
      depth_order:"P", premat:"premat", parms:"PARMS", equiv:"EQUIV", warmup:"w", context:"C",
      d_model:"D", heads:"H", grad_accum:"A", activation_checkpointing:"S", learning_rate:"c",
      min_learning_rate:"f", probe_start:"P_s", probe_end:"P_e", curve_start:"C_s", curve_end:"C_e",
      capture_period:"C_p", updated:"updated", menu:"menu",
    });

    function hidden_columns() {
      const stored = load_json(column_storage_key, []);
      return new Set(Array.isArray(stored) ? stored.map(String) : []);
    }

    function apply_column_visibility() {
      const hidden = hidden_columns();
      const table = document.querySelector(".runs-table");
      const headers = [...(table?.querySelectorAll("thead [data-instra-column-key]") || [])];
      for (const element of table?.querySelectorAll("[data-instra-column-key]") || []) {
        element.hidden = hidden.has(String(element.dataset.instraColumnKey || ""));
      }
      let fixed_width = 0;
      for (const header of headers) {
        if (header.hidden || header.dataset.instraColumnKey === "name") continue;
        const inline = Number.parseFloat(header.style.width || "");
        fixed_width += Number.isFinite(inline) ? inline : 56;
      }
      if (table) {
        const name_width = Number(localStorage.getItem("thog2_local_run_name_column_width")) || 390;
        table.style.setProperty("min-width", `${Math.max(260, fixed_width + (hidden.has("name") ? 0 : name_width))}px`, "important");
        const count = headers.filter(header => !header.hidden).length;
        table.querySelectorAll("tbody .group-row td").forEach(cell => { cell.colSpan = Math.max(1, count); });
      }
    }

    function ensure_columns_control() {
      const toolbar = document.querySelector(".runs-pane-header .toolbar");
      if (!toolbar) return;
      let button = by_id("instra_columns_button");
      if (!button) {
        button = document.createElement("button");
        button.id = "instra_columns_button";
        button.type = "button";
        button.className = "toolbar-button";
        button.textContent = "Columns";
        button.setAttribute("aria-expanded", "false");
        const group = by_id("group_button");
        toolbar.insertBefore(button, group || toolbar.firstChild);
      }
      let popover = by_id("instra_columns_popover");
      if (!popover) {
        popover = document.createElement("div");
        popover.id = "instra_columns_popover";
        popover.className = "instra-columns-popover";
        popover.hidden = true;
        popover.innerHTML = '<strong>Visible columns</strong><div class="instra-columns-list"></div>';
        document.body.appendChild(popover);
      }
      const list = popover.querySelector(".instra-columns-list");
      const headers = [...document.querySelectorAll(".runs-table thead [data-instra-column-key]")];
      const keys = headers.map(header => String(header.dataset.instraColumnKey || "")).filter(Boolean);
      const existing = [...list.querySelectorAll("input")].map(input => input.value);
      if (JSON.stringify(keys) !== JSON.stringify(existing)) {
        list.replaceChildren();
        const hidden = hidden_columns();
        for (const key of keys) {
          const label = document.createElement("label");
          const checkbox = document.createElement("input");
          checkbox.type = "checkbox";
          checkbox.value = key;
          checkbox.checked = !hidden.has(key);
          checkbox.addEventListener("change", () => {
            const next = hidden_columns();
            if (checkbox.checked) next.delete(key);
            else next.add(key);
            save_json(column_storage_key, [...next]);
            apply_column_visibility();
          });
          label.append(checkbox, document.createTextNode(column_labels[key] || key));
          list.appendChild(label);
        }
      }
      if (button.dataset.instraColumnsBound !== "true") {
        button.dataset.instraColumnsBound = "true";
        button.addEventListener("click", event => {
          event.stopPropagation();
          popover.hidden = !popover.hidden;
          button.setAttribute("aria-expanded", String(!popover.hidden));
          if (!popover.hidden) {
            const rect = button.getBoundingClientRect();
            popover.style.left = `${Math.max(8, Math.min(rect.left, window.innerWidth - 258))}px`;
            popover.style.top = `${Math.min(window.innerHeight - popover.offsetHeight - 8, rect.bottom + 6)}px`;
          }
        });
        popover.addEventListener("click", event => event.stopPropagation());
        document.addEventListener("click", () => {
          popover.hidden = true;
          button.setAttribute("aria-expanded", "false");
        });
      }
      apply_column_visibility();
    }

    function runs_signature() {
      const active_tick = (app.runs || []).some(run => is_active_run_state(run.run_state))
        ? Math.floor(Date.now() / 30000)
        : 0;
      return JSON.stringify({
        active_tick,
        current:String(app.current_run_id || ""), workspace:Boolean(app.workspace_mode), page:app.current_page,
        page_size:app.page_size, search:by_id("run_search")?.value || "",
        filter:by_id("state_filter")?.value || "all", sort:by_id("run_sort")?.value || "created",
        descending:app.sort_descending, grouped:app.group_by_host,
        selected:[...(app.selected || [])].map(String).sort(), visibility:app.visibility, colours:app.colours,
        pairs:[...(app.processing_paired_run_ids || [])].map(String).sort(),
        unmatched:[...(app.processing_unmatched_nsys_run_ids || [])].map(String).sort(),
        runs:(app.runs || []).map(run => [
          String(run_identifier(run)), run.revision || null, run.run_state || null, run.heartbeat_at || null,
          run.updated_at || null, run.maximum_update ?? null, run.chart_maximum_update ?? null,
          run.configuration?.premat ?? null, run.configuration?.premat_target_matrix ?? null,
        ]),
      });
    }

    const render_runs_before_sep21 = render_runs;
    let last_runs_signature = "";
    render_runs = function(...args) {
      const signature = runs_signature();
      if (signature === last_runs_signature) return;
      last_runs_signature = signature;
      const result = render_runs_before_sep21.apply(this, args);
      ensure_columns_control();
      apply_column_visibility();
      return result;
    };

    function standardize_throughput_plot(mount) {
      if (!mount || mount.dataset.plotReady !== "true") return;
      Plotly.relayout(mount, {
        "margin.l":58, "margin.r":18, "margin.t":16, "margin.b":48,
        "xaxis.title.text":"steps", "xaxis.title.standoff":8, "xaxis.type":"linear", "xaxis.automargin":true,
        "legend.orientation":"v", "legend.x":1, "legend.xanchor":"right",
        "legend.y":1, "legend.yanchor":"top", "legend.bgcolor":"rgba(255,255,255,.72)",
        "legend.font.size":9, "font.size":10,
      }).catch(() => {});
    }

    // vvv THOG chart discovery is asynchronous, so classify the loss card whenever train data or throughput changes.
    function layout_train_charts() {
      const train = document.querySelector('.local-metric-group[data-metric-group="train"]');
      const grid = train?.querySelector(".local-metric-grid");
      if (!grid) return;
      for (const card of grid.querySelectorAll(":scope > .chart-card")) {
        const identity = `${card.dataset.metricChartId || ""} ${card.querySelector(".chart-heading-copy h2")?.textContent || ""}`.toLowerCase();
        card.classList.toggle("instra-train-loss-card", /(^|[^a-z])loss([^a-z]|$)/.test(identity));
      }
    }
    // ^^^ THOG

    function place_training_throughput() {
      const legacy_group = by_id("training_chart_group");
      const card = by_id("training_throughput_card");
      const train = document.querySelector('.local-metric-group[data-metric-group="train"]');
      const grid = train?.querySelector(".local-metric-grid");
      if (!legacy_group || !card || !grid) return;
      card.classList.add("instra-training-throughput-card");
      card.dataset.metricGroup = "train";
      if (card.parentElement !== grid) grid.appendChild(card);
      legacy_group.hidden = true;
      const title = card.querySelector(".chart-heading-copy h2");
      if (title) title.textContent = "tokens throughput";
      const count = train.querySelector(".local-metric-group-count");
      if (count) count.textContent = String(grid.querySelectorAll(":scope > .chart-card").length);
      standardize_throughput_plot(by_id("training_throughput_plot"));
      layout_train_charts();                                                                                                                               // <<< THOG keep loss above the full-width throughput card
    }

    // vvv THOG in Runs view, ordinary training throughput does not make the Processing evidence group available
    const processing_sync_visibility_before_sep21 = processing_sync_visibility;
    processing_sync_visibility = function(...args) {
      const result = processing_sync_visibility_before_sep21.apply(this, args);
      const group = by_id("processing_chart_group");
      if (group && app.workspace_mode !== true) {
        const capture_available = Boolean(
          processing_view.trace_available
          || processing_view.resource_available
          || processing_view.compatibility_available
          || processing_view.timing_available
        );
        group.hidden = !(processing_view.charts_tab_visible && capture_available);
      }
      return result;
    };
    // ^^^ THOG

    function configure_throughput() {
      processing_view.throughput_axis_mode = "step";
      localStorage.setItem("thog2_processing_throughput_x_mode", "step");
      const select = by_id("processing_throughput_x_mode");
      if (select) {
        select.value = "step";
        select.hidden = true;
      }
      for (const [card_selector, mount_id] of [
        ['[data-chart="processing_throughput"]', "processing_throughput_plot"],
        ['[data-chart="training_throughput"]', "training_throughput_plot"],
      ]) {
        const heading = document.querySelector(`${card_selector} .chart-heading-copy h2`);
        if (heading) heading.textContent = "tokens throughput";
        standardize_throughput_plot(by_id(mount_id));
      }
      chart_titles.processing_throughput = "tokens throughput";
      chart_titles.training_throughput = "tokens throughput";
      for (const name of ["processing_throughput", "training_throughput"]) {
        app.dynamic_chart_metadata[name] = {
          x_source:"Optimizer update", x_label:"steps", y_source:"Training throughput",
          y_label:"tokens / second", default_x_axis_mode:"step", available_x_axis_modes:["step"],
        };
      }
      place_training_throughput();
    }

    const processing_render_throughput_before_sep21 = processing_render_throughput;
    processing_render_throughput = async function(payload) {
      processing_view.throughput_axis_mode = "step";
      const result = await processing_render_throughput_before_sep21(payload);
      configure_throughput();
      return result;
    };

    const metric_groups = window.__thog2_metric_groups;
    if (metric_groups?.refresh) {
      const refresh_metric_groups_before_sep21 = metric_groups.refresh;
      metric_groups.refresh = async function(...args) {
        const result = await refresh_metric_groups_before_sep21.apply(this, args);
        place_training_throughput();
        layout_train_charts();
        return result;
      };
    }
    const charts_scroll = by_id("charts_scroll");
    if (charts_scroll) {
      const metric_group_observer = new MutationObserver(records => {
        for (const record of records) {
          for (const removed of record.removedNodes) {
            if (!(removed instanceof Element)) continue;
            const card = removed.matches?.("#training_throughput_card")
              ? removed
              : removed.querySelector?.("#training_throughput_card");
            const legacy_grid = by_id("training_grid");
            if (card && legacy_grid && !card.isConnected) legacy_grid.appendChild(card);
          }
        }
        queueMicrotask(place_training_throughput);
      });
      metric_group_observer.observe(charts_scroll, {childList:true});
    }

    function timing_fingerprint() {
      const runs = typeof processing_update_timing_ordered_runs === "function"
        ? processing_update_timing_ordered_runs()
        : [];
      return JSON.stringify({
        workspace:Boolean(app.workspace_mode), current:String(app.current_run_id || ""),
        runs:runs.map(run => [
          String(run_identifier(run)), run.revision || run.data_updated_at || run.updated_at || null,
          run.run_state || null, is_visible(run_identifier(run)),
        ]),
      });
    }

    function configure_host_timeline(reset_range = false) {
      const card = by_id("processing_update_timing_timeline_card");
      const mount = by_id("processing_update_timing_timeline_plot");
      const entries = processing_view.timing_entries || [];
      if (!card || mount?.dataset.plotReady !== "true" || !entries.length) return;
      const steps = entries.map(entry => Number(entry.timing?.optimizer_update)).filter(Number.isFinite);
      const unique_steps = [...new Set(steps)];
      const step_text = unique_steps.length === 1
        ? `captured optimizer step ${unique_steps[0]}`
        : `captured optimizer steps ${unique_steps.join(", ")}`;
      const detail = card.querySelector(".chart-heading-copy p");
      if (detail) detail.textContent = (
        `${step_text} · elapsed host time from update entry (ms) · `
        + "--premat_instra__full_step_timing_capture_and_chart enable · "
        + `--premat_instra__full_step_timing_capture_and_chart_capture step ${unique_steps[0] ?? 5}`
      );
      const lane_names = entries.map(entry => {
        const label = typeof processing_update_timing_experiment_label === "function"
          ? processing_update_timing_experiment_label(entry)
          : String(entry.run?.artifact_name || entry.run_id || "run");
        return `${label} · step ${entry.timing?.optimizer_update ?? "—"}`;
      });
      const maximum_host_ms = Math.max(0, ...entries.map(entry => Number(
        entry.timing?.official_update_ms ?? entry.timing?.host_update_ms ?? 0
      )));
      const bar_indices = (mount.data || []).flatMap((trace, index) => trace?.type === "bar" ? [index] : []);
      if (bar_indices.length) Plotly.restyle(mount, {width:0.62}, bar_indices).catch(() => {});
      const layout = {
        "margin.l":Math.min(420, Math.max(160, 28 + Math.max(...lane_names.map(name => name.length)) * 5.7)),
        "margin.t":24, "margin.b":54,
        "xaxis.title.text":"elapsed host time from update entry (ms)",
        "xaxis.ticksuffix":" ms", "xaxis.uirevision":null,
        "yaxis.tickmode":"array", "yaxis.tickvals":entries.map((_entry, index) => index),
        "yaxis.ticktext":lane_names, "yaxis.range":[-0.45, Math.max(0.55, entries.length - 0.55)],
      };
      if (reset_range && maximum_host_ms > 0) {
        layout["xaxis.autorange"] = false;
        layout["xaxis.range"] = [0, maximum_host_ms * 1.01];
      }
      Plotly.relayout(mount, layout).catch(() => {});
      const viewport_height = Math.max(260, Number(by_id("charts_scroll")?.clientHeight || 520));
      const useful_height = Math.min(viewport_height - 16, Math.max(280, 154 + entries.length * 54));
      card.style.setProperty("--instra-host-timeline-height", `${useful_height}px`);
    }

    const processing_render_update_timing_before_sep21 = processing_render_update_timing;
    let last_timing_fingerprint = "";
    let timing_in_flight = null;
    let timing_queued_force = false;
    processing_render_update_timing = function(force = false) {
      const fingerprint = timing_fingerprint();
      if (!force && fingerprint === last_timing_fingerprint) return Promise.resolve();
      if (timing_in_flight) {
        timing_queued_force = timing_queued_force || force || fingerprint !== last_timing_fingerprint;
        return timing_in_flight;
      }
      timing_in_flight = (async () => {
        await processing_render_update_timing_before_sep21(force);
        last_timing_fingerprint = fingerprint;
        configure_host_timeline(true);
      })().finally(() => {
        timing_in_flight = null;
        if (timing_queued_force) {
          timing_queued_force = false;
          queueMicrotask(() => processing_render_update_timing(true));
        }
      });
      return timing_in_flight;
    };

    function apply_compatibility_bar_height() {
      const card = by_id("processing_compatibility_card");
      const mount = by_id("processing_resource_compatibility_plot");
      if (!card || mount?.dataset.plotReady !== "true" || !Array.isArray(mount.data)) return;
      const maximized = card.classList.contains("maximized");
      const scatter = mount.data.flatMap((trace, index) => trace.type === "scattergl" ? [index] : []);
      const bars = mount.data.flatMap((trace, index) => trace.type === "bar" ? [index] : []);
      if (scatter.length) Plotly.restyle(mount, {"line.width":maximized ? 17 : 8}, scatter).catch(() => {});
      if (bars.length) Plotly.restyle(mount, {width:maximized ? 0.36 : 0.23}, bars).catch(() => {});
    }

    const processing_render_before_sep21 = processing_render;
    processing_render = async function(...args) {
      const result = await processing_render_before_sep21.apply(this, args);
      configure_throughput();
      configure_host_timeline(true);
      apply_compatibility_bar_height();
      return result;
    };

    function ensure_file_trash_control() {
      const button = by_id("delete_selected_files");
      const bar = document.querySelector(".file-source-bar");
      const tabs = bar?.querySelector(".file-source-tabs");
      if (!button || !bar || !tabs) return;
      button.innerHTML = trash_svg;
      button.classList.add("file-delete-button");
      button.title = button.disabled ? "Select local files with the checkboxes to delete them" : button.title;
      if (button.parentElement !== bar || button.previousElementSibling !== tabs) tabs.insertAdjacentElement("afterend", button);
    }

    const render_files_before_sep21 = render_files;
    render_files = function(...args) {
      const result = render_files_before_sep21.apply(this, args);
      ensure_file_trash_control();
      return result;
    };

    document.addEventListener("click", event => {
      const eye = event.target.closest?.(".eye-button");
      const row = eye?.closest?.("tr[data-run-id]");
      if (row) {
        const run_id = String(row.dataset.runId || "");
        const run = (app.runs || []).find(candidate => String(run_identifier(candidate)) === run_id);
        const artifact = String(run?.artifact_name || run?.run_name || "").toUpperCase();
        const nsys = artifact.includes("_NSYS_") || artifact.includes("NSYS_PREMAT");
        if (nsys) {
          queueMicrotask(() => {
            if (!is_visible(run_id)) return;
            if (String(app.current_run_id || "") !== run_id) select_run(run_id, {manual:true});
            if (typeof processing_refresh === "function") processing_refresh(true);
          });
        }
      }
      if (event.target.closest?.("#processing_update_timing_timeline_card .maximize-button")) {
        for (const delay of [40, 180, 360]) window.setTimeout(() => configure_host_timeline(true), delay);
      }
      if (event.target.closest?.("#processing_compatibility_card .maximize-button")) {
        for (const delay of [40, 220, 400]) window.setTimeout(apply_compatibility_bar_height, delay);
      }
    }, true);

    install_run_colour_inputs();
    ensure_columns_control();
    ensure_file_trash_control();
    configure_throughput();
    configure_host_timeline(true);
    apply_compatibility_bar_height();
    last_runs_signature = "";
    render_runs();

    window.instra_sep21_repair_test_hooks = Object.freeze({
      light_run_palette, apply_column_visibility, ensure_columns_control, place_training_throughput,
      configure_throughput, configure_host_timeline, apply_compatibility_bar_height, runs_signature,
      timing_fingerprint, layout_train_charts,
    });
  }, 250);
});
// ^^^ THOG
