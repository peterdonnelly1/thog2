// vvv THOG
"use strict";

(function install_sep20_integrated_repairs() {
  const primary_row = Object.freeze([
    "#FF0000", "#00FF00", "#0000FF", "#00FFFF",
    "#FF00FF", "#FFFF00", "#FFFFFF", "#000000",
  ]);

  function hsl_hex(hue, saturation, lightness) {
    const s = saturation / 100;
    const l = lightness / 100;
    const chroma = (1 - Math.abs(2 * l - 1)) * s;
    const section = ((hue % 360) + 360) % 360 / 60;
    const second = chroma * (1 - Math.abs(section % 2 - 1));
    const pairs = [
      [chroma, second, 0], [second, chroma, 0], [0, chroma, second],
      [0, second, chroma], [second, 0, chroma], [chroma, 0, second],
    ];
    const [red, green, blue] = pairs[Math.floor(section) % 6];
    const match = l - chroma / 2;
    return `#${[red, green, blue].map(value => (
      Math.round((value + match) * 255).toString(16).padStart(2, "0")
    )).join("").toUpperCase()}`;
  }

  function tonal_rows(row_count, saturation, lightness, phase = 0) {
    return Array.from({length: row_count * 8}, (_value, index) => {
      const scrambled = (index * 7) % (row_count * 8);
      return hsl_hex(phase + scrambled * (360 / (row_count * 8)), saturation, lightness);
    });
  }

  // Seventeen rows exactly: six dark, five light, five very light, one exact primary row.
  const shared_palette = Object.freeze([
    ...tonal_rows(6, 76, 33, 0),
    ...tonal_rows(5, 68, 65, 4),
    ...tonal_rows(5, 58, 86, 8),
    ...primary_row,
  ]);
  window.instra_colour_palette = shared_palette;

  const style = document.createElement("style");
  style.id = "instra-sep20-integrated-repairs-style";
  style.textContent = `
    body { -webkit-user-select:text; user-select:text; }
    input, textarea, select, button, a, .panel-resizer, .modebar,
    .plot-container, .svg-container { -webkit-user-select:auto; user-select:auto; }
    #colour_popover, .processing-operations-colour-popover {
      max-height:calc(100vh - 16px) !important;
      overflow:hidden !important;
    }
    #colour_swatches, .processing-operations-colour-swatches {
      grid-template-columns:repeat(8,minmax(0,1fr)) !important;
      gap:2px !important;
      margin-top:7px !important;
    }
    #colour_swatches .colour-swatch, .processing-operations-colour-swatches .colour-swatch {
      aspect-ratio:auto !important;
      height:10px !important;
      min-height:10px !important;
    }
    #processing_update_timing_timeline_card.maximized > .chart-card-header {
      display:flex !important; visibility:visible !important; opacity:1 !important;
      flex:0 0 auto !important; min-height:58px !important; height:auto !important;
      position:relative !important; overflow:visible !important;
    }
    #processing_update_timing_timeline_plot .legendpoints path,
    #processing_update_timing_timeline_plot .legendpoints rect {
      transform:scale(2.15); transform-box:fill-box; transform-origin:center;
    }
    #processing_compatibility_card.maximized .processing-compatibility-key i {
      width:26px !important; height:26px !important; flex:0 0 26px !important;
    }
    .file-select-box { width:14px; height:14px; margin:0 8px 0 0; vertical-align:middle; }
    .file-delete-button { color:#a12a2a; }
    .file-delete-button:disabled { opacity:.38; }
  `;
  document.head.appendChild(style);

  function install_shared_palette() {
    window.instra_colour_palette = shared_palette;
    const container = by_id("colour_swatches");
    if (!container) return;
    container.replaceChildren();
    for (const colour of shared_palette) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "colour-swatch";
      button.style.background = colour;
      button.title = colour;
      button.setAttribute("aria-label", `Choose run colour ${colour}`);
      button.addEventListener("click", () => set_picker_colour(hex_to_rgb(colour)));
      container.appendChild(button);
    }
    container.dataset.instraPaletteVersion = "categorical-v4-17x8";
  }

  const open_colour_picker_before_sep20 = open_colour_picker;
  open_colour_picker = function(run_id, anchor) {
    const result = open_colour_picker_before_sep20(run_id, anchor);
    install_shared_palette();
    return result;
  };

  function remove_artifacts_tab() {
    document.querySelectorAll('[data-detail-tab="artifacts"]').forEach(element => element.remove());
  }

  function ensure_runs_trash_icon() {
    const button = by_id("delete_selected_runs");
    if (!button) return;
    button.hidden = false;
    if (!button.querySelector("svg")) {
      button.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16"/><path d="M9 7V4h6v3"/><path d="M6.5 7l1 13h9l1-13"/><path d="M10 11v5M14 11v5"/></svg>';
    }
  }

  function clone_live_figure(mount) {
    if (!mount || mount.dataset.plotReady !== "true" || !Array.isArray(mount.data)) return null;
    return {data: mount.data, layout: mount.layout || {}};
  }

  const figure_for_chart_before_sep20 = figure_for_chart;
  figure_for_chart = function(chart_name) {
    const known = figure_for_chart_before_sep20(chart_name);
    if (known) return known;
    const card = document.querySelector(`.chart-card[data-chart="${CSS.escape(chart_name)}"]`);
    return clone_live_figure(card?.querySelector(".plot-mount"));
  };

  function register_processing_figures() {
    document.querySelectorAll("#charts_scroll .processing-card[data-chart]").forEach(card => {
      const chart_name = String(card.dataset.chart || "");
      const mount = card.querySelector(".plot-mount");
      const figure = clone_live_figure(mount);
      if (!chart_name || !figure) return;
      app.dynamic_chart_figures[chart_name] = figure;
      app.dynamic_chart_metadata[chart_name] = app.dynamic_chart_metadata[chart_name] || {
        x_source:"Chart x axis", x_label:String(figure.layout?.xaxis?.title?.text || "x"),
        y_source:"Chart y axis", y_label:String(figure.layout?.yaxis?.title?.text || "value"),
        default_x_axis_mode:null, available_x_axis_modes:[],
      };
    });
  }

  async function register_throughput_figure(payload) {
    const mount = by_id("processing_throughput_plot");
    if (!mount || mount.dataset.plotReady !== "true") return;
    const runs = processing_throughput_workspace_runs();
    const rows_by_run = new Map(await Promise.all(runs.map(async run => [
      String(run_identifier(run)), await processing_throughput_rows_for_run(run, payload),
    ])));
    const variants = {
      step: row => Number(row.optimizer_update),
      relative_wall: row => Number(row.relative_wall_seconds) / 3600,
      relative_process: row => Number(row.process_time_seconds) / 3600,
      wall_time: row => Number(row.wall_time) * 1000,
    };
    const available = Object.keys(variants).filter(mode => [...rows_by_run.values()].every(rows => (
      rows.length && rows.every(row => Number.isFinite(variants[mode](row)))
    )));
    for (const trace of mount.data || []) {
      const rows = rows_by_run.get(String(trace.meta?.instra_workspace_run_id || "")) || [];
      trace.thog2_x_variants = Object.fromEntries(available.map(mode => [
        mode, rows.map(variants[mode]),
      ]));
    }
    app.dynamic_chart_figures.processing_throughput = clone_live_figure(mount);
    app.dynamic_chart_metadata.processing_throughput = {
      x_source:"Optimizer update or recorded time", x_label:"optimizer update",
      y_source:"Training throughput", y_label:"tokens / second",
      default_x_axis_mode:"step", available_x_axis_modes:available,
    };
  }

  const processing_render_before_sep20 = processing_render;
  processing_render = async function(payload, trace_available) {
    await processing_render_before_sep20(payload, trace_available);
    register_processing_figures();
    await register_throughput_figure(payload || {});
  };

  if (typeof processing_render_update_timing === "function") {
    const processing_render_update_timing_before_sep20 = processing_render_update_timing;
    processing_render_update_timing = async function(...args) {
      const result = await processing_render_update_timing_before_sep20.apply(this, args);
      configure_host_timeline();
      register_processing_figures();
      return result;
    };
  }

  function configure_host_timeline() {
    const card = by_id("processing_update_timing_timeline_card");
    const mount = by_id("processing_update_timing_timeline_plot");
    const detail = card?.querySelector(".chart-heading-copy p");
    if (detail) detail.textContent = (
      "Host wall-clock phases within each complete optimizer update, aligned to update entry. "
      + "The x-axis is elapsed milliseconds from that entry (not GPU time); each lane is one run."
    );
    if (!card || mount?.dataset.plotReady !== "true") return;
    const maximized = card.classList.contains("maximized");
    if (!maximized && mount.clientHeight > 0) {
      mount.dataset.hostTimelineReferenceHeight = String(mount.clientHeight);
    }
    const reference_height = Number(mount.dataset.hostTimelineReferenceHeight || 300);
    const current_height = Math.max(1, Number(mount.clientHeight || reference_height));
    const width = maximized
      ? Math.max(0.08, Math.min(0.62, 0.62 * reference_height / current_height))
      : 0.62;
    const bar_indices = (mount.data || []).flatMap((trace, index) => trace?.type === "bar" ? [index] : []);
    if (bar_indices.length) Plotly.restyle(mount, {width}, bar_indices).catch(() => {});
    const longest = Math.max(0, ...(processing_view.timing_entries || []).map(entry => (
      processing_update_timing_run_name(entry).length
    )));
    Plotly.relayout(mount, {
      "margin.l":Math.min(420, Math.max(190, 16 + longest * 5.7)),
      "margin.t":32,
      "yaxis.automargin":true,
      "legend.itemwidth":40,
    }).catch(() => {});
  }

  function halve_maximized_compatibility_bars() {
    const card = by_id("processing_compatibility_card");
    const mount = by_id("processing_resource_compatibility_plot");
    if (!card?.classList.contains("maximized") || mount?.dataset.plotReady !== "true") return;
    const scatter = mount.data.flatMap((trace, index) => trace.type === "scattergl" ? [index] : []);
    const bars = mount.data.flatMap((trace, index) => trace.type === "bar" ? [index] : []);
    if (scatter.length) Plotly.restyle(mount, {"line.width":17}, scatter).catch(() => {});
    if (bars.length) Plotly.restyle(mount, {width:0.36}, bars).catch(() => {});
  }

  document.addEventListener("click", event => {
    if (event.target.closest?.("#processing_update_timing_timeline_card .maximize-button")) {
      for (const delay of [40, 180]) setTimeout(configure_host_timeline, delay);
    }
    if (event.target.closest?.("#processing_compatibility_card .maximize-button")) {
      for (const delay of [40, 180]) setTimeout(halve_maximized_compatibility_bars, delay);
    }
  }, true);

  const selected_local_files = new Set();
  let selected_local_file_context = "";

  function update_file_delete_button() {
    const button = by_id("delete_selected_files");
    if (!button) return;
    button.disabled = app.file_source !== "instra" || selected_local_files.size === 0;
    button.title = selected_local_files.size
      ? `Delete ${selected_local_files.size} selected file${selected_local_files.size === 1 ? "" : "s"} from its real run-directory location`
      : "Select local files with the checkboxes to delete them";
    const select_all = by_id("select_all_local_files");
    const boxes = [...document.querySelectorAll("#files_body .file-select-box")];
    if (select_all) {
      select_all.disabled = app.file_source !== "instra" || boxes.length === 0;
      select_all.checked = boxes.length > 0 && boxes.every(box => box.checked);
      select_all.indeterminate = boxes.some(box => box.checked) && !select_all.checked;
    }
  }

  function ensure_file_delete_button() {
    const actions = document.querySelector(".file-source-actions");
    if (!actions) return;
    const heading = document.querySelector(".files-table thead th:first-child");
    if (heading && !by_id("select_all_local_files")) {
      const select_all = document.createElement("input");
      select_all.id = "select_all_local_files";
      select_all.type = "checkbox";
      select_all.className = "file-select-box";
      select_all.setAttribute("aria-label", "Select all local files in this folder");
      select_all.addEventListener("change", () => {
        for (const box of document.querySelectorAll("#files_body .file-select-box")) {
          box.checked = select_all.checked;
          const path = box.closest("tr")?.dataset.filePath;
          if (!path) continue;
          if (select_all.checked) selected_local_files.add(path);
          else selected_local_files.delete(path);
        }
        update_file_delete_button();
      });
      heading.prepend(select_all);
    }
    let button = by_id("delete_selected_files");
    if (!button) {
      button = document.createElement("button");
      button.id = "delete_selected_files";
      button.type = "button";
      button.className = "file-refresh-button file-delete-button";
      button.textContent = "⌫";
      button.setAttribute("aria-label", "Delete selected local files");
      button.addEventListener("click", async () => {
        const paths = [...selected_local_files];
        if (!paths.length || !window.confirm(`Delete ${paths.length} selected file${paths.length === 1 ? "" : "s"} from disk?`)) return;
        const failures = [];
        for (const path of paths) {
          const query = new URLSearchParams({run:app.current_run_id, path});
          try {
            const response = await fetch(`/api/local-file?${query}`, {method:"DELETE"});
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || `${response.status} ${response.statusText}`);
            selected_local_files.delete(path);
          } catch (error) {
            failures.push(`${path}: ${error.message}`);
          }
        }
        await refresh_files(true);
        show_toast(failures.length ? `File deletion failed for ${failures.length} item(s).` : `Deleted ${paths.length} file(s) from disk.`);
      });
      actions.insertBefore(button, by_id("refresh_files"));
    }
    update_file_delete_button();
  }

  const append_file_row_before_sep20 = append_file_row;
  append_file_row = function(body, entry) {
    append_file_row_before_sep20(body, entry);
    const row = body.lastElementChild;
    if (!row || app.file_source !== "instra" || entry.kind !== "file") return;
    row.dataset.filePath = entry.path;
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.className = "file-select-box";
    checkbox.checked = selected_local_files.has(entry.path);
    checkbox.setAttribute("aria-label", `Select ${entry.name} for deletion`);
    checkbox.addEventListener("click", event => event.stopPropagation());
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) selected_local_files.add(entry.path);
      else selected_local_files.delete(entry.path);
      update_file_delete_button();
    });
    const name_cell = row.firstElementChild;
    if (name_cell) name_cell.insertBefore(checkbox, name_cell.firstElementChild);
  };

  const render_files_before_sep20 = render_files;
  render_files = function() {
    const context = `${app.current_run_id || ""}\u0000${app.file_source || ""}\u0000${app.file_path || ""}`;
    if (context !== selected_local_file_context) {
      selected_local_file_context = context;
      selected_local_files.clear();
    }
    const result = render_files_before_sep20();
    ensure_file_delete_button();
    const root = app.file_payload?.root_path;
    if (root && app.file_source === "instra") {
      const status = by_id("file_source_status");
      if (status) {
        status.title = `Real location: ${root}`;
        if (!status.textContent.includes(root)) status.textContent += ` · ${root}`;
      }
    }
    return result;
  };

  const render_runs_before_sep20 = render_runs;
  render_runs = function(...args) {
    const result = render_runs_before_sep20.apply(this, args);
    ensure_runs_trash_icon();
    return result;
  };

  window.addEventListener("load", () => {
    setTimeout(() => {
      install_shared_palette();
      remove_artifacts_tab();
      ensure_runs_trash_icon();
      ensure_file_delete_button();
      configure_host_timeline();
      register_processing_figures();
    }, 0);
  });

  window.instra_sep20_repair_test_hooks = Object.freeze({
    shared_palette,
    register_processing_figures,
    configure_host_timeline,
    halve_maximized_compatibility_bars,
  });
})();
// ^^^ THOG
