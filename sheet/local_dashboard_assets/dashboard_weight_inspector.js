// vvv THOG
"use strict";

// Data operations deliberately use source figures, never Plotly's smoothed/logged
// y values. The same functions serve the inspector and dependency-free regressions.
const instra_weight_inspection = (() => {
  const number_or_null = value => {
    if (value === null || value === undefined || value === "") return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  };
  const step = trace => {
    const meta = trace?.meta || {};
    for (const value of [meta.instra_workspace_optimizer_update, meta.instra_dense_optimizer_update, meta.instra_thog_optimizer_update]) {
      const number = number_or_null(value);
      if (Number.isInteger(number) && number >= 0) return number;
    }
    const match = `${trace?.name || ""} ${trace?.hovertemplate || ""}`.match(/(?:^|[^A-Za-z0-9])(?:step\s+|U)(\d+)(?:\D|$)/i);
    return match ? Number(match[1]) : null;
  };
  const run_id = (trace, fallback = "") => String(trace?.meta?.instra_workspace_run_id || fallback);
  const scalar = trace => {
    const meta = trace?.meta || {};
    const model = number_or_null(meta.instra_weight_model_feature);
    const branch = number_or_null(meta.instra_weight_intermediate_feature);
    if (model !== null && branch !== null) return `m${model}:b${branch}`;
    return String(meta.instra_dense_scalar_id || meta.instra_thog_scalar_id || "unknown");
  };
  const series_key = (trace, fallback = "") => JSON.stringify([run_id(trace, fallback), step(trace), scalar(trace)]);
  const anchor = trace => trace?.meta?.instra_top_axis_anchor === true;
  const overlay = trace => trace?.meta?.instra_thog_executed_overlay === true;
  const coupling = (trace, chart_name) => {
    const meta = trace?.meta || {};
    let input = number_or_null(meta.instra_weight_model_feature);
    let output = number_or_null(meta.instra_weight_intermediate_feature);
    if (input !== null && output !== null) {
      if (["attn_out_head_N", "mlp_down"].includes(chart_name)) [input, output] = [output, input];
    } else {
      const match = scalar(trace).match(/^r(\d+)_c(\d+)$/);
      input = match ? Number(match[2]) : null;
      output = match ? Number(match[1]) : null;
    }
    return {input, output};
  };

  // One logical curve per run. A THOG executed-marker overlay is part of that
  // curve, not a second snapshot. Legacy multi-scalar runs choose one stable scalar.
  const latest_curves = traces => {
    const latest = new Map();
    for (const trace of traces) {
      if (anchor(trace)) continue;
      const update = step(trace);
      const id = run_id(trace);
      if (update !== null) latest.set(id, Math.max(update, latest.get(id) ?? -Infinity));
    }
    const selected = new Map();
    for (const trace of traces) {
      if (anchor(trace) || overlay(trace) || step(trace) !== latest.get(run_id(trace))) continue;
      const id = run_id(trace);
      const prior = selected.get(id);
      if (!prior || scalar(trace).localeCompare(scalar(prior), undefined, {numeric: true}) < 0) selected.set(id, trace);
    }
    const used_main = new Set();
    const used_overlay = new Set();
    return traces.filter(trace => {
      if (anchor(trace)) return true;
      const id = run_id(trace);
      const chosen = selected.get(id);
      if (trace === chosen && !used_main.has(id)) { used_main.add(id); return true; }
      if (!chosen || !overlay(trace) || series_key(trace) !== series_key(chosen) || used_overlay.has(id)) return false;
      used_overlay.add(id);
      return true;
    });
  };

  const build_table = (source, visible, runs, chart_name, fallback = "") => {
    const allowed = new Set((visible?.data || []).filter(trace => !anchor(trace)).map(trace => series_key(trace, fallback)));
    const selected = new Map();
    for (const trace of source?.data || []) {
      if (anchor(trace) || step(trace) === null) continue;
      const key = series_key(trace, fallback);
      if (!allowed.has(key)) continue;
      const prior = selected.get(key);
      const exact = Array.isArray(trace.meta?.instra_thog_integer_x) && Array.isArray(trace.meta?.instra_thog_integer_y);
      const priority = exact ? 3 : overlay(trace) ? 2 : 1;
      if (!prior || priority > prior.priority) selected.set(key, {trace, priority});
    }
    const row_map = new Map();
    const layers = new Set();
    const known_runs = new Map(runs.map(run => [String(run.id), run]));
    for (const {trace, priority} of selected.values()) {
      const id = run_id(trace, fallback);
      if (!known_runs.has(id)) continue;
      const pair = coupling(trace, chart_name);
      const update = step(trace);
      const pair_key = pair.input === null || pair.output === null ? scalar(trace) : `${pair.input}:${pair.output}`;
      const row_key = `${update}|${pair_key}`;
      if (!row_map.has(row_key)) row_map.set(row_key, {key: row_key, step: update, ...pair, values: new Map()});
      const row = row_map.get(row_key);
      const xs = priority === 3 ? trace.meta.instra_thog_integer_x : trace.x;
      const ys = priority === 3 ? trace.meta.instra_thog_integer_y : trace.y;
      if (!Array.isArray(xs) || !Array.isArray(ys) || xs.length !== ys.length) continue;
      const values = row.values.get(id) || new Map();
      for (let index = 0; index < xs.length; index += 1) {
        const layer = number_or_null(xs[index]);
        // Only executed layer positions belong in the raw weight table.
        if (!Number.isInteger(layer) || layer < 1) continue;
        layers.add(layer);
        if (!values.has(layer)) values.set(layer, number_or_null(ys[index]));
      }
      row.values.set(id, values);
    }
    const rows = [...row_map.values()].filter(row => [...row.values.values()].some(values => values.size))
      .sort((left, right) => left.step - right.step || (left.input ?? -1) - (right.input ?? -1) || (left.output ?? -1) - (right.output ?? -1));
    const columns = [...layers].sort((left, right) => left - right).flatMap(layer => runs.map(run => ({...run, layer, key: `${layer}|${run.id}`})));
    const multiple_pairs = new Set(rows.map(row => `${row.input}:${row.output}`)).size > 1;
    return {rows, columns, runs, multiple_pairs};
  };
  const value_at = (model, row, column) => {
    const col = model.columns[column];
    return col ? model.rows[row]?.values.get(String(col.id))?.get(col.layer) ?? null : null;
  };
  const precision = value => Number.isInteger(value) && value >= 0 && value <= 12 ? value : 4;
  const format = (value, places, missing = "—") => {
    if (value === null || value === undefined || !Number.isFinite(value)) return missing;
    const formatted = value.toFixed(precision(places));
    return /^-0(?:\.0+)?$/.test(formatted) ? formatted.slice(1) : formatted;
  };
  const bounds = selection => ({
    row_min: Math.min(selection.anchor.row, selection.focus.row), row_max: Math.max(selection.anchor.row, selection.focus.row),
    col_min: Math.min(selection.anchor.column, selection.focus.column), col_max: Math.max(selection.anchor.column, selection.focus.column),
  });
  const tsv = (model, selection, places) => {
    if (!selection) return "";
    const range = bounds(selection);
    const rows = [];
    for (let row = Math.max(0, range.row_min); row <= Math.min(model.rows.length - 1, range.row_max); row += 1) {
      const cells = [];
      for (let column = Math.max(0, range.col_min); column <= Math.min(model.columns.length - 1, range.col_max); column += 1) {
        cells.push(format(value_at(model, row, column), places, ""));
      }
      rows.push(cells.join("\t"));
    }
    return rows.join("\n");
  };
  const window_range = (scroll, size, header, cell, count) => ({
    start: Math.max(0, Math.floor(scroll / cell) - 1),
    end: Math.min(count, Math.ceil((scroll + Math.max(0, size - header)) / cell) + 1),
  });
  return Object.freeze({step, scalar, series_key, coupling, latest_curves, build_table, value_at, precision, format, bounds, tsv, window_range});
})();

if (typeof module !== "undefined" && module.exports) module.exports = instra_weight_inspection;
if (typeof window !== "undefined") window.addEventListener("load", () => {
  const install = () => {
    if (window.__instra_weight_inspector) return true;
    if (!window.__instra_further_weight_owner || !window.__instra_weight_group_settings) return false;
    const data = instra_weight_inspection;
    const chart_names = new Set(depth_weight_chart_names);
    let active = null;
    let button = null;
    const row_height = 28;
    const header_height = 56;
    const row_width = 110;
    const scope = () => app.workspace_mode ? "workspace" : `run:${app.current_run_id}`;
    const get_precision = () => data.precision(window.__instra_weight_group_settings.group_settings_for_scope(scope())?.inspection_precision);
    const context = () => window.__instra_weight_stability_final.context_key();
    const element = (tag, class_name, text) => {
      const node = document.createElement(tag);
      node.className = class_name;
      if (text !== undefined) node.textContent = text;
      return node;
    };
    const action = (label, title) => {
      const node = element("button", "weight-step-button", label);
      node.type = "button";
      node.title = title || label;
      return node;
    };
    const position = (node, left, top, width, height = row_height) => {
      Object.assign(node.style, {left: `${left}px`, top: `${top}px`, width: `${width}px`, height: `${height}px`});
    };
    const clamp_cell = (row, column) => ({
      row: Math.max(0, Math.min(active.model.rows.length - 1, row)),
      column: Math.max(0, Math.min(active.model.columns.length - 1, column)),
    });
    const update_metadata = () => {
      if (!active) return;
      const state = active;
      const cell = state.selection?.focus;
      const row = cell && state.model.rows[cell.row];
      const col = cell && state.model.columns[cell.column];
      state.metadata.textContent = row && col
        ? `Step ${row.step} · Layer ${col.layer} · Input coupling ${row.input ?? "unknown"} · Output coupling ${row.output ?? "unknown"} · ${col.name}`
        : "No recorded weights in this view.";
      state.metadata.title = state.metadata.textContent;
      const range = state.selection && data.bounds(state.selection);
      state.status.textContent = range
        ? `${range.row_max - range.row_min + 1} × ${range.col_max - range.col_min + 1} selected · ${state.places} decimal places`
        : "";
      state.copy.disabled = !range;
    };
    const paint = () => {
      if (!active) return;
      const state = active;
      state.frame = null;
      const {grid, canvas, model, cell_width} = state;
      const rows = data.window_range(grid.scrollTop, grid.clientHeight, header_height, row_height, model.rows.length);
      const cols = data.window_range(grid.scrollLeft, grid.clientWidth, row_width, cell_width, model.columns.length);
      const fragment = document.createDocumentFragment();
      const selected = state.selection && data.bounds(state.selection);
      for (let row_index = rows.start; row_index < rows.end; row_index += 1) {
        const row = model.rows[row_index];
        const top = header_height + row_index * row_height;
        for (let col_index = cols.start; col_index < cols.end; col_index += 1) {
          const col = model.columns[col_index];
          const value = data.value_at(model, row_index, col_index);
          const cell = element("div", "weight-inspection-cell", data.format(value, state.places));
          cell.setAttribute("role", "gridcell");
          cell.setAttribute("aria-rowindex", String(row_index + 3));
          cell.setAttribute("aria-colindex", String(col_index + 2));
          cell.id = `weight_inspection_cell_${row_index}_${col_index}`;
          cell.dataset.row = String(row_index);
          cell.dataset.column = String(col_index);
          cell.title = `Step ${row.step}, layer ${col.layer}, input ${row.input ?? "unknown"}, output ${row.output ?? "unknown"}, ${col.name}: ${value ?? "not recorded"}`;
          cell.style.setProperty("--run-colour", col.colour);
          const is_selected = Boolean(selected && row_index >= selected.row_min && row_index <= selected.row_max && col_index >= selected.col_min && col_index <= selected.col_max);
          cell.setAttribute("aria-selected", String(is_selected));
          cell.classList.toggle("selected", is_selected);
          cell.classList.toggle("active", state.selection?.focus.row === row_index && state.selection?.focus.column === col_index);
          cell.classList.toggle("layer-start", col_index % model.runs.length === 0);
          position(cell, row_width + col_index * cell_width, top, cell_width);
          fragment.appendChild(cell);
        }
        const header = element("div", "weight-inspection-row-header", model.multiple_pairs ? `${row.step} · ${row.input ?? "?"}→${row.output ?? "?"}` : String(row.step));
        header.setAttribute("role", "rowheader");
        header.setAttribute("aria-rowindex", String(row_index + 3));
        position(header, grid.scrollLeft, top, row_width);
        fragment.appendChild(header);
      }
      for (let col_index = cols.start; col_index < cols.end; col_index += 1) {
        const col = model.columns[col_index];
        const header = element("div", "weight-inspection-col-header", col.name);
        header.title = `Layer ${col.layer} · ${col.name}`;
        header.style.setProperty("--run-colour", col.colour);
        header.setAttribute("role", "columnheader");
        header.setAttribute("aria-colindex", String(col_index + 2));
        position(header, row_width + col_index * cell_width, grid.scrollTop + row_height, cell_width);
        fragment.appendChild(header);
      }
      const first_layer = Math.floor(cols.start / Math.max(1, model.runs.length));
      const last_layer = Math.ceil(cols.end / Math.max(1, model.runs.length));
      for (let layer_index = first_layer; layer_index < last_layer; layer_index += 1) {
        const col = model.columns[layer_index * model.runs.length];
        if (!col) continue;
        const header = element("div", "weight-inspection-layer-header", `Layer ${col.layer}`);
        header.setAttribute("role", "columnheader");
        position(header, row_width + layer_index * model.runs.length * cell_width, grid.scrollTop, model.runs.length * cell_width);
        fragment.appendChild(header);
      }
      const corner = element("div", "weight-inspection-corner", model.multiple_pairs ? "Step · input→output" : "Step ↓ / Layer →");
      position(corner, grid.scrollLeft, grid.scrollTop, row_width, header_height);
      fragment.appendChild(corner);
      canvas.replaceChildren(fragment);
      const focus = state.selection?.focus;
      if (focus && focus.row >= rows.start && focus.row < rows.end && focus.column >= cols.start && focus.column < cols.end) {
        grid.setAttribute("aria-activedescendant", `weight_inspection_cell_${focus.row}_${focus.column}`);
      } else grid.removeAttribute("aria-activedescendant");
      update_metadata();
    };
    const schedule_paint = () => {
      if (active && !active.frame) active.frame = requestAnimationFrame(paint);
    };
    const stop_drag = () => {
      if (!active) return;
      active.dragging = false;
      if (active.drag_frame) cancelAnimationFrame(active.drag_frame);
      active.drag_frame = null;
    };
    const close = (focus_button = true) => {
      if (!active) return;
      stop_drag();
      if (active.frame) cancelAnimationFrame(active.frame);
      active.resize_observer?.disconnect();
      active.card.classList.remove("weight-inspection-open");
      active.panel.remove();
      active = null;
      button?.setAttribute("aria-pressed", "false");
      if (focus_button) button?.focus();
      resize_visible_plots();
    };
    const raw_figure = chart_name => {
      const source = app.figures?.depth?.[chart_name];
      if (source) return source;
      const mount = by_id(`${chart_name}_plot`);
      if (mount?.dataset?.instraWeightContext === context()
          && mount.dataset.instraWeightView === window.__instra_weight_step_filter?.signature?.()) {
        return mount.__instraWeightFigure || null;
      }
      return null;
    };
    const update = () => {
      if (!active) return;
      const state = active;
      if (state.chart_name !== app.maximized_chart || state.context !== context()) { close(false); return; }
      const source = raw_figure(state.chart_name);
      const view = window.__instra_weight_step_filter?.signature?.();
      const pair = window.__instra_weight_viewer_selection?.selection?.();
      const places = get_precision();
      const runs = (app.workspace_mode ? window.__instra_workspace.visible_runs() : [current_run()].filter(Boolean))
        .map(run => ({id: String(run_identifier(run)), name: String(run.artifact_name || run.run_name || run_identifier(run)), colour: colour_for_run(String(run_identifier(run)))}));
      const signature = JSON.stringify([view, pair, places, runs]);
      if (state.source === source && state.signature === signature) return;
      const prior_rows = state.model?.rows || [];
      const prior_cols = state.model?.columns || [];
      const prior_selection = state.selection;
      state.model = data.build_table(source, source ? prepare_figure(source, state.chart_name) : null, runs, state.chart_name, String(app.current_run_id || ""));
      state.source = source;
      state.signature = signature;
      state.places = places;
      state.cell_width = Math.max(112, (places + 8) * 8);
      state.selection = null;
      if (state.model.rows.length && state.model.columns.length) {
        const remap = cell => {
          const row = state.model.rows.findIndex(value => value.key === prior_rows[cell.row]?.key);
          const column = state.model.columns.findIndex(value => value.key === prior_cols[cell.column]?.key);
          return clamp_cell(Math.max(0, row), Math.max(0, column));
        };
        state.selection = prior_selection
          ? {anchor: remap(prior_selection.anchor), focus: remap(prior_selection.focus)}
          : {anchor: {row: 0, column: 0}, focus: {row: 0, column: 0}};
      }
      state.canvas.style.width = `${row_width + state.model.columns.length * state.cell_width}px`;
      state.canvas.style.height = `${header_height + state.model.rows.length * row_height}px`;
      state.grid.setAttribute("aria-rowcount", String(state.model.rows.length + 2));
      state.grid.setAttribute("aria-colcount", String(state.model.columns.length + 1));
      state.empty.hidden = state.model.rows.length > 0;
      paint();
    };
    const copy_selection = async () => {
      if (!active?.selection) return;
      const state = active;
      const text = data.tsv(state.model, state.selection, state.places);
      try {
        if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(text);
        else throw new Error("Clipboard API unavailable");
      } catch (_error) {
        const input = element("textarea", "weight-inspection-copy-fallback");
        input.value = text;
        state.panel.appendChild(input);
        input.select();
        const copied = document.execCommand("copy");
        input.remove();
        state.grid.focus({preventScroll: true});
        if (!copied) { state.status.textContent = "Copy unavailable: select cells and press Ctrl+C."; return; }
      }
      if (active === state) state.status.textContent = "Copied as tab-separated values.";
    };
    const reveal_focus = () => {
      const {grid, selection, cell_width} = active;
      const left = selection.focus.column * cell_width;
      const top = selection.focus.row * row_height;
      if (left < grid.scrollLeft) grid.scrollLeft = left;
      else if (left + cell_width > grid.scrollLeft + grid.clientWidth - row_width) grid.scrollLeft = left + cell_width - grid.clientWidth + row_width;
      if (top < grid.scrollTop) grid.scrollTop = top;
      else if (top + row_height > grid.scrollTop + grid.clientHeight - header_height) grid.scrollTop = top + row_height - grid.clientHeight + header_height;
      schedule_paint();
    };
    const point_cell = (x, y) => {
      const rect = active.grid.getBoundingClientRect();
      return clamp_cell(Math.floor((y - rect.top + active.grid.scrollTop - header_height) / row_height), Math.floor((x - rect.left + active.grid.scrollLeft - row_width) / active.cell_width));
    };
    const drag_tick = () => {
      if (!active?.dragging) return;
      const state = active;
      const rect = state.grid.getBoundingClientRect();
      const edge_speed = (value, low, high) => value < low + 20 ? -Math.min(28, low + 20 - value) : value > high - 20 ? Math.min(28, value - high + 20) : 0;
      state.grid.scrollLeft += edge_speed(state.pointer_x, rect.left + row_width, rect.right - 15);
      state.grid.scrollTop += edge_speed(state.pointer_y, rect.top + header_height, rect.bottom - 15);
      state.selection.focus = point_cell(state.pointer_x, state.pointer_y);
      schedule_paint();
      state.drag_frame = requestAnimationFrame(drag_tick);
    };
    const open = () => {
      if (!chart_names.has(app.maximized_chart)) return;
      if (active) { close(); return; }
      const chart_name = app.maximized_chart;
      const card = document.querySelector(`.chart-card[data-chart="${chart_name}"]`);
      if (!card) return;
      const panel = element("section", "weight-inspection-panel");
      panel.setAttribute("aria-label", "Inspect recorded weights");
      const toolbar = element("div", "weight-inspection-toolbar");
      const back = action("←", "Back to chart");
      back.setAttribute("aria-label", "Back to chart");
      const metadata = element("div", "weight-inspection-metadata");
      const copy = action("copy", "Copy selected rectangle as tab-separated values");
      toolbar.append(back, metadata, copy);
      const hint = element("div", "weight-inspection-hint", "Recorded layer weights · drag or Shift-click a rectangle · Shift+arrows to extend · Ctrl+C to copy · — = not recorded");
      const grid = element("div", "weight-inspection-grid");
      grid.tabIndex = 0;
      grid.setAttribute("role", "grid");
      grid.setAttribute("aria-label", "Raw weights: step rows and layer columns");
      grid.setAttribute("aria-readonly", "true");
      grid.setAttribute("aria-multiselectable", "true");
      const canvas = element("div", "weight-inspection-canvas");
      grid.appendChild(canvas);
      const empty = element("div", "weight-inspection-empty", "No recorded layer weights for the selected coupling and steps.");
      const status = element("div", "weight-inspection-status");
      status.setAttribute("role", "status");
      panel.append(toolbar, hint, grid, empty, status);
      card.appendChild(panel);
      card.classList.add("weight-inspection-open");
      active = {chart_name, card, panel, grid, canvas, metadata, empty, status, copy, context: context(), model: {rows: [], columns: []}, selection: null};
      button.setAttribute("aria-pressed", "true");
      back.addEventListener("click", () => close());
      copy.addEventListener("click", copy_selection);
      grid.addEventListener("scroll", schedule_paint, {passive: true});
      grid.addEventListener("pointerdown", event => {
        const cell = event.target.closest(".weight-inspection-cell");
        if (!cell || event.button !== 0) return;
        event.preventDefault();
        grid.focus({preventScroll: true});
        const focus = {row: Number(cell.dataset.row), column: Number(cell.dataset.column)};
        active.selection = {anchor: event.shiftKey && active.selection ? active.selection.anchor : focus, focus};
        active.dragging = true;
        active.pointer_x = event.clientX;
        active.pointer_y = event.clientY;
        grid.setPointerCapture(event.pointerId);
        schedule_paint();
        active.drag_frame = requestAnimationFrame(drag_tick);
      });
      grid.addEventListener("pointermove", event => {
        if (!active?.dragging) return;
        active.pointer_x = event.clientX;
        active.pointer_y = event.clientY;
        active.selection.focus = point_cell(event.clientX, event.clientY);
        schedule_paint();
      });
      for (const name of ["pointerup", "pointercancel", "lostpointercapture"]) grid.addEventListener(name, stop_drag);
      grid.addEventListener("copy", event => {
        if (!active?.selection || !event.clipboardData) return;
        event.preventDefault();
        event.clipboardData.setData("text/plain", data.tsv(active.model, active.selection, active.places));
        active.status.textContent = "Copied as tab-separated values.";
      });
      grid.addEventListener("keydown", event => {
        if (event.key === "Escape") { event.preventDefault(); event.stopPropagation(); close(); return; }
        if (!active?.selection) return;
        const command = event.ctrlKey || event.metaKey;
        if (command && event.key.toLowerCase() === "a") {
          event.preventDefault();
          active.selection = {anchor: {row: 0, column: 0}, focus: {row: active.model.rows.length - 1, column: active.model.columns.length - 1}};
          schedule_paint(); return;
        }
        const moves = {ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1], PageUp: [-Math.max(1, Math.floor((grid.clientHeight - header_height) / row_height)), 0], PageDown: [Math.max(1, Math.floor((grid.clientHeight - header_height) / row_height)), 0]};
        let {row, column} = active.selection.focus;
        if (moves[event.key]) { row += moves[event.key][0]; column += moves[event.key][1]; }
        else if (event.key === "Home") { column = 0; if (command) row = 0; }
        else if (event.key === "End") { column = active.model.columns.length - 1; if (command) row = active.model.rows.length - 1; }
        else return;
        event.preventDefault();
        const focus = clamp_cell(row, column);
        active.selection = {anchor: event.shiftKey ? active.selection.anchor : focus, focus};
        reveal_focus();
      });
      if (typeof ResizeObserver === "function") {
        active.resize_observer = new ResizeObserver(schedule_paint);
        active.resize_observer.observe(grid);
      }
      update();
      grid.focus({preventScroll: true});
    };
    const sync_button = () => {
      const controls = by_id("weight_step_group_controls");
      if (!button && controls) {
        button = action("inspect weights", "Inspect raw recorded weights");
        button.id = "weight_inspect_button";
        button.setAttribute("aria-pressed", "false");
        controls.appendChild(button);
        button.addEventListener("click", event => { event.stopPropagation(); open(); });
      }
      if (button) button.hidden = !chart_names.has(app.maximized_chart);
      update();
    };
    const original_prepare = prepare_figure;
    prepare_figure = function(figure, chart_name) {
      const prepared = original_prepare(figure, chart_name);
      if (chart_names.has(chart_name) && window.__instra_weight_stability_final.mode() === "latest") {
        prepared.data = data.latest_curves(prepared.data || []);
      }
      return prepared;
    };
    const original_toggle = toggle_maximized_chart;
    toggle_maximized_chart = function(...args) { const result = original_toggle(...args); sync_button(); return result; };
    const original_restore = restore_maximized_chart;
    restore_maximized_chart = function(...args) { close(false); const result = original_restore(...args); sync_button(); return result; };
    const original_heading = render_run_heading;
    render_run_heading = function(...args) { const result = original_heading(...args); sync_button(); return result; };
    const original_render = render_figures;
    render_figures = async function(...args) { const result = await original_render(...args); sync_button(); return result; };
    const original_plot = render_plot;
    render_plot = async function(...args) { const result = await original_plot(...args); if (active && args[2] === active.chart_name) update(); return result; };
    // Capture Escape ahead of the chart-wide handler: return to the chart first.
    window.addEventListener("keydown", event => {
      if (event.key !== "Escape" || !active || !by_id("chart_settings_overlay")?.hidden) return;
      event.preventDefault(); event.stopImmediatePropagation(); close();
    }, true);
    const style = element("style", "");
    style.textContent = `
      #weight_inspect_button { margin-left: 28px; }
      #weight_inspect_button[hidden] { display: none !important; }
      #coefficients_chart_group.thog2-tab-maximized-group > .chart-group-header { height: auto; min-height: 35px; flex-wrap: wrap; padding: 4px 0; row-gap: 6px; }
      #coefficients_chart_group.thog2-tab-maximized-group .weight-step-group-controls { flex-wrap: wrap; row-gap: 6px; }
      .weight-inspection-open > .plot-shell { visibility: hidden; pointer-events: none; }
      .weight-inspection-panel { position: absolute; inset: 52px 0 0; z-index: 6; display: flex; flex-direction: column; min-width: 0; background: #fff; }
      .weight-inspection-toolbar { display: flex; align-items: center; gap: 12px; padding: 8px 12px; border-bottom: 1px solid #dce2e9; }
      .weight-inspection-metadata { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 12px; }
      .weight-inspection-hint, .weight-inspection-status { padding: 5px 12px; font-size: 11px; color: #536173; }
      .weight-inspection-status { min-height: 25px; border-top: 1px solid #dce2e9; }
      .weight-inspection-grid { flex: 1; min-height: 0; min-width: 0; overflow: scroll; overscroll-behavior: contain; outline-offset: -2px; user-select: none; touch-action: none; }
      .weight-inspection-canvas { position: relative; min-width: 100%; min-height: 100%; font: 12px ui-monospace, monospace; }
      .weight-inspection-cell, .weight-inspection-row-header, .weight-inspection-col-header, .weight-inspection-layer-header, .weight-inspection-corner { position: absolute; box-sizing: border-box; padding: 5px 8px; border-right: 1px solid #e0e5eb; border-bottom: 1px solid #e0e5eb; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
      .weight-inspection-cell { text-align: right; color: #172536; border-top: 2px solid var(--run-colour); background: color-mix(in srgb, var(--run-colour) 7%, white); cursor: cell; }
      .weight-inspection-cell.layer-start { border-left: 2px solid #bcc8d5; }
      .weight-inspection-cell.selected { background: #d6e9ff; box-shadow: inset 0 0 0 1px #6f9bc9; }
      .weight-inspection-cell.active { outline: 2px solid #176dad; outline-offset: -2px; }
      .weight-inspection-row-header { z-index: 2; text-align: right; background: #f2f5f9; font-weight: 600; }
      .weight-inspection-col-header { z-index: 3; background: #f4f7fa; border-bottom: 3px solid var(--run-colour); font-size: 10px; }
      .weight-inspection-layer-header { z-index: 3; background: #eaf0f6; text-align: center; font-weight: 600; }
      .weight-inspection-corner { z-index: 4; background: #eaf0f6; white-space: normal; font-size: 10px; }
      .weight-inspection-empty { padding: 12px; color: #536173; }
      .weight-inspection-empty[hidden] { display: none; }
      .weight-inspection-copy-fallback { position: fixed; left: -10000px; top: 0; }
    `;
    document.head.appendChild(style);
    window.__instra_weight_inspector = Object.freeze({open, close, sync: sync_button});
    sync_button();
    return true;
  };
  let attempts = 0;
  const when_ready = () => { attempts += 1; if (!install() && attempts < 240) setTimeout(when_ready, 25); };
  when_ready();
});
// ^^^ THOG
