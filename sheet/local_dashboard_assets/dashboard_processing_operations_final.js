// vvv THOG
"use strict";

(function install_processing_operations_final() {
  const style_id = "instra-processing-operations-final-style";
  if (!document.getElementById(style_id)) {
    const style = document.createElement("style");
    style.id = style_id;
    style.textContent = `
      .processing-header {
        align-items:center !important;
        gap:8px !important;
      }
      .processing-header-copy {
        min-width:0 !important;
        overflow:hidden !important;
        text-overflow:ellipsis !important;
        white-space:nowrap !important;
      }
      .processing-downloads {
        margin-left:auto !important;
        display:flex !important;
        align-items:center !important;
        align-self:center !important;
        flex-wrap:wrap !important;
        gap:4px !important;
        white-space:nowrap !important;
      }
      #processing_timeline_card .chart-card-header {
        position:relative !important;
        align-items:flex-start !important;
      }
      #processing_timeline_card .chart-card-actions {
        align-self:flex-start !important;
        margin-top:0 !important;
        display:flex !important;
        align-items:center !important;
        gap:5px !important;
      }
      #processing_timeline_card .processing-operations-key {
        position:absolute;
        z-index:5;
        top:60px;
        right:34px;
        left:88px;
        min-height:42px;
        max-height:54px;
        display:flex;
        align-content:center;
        align-items:center;
        flex-wrap:wrap;
        gap:4px 12px;
        overflow-y:auto;
        padding:4px 0;
        background:#fff;
      }
      #processing_timeline_card .processing-plot-shell {
        inset:118px 0 0 0 !important;
      }
      #processing_timeline_card:not(.maximized) {
        min-height:300px !important;
        height:300px !important;
      }
      #processing_timeline_card:not(.maximized) .processing-plot-shell {
        min-height:182px !important;
        height:auto !important;
        flex:1 1 auto !important;
      }
      .processing-operations-key-item {
        display:inline-flex;
        align-items:center;
        gap:4px;
        color:#434a54;
        font-size:9px;
        white-space:nowrap;
      }
      .processing-operations-key-item.is-hidden {
        opacity:.42;
      }
      .processing-operations-key-colour,
      .processing-operations-key-pattern {
        width:13px;
        height:13px;
        flex:0 0 13px;
        padding:0;
        border:1px solid rgba(0,0,0,.20);
        border-radius:2px;
      }
      .processing-operations-key-colour {
        cursor:pointer;
      }
      .processing-operations-key-pattern {
        background:repeating-linear-gradient(135deg,#8a8f97 0 4px,#fff 4px 8px);
      }
      .processing-operations-key-toggle {
        padding:0;
        border:0;
        background:transparent;
        color:inherit;
        font:inherit;
        cursor:pointer;
      }
      .processing-operations-colour-popover {
        position:fixed;
        z-index:160;
        width:282px;
        max-height:min(360px,calc(100vh - 16px));
        overflow:auto;
        padding:10px;
        border:1px solid #cfd2d8;
        border-radius:7px;
        background:#fff;
        box-shadow:0 8px 28px rgba(23,25,30,.24);
      }
      .processing-operations-colour-popover[hidden] { display:none !important; }
      .processing-operations-colour-title {
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:8px;
        margin-bottom:8px;
        color:#454c55;
        font-size:10px;
        font-weight:650;
      }
      .processing-operations-colour-reset {
        padding:3px 7px;
        border:1px solid rgba(127,127,127,.30);
        border-radius:4px;
        background:#fff;
        color:inherit;
        font-size:9px;
        cursor:pointer;
      }
      .processing-operations-colour-swatches {
        display:grid;
        grid-template-columns:repeat(8,1fr);
        gap:5px;
      }
      .processing-operations-colour-swatch {
        aspect-ratio:1;
        padding:0;
        border:1px solid rgba(0,0,0,.10);
        border-radius:3px;
        cursor:pointer;
      }
      .processing-operations-show-all {
        height:27px;
        padding:0 8px;
        border:1px solid rgba(127,127,127,.32);
        border-radius:4px;
        background:#f7f7f8;
        color:inherit;
        cursor:pointer;
        font-size:10px;
        font-weight:400;
        white-space:nowrap;
      }
      .processing-operations-show-all:hover {
        background:#eceafc;
        border-color:#b8afea;
        color:#4732b7;
      }
      .processing-operations-show-all:disabled {
        cursor:default;
        opacity:.45;
        background:#f7f7f8;
        border-color:rgba(127,127,127,.32);
        color:inherit;
      }
      .processing-operations-hscroll {
        display:block;
        overflow-x:scroll;
        overflow-y:hidden;
        height:16px;
        margin:0 34px 2px 88px;
        scrollbar-gutter:stable;
      }
      .processing-operations-hscroll[hidden] {
        display:none !important;
      }
      .processing-operations-hscroll-track {
        height:1px;
        min-width:100%;
        pointer-events:none;
      }
      #training_chart_group {
        width:100%;
        min-height:0;
        padding-bottom:12px;
      }
      #training_chart_group > .chart-grid {
        min-height:0;
      }
      #training_throughput_card {
        flex:0 0 100%;
        width:100%;
        max-width:100%;
      }
      #training_throughput_card .plot-shell {
        min-height:210px;
      }
      #processing_contention_card:not(.maximized) {
        min-height:220px !important;
        height:220px !important;
      }
      #processing_contention_card:not(.maximized) .processing-plot-shell {
        min-height:146px !important;
        height:146px !important;
        flex:0 0 146px !important;
      }
    `;
    document.head.appendChild(style);
  }

  const family_colours = Object.freeze({
    QKV:"#4d79a7",
    O:"#7a59d1",
    UP:"#2e9d57",
    DOWN:"#e76f38",
  });
  const operation_colours = Object.freeze({
    attention:"#00a6d6",
    layernorm:"#8c6d5a",
    activation:"#d9a300",
    residual:"#607d8b",
    lm_head:"#6f42c1",
    loss:"#b56b21",
    misc:"#8c8c8c",
    other:"#8c8c8c",
  });
  const operation_colour_storage_key = "thog2_processing_operation_colours_v1";
  const custom_operation_colours = (
    typeof load_json === "function"
      ? load_json(operation_colour_storage_key, {})
      : {}
  );
  const lane_y = Object.freeze({MAIN:1.00, PREMAT:0.70, OTHER:0.40, UNKNOWN:0.15});
  const lane_width = 0.30;
  const layer_top_y = 1.42;
  const layer_bottom_y = 0.28;
  const normal_y_range = [0.20, 1.50];
  const maximized_y_range = [0.20, 1.50];
  processing_view.operations_hidden_keys = processing_view.operations_hidden_keys || new Set();
  processing_view.operations_colour_defaults = processing_view.operations_colour_defaults || {};
  processing_view.operations_ancillary_white = false;
  const ancillary_operations = new Set(["misc", "layernorm", "lm_head", "loss"]);

  function operation_colour_key(row) {
    const family = String(row.family || "-").toUpperCase();
    const operation = String(row.operation || "misc").toLowerCase();
    if (Object.hasOwn(family_colours, family)) return `FAMILY:${family}`;
    return `OPERATION:${operation}:${semantic_label(row)}`;
  }

  function legacy_operation_colour_key(row) {
    const owner = String(row.owner || "UNKNOWN").toUpperCase();
    const family = String(row.family || "-").toUpperCase();
    const operation = String(row.operation || "misc").toLowerCase();
    return `${owner}:${family}:${operation}:${semantic_label(row)}`;
  }

  function family_colour(row) {
    const family = String(row.family || "").toUpperCase();
    const operation = String(row.operation || "misc").toLowerCase();
    const fallback = family_colours[family] || operation_colours[operation] || "#8c8c8c";
    const key = operation_colour_key(row);
    const legacy_main_key = Object.hasOwn(family_colours, family)
      ? `MAIN:${family}:consume:${family} GEMM`
      : null;
    processing_view.operations_colour_defaults[key] = fallback;
    return custom_operation_colours[key]
      || (legacy_main_key ? custom_operation_colours[legacy_main_key] : null)
      || (!Object.hasOwn(family_colours, family)
        ? custom_operation_colours[legacy_operation_colour_key(row)]
        : null)
      || fallback;
  }

  function semantic_label(row) {
    const operation = String(row.operation || "misc").toLowerCase();
    const family = String(row.family || "").toUpperCase();
    if (operation === "materialize") return `${family || "?"} MAT`;
    if (operation === "consume") return `${family || "?"} GEMM`;
    return operation === "lm_head" ? "LM HEAD" : operation.toUpperCase();
  }

  function hover_label(row) {
    const owner = String(row.owner || "UNKNOWN").toUpperCase();
    const family = String(row.family || "").toUpperCase();
    if (owner === "PREMAT") return family ? `PREMAT · ${family}` : "PREMAT";
    return semantic_label(row);
  }

  function marker_for(row) {
    const owner = String(row.owner || "").toUpperCase();
    const operation = String(row.operation || "").toLowerCase();
    const colour = family_colour(row);
    const marker = {color:colour, line:{width:0}};
    if (owner === "PREMAT" || operation === "materialize") {
      marker.pattern = {
        shape:"/",
        fgcolor:"rgba(255,255,255,0.78)",
        bgcolor:colour,
        solidity:0.30,
        size:7,
        fillmode:"overlay",
      };
    }
    return marker;
  }

  function layer_guides(intervals) {
    const first_by_layer = new Map();
    for (const row of intervals) {
      if (String(row.owner || "") !== "MAIN") continue;
      const layer = Number(row.layer);
      if (!Number.isFinite(layer)) continue;
      const start = Number(row.start_us) / 1000.0;
      if (!Number.isFinite(start)) continue;
      if (!first_by_layer.has(layer) || start < first_by_layer.get(layer)) first_by_layer.set(layer, start);
    }
    const sorted = [...first_by_layer.entries()].sort((left, right) => left[1] - right[1]);
    const shapes = [];
    for (const [_layer, x] of sorted) {
      shapes.push({
        type:"line", xref:"x", yref:"y", x0:x, x1:x,
        y0:layer_bottom_y, y1:layer_top_y,
        line:{color:"rgba(96,103,113,0.58)", width:1}, layer:"above",
      });
    }
    const annotations = [];
    for (const [layer, x] of sorted) {
      for (const [position, y] of [["top", layer_top_y], ["bottom", layer_bottom_y]]) {
        annotations.push({
          xref:"x", yref:"y", x, y, text:String(layer + 1), showarrow:false,
          xanchor:"center", yanchor:"middle", font:{size:13, color:"#343a43"},
          bgcolor:"rgba(255,255,255,0.94)", borderpad:2,
          name:`processing-layer-${layer}-${position}`, captureevents:true,
          hovertext:`Zoom to layer ${layer + 1}`,
        });
      }
    }
    return {
      sorted,
      shapes,
      annotations,
    };
  }

  function layer_zoom_range(sorted, capture_ms, layer) {
    const index = sorted.findIndex(([value]) => Number(value) === Number(layer));
    if (index < 0) return null;
    const start = Number(sorted[index][1]);
    const end = index + 1 < sorted.length
      ? Number(sorted[index + 1][1])
      : Math.max(start, Number(capture_ms || start));
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return null;
    const previous_span = index > 0
      ? start - Number(sorted[index - 1][1])
      : end - start;
    const next_span = index + 2 < sorted.length
      ? Number(sorted[index + 2][1]) - end
      : end - start;
    return [
      Math.max(0, start - Math.max(0, previous_span) * 0.12),
      end + Math.max(0, next_span) * 0.12,
    ];
  }

  function layer_scroll_metrics(capture_ms, range, viewport_width) {
    const capture = Math.max(0, Number(capture_ms) || 0);
    const start = Math.max(0, Number(range?.[0]) || 0);
    const end = Math.min(capture, Number(range?.[1]) || capture);
    const span = Math.max(0, end - start);
    const viewport = Math.max(1, Number(viewport_width) || 1);
    if (!(capture > 0) || !(span > 0) || span >= capture) {
      return {virtual_width:viewport, max_scroll:0, scroll_left:0, span:capture};
    }
    const virtual_width = Math.max(viewport + 1, Math.round(viewport * capture / span));
    const max_scroll = Math.max(1, virtual_width - viewport);
    const scroll_left = Math.max(0, Math.min(max_scroll, start / (capture - span) * max_scroll));
    return {virtual_width, max_scroll, scroll_left, span};
  }

  function layer_range_for_scroll(capture_ms, span, scroll_left, max_scroll) {
    const capture = Math.max(0, Number(capture_ms) || 0);
    const width = Math.max(0, Math.min(capture, Number(span) || 0));
    const maximum = Math.max(0, Number(max_scroll) || 0);
    const position = Math.max(0, Math.min(maximum, Number(scroll_left) || 0));
    const start = maximum > 0 ? position / maximum * Math.max(0, capture - width) : 0;
    return [start, start + width];
  }

  function ensure_layer_zoom_controls() {
    const card = by_id("processing_timeline_card");
    const actions = card?.querySelector(".chart-card-actions");
    const shell = card?.querySelector(".processing-plot-shell");
    if (!actions || !shell) return;
    let button = by_id("processing_operations_reset_zoom");
    if (!button) {
      button = document.createElement("button");
      button.id = "processing_operations_reset_zoom";
      button.type = "button";
      button.className = "processing-operations-show-all";
      button.textContent = "Reset zoom";
      button.title = "Restore the complete MAIN/PREMAT capture time range";
      button.disabled = true;
      button.addEventListener("click", async event => {
        event.preventDefault();
        event.stopPropagation();
        await reset_layer_zoom(by_id("processing_timeline_plot"));
      });
      actions.insertBefore(button, actions.firstChild);
    }
    let scrollbar = by_id("processing_operations_hscroll");
    if (!scrollbar) {
      scrollbar = document.createElement("div");
      scrollbar.id = "processing_operations_hscroll";
      scrollbar.className = "processing-operations-hscroll";
      scrollbar.hidden = true;
      scrollbar.title = "Pan the zoomed layer window across the complete capture";
      scrollbar.innerHTML = '<div class="processing-operations-hscroll-track"></div>';
      shell.insertAdjacentElement("afterend", scrollbar);
      scrollbar.addEventListener("scroll", () => {
        const mount = by_id("processing_timeline_plot");
        if (!mount || mount._processing_layer_scroll_syncing || !mount._processing_layer_zoom_range) return;
        if (mount._processing_layer_scroll_frame) cancelAnimationFrame(mount._processing_layer_scroll_frame);
        mount._processing_layer_scroll_frame = requestAnimationFrame(() => {
          mount._processing_layer_scroll_frame = null;
          const range = layer_range_for_scroll(
            mount._processing_layer_zoom_capture_ms,
            mount._processing_layer_zoom_range[1] - mount._processing_layer_zoom_range[0],
            scrollbar.scrollLeft,
            scrollbar.scrollWidth - scrollbar.clientWidth,
          );
          mount._processing_layer_zoom_range = range;
          Plotly.relayout(mount, {"xaxis.range":range}).catch(() => {});
        });
      }, {passive:true});
      if (typeof ResizeObserver === "function") {
        scrollbar._processing_resize_observer = new ResizeObserver(() => {
          const mount = by_id("processing_timeline_plot");
          if (mount?._processing_layer_zoom_range) sync_layer_scrollbar(mount);
        });
        scrollbar._processing_resize_observer.observe(shell);
      }
    }
  }

  function sync_layer_scrollbar(mount) {
    const scrollbar = by_id("processing_operations_hscroll");
    const reset = by_id("processing_operations_reset_zoom");
    if (!scrollbar || !mount?._processing_layer_zoom_range) {
      if (scrollbar) scrollbar.hidden = true;
      if (reset) reset.disabled = true;
      return;
    }
    if (reset) reset.disabled = false;
    const track = scrollbar.firstElementChild;
    const metrics = layer_scroll_metrics(
      mount._processing_layer_zoom_capture_ms,
      mount._processing_layer_zoom_range,
      scrollbar.clientWidth || mount.clientWidth,
    );
    if (!track || metrics.max_scroll <= 0) {
      scrollbar.hidden = true;
      return;
    }
    scrollbar.hidden = false;
    track.style.width = `${metrics.virtual_width}px`;
    mount._processing_layer_scroll_syncing = true;
    scrollbar.scrollLeft = metrics.scroll_left;
    requestAnimationFrame(() => { mount._processing_layer_scroll_syncing = false; });
    if (reset) reset.disabled = false;
  }

  function apply_layer_zoom(mount, range) {
    if (!mount || !Array.isArray(range)) return;
    mount._processing_layer_zoom_range = range.slice();
    const reset = by_id("processing_operations_reset_zoom");
    if (reset) reset.disabled = false;
    Plotly.relayout(mount, {"xaxis.range":range}).then(() => sync_layer_scrollbar(mount)).catch(() => {});
  }

  async function reset_layer_zoom(mount) {
    if (!mount) return;
    const capture_ms = Number(mount._processing_layer_zoom_capture_ms || 0);
    mount._processing_layer_zoom_range = null;
    const scrollbar = by_id("processing_operations_hscroll");
    const reset = by_id("processing_operations_reset_zoom");
    if (scrollbar) scrollbar.hidden = true;
    if (reset) reset.disabled = true;
    const update = capture_ms > 0
      ? {"xaxis.autorange":false, "xaxis.range":[0, capture_ms]}
      : {"xaxis.autorange":true};
    try {
      await Plotly.relayout(mount, update);
    } catch (_error) {
      // Leave the controls reset even if Plotly is tearing down this run.
    }
  }

  function install_layer_zoom(mount, guides, capture_ms) {
    if (!mount) return;
    mount._processing_layer_zoom_guides = guides;
    mount._processing_layer_zoom_capture_ms = capture_ms;
    if (mount._processing_layer_zoom_installed === true) return;
    mount._processing_layer_zoom_installed = true;
    mount.on("plotly_clickannotation", event => {
      const name = String(event?.annotation?.name || "");
      const match = /^processing-layer-(\d+)-(?:top|bottom)$/.exec(name);
      if (!match) return;
      const range = layer_zoom_range(
        mount._processing_layer_zoom_guides?.sorted || [],
        mount._processing_layer_zoom_capture_ms,
        Number(match[1]),
      );
      if (range) apply_layer_zoom(mount, range);
    });
  }

  function operations_card_maximized() {
    return by_id("processing_timeline_card")?.classList.contains("maximized") === true;
  }

  function current_y_range() {
    return operations_card_maximized() ? maximized_y_range : normal_y_range;
  }

  function operations_colour_palette() {
    const shared = window.instra_colour_palette;
    if (Array.isArray(shared) && shared.length) return shared;
    return [...new Set([
      ...Object.values(family_colours),
      ...Object.values(operation_colours),
      "#000000", "#FFFFFF",
    ].map(colour => String(colour).toUpperCase()))];
  }

  function close_operations_colour_picker() {
    const popover = by_id("processing_operations_colour_popover");
    if (popover) popover.hidden = true;
    processing_view.operations_colour_picker_key = null;
  }

  function operation_trace_indices(mount, key, patterned = false) {
    if (!mount || !Array.isArray(mount.data)) return [];
    const indices = [];
    mount.data.forEach((trace, index) => {
      const meta = trace.meta || {};
      if (patterned ? meta.operations_patterned === true : meta.operations_colour_key === key) {
        indices.push(index);
      }
    });
    return indices;
  }

  function save_operation_colours() {
    if (typeof save_json === "function") {
      save_json(operation_colour_storage_key, custom_operation_colours);
    }
  }

  async function apply_operation_colour(key, colour) {
    if (!key) return;
    if (colour) custom_operation_colours[key] = String(colour).toUpperCase();
    else delete custom_operation_colours[key];
    save_operation_colours();
    let resolved = custom_operation_colours[key]
      || processing_view.operations_colour_defaults[key]
      || "#8C8C8C";
    const mount = by_id("processing_timeline_plot");
    const indices = operation_trace_indices(mount, key, false);
    if (
      processing_view.operations_ancillary_white
      && indices.some(index => ancillary_operations.has(String(mount?.data?.[index]?.meta?.operations_operation || "")))
    ) resolved = "#FFFFFF";
    if (indices.length) {
      await Plotly.restyle(mount, {
        "marker.color":resolved,
        "marker.pattern.bgcolor":resolved,
      }, indices);
    }
    render_operations_key(mount?.data || []);
  }

  function ensure_operations_colour_picker() {
    let popover = by_id("processing_operations_colour_popover");
    if (popover) return popover;
    popover = document.createElement("div");
    popover.id = "processing_operations_colour_popover";
    popover.className = "processing-operations-colour-popover";
    popover.hidden = true;
    popover.innerHTML = `
      <div class="processing-operations-colour-title">
        <span id="processing_operations_colour_title">Operation colour</span>
        <button class="processing-operations-colour-reset" type="button">Default</button>
      </div>
      <div class="colour-swatches processing-operations-colour-swatches"></div>`;
    document.body.appendChild(popover);
    popover.querySelector(".processing-operations-colour-reset")?.addEventListener("click", async event => {
      event.preventDefault();
      await apply_operation_colour(processing_view.operations_colour_picker_key, null);
      close_operations_colour_picker();
    });
    const swatches = popover.querySelector(".processing-operations-colour-swatches");
    for (const colour of operations_colour_palette()) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "colour-swatch processing-operations-colour-swatch";
      button.style.background = colour;
      button.title = colour;
      button.setAttribute("aria-label", `Choose operation colour ${colour}`);
      button.addEventListener("click", async event => {
        event.preventDefault();
        await apply_operation_colour(processing_view.operations_colour_picker_key, colour);
        close_operations_colour_picker();
      });
      swatches?.appendChild(button);
    }
    return popover;
  }

  function open_operations_colour_picker(key, label, anchor) {
    const popover = ensure_operations_colour_picker();
    if (!popover || !anchor) return;
    processing_view.operations_colour_picker_key = key;
    const title = by_id("processing_operations_colour_title");
    if (title) title.textContent = `${label} colour`;
    popover.hidden = false;
    const rect = anchor.getBoundingClientRect();
    const width = 282;
    const height = Math.min(360, Math.max(160, popover.offsetHeight || 300));
    popover.style.left = `${Math.max(8, Math.min(rect.left, window.innerWidth - width - 8))}px`;
    popover.style.top = `${Math.max(8, Math.min(rect.bottom + 6, window.innerHeight - height - 8))}px`;
  }

  async function toggle_operations_key(key, patterned = false) {
    const mount = by_id("processing_timeline_plot");
    const indices = operation_trace_indices(mount, key, patterned);
    if (!indices.length) return;
    const state_key = patterned ? "__MATERIALISATION__" : key;
    const hide = !processing_view.operations_hidden_keys.has(state_key);
    if (hide) processing_view.operations_hidden_keys.add(state_key);
    else processing_view.operations_hidden_keys.delete(state_key);
    await Plotly.restyle(mount, {visible:hide ? false : true}, indices);
    render_operations_key(mount.data || []);
  }

  function ensure_operations_key() {
    let key = by_id("processing_operations_key");
    if (key) return key;
    const card = by_id("processing_timeline_card");
    const shell = card?.querySelector(".processing-plot-shell");
    if (!card || !shell) return null;
    key = document.createElement("div");
    key.id = "processing_operations_key";
    key.className = "processing-operations-key";
    key.setAttribute("aria-label", "MAIN and PREMAT operation key");
    shell.insertAdjacentElement("beforebegin", key);
    return key;
  }

  function render_operations_key(traces) {
    const key = ensure_operations_key();
    if (!key) return;
    key.replaceChildren();
    const items = new Map();
    let has_materialisation = false;
    for (const trace of traces || []) {
      const meta = trace.meta || {};
      if (meta.operations_patterned === true) {
        has_materialisation = true;
        continue;
      }
      if (!meta.operations_colour_key || items.has(meta.operations_colour_key)) continue;
      items.set(meta.operations_colour_key, {
        label:String(meta.operations_label || trace.name || "operation"),
        colour:String(trace.marker?.color || "#8C8C8C"),
      });
    }
    for (const [colour_key, item] of items) {
      const wrapper = document.createElement("span");
      wrapper.className = "processing-operations-key-item";
      wrapper.classList.toggle("is-hidden", processing_view.operations_hidden_keys.has(colour_key));
      const colour = document.createElement("button");
      colour.type = "button";
      colour.className = "processing-operations-key-colour";
      colour.style.background = item.colour;
      colour.title = `Choose ${item.label} colour`;
      colour.setAttribute("aria-label", colour.title);
      colour.addEventListener("click", event => {
        event.preventDefault();
        event.stopPropagation();
        open_operations_colour_picker(colour_key, item.label, colour);
      });
      const toggle = document.createElement("button");
      toggle.type = "button";
      toggle.className = "processing-operations-key-toggle";
      toggle.textContent = item.label;
      toggle.title = `Show or hide ${item.label}`;
      toggle.addEventListener("click", () => toggle_operations_key(colour_key, false));
      wrapper.append(colour, toggle);
      key.appendChild(wrapper);
    }
    if (has_materialisation) {
      const wrapper = document.createElement("span");
      wrapper.className = "processing-operations-key-item";
      wrapper.classList.toggle("is-hidden", processing_view.operations_hidden_keys.has("__MATERIALISATION__"));
      const pattern = document.createElement("span");
      pattern.className = "processing-operations-key-pattern";
      pattern.setAttribute("aria-hidden", "true");
      const toggle = document.createElement("button");
      toggle.type = "button";
      toggle.className = "processing-operations-key-toggle";
      toggle.textContent = "PREMAT";
      toggle.title = "Show or hide all striped materialisation and PREMAT operations";
      toggle.addEventListener("click", () => toggle_operations_key("__MATERIALISATION__", true));
      wrapper.append(pattern, toggle);
      key.appendChild(wrapper);
    }
  }

  function ensure_show_all_button() {
    const card = by_id("processing_timeline_card");
    const actions = card?.querySelector(".chart-card-actions");
    if (!actions) return;
    let button = by_id("processing_operations_show_all");
    if (!button) {
      button = document.createElement("button");
      button.id = "processing_operations_show_all";
      button.type = "button";
      button.className = "processing-operations-show-all";
      button.textContent = "Show All";
      button.title = "Restore every MAIN/PREMAT operation hidden through the legend";
      button.addEventListener("click", async event => {
        event.preventDefault();
        event.stopPropagation();
        const mount = by_id("processing_timeline_plot");
        if (!mount || mount.dataset.plotReady !== "true" || !Array.isArray(mount.data)) return;
        const indices = mount.data.map((_trace, index) => index);
        if (indices.length) await Plotly.restyle(mount, {visible:true}, indices);
        processing_view.operations_hidden_keys.clear();
        processing_view.operations_ancillary_white = false;
        await restore_operation_trace_colours(mount);
        render_operations_key(mount.data || []);
      });
      actions.insertBefore(button, actions.firstChild);
    }
  }

  function resolved_trace_colour(trace) {
    const key = String(trace?.meta?.operations_colour_key || "");
    return custom_operation_colours[key]
      || processing_view.operations_colour_defaults[key]
      || String(trace?.meta?.operations_base_colour || trace?.marker?.color || "#8C8C8C");
  }

  async function restore_operation_trace_colours(mount) {
    if (!mount || !Array.isArray(mount.data)) return;
    for (let index = 0; index < mount.data.length; index += 1) {
      const trace = mount.data[index];
      const colour = resolved_trace_colour(trace);
      await Plotly.restyle(mount, {
        "marker.color":colour,
        "marker.pattern.bgcolor":colour,
      }, [index]);
    }
  }

  function ensure_ancillary_button() {
    const actions = by_id("processing_timeline_card")?.querySelector(".chart-card-actions");
    const show_all = by_id("processing_operations_show_all");
    if (!actions || !show_all || by_id("processing_operations_hide_ancillary")) return;
    const button = document.createElement("button");
    button.id = "processing_operations_hide_ancillary";
    button.type = "button";
    button.className = "processing-operations-show-all";
    button.textContent = "Core only";
    button.title = "Render MISC, LAYERNORM, LM HEAD and LOSS in white; Show All restores their colours";
    button.addEventListener("click", async event => {
      event.preventDefault();
      event.stopPropagation();
      const mount = by_id("processing_timeline_plot");
      if (!mount || mount.dataset.plotReady !== "true" || !Array.isArray(mount.data)) return;
      processing_view.operations_ancillary_white = true;
      const indices = mount.data.flatMap((trace, index) => (
        ancillary_operations.has(String(trace?.meta?.operations_operation || "")) ? [index] : []
      ));
      if (indices.length) await Plotly.restyle(mount, {
        "marker.color":"#FFFFFF",
        "marker.pattern.bgcolor":"#FFFFFF",
      }, indices);
      render_operations_key(mount.data || []);
    });
    actions.insertBefore(button, show_all);
  }

  function ensure_training_throughput_group() {
    let group = by_id("training_chart_group");
    if (group) return group;
    const processing = by_id("processing_chart_group");
    if (!processing?.parentElement) return null;
    group = document.createElement("section");
    group.className = "chart-group training-group";
    group.id = "training_chart_group";
    group.dataset.chartGroup = "training";
    group.hidden = true;
    group.innerHTML = `
      <header class="chart-group-header">
        <button class="chart-group-toggle" id="training_group_toggle" type="button" aria-expanded="true" aria-controls="training_grid">
          <span class="group-caret" aria-hidden="true">⌄</span><strong>training</strong><span class="group-count">1</span>
        </button>
      </header>
      <div class="chart-grid" id="training_grid">
        <article class="chart-card training-throughput-card" id="training_throughput_card" data-chart="training_throughput">
          <header class="chart-card-header"><div class="chart-heading-copy"><h2>Training throughput (tok/s)</h2></div><div class="chart-card-actions"><button class="processing-throughput-z-button" id="training_throughput_z_button" type="button" title="Cycle which throughput curve is drawn on top">z</button><button class="maximize-button" data-maximize="training_throughput" type="button" aria-label="Maximize training throughput" title="Maximize chart"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true" style="pointer-events:none;vertical-align:middle"><rect x="4" y="4" width="16" height="16" rx="1"/></svg></button></div></header>
          <div class="plot-shell"><div class="plot-mount" id="training_throughput_plot"></div></div>
          <div class="panel-resizer panel-resizer-east" data-resize="east" title="Drag to resize chart width"></div>
          <div class="panel-resizer panel-resizer-south" data-resize="south" title="Drag to resize chart height"></div>
          <div class="panel-resizer panel-resizer-corner" data-resize="both" title="Drag to resize chart"></div>
        </article>
      </div>`;
    processing.insertAdjacentElement("afterend", group);
    if (typeof ensure_chart_settings_button === "function") ensure_chart_settings_button(group.querySelector(".chart-card"));
    by_id("training_throughput_z_button")?.addEventListener("click", () => {
      window.processing_gpu_cycle_throughput_z?.();
    });
    if (typeof apply_saved_panel_sizes === "function") apply_saved_panel_sizes();
    return group;
  }

  async function mirror_training_throughput() {
    const group = ensure_training_throughput_group();
    const source = by_id("processing_throughput_plot");
    const target = by_id("training_throughput_plot");
    if (!group || !source || !target || source.dataset.plotReady !== "true") return;
    const traces = Array.isArray(source.data) ? source.data : [];
    processing_view.training_throughput_available = traces.length > 0;
    group.hidden = !(
      processing_view.charts_tab_visible
      && processing_view.training_throughput_available
    );
    const layout = {...source.layout, autosize:true};
    if (target.dataset.plotReady === "true") await Plotly.react(target, traces, layout, plot_config);
    else {
      await Plotly.newPlot(target, traces, layout, plot_config);
      target.dataset.plotReady = "true";
    }
    const source_z = by_id("processing_throughput_z_button");
    const training_z = by_id("training_throughput_z_button");
    if (training_z && source_z) training_z.title = source_z.title;
    if (typeof ResizeObserver === "function" && !group._training_resize_observer) {
      group._training_resize_observer = new ResizeObserver(() => {
        if (target.dataset.plotReady === "true" && group.offsetParent !== null) Plotly.Plots.resize(target);
      });
      group._training_resize_observer.observe(group);
    }
  }

  function enforce_timeline_geometry() {
    const mount = by_id("processing_timeline_plot");
    if (!mount || mount.dataset.plotReady !== "true") return;
    const range = current_y_range();
    Plotly.relayout(mount, {"yaxis.range":range}).catch(() => {});
  }

  processing_render_timeline = async function(payload) {
    ensure_layer_zoom_controls();
    ensure_show_all_button();
    ensure_ancillary_button();
    processing_view.operations_payload = payload;
    const intervals = Array.isArray(payload.intervals) ? payload.intervals : [];
    const groups = new Map();
    for (const row of intervals) {
      const owner = ["MAIN", "PREMAT", "OTHER", "UNKNOWN"].includes(String(row.owner || ""))
        ? String(row.owner)
        : "UNKNOWN";
      if (!(["MAIN", "PREMAT"].includes(owner))) continue;
      const label = semantic_label(row);
      const family = String(row.family || "").toUpperCase();
      const operation = String(row.operation || "misc").toLowerCase();
      const key = `${owner}:${family}:${operation}:${label}`;
      if (!groups.has(key)) groups.set(key, {owner, label, rows:[]});
      groups.get(key).rows.push(row);
    }

    const traces = [];
    for (const group of groups.values()) {
      const exemplar = group.rows[0] || {};
      const colour_key = operation_colour_key(exemplar);
      const patterned = group.owner === "PREMAT" || String(exemplar.operation || "").toLowerCase() === "materialize";
      const state_key = patterned ? "__MATERIALISATION__" : colour_key;
      const marker = marker_for(exemplar);
      traces.push({
        type:"bar",
        orientation:"h",
        name:group.owner === "PREMAT"
          ? `PREMAT${String(exemplar.family || "").trim() ? ` ${String(exemplar.family).toUpperCase()}` : ""}`
          : `${group.owner} ${group.label}`,
        showlegend:false,
        visible:processing_view.operations_hidden_keys.has(state_key) ? false : true,
        x:group.rows.map(row => Math.max(0, Number(row.end_us) - Number(row.start_us)) / 1000.0),
        base:group.rows.map(row => Number(row.start_us) / 1000.0),
        y:group.rows.map(() => lane_y[group.owner] ?? lane_y.UNKNOWN),
        width:lane_width,
        marker,
        meta:{
          operations_owner:group.owner,
          operations_operation:String(exemplar.operation || "misc").toLowerCase(),
          operations_patterned:patterned,
          operations_colour_key:colour_key,
          operations_label:group.label,
          operations_base_colour:marker.color,
        },
        customdata:group.rows.map(row => [
          hover_label(row),
          row.family || "",
          row.layer === "" || row.layer === null || row.layer === undefined ? "—" : Number(row.layer) + 1,
          row.kernel_name || "",
        ]),
        hovertemplate:"%{customdata[0]} · layer %{customdata[2]}<br>%{customdata[3]}<br>%{x:.4f} ms<extra></extra>",
      });
    }

    const guides = layer_guides(intervals);
    const tick_owners = ["PREMAT", "MAIN"];
    const capture_ms = Number(payload.metadata?.capture_duration_ms || 0);
    await processing_plot("processing_timeline_plot", traces, {
      margin:{l:88, r:34, t:8, b:38},
      barmode:"overlay",
      hovermode:"closest",
      showlegend:false,
      xaxis:{title:"capture time (ms)", range:capture_ms > 0 ? [0, capture_ms] : undefined},
      yaxis:{
        tickmode:"array",
        tickvals:tick_owners.map(owner => lane_y[owner]),
        ticktext:tick_owners,
        range:current_y_range(),
        automargin:true,
        fixedrange:true,
        zeroline:false,
        showgrid:false,
      },
      shapes:guides.shapes,
      annotations:guides.annotations,
      bargap:0,
    });
    render_operations_key(traces);
    install_layer_zoom(by_id("processing_timeline_plot"), guides, capture_ms);
    reset_layer_zoom(by_id("processing_timeline_plot"));
    const heading = by_id("processing_timeline_card")?.querySelector(".chart-heading-copy h2");
    if (heading) heading.textContent = "GPT Level Operations by Stream (MAIN/PREMAT)";
    processing_gpu_link_time_axes();
  };

  processing_render_contention = async function(payload) {
    const family_order = ["QKV", "O", "UP", "DOWN"];
    const summary = processing_resolved_matrix_summary(payload);
    const families = family_order.filter(family => summary[family]);
    const premat_pct = families.map(family => Number(summary[family].premat_concurrent_with_main_pct));
    const main_pct = families.map(family => Number(summary[family].main_busy_concurrent_with_premat_pct));
    const traces = families.length ? [
      {
        type:"bar",
        orientation:"v",
        name:"PREMAT work concurrent with MAIN",
        x:families,
        y:premat_pct,
        text:premat_pct.map(value => Number.isFinite(value) ? `${value.toFixed(2)}%` : ""),
        textposition:"outside",
        cliponaxis:false,
        hovertemplate:"%{x}<br>%{y:.2f}% of PREMAT GPU work coincides with any MAIN kernel<extra></extra>",
      },
      {
        type:"bar",
        orientation:"v",
        name:"MAIN busy time concurrent with PREMAT",
        x:families,
        y:main_pct,
        text:main_pct.map(value => Number.isFinite(value) ? `${value.toFixed(2)}%` : ""),
        textposition:"outside",
        cliponaxis:false,
        hovertemplate:"%{x}<br>%{y:.2f}% of MAIN GPU busy time coincides with PREMAT<extra></extra>",
      },
    ] : [];
    await processing_plot("processing_contention_plot", traces, {
      margin:{l:64, r:24, t:30, b:46},
      barmode:"group",
      xaxis:{
        title:"matrix family",
        categoryorder:"array",
        categoryarray:family_order,
        fixedrange:true,
      },
      yaxis:{
        title:"temporal overlap (%)",
        range:[0,100],
        dtick:20,
        ticksuffix:"%",
        fixedrange:true,
        zeroline:true,
      },
      legend:{orientation:"h", y:1.16, font:{size:9}},
      annotations:families.length ? [] : [{
        text:"No PREMAT materialisation intervals in this capture",
        showarrow:false, xref:"paper", yref:"paper", x:0.5, y:0.5,
      }],
    });
  };

  document.addEventListener("click", event => {
    const colour_popover = by_id("processing_operations_colour_popover");
    if (
      colour_popover
      && !colour_popover.hidden
      && !colour_popover.contains(event.target)
      && !event.target.closest?.(".processing-operations-key-colour")
    ) close_operations_colour_picker();
    const button = event.target.closest?.('#processing_timeline_card .maximize-button');
    if (!button) return;
    setTimeout(enforce_timeline_geometry, 40);
    setTimeout(enforce_timeline_geometry, 180);
  }, true);

  const processing_render_throughput_before_training_group = processing_render_throughput;
  processing_render_throughput = async function(payload) {
    await processing_render_throughput_before_training_group(payload);
    await mirror_training_throughput();
  };

  function sync_training_group_presentation() {
    const group = by_id("training_chart_group");
    if (group) group.hidden = !(
      processing_view.charts_tab_visible
      && processing_view.training_throughput_available
    );
    const processing = by_id("processing_chart_group");
    if (!group || !processing || group.parentElement !== processing.parentElement) return;
    // Processing evidence belongs to the selected run.  Put a selected trace
    // first so a compact live-training mirror can never obscure an historic
    // NSYS/NCU pair; throughput-only runs retain Training as their first view.
    if (processing_view.trace_available) processing.parentElement.insertBefore(processing, group);
    else processing.insertAdjacentElement("beforebegin", group);
  }

  if (typeof processing_sync_visibility === "function") {
    const processing_sync_visibility_before_training_group = processing_sync_visibility;
    processing_sync_visibility = function() {
      processing_sync_visibility_before_training_group();
      sync_training_group_presentation();
    };
  }

  const processing_apply_detail_tab_before_training_group = window.processing_apply_detail_tab;
  window.processing_apply_detail_tab = charts_selected => {
    processing_apply_detail_tab_before_training_group?.(charts_selected);
    const group = by_id("training_chart_group");
    if (group) group.hidden = !(
      Boolean(charts_selected)
      && processing_view.training_throughput_available
    );
  };

  window.addEventListener("load", () => {
    ensure_training_throughput_group();
    ensure_layer_zoom_controls();
    ensure_show_all_button();
    const header = by_id("processing_timeline_card")?.querySelector(".chart-card-header");
    if (header && typeof MutationObserver === "function") {
      const observer = new MutationObserver(() => {
        ensure_layer_zoom_controls();
        ensure_show_all_button();
      });
      observer.observe(header, {childList:true, subtree:true});
    }
  });

  window.processing_operations_test_hooks = Object.freeze({
    layer_guides,
    layer_zoom_range,
    layer_scroll_metrics,
    layer_range_for_scroll,
    marker_for,
    operation_colour_key,
    operations_colour_palette,
    operation_trace_indices,
    apply_operation_colour,
    render_operations_key,
    reset_layer_zoom,
    resolved_trace_colour,
    sync_training_group_presentation,
  });
})();
// ^^^ THOG
