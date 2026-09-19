// vvv THOG
"use strict";

/*
 * Processing GPU Resource Compatibility - intended A/B/C composition.
 *
 * A. Stream Resource Pressure (NSYS), with GPC clock on its own synchronized track.
 * B. MAIN / PREMAT semantic operation timeline (NSYS).
 * C. Narrow PREMAT compatibility strip (NCU), mapped only onto MAIN intervals for
 *    which representative NCU evidence genuinely exists.
 *
 * Separate NSYS and NCU captures are paired server-side. Workspace remains a
 * general multi-run viewer and is not the profiler-composition mechanism.
 */
(function install_processing_gpu_resource_compatibility_v1() {
  const style_id = "instra-processing-gpu-resource-compatibility-v1-style";
  if (!document.getElementById(style_id)) {
    const style = document.createElement("style");
    style.id = style_id;
    style.textContent = `
      .workspace-icon {
        width:22px; height:18px; display:grid;
        grid-template-columns:9px 9px; grid-template-rows:8px 8px; gap:2px;
      }
      .workspace-icon i {
        display:block; min-width:0; min-height:0;
        border:1.5px solid currentColor; border-radius:2px;
      }
      .workspace-icon i:first-child { grid-row:1 / span 2; }

      #processing_resource_card,
      #processing_timeline_card,
      #processing_compatibility_card {
        flex:0 0 100% !important; width:100% !important; max-width:100% !important;
      }
      #processing_compatibility_card { min-height:142px; }
      #processing_compatibility_card .chart-card-header {
        min-height:52px; height:auto; align-items:flex-start;
      }
      #processing_compatibility_card .processing-plot-shell {
        min-height:82px !important; height:96px;
      }
      #processing_compatibility_card.maximized .processing-plot-shell {
        height:auto; min-height:0 !important; flex:1 1 auto;
      }
      .processing-compatibility-source {
        color:#69717d; font-size:10px; line-height:1.35;
      }
      .processing-compatibility-key {
        display:flex; align-items:center; flex-wrap:wrap; gap:8px 12px;
        color:#69717d; font-size:10px;
      }
      .processing-compatibility-key span { display:inline-flex; align-items:center; gap:4px; }
      .processing-compatibility-key i {
        width:10px; height:10px; display:inline-block; border-radius:2px;
      }
      .processing-resource-legend-controls {
        display:inline-flex; align-items:center; gap:4px; margin-right:4px;
      }
      .processing-resource-legend-controls button {
        height:27px; padding:0 7px; border:1px solid rgba(127,127,127,.32);
        border-radius:4px; background:#f7f7f8; color:inherit; cursor:pointer;
        font-size:10px; font-weight:400;
      }
      .processing-resource-legend-controls button:hover {
        background:#eceafc; border-color:#b8afea; color:#4732b7;
      }
      .processing-throughput-z-button {
        display:inline-flex !important; align-items:center !important;
        justify-content:center !important; text-align:center !important;
        font-family:inherit !important; font-size:inherit !important;
        font-weight:400 !important; font-style:normal !important;
        line-height:1 !important; padding:0 !important;
      }
    `;
    document.head.appendChild(style);
  }

  const class_colours = Object.freeze({
    GREEN: "#2e9d57",
    YELLOW: "#f2cc0c",
    ORANGE: "#f28c28",
    RED: "#d63c3c",
  });

  const compatibility_rows = payload => {
    const source = payload?.premat_compatibility;
    if (Array.isArray(source)) return source;
    if (source && Array.isArray(source.rows)) return source.rows;
    return [];
  };

  function compatibility_source(payload) {
    return payload?.premat_compatibility_source || null;
  }

  function ensure_compatibility_card() {
    let card = by_id("processing_compatibility_card");
    if (card) return card;
    const grid = by_id("processing_grid");
    if (!grid) return null;
    card = document.createElement("article");
    card.className = "processing-card chart-card processing-compatibility-card";
    card.id = "processing_compatibility_card";
    card.dataset.chart = "processing_compatibility";
    card.hidden = true;
    card.innerHTML = `
      <header class="chart-card-header">
        <div class="chart-heading-copy">
          <h2>C. PREMAT compatibility</h2>
          <p>NCU structural evidence mapped only onto profiled MAIN operation/layer pairs; colour is controlled by the most constrained captured PREMAT stage; blank means unmeasured.</p>
          <p class="processing-compatibility-source" id="processing_compatibility_source"></p>
          <div class="processing-compatibility-key" aria-label="Compatibility key">
            <span><i style="background:${class_colours.GREEN}"></i>GREEN full-MAIN headroom</span>
            <span><i style="background:${class_colours.YELLOW}"></i>YELLOW tail/partial residency</span>
            <span><i style="background:${class_colours.ORANGE}"></i>ORANGE constrained</span>
            <span><i style="background:${class_colours.RED}"></i>RED no pair co-residency</span>
          </div>
        </div>
        <div class="chart-card-actions"></div>
      </header>
      <div class="processing-plot-shell"><div class="processing-plot plot-mount" id="processing_resource_compatibility_plot"></div></div>
      <div class="panel-resizer panel-resizer-east" data-resize="east" title="Drag to resize chart width"></div>
      <div class="panel-resizer panel-resizer-south" data-resize="south" title="Drag to resize chart height"></div>
      <div class="panel-resizer panel-resizer-corner" data-resize="both" title="Drag to resize chart"></div>`;
    grid.appendChild(card);
    if (typeof ResizeObserver === "function") {
      const observer = new ResizeObserver(() => processing_resize_ready_card(card));
      observer.observe(card);
      if (Array.isArray(processing_resize_observers)) processing_resize_observers.push(observer);
    }
    return card;
  }

  function ensure_resource_controls() {
    const card = by_id("processing_resource_card");
    const actions = card?.querySelector(".chart-card-actions");
    if (!actions || actions.querySelector(".processing-resource-legend-controls")) return;
    const controls = document.createElement("div");
    controls.className = "processing-resource-legend-controls";
    for (const [label, mode] of [["Show all", "show"], ["Hide all", "hide"], ["Reset", "reset"]]) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = label;
      button.dataset.resourceVisibility = mode;
      button.addEventListener("click", async event => {
        event.preventDefault();
        event.stopPropagation();
        const mount = by_id("processing_resource_plot");
        if (!mount || mount.dataset.plotReady !== "true" || !Array.isArray(mount.data)) return;
        const jobs = [];
        mount.data.forEach((trace, index) => {
          let visible = true;
          if (mode === "hide") visible = "legendonly";
          if (mode === "reset") {
            const spec = (processing_gpu_resource_specs || []).find(item => item.key === trace.legendgroup);
            visible = spec?.visible ? true : "legendonly";
          }
          jobs.push(Plotly.restyle(mount, {visible}, [index]));
        });
        await Promise.all(jobs);
      });
      controls.appendChild(button);
    }
    actions.insertBefore(controls, actions.firstChild);
  }

  function ensure_design_layout() {
    const resource = typeof processing_resource_ensure_card === "function"
      ? processing_resource_ensure_card()
      : by_id("processing_resource_card");
    const timeline = by_id("processing_timeline_card");
    const compatibility = ensure_compatibility_card();
    const grid = by_id("processing_grid");
    if (!grid) return;

    // Resource-pressure v1 originally created the compatibility shell inside A.
    // Move that existing mount into the dedicated narrow C card if necessary.
    const old_shell = by_id("processing_resource_compatibility_shell");
    if (old_shell && compatibility && old_shell.closest("#processing_compatibility_card") !== compatibility) {
      old_shell.remove();
    }
    const duplicate_mount = compatibility?.querySelector("#processing_resource_compatibility_plot");
    if (!duplicate_mount && compatibility) {
      const shell = document.createElement("div");
      shell.className = "processing-plot-shell";
      shell.innerHTML = '<div class="processing-plot plot-mount" id="processing_resource_compatibility_plot"></div>';
      compatibility.appendChild(shell);
    }

    // A, then B, then C. Existing supplementary Processing cards remain below.
    if (resource && timeline && resource.nextElementSibling !== timeline) grid.insertBefore(resource, timeline);
    if (timeline && compatibility && timeline.nextElementSibling !== compatibility) {
      timeline.insertAdjacentElement("afterend", compatibility);
    }

    const resource_heading = resource?.querySelector(".chart-heading-copy h2");
    if (resource_heading) resource_heading.textContent = "A. Stream Resource Pressure";
    const timeline_heading = timeline?.querySelector(".chart-heading-copy h2");
    if (timeline_heading) timeline_heading.textContent = "GPT Level Operations by Stream (MAIN/PREMAT)";
    ensure_resource_controls();
  }

  function has_layer(row) {
    return row?.main_layer !== "" && row?.main_layer !== null && row?.main_layer !== undefined;
  }

  // Do not smear one representative layer across unrelated layers. A profiled
  // L8 DOWN result applies to L8 DOWN only unless an explicitly layer-generic
  // compatibility row exists.
  processing_gpu_matching_compatibility = function(rows, interval) {
    const candidates = (rows || []).filter(row => {
      if (String(row.main_operation || "") && String(row.main_operation) !== String(interval.operation || "")) return false;
      if (String(row.main_family || "") && String(row.main_family) !== String(interval.family || "")) return false;
      return true;
    });
    const exact = candidates.filter(row => has_layer(row) && Number(row.main_layer) === Number(interval.layer));
    const generic = candidates.filter(row => !has_layer(row));
    const selected = exact.length ? exact : generic;
    return selected.sort((left, right) => processing_gpu_compatibility_rank(left.compatibility_class) - processing_gpu_compatibility_rank(right.compatibility_class))[0] || null;
  };

  function semantic_detail(row) {
    const layer = row.layer === "" || row.layer === null || row.layer === undefined ? "—" : Number(row.layer) + 1;
    return `${processing_escape(row.owner)} · ${processing_escape(row.operation || "misc")} · ${processing_escape(row.family || "")} · L${layer}`
      + `<br>${processing_escape(row.kernel_name || "")}`;
  }

  // B: absolute-time line segments give us visually substantial operation bars
  // while retaining a true shared x coordinate for cross-chart synchronization.
  processing_render_timeline = async function(payload) {
    const groups = processing_trace_groups(payload.intervals || []);
    const traces = [];
    const lane_order = ["MAIN", "PREMAT", "OTHER", "UNKNOWN"];
    const owners = lane_order.filter(owner => (payload.intervals || []).some(row => String(row.owner || "UNKNOWN") === owner));
    for (const [key, rows] of groups) {
      if (!rows.length) continue;
      const owner = String(rows[0].owner || "UNKNOWN");
      const operation = String(rows[0].operation || "misc");
      const x = [];
      const y = [];
      const hover = [];
      for (const row of rows) {
        const start = Number(row.start_us) / 1000.0;
        const end = Number(row.end_us) / 1000.0;
        const middle = (start + end) / 2.0;
        const detail = `${semantic_detail(row)}<br>${(end - start).toFixed(4)} ms`;
        x.push(start, middle, end, null);
        y.push(owner, owner, owner, null);
        hover.push(detail, detail, detail, "");
      }
      traces.push({
        type: "scattergl", mode: "lines", name: key.replace(":", " "),
        x, y, hovertext: hover, hoverinfo: "text", connectgaps: false,
        line: {
          width: owner === "PREMAT" ? 12 : 10,
          color: processing_gpu_operation_colour(operation, owner),
        },
      });
    }
    const capture_ms = Number(payload.metadata?.capture_duration_ms || 0);
    await processing_plot("processing_timeline_plot", traces, {
      margin: {l: 92, r: 34, t: 12, b: 48}, hovermode: "x",
      legend: {orientation: "h", y: 1.08},
      xaxis: {title: "capture time (ms)", range: capture_ms > 0 ? [0, capture_ms] : undefined},
      yaxis: {categoryorder: "array", categoryarray: owners, automargin: true},
    });
    processing_gpu_link_time_axes();
  };

  function compatibility_hover(row, interval, klass) {
    const main_layer = row.main_layer === "" || row.main_layer === null || row.main_layer === undefined ? "—" : Number(row.main_layer) + 1;
    const premat_layer = row.premat_layer === "" || row.premat_layer === null || row.premat_layer === undefined ? "—" : Number(row.premat_layer) + 1;
    const stage_index = Number(row.premat_stage_index || 1);
    const stage_count = Number(row.premat_stage_count || 1);
    const stage = `stage ${stage_index}/${stage_count}`;
    return `${klass} · MAIN ${processing_escape(interval.operation || row.main_operation || "?")} ${processing_escape(interval.family || row.main_family || "?")} L${main_layer}`
      + ` → PREMAT ${processing_escape(row.premat_family || "?")} L${premat_layer}<br>`
      + `PREMAT ${stage}${row.premat_stage_is_most_constrained ? " · most constrained captured stage" : ""}<br>`
      + `${processing_escape(row.premat_cuda_kernel_name || "unknown kernel")}<br>`
      + `Pair compatible: ${row.pair_can_co_reside ? "YES" : "NO"}<br>`
      + `PREMAT blocks with MAIN at full residency: ${Number(row.premat_blocks_with_full_main_residency || 0)}<br>`
      + `Limiter: ${processing_escape(row.limiting_resource || "—")}<br>`
      + `MAIN block: ${Number(row.main_warps_per_block || 0)} warps · ${Number(row.main_registers_per_block || 0).toLocaleString()} regs · ${Number(row.main_shared_mem_bytes || 0).toLocaleString()} B shared<br>`
      + `PREMAT block: ${Number(row.premat_warps_per_block || 0)} warps · ${Number(row.premat_registers_per_block || 0).toLocaleString()} regs · ${Number(row.premat_shared_mem_bytes || 0).toLocaleString()} B shared`;
  }

  // C: one narrow, absolute-time structural strip. Unprofiled intervals stay blank.
  processing_gpu_render_compatibility = async function(payload) {
    ensure_design_layout();
    const card = ensure_compatibility_card();
    const mount = by_id("processing_resource_compatibility_plot");
    if (!card || !mount) return;
    const compatibility = compatibility_rows(payload);
    processing_view.compatibility_available = compatibility.length > 0;
    const source = compatibility_source(payload);
    const source_copy = by_id("processing_compatibility_source");
    if (source_copy) {
      source_copy.textContent = source?.artifact_name
        ? `NCU companion: ${source.artifact_name}`
        : (compatibility.length ? "NCU compatibility from selected run" : "No matching NCU evidence available");
    }
    if (!compatibility.length) {
      card.hidden = true;
      clear_plot(mount);
      return;
    }

    const intervals = Array.isArray(payload.intervals) ? payload.intervals : [];
    if (!intervals.length) {
      // Directly selecting an NCU run remains useful, but deliberately compact:
      // it is a structural summary, not a synthetic time trace.
      const grouped = new Map();
      for (const row of compatibility) {
        const klass = String(row.compatibility_class || "").toUpperCase();
        if (!class_colours[klass]) continue;
        if (!grouped.has(klass)) grouped.set(klass, {x: [], y: [], hover: []});
        const item = grouped.get(klass);
        const main_layer = row.main_layer === "" || row.main_layer === null || row.main_layer === undefined ? "—" : Number(row.main_layer) + 1;
        const premat_layer = row.premat_layer === "" || row.premat_layer === null || row.premat_layer === undefined ? "—" : Number(row.premat_layer) + 1;
        const synthetic_interval = {operation: row.main_operation, family: row.main_family};
        item.x.push(1);
        item.y.push(`MAIN ${row.main_family || "?"} L${main_layer} → PREMAT ${row.premat_family || "?"} L${premat_layer} · stage ${Number(row.premat_stage_index || 1)}/${Number(row.premat_stage_count || 1)}`);
        item.hover.push(compatibility_hover(row, synthetic_interval, klass));
      }
      const traces = [...grouped.entries()].map(([klass, item]) => ({
        type: "bar", orientation: "h", name: klass,
        x: item.x, y: item.y, hovertext: item.hover, hoverinfo: "text",
        marker: {color: class_colours[klass]}, width: 0.46, showlegend: false,
      }));
      await processing_plot("processing_resource_compatibility_plot", traces, {
        margin: {l: 210, r: 34, t: 8, b: 18}, barmode: "group", hovermode: "closest",
        xaxis: {visible: false, range: [0, 1]}, yaxis: {automargin: true},
      });
      card.hidden = !processing_view.charts_tab_visible;
      return;
    }

    const by_key = new Map();
    const families = [...new Set(compatibility.map(row => String(row.premat_family || "PREMAT")).filter(Boolean))];
    for (const interval of intervals) {
      if (String(interval.owner) !== "MAIN") continue;
      for (const premat_family of families) {
        const match = processing_gpu_matching_compatibility(
          compatibility.filter(row => String(row.premat_family || "PREMAT") === premat_family),
          interval,
        );
        if (!match) continue;
        const klass = String(match.compatibility_class || "").toUpperCase();
        if (!class_colours[klass]) continue;
        const key = `${klass}:${premat_family}`;
        if (!by_key.has(key)) by_key.set(key, {klass, premat_family, x: [], y: [], hover: []});
        const item = by_key.get(key);
        const start = Number(interval.start_us) / 1000.0;
        const end = Number(interval.end_us) / 1000.0;
        const middle = (start + end) / 2.0;
        const detail = compatibility_hover(match, interval, klass);
        item.x.push(start, middle, end, null);
        item.y.push(`PREMAT ${premat_family}`, `PREMAT ${premat_family}`, `PREMAT ${premat_family}`, null);
        item.hover.push(detail, detail, detail, "");
      }
    }
    const traces = [...by_key.values()].map(item => ({
      type: "scattergl", mode: "lines", name: item.klass,
      x: item.x, y: item.y, hovertext: item.hover, hoverinfo: "text",
      connectgaps: false, showlegend: false,
      line: {width: 16, color: class_colours[item.klass]},
    }));
    const capture_ms = Number(payload.metadata?.capture_duration_ms || 0);
    await processing_plot("processing_resource_compatibility_plot", traces, {
      margin: {l: 92, r: 34, t: 4, b: 20}, hovermode: "x",
      xaxis: {
        range: capture_ms > 0 ? [0, capture_ms] : undefined,
        showticklabels: false, ticks: "", title: "",
      },
      yaxis: {categoryorder: "array", categoryarray: families.map(family => `PREMAT ${family}`), automargin: true},
      annotations: traces.length ? [] : [{
        text: "No profiled MAIN operation/layer pair occurs in this NSYS capture",
        showarrow: false, xref: "paper", yref: "paper", x: 0.5, y: 0.5,
      }],
    });
    card.hidden = !processing_view.charts_tab_visible;
    processing_gpu_link_time_axes();
  };

  function sync_x_range(event) {
    if (event?.["xaxis.autorange"] === true) return {"xaxis.autorange": true};
    const left = Number(event?.["xaxis.range[0]"]);
    const right = Number(event?.["xaxis.range[1]"]);
    if (!Number.isFinite(left) || !Number.isFinite(right)) return null;
    return {"xaxis.range": [left, right]};
  }

  function synchronized_mounts() {
    return [
      by_id("processing_resource_plot"),
      by_id("processing_resource_clock_plot"),
      by_id("processing_timeline_plot"),
      by_id("processing_resource_compatibility_plot"),
    ].filter(mount => mount && mount.dataset.plotReady === "true" && mount.offsetParent !== null);
  }

  processing_gpu_link_time_axes = function() {
    for (const source of synchronized_mounts()) {
      if (source._processing_intended_sync === true) continue;
      source._processing_intended_sync = true;
      source.on("plotly_relayout", event => {
        if (processing_view.gpu_axis_sync) return;
        const update = sync_x_range(event || {});
        if (!update) return;
        processing_view.gpu_axis_sync = true;
        Promise.all(
          synchronized_mounts()
            .filter(target => target !== source)
            .map(target => Promise.resolve(Plotly.relayout(target, update)))
        ).finally(() => { processing_view.gpu_axis_sync = false; });
      });
      source.on("plotly_hover", event => {
        if (processing_view._gpu_hover_sync) return;
        const x = Number(event?.points?.[0]?.x);
        if (!Number.isFinite(x)) return;
        processing_view._gpu_hover_sync = true;
        for (const target of synchronized_mounts()) {
          if (target === source) continue;
          try { Plotly.Fx.hover(target, {xval: x}, ["xy"]); } catch (_error) {}
        }
        setTimeout(() => { processing_view._gpu_hover_sync = false; }, 0);
      });
      source.on("plotly_unhover", () => {
        if (processing_view._gpu_hover_sync) return;
        for (const target of synchronized_mounts()) {
          if (target === source) continue;
          try { Plotly.Fx.unhover(target); } catch (_error) {}
        }
      });
    }
  };
  processing_resource_link_time_axes = processing_gpu_link_time_axes;

  const processing_sync_visibility_before_intended_tool = processing_sync_visibility;
  processing_sync_visibility = function() {
    processing_sync_visibility_before_intended_tool();
    ensure_design_layout();
    const resource = by_id("processing_resource_card");
    const compatibility = by_id("processing_compatibility_card");
    if (resource) {
      resource.hidden = !(processing_view.charts_tab_visible && processing_view.resource_available);
      resource.classList.remove("processing-resource-compat-only");
    }
    if (compatibility) {
      compatibility.hidden = !(processing_view.charts_tab_visible && processing_view.compatibility_available);
    }
  };

  function compatibility_download_url(payload, filename) {
    const run_id = String(compatibility_source(payload)?.dashboard_run_id || processing_current_run() || "");
    return `/api/local-file?run=${encodeURIComponent(run_id)}&path=${encodeURIComponent(`processing/${filename}`)}&download=1`;
  }

  function expose_compatibility_downloads(payload) {
    const sample_link = by_id("processing_download_stream_resources") || by_id("processing_download_samples");
    if (!sample_link?.parentElement) return;
    const files = payload?.premat_compatibility_files || {};
    for (const [id, label, filename] of [
      ["processing_download_ncu_resources", "NCU resources", files.kernel_resources],
      ["processing_download_compatibility", "Compatibility", files.csv || files.json],
    ]) {
      let link = by_id(id);
      if (!link) {
        link = sample_link.cloneNode(false);
        link.id = id;
        link.textContent = label;
        sample_link.parentElement.appendChild(link);
      }
      link.hidden = !filename;
      if (filename) link.href = compatibility_download_url(payload, filename);
    }
  }

  function update_status(payload, trace_available) {
    const rows = compatibility_rows(payload);
    const status = by_id("processing_status");
    if (!status || !rows.length) return;
    const source = compatibility_source(payload);
    if (trace_available) {
      status.textContent = `NSYS temporal + NCU structural evidence · ${rows.length} profiled compatibility pair${rows.length === 1 ? "" : "s"}`
        + (source?.artifact_name ? ` · NCU ${source.artifact_name}` : "");
    } else {
      status.textContent = `Nsight Compute structural evidence · ${rows.length} profiled compatibility pair${rows.length === 1 ? "" : "s"}`;
    }
  }

  function update_group_count() {
    const group_count = by_id("processing_group_count");
    if (!group_count) return;
    const visible = [...document.querySelectorAll("#processing_grid > .processing-card")]
      .filter(card => !card.hidden).length;
    group_count.textContent = String(visible);
  }

  const processing_render_before_intended_tool = processing_render;
  processing_render = async function(payload, trace_available) {
    ensure_design_layout();
    await processing_render_before_intended_tool(payload, trace_available);

    const rows = compatibility_rows(payload);
    processing_view.compatibility_available = rows.length > 0;
    if (!trace_available && rows.length && typeof processing_resource_render === "function") {
      // Direct NCU selection: render C only; do not pretend an NSYS trace exists.
      await processing_resource_render(payload);
    } else if (trace_available && rows.length) {
      // Resource rendering normally invoked C already, but render again after the
      // final A/B layout is established so the strip is authoritative.
      await processing_gpu_render_compatibility(payload);
    }

    expose_compatibility_downloads(payload);
    update_status(payload, Boolean(trace_available));
    processing_sync_visibility();
    processing_gpu_link_time_axes();
    requestAnimationFrame(() => {
      for (const id of ["processing_resource_card", "processing_timeline_card", "processing_compatibility_card"]) {
        processing_resize_ready_card(by_id(id));
      }
      update_group_count();
    });
  };

  window.addEventListener("load", () => {
    ensure_design_layout();
    processing_sync_visibility();
  });
})();
// ^^^ THOG
