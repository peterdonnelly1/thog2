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
        flex-wrap:nowrap !important;
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
  const lane_y = Object.freeze({MAIN:1.00, PREMAT:0.70, OTHER:0.40, UNKNOWN:0.15});
  const normal_y_range = [0.53, 1.17];
  const maximized_y_range = [-1.35, 3.45];

  function family_colour(row) {
    const family = String(row.family || "").toUpperCase();
    const operation = String(row.operation || "misc").toLowerCase();
    return family_colours[family] || operation_colours[operation] || "#8c8c8c";
  }

  function semantic_label(row) {
    const operation = String(row.operation || "misc").toLowerCase();
    const family = String(row.family || "").toUpperCase();
    if (operation === "materialize") return `${family || "?"} MAT`;
    if (operation === "consume") return `${family || "?"} GEMM`;
    return operation === "lm_head" ? "LM HEAD" : operation.toUpperCase();
  }

  function marker_for(row) {
    const operation = String(row.operation || "").toLowerCase();
    const colour = family_colour(row);
    const marker = {color:colour, line:{width:0}};
    if (operation === "consume") {
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
      for (const [y0, y1] of [[0.59, 0.81], [0.89, 1.11]]) {
        shapes.push({
          type:"line", xref:"x", yref:"y", x0:x, x1:x, y0, y1,
          line:{color:"rgba(112,118,127,0.56)", width:1}, layer:"above",
        });
      }
    }
    return {
      sorted,
      shapes,
      annotations:sorted.map(([layer, x]) => ({
        xref:"x", yref:"y", x, y:(lane_y.MAIN + lane_y.PREMAT) / 2, text:String(layer + 1), showarrow:false,
        xanchor:"center", yanchor:"middle", font:{size:12, color:"#454c55"},
        name:`processing-layer-${layer}`, captureevents:true,
        hovertext:`Zoom to layer ${layer + 1}`,
      })),
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

  function install_layer_zoom(mount, guides, capture_ms) {
    if (!mount) return;
    mount._processing_layer_zoom_guides = guides;
    mount._processing_layer_zoom_capture_ms = capture_ms;
    if (mount._processing_layer_zoom_installed === true) return;
    mount._processing_layer_zoom_installed = true;
    mount.on("plotly_clickannotation", event => {
      const name = String(event?.annotation?.name || "");
      const match = /^processing-layer-(\d+)$/.exec(name);
      if (!match) return;
      const range = layer_zoom_range(
        mount._processing_layer_zoom_guides?.sorted || [],
        mount._processing_layer_zoom_capture_ms,
        Number(match[1]),
      );
      if (range) Plotly.relayout(mount, {"xaxis.range":range}).catch(() => {});
    });
  }

  function operations_card_maximized() {
    return by_id("processing_timeline_card")?.classList.contains("maximized") === true;
  }

  function current_y_range() {
    return operations_card_maximized() ? maximized_y_range : normal_y_range;
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
      });
      actions.insertBefore(button, actions.firstChild);
    }
  }

  function enforce_timeline_geometry() {
    const mount = by_id("processing_timeline_plot");
    if (!mount || mount.dataset.plotReady !== "true") return;
    const range = current_y_range();
    Plotly.relayout(mount, {"yaxis.range":range}).catch(() => {});
  }

  processing_render_timeline = async function(payload) {
    ensure_show_all_button();
    const intervals = Array.isArray(payload.intervals) ? payload.intervals : [];
    const groups = new Map();
    for (const row of intervals) {
      const owner = ["MAIN", "PREMAT", "OTHER", "UNKNOWN"].includes(String(row.owner || ""))
        ? String(row.owner)
        : "UNKNOWN";
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
      traces.push({
        type:"bar",
        orientation:"h",
        name:`${group.owner} ${group.label}`,
        legendgroup:`${group.owner}:${group.label}`,
        x:group.rows.map(row => Math.max(0, Number(row.end_us) - Number(row.start_us)) / 1000.0),
        base:group.rows.map(row => Number(row.start_us) / 1000.0),
        y:group.rows.map(() => lane_y[group.owner] ?? lane_y.UNKNOWN),
        width:0.12,
        marker:marker_for(exemplar),
        customdata:group.rows.map(row => [
          semantic_label(row),
          row.family || "",
          row.layer === "" || row.layer === null || row.layer === undefined ? "—" : Number(row.layer) + 1,
          row.kernel_name || "",
        ]),
        hovertemplate:"%{customdata[0]} · layer %{customdata[2]}<br>%{customdata[3]}<br>%{x:.4f} ms<extra>%{fullData.name}</extra>",
      });
    }

    const guides = layer_guides(intervals);
    const active_extra = ["OTHER", "UNKNOWN"].filter(owner => intervals.some(row => String(row.owner || "") === owner));
    const tick_owners = ["PREMAT", "MAIN", ...active_extra];
    const capture_ms = Number(payload.metadata?.capture_duration_ms || 0);
    await processing_plot("processing_timeline_plot", traces, {
      margin:{l:88, r:34, t:8, b:38},
      barmode:"overlay",
      hovermode:"closest",
      legend:{orientation:"h", y:1.14, font:{size:9}, groupclick:"togglegroup"},
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
    install_layer_zoom(by_id("processing_timeline_plot"), guides, capture_ms);
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
    const button = event.target.closest?.('#processing_timeline_card .maximize-button');
    if (!button) return;
    setTimeout(enforce_timeline_geometry, 40);
    setTimeout(enforce_timeline_geometry, 180);
  }, true);

  window.addEventListener("load", () => {
    ensure_show_all_button();
    const header = by_id("processing_timeline_card")?.querySelector(".chart-card-header");
    if (header && typeof MutationObserver === "function") {
      const observer = new MutationObserver(() => ensure_show_all_button());
      observer.observe(header, {childList:true, subtree:true});
    }
  });

  window.processing_operations_test_hooks = Object.freeze({
    layer_guides,
    layer_zoom_range,
  });
})();
// ^^^ THOG
