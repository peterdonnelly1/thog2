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
    return {
      shapes:sorted.map(([_layer, x]) => ({
        type:"line", xref:"x", yref:"y", x0:x, x1:x, y0:0.865, y1:1.085,
        line:{color:"rgba(125,130,138,0.50)", width:1}, layer:"above",
      })),
      annotations:sorted.map(([layer, x]) => ({
        xref:"x", yref:"y", x, y:0.835, text:String(layer + 1), showarrow:false,
        xanchor:"center", yanchor:"top", font:{size:9, color:"#7a8088"},
      })),
    };
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
      const operation = String(exemplar.operation || "misc").toLowerCase();
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
    processing_gpu_link_time_axes();
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
})();
// ^^^ THOG
