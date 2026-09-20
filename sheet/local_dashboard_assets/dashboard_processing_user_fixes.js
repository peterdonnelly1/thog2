// vvv THOG
"use strict";

(function install_processing_user_fixes() {
  const style = document.createElement("style");
  style.id = "instra-processing-user-fixes-style";
  style.textContent = `
    .processing-downloads { min-width:0 !important; overflow-x:auto; scrollbar-width:thin; }
    .processing-download-dock {
      flex:0 0 100%; width:100%; min-width:0; height:34px;
      padding:2px 8px 4px; overflow:hidden; background:#fff;
    }
    .processing-download-dock > .processing-downloads {
      width:100%; height:28px; max-width:none; margin:0 !important;
      display:flex !important; align-items:flex-start !important; flex-wrap:nowrap !important;
      overflow-x:auto; overflow-y:hidden;
    }
    #processing_grid.is-maximized {
      flex-direction:column !important; flex-wrap:nowrap !important;
    }
    #processing_grid.is-maximized > .processing-download-dock {
      flex:0 0 34px !important; width:100% !important;
    }
    .processing-paired-download-groups { display:flex; align-items:center; justify-content:flex-start; flex-wrap:nowrap; gap:3px 7px; white-space:nowrap; }
    .processing-paired-download-group { display:inline-flex; align-items:center; gap:4px; }
    .processing-paired-download-label { font-size:9px; font-weight:850; color:#3f4650; letter-spacing:.04em; }
    #processing_timeline_card.maximized .processing-plot-shell { padding-top:0 !important; padding-bottom:0 !important; }
    #processing_compatibility_card .processing-compatibility-key { margin-top:8px; }
    #processing_compatibility_card.maximized .processing-compatibility-key i {
      width:26px; height:26px; flex:0 0 26px;
    }
  `;
  document.head.appendChild(style);

  function ensure_compatibility_maximize_button() {
    const card = by_id("processing_compatibility_card");
    const actions = card?.querySelector(".chart-card-actions");
    if (!actions || actions.querySelector(".maximize-button")) return;
    const button = typeof processing_gpu_maximize_button === "function"
      ? processing_gpu_maximize_button("processing_compatibility", "PREMAT Compatibility")
      : document.createElement("button");
    if (!button.classList.contains("maximize-button")) {
      button.className = "maximize-button";
      button.dataset.maximize = "processing_compatibility";
      button.type = "button";
      button.title = "Maximize chart";
      button.setAttribute("aria-label", "Maximize PREMAT Compatibility");
      button.innerHTML = typeof chart_size_icon === "function" ? chart_size_icon(false) : "□";
    }
    actions.appendChild(button);
  }

  const download_order = [
    "everything", "pair_manifest", "bundle", "samples", "stream_resources", "intervals",
    "lifecycle_events", "lifecycle_summary", "contention_intervals", "operation_resource_stats",
    "attribution_resource_stats", "metric_audit", "summary", "metadata",
    "raw_trace", "raw_ncu", "ncu_raw_csv", "ncu_semantic_csv", "kernel_resources", "csv", "json",
  ];
  const download_labels = {
    everything:"Everything", pair_manifest:"Manifest", bundle:"Bundle",
    samples:"Samples", stream_resources:"Streams",
    intervals:"Intervals", summary:"Summary", metadata:"Metadata",
    lifecycle_events:"Lifecycle", lifecycle_summary:"Life summary",
    contention_intervals:"Contention",
    operation_resource_stats:"Op stats",
    attribution_resource_stats:"Attrib stats", metric_audit:"Audit",
    raw_trace:"Raw", raw_ncu:"Raw", ncu_raw_csv:"Metrics",
    ncu_semantic_csv:"Semantic", kernel_resources:"Resources",
    csv:"Compat CSV", json:"Compat JSON",
  };
  const download_help = {
    everything:"Complete paired NSYS + NCU archive.\nContains every downloadable from both runs, including raw profiler reports, normalized bundles, CSV/JSON evidence, lifecycle data and the pair manifest.",
    pair_manifest:"Pair provenance manifest (JSON).\nRecords the exact NSYS and NCU artifacts, pairing key, file inventory, sizes and SHA-256 hashes used for this analysis.",
    bundle:"This run's normalized Processing archive.\nContains structured samples, operation intervals, summaries, resource attribution, metric audit and metadata for reproducible analysis.",
    samples:"Time-ordered NSYS GPU metric samples (CSV).\nIncludes capture timestamps and normalized device-wide counters at the requested sampling frequency.",
    stream_resources:"Analysis-ready resource timeline (CSV).\nCombines normalized NSYS counters with MAIN-only, PREMAT-only, overlap and mixed/unattributed interval classification.",
    intervals:"GPU kernel intervals (CSV).\nEach row contains start/end time, stream, inferred owner, semantic operation, matrix family, layer and kernel name.",
    lifecycle_events:"PREMAT scheduler lifecycle events (CSV).\nCapture-relative request, admission, submission, readiness, deadline, wait, consumption and release evidence for each matrix job.",
    lifecycle_summary:"One row per PREMAT job (CSV).\nCondenses scheduler lifecycle and GPU interval timing into analysis-ready readiness, lead, wait and completion fields.",
    contention_intervals:"Conservative MAIN/PREMAT contention intervals (CSV).\nSolid rows combine NSYS eligibility timing with NCU structural exclusion; pressure rows are explicitly non-causal device-wide overlap evidence.",
    operation_resource_stats:"Duration-weighted NSYS resource statistics by semantic operation/family/layer (CSV).\nUse this to compare MAIN and PREMAT phases without reprocessing raw samples.",
    attribution_resource_stats:"Duration-weighted resource statistics by attribution state (CSV).\nSeparates MAIN-only, PREMAT-only, simultaneous overlap and mixed/unattributed sampling bins.",
    metric_audit:"Metric provenance and quality audit (CSV).\nShows the exact Nsight metric selected for every displayed series, transformations, units, sample counts and missing-data status.",
    summary:"Per-matrix Processing summary (CSV).\nIncludes PREMAT busy time, MAIN overlap, concurrency percentages and capture-level totals.",
    metadata:"Processing capture metadata (JSON).\nIncludes schema version, run configuration, capture update/frequency, timing origin, source profiler and generated-file map.",
    raw_trace:"Original Nsight Systems .nsys-rep capture.\nOpen in Nsight Systems for complete timeline inspection beyond Instra's normalized views.",
    raw_ncu:"Original Nsight Compute .ncu-rep capture.\nOpen in Nsight Compute for the complete collected kernel/resource report and source metrics.",
    ncu_raw_csv:"Unmodified Nsight Compute raw-page metric export (CSV).\nKernel names are original CUDA names; use with the semantic export to audit every normalization step.",
    ncu_semantic_csv:"Nsight Compute raw-page export with NVTX semantic kernel names (CSV).\nMaps launches to MAIN/PREMAT owner, operation, matrix family and zero-based layer.",
    kernel_resources:"Captured NCU kernel resources (CSV).\nContains the dominant MAIN kernel and every distinct PREMAT sub-kernel per family/layer, including threads, warps, registers, shared memory, occupancy and SM capacities.",
    csv:"MAIN × PREMAT structural compatibility catalogue (CSV).\nTests block co-residency against register, shared-memory, warp, thread and block-slot limits; it does not claim observed overlap.",
    json:"MAIN × PREMAT structural compatibility catalogue (JSON).\nSame compatibility evidence as the CSV, with schema and interpretation metadata for programmatic analysis.",
  };
  const download_id_alias = Object.freeze({
    raw:"raw_trace",
    ncu_resources:"kernel_resources",
    compatibility:"csv",
  });

  function processing_download_url_for_run(run_id, filename) {
    return `/api/local-file?run=${encodeURIComponent(run_id)}&path=${encodeURIComponent(`processing/${filename}`)}&download=1`;
  }

  function download_entries(files) {
    const entries = Object.entries(files || {}).filter(([_key, filename]) => typeof filename === "string" && filename);
    entries.sort((left, right) => {
      const a = download_order.indexOf(left[0]);
      const b = download_order.indexOf(right[0]);
      return (a < 0 ? 999 : a) - (b < 0 ? 999 : b) || left[0].localeCompare(right[0]);
    });
    const seen = new Set();
    return entries.filter(([_key, filename]) => {
      if (seen.has(filename)) return false;
      seen.add(filename);
      return true;
    });
  }

  function render_paired_downloads(payload) {
    const host = document.querySelector("#processing_chart_group .processing-downloads");
    if (!host) return;
    host.querySelector(".processing-paired-download-groups")?.remove();
    const original_children = [...host.children];
    original_children.forEach(child => { child.hidden = false; });
    const pair = payload?.paired_processing_downloads;
    if (!pair?.nsys?.dashboard_run_id || !pair?.ncu?.dashboard_run_id) return;
    original_children.forEach(child => { child.hidden = true; });

    const groups = document.createElement("span");
    groups.className = "processing-paired-download-groups";
    for (const [role, label] of [["pair", "PAIR"], ["nsys", "NSYS"], ["ncu", "NCU"]]) {
      const source = pair[role] || {};
      const run_id = source.dashboard_run_id || (role === "pair" ? pair.nsys.dashboard_run_id : "");
      if (!run_id || !download_entries(source.files).length) continue;
      const group = document.createElement("span");
      group.className = "processing-paired-download-group";
      group.title = String(source.artifact_name || source.dashboard_run_id || "");
      const heading = document.createElement("strong");
      heading.className = "processing-paired-download-label";
      heading.textContent = `${label}:`;
      group.appendChild(heading);
      for (const [key, filename] of download_entries(source.files)) {
        const link = document.createElement("a");
        link.textContent = download_labels[key] || String(key).replaceAll("_", " ");
        link.href = processing_download_url_for_run(run_id, filename);
        link.download = filename;
        link.title = download_help[key] || `Download ${link.textContent}.\nFile: ${filename}`;
        link.setAttribute("aria-label", `${link.textContent}. ${link.title.replaceAll("\n", " ")}`);
        group.appendChild(link);
      }
      groups.appendChild(group);
    }
    host.appendChild(groups);
  }

  function ensure_processing_download_dock() {
    const host = document.querySelector("#processing_chart_group .processing-downloads");
    const grid = by_id("processing_grid");
    if (!host || !grid) return host;
    let dock = by_id("processing_download_dock");
    if (!dock) {
      dock = document.createElement("div");
      dock.id = "processing_download_dock";
      dock.className = "processing-download-dock";
      grid.insertBefore(dock, grid.firstElementChild);
    }
    if (host.parentElement !== dock) dock.appendChild(host);
    return host;
  }

  function decorate_direct_downloads() {
    const host = document.querySelector("#processing_chart_group .processing-downloads");
    if (!host) return;
    for (const link of host.querySelectorAll(":scope > a")) {
      const id_key = String(link.id || "").replace(/^processing_download_/, "");
      const label_key = Object.keys(download_labels).find(
        key => download_labels[key] === String(link.textContent || "").trim(),
      );
      const key = download_id_alias[id_key] || id_key || label_key;
      if (!key || !download_help[key]) continue;
      link.textContent = download_labels[key] || link.textContent;
      link.title = download_help[key];
      link.setAttribute("aria-label", `${link.textContent}. ${link.title.replaceAll("\n", " ")}`);
    }
  }

  function resource_heading_annotations(payload) {
    const rows = Array.isArray(payload?.stream_resources) ? payload.stream_resources : [];
    const panes = typeof processing_resource_panes === "function" ? processing_resource_panes(rows) : [];
    return panes.map(pane => ({
      xref:"paper", yref:"paper", x:0, y:Math.min(1.02, Number(pane.domain?.[1] || 1) + 0.018),
      text:`<b>${processing_escape(pane.label)}</b>`, showarrow:false,
      xanchor:"left", yanchor:"bottom", font:{size:11, color:"#4d5560"},
    }));
  }

  const processing_resource_render_before_user_fixes = processing_resource_render;
  processing_resource_render = async function(payload) {
    await processing_resource_render_before_user_fixes(payload);
    const mount = by_id("processing_resource_plot");
    if (mount?.dataset.plotReady === "true" && processing_view.resource_available) {
      await Plotly.relayout(mount, {annotations:resource_heading_annotations(payload)});
    }
  };

  const processing_render_contention_before_user_fixes = processing_render_contention;
  processing_render_contention = async function(payload) {
    await processing_render_contention_before_user_fixes(payload);
    const mount = by_id("processing_contention_plot");
    if (mount?.dataset.plotReady === "true") {
      await Plotly.relayout(mount, {"xaxis.domain":[0, 0.25]});
    }
  };

  function apply_operations_maximized_geometry() {
    const card = by_id("processing_timeline_card");
    const mount = by_id("processing_timeline_plot");
    if (!card || mount?.dataset.plotReady !== "true") return;
    const maximized = card.classList.contains("maximized");
    const width = 0.30;
    const indices = Array.isArray(mount.data)
      ? mount.data.flatMap((trace, index) => trace?.meta?.operations_contention === true ? [] : [index])
      : [];
    if (indices.length) Plotly.restyle(mount, {width}, indices).catch(() => {});
    Plotly.relayout(mount, {
      "yaxis.range":[0.30, 1.40],
      "margin.t":8,
      "margin.b":maximized ? 42 : 38,
    }).catch(() => {});
  }

  function apply_compatibility_geometry() {
    const card = by_id("processing_compatibility_card");
    const mount = by_id("processing_resource_compatibility_plot");
    if (!card || mount?.dataset.plotReady !== "true" || !Array.isArray(mount.data)) return;
    const maximized = card.classList.contains("maximized");
    const scatter_indices = mount.data.flatMap((trace, index) => trace.type === "scattergl" ? [index] : []);
    const bar_indices = mount.data.flatMap((trace, index) => trace.type === "bar" ? [index] : []);
    if (scatter_indices.length) Plotly.restyle(mount, {"line.width":maximized ? 34 : 16}, scatter_indices).catch(() => {});
    if (bar_indices.length) Plotly.restyle(mount, {width:maximized ? 0.72 : 0.46}, bar_indices).catch(() => {});
    Plotly.relayout(mount, {
      "margin.t":maximized ? 18 : 4,
      "margin.b":maximized ? 34 : 20,
      autosize:true,
    }).catch(() => {});
  }

  function timing_name_annotations() {
    return (processing_view.timing_entries || []).map((entry, index) => ({
      xref:"paper", yref:"y", x:0, y:index + 0.36,
      text:processing_escape(processing_update_timing_run_name(entry)),
      showarrow:false, xanchor:"left", yanchor:"bottom",
      font:{size:10, color:"#3f4650"},
    }));
  }

  function decorate_timing_timeline() {
    const mount = by_id("processing_update_timing_timeline_plot");
    const entries = processing_view.timing_entries || [];
    if (mount?.dataset.plotReady !== "true" || !entries.length) return;
    Plotly.relayout(mount, {
      annotations:timing_name_annotations(),
      "yaxis.range":[-0.5, Math.max(0.55, entries.length - 0.08)],
      "margin.l":86,
      "margin.t":24,
    }).catch(() => {});
  }

  function resize_card_plots(card) {
    if (!card || card.offsetParent === null) return;
    for (const mount of card.querySelectorAll(".plot-mount")) {
      if (mount.dataset.plotReady !== "true") continue;
      Plotly.Plots.resize(mount);
      Plotly.relayout(mount, {autosize:true}).catch(() => {});
    }
  }

  const geometry_settle_timers = new WeakMap();
  function settle_card_geometry(card) {
    if (!card) return;
    window.clearTimeout(geometry_settle_timers.get(card));
    requestAnimationFrame(() => resize_card_plots(card));
    geometry_settle_timers.set(card, window.setTimeout(() => {
      resize_card_plots(card);
      if (card.id === "processing_timeline_card") apply_operations_maximized_geometry();
      if (card.id === "processing_compatibility_card") apply_compatibility_geometry();
    }, 180));
  }

  let comparison_refresh_timer = null;
  let comparison_refresh_generation = 0;
  let last_comparison_fingerprint = "";
  let comparison_refresh_in_flight = false;
  let comparison_refresh_pending = false;

  function comparison_fingerprint() {
    const runs = typeof processing_throughput_workspace_runs === "function"
      ? processing_throughput_workspace_runs()
      : [];
    return JSON.stringify(runs.map(run => ({
      id:String(run_identifier(run)),
      revision:run.revision || run.data_updated_at || run.updated_at || null,
      state:String(run.run_state || ""),
      visible:is_visible(run_identifier(run)),
    })));
  }

  function schedule_comparison_refresh(force = false) {
    window.clearTimeout(comparison_refresh_timer);
    comparison_refresh_timer = window.setTimeout(async () => {
      const fingerprint = comparison_fingerprint();
      if (!force && fingerprint === last_comparison_fingerprint) return;
      if (comparison_refresh_in_flight) {
        comparison_refresh_pending = true;
        return;
      }
      comparison_refresh_in_flight = true;
      last_comparison_fingerprint = fingerprint;
      const generation = ++comparison_refresh_generation;
      try {
        const payload = processing_view.throughput_last_payload;
        if (payload) await processing_render_throughput(payload);
        if (generation !== comparison_refresh_generation) return;
        await processing_render_update_timing(true);
        if (generation !== comparison_refresh_generation) return;
        decorate_timing_timeline();
        settle_card_geometry(document.querySelector('.chart-card[data-chart="processing_throughput"]'));
        settle_card_geometry(by_id("processing_update_timing_timeline_card"));
      } finally {
        comparison_refresh_in_flight = false;
        if (comparison_refresh_pending) {
          comparison_refresh_pending = false;
          schedule_comparison_refresh(false);
        }
      }
    }, 120);
  }

  const render_runs_before_processing_user_fixes = render_runs;
  render_runs = function(...args) {
    const result = render_runs_before_processing_user_fixes.apply(this, args);
    schedule_comparison_refresh(false);
    return result;
  };

  const processing_render_update_timing_before_user_fixes = processing_render_update_timing;
  processing_render_update_timing = async function(force = false) {
    await processing_render_update_timing_before_user_fixes(force);
    decorate_timing_timeline();
  };

  const processing_render_before_user_fixes = processing_render;
  processing_render = async function(payload, trace_available) {
    ensure_processing_download_dock();
    await processing_render_before_user_fixes(payload, trace_available);
    const effective = processing_view.companion_enriched_payload || payload;
    ensure_compatibility_maximize_button();
    decorate_direct_downloads();
    render_paired_downloads(effective);
    apply_operations_maximized_geometry();
    apply_compatibility_geometry();
    schedule_comparison_refresh(false);
  };

  document.addEventListener("click", event => {
    const button = event.target.closest?.("#processing_chart_group .maximize-button");
    if (!button) return;
    const chart_name = String(button.dataset.maximize || "");
    setTimeout(() => {
      const card = document.querySelector(`.chart-card[data-chart="${chart_name}"]`);
      if (chart_name === "processing_timeline") apply_operations_maximized_geometry();
      if (chart_name === "processing_compatibility") apply_compatibility_geometry();
      if (chart_name === "processing_throughput" && processing_view.throughput_last_payload) {
        processing_render_throughput(processing_view.throughput_last_payload);
      }
      if (chart_name === "processing_update_timing_timeline") {
        processing_render_update_timing(true);
      }
      settle_card_geometry(card);
    }, 0);
    if (chart_name === "processing_timeline") {
      setTimeout(apply_operations_maximized_geometry, 180);
      setTimeout(() => settle_card_geometry(by_id("processing_timeline_card")), 320);
    }
    if (chart_name === "processing_compatibility") setTimeout(apply_compatibility_geometry, 180);
  }, true);

  window.processing_user_fix_test_hooks = {
    download_entries,
    processing_download_url_for_run,
    resource_heading_annotations,
    timing_name_annotations,
    apply_operations_maximized_geometry,
    render_paired_downloads,
    decorate_direct_downloads,
    ensure_processing_download_dock,
    apply_compatibility_geometry,
    download_help,
  };

  window.addEventListener("load", () => {
    ensure_processing_download_dock();
    ensure_compatibility_maximize_button();
    schedule_comparison_refresh(true);
  });
})();
// ^^^ THOG
