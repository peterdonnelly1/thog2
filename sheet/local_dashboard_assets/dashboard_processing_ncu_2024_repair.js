// vvv THOG
"use strict";

/*
 * NCU 2024.3 / INSTRA bridge.
 *
 * - render compatibility-only NCU runs without pretending they are NSYS traces;
 * - in Workspace, pair a visible NSYS temporal payload with visible NCU
 *   compatibility data so both kinds of evidence are visible together;
 * - fall back to the immutable compatibility JSON directly if the aggregate
 *   /api/processing payload has not yet incorporated it;
 * - retain the high-resolution timeline widths and Workspace rail icon.
 */
(function install_processing_ncu_2024_repair() {
  const style_id = "instra-processing-ncu-2024-repair-style";
  if (!document.getElementById(style_id)) {
    const style = document.createElement("style");
    style.id = style_id;
    style.textContent = `
      .workspace-icon {
        width:22px;
        height:18px;
        display:grid;
        grid-template-columns:9px 9px;
        grid-template-rows:8px 8px;
        gap:2px;
      }
      .workspace-icon i {
        display:block;
        min-width:0;
        min-height:0;
        border:1.5px solid currentColor;
        border-radius:2px;
      }
      .workspace-icon i:first-child { grid-row:1 / span 2; }
    `;
    document.head.appendChild(style);
  }

  const compatibility_rows = payload => (
    typeof processing_gpu_compatibility_rows === "function"
      ? processing_gpu_compatibility_rows(payload)
      : []
  );

  async function fetch_direct_compatibility(run_id) {
    if (!run_id) return null;
    const path = "processing/processing_premat_compatibility.json";
    const url = `/api/local-file?run=${encodeURIComponent(run_id)}&path=${encodeURIComponent(path)}`;
    try {
      const response = await fetch(url, {cache: "no-store"});
      if (!response.ok) return null;
      const value = await response.json();
      if (!(Array.isArray(value) || Array.isArray(value?.rows))) return null;
      return value;
    } catch (_error) {
      return null;
    }
  }

  async function ensure_current_compatibility(payload) {
    if (compatibility_rows(payload).length) return payload;
    const run_id = processing_current_run();
    const direct = await fetch_direct_compatibility(run_id);
    if (!direct) return payload;
    return {
      ...payload,
      premat_compatibility: direct,
      premat_compatibility_files: {
        ...(payload?.premat_compatibility_files || {}),
        json: "processing_premat_compatibility.json",
        csv: "processing_premat_compatibility.csv",
        kernel_resources: "processing_ncu_kernel_resources.csv",
      },
    };
  }

  async function fetch_processing(run_id) {
    try {
      const response = await fetch_json(`/api/processing?run=${encodeURIComponent(run_id)}`);
      return response?.available && response?.data ? response : null;
    } catch (_error) {
      return null;
    }
  }

  async function resolve_workspace_processing(payload, trace_available) {
    if (app.workspace_mode !== true) {
      return {payload: await ensure_current_compatibility(payload), trace_available: Boolean(trace_available), paired: false};
    }

    const runs = window.__instra_workspace?.visible_runs?.()
      || (app.runs || []).filter(run => is_visible(run_identifier(run)));
    if (!runs.length) {
      return {payload: await ensure_current_compatibility(payload), trace_available: Boolean(trace_available), paired: false};
    }

    const results = await Promise.all(runs.map(async run => ({
      run,
      run_id: run_identifier(run),
      response: await fetch_processing(run_identifier(run)),
    })));
    const available = results.filter(item => item.response?.data);
    const trace_source = available.find(item => item.response.trace_available === true) || null;
    let compatibility_source = available.find(item => compatibility_rows(item.response.data).length > 0) || null;

    if (!compatibility_source) {
      for (const item of results) {
        const direct = await fetch_direct_compatibility(item.run_id);
        if (!direct) continue;
        compatibility_source = {
          ...item,
          response: {
            available: true,
            trace_available: false,
            data: {
              metadata: null,
              intervals: [],
              samples: [],
              summary: [],
              premat_compatibility: direct,
              premat_compatibility_files: {
                json: "processing_premat_compatibility.json",
                csv: "processing_premat_compatibility.csv",
                kernel_resources: "processing_ncu_kernel_resources.csv",
              },
            },
          },
        };
        break;
      }
    }

    let merged = trace_source?.response?.data
      ? {...trace_source.response.data}
      : {...payload};

    // processing_render_throughput treats the selected run specially, so keep
    // the selected run's retained throughput even if NSYS evidence came from a
    // different visible run.
    merged.throughput = Array.isArray(payload?.throughput) ? payload.throughput : [];

    if (compatibility_source?.response?.data?.premat_compatibility) {
      merged.premat_compatibility = compatibility_source.response.data.premat_compatibility;
      merged.premat_compatibility_files = compatibility_source.response.data.premat_compatibility_files || {
        json: "processing_premat_compatibility.json",
        csv: "processing_premat_compatibility.csv",
        kernel_resources: "processing_ncu_kernel_resources.csv",
      };
    }

    return {
      payload: merged,
      trace_available: Boolean(trace_source?.response?.trace_available),
      paired: Boolean(trace_source && compatibility_source),
    };
  }

  async function thicken_processing_timeline_bars() {
    const mount = by_id("processing_timeline_plot");
    if (!mount || mount.dataset.plotReady !== "true" || !Array.isArray(mount.data)) return;
    const jobs = [];
    mount.data.forEach((trace, index) => {
      const name = String(trace?.name || "");
      const width = name.startsWith("PREMAT ") ? 12 : (name.startsWith("MAIN ") ? 10 : null);
      if (width === null) return;
      jobs.push(Plotly.restyle(mount, {"line.width": width}, [index]));
    });
    if (jobs.length) await Promise.all(jobs);
  }

  function expose_ncu_downloads(payload) {
    if (typeof processing_gpu_ensure_download_links === "function") {
      processing_gpu_ensure_download_links(payload);
    }
  }

  function set_ncu_ready_status(payload, paired) {
    const rows = compatibility_rows(payload);
    if (!rows.length) return;
    const status = by_id("processing_status");
    if (status) {
      status.textContent = paired
        ? `NSYS temporal + NCU structural evidence · ${rows.length} compatibility pair${rows.length === 1 ? "" : "s"}`
        : `Nsight Compute structural evidence · ${rows.length} compatibility pair${rows.length === 1 ? "" : "s"}`;
    }
  }

  const processing_render_before_ncu_2024_repair = processing_render;
  processing_render = async function(payload, trace_available) {
    const resolved = await resolve_workspace_processing(payload || {}, trace_available);
    await processing_render_before_ncu_2024_repair(resolved.payload, resolved.trace_available);

    const rows = compatibility_rows(resolved.payload);
    if (!resolved.trace_available && rows.length && typeof processing_resource_render === "function") {
      processing_view.available = true;
      await processing_resource_render(resolved.payload);
      processing_view.compatibility_available = true;
      const group = by_id("processing_chart_group");
      if (group) group.hidden = !processing_view.charts_tab_visible;
      const group_count = by_id("processing_group_count");
      if (group_count) group_count.textContent = "2";
      if (typeof processing_sync_visibility === "function") processing_sync_visibility();
      requestAnimationFrame(() => processing_resize_ready_card(by_id("processing_resource_card")));
    }

    expose_ncu_downloads(resolved.payload);
    set_ncu_ready_status(resolved.payload, resolved.paired);
    await thicken_processing_timeline_bars();
  };
})();
// ^^^ THOG
