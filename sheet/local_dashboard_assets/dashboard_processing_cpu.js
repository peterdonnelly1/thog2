// vvv THOG CPU work, copy evidence and memory overlays share the established Processing controls
"use strict";
(function install_cpu_processing() {
  const cpu = payload => payload?.metadata?.materialisation_device === "cpu_and_gpu";
  const value = (number, suffix="") => number === null || number === undefined ? "unknown" : Number.isFinite(Number(number)) ? Number(number).toFixed(3)+suffix : "unknown";
  function ensure_summary() {
    let card=by_id("processing_cpu_summary_card");
    if (!card) {
      const neighbour=by_id("processing_matrix_summary_card");
      if (!neighbour) return null;
      card=document.createElement("section");
      card.id="processing_cpu_summary_card";card.className="processing-card chart-card";card.dataset.chart="processing_cpu_summary";
      chart_titles.processing_cpu_summary="CPU preparation and copies";
      card.innerHTML='<div class="chart-card-header"><div class="chart-heading-copy"><h2>CPU preparation and copies</h2><p>Independent overlays; GPU device metrics remain unsplit</p></div><div class="chart-card-actions"><button class="maximize-button" data-maximize="processing_cpu_summary" type="button" aria-label="Maximize CPU preparation and copies" title="Maximize chart">'+chart_size_icon(false)+'</button></div></div><div id="processing_cpu_summary_body" style="padding:12px;overflow:auto;flex:1;min-height:0"></div><div class="panel-resizer panel-resizer-east" data-resize="east" title="Drag to resize chart width"></div><div class="panel-resizer panel-resizer-south" data-resize="south" title="Drag to resize chart height"></div><div class="panel-resizer panel-resizer-corner" data-resize="both" title="Drag to resize chart"></div>';
      neighbour.insertAdjacentElement("afterend",card);
    }
    return card;
  }
  function summary(payload) {
    const card=ensure_summary();if(!card)return;
    card.hidden=!(cpu(payload)&&processing_view.charts_tab_visible);
    const body=by_id("processing_cpu_summary_body");if(!cpu(payload)||!body)return;
    const s=payload.cpu_summary||{},m=payload.cpu_runtime||{};
    const rows=[
      ["Eligible matrix uses",s.eligible_matrix_uses],["Full hits",s.full_hits],["Complete misses",s.complete_misses],["Not targeted",s.not_targeted_uses],
      ["Unique CPU matrices",s.unique_cpu_matrices],["CPU task service",value(s.cpu_service_ms," ms")],
      ["H2D / D2H busy",value(s.h2d_busy_us," μs")+" / "+value(s.d2h_busy_us," μs")],
      ["CUDA copy coverage",s.cuda_copy_coverage||"unknown"],["Late uploads / bytes",`${s.late_upload_count??"unknown"} / ${s.late_upload_bytes??"unknown"}`],
      ["H2D bytes / copy versus Main overlap",`${s.h2d_bytes??"unknown"} / ${value(s.copy_main_overlap_us," μs")}`],
      ["CPU cache reuses",s.cpu_cache_reuse_count],["Phase and family outcomes",JSON.stringify(s.phase_family||{})],
      ["Qualified / unavailable predictions",`${s.prediction_qualified_uses??"unknown"} / ${s.prediction_unavailable_uses??"unknown"}`],["Prediction / arrival errors (ms)",JSON.stringify({prediction:s.prediction_error_ms,arrival:s.arrival_error_ms})],["Release lag (ms)",JSON.stringify(s.release_lag_ms||[])],["Older lifecycle records omitted",s.lifecycle_dropped_events],
      ["CPU cache / peak bytes",`${m.cpu_cache_bytes??"unknown"} / ${m.cpu_cache_peak_bytes??"unknown"}`],
      ["GPU staging / peak / limit bytes",`${m.staging_bytes??"unknown"} / ${m.staging_peak_bytes??"unknown"} / ${m.staging_limit_bytes??"unknown"}`],
      ["Pending / copying / available / autograd bytes",`${m.pending_upload_bytes??"unknown"} / ${m.copying_bytes??"unknown"} / ${m.available_bytes??"unknown"} / ${m.autograd_retained_bytes??"unknown"}`],
      ["Pinned upload / snapshot / preparation bytes",`${m.pinned_upload_bytes??"unknown"} / ${m.pinned_snapshot_bytes??"unknown"} / ${m.pin_preparation_in_progress_bytes??"unknown"}`],
      ["Workers × native threads",`${m.workers_resolved??"unknown"} × ${m.threads_per_worker_resolved??"unknown"}`],
      ["Co-residency","N/A: CPU materialisation"],["Fallback reasons",JSON.stringify(s.fallback_reasons||{})],
      ["Control host time (ms)",JSON.stringify(m.control_path_host_ms||{})],
      ["Precursor evidence",(payload.cpu_tasks||[]).filter(row=>row.precursor||!row.clock_alignment_known).map(row=>`${row.cpu_matrix_id}: ${row.start_ns??"unknown"}–${row.end_ns??"unknown"} ns (${row.precursor?"before capture":"clock alignment unknown"})`).join("; ")||"none observed"],
    ];
    body.innerHTML='<table class="processing-summary"><tbody>'+rows.map(([label,data])=>`<tr><th>${processing_escape(label)}</th><td>${processing_escape(data??"unknown")}</td></tr>`).join("")+'</tbody></table>';
  }
  const old_downloads=processing_set_downloads;
  processing_set_downloads=function(metadata) {
    old_downloads(metadata);
    const parent=document.querySelector(".processing-downloads");if(!parent)return;
    for(const [key,label] of [["transfers","COPY CSV"],["cpu_tasks","CPU CSV"],["cpu_memory","Memory CSV"],["cpu_lifecycle","Lifecycle CSV"]]){
      let a=by_id("processing_download_"+key);
      if(!a){a=document.createElement("a");a.id="processing_download_"+key;a.className="processing-download";a.textContent=label;parent.appendChild(a);}
      const filename=metadata?.files?.[key];a.hidden=!filename;
      if(filename){a.href=processing_download_url(filename);a.download=filename;}
    }
  };
  const old_render=processing_render;
  processing_render=async function(payload,trace_available){await old_render(payload,trace_available);summary(payload);};
  const old_visibility=processing_sync_visibility;
  processing_sync_visibility=function(){old_visibility();const card=by_id("processing_cpu_summary_card");if(card)card.hidden=!(cpu(processing_view.operations_payload)&&processing_view.charts_tab_visible);};
  const old_phases=processing_update_timing_phase_traces;
  processing_update_timing_phase_traces=function(entries){
    const traces=old_phases(entries);
    entries.forEach((entry,lane)=>{
      const overlay=entry.timing?.cpu_overlay;if(!overlay?.available)return;
      const tasks=(overlay.tasks||[]).filter(row=>row.host_start_ms!==null&&row.host_end_ms!==null&&row.host_end_ms>=0);
      if(tasks.length)traces.push({type:"bar",orientation:"h",name:"CPU tasks (overlay)",x:tasks.map(row=>row.host_end_ms-Math.max(0,row.host_start_ms)),base:tasks.map(row=>Math.max(0,row.host_start_ms)),y:tasks.map(()=>lane+0.23),width:0.10,marker:{color:"#328b69"},hovertext:tasks.map(row=>processing_escape(`${row.cpu_matrix_id} · CPU work counted once; nonadditive`)),hoverinfo:"text"});
      const origin=entry.timing.host_update_start_ns;
      const boundaries=(overlay.lifecycle||[]).filter(row=>["upload_submitted","matrix_use","gpu_storage_released"].includes(row.event)&&row.host_time_ns>=origin);
      if(origin&&boundaries.length)traces.push({type:"scatter",mode:"markers",name:"COPY / readiness boundaries (host)",x:boundaries.map(row=>(row.host_time_ns-origin)/1e6),y:boundaries.map(()=>lane-0.23),marker:{symbol:"line-ns",size:10,color:"#427ba8"},hovertext:boundaries.map(row=>processing_escape(`${row.event} · ${row.upload_id||row.matrix_use_id||"unknown identity"} · host boundary; DMA duration requires CUDA evidence`)),hoverinfo:"text"});
    });
    return traces;
  };
  const old_plot=processing_plot;
  processing_plot=async function(mount_id,traces,layout){
    const payload=processing_view.operations_payload;
    if(mount_id==="processing_resource_plot"&&cpu(payload)){
      const bands=(payload.transfers||[]).map(row=>({type:"rect",xref:"x",yref:"paper",x0:row.start_us/1000,x1:row.end_us/1000,y0:0,y1:1,line:{width:0},fillcolor:row.direction==="H2D"?"rgba(50,110,180,.15)":"rgba(190,120,40,.15)",layer:"below",name:row.direction+" DMA coverage",showlegend:true,legendgroup:"COPY_"+row.direction}));
      layout={...layout,shapes:[...(layout.shapes||[]),...bands]};
    }
    return old_plot(mount_id,traces,layout);
  };
  window.processing_cpu_test_hooks=Object.freeze({summary,cpu});
})();
// ^^^ THOG
