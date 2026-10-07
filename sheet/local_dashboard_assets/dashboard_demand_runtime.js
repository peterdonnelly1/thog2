// vvv THOG one demand-driven runtime owner for live weights, hidden plots and immutable throughput
"use strict";
(function install_demand_runtime() {
  const family_signatures=new Map(), rendered_figures=new WeakMap(), pending_plots=new Map(), plot_jobs=new WeakMap();
  let refresh_owner=null, flush_frame=null;
  const context_key=()=>app.workspace_mode ? window.__instra_workspace?.selection_key?.() : `run:${app.current_run_id}`;
  function prune_pending() {
    const context=context_key();
    for(const [mount,job] of pending_plots)if(!mount.isConnected || job.context!==context)pending_plots.delete(mount);
    const ids=new Set((app.workspace_mode ? window.__instra_workspace.visible_runs() : [current_run()]).filter(Boolean).map(run=>String(run_identifier(run))));
    for(const id of throughput_rows.keys())if(!ids.has(id))throughput_rows.delete(id);
  }
  function chart_visible(mount) {
    // vvv THOG settings previews are visible independently of maximized cards and the chart viewport
    if(mount?.id==="chart_settings_preview")return !document.hidden && mount.isConnected && !by_id("chart_settings_overlay")?.hidden;
    // ^^^ THOG
    if(document.hidden || !instra_charts_visible() || by_id("charts_scroll")?.hidden || !mount?.isConnected)return false;
    const card=mount.closest(".chart-card");
    if(!card || card.hidden || card.offsetParent===null)return false;
    if(app.maximized_chart && card.dataset.chart!==app.maximized_chart)return false;
    const bounds=card.getBoundingClientRect(), viewport=by_id("charts_scroll").getBoundingClientRect();
    return bounds.bottom>viewport.top-120 && bounds.top<viewport.bottom+120;
  }
  function schedule_flush() {
    if(flush_frame!==null)return;
    flush_frame=requestAnimationFrame(async()=>{
      flush_frame=null;
      for(const [mount,job] of pending_plots) {
        if(!mount.isConnected || job.context!==context_key()){pending_plots.delete(mount);continue;}
        if(!chart_visible(mount))continue;
        pending_plots.delete(mount);
        try{await job.draw();}catch(error){console.warn("Deferred chart render failed",error);}
      }
      void refresh_current_run();
    });
  }
  // One active render and one replaceable newest job per mount; cleared mounts release all retained arguments.
  function serialize_plot(mount,draw) {
    let state=plot_jobs.get(mount);
    if(!state){state={promise:null,latest:null,generation:0};plot_jobs.set(mount,state);}
    state.latest={draw,context:context_key(),generation:state.generation};
    if(state.promise)return state.promise;
    const task=(async()=>{
      while(state.latest) {
        const job=state.latest;state.latest=null;
        if(job.context!==context_key() || !mount.isConnected)continue;
        await job.draw();
        if((job.context!==context_key() || job.generation!==state.generation) && mount._fullLayout){Plotly.purge(mount);delete mount.dataset.plotReady;}
      }
    })();
    state.promise=task.finally(()=>{state.promise=null;state.latest=null;});
    return state.promise;
  }
  function defer_or_draw(mount,draw) {
    if(!mount)return Promise.resolve();
    if(!chart_visible(mount)){pending_plots.set(mount,{draw,context:context_key()});return Promise.resolve();}
    pending_plots.delete(mount);
    return serialize_plot(mount,draw);
  }
  const clear_plot_before=clear_plot;
  clear_plot=function(mount) {
    pending_plots.delete(mount);rendered_figures.delete(mount);
    const state=plot_jobs.get(mount);if(state){state.latest=null;state.generation++;}
    return clear_plot_before(mount);
  };
  function sync_weight_availability() {
    const status=app.current_status || current_run(), group=by_id("depth_chart_group");
    if(!group || !status)return;
    const cohort=app.workspace_mode ? window.__instra_workspace?.visible_runs?.() || [] : [status];
    const has_depth=cohort.some(run=>Number(run.depth_snapshot_count)>0), has_heatmap=!app.workspace_mode && Number(status.heatmap_count)>0;
    group.hidden=!(has_depth || has_heatmap);
    const heatmap=by_id("heatmap_plot")?.closest(".chart-card");if(heatmap)heatmap.hidden=!has_heatmap;
    for(const chart_name of depth_weight_chart_names){const card=by_id(`${chart_name}_plot`)?.closest(".chart-card");if(card)card.hidden=!has_depth;}
  }
  async function draw_requested_figures() {
    if(!app.figures || !app.current_run_id)return;
    sync_weight_availability();
    for(const chart_name of ["heatmap",...depth_weight_chart_names]) {
      const mount=by_id(`${chart_name}_plot`), figure=chart_name==="heatmap" ? app.figures.heatmap : app.figures.depth?.[chart_name];
      if(!mount || !figure || !chart_visible(mount))continue;
      const signature=JSON.stringify([context_key(),family_signatures.get(chart_name),normalize_chart_settings(chart_name),
        chart_name==="heatmap" ? heatmap_settings_for_current_run() : null,colour_for_run(app.current_run_id),load_json("thog2_local_trajectory_scale_modes",{})]);
      if(rendered_figures.get(mount)===signature && mount.dataset.plotReady==="true" && !mount.data?.some(trace=>trace.type==="scattergl"))continue;
      const requested_context=context_key();
      await render_plot(mount,figure,chart_name);
      if(requested_context!==context_key())return;
      rendered_figures.set(mount,signature);
      const placeholder=by_id(`${chart_name}_placeholder`);if(placeholder)placeholder.hidden=true;
    }
  }
  function source_signature(chart_name) {
    const cohort=app.workspace_mode ? window.__instra_workspace?.visible_runs?.() || [] : [app.current_status || current_run()];
    const settings=chart_name==="heatmap" ? heatmap_settings_for_current_run() : normalize_chart_settings(chart_name);
    return JSON.stringify([context_key(),settings,cohort.map(run=>chart_name==="heatmap"
      ? [run?.heatmap_count,run?.heatmap_maximum_update,run?.heatmap_settings]
      : [run_identifier(run),run?.depth_snapshot_count,run?.depth_minimum_update,run?.depth_maximum_update])]);
  }
  async function fetch_requested_figures(signal) {
    sync_weight_availability();
    app.figures ||= {heatmap:null,depth:{},heatmap_dimensions:{layers:0,probes:0}};
    for(const chart_name of ["heatmap",...depth_weight_chart_names]) {
      const mount=by_id(`${chart_name}_plot`);if(!chart_visible(mount))continue;
      const signature=source_signature(chart_name);if(family_signatures.get(chart_name)===signature)continue;
      const requested_context=context_key();let payload;
      if(chart_name==="heatmap") {
        const settings=heatmap_settings_for_current_run();
        payload=await fetch_json(`/api/figure-family?run=${encodeURIComponent(app.current_run_id)}&family=heatmap&probe_count=${settings.probe_count || 100}&window_mode=${settings.window_mode || "rolling"}`,{signal});
      } else {
        const settings=normalize_chart_settings(chart_name);
        const request=url=>{const parsed=new URL(url,location.origin);return fetch_json(`/api/weight-figure?run=${encodeURIComponent(parsed.searchParams.get("run"))}&chart=${chart_name}&current_only=${Number(settings.current_weights_only)}&snapshots=${settings.max_snapshots}&window=${settings.snapshot_window_mode}`,{signal});};
        payload=app.workspace_mode ? await window.__instra_workspace.fetch_depth_payload(request,signal)
          : await request(`/api/figure-family?run=${encodeURIComponent(app.current_run_id)}`);
      }
      if(signal.aborted || requested_context!==context_key())return;
      if(chart_name==="heatmap")Object.assign(app.figures,{heatmap:payload.heatmap,heatmap_dimensions:payload.heatmap_dimensions});
      else {
        if(payload.depth?.[chart_name])app.figures.depth[chart_name]=payload.depth[chart_name];
        else {delete app.figures.depth[chart_name];clear_plot(mount);}
        app.figures.weight_step_range=payload.weight_step_range;
      }
      if(!payload.incomplete)family_signatures.set(chart_name,signature);
    }
    await draw_requested_figures();
  }
  refresh_current_run=async function() {
    if(!app.current_run_id || !instra_charts_visible())return;
    prune_pending();
    const context=context_key();
    if(refresh_owner){if(refresh_owner.context===context)return refresh_owner.promise;refresh_owner.controller.abort();}
    const owner={context,controller:new AbortController(),promise:null};refresh_owner=owner;app.refresh_in_flight=true;
    const deadline=setTimeout(()=>owner.controller.abort(),15000);
    owner.promise=(async()=>{
      try {
        const status=await fetch_json(`/api/status?run=${encodeURIComponent(app.current_run_id)}`,{signal:owner.controller.signal});
        if(context!==context_key() || owner.controller.signal.aborted)return;
        app.current_status=status;render_run_heading();
        await fetch_requested_figures(owner.controller.signal);
      } catch(error){if(error.name!=="AbortError" && context===context_key())show_toast(`Run refresh failed: ${error.message}`);}
      finally{clearTimeout(deadline);if(refresh_owner===owner){refresh_owner=null;app.refresh_in_flight=false;}}
    })();
    return owner.promise;
  };
  const throughput_rows=new Map();let throughput_signature=null, throughput_owner=null;
  function sampled_trace(trace,limit) {
    if(trace.x.length<=limit)return trace;
    const indices=[0], buckets=Math.max(1,Math.floor((limit-2)/2)), count=trace.x.length-2;
    for(let bucket=0;bucket<buckets;bucket++) {
      const begin=1+Math.floor(bucket*count/buckets),end=1+Math.floor((bucket+1)*count/buckets);
      let low=begin,high=begin;
      for(let index=begin+1;index<end;index++){if(trace.y[index]<trace.y[low])low=index;if(trace.y[index]>trace.y[high])high=index;}
      indices.push(Math.min(low,high));if(low!==high)indices.push(Math.max(low,high));
    }
    indices.push(trace.x.length-1);
    return {...trace,x:indices.map(index=>trace.x[index]),y:indices.map(index=>trace.y[index]),customdata:indices.map(index=>trace.customdata?.[index])};
  }
  function throughput_run_signature(run){return JSON.stringify([run.revision,run.data_updated_at,run.acquired_at]);}
  async function render_throughput(payload) {
    processing_view.throughput_last_payload=payload;
    const target=by_id("training_throughput_plot");
    if(payload.throughput?.length)by_id("training_throughput_card").hidden=false;
    if(!chart_visible(target)) {
      if(target)pending_plots.set(target,{draw:()=>render_throughput(processing_view.throughput_last_payload || {}),context:context_key()});
      return;
    }
    const runs=processing_throughput_workspace_runs(), ids=new Set(runs.map(run=>String(run_identifier(run))));
    for(const id of throughput_rows.keys())if(!ids.has(id))throughput_rows.delete(id);
    const context=context_key(), signature=JSON.stringify([context,processing_view.throughput_axis_mode,processing_view.throughput_z_offset,
      runs.map(run=>[run_identifier(run),throughput_run_signature(run),colour_for_run(run_identifier(run))]),
      payload.throughput?.length,payload.throughput?.at(-1)]);
    if(signature===throughput_signature && (target?.dataset.plotReady==="true" || pending_plots.has(target)))return;
    const controller=new AbortController();
    if(throughput_owner)throughput_owner.abort();throughput_owner=controller;
    const deadline=setTimeout(()=>controller.abort(),15000);
    try {
      const entries=await window.__instra_workspace.map_with_concurrency(runs,4,async run=>{
        const run_id=String(run_identifier(run)), revision=throughput_run_signature(run), cached=throughput_rows.get(run_id);
        if(cached?.revision===revision)return {run,series:cached.series};
        const current=run_id===processing_current_run() && Array.isArray(payload.throughput) && !cached;
        const rows=processing_valid_throughput_rows(current ? payload.throughput : (await fetch_json(`/api/processing-throughput?run=${encodeURIComponent(run_id)}`,{signal:controller.signal})).throughput);
        const series={y:rows.map(row=>Number(row.tokens_per_second)),variants:Object.fromEntries(
          ["step","relative_wall","relative_process","wall_time"].map(mode=>[mode,rows.map(row=>processing_gpu_throughput_value(row,mode))]))};
        if(!controller.signal.aborted)throughput_rows.set(run_id,{revision,series});
        return {run,series};
      },controller.signal);
      if(controller.signal.aborted || context!==context_key())return;
      const populated=entries.filter(entry=>entry?.series.y.length);
      let mode=processing_view.throughput_axis_mode || "step";
      if(populated.some(entry=>!entry.series.variants[mode]?.some(Number.isFinite)))mode="step";
      const select=by_id("processing_throughput_x_mode");
      if(select){for(const option of select.options)option.disabled=option.value!=="step" && populated.some(entry=>!entry.series.variants[option.value]?.some(Number.isFinite));select.value=mode;}
      processing_view.throughput_axis_mode=mode;
      const traces=populated.map((_,index)=>populated[(index+processing_view.throughput_z_offset)%populated.length]).map(entry=>{
        const run_id=String(run_identifier(entry.run)), source=entry.series,values=source.variants[mode];
        const indices=values.flatMap((value,index)=>Number.isFinite(value) ? [index] : []), complete=indices.length===values.length;
        const selected=array=>complete ? array : indices.map(index=>array[index]);
        return {type:"scatter",mode:"lines",name:entry.run.artifact_name || entry.run.run_name || run_id,meta:{instra_workspace_run_id:run_id},
          x:selected(values),y:selected(source.y),thog2_x_variants:Object.fromEntries(Object.entries(source.variants).map(([key,array])=>[key,selected(array)])),
          customdata:selected(source.variants.step),line:{color:colour_for_run(run_id),width:2.4},
          hovertemplate:"%{x}<br>%{y:,.0f} tok/s<extra>%{fullData.name}</extra>"};
      });
      const layout={autosize:true,margin:{l:72,r:24,t:12,b:54},xaxis:{title:{text:{step:"optimizer update",relative_wall:"relative wall time (hours)",relative_process:"relative process time (hours)",wall_time:"wall time"}[mode]},type:mode==="wall_time" ? "date" : "linear"},
        yaxis:{title:{text:"tokens / second"},rangemode:"tozero"},showlegend:false,uirevision:context};
      processing_view.throughput_last_payload=payload;processing_view.training_throughput_available=traces.length>0;
      const figure={data:traces,layout};app.dynamic_chart_figures.training_throughput=figure;app.dynamic_chart_figures.processing_throughput=figure;
      by_id("training_throughput_card").hidden=!traces.length;
      const limit=Math.max(128,Math.min(1600,Math.floor(48000/Math.max(1,traces.length))));
      await processing_plot("training_throughput_plot",traces.map(trace=>{const {thog2_x_variants,...plot_trace}=trace;return sampled_trace(plot_trace,limit);}),layout);
      if(entries.every(Boolean))throughput_signature=signature;
    } finally {clearTimeout(deadline);if(throughput_owner===controller)throughput_owner=null;}
  }
  window.addEventListener("load",()=>setTimeout(()=>{
    const prepare_figure_before=prepare_figure;
    prepare_figure=function(figure,name,...args) {
      const prepared=prepare_figure_before(figure,name,...args);
      if(String(name).startsWith("local_metric_") || depth_weight_chart_names.includes(name)) {
        const limit=Math.max(4,Math.min(1600,Math.floor(48000/Math.max(1,prepared.data?.length || 0))));
        prepared.data=(prepared.data || []).map(trace=>{
          if(!["scatter","scattergl"].includes(trace.type))return trace;
          const {thog2_x_variants,...plotted}=trace;
          return sampled_trace({...plotted,type:"scatter"},limit);
        });
      }
      return prepared;
    };
    const render_plot_before=render_plot, processing_plot_before=processing_plot;
    // vvv THOG derive exact one-tenth minor intervals from resolved major ticks, including after zoom
    function sync_minor_grid(mount) {
      if(mount._instra_minor_grid_busy || !mount._fullLayout)return;
      const update={};
      for(const name of ["xaxis","yaxis"]) {
        const axis=mount._fullLayout[name],interval=Number(axis?.dtick)/10;
        if(axis?.minor?.showgrid && Number.isFinite(interval) && interval>0 && mount.layout?.[name]?.minor?.dtick!==interval)
          update[`${name}.minor.dtick`]=interval;
      }
      if(!Object.keys(update).length)return;
      mount._instra_minor_grid_busy=true;
      return Plotly.relayout(mount,update).finally(()=>{mount._instra_minor_grid_busy=false;});
    }
    render_plot=function(mount,figure,name,...args){return defer_or_draw(mount,async()=>{
      await render_plot_before(mount,figure,name,...args);
      if(!/loss/i.test(name+" "+chart_titles[name]) || !mount._fullLayout)return;
      await sync_minor_grid(mount);
      if(typeof mount.on==="function") {
        mount._instra_minor_grid_listener ||= ()=>{void sync_minor_grid(mount)?.catch(error=>console.warn("Minor grid update failed",error));};
        mount.removeListener?.("plotly_relayout",mount._instra_minor_grid_listener);
        mount.on("plotly_relayout",mount._instra_minor_grid_listener); // <<< THOG restore the listener after preview purges without duplicates
      }
    });};
    // ^^^ THOG
    processing_plot=function(id,traces,layout){const mount=by_id(id);return defer_or_draw(mount,()=>processing_plot_before(id,traces,layout));};
    render_figures=draw_requested_figures;
    processing_render_throughput=render_throughput;
    const timing_render_before=processing_render_update_timing;
    processing_render_update_timing=function(...args) {
      if(!["processing_update_timing_timeline_plot","processing_update_timing_microsteps_plot"]
        .some(id=>chart_visible(by_id(id))))return Promise.resolve();
      return timing_render_before(...args);
    };
    const processing_refresh_before=processing_refresh;
    let probe_owner=null,probe_key=null,probe_time=0;
    processing_refresh=async function(force=false) {
      if(document.hidden || !instra_charts_visible() || !processing_view.charts_tab_visible)return;
      const context=context_key(), run_id=app.current_run_id;
      const throughput=by_id("training_throughput_plot");
      if(chart_visible(throughput))await render_throughput({});
      const group=by_id("processing_chart_group"),bounds=group?.getBoundingClientRect(),viewport=by_id("charts_scroll")?.getBoundingClientRect();
      const processing_visible=Boolean(group && !group.hidden && group.offsetParent!==null && bounds.bottom>viewport.top && bounds.top<viewport.bottom &&
        (!app.maximized_chart || String(app.maximized_chart).startsWith("processing_")));
      if(processing_visible)return processing_refresh_before(force);
      if(probe_owner || context===probe_key && Date.now()-probe_time<10000)return;
      probe_key=context;probe_time=Date.now();probe_owner=new AbortController();
      const controller=probe_owner,deadline=setTimeout(()=>controller.abort(),5000);
      try {
        const runs=app.workspace_mode ? window.__instra_workspace.visible_runs() : [current_run()];
        const statuses=await window.__instra_workspace.map_with_concurrency(runs.filter(Boolean),4,run=>
          fetch_json(`/api/processing-status?run=${encodeURIComponent(run_identifier(run))}`,{signal:controller.signal}),controller.signal);
        if(context!==context_key())return;
        processing_view.trace_available=statuses.some(status=>status?.trace_available);
        processing_view.timing_available=statuses.some(status=>status?.timing_available);
        processing_view.available=processing_view.trace_available || processing_view.timing_available;
        processing_sync_visibility();
      }catch(error){if(error.name!=="AbortError")console.warn("Processing discovery failed",error);}
      finally{clearTimeout(deadline);if(probe_owner===controller)probe_owner=null;}
    };
    const restore_before=restore_maximized_chart, toggle_before=toggle_maximized_chart, group_before=toggle_chart_group;
    restore_maximized_chart=function(...args){const result=restore_before(...args);schedule_flush();return result;};
    toggle_maximized_chart=function(...args){const result=toggle_before(...args);schedule_flush();return result;};
    toggle_chart_group=function(...args){const result=group_before(...args);schedule_flush();return result;};
    by_id("charts_scroll")?.addEventListener("scroll",schedule_flush,{passive:true});
    window.addEventListener("resize",schedule_flush);
    document.addEventListener("visibilitychange",()=>{if(!document.hidden)schedule_flush();else{refresh_owner?.controller.abort();throughput_owner?.abort();}});
    document.addEventListener("click",event=>{if(event.target.closest('[data-detail-tab],#runs_nav,#workspace_nav'))schedule_flush();});
    window.instra_demand_runtime={chart_visible,pending_plots,family_signatures,schedule_flush,sampled_trace};
    schedule_flush();
  },400));
  // vvv THOG guard the complete table pipeline after all table owners have installed their wrappers
  window.addEventListener("load",()=>setTimeout(()=>{
    const previous=render_runs,signature=window.instra_sep21_repair_test_hooks?.runs_signature;
    if(!signature)return;
    let retained=null;
    const statistics={rendered:0,skipped:0};
    render_runs=function(...args) {
      const key=signature();
      if(retained===key && by_id("runs_body")?.childElementCount) { statistics.skipped++;return; }
      const result=previous.apply(this,args);
      retained=signature();statistics.rendered++;
      return result;
    };
    window.instra_demand_runtime.table_render_statistics=statistics;
  },2200));
  // ^^^ THOG
})();
// ^^^ THOG
