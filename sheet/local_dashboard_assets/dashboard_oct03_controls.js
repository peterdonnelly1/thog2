// vvv THOG persistent Grid tones, table gestures, exact visible-curve exports and ephemeral Processing navigation state
"use strict";
window.addEventListener("load", () => setTimeout(() => {
  const style=document.createElement("style");
  style.textContent=`
    .runs-table th[data-instra-column-key="name"] { position:sticky !important; top:0; z-index:5; }
    .runs-table td.name-column { position:static !important; vertical-align:middle; }
    .runs-table th[data-instra-column-key="state"] { text-align:center !important; }
    .runs-table th[data-instra-column-key="gb"] { text-transform:none !important; }
    .runs-table th[data-instra-column-key] { cursor:grab; }
    .runs-table th.column-drag-target { box-shadow:inset 3px 0 #1590a8; }
    .runs-table td.menu-column { white-space:nowrap; }
    .file-delete-button, .runner-recipe-delete { color:#b3262e !important; }
    .file-delete-button svg, .runner-recipe-delete svg {
      fill:none; stroke:#b3262e !important; width:18px; height:18px; stroke-width:1.8; stroke-linecap:round; stroke-linejoin:round;
    }
    .run-column-resizer { position:absolute; z-index:6; right:0; top:0; bottom:0; width:7px; cursor:col-resize; touch-action:none; }
    .run-column-resizer:hover { border-right:2px solid #1590a8; }
    body.resizing-run-column { cursor:col-resize; user-select:none; }
    .chart-download-options { position:relative; font-size:11px; flex:0 0 auto; }
    .chart-download-options > summary { cursor:pointer; padding:5px; list-style:none; }
    .chart-card.maximized .chart-card-actions { padding-right:85px !important; }
    .chart-download-options[open] { z-index:90; }
    .chart-download-menu { position:absolute; right:0; top:100%; min-width:255px; padding:5px; background:#fff; border:1px solid #cfd2d8; border-radius:4px; box-shadow:0 4px 12px #0002; }
    .chart-download-menu button { display:block; width:100%; text-align:left; white-space:nowrap; padding:7px; border:0; background:#fff; cursor:pointer; font-size:11px; }
    .chart-download-menu button:hover { background:#eef0f3; }
    #charts_scroll.maximized-mode > .chart-group:not(.maximized) { display:none !important; }
    #processing_chart_group.maximized { display:flex; flex-direction:column; box-sizing:border-box; }
    .processing-group.maximized > .processing-grid.is-maximized { flex:1 1 0; min-height:0; padding-bottom:0; box-sizing:border-box; }
    #processing_chart_group.maximized > .processing-grid.chart-grid.is-maximized > .processing-card.chart-card.maximized { flex:1 1 0 !important; height:auto !important; max-height:100% !important; min-height:0 !important; box-sizing:border-box; }
    .processing-card.maximized > .processing-plot-shell, .processing-card.maximized .plot-mount { min-height:0 !important; height:100%; box-sizing:border-box; }
  `;
  document.head.appendChild(style);
  // vvv THOG Space toggles the focused row checkbox without scrolling or moving run selection
  document.addEventListener("keydown",event=>{
    if(![" ","Spacebar"].includes(event.key) || event.altKey || event.ctrlKey || event.metaKey)return;
    const target=event.target;
    if(target.closest?.('.settings-overlay:not([hidden]),dialog[open]') || target.isContentEditable)return;
    let row=target.closest?.(".runs-table tr[data-run-id]");
    if(!row && target===document.body && app.current_run_id && !app.workspace_mode && !by_id("workspace")?.hidden && instra_charts_visible())
      row=document.querySelector(`.runs-table tr[data-run-id="${CSS.escape(app.current_run_id)}"]`);
    if(!row || row.offsetParent===null || target.matches?.('textarea,select,input:not([type="checkbox"]),button:not(.run-link),a'))return;
    event.preventDefault();event.stopImmediatePropagation();
    if(event.repeat)return;
    const checkbox=row.querySelector('.check-column input[type="checkbox"]');
    if(!checkbox)return;
    checkbox.checked=!checkbox.checked;checkbox.dispatchEvent(new Event("change",{bubbles:true}));
    if(target===document.body)row.focus({preventScroll:true});
  },true);
  // ^^^ THOG

  function rgb_to_hsl(rgb) {
    const [red,green,blue]=rgb.map(value=>value/255),maximum=Math.max(red,green,blue),minimum=Math.min(red,green,blue);
    const delta=maximum-minimum,lightness=(maximum+minimum)/2;
    let hue=0;
    if(delta) hue=maximum===red ? ((green-blue)/delta+6)%6 : maximum===green ? (blue-red)/delta+2 : (red-green)/delta+4;
    return {hue:hue*60,saturation:delta ? 100*delta/(1-Math.abs(2*lightness-1)) : 0,lightness:lightness*100};
  }
  function grid_members(run_id) {
    const key=grid_identity(run_for_id(run_id));
    return key ? app.runs.filter(run=>grid_identity(run)===key) : [];
  }
  function refresh_colours(run_id) {
    grid_palette_source=null;
    render_runs();render_run_heading();
    queue_current_recolour(run_id);
    window.__thog2_metric_groups?.invalidate?.();
    window.__thog2_metric_groups?.refresh?.();
  }
  function reset_grid_colours(run_id,all) {
    const members=all ? grid_members(run_id) : [run_for_id(run_id)].filter(Boolean);
    for(const run of members)delete app.colours[run_identifier(run)];
    save_json("thog2_local_run_colours",app.colours);
    close_run_menu();refresh_colours(run_id);
  }
  let picker_scope="run";
  const picker_title=document.createElement("strong");picker_title.textContent="Run colour";
  by_id("colour_popover").prepend(picker_title);
  const open_picker_before=open_colour_picker, set_picker_before=set_picker_colour;
  open_colour_picker=function(run_id,anchor) {
    picker_scope="run";picker_title.textContent="Run colour";
    return open_picker_before(run_id,anchor);
  };
  set_picker_colour=function(rgb,persist=true) {
    if(picker_scope!=="grid" || !persist || !app.colour_run_id)return set_picker_before(rgb,persist);
    const run_id=app.colour_run_id,key=grid_identity(run_for_id(run_id));
    if(!key)return;
    set_picker_before(rgb,false);
    grid_tones[key]=rgb_to_hsl(rgb);
    save_json("thog2_grid_tones_v1",grid_tones);
    for(const run of grid_members(run_id))delete app.colours[run_identifier(run)];
    save_json("thog2_local_run_colours",app.colours);
    refresh_colours(run_id);
  };
  const menu=by_id("run_menu");
  function menu_button(id,text,action) {
    const button=document.createElement("button");button.id=id;button.type="button";button.setAttribute("role","menuitem");
    button.textContent=text;button.addEventListener("click",action);menu.insertBefore(button,by_id("delete_run").previousElementSibling);
    return button;
  }
  const reset_run=menu_button("reset_run_grid_colour","Reset this run to Grid colour",()=>reset_grid_colours(app.menu_run_id,false));
  const reset_grid=menu_button("reset_grid_colours","Reset all manual colours in this Grid",()=>reset_grid_colours(app.menu_run_id,true));
  const change_grid=menu_button("change_grid_tone","Change Grid centre colour…",()=>{
    const run_id=app.menu_run_id,anchor=document.querySelector(`tr[data-run-id="${CSS.escape(run_id)}"] .colour-dot`);
    close_run_menu();if(!anchor)return;
    open_picker_before(run_id,anchor);picker_scope="grid";
    picker_title.textContent="Grid centre colour";
    const tone=grid_tones[grid_identity(run_for_id(run_id))];
    const probe=document.createElement("span");probe.style.color=tone ? `hsl(${tone.hue} ${tone.saturation}% ${tone.lightness}%)` : `hsl(${grid_hues[grid_identity(run_for_id(run_id))]} 56% 54%)`;
    document.body.appendChild(probe);
    const channels=getComputedStyle(probe).color.match(/[\d.]+/g)?.slice(0,3).map(Number);probe.remove();
    if(channels?.length===3)set_picker_colour(channels,false);
  });
  change_grid.title="Permanently change this Grid's tone range and all its run colours";
  const open_menu_before=open_run_menu;
  open_run_menu=function(run_id,anchor) {
    const result=open_menu_before(run_id,anchor),has_grid=Boolean(grid_identity(run_for_id(run_id)));
    reset_run.hidden=reset_grid.hidden=change_grid.hidden=!has_grid;
    const rect=anchor.getBoundingClientRect();
    menu.style.top=`${Math.max(8,Math.min(rect.bottom+4,window.innerHeight-menu.scrollHeight-8))}px`;
    return result;
  };

  const layout_key="thog2_run_columns_layout_v1";
  const layout=load_json(layout_key,{order:[],widths:{}});
  if(!Array.isArray(layout.order))layout.order=[];
  if(!layout.widths || typeof layout.widths!=="object")layout.widths={};
  let dragging_key=null,suppress_heading_click_until=0;
  const default_widths=new Map();
  const save_layout=()=>save_json(layout_key,layout);
  function apply_table_layout() {
    const table=document.querySelector(".runs-table"),header_row=table?.querySelector("thead tr");
    if(!header_row)return;
    const headers=[...header_row.children].filter(header=>header.dataset.instraColumnKey);
    const defaults=headers.map(header=>header.dataset.instraColumnKey);
    const keys=[...new Set([...layout.order.filter(key=>defaults.includes(key)),...defaults])];
    // vvv THOG migrate saved layouts without placing the new grid control at the far right
    if(!layout.order.includes("grid_visibility") && keys.includes("grid_visibility")) {
      keys.splice(keys.indexOf("grid_visibility"),1);keys.splice(keys.indexOf("visibility"),0,"grid_visibility");
    }
    // ^^^ THOG
    for(const row of [header_row,...table.querySelectorAll("tbody tr[data-run-id]")]) {
      const cells=new Map([...row.children].map(cell=>[cell.dataset.instraColumnKey,cell]));
      if(keys.every((key,index)=>row.children[index]===cells.get(key)))continue;
      const fragment=document.createDocumentFragment();
      for(const key of keys)if(cells.has(key))fragment.appendChild(cells.get(key));
      row.appendChild(fragment);
    }
    let total=0;
    for(const header of headers) {
      const key=header.dataset.instraColumnKey;
      if(!default_widths.has(key))default_widths.set(key,key==="menu" ? 36 : Number.parseFloat(header.style.width)||56);
      const stored=Number(layout.widths[key]);
      const width=Number.isFinite(stored) && stored>=28 ? Math.min(2200,stored) : default_widths.get(key);
      for(const property of ["width","min-width","max-width"])header.style.setProperty(property,`${width}px`,"important");
      if(key==="name")table.style.setProperty("--instra-run-name-width",`${width}px`);
      if(!header.hidden)total+=width;
      header.querySelector(".run-name-column-resizer")?.remove();
      if(header.dataset.instraGesturesBound==="true")continue;
      header.dataset.instraGesturesBound="true";header.draggable=true;
      const handle=document.createElement("span");handle.className="run-column-resizer";
      handle.setAttribute("role","separator");handle.setAttribute("aria-orientation","vertical");handle.setAttribute("aria-label",`Resize ${header.textContent.trim()} column`);
      handle.addEventListener("pointerdown",event=>{
        if(event.button!==0)return;event.preventDefault();event.stopPropagation();
        const start_x=event.clientX,start_width=header.getBoundingClientRect().width;
        header.draggable=false;document.body.classList.add("resizing-run-column");
        const move=event=>{layout.widths[key]=Math.round(Math.max(28,Math.min(2200,start_width+event.clientX-start_x)));apply_table_layout();};
        const finish=()=>{
          window.removeEventListener("pointermove",move,true);window.removeEventListener("pointerup",finish,true);window.removeEventListener("pointercancel",finish,true);
          header.draggable=true;document.body.classList.remove("resizing-run-column");suppress_heading_click_until=Date.now()+300;
          if(key==="name")localStorage.setItem("thog2_local_run_name_column_width",String(layout.widths[key]));
          save_layout();
        };
        window.addEventListener("pointermove",move,true);window.addEventListener("pointerup",finish,true);window.addEventListener("pointercancel",finish,true);
      });
      handle.addEventListener("dblclick",event=>{event.preventDefault();event.stopPropagation();delete layout.widths[key];save_layout();render_runs();apply_table_layout();});
      header.appendChild(handle);
      header.addEventListener("dragstart",event=>{dragging_key=key;event.dataTransfer.setData("text/plain",key);event.dataTransfer.effectAllowed="move";});
      header.addEventListener("dragover",event=>{if(dragging_key){event.preventDefault();header.classList.add("column-drag-target");}});
      header.addEventListener("dragleave",()=>header.classList.remove("column-drag-target"));
      header.addEventListener("drop",event=>{
        event.preventDefault();header.classList.remove("column-drag-target");
        if(!dragging_key || dragging_key===key)return;
        const order=[...header_row.children].map(header=>header.dataset.instraColumnKey).filter(key=>key!==dragging_key);
        order.splice(order.indexOf(key),0,dragging_key);layout.order=order;save_layout();apply_table_layout();
        suppress_heading_click_until=Date.now()+300;dragging_key=null;
      });
      header.addEventListener("dragend",()=>{dragging_key=null;header_row.querySelectorAll(".column-drag-target").forEach(node=>node.classList.remove("column-drag-target"));});
    }
    table.style.setProperty("min-width",`${total}px`,"important");table.style.setProperty("width",`${total}px`,"important");
    // vvv THOG use the toolbar bin for checkbox selections; the run menu retains single-run deletion
    for(const trash of table.querySelectorAll(".run-row-trash"))trash.remove();
    // ^^^ THOG
  }
  document.querySelector(".runs-table thead")?.addEventListener("click",event=>{
    if(Date.now()<suppress_heading_click_until || event.target.closest(".run-column-resizer")){event.preventDefault();event.stopImmediatePropagation();}
  },true);
  const render_runs_before=render_runs;
  // vvv THOG catalogue refreshes preserve keyboard focus on the same run and control
  render_runs=function(...args){
    const focused=document.activeElement,focused_row=focused?.closest?.(".runs-table tr[data-run-id]");
    const run_id=focused_row?.dataset.runId;
    const selector=focused?.matches?.('input[type="checkbox"]') ? '.check-column input' :
      focused?.matches?.('.run-link') ? '.run-link' : focused?.matches?.('.grid-visibility-button') ? '.grid-visibility-button' :
      focused?.matches?.('.eye-button') ? '.visibility-column .eye-button' : null;
    const result=render_runs_before.apply(this,args);apply_table_layout();
    if(run_id && !focused.isConnected) {
      const row=document.querySelector(`.runs-table tr[data-run-id="${CSS.escape(run_id)}"]`);
      (selector ? row?.querySelector(selector) : row)?.focus({preventScroll:true});
    }
    return result;
  };
  // ^^^ THOG
  by_id("instra_columns_popover")?.addEventListener("change",()=>queueMicrotask(apply_table_layout));

  let processing_epoch=0;
  function reset_processing_zoom() {
    processing_epoch++;
    for(const mount of document.querySelectorAll("#processing_chart_group .plot-mount")) {
      mount._processing_layer_zoom_range=null;mount._processing_layer_zoom_run_id=null;
      if(mount.dataset.plotReady!=="true")continue;
      const data=mount.layout?.xaxis;
      if(data){delete data.range;data.autorange=true;}
      mount._instra_last_resize=null;
    }
    const scrollbar=by_id("processing_operations_hscroll");if(scrollbar)scrollbar.hidden=true;
    const reset=by_id("processing_operations_reset_zoom");if(reset)reset.disabled=true;
    processing_view.revision=null;
  }
  const select_run_before=select_run;
  select_run=function(run_id,...args) {
    if(run_id!==app.current_run_id) {
      reset_processing_zoom();
      for(const key of ["trace_available","resource_available","compatibility_available","timing_available"])processing_view[key]=false;
      by_id("processing_chart_group").hidden=true;
    }
    const result=select_run_before.call(this,run_id,...args);
    void refresh_premat();
    return result;
  };
  const processing_plot_before=processing_plot;
  processing_plot=function(id,traces,layout) {
    return processing_plot_before(id,traces,{...layout,uirevision:`${app.current_run_id}:${processing_epoch}:${id}`});
  };
  document.addEventListener("click",event=>{
    if(event.target.closest('[data-detail-tab], [data-runner-tab], #runs_nav, #workspace_nav, #runner_nav, #networks_nav, #settings_nav'))reset_processing_zoom();
  },true);
  const processing_visibility_before=processing_sync_visibility;
  processing_sync_visibility=function(...args) {
    const result=processing_visibility_before.apply(this,args);
    if(!premat_run_enabled())for(const id of ["processing_contention_card","processing_matrix_summary_card","processing_compatibility_card"]){const card=by_id(id);if(card)card.hidden=true;}
    return result;
  };

  const array=value=>Array.isArray(value) ? value : ArrayBuffer.isView(value) ? Array.from(value) : [];
  function chart_export_payload(card) {
    const name=card.dataset.chart,mount=card.querySelector(".plot-mount"),figure=figure_for_chart(name);
    const sources=figure?.data || mount?.data || [];
    const metric=card.id.includes("throughput") || name.includes("throughput") ? "tokens_throughput" : "training_loss";
    const series=[];
    const trace_run_id=trace=>String(trace.meta?.instra_workspace_run_id || trace.meta?.instra_run_id || app.current_run_id || "");
    for(const trace of sources) {
      const displayed=(mount?.data || []).find(item=>item.name===trace.name && trace_run_id(item)===trace_run_id(trace));
      if(displayed?.visible===false || displayed?.visible==="legendonly" || trace.visible===false || trace.visible==="legendonly")continue;
      const run_id=trace_run_id(trace);
      if(app.workspace_mode && !is_visible(run_id))continue;
      const x=array(trace.x),y=array(trace.y);if(!x.length || !y.length)continue;
      const run=run_for_id(run_id);
      const variants=trace.thog2_x_variants || {};
      const shown_x=array(displayed?.x);
      series.push({run_id,run_name:run?.artifact_name || run?.run_name || run_id,name:trace.name || metric,
        colour:colour_for_run(run_id),x,y,displayed_x:shown_x.length===y.length ? shown_x : x,
        x_variants:variants,point_sources:array(trace.customdata)});
    }
    return {schema:"instra.visible_curves.v1",metric,exported_at:new Date().toISOString(),
      selected_run_id:app.current_run_id,selected_run_name:current_run()?.artifact_name || current_run()?.run_name || app.current_run_id || "multiview",
      x_axis:mount?.layout?.xaxis?.title?.text || mount?.layout?.xaxis?.title || "step",
      y_axis:mount?.layout?.yaxis?.title?.text || mount?.layout?.yaxis?.title || metric,
      data_scope:"All retained raw points of every visible curve; zoom does not truncate the export.",series};
  }
  function save_blob(blob,name) {
    const url=URL.createObjectURL(blob),link=document.createElement("a");link.href=url;link.download=name;
    document.body.appendChild(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  async function download_chart(card,format) {
    const payload=chart_export_payload(card);
    if(!payload.series.length){show_toast("No visible curve data to download.");return;}
    const filename=`${payload.selected_run_name.replace(/[<>:"/\\|?*\x00-\x1f]/g,"_")}_${payload.metric}.${format}`;
    if(format==="json")save_blob(new Blob([JSON.stringify(payload,null,2)],{type:"application/json"}),filename);
    else {
      try {
        const controller=new AbortController(),deadline=setTimeout(()=>controller.abort(),30000);
        let response;
        try{response=await fetch("/api/chart-export",{method:"POST",signal:controller.signal,headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});}
        finally{clearTimeout(deadline);}
        if(!response.ok){const error=await response.json();throw new Error(error.error || `HTTP ${response.status}`);}
        save_blob(await response.blob(),filename);
      }catch(error){show_toast(`Chart export failed: ${error.message}`);}
    }
  }
  function ensure_downloads() {
    for(const card of document.querySelectorAll('.chart-card[data-chart]')) {
      const title=card.querySelector("h2")?.textContent || "";
      if(!card.dataset.chart.includes("throughput") && !(card.dataset.metricChartId?.toLowerCase().includes("loss")) && !/^loss$/i.test(title.trim()))continue;
      if(card.querySelector(".chart-download-options"))continue;
      const details=document.createElement("details");details.className="chart-download-options";
      details.innerHTML='<summary title="Download visible curve data">Download</summary><div class="chart-download-menu"></div>';
      for(const [format,label] of [["json","Download as JSON"],["xls","Download as Excel 97-2003 Workbook"]]) {
        const button=document.createElement("button");button.type="button";button.textContent=label;
        button.addEventListener("click",()=>{details.open=false;void download_chart(card,format);});details.lastElementChild.appendChild(button);
      }
      (card.querySelector(".chart-card-actions") || card.querySelector(".chart-card-header"))?.appendChild(details);
    }
  }
  const metric_refresh_before=window.__thog2_metric_groups.refresh;
  window.__thog2_metric_groups.refresh=async function(...args){try{return await metric_refresh_before.apply(this,args);}finally{ensure_downloads();}};
  const processing_throughput_before=processing_render_throughput;
  processing_render_throughput=async function(...args){const result=await processing_throughput_before.apply(this,args);ensure_downloads();return result;};
  const render_plot_before=render_plot;
  render_plot=function(mount,figure,name,...args){
    const card=mount?.closest(".local-metric-card");
    if(card)delete card.dataset.instraDeferredMetric;
    ensure_downloads();
    return render_plot_before.call(this,mount,figure,name,...args);
  };
  const restore_before=restore_maximized_chart;
  restore_maximized_chart=function(...args){const result=restore_before.apply(this,args),run_id=app.current_run_id;
    requestAnimationFrame(async()=>{
      for(const card of document.querySelectorAll('.local-metric-card[data-instra-deferred-metric="true"]')) {
        if(run_id!==app.current_run_id || app.maximized_chart)return;
        const figure=app.dynamic_chart_figures[card.dataset.chart],mount=card.querySelector(".plot-mount");
        if(figure && mount?.offsetParent!==null)try{await render_plot(mount,figure,card.dataset.chart);}catch(_){}
      }
    });
    return result;
  };
  ensure_downloads();apply_table_layout();
  window.instra_oct03_controls={apply_table_layout,reset_grid_colours,chart_export_payload,reset_processing_zoom,rgb_to_hsl};
},350));
// ^^^ THOG
