// vvv THOG real Plotly, pointer, download, recap and bounded CPU-evidence rendering in both browsers
"use strict";
const assert=require("node:assert/strict"),{chromium,firefox}=require("playwright");
const address=process.env.INSTRA_TEST_URL||"http://127.0.0.1:8765";
const runtime={staging_bytes:256,staging_peak_bytes:512,staging_limit_bytes:512,cpu_cache_bytes:1024,cpu_cache_peak_bytes:1024,workers_resolved:1,threads_per_worker_resolved:1};
const data={
 metadata:{schema_version:5,materialisation_device:"cpu_and_gpu",capture_duration_ms:10,capture_frequency_hz:10000,capture:{optimizer_update:2},warnings:[],run:{run_name:"cpu_fixture"},files:{transfers:"cpu_fixture_processing_transfers.csv",cpu_tasks:"cpu_fixture_processing_cpu_tasks.csv",cpu_memory:"cpu_fixture_processing_cpu_memory.csv",cpu_lifecycle:"cpu_fixture_processing_cpu_lifecycle.csv"}},
 intervals:[{start_us:0,end_us:2000,owner:"MAIN",layer:0,family:"O",operation:"consume",kernel_name:"mm"},{start_us:5000,end_us:8000,owner:"MAIN",layer:1,family:"O",operation:"consume",kernel_name:"mm"}],
 cpu_tasks:[{start_us:300,end_us:1700,clock_alignment_known:true,clock_uncertainty_ms:0.01,cpu_matrix_id:"snapshot:O:0",cpu_task_id:"task",snapshot_id:"snapshot",family:"O",layer_index:0,start_ns:1300000,end_ns:2700000},{start_us:null,end_us:null,clock_alignment_known:false,cpu_matrix_id:"precursor",start_ns:500,end_ns:700}],
 transfers:[{start_us:1700,end_us:1900,direction:"H2D",bytes:256,upload_id:"upload",snapshot_id:"snapshot",raw_start_ns:1700000,raw_end_ns:1900000,left_censored:false,right_censored:false}],
 samples:[],stream_resources:[],main_idle_intervals:[],summary:[],matrix_summary:{},operation_resource_stats:[],attribution_resource_stats:[],metric_audit:[],throughput:[],
 cpu_summary:{eligible_matrix_uses:4,full_hits:1,complete_misses:3,not_targeted_uses:12,unique_cpu_matrices:1,cpu_service_ms:1.4,h2d_busy_us:200,d2h_busy_us:null,cuda_copy_coverage:"available",late_upload_count:1,fallback_reasons:{upload_not_complete:3}},cpu_runtime:runtime,
};
const snapshot={schema_version:5,optimizer_update:2,pass_complete:true,pass_sequence:4,materialisation_device:"cpu_and_gpu",phase:"original_forward",attention_mode:"fused",target_matrix:2,target_layer:0,layer_indices:[0,1,2,3],n_layer:4,cpu_runtime:runtime,events:[],cpu_lifecycle:[{event:"cpu_matrix_ready",phase:"cpu_preparation",cpu_matrix_id:"snapshot:O:0",cpu_task_id:"task",snapshot_id:"snapshot",start_ns:1300000,end_ns:2700000},{event:"matrix_use",phase:"checkpoint_recompute",matrix_use_id:"replay-use",cpu_matrix_id:"snapshot:O:0",cpu_task_id:"task",snapshot_id:"snapshot"}]};
let sequence=0;
for(let layer=0;layer<4;layer++)for(const family of ["QKV","O","UP","DOWN"]){
 const common={layer_index:layer,family,phase:"original_forward",predicted_retained_bytes:256,snapshot_id:"snapshot",cpu_matrix_id:"snapshot:"+family+":"+layer,cpu_task_id:"task",matrix_use_id:"use:"+layer+":"+family,elapsed_ms:sequence/10};
 if(layer===0&&family==="O"){
  snapshot.events.push({...common,sequence:sequence++,event:"premat_submission",owner:"cpu_upload",new_state:"MATERIALISING"});
  snapshot.events.push({...common,sequence:sequence++,event:"available",owner:"cpu_upload",new_state:"AVAILABLE"});
 }
 const final_outcome=family!=="O"?"NOT TARGETED":layer===0?"FULL HIT":"COMPLETE MISS";
 snapshot.events.push({...common,sequence:sequence++,event:"matrix_acquired",owner:final_outcome==="FULL HIT"?"cpu_upload":"main",new_state:"CONSUMING",final_outcome,fallback_reason:layer?"upload_not_complete":""});
 snapshot.events.push({...common,sequence:sequence++,event:"consumed",owner:final_outcome==="FULL HIT"?"cpu_upload":"main",new_state:"CONSUMED",final_outcome});
}
snapshot.events.push({event:"matrix_acquired",family:"O",layer_index:0,sequence:sequence++,phase:"checkpoint_recompute",new_state:"CONSUMING",owner:"main",final_outcome:"COMPLETE MISS"});
snapshot.events.push({event:"pass_end",sequence:sequence++});
(async()=>{
 const selected_browsers=(process.env.INSTRA_TEST_BROWSERS||"chromium,firefox").split(",").map(name=>({chromium,firefox}[name]));
 assert.ok(selected_browsers.length&&selected_browsers.every(Boolean));
 for(const type of selected_browsers){
  const browser=await type.launch({headless:true,...(type===firefox?{env:{...process.env,MOZ_DISABLE_CONTENT_SANDBOX:"1"}}:{})});
  try{
   const page=await browser.newPage({viewport:{width:1600,height:1000},acceptDownloads:true}),errors=[];
   page.setDefaultTimeout(25000);page.on("pageerror",e=>{errors.push(e.message);console.error("CPU BROWSER PAGE ERROR",e.stack);});
   await page.route(/\/api\/processing\?/,route=>route.fulfill({json:{available:true,trace_available:true,revision:"cpu-v5",data}}));
   await page.route(/\/api\/local-file\?.*processing.*csv/,route=>route.fulfill({status:200,headers:{"Content-Type":"text/csv","Content-Disposition":"attachment"},body:"snapshot_id,cpu_task_id\nsnapshot,task\n"}));
   await page.goto(address+"/runs/fixture_00");
   await page.waitForFunction(()=>window.processing_cpu_test_hooks&&app.runs.length===12);
   await page.evaluate(async payload=>{local_set_detail_tab("charts");processing_view.charts_tab_visible=true;await processing_render(payload,true);},data);
   await page.waitForFunction(()=>by_id("processing_timeline_plot")?.data?.some(trace=>trace.meta?.operations_owner==="CPU"));
   const lanes=await page.locator("#processing_timeline_plot").evaluate(node=>node.layout.yaxis.ticktext);
   assert.deepEqual(lanes,["CPU","COPY D2H","COPY H2D","PREMAT","MAIN"]);
   assert.ok(await page.locator("#processing_cpu_summary_body").textContent().then(t=>t.includes("N/A: CPU materialisation")&&t.includes("clock alignment unknown")));
   const controls=await page.locator(".processing-operations-key-toggle").count();assert.ok(controls>0);
   const first=page.locator(".processing-operations-key-toggle").first();await first.click();await first.click();
   await page.evaluate(()=>processing_operations_test_hooks.install_layer_zoom(by_id("processing_timeline_plot"),processing_operations_test_hooks.layer_guides(processing_view.operations_payload.intervals),10));
   await page.evaluate(()=>by_id("processing_timeline_plot").emit("plotly_clickannotation",{annotation:{name:"processing-layer-0-top"}}));
   await page.waitForFunction(()=>!by_id("processing_operations_reset_zoom").disabled);
   await page.locator("#processing_operations_reset_zoom").click();
   assert.equal(await page.locator("#processing_operations_reset_zoom").isDisabled(),true);
   await page.locator('#processing_timeline_card .maximize-button').click();
   await page.waitForFunction(()=>by_id("processing_timeline_card").classList.contains("maximized"));
   await page.locator('#processing_timeline_card .maximize-button').click();
   await page.waitForFunction(()=>!by_id("processing_timeline_card").classList.contains("maximized"));
   for(const key of ["transfers","cpu_tasks","cpu_memory","cpu_lifecycle"]){
    const link=page.locator('.processing-downloads a[download="'+data.metadata.files[key]+'"]:visible');await link.waitFor({state:"visible"});assert.ok(await link.isVisible(),key);
    const download=page.waitForEvent("download");await link.click();assert.ok((await download).url().includes("cpu_fixture"));
   }
   const summary_maximize=page.locator('#processing_cpu_summary_card .maximize-button');
   // Exercise keyboard activation here as well as the real pointer activation
   // already checked on the timeline card. Restore below still uses a pointer click.
   await summary_maximize.press("Enter");
   await page.waitForFunction(()=>by_id("processing_cpu_summary_card").classList.contains("maximized"));
   await page.locator('#processing_cpu_summary_card .maximize-button').click();
   await page.waitForFunction(()=>!by_id("processing_cpu_summary_card").classList.contains("maximized"));
   // Restore schedules plot layout in animation frames; finish that layout before
   // locating the resize handle and sending a native pointer gesture.
   await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))));
   await page.evaluate(()=>document.activeElement?.blur());
   const resize=page.locator("#processing_cpu_summary_card .panel-resizer-corner");
   await resize.evaluate(handle=>handle.scrollIntoView({block:"center",behavior:"instant"}));
   await resize.evaluate(handle=>{
    window.cpu_resize_pointer_events=[];
    let pointer_id=null;
    const names=["pointerdown","pointermove","pointerup","pointercancel"];
    function observe(event){
     if(event.type==="pointerdown"){if(event.target!==handle)return;pointer_id=event.pointerId;}
     if(event.pointerId!==pointer_id)return;
     cpu_resize_pointer_events.push({type:event.type,trusted:event.isTrusted});
     if(event.type==="pointerup"||event.type==="pointercancel"){
      pointer_id=null;for(const name of names)window.removeEventListener(name,observe,true);
     }
    }
    for(const name of names)window.addEventListener(name,observe,true);
   });
   await resize.dragTo(page.locator("#processing_cpu_summary_body"));
   await page.waitForFunction(()=>cpu_resize_pointer_events.some(event=>event.type==="pointerup"&&event.trusted));
   const pointer_events=await page.evaluate(()=>window.cpu_resize_pointer_events);
   assert.ok(pointer_events.some(event=>event.type==="pointerdown"&&event.trusted));
   assert.ok(pointer_events.some(event=>event.type==="pointermove"&&event.trusted));
   assert.ok(pointer_events.some(event=>event.type==="pointerup"&&event.trusted));
   await page.waitForFunction(()=>JSON.parse(localStorage.getItem("thog2_local_panel_sizes")||"{}").processing_cpu_summary?.height>0);
   const recap=await page.evaluate(s=>{
    premat_start_snapshot(s);premat_finish_playback();premat_show_panel("inspector");
    const model=premat_build_model(s),rows=premat_inspector_rows(s,model);
    return {outcomes:rows.map(row=>row.outcome),csv:premat_detailed_history_csv([s]),raw:premat_raw_event_history_csv([s]),frames:model.frames.map(frame=>frame.frame_state)};
   },snapshot);
   assert.equal(recap.outcomes.filter(x=>x==="FULL HIT").length,1);assert.equal(recap.outcomes.filter(x=>x==="COMPLETE MISS").length,3);assert.equal(recap.outcomes.filter(x=>x==="NOT TARGETED").length,12);
   assert.ok(recap.frames.includes("UPLOAD PENDING"));assert.match(recap.csv,/snapshot_id,cpu_task_id/);assert.match(recap.csv,/CPU|cpu_matrix_ready/);assert.match(recap.raw,/checkpoint_recompute/);
   await page.locator("#runner_nav").click();await page.waitForFunction(()=>window.instra_runner_test_hooks);
   const conditional=await page.evaluate(()=>({
    inactive:instra_runner_test_hooks.cpu_field_active("--premat_cpu_workers",{"--premat_materialisation_device":["gpu"]}),
    active:instra_runner_test_hooks.cpu_field_active("--premat_cpu_workers",{"--premat_materialisation_device":["cpu_and_gpu"]}),
    lead:instra_runner_test_hooks.cpu_field_active("--premat_cpu_transfer_lead_ms",{"--premat_materialisation_device":["cpu_and_gpu"],"--premat_cpu_transfer_timing":["predicted_gemm_start"]}),
    gpu:instra_runner_test_hooks.cpu_field_active("--premat_timing",{"--premat_materialisation_device":["cpu_and_gpu"]}),
   }));assert.deepEqual(conditional,{inactive:false,active:true,lead:true,gpu:false});
   await page.locator("#runs_nav").click();
   await page.evaluate(()=>local_set_detail_tab("charts"));
   const before=await page.evaluate(()=>({dom:document.querySelectorAll("*").length,trace:by_id("processing_timeline_plot").data.length}));
   const stop=Date.now()+Number(process.env.INSTRA_CPU_SOAK_SECONDS||60)*1000;
   let iterations=0;
   while(Date.now()<stop){await page.evaluate(async payload=>{await processing_render(payload,true);},data);iterations++;await page.waitForTimeout(400);}
   const after=await page.evaluate(()=>({dom:document.querySelectorAll("*").length,trace:by_id("processing_timeline_plot").data.length}));
   assert.equal(after.trace,before.trace);assert.ok(after.dom<=before.dom+60,JSON.stringify({before,after}));assert.deepEqual(errors,[]);
   await page.unrouteAll({behavior:"wait"});
   console.log(JSON.stringify({browser:type.name(),cpu_evidence:true,recap:true,downloads:4,zoom:true,maximize:true,legend:true,runner:true,soak_seconds:Number(process.env.INSTRA_CPU_SOAK_SECONDS||60),iterations,before,after,page_errors:errors}));
  }finally{await browser.close();}
 }
})().catch(error=>{console.error(error.stack);process.exit(1);});
// ^^^ THOG
