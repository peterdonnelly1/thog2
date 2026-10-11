// vvv THOG qualified prediction and censored CPU overlays preserve the exclusive update chart
"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),vm=require("node:vm"),path=require("node:path");
const exclusive={name:"exclusive phases",x:[1,2,3]};
const context={window:{},by_id:()=>null,document:{querySelector:()=>null},processing_view:{},
  processing_set_downloads:()=>{},processing_render:async()=>{},processing_sync_visibility:()=>{},
  processing_update_timing_phase_traces:()=>[exclusive],processing_plot:async()=>{},processing_escape:String};
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(__dirname,"../sheet/local_dashboard_assets/dashboard_processing_cpu.js"),"utf8"),context);
const use={event:"matrix_use",matrix_use_id:"use",upload_id:"upload",host_time_ns:1_004_000,
  predicted_gemm_start_ns:1_006_000,intended_gpu_available_ns:1_003_000,actual_gpu_available_ns:1_003_500,readiness_check_ns:1_004_000};
const entry={colour:"#123456",timing:{host_update_start_ns:1_000_000,cpu_overlay:{available:true,additive:false,
  tasks:[{cpu_matrix_id:"unfinished",host_start_ms:-0.1,host_end_ms:1,right_censored:true},
    {cpu_matrix_id:"precursor",host_start_ms:-1,host_end_ms:-0.5},{cpu_matrix_id:"unknown"}],
  lifecycle:[use,{...use,event:"upload_complete_observed"},{event:"matrix_use",matrix_use_id:"unknown",predicted_gemm_start_ns:null}]}}};
const traces=context.processing_update_timing_phase_traces([entry]);
assert.equal(traces[0],exclusive);
const tasks=traces.find(trace=>trace.name==="CPU tasks (overlay)");
assert.deepEqual(Array.from(tasks.x),[1]);assert.deepEqual(Array.from(tasks.base),[0]);
assert.ok(tasks.hovertext[0].includes("right-censored"));
for(const [name,position] of [["Estimated GEMM start",0.006],["Intended upload availability",0.003],["Observed upload availability",0.0035],["Readiness check",0.004]]){
  const trace=traces.find(item=>item.name===name);
  assert.deepEqual(Array.from(trace.x),[position]);assert.equal(trace.marker.color,"#123456");
}
assert.equal(context.processing_update_timing_phase_traces([{timing:{schema_version:2}}]).length,1);
assert.equal(context.processing_update_timing_phase_traces([{timing:{host_update_start_ns:1_000_000,cpu_overlay:{available:true,tasks:[],lifecycle:[{predicted_gemm_start_ns:null}]}}}]).length,1);
console.log("PASS independent prediction/readiness markers, censored CPU intervals, unknown-clock omission and legacy phase totals");
// ^^^ THOG
