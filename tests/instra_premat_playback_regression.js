// vvv THOG CPU event recovery, long accumulation playback, pause, Skip and new-data interruptions
'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const {JSDOM}=require('jsdom');
const dom=new JSDOM(fs.readFileSync('sheet/local_dashboard_assets/index.html','utf8'),{url:'http://localhost'});
const timers=new Map();let timer_serial=0;
const context={window:dom.window,document:dom.window.document,by_id:id=>dom.window.document.getElementById(id),
  app:{current_run_id:'fixture',runs:[]},current_run:()=>({configuration:{premat:'enabled'}}),
  fetch_json:async()=>({snapshots:[]}),AbortController,URL,Blob,
  setInterval:()=>0,
  setTimeout:callback=>{timers.set(++timer_serial,callback);return timer_serial;},clearTimeout:id=>timers.delete(id),console};
vm.createContext(context);
vm.runInContext(fs.readFileSync('sheet/local_dashboard_assets/dashboard_premat.js','utf8'),context);
const run=code=>vm.runInContext(code,context);
function snapshot(step,micro,mode='complete_step') {
  return {optimizer_update:step,micro_step:micro,pass_complete:true,playback_mode:mode,attention_mode:'fused',layer_indices:[0],
    events:[{sequence:1,event:'materialising',family:'O',layer_index:0,owner:'main',state:'MATERIALISING'},
      {sequence:2,event:'consuming',family:'O',layer_index:0,owner:'main',state:'CONSUMING'},
      {sequence:3,event:'consumed',family:'O',layer_index:0,owner:'main',state:'CONSUMED'},{sequence:4,event:'pass_end'}]};
}
context.snapshot=snapshot;
run("premat_view.run_id = 'fixture';");
run('premat_receive_snapshot(snapshot(1,0));');
const original_timer=run('premat_view.playback_timer');
run('for(let micro=1;micro<6;micro++)premat_receive_snapshot(snapshot(1,micro));premat_receive_snapshot(snapshot(2,0));');
assert.equal(run('premat_view.active_snapshot.optimizer_update'),1);
assert.equal(run('premat_view.active_snapshot.micro_step'),0);
assert.equal(run('premat_view.playback_timer'),original_timer,'arrival must not cancel or reset the active frame');
const played=[];
for(let micro=0;micro<6;micro++) {
  played.push(run('premat_view.active_snapshot.micro_step'));
  while(run('premat_view.active_snapshot.optimizer_update===1 && premat_view.active_snapshot.micro_step=== '+micro+' && premat_view.playback_running'))run('premat_advance_playback()');
}
assert.deepEqual(played,[0,1,2,3,4,5]);
assert.equal(run('premat_view.active_snapshot.optimizer_update'),2);
run('premat_reset();premat_receive_snapshot(snapshot(10,0));for(let micro=1;micro<180;micro++)premat_receive_snapshot(snapshot(10,micro));for(let step=11;step<200;step++)premat_receive_snapshot(snapshot(step,0));');
assert.equal(run('premat_view.pending_snapshots.filter(s=>s.optimizer_update===10).length'),179,'protect all active-step siblings under queue pressure');
assert.ok(run('premat_view.pending_snapshots.length')<308,'future backlog must remain bounded');
for(let micro=0;micro<180;micro++) {
  assert.equal(run('premat_view.active_snapshot.micro_step'),micro);
  run('premat_finish_playback()');
}
assert.ok(run('premat_view.active_snapshot.optimizer_update')>10);
run('premat_reset();premat_receive_snapshot(snapshot(1,0));premat_toggle_playback();for(let micro=1;micro<6;micro++)premat_receive_snapshot(snapshot(1,micro));premat_receive_snapshot(snapshot(2,3));');
assert.equal(run('premat_view.playback_timer'),null,'pause must cancel the timer');
run('premat_skip_playback();');
assert.equal(run('premat_view.active_snapshot.optimizer_update'),2);
assert.equal(run('premat_view.active_snapshot.micro_step'),3);
assert.equal(run('premat_view.pending_snapshots.length'),0);
assert.equal(run('premat_view.playing'),true);
assert.equal(run('premat_view.playback_mode'),'complete_step','Skip must preserve the selected mode');
run('premat_skip_playback();');
assert.equal(run('premat_view.playback_running'),false,'Skip without queued data must abandon the current animation');
run('premat_receive_snapshot(snapshot(3,4));');
assert.equal(run('premat_view.active_snapshot.optimizer_update'),3);
run('premat_reset();premat_receive_snapshot(snapshot(1,0,"latest"));premat_receive_snapshot(snapshot(2,4,"latest"));');
assert.equal(run('premat_view.active_snapshot.micro_step'),4,'latest mode must interrupt immediately');
run('premat_toggle_playback();premat_receive_snapshot(snapshot(3,1,"latest"));premat_receive_snapshot(snapshot(4,2,"latest"));');
assert.equal(run('premat_view.pending_snapshots.length'),1,'paused latest mode must coalesce incoming data');
run('premat_toggle_playback();');
assert.equal(run('premat_view.active_snapshot.optimizer_update'),4,'resuming Latest must select the newest paused capture immediately');
run('premat_toggle_playback_mode();premat_receive_snapshot(snapshot(5,0,"latest"));');
assert.equal(run('premat_view.playback_mode'),'complete_step','chart button overrides the hyperparameter for this run');
run('premat_reset();');assert.equal(run('premat_view.playback_override'),null,'run changes clear overrides');

// Recreate the six-microstep reporting failure with explicit authoritative outcomes; never infer hits.
for(const corrupted of [false,true]) {
  const hits=[0,0,29,63,64,63];
  const snapshots=hits.map((count,micro)=>{
    const events=[];let sequence=0;
    for(let matrix=0;matrix<64;matrix++) {
      const family=['QKV','O','UP','DOWN'][matrix%4],layer_index=Math.floor(matrix/4);
      const final_outcome=matrix<count?'FULL HIT':'COMPLETE MISS';
      const metadata={family,layer_index,final_outcome,owner:matrix<count?'cpu_upload':'main',phase:'original_forward',
        fallback_reason:matrix<count?null:micro<2?'cpu_preparation_not_ready':'upload_not_complete'};
      events.push({...metadata,sequence:++sequence,event:corrupted?'cpu_matrix_ready':'matrix_acquired',event_type:'matrix_acquired',state:'CONSUMING'});
      events.push({...metadata,sequence:++sequence,event:corrupted?'cpu_matrix_ready':'consumed',event_type:'consumed',state:'CONSUMED'});
    }
    return {optimizer_update:1,micro_step:micro,pass_complete:true,materialisation_device:'cpu_and_gpu',attention_mode:'fused',layer_indices:Array.from({length:16},(_,i)=>i),events};
  });
  context.snapshots=snapshots;
  const counts=JSON.parse(run('JSON.stringify(premat_history_model(snapshots).rows.sort((left,right)=>left.micro_step-right.micro_step).map(row=>row.totals))'));
  assert.deepEqual(counts.map(row=>row.full_hits),hits);
  assert.equal(counts.reduce((total,row)=>total+row.full_hits,0),219);
  assert.equal(counts.reduce((total,row)=>total+row.complete_misses,0),165);
  const csv=run('premat_detailed_history_csv(snapshots)');
  assert.equal((csv.match(/NOT YET REACHED/g)||[]).length,0);
  assert.equal(snapshots[2].events[0].event,corrupted?'cpu_matrix_ready':'matrix_acquired','history recovery must not mutate raw evidence');
}
assert.equal(run('premat_snapshot_label(snapshot(11,5))'),'Microstep 6 of Optimiser Step 11');
// Skip with no queued data must choose the newest fresh snapshot, not the first of a batch.
(async()=>{
 run('premat_reset();premat_receive_snapshot(snapshot(20,0));premat_skip_playback();');
 context.fetch_json=async()=>({snapshots:[snapshot(21,0),snapshot(21,1),snapshot(22,4)]});
 await context.refresh_premat_payload();
 assert.equal(run('premat_view.active_snapshot.optimizer_update'),22);
 assert.equal(run('premat_view.active_snapshot.micro_step'),4);
 assert.equal(run('premat_view.skip_to_latest'),false);
 run('premat_clear_timer()');dom.window.close();
 console.log('PASS 6-microstep completion, 180-microstep queue protection, bounded backlog, pause, Skip, latest mode, overrides, reset, 384 CPU outcomes and legacy event recovery');
})().catch(error=>{console.error(error.stack);process.exitCode=1;});
// ^^^ THOG
