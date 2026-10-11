// vvv THOG native Runner edits and live microstep navigation in Chromium and Firefox
'use strict';
const assert=require('node:assert/strict'),{chromium,firefox}=require('playwright');
const address=process.env.INSTRA_TEST_URL;
async function check(type){
 const browser=await type.launch({headless:true,...(type===chromium&&process.env.INSTRA_CHROMIUM_PATH?{executablePath:process.env.INSTRA_CHROMIUM_PATH,args:['--no-sandbox','--disable-dev-shm-usage','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader']} : {}),...(type===firefox?{env:{...process.env,MOZ_DISABLE_CONTENT_SANDBOX:'1'}}:{})});
 try{
  const page=await browser.newPage({viewport:{width:1600,height:1000}}),errors=[];
  page.on('pageerror',error=>errors.push(error.message));page.setDefaultTimeout(20000);
  await page.addInitScript(()=>localStorage.setItem('thog2_run_columns_layout_v1',JSON.stringify({order:['run_name','updated','state'],widths:{}})));
  const microsteps=[0,1,2].map(micro=>({optimizer_update:11,micro_step:micro,pass_sequence:micro+1,pass_complete:true,attention_mode:'fused',n_layer:2,layer_indices:[0,1],events:[{sequence:1,event:'pass_end'}]}));
  await page.route(/\/api\/premat\?/,route=>{
   const query=new URL(route.request().url()).searchParams;
   const after=Number(query.get('after')||0),micro=Number(query.get('after_micro')??-1);
   const snapshots=microsteps.filter(s=>s.optimizer_update>after||s.optimizer_update===after&&s.micro_step>micro);
   route.fulfill({json:{latest:microsteps[2],snapshots,latest_update:11,latest_micro_step:2,snapshot_count:3,history:microsteps,unchanged:!snapshots.length}});
  });
  await page.goto(address+'/runs/fixture_00');
  await page.waitForFunction(()=>window.instra_runner_test_hooks&&app.runs.length===12);
  await page.waitForSelector('.runs-table thead [data-instra-column-key="nsight"]');
  const columns=await page.locator('.runs-table thead [data-instra-column-key]').evaluateAll(nodes=>nodes.map(n=>n.dataset.instraColumnKey));
  assert.equal(columns.indexOf('nsight'),columns.indexOf('updated')-1,'saved column layout migration');
  await page.locator('#runner_nav').click();
  const category=name=>page.locator(`.runner-categories button[data-runner-category="${name}"]`).click();
  const field=key=>page.locator(`input[data-runner-field="${key}"]`);
  await category('CPU Premat');
  const keys=await page.locator('.runner-fields input[data-runner-field]').evaluateAll(nodes=>nodes.map(n=>n.dataset.runnerField));
  assert.equal(keys[0],'--premat','PREMAT enable control must lead CPU Premat');
  await field('--premat').fill('enabled');await field('--premat_materialisation_device').fill('cpu_and_gpu');
  await category('NSIGHT');
  const mode=page.locator('[data-runner-profiling-mode]');await mode.selectOption('pair');
  for(const [key,value] of Object.entries({'--premat_processing_logging':'enabled','--premat_processing_profiler':'nsys','--premat_processing_logging_capture_update':'10','--premat_instra__full_step_timing_capture_and_chart':'enable','--premat_instra__full_step_timing_capture_and_chart_capture':'step 10'}))assert.equal(await field(key).inputValue(),value,key);
  assert.equal(await field('--premat_processing_profiler').getAttribute('readonly'),'');
  assert.ok((await mode.getAttribute('title')).includes('--premat_processing_profiler = nsys'));
  assert.ok((await mode.getAttribute('title')).includes('N/A'));
  await mode.selectOption('ncu');assert.equal(await field('--premat_processing_profiler').inputValue(),'ncu');
  await mode.selectOption('none');assert.equal(await field('--premat_processing_logging').inputValue(),'disabled');
  assert.equal(await field('--premat_processing_profiler').getAttribute('readonly'),null);
  await mode.selectOption('pair');
  await field('--premat_processing_logging_capture_update').fill('2');
  await field('--premat_instra__full_step_timing_capture_and_chart_capture').fill('step 2');
  await category('Premat');assert.equal(await page.locator('input[data-runner-field^="--premat_cpu_"]').count(),0);
  await field('--premat_instra_microsteps').fill('first');
  await category('CPU Premat');assert.equal(await field('--premat_instra_microsteps').inputValue(),'first');
  const label=`Microstep NSIGHT ${type.name()} ${Date.now()}`;await field('Recipe label').fill(label);
  const save=page.waitForResponse(response=>response.url().endsWith('/api/runner/action')&&response.request().postDataJSON()?.action==='save');
  await page.locator('.runner-actions button',{hasText:/^Save$/}).click();assert.ok((await save).ok());
  const saved=(await (await page.request.get(address+'/api/runner')).json()).recipes.find(r=>r.recipe.label===label);
  assert.ok(saved);assert.deepEqual(saved.recipe.profilers,['nsys','ncu']);
  assert.equal(saved.recipe.parameters['--premat_instra_microsteps'],'first');
  assert.equal(saved.recipe.parameters['--premat_processing_logging_capture_update'],2);
  assert.equal(saved.recipe.parameters['--premat_instra__full_step_timing_capture_and_chart_capture'],'step 2');
  await page.locator('#runs_nav').click();
  await page.locator('[data-detail-tab="premat"]').click();
  await page.waitForFunction(()=>premat_view.latest_received_micro_step===2);
  const queued=await page.evaluate(()=>[premat_view.active_snapshot,...premat_view.pending_snapshots].map(s=>s.micro_step));
  assert.deepEqual(queued,[0,1,2]);
  await page.evaluate(()=>{premat_finish_playback();premat_start_snapshot(premat_view.pending_snapshots.pop());premat_finish_playback();});
  assert.equal(await page.locator('#premat_step').textContent(),'Microstep 3 of Optimiser Step 11');
  await page.setViewportSize({width:900,height:1000});
  await page.locator('#premat_history_button').click();
  await page.waitForFunction(()=>premat_view.history_snapshots.length===3);
  const labels=await page.locator('#premat_history_body [data-premat-history-step]').allTextContents();
  assert.ok(labels.some(label=>label.includes('Microstep 3 of Optimiser Step 11')));
  const csv=await page.evaluate(()=>premat_history_csv(premat_view.history_model));
  assert.ok(csv.startsWith('"step","microstep",'));assert.ok(csv.includes('11,3,'));
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({browser:type.name(),nsight_controls:true,automatic_values:true,cpu_tab:true,saved_recipe:true,microstep_queue:true,history:true,column_migration:true,page_errors:errors}));
 }finally{await browser.close();}
}
(async()=>{for(const name of (process.env.INSTRA_TEST_BROWSERS||'chromium,firefox').split(','))await check({chromium,firefox}[name]);})().catch(error=>{console.error(error.stack);process.exitCode=1;});
// ^^^ THOG
