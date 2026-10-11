// vvv THOG exercise native Recipe choices, all Save forms, palette consistency and Premat buttons
'use strict';
const assert=require('node:assert/strict'),{chromium,firefox}=require('playwright');
const address=process.env.INSTRA_TEST_URL;
async function check(browser_type) {
 const browser=await browser_type.launch({headless:true,...(browser_type===chromium&&process.env.INSTRA_CHROMIUM_PATH?{executablePath:process.env.INSTRA_CHROMIUM_PATH,args:['--no-sandbox','--disable-dev-shm-usage']} : {}),...(browser_type===firefox?{env:{...process.env,MOZ_DISABLE_CONTENT_SANDBOX:'1'}}:{})});
 try {
  const page=await browser.newPage({viewport:{width:1600,height:1000}}),errors=[];
  page.setDefaultTimeout(20000);page.on('pageerror',error=>errors.push(error.message));
  await page.goto(address+'/runs/fixture_00');await page.waitForFunction(()=>window.instra_runner_test_hooks);
  await page.locator('#runner_nav').click();
  const option_help=await page.evaluate(()=>instra_runner_test_hooks.field_help({type:'str',help:'Basis version',options_help:'auto; chebyshev_first_kind_roots_v1; dct_ii_orthonormal_v1'}));
  assert.ok(option_help.includes('auto; chebyshev_first_kind_roots_v1; dct_ii_orthonormal_v1'),'open-ended option help must survive enum precedence');
  const category=name=>page.locator(`.runner-categories button[data-runner-category="${name}"]`).click();
  const field=key=>page.locator(`input[data-runner-field="${key}"]`);
  const picker=key=>page.locator(`select[data-runner-choices="${key}"]`);
  await category('CPU Premat');
  assert.equal(await page.locator('.runner-fields input[data-runner-field]').first().getAttribute('data-runner-field'),'--premat');
  await picker('--premat').selectOption('enabled');await picker('--premat_materialisation_device').selectOption('cpu_and_gpu');
  for(const [key,values] of Object.entries({'--premat_cpu_preparation':['eager','scheduled','demand_driven'],
    '--premat_cpu_transfer_timing':['as_the_code_flies','previous_gemm_leading_edge','as_soon_as_ready','demand_driven','predicted_gemm_start'],
    '--premat_cpu_checkpoint_replay':['enabled','disabled']})) {
   const title=await field(key).getAttribute('title'),options=await picker(key).locator('option').evaluateAll(nodes=>nodes.map(node=>node.value));
   for(const value of values) {assert.ok(title.includes(value),key+' help '+value);assert.ok(options.includes(value),key+' picker '+value);}
  }
  await picker('--premat_cpu_layer_batch_size').selectOption('all_layers');assert.equal(await field('--premat_cpu_layer_batch_size').inputValue(),'all_layers');
  await field('--premat_cpu_layer_batch_size').fill('17');assert.equal(await field('--premat_cpu_layer_batch_size').getAttribute('aria-invalid'),'false');
  await field('--premat_cpu_layer_batch_size').fill('1');assert.equal(await field('--premat_cpu_layer_batch_size').getAttribute('aria-invalid'),'true');
  await picker('--premat_cpu_layer_batch_size').selectOption('2');
  await field('--premat_cpu_preparation').fill('eager, ');await picker('--premat_cpu_preparation').selectOption('scheduled');
  assert.equal(await field('--premat_cpu_preparation').inputValue(),'eager, scheduled');
  await picker('--premat_cpu_transfer_timing').selectOption('predicted_gemm_start');assert.equal(await field('--premat_cpu_transfer_lead_ms').isEnabled(),true);
  await picker('--premat_enable_gpu_timing_diagnostic').selectOption('false');
  await picker('--premat_instra_playback').selectOption('latest');
  await picker('--premat_materialisation_device').selectOption('gpu');assert.equal(await picker('--premat_cpu_preparation').isDisabled(),true);
  await picker('--premat_materialisation_device').selectOption('cpu_and_gpu');
  const label=`Recipe controls ${browser_type.name()} ${Date.now()}`;await field('Recipe label').fill(label);
  const save=async()=>{
   const response=page.waitForResponse(response=>response.url().endsWith('/api/runner/action')&&['save','update_recipe'].includes(response.request().postDataJSON()?.action));
   await page.locator('.runner-actions button',{hasText:/^Save$/}).click();const result=await response;assert.ok(result.ok(),await result.text());
   await page.waitForFunction(()=>document.querySelector('.runner-categories button.active')?.dataset.runnerCategory==='Frequently Used');
   return result.request().postDataJSON().action;
  };
  assert.equal(await save(),'save');
  const state=await (await page.request.get(address+'/api/runner')).json();const saved=state.recipes.find(row=>row.recipe.label===label);
  assert.ok(saved);assert.equal(saved.recipe.parameters['--premat_instra_playback'],'latest');
  assert.deepEqual(saved.recipe.parameters['--premat_cpu_preparation'],['eager','scheduled']);
  await category('CPU Premat');assert.equal(await save(),'save','unchanged save');
  await page.locator('.runner-recipe-row',{has:page.getByRole('button',{name:label,exact:true})}).getByRole('button',{name:'Edit',exact:true}).click();
  await category('CPU Premat');await picker('--premat_instra_playback').selectOption('complete_step');
  assert.equal(await save(),'update_recipe','in-place save');
  await page.locator('.runner-recipe-row',{has:page.getByRole('button',{name:label,exact:true})}).getByRole('button',{name:label,exact:true}).click();
  await category('CPU Premat');await picker('--premat_instra_playback').selectOption('latest');assert.equal(await save(),'save','copy save');

  const states=['ready','queued','dispatching','running','blocked','stopping','flushing','unknown','completed','failed','cancelled'];
  const grid={grid_id:'colour_grid',grid_tag:'SCR-99999',label:'State colour checks',recipe_id:saved.recipe_id,state:'running',created_at:new Date().toISOString(),
    recipe:saved.recipe,runs:states.map((state,index)=>({run_id:'state_'+index,grid_id:'colour_grid',state,parameters:{'--geometry-preset':'depth','--max-iters':50},
      host_label:'scruffy',gpu:{ordinal:0,model:'fixture',uuid:'fixture'},profiler:'none',attempts:[]}))};
  await page.route(/\/api\/runner$/,async route=>{const response=await route.fetch();const data=await response.json();data.grids=[grid];await route.fulfill({response,json:data});});
  await page.locator('#runner_tabs button[data-runner-tab="progress"]').click();
  await page.waitForFunction(()=>document.querySelectorAll('.runner-run-identity .runner-status').length===11);
  const colors=await page.evaluate(states=>states.map(state=>{
    const legend=document.querySelector(`.runner-state-legend .runner-status-${state}`);
    const cell=document.querySelector(`.runner-run-identity .runner-status-${state}`);
    return {state,key:getComputedStyle(legend).color,cell:getComputedStyle(cell).color};
  }),states);
  for(const row of colors)assert.equal(row.cell,row.key,row.state);
  for(const row of colors.slice(0,4))assert.equal(row.cell,'rgb(22, 131, 59)');
  for(const row of colors.slice(6))assert.equal(row.cell,'rgb(179, 38, 46)');

  await page.locator('#runs_nav').click();await page.locator('[data-detail-tab="premat"]').click();
  await page.evaluate(()=>{
    premat_reset();premat_view.run_id=app.current_run_id;premat_view.playing=true;
    const sample=(step,micro)=>({optimizer_update:step,micro_step:micro,pass_complete:true,playback_mode:'complete_step',attention_mode:'fused',layer_indices:[0],
      events:[{sequence:1,event:'materialising',family:'O',layer_index:0,owner:'main',state:'MATERIALISING'},{sequence:2,event:'pass_end'}]});
    premat_receive_snapshot(sample(100,0));premat_receive_snapshot(sample(100,1));premat_receive_snapshot(sample(101,4));premat_toggle_playback();
  });
  await page.locator('#premat_skip').click();
  assert.equal(await page.locator('#premat_step').textContent(),'Microstep 5 of Optimiser Step 101');
  assert.equal(await page.locator('#premat_playback_mode').getAttribute('aria-pressed'),'true');
  await page.locator('#premat_playback_mode').click();assert.equal(await page.locator('#premat_playback_mode').textContent(),'Latest data');
  await page.evaluate(()=>premat_receive_snapshot({optimizer_update:102,micro_step:2,pass_complete:true,attention_mode:'fused',layer_indices:[0],events:[{sequence:1,event:'pass_end'}]}));
  assert.equal(await page.locator('#premat_step').textContent(),'Microstep 3 of Optimiser Step 102');
  await page.setViewportSize({width:900,height:1000});
  const overflow=await page.locator('.premat-header-controls').evaluate(element=>element.scrollWidth>element.clientWidth+1);
  assert.equal(overflow,false,'playback controls must wrap at narrower widths');
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({browser:browser_type.name(),native_popups:true,valid_option_help:true,sweeps:true,save_variants:4,state_colours:colors,skip:true,playback_mode:true,page_errors:errors}));
 } finally {await browser.close();}
}
(async()=>{for(const name of (process.env.INSTRA_TEST_BROWSERS||'chromium,firefox').split(','))await check({chromium,firefox}[name]);})().catch(error=>{console.error(error.stack);process.exitCode=1;});
// ^^^ THOG
