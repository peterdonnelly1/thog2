// vvv THOG real-browser width acceptance; launch through test_residual_width_browser.py
const assert = require('node:assert/strict');
const fs = require('node:fs');
const {chromium} = require('playwright');

(async () => {
  const options = {headless:true};
  if (process.env.THOG_WIDTH_BROWSER_EXECUTABLE) options.executablePath = process.env.THOG_WIDTH_BROWSER_EXECUTABLE;
  if (process.env.THOG_WIDTH_CHROMIUM_PACKAGE) {
    const {default:pack} = await import(process.env.THOG_WIDTH_CHROMIUM_PACKAGE);
    options.args = pack.args;
  }
  const browser = await chromium.launch(options);
  const page = await browser.newPage({viewport:{width:1600,height:1100}});
  page.setDefaultTimeout(10000);
  const errors = [], requests = [];
  page.on('pageerror', error => errors.push(String(error)));
  page.on('console', message => {if(message.type()==='warning' && /Width|Deferred chart/.test(message.text()))errors.push(message.text());});
  page.on('request', request => {if(request.url().includes('/api/width-figure'))requests.push(request.url());});
  try {
    await page.goto(process.argv[2] + '/runs/fixture_00');
    await page.waitForFunction(() => app.current_status?.width_snapshot_count===3 && window.instra_demand_runtime && window.__instra_workspace);
    await page.locator('#runs_nav').click();
    await page.locator('[data-detail-tab="charts"]').click();
    const show = async name => {
      await page.evaluate(() => window.instra_width.refresh());
      await page.waitForTimeout(150);
      await page.evaluate(() => {if(app.maximized_chart)restore_maximized_chart();});
      await page.locator(`#${name}_plot`).scrollIntoViewIfNeeded();
      await page.evaluate(() => window.instra_width.refresh());
      await page.waitForFunction(name => by_id(`${name}_plot`).dataset.plotReady==='true' && by_id(`${name}_plot`).data?.length>0, name);
    };
    for (const name of ['width_energy','width_probes','width_curves']) await show(name);
    assert.equal(await page.locator('[data-width-step="width_energy"] option').count(), 3);
    await page.locator('[data-width-step="width_energy"]').selectOption('1');
    await page.waitForFunction(() => by_id('width_energy_plot').data?.[0]?.meta?.instra_workspace_optimizer_update===1);
    await page.locator('[data-width-site="width_energy"]').selectOption('block_1.residual_output');
    await page.waitForFunction(() => by_id('width_energy_plot').data?.[0]?.meta?.site==='block_1.residual_output');
    const curves = await page.evaluate(() => by_id('width_curves_plot').data);
    assert.ok(curves.length<=8 && curves.every(trace => trace.x.length<=8 && trace.x.length===trace.y.length));
    assert.match(await page.evaluate(() => by_id('width_energy_plot').layout.xaxis.title.text), /basis mode/);
    assert.match(await page.evaluate(() => by_id('width_probes_plot').layout.yaxis.title.text), /cross-entropy/);
    await page.locator('[data-maximize="width_curves"]').click();
    await page.waitForFunction(() => app.maximized_chart==='width_curves');
    await page.locator('[data-maximize="width_curves"]').click();
    await page.waitForFunction(() => app.maximized_chart===null);

    // Existing ordinary visibility controls choose the Multiview cohort.
    await page.evaluate(() => {
      for (const run of app.runs) app.visibility[run_identifier(run)] = ['fixture_00','fixture_01'].includes(run_identifier(run));
      save_json('thog2_local_run_visibility', app.visibility);render_runs();
    });
    await page.locator('#workspace_nav').click();
    await page.waitForFunction(() => app.workspace_mode===true && app.instra_loss_autofocused===true);
    // Multiview normally opens by maximizing Loss; restore its ordinary grid to inspect width views.
    await page.waitForTimeout(200);
    await page.evaluate(() => restore_maximized_chart());
    await show('width_energy');
    await page.waitForFunction(() => new Set(by_id('width_energy_plot').data?.map(trace=>trace.meta?.instra_workspace_run_id)).size===2);
    const paired = await page.evaluate(() => by_id('width_energy_plot').data.map(trace=>({run:trace.meta.instra_workspace_run_id,step:trace.meta.instra_workspace_optimizer_update,D:trace.meta.reference_width,r:trace.meta.residual_width})));
    assert.deepEqual(new Set(paired.map(row=>row.run)), new Set(['fixture_00','fixture_01']));
    assert.ok(paired.every(row=>row.step===1 && row.D===8 && row.r===3));
    await page.locator('#runs_nav').click();
    await page.evaluate(() => {select_run('fixture_01',{manual:true});select_run('fixture_00',{manual:true});});
    await page.waitForFunction(() => app.current_status?.local_run_id==='fixture_00');
    await show('width_energy');
    await page.waitForFunction(() => by_id('width_energy_plot').data?.every(trace=>trace.meta.instra_workspace_run_id==='fixture_00'));
    await page.evaluate(() => select_run('fixture_02',{manual:true}));
    await page.waitForFunction(() => app.current_status?.local_run_id==='fixture_02' && by_id('width_chart_group').hidden);
    assert.equal(await page.locator('#width_chart_group').isVisible(), false);
    const response = await page.request.get(process.argv[2]+'/api/width-figure?run=fixture_00&chart=width_probes&step=1');
    assert.equal(response.status(),200);
    const payload = await response.json();
    assert.equal(payload.available_steps.length,3);
    assert.match(payload.probe_interpretation,/not retrained/);
    // Exercise real HTTP Save/Preview and public parser resolution without allocating fixture GPUs.
    await page.locator('#runner_nav').click();
    await page.waitForSelector('#runner_parameter_search');
    const set_runner_field = async (key,value) => {
      await page.locator('#runner_parameter_search').fill(key);
      await page.locator(`input[data-runner-field="${key}"]`).fill(value);
    };
    await set_runner_field('--geometry-preset','width');
    await page.locator('#runner_parameter_search').fill('--select-width');
    assert.equal(await page.locator('input[data-runner-field="--select-width"]').inputValue(),'true');
    await set_runner_field('WIDTH.order','16');
    await set_runner_field('WIDTH.compressor','dct');
    await set_runner_field('DEPTH.compressor','chebyshev');
    await set_runner_field('DEPTH.compressor_version','auto');
    await set_runner_field('DEPTH.order','');
    await set_runner_field('--instrumentation__width_activation_curves__end_step','-1');
    assert.equal(await page.locator('#runner_validation').textContent(),'');
    const preview_recipes = [];
    const preview_results = [];
    await page.locator('#runner_parameter_search').fill('');
    await page.locator('.runner-categories').getByRole('button',{name:'Frequently Used',exact:true}).click();
    assert.equal(await page.locator('input[data-runner-field="WIDTH.order"]').isVisible(),true);
    await page.locator('.runner-categories').getByRole('button',{name:'Geometry',exact:true}).click();
    assert.equal(await page.locator('input[data-runner-field="WIDTH.order"]').isVisible(),true);
    assert.equal(await page.locator('input[data-runner-field="--select-width"]').inputValue(),'true');
    const save_response = page.waitForResponse(response=>response.url().endsWith('/api/runner/action') && response.request().postDataJSON().action==='save');
    await page.locator('#runner_detail').getByRole('button',{name:'Save',exact:true}).click();
    assert.equal((await save_response).status(),200);
    const preview = async count => {
      await page.locator('#runner_parameter_search').fill('');
      await page.locator('.runner-categories').getByRole('button',{name:'Frequently Used',exact:true}).click();
      const pending = page.waitForResponse(response=>response.url().endsWith('/api/runner/action') && response.request().postDataJSON().action==='preview');
      await page.locator('#runner_detail').getByRole('button',{name:'Preview',exact:true}).click();
      const response = await pending;
      assert.equal(response.status(),200,await response.text());
      preview_recipes.push(response.request().postDataJSON().recipe);
      preview_results.push(await response.json());
      await page.locator('#runner_preview').getByRole('heading',
        {name:`Total Runs in Grid: ${count} · physical executions: ${count}`,exact:true}).waitFor();
    };
    await preview(1);
    assert.deepEqual(preview_recipes[0].parameters['WIDTH.order'],[16]);
    assert.equal(preview_recipes[0].parameters['--select-width'],true);
    assert.equal(preview_results[0].runs[0].parameters['--geometry-preset'],'width-type-I');
    assert.equal(Object.hasOwn(preview_recipes[0].parameters,'DEPTH.order'),false);
    assert.equal(preview_recipes[0].parameters['--instrumentation__width_activation_curves__end_step'],-1);
    await set_runner_field('--geometry-preset','width-type-I');
    await set_runner_field('WIDTH.order','16, 32, 64');
    await preview(3);
    assert.deepEqual(preview_recipes[1].parameters['WIDTH.order'],[16,32,64]);
    assert.equal(Object.hasOwn(preview_recipes[1].parameters,'DEPTH.order'),false);
    await set_runner_field('DEPTH.order','1, 2');
    await preview(6);
    await set_runner_field('DEPTH.order','');
    await set_runner_field('--basis-family','dct');
    const invalid_response = page.waitForResponse(response=>response.url().endsWith('/api/runner/action') && response.request().postDataJSON().action==='preview');
    await page.locator('#runner_parameter_search').fill('');
    await page.locator('.runner-categories').getByRole('button',{name:'Frequently Used',exact:true}).click();
    await page.locator('#runner_detail').getByRole('button',{name:'Preview',exact:true}).click();
    const rejected = await invalid_response;
    assert.equal(rejected.status(),400);
    assert.match((await rejected.json()).error,/explicit.*basis/);
    await set_runner_field('--basis-family','');
    await preview(3);
    assert.deepEqual(errors,[]);
    const record={browser:await browser.version(),checks:['three Plotly charts','step selection','residual-site selection','bounded curves','correct axis labels','maximize/restore','paired Multiview identity and step','rapid run switching','legacy run hides width charts','API pairing metadata','Runner real Save','Runner real width-only single Preview with optimizer','Runner real width-only grid Preview with optimizer','Runner real joint grid Preview','Runner width aliases and automatic selection','WIDTH.order in Frequently Used and Geometry','Runner HTTP validation error and recovery','Runner unbounded capture window edit'],width_requests:requests.length,console_errors:errors,pairing:paired,runner_preview_parameters:preview_recipes.map(recipe=>recipe.parameters),runner_preview_counts:preview_results.map(result=>result.total_runs)};
    if (process.argv[3]) fs.writeFileSync(process.argv[3],JSON.stringify(record,null,2)+'\n');
    console.log(JSON.stringify(record));
  } finally {
    await browser.close();
  }
})().catch(error=>{console.error(error);process.exitCode=1;});
// ^^^ THOG
