// vvv THOG exercise the feedback in both browsers, including slow data and stale plastic panels
"use strict";
const assert=require("node:assert/strict"),{chromium,firefox}=require("playwright");
const address=process.env.INSTRA_TEST_URL || "http://127.0.0.1:8766";
async function check(type) {
  const browser=await type.launch({headless:true}),context=await browser.newContext({viewport:{width:1600,height:1100}}),page=await context.newPage();
  const errors=[];page.on("pageerror",error=>errors.push(error.message));
  await page.goto(`${address}/runs/fixture_00`);
  await page.waitForFunction(()=>typeof app!=="undefined" && app.runs.length===12);
  await page.locator("#runner_nav").click();
  await page.locator(".runner-recipe-row > button:first-child",{hasText:"Fixture Recipe"}).click();
  const category=name=>page.locator(".runner-categories button",{hasText:new RegExp(`^${name}$`)});
  const field=key=>page.locator(`input[data-runner-field="${key}"]`);
  for(const key of ["--log-interval","--eval-iters","--eval-interval"])assert.equal(await field(key).count(),1);
  const position=await field("--optimizer").evaluate(node=>({field:node.getBoundingClientRect().right,button:node.parentElement.querySelector("button").getBoundingClientRect().left}));
  assert.ok(position.button>=position.field && position.button-position.field<15,JSON.stringify(position));
  const pale=await category("Premat").evaluate(node=>getComputedStyle(node).backgroundColor);
  assert.equal(pale,"rgb(228, 242, 255)");
  await category("Premat").click();
  assert.notEqual(await page.locator(".runner-fields").evaluate(node=>getComputedStyle(node).backgroundColor),pale);
  for(const [name,keys,value] of [["Coarse",["--plastic__coarse_phase"],"enabled"],
    ["Variable Depth",["--plastic__do_learn_layer_count","--no-plastic__do_learn_layer_count"],"true"],
    ["Chaos Bumps",["--chaos_bump__sampling__enabled","--no-chaos_bump__sampling__enabled"],"true"]]) {
    await category(name).click();
    assert.deepEqual(await page.locator(".runner-fields input").evaluateAll((nodes,count)=>nodes.slice(0,count).map(node=>node.dataset.runnerField),keys.length),keys);
    await field(keys[0]).fill(value);
    assert.equal(await category(name).evaluate(node=>getComputedStyle(node).backgroundColor),pale);
    if(keys.length===2) {
      await field(keys[1]).fill("true");
      assert.equal(await category(name).evaluate(node=>node.classList.contains("runner-category-enabled")),false);
      await field(keys[1]).fill("");
      assert.equal(await category(name).evaluate(node=>node.classList.contains("runner-category-enabled")),true);
    }
  }
  await category("NSIGHT").click();
  await page.locator(".runner-fields select").selectOption("nsys");
  assert.equal(await category("NSIGHT").evaluate(node=>getComputedStyle(node).backgroundColor),pale);
  await page.locator(".runner-fields select").selectOption("none");
  assert.equal(await category("NSIGHT").evaluate(node=>node.classList.contains("runner-category-enabled")),false);
  const search=page.locator("#runner_parameter_search");
  await search.fill("log_interval");
  assert.equal(await category("Frequently Used").count(),1);
  assert.equal(await category("Run Control Parameters").count(),1);
  assert.equal(await category("Premat").count(),0);
  assert.match(await field("--log-interval").locator("..").locator(".runner-field-categories").textContent(),/Frequently Used.*Run Control Parameters/);
  await category("Run Control Parameters").click();
  assert.equal(await field("--plastic__log_interval_coarse").count(),0);
  await search.fill("");
  for(const key of ["--log-interval","--eval-iters","--eval-interval"])assert.equal(await field(key).count(),1);
  for(const name of await page.locator(".runner-categories button").allTextContents()) {
    await category(name).click();
    assert.equal(await page.locator(".runner-parameter-panel").evaluate(panel=>{
      const bottom=panel.getBoundingClientRect().bottom;
      return panel.contains(document.getElementById("runner_parameter_search")) &&
        [...panel.querySelectorAll(".runner-fields input,.runner-fields select")].every(field=>field.getBoundingClientRect().bottom<=bottom);
    }),true,`${name} escaped its parameter panel`);
  }
  await category("Frequently Used").click();
  await page.screenshot({path:`/tmp/instra-feedback-${type.name()}-runner.png`});
  // A new selection starts with the current Grid rather than every historical run.
  await page.locator("#runs_nav").click();await page.locator("#workspace_nav").click();
  await page.waitForFunction(()=>window.__instra_workspace.visible_runs().length===4);
  const chosen=await page.evaluate(()=>({...app.visibility}));
  await page.reload();await page.waitForFunction(()=>typeof app!=="undefined"&&app.runs.length===12);
  await page.locator("#workspace_nav").click();
  assert.deepEqual(await page.evaluate(()=>({...app.visibility})),chosen,"reload changed saved eyes");
  // Loss must render while a slow run still blocks metadata and its own loss response.
  await page.route(/\/api\/chart-group(?:s)?\?.*run=fixture_11/,async route=>{
    await new Promise(resolve=>setTimeout(resolve,5000));try{await route.continue();}catch(_){/* cancelled obsolete request */}
  });
  await page.locator("#runs_nav").click();
  await page.evaluate(()=>{for(const run of app.runs)app.visibility[run_identifier(run)]=true;save_json("thog2_local_run_visibility",app.visibility);render_runs();});
  await page.locator("#workspace_nav").click();
  await page.waitForFunction(()=>document.querySelector('[data-metric-chart-id="train/loss"] .plot-mount')?.data?.length>0,{},{timeout:3500});
  assert.equal(await page.evaluate(()=>document.querySelector('.local-metric-group[data-metric-group="train"] .local-metric-grid').firstElementChild.dataset.metricChartId),"train/loss");
  await page.evaluate(()=>{for(const run of app.runs)app.visibility[run_identifier(run)]=run.runner_grid_tag==="G-00002";save_json("thog2_local_run_visibility",app.visibility);render_runs();});
  await page.waitForFunction(()=>document.querySelector('[data-metric-chart-id="train/loss"] .plot-mount')?.data?.length===4,{},{timeout:3500});
  await page.waitForTimeout(5500);
  assert.equal(await page.evaluate(()=>document.querySelector('[data-metric-chart-id="train/loss"] .plot-mount').data.length),4,"obsolete response restored deselected runs");
  await page.unroute(/\/api\/chart-group(?:s)?\?.*run=fixture_11/);
  // Show genuine plastic data for one selected run, then switch to a non-plastic Grid.
  await page.route(/\/api\/chart-groups\?/,async route=>{
    const id=new URL(route.request().url()).searchParams.get("run");
    await route.fulfill({json:{available:true,groups:[{name:"train",revision:2,chart_count:1},...(id==="fixture_00"?[{name:"plastic",revision:2,chart_count:1}]:[])]}});
  });
  await page.route(/\/api\/chart-group\?/,async route=>{
    if(new URL(route.request().url()).searchParams.get("group")!=="plastic")return route.continue();
    await route.fulfill({json:{available:true,group:{name:"plastic",revision:2,charts:[{id:"plastic/ncols",title:"Ncols",series:[{name:"Ncols",x:[1,2],y:[8,9]}]}]}}});
  });
  await page.evaluate(()=>{if(app.maximized_chart)restore_maximized_chart();for(const run of app.runs)app.visibility[run_identifier(run)]=run_identifier(run)==="fixture_00";render_runs();});
  await page.waitForSelector('.local-metric-group[data-metric-group="plastic"]');
  await page.locator('.local-metric-group[data-metric-group="plastic"] .chart-group-toggle').click();
  await page.waitForSelector('[data-metric-chart-id="plastic/ncols"]');
  await page.evaluate(()=>toggle_maximized_chart(document.querySelector('[data-metric-chart-id="plastic/ncols"]').dataset.chart));
  await page.evaluate(()=>{for(const run of app.runs)app.visibility[run_identifier(run)]=run.runner_grid_tag==="G-00002";render_runs();});
  await page.waitForFunction(()=>!document.querySelector('.local-metric-group[data-metric-group="plastic"]'));
  assert.equal(await page.locator('[data-metric-chart-id="plastic/ncols"]').count(),0);
  assert.equal(await page.evaluate(()=>Boolean(app.maximized_chart && !document.querySelector('.chart-card.maximized'))),false,"removed plastic card left an empty maximized view");
  const main_eyes=await page.evaluate(()=>localStorage.getItem("thog2_local_run_visibility"));
  const frame=await page.context().newPage();
  await frame.goto(`${address}/?runner_grid_tag=G-00003`);
  await frame.waitForFunction(()=>typeof app!=="undefined"&&app.runs.length===4);
  await frame.evaluate(()=>document.getElementById("workspace_nav").click());
  await frame.waitForFunction(()=>window.__instra_workspace.visible_runs().length===4);
  assert.equal(await frame.evaluate(()=>localStorage.getItem("thog2_local_run_visibility")),main_eyes,"embedded Grid changed the main eye selection");
  await frame.close();
  assert.deepEqual(errors,[],"uncaught browser errors");
  await browser.close();console.log("PASS",type.name(),"all category feedback, saved eyes, slow loss, cancellation and stale plastic cleanup");
}
(async()=>{for(const type of [chromium,firefox])await check(type);})().catch(error=>{console.error(error);process.exit(1);});
// ^^^ THOG
