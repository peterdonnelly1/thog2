// vvv THOG exercise the complete dashboard in Chromium and Firefox using an isolated fixture
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const {chromium,firefox} = require("playwright");
const base_url = process.env.INSTRA_TEST_URL || "http://127.0.0.1:8765";
const asset = name => fs.readFileSync(`sheet/local_dashboard_assets/${name}`,"utf8");
async function check_browser(browser_type) {
  const browser = await browser_type.launch({headless:true});
  const page = await browser.newPage({viewport:{width:1600,height:1100}});
  const errors=[];page.on("pageerror",error=>{errors.push(error.message);if(errors.length<3)console.log("ERROR STACK",error.stack);});
  let failed_responses=0,chart_requests=0;
  page.on("request",request=>{if(/\/api\/(?:chart-group|processing)/.test(request.url()))chart_requests++;});
  page.on("response",response=>{if(response.status()>=400 && failed_responses++<5)console.log("HTTP",response.status(),response.url());});
  page.on("dialog",dialog=>dialog.dismiss());
  await page.addInitScript(()=>{
    let plotly;
    Object.defineProperty(window,"Plotly",{configurable:true,get(){return plotly;},set(value){
      plotly=value;
      for (const [owner,names] of [[value,["react","newPlot","relayout","purge"]],[value.Plots,["resize"]]]) {
        for(const name of names) {
          const original=owner[name];
          owner[name]=function(...args){
            const context=`Plotly.${name}: ${args[0]?.id || args[0]} connected=${args[0]?.isConnected}`;
            const convert=error=>{throw new Error(`${context}: ${error}`);};
            try{const result=original.apply(this,args);return result?.catch ? result.catch(convert) : result;}catch(error){return convert(error);}
          };
        }
      }
    }});
  });
  await page.goto(`${base_url}/runs/fixture_00`);
  await page.waitForFunction(()=>typeof app!=="undefined" && app.runs.length>=12);
  await page.waitForTimeout(1800);
  console.log(browser_type.name(),"loaded",await page.evaluate(()=>({runs:app.runs.length,nav:[...document.querySelectorAll('.icon-rail .selected')].map(node=>node.id)})));
  await page.locator("#runner_nav").click();
  await page.waitForSelector("#runner_detail input[data-runner-field]");
  await page.locator(".runner-recipe-row > button:first-child",{hasText:"Fixture Recipe"}).click();
  assert.equal(await page.locator(".runner-recipe-label button").count(),0);
  assert.equal(await page.locator(".runner-gpu-default").count(),0);
  assert.equal(await page.evaluate(()=>document.getElementById("runner_parameter_search").parentElement.nextElementSibling.className),"runner-categories");
  const optimizer=page.locator('input[data-runner-field="--optimizer"]');
  await optimizer.fill("imaginary_optimizer");
  assert.equal(await optimizer.getAttribute("aria-invalid"),"true");
  assert.equal(await optimizer.evaluate(node=>getComputedStyle(node).borderTopColor),"rgb(255, 0, 0)");
  assert.match(await optimizer.getAttribute("title"),/Available options:.*adamw/);
  await optimizer.fill("adamw");
  assert.equal(await optimizer.getAttribute("aria-invalid"),"false");
  const position=await optimizer.evaluate(node=>({field:node.getBoundingClientRect().left,button:node.parentElement.querySelector("button").getBoundingClientRect().right}));
  assert.ok(position.button<=position.field,JSON.stringify(position));
  await page.locator(".runner-categories button",{hasText:/^Premat$/}).click();
  assert.equal(await page.locator(".runner-premat-enabled").count(),1);
  await page.locator('[data-runner-tab="history"]').click();
  for (const [label,outcome] of [["Grid 1","complete"],["Grid 2","partial"],["Grid 3","none"]]) {
    const entry=page.locator(".runner-history-grid-row > button:first-child",{hasText:label});
    assert.ok((await entry.getAttribute("class")).includes(`runner-outcome-${outcome}`));
    assert.equal(await entry.evaluate(node=>getComputedStyle(node).fontWeight),"400");
  }
  await page.locator('[data-runner-tab="progress"]').click();
  await page.waitForSelector(".runner-run-headings");
  assert.deepEqual(await page.locator(".runner-run-headings strong").allTextContents(),["Run ID","Host","GPU","State","Step","Loss","Best loss","Profiling"]);
  for (const id of ["workspace_nav","runs_nav","runner_nav","workspace_nav","runs_nav"]) {
    await page.locator(`#${id}`).click();
    await page.waitForTimeout(100);
    const selected=await page.evaluate(()=>["runner_nav","workspace_nav","runs_nav","networks_nav","settings_nav"].filter(id=>document.getElementById(id).classList.contains("selected")));
    assert.deepEqual(selected,[id]);
  }
  await page.waitForFunction(()=>document.getElementById("training_throughput_plot")?.data?.length===1,{},{timeout:15000});
  await page.waitForFunction(()=>document.querySelector('.chart-card.maximized[data-metric-chart-id="train/loss"]'),{},{timeout:15000});
  assert.equal(await page.evaluate(()=>document.querySelector('.local-metric-grid').firstElementChild.dataset.metricChartId),"train/loss");
  // Deliberately delay Multiview responses and leave while they are pending.
  await page.route("**/api/chart-group?**",async route=>{await new Promise(resolve=>setTimeout(resolve,200));await route.continue();});
  for(let cycle=0;cycle<15;cycle++) {
    await page.locator("#workspace_nav").click();
    await page.waitForTimeout(cycle%3===0?250:30);
    await page.locator("#runs_nav").click();
    await page.waitForTimeout(60);
  }
  await page.unroute("**/api/chart-group?**");
  await page.waitForFunction(()=>document.getElementById("training_throughput_plot")?.data?.length===1,{},{timeout:15000});
  assert.equal(await page.evaluate(()=>app.workspace_mode),false);
  assert.equal(await page.evaluate(()=>document.getElementById("training_throughput_plot").data[0].meta.instra_workspace_run_id),"fixture_00");
  // Multi-Grid overlays keep every eye-selected member.
  await page.locator("#workspace_nav").click();
  await page.waitForFunction(()=>document.getElementById("training_throughput_plot")?.data?.length===12,{},{timeout:15000});
  await page.waitForFunction(()=>document.querySelector('[data-metric-chart-id="train/loss"] .plot-mount')?.data?.length===12,{},{timeout:15000});
  await page.locator("#runner_nav").click();
  await page.waitForTimeout(750);
  const hidden_requests=chart_requests;
  await page.waitForTimeout(2500);
  assert.equal(chart_requests,hidden_requests,"hidden charts continued polling behind Runner");
  await page.locator("#workspace_nav").click();
  await page.waitForFunction(()=>document.querySelector('[data-metric-chart-id="train/loss"] .plot-mount')?.data?.length===12,{},{timeout:15000});
  // Measure responsive event-loop progress over a sustained live dashboard session.
  await page.evaluate(()=>{
    window.instra_test_ticks=0;window.instra_test_mutations=0;
    setInterval(()=>window.instra_test_ticks++,100);
    new MutationObserver(records=>window.instra_test_mutations+=records.length).observe(document.body,{subtree:true,childList:true,attributes:true});
  });
  const soak_seconds=Number(process.env.INSTRA_SOAK_SECONDS || 20);
  for(let elapsed=0;elapsed<soak_seconds;elapsed+=5) {
    await page.waitForTimeout(5000);
    console.log(browser_type.name(),"soak",elapsed+5,"s",await page.evaluate(()=>({ticks:window.instra_test_ticks,mutations:window.instra_test_mutations})));
  }
  assert.ok(await page.evaluate(()=>window.instra_test_ticks)>=soak_seconds*7,"dashboard event loop stalled");
  assert.deepEqual(errors,[],"uncaught browser errors");
  await browser.close();
  console.log("PASS",browser_type.name(),"Recipe validation, placement, History, navigation, loss/throughput, delayed responses and soak");
}
async function check_deletion() {
  const browser=await chromium.launch({headless:true});
  const page=await browser.newPage();
  await page.goto(`${base_url}/runs/fixture_04`);
  await page.waitForFunction(()=>typeof app!=="undefined" && app.runs.length===12);
  let confirmations=0,requests=0;
  page.on("dialog",async dialog=>{confirmations++;await dialog.accept();});
  page.on("request",request=>{if(request.method()==="DELETE")requests++;});
  // Make a catalogue Grid active and verify the menu warns before confirmation or deletion.
  await page.route("**/api/runner",async route=>{
    const response=await route.fetch(),state=await response.json();
    state.grids.find(grid=>grid.grid_tag==="G-00002").state="running";
    await route.fulfill({response,json:state});
  });
  await page.evaluate(()=>open_run_menu("fixture_04",document.getElementById("run_title")));
  await page.locator("#delete_grid_runs").click();
  await page.waitForFunction(()=>document.body.textContent.includes("Stop the Grid first"));
  assert.equal(confirmations,0);assert.equal(requests,0);
  await page.unroute("**/api/runner");
  // A terminal Grid removes four runs in one request and clears its selected rows.
  await page.evaluate(()=>{app.selected=new Set(["fixture_04","fixture_05"]);render_runs();open_run_menu("fixture_04",document.getElementById("run_title"));});
  const started=Date.now();
  await page.locator("#delete_grid_runs").click();
  await page.waitForFunction(()=>app.runs.length===8 && !app.selected.has("fixture_04") && !app.selected.has("fixture_05"));
  assert.ok(Date.now()-started<10000,"four-run Grid deletion exceeded ten seconds");
  assert.equal(confirmations,1);assert.equal(requests,1);
  await page.evaluate(()=>{app.selected=new Set(["fixture_00","fixture_01","fixture_02"]);render_runs();});
  await page.locator("#delete_selected_runs").click();
  await page.waitForFunction(()=>app.runs.length===5 && app.selected.size===0);
  assert.equal(confirmations,2);assert.equal(requests,2,"bulk deletion sent a request for every run");
  await browser.close();
  console.log("PASS Grid deletion confirmation/active warning, actual batch deletion and checkbox cleanup");
}
(async()=>{
  for(const browser_type of [chromium,firefox]) await check_browser(browser_type);
  await check_deletion();
})().catch(error=>{console.error(error);process.exit(1);});
// ^^^ THOG
