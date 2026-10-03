// vvv THOG verify same-number host isolation, dense outlines, Ready and rendered run-name hovers
"use strict";
const assert=require("node:assert/strict");
const {chromium}=require("playwright");
const address=process.env.INSTRA_TEST_URL||"http://127.0.0.1:8765";
(async()=>{
  const browser=await chromium.launch({headless:true,executablePath:process.env.INSTRA_CHROMIUM_PATH,
    args:["--no-sandbox","--disable-dev-shm-usage"]});
  try {
    const page=await browser.newPage({viewport:{width:1600,height:1100}});
    const errors=[];page.on("pageerror",error=>errors.push(error.message));
    const runs=await (await page.request.get(address+"/api/runs")).json();
    for(let index=0;index<8;index++){
      const run=runs.runs.find(run=>run.dashboard_run_id===`fixture_${String(index).padStart(2,"0")}`);
      run.runner_grid_tag="G-00021";
      run.thog_host_id=index<4?"thog_host.scruffy":"thog_host.dreedle";
      run.host_label=index<4?"scruffy":"dreedle";
      run.remote_copy=index>=4;
      delete run.runner_grid_id;delete run.runner_grid_owner_host_id;
    }
    const snapshot=await (await page.request.get(address+"/api/runner")).json();
    snapshot.recipes[0].state="ready";
    snapshot.grids[0].grid_tag="G-00021";
    await page.route(address+"/api/runs",route=>route.fulfill({json:runs}));
    await page.route(address+"/api/runner",route=>route.fulfill({json:snapshot}));
    await page.goto(address+"/runs/fixture_00");
    await page.waitForFunction(()=>typeof app!=="undefined"&&app.runs.length===12);
    await page.locator("#runs_nav").click();
    await page.evaluate(()=>{for(const run of app.runs)app.visibility[run_identifier(run)]=false;render_runs();});
    const row=id=>page.locator(`#runs_body tr[data-run-id="${id}"]`);
    await row("fixture_00").locator(".eye-button").click();
    assert.deepEqual(await page.evaluate(()=>app.runs.filter(run=>is_visible(run_identifier(run))).map(run_identifier).sort()),
      ["fixture_00","fixture_01","fixture_02","fixture_03"]);
    await row("fixture_01").locator(".eye-button").click();
    assert.equal(await page.evaluate(()=>is_visible("fixture_00")&&!is_visible("fixture_01")&&!is_visible("fixture_04")),true);
    await row("fixture_04").locator(".run-menu-button").click();
    assert.equal(await page.locator("#delete_grid_runs").isVisible(),false);
    await page.locator("#runs_nav").click();
    await row("fixture_00").locator(".run-menu-button").click();
    assert.equal(await page.locator("#delete_grid_runs").isVisible(),true);
    const header=page.locator(".runs-table th").filter({hasText:/^GB pk$/});
    const dimensions=await header.evaluate(node=>{
      const range=document.createRange();range.selectNodeContents(node);
      return {text:range.getBoundingClientRect().width,cell:node.clientWidth,padding:
        parseFloat(getComputedStyle(node).paddingLeft)+parseFloat(getComputedStyle(node).paddingRight)};
    });
    assert.ok(dimensions.text<=dimensions.cell-dimensions.padding,JSON.stringify(dimensions));
    await page.locator("#runner_nav").click();
    await page.waitForSelector("#runner_detail input[data-runner-field]");
    assert.equal(await page.locator("#runner_list .runner-status").first().textContent(),"ready");
    const geometry=page.locator('input[data-runner-field="--geometry-preset"]');
    const order=page.locator('input[data-runner-field="DEPTH.order"]');
    await geometry.fill("dense");
    assert.equal(await order.getAttribute("aria-invalid"),"true");
    await geometry.fill("dense,depth");
    assert.equal(await order.getAttribute("aria-invalid"),"false");
    const label=page.locator('input[data-runner-field="Recipe label"]');
    await label.fill(" ");
    assert.equal(await label.getAttribute("aria-invalid"),"true");
    await page.locator("#workspace_nav").click();
    await page.waitForFunction(()=>document.querySelector('[data-metric-chart-id="train/loss"] .plot-mount')?.data?.length===3);
    const run_name=await page.evaluate(()=>{
      const plot=document.querySelector('[data-metric-chart-id="train/loss"] .plot-mount');
      const index=plot.data.findIndex(trace=>trace.meta.instra_workspace_run_id==="fixture_00");
      Plotly.Fx.hover(plot,[{curveNumber:index,pointNumber:1}]);
      return plot.data[index].meta.instra_run_name;
    });
    await page.waitForSelector(".hoverlayer .hovertext");
    assert.ok((await page.locator(".hoverlayer .hovertext").allTextContents()).some(text=>text.includes(run_name)),run_name);
    await page.screenshot({path:"/tmp/instra-oct01-host-grid-hover.png"});
    assert.deepEqual(errors,[]);
    console.log("PASS chromium host-isolated eyes/local deletion menu, GB pk width, dense-only outlines, Ready and rendered run-name hover");
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
// ^^^ THOG
