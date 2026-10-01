// vvv THOG verify actual controls, curve emphasis, selection colours, confirmations and aligned wall-clock cells
"use strict";
const assert = require("node:assert/strict");
const {chromium, firefox} = require("playwright");
const address = process.env.INSTRA_TEST_URL || "http://127.0.0.1:8765";
async function check(type) {
  const browser = await type.launch({headless:true,
    ...(type===firefox && process.env.INSTRA_TEST_UNSANDBOXED==="1" ? {firefoxUserPrefs:{"security.sandbox.content.level":0,"fission.autostart":false,"dom.ipc.processCount":1}} : {}), ...(type === chromium ? {
    executablePath:process.env.INSTRA_CHROMIUM_PATH,args:["--no-sandbox","--disable-dev-shm-usage"]} : {})});
  try {
    const page = await browser.newPage({viewport:{width:1600,height:1100}});
    const errors = [];
    page.on("pageerror", error=>errors.push(error.message));
    const snapshot = await (await page.request.get(address+"/api/runner")).json();
    for (const grid of snapshot.grids) for (const run of grid.runs) {
      run.started_at = "2026-10-01T22:59:00Z";
      if (run.state !== "running") run.finished_at = "2026-10-02T00:05:00Z";
    }
    await page.route(address+"/api/runner",route=>route.fulfill({json:snapshot}));
    await page.route(address+"/api/deletions",route=>route.fulfill({json:{deletion_confirmation_timeout_days:7,
      pending:[{run_id:"pending",outstanding_hosts:["dreedle"],timeout_days:7}]}}));
    await page.goto(address+"/runs/fixture_00");
    await page.waitForFunction(()=>typeof app!=="undefined"&&app.runs.length===12);
    await page.waitForSelector("#instra-sep21-repairs-style",{state:"attached"});
    await page.evaluate(async()=>{
      await select_run("fixture_03",{manual:true});
      for (const run of app.runs) app.visibility[run_identifier(run)]=run.runner_grid_tag==="G-00001";
      render_runs();
    });
    const row = id=>page.locator(`#runs_body tr[data-run-id="${id}"]`);
    const colour = id=>row(id).locator(".run-link").evaluate(node=>getComputedStyle(node).color);
    assert.notEqual(await colour("fixture_00"),await colour("fixture_03"));
    await row("fixture_00").locator(".run-menu-button").click();
    const gap = await page.locator("#delete_grid_separator").evaluate(node=>({height:node.getBoundingClientRect().height,
      top:parseFloat(getComputedStyle(node).marginTop),bottom:parseFloat(getComputedStyle(node).marginBottom)}));
    assert.ok(gap.top>=7 && gap.bottom>=7,JSON.stringify(gap));
    const dialog = page.waitForEvent("dialog");
    const delete_click = page.locator("#delete_grid_runs").click();
    const confirmation = await dialog;
    assert.equal(confirmation.message(),"Delete all (4) runs belonging to G-00001? Checkpoints, other logs and W&B runs remain.");
    await confirmation.dismiss();
    await delete_click;
    await page.evaluate(()=>close_run_menu());
    await page.locator("#workspace_nav").click();
    const card = page.locator('[data-metric-chart-id="train/loss"]');
    await page.waitForFunction(()=>document.querySelector('[data-metric-chart-id="train/loss"] .plot-mount')?.data?.length===4);
    assert.deepEqual(await card.locator(".metric-z-cycle").allTextContents(),["!+","!-","z+","z-"]);
    const traces = ()=>card.locator(".plot-mount").evaluate(node=>node.data.map(trace=>({id:trace.meta.instra_workspace_run_id,width:trace.line.width})));
    const initial = await traces(), first_front = initial.at(-1).id;
    assert.equal(initial.at(-1).width,4.6);
    assert.equal(await row(first_front).evaluate(node=>node.classList.contains("front-curve-run")),true);
    const selected_colour = await colour("fixture_03");
    if (first_front!=="fixture_03") assert.notEqual(await colour(first_front),selected_colour);
    await card.locator("button",{hasText:/^!\+$/}).click();
    await card.locator("button",{hasText:/^z\+$/}).click();
    let data=await traces(), next_front=data.at(-1).id;
    assert.notEqual(next_front,first_front);
    assert.equal(data.find(trace=>trace.id===first_front).width,4.6);
    assert.equal(data.at(-1).width,4.6);
    assert.equal(await row(next_front).evaluate(node=>node.classList.contains("front-curve-run")),true);
    await card.locator("button",{hasText:/^z-$/}).click();
    assert.equal((await traces()).at(-1).id,first_front);
    await card.locator("button",{hasText:/^!-$/}).click();
    await card.locator("button",{hasText:/^z\+$/}).click();
    assert.equal((await traces()).find(trace=>trace.id===first_front).width,2.4);
    await page.waitForTimeout(2500);
    assert.equal((await traces()).at(-1).id,next_front,"poll reset z order");
    assert.equal(await row(next_front).evaluate(node=>node.classList.contains("front-curve-run")),true);
    await card.locator(".maximize-button").click();
    assert.equal(await card.locator(".metric-z-cycle").filter({hasText:/^z-$/}).isVisible(),true);
    await card.locator(".maximize-button").click();
    await page.locator("#runner_nav").click();
    await page.locator('[data-runner-tab="history"]').click();
    await page.locator(".runner-history-grid-row > button").filter({hasText:"Grid 1"}).click();
    assert.deepEqual(await page.locator(".runner-run-headings strong").allTextContents(),["Run ID","start","end","Host","GPU","State","Step","Loss","Best loss","Profiling"]);
    const boundaries = await page.locator(".runner-run-identity").evaluateAll(rows=>rows.map(row=>[...row.children].slice(0,3).map(node=>({left:node.getBoundingClientRect().left,text:node.textContent,title:node.title}))));
    assert.ok(boundaries.length===4 && boundaries.every(values=>values[1].text!=="—"&&values[2].text!=="—"));
    for(let column=0;column<3;column++)assert.ok(boundaries.every(values=>Math.abs(values[column].left-boundaries[0][column].left)<1));
    assert.ok(boundaries[0][1].title.includes("2026")&&boundaries[0][2].title.includes("2026"));
    await page.screenshot({path:`/workspace/scratch/67907f56323f/instra-timing-${type.name()}.png`});
    await page.locator('[data-runner-tab="progress"]').click();
    assert.deepEqual(await page.locator(".runner-run-headings strong").allTextContents(),["Run ID","start","end","Host","GPU","State","Step","Loss","Best loss","Profiling"]);
    assert.equal(await page.locator(".runner-run-identity span").nth(2).textContent(),"—");
    await page.locator("#runs_nav").click();
    await page.evaluate(()=>{app.runs.find(run=>run_identifier(run)==="fixture_00").remote_copy=true;render_runs();});
    await row("fixture_00").locator(".run-menu-button").click();
    assert.equal(await page.locator("#delete_grid_runs").isVisible(),false);
    const force_dialog=page.waitForEvent("dialog");
    const force_click = page.locator("#force_delete_local_copy").click();
    const force=await force_dialog;
    assert.equal(force.message(),"This deletes only this Instra’s local copy. The authoritative run on the producing host is unaffected. If that run still exists, monitoring will download it again.");
    await force.dismiss();
    await force_click;
    await page.screenshot({path:`/workspace/scratch/67907f56323f/instra-controls-${type.name()}.png`});
    assert.deepEqual(errors,[]);
    console.log("PASS",type.name(),"curve controls/bolding/polling, eyes/front rows, deletion messages/spacing, aligned start/end clocks");
  } finally {await browser.close();}
}
(async()=>{for(const name of (process.env.INSTRA_TEST_BROWSERS||"chromium,firefox").split(","))await check({chromium,firefox}[name]);})().catch(error=>{console.error(error);process.exitCode=1;});
// ^^^ THOG
