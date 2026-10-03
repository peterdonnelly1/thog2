// vvv THOG exercise October fixes in the complete dashboard with real Plotly and persistent browser storage
"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs");
const {chromium,firefox}=require("playwright");
const address=process.env.INSTRA_TEST_URL||"http://127.0.0.1:8765";
async function check(type) {
  const browser=await type.launch({headless:true,...(type===chromium && process.env.INSTRA_CHROMIUM_PATH ?
    {executablePath:process.env.INSTRA_CHROMIUM_PATH,args:["--no-sandbox","--disable-dev-shm-usage","--no-zygote","--use-gl=angle","--use-angle=swiftshader","--enable-unsafe-swiftshader"]}: {})});
  const context=await browser.newContext({viewport:{width:1600,height:1100},acceptDownloads:true,timezoneId:"UTC"});
  const page=await context.newPage(),errors=[];
  page.on("pageerror",error=>{errors.push(error.message);console.log("PAGE ERROR",error.stack);});
  const snapshot=await (await page.request.get(address+"/api/runner")).json();
  for(const grid of snapshot.grids)for(const run of grid.runs)run.attempts=[{started_at:"2026-10-05T07:07:12Z",finished_at:"2026-10-05T19:08:13Z"}];
  await page.route(address+"/api/runner",route=>route.fulfill({json:snapshot}));
  await page.goto(address+"/runs/fixture_00");
  await page.waitForFunction(()=>window.instra_oct03_controls && app.runs.length===12);
  const row=id=>page.locator(`#runs_body tr[data-run-id="${id}"]`);
  const heading=key=>page.locator(`.runs-table th[data-instra-column-key="${key}"]`);
  assert.deepEqual(await page.locator(".runs-table thead tr > th").evaluateAll(nodes=>nodes.slice(0,2).map(node=>node.dataset.instraColumnKey)),["menu","select"]);
  assert.equal(await heading("state").evaluate(node=>getComputedStyle(node).textAlign),"center");
  assert.equal(await heading("gb").textContent(),"GB pk");
  assert.equal(await heading("gb").evaluate(node=>getComputedStyle(node).textTransform),"none");
  const alignment=await page.evaluate(()=>({heading:document.querySelector('.runs-table th[data-instra-column-key="name"]').getBoundingClientRect().bottom,
    names:[...document.querySelectorAll('.runs-table td.name-column')].map(node=>node.getBoundingClientRect().top)}));
  assert.ok(alignment.names.every(top=>top>=alignment.heading),JSON.stringify(alignment));
  assert.equal(await row("fixture_00").locator(".run-row-trash svg").evaluate(node=>getComputedStyle(node).stroke),"rgb(179, 38, 46)");

  // Resize by mouse, reorder by native HTML drag events, then reload to check persistence and row identity.
  const handle=heading("loss").locator(".run-column-resizer"),before_width=(await heading("loss").boundingBox()).width;
  const bounds=await handle.boundingBox();await page.mouse.move(bounds.x+bounds.width/2,bounds.y+bounds.height/2);
  await page.mouse.down();await page.mouse.move(bounds.x+80,bounds.y+bounds.height/2);await page.mouse.up();
  assert.ok((await heading("loss").boundingBox()).width>before_width+65);
  const transfer=await page.evaluateHandle(()=>new DataTransfer());
  await heading("state").dispatchEvent("dragstart",{dataTransfer:transfer});
  await heading("loss").dispatchEvent("dragover",{dataTransfer:transfer});
  await heading("loss").dispatchEvent("drop",{dataTransfer:transfer});
  await heading("state").dispatchEvent("dragend",{dataTransfer:transfer});
  const stored_layout=await page.evaluate(()=>JSON.parse(localStorage.getItem("thog2_run_columns_layout_v1")));
  assert.ok(stored_layout.order.indexOf("state")<stored_layout.order.indexOf("loss"));
  await page.reload();await page.waitForFunction(()=>window.instra_oct03_controls && app.runs.length===12);
  assert.deepEqual(await page.locator(".runs-table thead tr > th").evaluateAll(nodes=>nodes.map(node=>node.dataset.instraColumnKey)),stored_layout.order);
  assert.ok((await heading("loss").boundingBox()).width>before_width+65);
  await page.evaluate(()=>{render_runs();render_runs();});
  assert.equal(await heading("name").locator(".run-name-column-resizer").count(),0);
  assert.equal(await row("fixture_00").locator('[data-instra-column-key="name"]').getAttribute("class"),"name-column");
  console.log("PASS",type.name(),"table alignment, red per-run trash, dragging, resizing and persisted column identity");

  const defaults=await page.evaluate(()=>[0,1,2,3].map(i=>colour_for_run(`fixture_0${i}`)));
  async function colour(id,hex) {await row(id).locator(".colour-dot").click();await page.locator("#colour_hex").fill(hex);await page.locator("#colour_hex").dispatchEvent("change");await page.locator("#runs_nav").click();}
  await colour("fixture_00","#CC3377");
  await row("fixture_00").locator(".run-menu-button").click();await page.locator("#reset_run_grid_colour").click();
  assert.equal(await page.evaluate(()=>colour_for_run("fixture_00")),defaults[0]);
  await colour("fixture_00","#CC3377");await colour("fixture_01","#AACC33");
  await row("fixture_03").locator(".run-menu-button").click();await page.locator("#reset_grid_colours").click();
  assert.deepEqual(await page.evaluate(()=>[0,1,2,3].map(i=>colour_for_run(`fixture_0${i}`))),defaults);
  async function centre(id,hex) {await row(id).locator(".run-menu-button").click();await page.locator("#change_grid_tone").click();await page.locator("#colour_hex").fill(hex);await page.locator("#colour_hex").dispatchEvent("change");await page.locator("#runs_nav").click();}
  await centre("fixture_03","#8888DD");const centred=await page.evaluate(()=>[0,1,2,3].map(i=>colour_for_run(`fixture_0${i}`)));
  await centre("fixture_00","#8888DD");assert.deepEqual(await page.evaluate(()=>[0,1,2,3].map(i=>colour_for_run(`fixture_0${i}`))),centred);
  assert.notDeepEqual(centred,defaults);
  await colour("fixture_00","#CC3377");await row("fixture_00").locator(".run-menu-button").click();await page.locator("#reset_grid_colours").click();
  assert.deepEqual(await page.evaluate(()=>[0,1,2,3].map(i=>colour_for_run(`fixture_0${i}`))),centred,"manual reset undid permanent centre");
  await page.reload();await page.waitForFunction(()=>window.instra_oct03_controls && app.runs.length===12);
  assert.deepEqual(await page.evaluate(()=>[0,1,2,3].map(i=>colour_for_run(`fixture_0${i}`))),centred);
  await row("fixture_00").locator(".run-menu-button").click();
  const separator=await page.locator("#delete_run").evaluate(node=>{const rule=node.nextElementSibling,s=getComputedStyle(rule);return [s.marginTop,s.marginBottom];});
  assert.equal(separator[0],separator[1]);await page.locator("#runs_nav").click();
  console.log("PASS",type.name(),"single/Grid colour resets, permanent centre from any member and restart persistence");

  // Real log endpoint: ANSI formatting, newest first, incremental polling, stale response cancellation.
  await page.locator('[data-detail-tab="logs"]').click();
  await page.waitForFunction(()=>document.getElementById("local_log_output")?.textContent.includes("CLI newest"));
  assert.equal(await page.locator("#local_log_output .local-log-line").first().textContent(),"CLI newest");
  assert.equal(await page.locator("#local_log_output span",{hasText:"CLI oldest"}).evaluate(node=>node.style.color),"rgb(0, 205, 0)");
  await page.evaluate(()=>{instra_run_logs.ingest_log("partial",false);instra_run_logs.ingest_log(" completed\n",false);});
  assert.equal(await page.locator("#local_log_output .local-log-line").first().textContent(),"partial completed");
  await page.evaluate(()=>instra_run_logs.ingest_log(Array.from({length:6000},(_,i)=>`line ${i}`).join("\n")+"\n",true));
  assert.equal(await page.locator("#local_log_output .local-log-line").count(),5000);
  await page.locator('[data-detail-tab="charts"]').click();
  assert.equal(await page.locator("#run_logs_pane").isVisible(),false);
  assert.equal(await page.evaluate(()=>document.getElementById("processing_chart_group").hidden),true,"run without processing capture showed the Processing group");
  const captured=await (await page.request.get(address+"/api/runs")).json();
  for(const run of captured.runs)if(run.dashboard_run_id==="fixture_11" || run.run_id==="fixture_11")run.configuration={...run.configuration,premat:"disabled"};
  await page.route(address+"/api/runs",route=>route.fulfill({json:captured}));
  await page.reload();await page.waitForFunction(()=>window.instra_oct03_controls && app.runs.length===12);
  await page.evaluate(()=>select_run("fixture_11"));await page.waitForTimeout(800);
  assert.equal(await page.locator('[data-detail-tab="premat"]').isVisible(),false);
  await page.unroute(address+"/api/runs");await page.evaluate(()=>select_run("fixture_00"));await page.waitForTimeout(800);
  console.log("PASS",type.name(),"Logs ANSI/order/partial lines/bounded retention, disabled PREMAT and absent capture");

  await page.locator("#runner_nav").click();await page.locator('[data-runner-tab="recipes"]').click();
  await page.locator(".runner-recipe-row > button:first-child",{hasText:"Fixture Recipe"}).click();
  const buttons=await page.locator(".runner-actions > button").allTextContents();
  assert.ok(buttons.indexOf("Reset")===buttons.indexOf("Save")+1,JSON.stringify(buttons));
  const field=page.locator('input[data-runner-field="--max-iters"]');await field.fill("999");
  await page.locator(".runner-actions > button",{hasText:/^Reset$/}).click();
  assert.notEqual(await field.inputValue(),"999");
  await page.locator('[data-runner-tab="history"]').click();
  assert.ok((await page.locator(".runner-history-wall-time").allTextContents()).filter(text=>text!=="—").every(text=>/^\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/.test(text)));
  await page.locator(".runner-history-grid-row > button:first-child",{hasText:"Grid 1"}).click();
  assert.deepEqual((await page.locator(".runner-run-headings strong").allTextContents()).slice(0,4),["Run ID","--geometry-preset","start","end"]);
  const summary=page.locator(".runner-run > .runner-run-identity").first();
  const values=await summary.locator(":scope > *").allTextContents();
  assert.equal(values[1],"depth");assert.equal(values[2],"26-10-05 07:07:12");assert.equal(values[3],"26-10-05 19:08:13");
  console.log("PASS",type.name(),"Recipe Reset placement/defaults, 24-hour dated History and geometry column");

  await page.locator("#runs_nav").click();
  await page.evaluate(()=>{for(const run of app.runs)app.visibility[run_identifier(run)]=["fixture_00","fixture_01"].includes(run_identifier(run));save_json("thog2_local_run_visibility",app.visibility);render_runs();});
  await page.locator("#workspace_nav").click();
  await page.waitForFunction(()=>document.querySelector('[data-metric-chart-id="train/loss"] .plot-mount')?.data?.length===2);
  await page.waitForFunction(()=>document.getElementById("training_throughput_plot")?.data?.length===2);
  await page.evaluate(()=>restore_maximized_chart());await page.waitForTimeout(250);
  const started=Date.now();await page.locator('[data-metric-chart-id="train/loss"] .maximize-button').click();
  await page.waitForFunction(()=>document.querySelector('[data-metric-chart-id="train/loss"]').classList.contains("maximized"));
  await page.waitForTimeout(250);const latency=Date.now()-started;assert.ok(latency<2000,`two-curve maximize took ${latency}ms`);
  const widths=await page.evaluate(()=>document.querySelector('[data-metric-chart-id="train/loss"] .plot-mount').data.map(trace=>trace.line.width));
  assert.ok(Math.max(...widths)<=3.5 && Math.max(...widths)>Math.min(...widths));
  const loss=page.locator('[data-metric-chart-id="train/loss"]');
  async function download(card,format) {
    await card.locator(".chart-download-options > summary").click();
    const event=page.waitForEvent("download");await card.getByRole("button",{name:format==="json"?"Download as JSON":"Download as Excel 97-2003 Workbook",exact:true}).click();
    const file=await event;return {name:file.suggestedFilename(),data:fs.readFileSync(await file.path())};
  }
  const json_download=await download(loss,"json"),payload=JSON.parse(json_download.data);
  assert.equal(payload.metric,"training_loss");assert.equal(payload.series.length,2);assert.ok(payload.series.every(curve=>curve.x.length===5 && curve.y.length===5));
  assert.equal(json_download.name,`${payload.selected_run_name}_training_loss.json`);
  await page.evaluate(async()=>{const mount=document.querySelector('[data-metric-chart-id="train/loss"] .plot-mount');mount.data[0].meta={...mount.data[0].meta,display_only:true};await Plotly.restyle(mount,{visible:"legendonly"},[0]);});
  const one_visible=JSON.parse((await download(loss,"json")).data);
  assert.equal(one_visible.series.length,1,"legend-hidden curve leaked into exported data");
  await page.evaluate(()=>Plotly.restyle(document.querySelector('[data-metric-chart-id="train/loss"] .plot-mount'),{visible:true},[0]));
  const xls_download=await download(loss,"xls");assert.equal(xls_download.data.subarray(0,8).toString("hex"),"d0cf11e0a1b11ae1");
  assert.equal(xls_download.name,`${payload.selected_run_name}_training_loss.xls`);
  await page.evaluate(()=>restore_maximized_chart());
  const throughput=await download(page.locator("#training_throughput_card"),"json");
  const throughput_payload=JSON.parse(throughput.data);assert.equal(throughput_payload.metric,"tokens_throughput");assert.equal(throughput_payload.series.length,2);
  console.log("PASS",type.name(),`two-curve maximize ${latency}ms, visible-curve JSON/BIFF8 downloads and tokens throughput`);

  // Use the live Plotly Processing renderer to check viewport fit and zoom reset across navigation.
  await page.locator("#runs_nav").click();await page.evaluate(()=>{
    processing_view.available=true;processing_view.trace_available=true;processing_view.charts_tab_visible=true;processing_sync_visibility();
    processing_plot("processing_timeline_plot",[{type:"scatter",x:[0,10,20,30],y:[1,2,3,4]}],{xaxis:{range:[0,30]},yaxis:{autorange:true}});
  });await page.waitForTimeout(400);
  await page.evaluate(async()=>{const mount=document.getElementById("processing_timeline_plot");mount._processing_layer_zoom_range=[5,10];mount._processing_layer_zoom_run_id=app.current_run_id;await Plotly.relayout(mount,{"xaxis.range":[5,10],"xaxis.autorange":false});});
  await page.locator('[data-detail-tab="overview"]').click();
  assert.equal(await page.evaluate(()=>document.getElementById("processing_timeline_plot")._processing_layer_zoom_range),null);
  await page.locator('[data-detail-tab="charts"]').click();
  await page.evaluate(async()=>{await processing_plot("processing_timeline_plot",[{type:"scatter",x:[0,10,20,30],y:[1,2,3,4]}],{xaxis:{range:[0,30]},yaxis:{autorange:true}});});
  assert.deepEqual(await page.evaluate(()=>document.getElementById("processing_timeline_plot")._fullLayout.xaxis.range),[0,30]);
  await page.evaluate(()=>{processing_view.available=true;processing_view.trace_available=true;processing_view.charts_tab_visible=true;processing_sync_visibility();by_id("processing_contention_card").hidden=false;toggle_maximized_chart("processing_contention");});
  await page.waitForTimeout(350);
  const fit=await page.locator("#processing_contention_card").evaluate(node=>({bottom:node.getBoundingClientRect().bottom,height:innerHeight,top:node.getBoundingClientRect().top}));
  assert.ok(fit.bottom<=fit.height+2 && fit.bottom>fit.top+100,JSON.stringify(fit));
  await page.evaluate(()=>{restore_maximized_chart();select_run("fixture_01");});
  assert.equal(await page.evaluate(()=>document.getElementById("processing_timeline_plot")._processing_layer_zoom_range),null);
  console.log("PASS",type.name(),"Processing zoom reset on view/run changes and maximized temporal-overlap viewport fit");

  // Repeated delayed requests and run/view changes must leave one responsive view with no detached Plotly jobs.
  await page.route(/\/api\/chart-group\?/,async route=>{await new Promise(resolve=>setTimeout(resolve,100));try{await route.continue();}catch(_) {}});
  for(let cycle=0;cycle<30;cycle++) {
    await page.evaluate(i=>select_run(`fixture_${String(i%12).padStart(2,"0")}`),cycle);
    await page.locator(cycle%2?"#workspace_nav":"#runs_nav").click();
    await page.waitForTimeout(40);
  }
  await page.waitForTimeout(200);await page.unroute(/\/api\/chart-group\?/);
  await page.locator("#runs_nav").click();await page.evaluate(()=>select_run("fixture_00"));await page.waitForTimeout(1000);
  assert.equal(await page.evaluate(()=>app.workspace_mode),false);
  assert.deepEqual(errors,[]);
  await page.screenshot({path:`/tmp/instra-oct03-${type.name()}-final.png`});
  await browser.close();console.log("PASS",type.name(),"30 delayed navigation cycles without JavaScript/Plotly errors");
}
(async()=>{for(const name of (process.env.INSTRA_TEST_BROWSERS||"chromium,firefox").split(","))await check({chromium,firefox}[name]);})().catch(error=>{console.error(error.stack);process.exit(1);});
// ^^^ THOG
