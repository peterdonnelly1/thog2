// vvv THOG exercise October 7 controls in Chromium and Firefox with actual Plotly and Recipe persistence
"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),{chromium,firefox}=require("playwright");
const address=process.env.INSTRA_TEST_URL || "http://127.0.0.1:8765";
const results=[];
async function check(type) {
  const browser=await type.launch({headless:true,...(type===chromium && process.env.INSTRA_CHROMIUM_PATH ? {
    executablePath:process.env.INSTRA_CHROMIUM_PATH,args:["--no-sandbox","--no-zygote","--disable-dev-shm-usage","--use-gl=angle","--use-angle=swiftshader"],
  } : {}),...(type===firefox ? {env:{...process.env,MOZ_DISABLE_CONTENT_SANDBOX:"1"}} : {})});
  try {
    // Exercise two desktop heights while using actual native pointer and keyboard input.
    const page=await browser.newPage({viewport:{width:1800,height:type===firefox ? 900 : 1100},timezoneId:"UTC"});
    page.setDefaultTimeout(20000);
    async function click_settings_button(id) {
      const button=page.locator(`#${id}`);
      // The host drops some native Firefox footer mouse events even on a plain test page.
      // Native Enter also verifies button accessibility; Chromium covers the native mouse path.
      if(type===firefox){await button.focus();await page.keyboard.press("Enter");}
      else await button.click();
    }
    const errors=[],deletes=[],legacy_deletes=[];let remote_mode=false,dense_mode=false;
    page.on("pageerror",error=>errors.push(error.message));
    await page.route(/\/api\/run\?/,route=>{
      if(route.request().method()!=="DELETE")return route.continue();
      legacy_deletes.push(route.request().url());return route.fulfill({json:{deleted:true}});
    });
    await page.route(/\/api\/runner$/,async route=>{
      const response=await route.fetch(),snapshot=await response.json();
      for(const grid of snapshot.grids)for(const run of grid.runs)if(run.state==="running") {
        run.started_at=new Date(Date.now()-2000).toISOString();run.attempts=[{started_at:run.started_at}];run.estimated_duration_seconds=60;
      }
      return route.fulfill({json:snapshot});
    });
    await page.route(/\/api\/runs$/,async route=>{
      if(route.request().method()!=="DELETE") {
        const response=await route.fetch(),catalogue=await response.json();
        for(const run of catalogue.runs) {
          run.remote_copy=remote_mode && ["fixture_00","fixture_01"].includes(run.dashboard_run_id);
          if(dense_mode && run.dashboard_run_id==="fixture_00")run.preset="dense";
        }
        return route.fulfill({json:catalogue});
      }
      const payload=route.request().postDataJSON();deletes.push(payload);
      return route.fulfill({json:{deleted_run_ids:[],queued_run_ids:payload.force_local ? payload.run_ids : [],errors:[]}});
    });
    await page.goto(address+"/runs/fixture_00");
    await page.waitForFunction(()=>window.instra_oct03_controls && app.runs.length===12 && window.instra_demand_runtime);
    const row=id=>page.locator(`#runs_body tr[data-run-id="${id}"]`);
    const bin=page.locator("#delete_selected_runs");
    assert.equal(await page.locator(".run-row-trash").count(),0);
    assert.equal(await row("fixture_00").locator(".colour-dot").evaluate(node=>getComputedStyle(node).borderRadius),"0px");
    const inactive=await bin.evaluate(node=>getComputedStyle(node.querySelector("svg")).stroke);
    assert.equal(inactive,"rgb(164, 170, 178)");
    const checkbox=row("fixture_00").locator('.check-column input');
    await checkbox.focus();
    const scroll_before=await page.evaluate(()=>({body:window.scrollY,table:by_id("runs_body").closest(".runs-table-wrap").scrollTop}));
    await page.keyboard.press("Space");assert.equal(await checkbox.isChecked(),true);
    assert.equal(await page.evaluate(()=>document.activeElement.closest("tr").dataset.runId),"fixture_00");
    assert.equal(await bin.isEnabled(),true);
    assert.equal(await bin.evaluate(node=>getComputedStyle(node.querySelector("svg")).stroke),"rgb(52, 59, 68)");
    assert.equal(await bin.getAttribute("title"),"Delete Instra data for 1 selected runs");
    await page.evaluate(()=>render_runs());
    assert.equal(await page.evaluate(()=>document.activeElement.closest("tr").dataset.runId),"fixture_00");
    await page.keyboard.press("Space");assert.equal(await checkbox.isChecked(),false);
    assert.deepEqual(await page.evaluate(()=>({body:window.scrollY,table:by_id("runs_body").closest(".runs-table-wrap").scrollTop})),scroll_before);
    await row("fixture_00").locator(".run-link").focus();await page.keyboard.press("Space");
    assert.equal(await checkbox.isChecked(),true);await page.keyboard.press("Space");
    assert.equal(await checkbox.isChecked(),false);
    console.log("PASS",type.name(),"toolbar bin, square colours and Space focus/scroll");

    await page.evaluate(()=>{for(const run of app.runs)app.visibility[run_identifier(run)]=false;app.grid_visibility={};render_runs();});
    const grid=row("fixture_00").locator(".grid-visibility-button");
    await grid.click();
    assert.deepEqual(await page.evaluate(()=>app.runs.filter(run=>is_visible(run_identifier(run))).map(run=>run_identifier(run)).sort()),["fixture_00","fixture_01","fixture_02","fixture_03"]);
    assert.equal(await row("fixture_03").locator(".grid-visibility-button").getAttribute("aria-pressed"),"true");
    await row("fixture_01").locator(".visibility-column .eye-button").click();
    assert.equal(await page.evaluate(()=>is_visible("fixture_01")),false);
    assert.equal(await grid.getAttribute("aria-pressed"),"true");
    await row("fixture_03").locator(".grid-visibility-button").click();
    assert.equal(await page.evaluate(()=>app.runs.some(run=>is_visible(run_identifier(run)))),false);
    await row("fixture_02").locator(".visibility-column .eye-button").click();
    assert.deepEqual(await page.evaluate(()=>app.runs.filter(run=>is_visible(run_identifier(run))).map(run=>run_identifier(run))),["fixture_02"]);
    assert.equal(await grid.getAttribute("aria-pressed"),"false");
    const positions=await row("fixture_00").evaluate(node=>[node.querySelector(".grid-visibility-column").cellIndex,node.querySelector(".visibility-column").cellIndex]);
    assert.equal(positions[1],positions[0]+1);
    await page.reload();await page.waitForFunction(()=>window.instra_oct03_controls && app.runs.length===12);
    assert.equal(await grid.getAttribute("aria-pressed"),"false");assert.equal(await page.evaluate(()=>is_visible("fixture_02")),true);
    assert.equal(await page.evaluate(()=>{const template=app.runs[0];app.runs=[...app.runs,{...template,dashboard_run_id:"new_member",local_run_id:"new_member"}];return is_visible("new_member");}),false);
    await page.evaluate(()=>{app.runs=app.runs.filter(run=>run_identifier(run)!=="new_member");});
    console.log("PASS",type.name(),"grid on/off, individual override, persistence and new member inheritance");

    remote_mode=true;await page.evaluate(()=>refresh_catalog());
    await checkbox.check();await row("fixture_02").locator('.check-column input').check();
    assert.equal(await bin.isEnabled(),false,"mixed bulk selection was allowed");
    assert.equal(await bin.evaluate(node=>getComputedStyle(node.querySelector("svg")).stroke),"rgb(52, 59, 68)");
    await row("fixture_02").locator('.check-column input').uncheck();await row("fixture_01").locator('.check-column input').check();
    assert.equal(await bin.isEnabled(),true);assert.match(await bin.getAttribute("title"),/force delete local copies/);
    let dialog_promise=page.waitForEvent("dialog"),click=bin.click(),dialog=await dialog_promise;
    assert.match(dialog.message(),/^Delete Instra data for 2 selected runs/);assert.match(dialog.message(),/excluded until you resume/);
    await dialog.dismiss();await click;assert.equal(deletes.length,0,"cancelled delete reached backend");
    await row("fixture_00").locator(".run-menu-button").click();assert.equal(await page.locator("#force_delete_local_copy").isVisible(),true);
    await page.evaluate(()=>close_run_menu());
    dialog_promise=page.waitForEvent("dialog");click=bin.click();dialog=await dialog_promise;await dialog.accept();await click;
    await page.waitForFunction(()=>app.selected.size===0 && app.runs.length===12 && by_id("toast").textContent.startsWith("Queued local copy deletion for 2 runs"));
    assert.deepEqual(deletes,[{run_ids:["fixture_00","fixture_01"],force_local:true}]);
    assert.deepEqual(legacy_deletes,[],"toolbar also invoked the retired per-run deletion path");
    remote_mode=false;
    await page.evaluate(()=>refresh_catalog());
    console.log("PASS",type.name(),"all-remote forced deletion, mixed rejection, cancellation and single-run menu");

    await page.evaluate(()=>{for(const run of app.runs)app.visibility[run_identifier(run)]=["fixture_00","fixture_01"].includes(run_identifier(run));render_runs();});
    await page.locator("#workspace_nav").click();
    try {
      await page.waitForFunction(()=>document.querySelector('[data-metric-chart-id="train/loss"] .plot-mount')?.data?.length===2);
    } catch(error) {
      console.error("Loss readiness",await page.evaluate(()=>{
        const mount=document.querySelector('[data-metric-chart-id="train/loss"] .plot-mount');
        return {workspace_mode:app.workspace_mode,visibility:app.visibility,data:mount?.data?.length,ready:mount?.dataset.plotReady,
          rect:mount?.getBoundingClientRect().toJSON(),group:mount?.closest('.local-metric-group')?.className,
          current:app.current_run_id,visible:app.runs.filter(run=>is_visible(run_identifier(run))).map(run=>run_identifier(run))};
      }),errors);throw error;
    }
    const loss=page.locator('[data-metric-chart-id="train/loss"]'),mount=loss.locator(".plot-mount");
    await row("fixture_00").locator(".grid-visibility-button").click();
    await page.waitForFunction(()=>document.querySelector('[data-metric-chart-id="train/loss"] .plot-mount')?.data?.length===4);
    await row("fixture_01").locator(".visibility-column .eye-button").click();
    await page.waitForFunction(()=>document.querySelector('[data-metric-chart-id="train/loss"] .plot-mount')?.data?.length===3);
    await row("fixture_00").locator(".grid-visibility-button").click();
    await page.waitForFunction(()=>!app.runs.some(run=>is_visible(run_identifier(run))) && !document.querySelector('[data-metric-chart-id="train/loss"] .plot-mount')?.data?.length);
    await row("fixture_00").locator(".grid-visibility-button").click();
    await page.waitForFunction(()=>document.querySelector('[data-metric-chart-id="train/loss"] .plot-mount')?.data?.length===4);
    await row("fixture_02").locator(".visibility-column .eye-button").click();await row("fixture_03").locator(".visibility-column .eye-button").click();
    await page.waitForFunction(()=>document.querySelector('[data-metric-chart-id="train/loss"] .plot-mount')?.data?.length===2);
    await loss.locator(".chart-settings-button").click();
    await page.waitForFunction(()=>by_id("chart_settings_preview").dataset.plotReady==="true");
    assert.equal(await page.evaluate(()=>by_id("chart_settings_preview").data.length),2);
    await page.locator("#chart_x_min").fill("2");await page.locator("#chart_x_max").fill("4");
    await page.locator("#chart_y_min").fill("4.5");await page.locator("#chart_y_max").fill("5.1");
    await page.waitForFunction(()=>JSON.stringify(by_id("chart_settings_preview")._fullLayout.xaxis.range)==="[2,4]");
    await page.locator('[data-chart-settings-tab="display"]').click();
    await page.locator("#chart_smoothing").evaluate(node=>{node.value="0.4";node.dispatchEvent(new Event("input",{bubbles:true}));});
    await page.locator("#chart_show_minor_grid").check();
    await page.waitForFunction(()=>{
      const mount=by_id("chart_settings_preview");return mount._fullLayout?.xaxis.minor?.dtick===mount._fullLayout?.xaxis.dtick/10;
    });
    await page.locator('[data-chart-settings-tab="data"]').click();
    await click_settings_button("reset_chart_ranges");
    try {await page.waitForFunction(()=>["x_min","x_max","y_min","y_max"].every(name=>by_id(`chart_${name}`).value===""));}
    catch(error) {
      console.error("Reset geometry",await page.evaluate(()=>({viewport:[innerWidth,innerHeight],button:by_id("reset_chart_ranges").getBoundingClientRect().toJSON(),
        values:["x_min","x_max","y_min","y_max"].map(name=>by_id(`chart_${name}`).value)})),errors);throw error;
    }
    await page.waitForFunction(()=>by_id("chart_settings_preview")._fullLayout.xaxis.autorange && by_id("chart_settings_preview")._fullLayout.yaxis.autorange);
    for(const name of ["x_min","x_max","y_min","y_max"])assert.equal(await page.locator(`#chart_${name}`).inputValue(),"");
    assert.equal(await page.locator("#chart_smoothing").inputValue(),"0.4");
    assert.equal(await page.locator("#chart_show_minor_grid").isChecked(),true);
    await click_settings_button("save_chart_settings");
    await page.waitForFunction(()=>document.querySelector('[data-metric-chart-id="train/loss"] .plot-mount')._fullLayout?.xaxis.minor.showgrid===true);
    const intervals=await mount.evaluate(node=>["xaxis","yaxis"].map(name=>({major:node._fullLayout[name].dtick,minor:node._fullLayout[name].minor.dtick,color:node._fullLayout[name].minor.gridcolor,width:node._fullLayout[name].minor.gridwidth})));
    assert.ok(intervals.every(axis=>axis.minor===axis.major/10 && axis.color==="#f4f5f7" && axis.width===0.5));
    await mount.evaluate(node=>Plotly.relayout(node,{"xaxis.range":[1,3],"yaxis.range":[4.6,4.8]}));
    await page.waitForFunction(()=>{const m=document.querySelector('[data-metric-chart-id="train/loss"] .plot-mount');return m._fullLayout.xaxis.minor.dtick===m._fullLayout.xaxis.dtick/10 && m._fullLayout.yaxis.minor.dtick===m._fullLayout.yaxis.dtick/10;});
    console.log("PASS",type.name(),"live loss preview, range-only Reset and exact minor spacing after zoom");

    const traces=()=>mount.evaluate(node=>node.data.map(trace=>({id:trace.meta.instra_workspace_run_id,width:trace.line.width})));
    const first_front=(await traces()).at(-1).id;
    await loss.getByRole("button",{name:"Remove persistent bolding from the current front curve",exact:true}).click();
    assert.equal((await traces()).at(-1).width,2.4,"unbold did not affect the front curve");
    await loss.getByRole("button",{name:"Keep the current front curve bold when its z-order changes",exact:true}).click();
    await loss.getByRole("button",{name:"Bring the next Workspace run to the front",exact:true}).click();
    assert.equal((await traces()).find(trace=>trace.id===first_front).width,3.5);
    await page.waitForTimeout(2300);assert.equal((await traces()).find(trace=>trace.id===first_front).width,3.5);
    await loss.locator(".chart-settings-button").click();
    await page.waitForFunction(()=>by_id("chart_settings_preview").dataset.plotReady==="true");
    await page.locator("#chart_settings_preview").evaluate(node=>Plotly.relayout(node,{"xaxis.range":[1,3],"yaxis.range":[4.6,4.8]}));
    await page.waitForFunction(()=>{const m=by_id("chart_settings_preview");return m._fullLayout.xaxis.minor.dtick===m._fullLayout.xaxis.dtick/10 && m._fullLayout.yaxis.minor.dtick===m._fullLayout.yaxis.dtick/10;});
    await page.locator('[data-chart-settings-tab="display"]').click();assert.equal(await page.locator("#chart_show_minor_grid").isChecked(),true);
    await page.locator("#chart_show_minor_grid").uncheck();await click_settings_button("save_chart_settings");
    await page.waitForFunction(()=>document.querySelector('[data-metric-chart-id="train/loss"] .plot-mount')._fullLayout.xaxis.minor?.showgrid!==true);
    console.log("PASS",type.name(),"pin/unpin front curves and persisted minor-grid preference");

    await page.locator("#runner_nav").click();await page.locator('[data-runner-tab="recipes"]').click();
    const recipe_row=page.locator(".runner-recipe-row").filter({hasText:"Fixture Recipe"});
    assert.deepEqual((await recipe_row.locator("button").allTextContents()).slice(1),["Edit","Rename","Delete"]);
    await page.locator(".runner-recipe-add-row button").click();
    const label=page.locator('[data-runner-field="Recipe label"]');
    const unique=`October 7 ${type.name()}`;await label.fill(unique);
    await page.locator(".runner-actions > button",{hasText:/^Save$/}).click();
    await page.waitForSelector(`.runner-recipe-row button[title="${unique}"]`);
    let state=await (await page.request.get(address+"/api/runner")).json();
    const saved=state.recipes.find(item=>item.recipe.label===unique),count=state.recipes.length;
    const saved_row=()=>page.locator(".runner-recipe-row").filter({has:page.locator(`button[title="${unique}"]`)});
    await saved_row().getByRole("button",{name:"Edit",exact:true}).click();
    const max_iters=page.locator('input[data-runner-field="--max-iters"]');await max_iters.fill("77");
    await page.locator(".runner-actions > button",{hasText:/^Save$/}).click();
    await page.waitForFunction(()=>!document.querySelector("#runner_message").textContent.endsWith("…"));
    state=await (await page.request.get(address+"/api/runner")).json();
    assert.equal(state.recipes.length,count);assert.equal(state.recipes.find(item=>item.recipe_id===saved.recipe_id).recipe.parameters["--max-iters"],77);
    await saved_row().getByRole("button",{name:"Rename",exact:true}).click();
    await page.locator("#runner_rename_input").fill(unique+" renamed");await page.locator("#runner_rename_form button[type=submit]").click();
    await page.waitForSelector(`.runner-recipe-row button[title="${unique} renamed"]`);
    state=await (await page.request.get(address+"/api/runner")).json();
    assert.equal(state.recipes.length,count);assert.equal(state.recipes.find(item=>item.recipe_id===saved.recipe_id).recipe.label,unique+" renamed");
    const name_widths=await page.evaluate(()=>{
      const view=by_id("runner_view"),row=document.querySelector(".runner-recipe-row"),name=row.querySelector("button");
      const edit=[...row.querySelectorAll("button")].find(button=>button.textContent==="Edit"),current=name.getBoundingClientRect().width;
      edit.style.display="none";view.classList.remove("runner-recipes");
      const previous=name.getBoundingClientRect().width;
      edit.style.removeProperty("display");view.classList.add("runner-recipes");return {current,previous};
    });
    assert.ok(name_widths.current>=name_widths.previous-1,JSON.stringify(name_widths));
    const recipe_width=await page.locator("#runner_detail").evaluate(node=>node.getBoundingClientRect().left);
    await page.locator('[data-runner-tab="progress"]').click();
    assert.ok(recipe_width>await page.locator("#runner_detail").evaluate(node=>node.getBoundingClientRect().left)+50,"Edit shrank the name column instead of expanding the left panel");
    assert.deepEqual(await page.locator(".runner-run-headings strong").allTextContents(),["Run ID","preset","start","end","est. end","Host","GPU","State","Step","Loss","Best loss"]);
    const progress=page.locator(".runner-run-identity").first();
    assert.match(await progress.locator("span").nth(2).textContent(),/^\d{2}-\d{2}-\d{2}  \d{2}:\d{2}:\d{2}$/);
    assert.match(await progress.locator("span").nth(4).textContent(),/^\d{2}-\d{2}-\d{2}  \d{2}:\d{2}:\d{2}$/);
    await page.locator('[data-runner-tab="history"]').click();await page.locator(".runner-history-grid-row > button:first-child",{hasText:"Grid 1"}).click();
    assert.equal((await page.locator(".runner-run-headings strong").allTextContents())[1],"preset");
    assert.ok((await page.locator(".runner-history-wall-time").allTextContents()).filter(value=>value!=="—").every(value=>/^\d{2}-\d{2}-\d{2}  \d{2}:\d{2}:\d{2}$/.test(value)));
    await page.locator("#runs_nav").click();
    await page.evaluate(()=>select_run("fixture_00"));
    await page.waitForFunction(()=>!app.workspace_mode && document.querySelector('[data-metric-chart-id="train/loss"] .plot-mount')?.data?.length===1);
    await loss.locator(".chart-settings-button").click();
    await page.waitForFunction(()=>by_id("chart_settings_preview").dataset.plotReady==="true" && by_id("chart_settings_preview").data?.length===1);
    await page.locator("#chart_x_min").fill("1");await page.locator("#chart_x_max").fill("3");
    await page.waitForFunction(()=>JSON.stringify(by_id("chart_settings_preview")._fullLayout.xaxis.range)==="[1,3]");
    await click_settings_button("reset_chart_ranges");
    await page.waitForFunction(()=>["x_min","x_max","y_min","y_max"].every(name=>by_id(`chart_${name}`).value===""));
    await click_settings_button("cancel_chart_settings");
    dense_mode=true;await page.evaluate(()=>refresh_catalog());
    // An existing poll may have started before the fixture changed its preset.
    await page.waitForFunction(()=>app.runs.find(run=>run_identifier(run)==="fixture_00")?.preset==="dense");
    assert.ok(Number(await row("fixture_00").locator('[data-instra-column-key="preset"]').evaluate(node=>getComputedStyle(node).fontWeight))>=700);
    assert.deepEqual(errors,[]);
    console.log("PASS",type.name(),"Edit/Rename identity, panel width, Progress columns/ETA, date spacing and bold dense");
    results.push({browser:type.name(),items:Array.from({length:18},(_,index)=>index+1),page_errors:errors});
  } finally {await browser.close();}
}
(async()=>{for(const name of (process.env.INSTRA_TEST_BROWSERS || "chromium,firefox").split(","))await check({chromium,firefox}[name]);
  if(process.env.INSTRA_TEST_EVIDENCE)fs.writeFileSync(process.env.INSTRA_TEST_EVIDENCE,JSON.stringify(results,null,2)+"\n");
})().catch(error=>{console.error(error);process.exitCode=1;});
// ^^^ THOG
