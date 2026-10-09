// vvv THOG exercise the complete settings, table, help and retained-render behavior in a real browser
"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),browser_type=require("playwright")[process.env.INSTRA_TEST_BROWSER || "chromium"];
const address=process.env.INSTRA_TEST_URL || "http://127.0.0.1:8765";
(async()=>{
  const browser=await browser_type.launch({headless:true,...(browser_type.name()==="chromium" && process.env.INSTRA_CHROMIUM_PATH ? {
    executablePath:process.env.INSTRA_CHROMIUM_PATH,args:["--no-sandbox","--no-zygote","--disable-dev-shm-usage"],
  } : {}),...(browser_type.name()==="firefox" ? {env:{...process.env,MOZ_DISABLE_CONTENT_SANDBOX:"1"}} : {})});
  try {
    const page=await browser.newPage({viewport:{width:1600,height:1000}}),errors=[],actions=[];
    page.setDefaultTimeout(20000);
    page.on("pageerror",error=>errors.push(error.message));
    let excluded=true;
    await page.route(/\/api\/runs$/,async route=>{
      const response=await route.fetch(),payload=await response.json();
      const run=payload.runs.find(run=>run.dashboard_run_id==="fixture_00");
      run.preset="width-type-I";run.configuration={...run.configuration,geometry_preset:"width-type-I",n_embd:1024,width_order:512,width_enabled:true};
      return route.fulfill({json:payload});
    });
    await page.route(/\/api\/deletions$/,route=>route.fulfill({json:{deletion_confirmation_timeout_days:7,
      pending:[{run_id:"pending_run",outstanding_hosts:["dreedle"]}],
      excluded_local_copies:excluded ? [{run_id:"excluded_run",producing_host:"dreedle",host_id:"host",source_chart_path:"run/charts.sqlite3"}] : [],
    }}));
    await page.route(/\/api\/deletion\/action$/,route=>{
      const payload=route.request().postDataJSON();actions.push(payload);
      if(payload.action==="resume_local_copy")excluded=false;
      return route.fulfill({json:{saved:true,resumed:true}});
    });
    await page.goto(address+"/runs/fixture_00");
    await page.waitForFunction(()=>window.instra_demand_runtime?.table_render_statistics && app.runs.length===12);
    const row=page.locator('#runs_body tr[data-run-id="fixture_00"]');
    const columns=await page.locator('.runs-table thead tr').evaluate(node=>[...node.children].map(cell=>cell.dataset.instraColumnKey));
    assert.deepEqual(columns.slice(columns.indexOf("layers"),columns.indexOf("layers")+4),["layers","depth_order","d_model","width_order"]);
    assert.equal(await row.locator('[data-instra-column-key="preset"]').textContent(),"width");
    assert.equal(await row.locator('[data-instra-column-key="width_order"]').textContent(),"512");
    assert.equal(await row.locator('[data-instra-column-key="d_model"]').textContent(),"1024");
    const pair=await page.locator('.runs-table thead [data-instra-column-key="width_order"]').evaluate(node=>({text:node.textContent.trim(),transform:getComputedStyle(node).textTransform,color:getComputedStyle(node).color}));
    assert.equal(pair.text,"r");assert.equal(pair.transform,"none");assert.equal(pair.color,"rgb(31, 74, 122)");
    const retention=await page.evaluate(()=>{
      render_runs(); // Prime the complete pipeline's retained signature once after installation.
      const row=by_id("runs_body").firstElementChild,statistics=instra_demand_runtime.table_render_statistics;
      const before={...statistics};for(let i=0;i<100;i++)render_runs();
      return {same:row===by_id("runs_body").firstElementChild,rendered:statistics.rendered-before.rendered,skipped:statistics.skipped-before.skipped};
    });
    assert.deepEqual(retention,{same:true,rendered:0,skipped:100});
    await page.evaluate(()=>{app.runs.find(run=>run_identifier(run)==="fixture_00").configuration.width_order=1024;render_runs();});
    assert.equal(await row.locator('[data-instra-column-key="width_order"]').textContent(),"1024");
    await page.evaluate(()=>{app.runs.find(run=>run_identifier(run)==="fixture_00").preset="dense";render_runs();});
    assert.equal(await row.locator('[data-instra-column-key="preset"]').evaluate(node=>node.classList.contains("instra-dense-preset")),true);
    await page.evaluate(()=>{app.runs.find(run=>run_identifier(run)==="fixture_00").preset="width-type-I";render_runs();});
    assert.equal(await row.locator('[data-instra-column-key="preset"]').evaluate(node=>node.classList.contains("instra-dense-preset")),false);
    console.log("PASS width label, adjacent blue D/r, retained table, and data invalidation");

    await page.locator("#settings_nav").click();
    assert.equal(await page.locator("#deletion_pending_details").getAttribute("open"),null);
    assert.equal(await page.locator("#deletion_pending_status").isVisible(),false);
    assert.equal(await page.locator("#initial_run_colour").inputValue(),"light");
    assert.equal(await page.locator("#grid_eye_grouping").count(),0);
    assert.equal(await page.locator("#heatmap_settings_section").getAttribute("open"),null);
    await page.locator("#heatmap_settings_section > summary").click();
    assert.equal(await page.locator("#heatmap_setting_abs_limit").isVisible(),true);
    await page.locator("#heatmap_settings_section > summary").click();
    await page.locator("#deletion_pending_details summary").click();
    await page.waitForFunction(()=>by_id("deletion_pending_status").textContent.includes("pending_run: waiting on dreedle"));
    await page.locator("#excluded_local_copies_details summary").click();
    await page.locator("#excluded_local_copies_status button").click();
    await page.waitForFunction(()=>by_id("excluded_local_copies_status").textContent==="No local copies are excluded.");
    assert.deepEqual(actions[0],{action:"resume_local_copy",host_id:"host",source_chart_path:"run/charts.sqlite3"});
    for(const [band,minimum,maximum] of [["very_light",81,91],["light",65,79],["medium",46,62],["dark",27,39]]) {
      await page.locator("#initial_run_colour").selectOption(band);await page.locator("#save_settings").click();
      await page.waitForFunction(()=>by_id("settings_overlay").hidden);
      const shades=await page.evaluate(()=>app.runs.map(run=>Number(colour_for_run(run_identifier(run)).match(/([\d.]+)%\)$/)?.[1])));
      assert.ok(shades.every(shade=>shade>=minimum && shade<=maximum),JSON.stringify({band,shades}));
      await page.locator("#settings_nav").click();assert.equal(await page.locator("#initial_run_colour").inputValue(),band);
    }
    await page.locator("#initial_run_colour").selectOption("light");await page.locator("#save_settings").click();
    await page.waitForFunction(()=>by_id("settings_overlay").hidden);
    await page.reload();await page.waitForFunction(()=>window.instra_demand_runtime?.table_render_statistics);
    assert.equal(await page.evaluate(()=>app.initial_run_colour),"light");
    await page.locator("#settings_nav").click();
    await page.setViewportSize({width:480,height:800});
    assert.equal(await page.locator(".global-settings-window .settings-content").evaluate(node=>getComputedStyle(node).gridTemplateColumns.split(" ").length),1);
    assert.ok(await page.locator("#save_settings").isVisible());
    await page.setViewportSize({width:1600,height:1000});
    await page.screenshot({path:`/tmp/instra-oct08-settings-${browser_type.name()}.png`});
    await page.locator("#cancel_settings").click();
    console.log("PASS settings layout, collapsed pending data, explicit resume and all four colour bands");

    await page.locator("#runner_nav").click();
    await page.waitForFunction(()=>window.instra_runner_test_hooks);
    const snapshot=await (await page.request.get(address+"/api/runner")).json();
    const help=await page.evaluate(catalogue=>Object.fromEntries(Object.entries(catalogue).map(([name,spec])=>[name,instra_runner_test_hooks.field_help(spec)])),snapshot.catalogue);
    for(const name of ["--basis-family","--basis-version","DEPTH.compressor","DEPTH.compressor_version","WIDTH.compressor","WIDTH.compressor_version","--mlp-hidden-compressor","--hyperblock-compressor","--hyperblock-compressor-version"])
      assert.match(help[name],/equivalent to DCT-II up to ordering and signs/);
    for(const name of ["DEPTH.compressor_version","WIDTH.compressor_version","--basis-version"])
      assert.match(help[name],/chebyshev_first_kind_roots_v1.*dct_ii_orthonormal_v1.*haar_balanced_binary_orthonormal_v1/);
    assert.match(help["--attention-geometry"],/head_aware_block/);
    assert.match(help["--reset-optimizer"],/true, false/);
    assert.equal(await page.locator("#runner_rename_form button[type=submit]").textContent(),"Save to New Name");
    await page.locator('[data-runner-tab="progress"]').click();
    await page.waitForSelector(".runner-state-legend");
    const states=await page.locator(".runner-state-legend span").evaluateAll(nodes=>nodes.map(node=>({state:node.textContent,color:getComputedStyle(node).color})));
    assert.ok(states.slice(0,4).every(item=>item.color==="rgb(22, 131, 59)"));
    assert.ok(states.slice(6).every(item=>item.color==="rgb(179, 38, 46)"));
    assert.deepEqual(errors,[]);
    console.log("PASS registered hover options, Chebyshev/DCT equivalence, Save to New Name and state colours");
    await page.unrouteAll({behavior:"wait"});
    if(process.env.INSTRA_TEST_EVIDENCE)fs.writeFileSync(process.env.INSTRA_TEST_EVIDENCE,JSON.stringify({browser:browser_type.name(),retention,columns,states,page_errors:errors},null,2)+"\n");
  } finally {await browser.close();}
})().catch(error=>{console.error(error.stack);process.exit(1);});
// ^^^ THOG
