// vvv THOG exercise the complete runtime under live catalogue writes, wide comparisons and mature throughput histories
"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),{chromium,firefox}=require("playwright");
const address=process.env.INSTRA_TEST_URL || "http://127.0.0.1:8765",duration=Number(process.env.INSTRA_DEMAND_SOAK_SECONDS || 120);
async function check(type) {
  const browser=await type.launch({headless:true,timeout:30000,...(type===firefox && process.env.INSTRA_FIREFOX_TEST_NO_SANDBOX ?
    {env:{...process.env,MOZ_DISABLE_CONTENT_SANDBOX:"1"}} : {}),...(type===chromium && process.env.INSTRA_CHROMIUM_PATH ?
    {executablePath:process.env.INSTRA_CHROMIUM_PATH,args:["--no-sandbox","--disable-dev-shm-usage","--no-zygote","--use-gl=angle","--use-angle=swiftshader","--enable-unsafe-swiftshader"]}: {})});
  try {
    const page=await browser.newPage({viewport:{width:1600,height:1100}}),errors=[],counts={weights:[],figures:0,throughput:0,groups:{}};
    page.on("pageerror",error=>{errors.push(error.message);console.log("PAGE ERROR",error.stack);});
    page.on("console",message=>{if(message.text().startsWith("LONG TASK"))console.log(type.name(),message.text());});
    const initial=await (await page.request.get(address+"/api/runs")).json();
    const runs=Array.from({length:32},(_,index)=>({...initial.runs[index%initial.runs.length],dashboard_run_id:`fixture_${String(index).padStart(2,"0")}`,
      local_run_id:`fixture_${String(index).padStart(2,"0")}`,artifact_name:`G-00001_history_${index}`,run_name:`history_${index}`,
      run_state:"running",depth_snapshot_count:64,depth_minimum_update:500,depth_maximum_update:32000,heatmap_count:0}));
    let data_revision=1,loss_revision=1,fail_weight_once=false,fail_throughput_once=true;
    await page.addInitScript(()=>{
      localStorage.setItem("thog2_local_run_visibility",JSON.stringify(Object.fromEntries(Array.from({length:32},(_,i)=>[`fixture_${String(i).padStart(2,"0")}`,true]))));
      window.demand_stats={draws:{},max_pause_ms:0,last:performance.now(),long_tasks:[]};
      if(window.PerformanceObserver?.supportedEntryTypes.includes("longtask"))new PerformanceObserver(list=>{
        for(const task of list.getEntries())if(task.duration>500){demand_stats.long_tasks.push({start:task.startTime,duration:task.duration});console.log("LONG TASK",task.startTime,task.duration);}
      }).observe({entryTypes:["longtask"]});
      setInterval(()=>{const now=performance.now();demand_stats.max_pause_ms=Math.max(demand_stats.max_pause_ms,now-demand_stats.last);demand_stats.last=now;},100);
      let plotly;Object.defineProperty(window,"Plotly",{configurable:true,get:()=>plotly,set:value=>{
        plotly=value;for(const key of ["newPlot","react"]){const before=value[key];value[key]=function(mount,...args){const name=mount.id;demand_stats.draws[name]=(demand_stats.draws[name]||0)+1;return before.call(this,mount,...args);};}
      }});
    });
    const status=run=>({...run,revision:[0,0,64,32000,0,0,data_revision]});
    await page.route(/\/api\/runs$/,route=>route.fulfill({json:{...initial,runs:runs.map(status)}}));
    await page.route(/\/api\/status\?/,route=>route.fulfill({json:status(runs[0])}));
    await page.route(/\/api\/figures\?/,route=>{counts.figures++;return route.fulfill({json:{heatmap:null,depth:{}}});});
    await page.route(/\/api\/processing\?/,route=>route.fulfill({json:{available:false,trace_available:false,revision:null,data:null}}));
    await page.route(/\/api\/processing-status\?/,route=>route.fulfill({json:{trace_available:false,timing_available:false,revision:{}}}));
    await page.route(/\/api\/chart-groups\?/,route=>{
      const id=new URL(route.request().url()).searchParams.get("run");
      return route.fulfill({json:{available:true,groups:[{name:"train",chart_count:1,revision:id==="fixture_00" ? loss_revision : 1}]}});
    });
    await page.route(/\/api\/chart-group\?/,route=>{
      const id=new URL(route.request().url()).searchParams.get("run"),revision=id==="fixture_00" ? loss_revision : 1;
      counts.groups[id]=(counts.groups[id]||0)+1;
      const x=Array.from({length:1600},(_,index)=>index),y=x.map(index=>4+Math.sin(index/100)/10-index/100000+revision/10000);
      return route.fulfill({json:{available:true,group:{name:"train",revision,charts:[{id:"train/loss",title:"loss",x_title:"step",series:[{name:"loss",x,y}]}]}}});
    });
    const throughput=Array.from({length:40000},(_,index)=>({optimizer_update:index,tokens_per_second:index===17291 ? 999999 : 10000+100*Math.sin(index/20),
      wall_time:1.7e9+index,relative_wall_seconds:index,process_time_seconds:index}));
    await page.route(/\/api\/processing-throughput\?/,route=>{
      counts.throughput++;
      if(fail_throughput_once && new URL(route.request().url()).searchParams.get("run")==="fixture_07"){
        fail_throughput_once=false;return route.fulfill({status:503,json:{error:"temporary history read failure"}});
      }
      return route.fulfill({json:{throughput}});
    });
    await page.route(/\/api\/weight-figure\?/,route=>{
      const query=new URL(route.request().url()).searchParams,chart=query.get("chart");counts.weights.push(chart);
      if(fail_weight_once && query.get("run")==="fixture_07"){
        fail_weight_once=false;return route.fulfill({status:503,json:{error:"temporary weight read failure"}});
      }
      const figure={data:[{type:"scatter",mode:"lines",name:"step 32000",x:[1,2,3],y:[1,2,1]}],layout:{xaxis:{},yaxis:{}}};
      return route.fulfill({json:{depth:{[chart]:figure},weight_step_range:{minimum:32000,maximum:32000,snapshot_count:1}}});
    });
    await page.goto(address+"/runs/fixture_00");
    await page.waitForFunction(()=>window.instra_demand_runtime && app.runs.length===32);
    await page.locator("#workspace_nav").click();
    await page.waitForFunction(()=>document.querySelector('[data-metric-chart-id="train/loss"] .plot-mount')?.data?.length===32);
    await page.waitForFunction(()=>String(app.maximized_chart).startsWith("local_metric_"));
    await page.waitForTimeout(1000);
    const mount=await page.locator('[data-metric-chart-id="train/loss"] .plot-mount').elementHandle();
    await page.evaluate(async()=>{
      const mount=document.querySelector('[data-metric-chart-id="train/loss"] .plot-mount');mount.dataset.demandIdentity="original";
      await Plotly.relayout(mount,{"xaxis.range":[400,800]});
      demand_stats.max_pause_ms=0;demand_stats.last=performance.now();
    });
    const start_counts=JSON.parse(JSON.stringify(counts));
    let clock=setInterval(()=>{data_revision++;loss_revision++;},1500);
    const client=type===chromium ? await page.context().newCDPSession(page) : null;
    if(client)await client.send("HeapProfiler.collectGarbage");
    const initial_heap=client ? (await client.send("Runtime.getHeapUsage")).usedSize : null;
    await page.evaluate(()=>{demand_stats.max_pause_ms=0;demand_stats.last=performance.now();demand_stats.long_tasks=[];});
    if(client && process.env.INSTRA_TEST_PROFILE){await client.send("Profiler.enable");await client.send("Profiler.start");}
    const deadline=Date.now()+duration*1000;let health_requests=0;
    while(Date.now()<deadline) {
      const response=await page.request.get(address+"/api/health",{timeout:3000});assert.equal(response.status(),200);health_requests++;
      await page.waitForTimeout(1000);
      if(health_requests%20===0)console.log("DEMAND SOAK",type.name(),health_requests,"health checks");
    }
    clearInterval(clock);clock=null;
    const stats=await page.evaluate(()=>({...demand_stats,identity:document.querySelector('[data-metric-chart-id="train/loss"] .plot-mount').dataset.demandIdentity,
      range:document.querySelector('[data-metric-chart-id="train/loss"] .plot-mount')._fullLayout.xaxis.range,pending:instra_demand_runtime.pending_plots.size}));
    if(client && process.env.INSTRA_TEST_PROFILE){const profile=(await client.send("Profiler.stop")).profile;fs.writeFileSync("/tmp/instra-demand-cpu.json",JSON.stringify(profile));}
    console.log("DEMAND CLOCK",type.name(),JSON.stringify(stats));
    assert.equal(stats.identity,"original","live writes replaced the chart mount");assert.deepEqual(stats.range,[400,800],"live refresh discarded the user's zoom");
    assert.equal(counts.figures,0,"the monolithic weight endpoint was requested");
    assert.equal(counts.throughput,start_counts.throughput,"hidden throughput was fetched during loss comparison");
    assert.equal(counts.weights.length,start_counts.weights.length,"hidden weight histories were fetched");
    for(const [id,count] of Object.entries(counts.groups))if(id!=="fixture_00")assert.equal(count,start_counts.groups[id],`unchanged history ${id} was refetched`);
    assert.ok(counts.groups.fixture_00>start_counts.groups.fixture_00+10,"the live loss did not advance");
    assert.ok(stats.max_pause_ms<2000,`UI paused for ${stats.max_pause_ms}ms`);
    assert.equal(await page.evaluate(()=>document.querySelector('[data-metric-chart-id="train/loss"] .plot-mount').data.every(trace=>trace.type==="scatter")),true);
    let retained_heap=null;if(client){await client.send("HeapProfiler.collectGarbage");retained_heap=(await client.send("Runtime.getHeapUsage")).usedSize;assert.ok(retained_heap-initial_heap<64*1048576);}
    console.log("PASS",type.name(),JSON.stringify({seconds:duration,health_requests,initial_heap_mib:initial_heap/1048576,retained_heap_mib:retained_heap/1048576,...stats}));

    // Only the requested weight family may materialise, including a thirty-two-run Workspace.
    fail_weight_once=true;
    await page.evaluate(()=>{toggle_maximized_chart("mlp_up");instra_demand_runtime.schedule_flush();});
    await page.waitForFunction(()=>by_id("mlp_up_plot").data?.length===32);
    assert.equal(await page.evaluate(()=>by_id("mlp_up_plot").data.every(trace=>trace.type==="scatter")),true,"weights still allocated WebGL traces");
    assert.ok(counts.weights.slice(start_counts.weights.length).every(name=>name==="mlp_up"));
    assert.equal(fail_weight_once,false);
    console.log("PASS",type.name(),"only the selected weight chart was fetched and rendered, with recovery from a failed request");

    // Rendering is bounded while exports retain every original throughput point and its spike.
    await page.evaluate(()=>{by_id("training_throughput_card").hidden=false;toggle_maximized_chart("training_throughput");});
    await page.waitForFunction(()=>by_id("training_throughput_plot").data?.length===32,{},{timeout:60000});
    const series=await page.evaluate(()=>({plotted:by_id("training_throughput_plot").data.map(trace=>trace.x.length),
      exported:instra_oct03_controls.chart_export_payload(by_id("training_throughput_card")).series.map(trace=>trace.x.length),
      spike:by_id("training_throughput_plot").data.every(trace=>trace.y.includes(999999))}));
    assert.ok(series.plotted.every(count=>count<=1600));assert.ok(series.exported.every(count=>count===40000));assert.equal(series.spike,true);
    const stable_requests=counts.throughput;await page.waitForTimeout(5000);assert.equal(counts.throughput,stable_requests,"unchanged throughput was repeatedly downloaded");
    console.log("PASS",type.name(),"32 x 40,000 raw throughput points, bounded plots, exact exports and unchanged-history caching");
    await page.locator("#runner_nav").click();await page.waitForTimeout(1000);
    const paused_counts=JSON.stringify(counts);await page.waitForTimeout(3500);assert.equal(JSON.stringify(counts),paused_counts,"charts polled under Runner");
    await page.locator("#runs_nav").click();await page.evaluate(()=>select_run("fixture_01"));await page.waitForTimeout(1000);
    assert.deepEqual(errors,[]);
    await page.screenshot({path:`/tmp/instra-demand-${type.name()}.png`});
    console.log("PASS",type.name(),"Runner quiescence and subsequent run navigation without JavaScript errors");
  } finally {await browser.close();}
}
(async()=>{for(const name of (process.env.INSTRA_TEST_BROWSERS || "chromium,firefox").split(","))await check({chromium,firefox}[name]);})().catch(error=>{console.error(error.stack);process.exit(1);});
// ^^^ THOG
