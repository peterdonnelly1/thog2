// vvv THOG sustained live twelve-run rendering, hidden-card deferral, request bounds and browser heap regression
"use strict";
const assert=require("node:assert/strict"),{chromium}=require("playwright");
const url=process.env.INSTRA_TEST_URL||"http://127.0.0.1:8765";
const duration=Number(process.env.INSTRA_SOAK_SECONDS||180);
(async()=>{
  const browser=await chromium.launch({headless:true,...(process.env.INSTRA_CHROMIUM_PATH?
    {executablePath:process.env.INSTRA_CHROMIUM_PATH,args:["--no-sandbox","--disable-dev-shm-usage","--no-zygote","--use-gl=angle","--use-angle=swiftshader","--enable-unsafe-swiftshader"]}: {})});
  try {
    const page=await browser.newPage({viewport:{width:1600,height:1100}}),errors=[];
    page.on("pageerror",error=>{errors.push(error.message);console.log("PAGE ERROR",error.stack);});
    await page.addInitScript(()=>{
      localStorage.setItem("thog2_local_run_visibility",JSON.stringify(Object.fromEntries(Array.from({length:12},(_,i)=>[`fixture_${String(i).padStart(2,"0")}`,true]))));
      window.soak_stats={draws:{},resizes:0,max_pause_ms:0,previous:performance.now(),pauses:[]};
      soak_stats.long_tasks=[];
      new PerformanceObserver(list=>{for(const task of list.getEntries())if(task.duration>500)soak_stats.long_tasks.push({start:task.startTime,duration:task.duration});}).observe({entryTypes:["longtask"]});
      setInterval(()=>{const now=performance.now(),gap=now-soak_stats.previous;soak_stats.max_pause_ms=Math.max(soak_stats.max_pause_ms,gap);soak_stats.previous=now;if(gap>1000)soak_stats.pauses.push({time:Date.now(),gap,visibility:document.visibilityState});},100);
      let value;Object.defineProperty(window,"Plotly",{configurable:true,get:()=>value,set:next=>{
        value=next;
        for(const key of ["newPlot","react"]) {const base=next[key];next[key]=function(mount,...args){const name=mount?.closest?.('.chart-card')?.dataset.metricChartId||mount?.id;soak_stats.draws[name]=(soak_stats.draws[name]||0)+1;return base.call(this,mount,...args);};}
        const base=next.Plots.resize;next.Plots.resize=function(...args){soak_stats.resizes++;return base.apply(this,args);};
      }});
    });
    const begin=Date.now(),revision=()=>1+Math.floor((Date.now()-begin)/1500);
    await page.route(/\/api\/chart-groups\?/,route=>route.fulfill({json:{available:true,groups:[{name:"train",chart_count:2,revision:revision()}]}}));
    await page.route(/\/api\/chart-group\?/,route=>{
      const count=2048+Math.floor((Date.now()-begin)/1000)*8;
      const x=Array.from({length:count},(_,i)=>i),y=x.map(i=>4+Math.sin(i/100)/10-i/100000);
      return route.fulfill({json:{available:true,group:{name:"train",revision:revision(),charts:[
        {id:"train/loss",title:"loss",x_title:"step",series:[{name:"loss",x,y}]},
        {id:"train/accuracy",title:"accuracy",x_title:"step",series:[{name:"accuracy",x,y:x.map(i=>i/10000)}]},
      ]}}});
    });
    await page.goto(url+"/runs/fixture_00");await page.waitForFunction(()=>window.instra_oct03_controls && app.runs.length===12);
    await page.locator("#workspace_nav").click();
    await page.waitForFunction(()=>document.querySelector('[data-metric-chart-id="train/loss"] .plot-mount')?.data?.length===12);
    await page.waitForFunction(()=>app.maximized_chart?.startsWith("local_metric_"));
    await page.waitForTimeout(2000);
    await page.evaluate(()=>{soak_stats.previous=performance.now();soak_stats.max_pause_ms=0;soak_stats.pauses=[];});
    const host_pauses=[];let host_previous=Date.now();
    const host_clock=setInterval(()=>{const now=Date.now(),gap=now-host_previous;if(gap>1000)host_pauses.push({time:now,gap});host_previous=now;},100);
    const client=await page.context().newCDPSession(page);await client.send("HeapProfiler.collectGarbage");
    const initial=(await client.send("Runtime.getHeapUsage")).usedSize;
    const initial_hidden=await page.evaluate(()=>soak_stats.draws["train/accuracy"]||0);
    let peak=initial,health_requests=0;
    const deadline=Date.now()+duration*1000;
    while(Date.now()<deadline) {
      const response=await page.request.get(url+"/api/runs",{timeout:3000});assert.equal(response.status(),200);health_requests++;
      peak=Math.max(peak,(await client.send("Runtime.getHeapUsage")).usedSize);
      if(health_requests%10===0)console.log("SOAK",Math.round((Date.now()-(deadline-duration*1000))/1000),"seconds",Math.round(peak/1024/1024),"MiB peak");
      await page.waitForTimeout(1000);
    }
    const stats=await page.evaluate(()=>({...soak_stats,curves:document.querySelector('[data-metric-chart-id="train/loss"] .plot-mount').data.length}));
    await client.send("HeapProfiler.collectGarbage");const retained=(await client.send("Runtime.getHeapUsage")).usedSize;
    clearInterval(host_clock);
    console.log("SOAK CLOCKS",JSON.stringify({browser:stats.pauses,host:host_pauses,long_tasks:stats.long_tasks}));
    assert.equal(stats.curves,12);assert.equal(stats.draws["train/accuracy"]||0,initial_hidden,"hidden chart was repeatedly rendered");
    assert.ok(stats.draws["train/loss"]>30,JSON.stringify(stats));
    assert.ok(stats.max_pause_ms<5000,`main-thread pause ${stats.max_pause_ms}ms`);
    assert.ok(retained-initial<128*1024*1024,`retained heap increased ${(retained-initial)/1024/1024} MiB`);
    assert.ok(peak<512*1024*1024,`peak heap ${peak/1024/1024} MiB`);
    assert.deepEqual(errors,[]);
    console.log("PASS",JSON.stringify({seconds:duration,health_requests,...stats,initial_heap_mib:initial/1024/1024,retained_heap_mib:retained/1024/1024,peak_heap_mib:peak/1024/1024}));
  } finally {await browser.close();}
})().catch(error=>{console.error(error.stack);process.exit(1);});
// ^^^ THOG
