// vvv THOG
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {JSDOM} = require("jsdom");
const file = path.join(__dirname,"../sheet/local_dashboard_assets/dashboard_thogopt.js");
const helpers = require(file);
assert.equal(helpers.formatted(1/3,4),"0.3333");
assert.equal(helpers.formatted(null,4),"—");
assert.deepEqual(helpers.aligned_difference([[3,-2]],[[1,-1]]),[[2,-1]]);
assert.throws(()=>helpers.aligned_difference([[1]],[[1,2]]),/shapes/);
const dom = new JSDOM('<html><body><div id="charts_scroll"></div></body></html>',{runScripts:"outside-only",url:"http://localhost"});
const w=dom.window;
w.setInterval=()=>0;
w.app={current_run_id:'first',dynamic_chart_figures:{},dynamic_chart_metadata:{}};
w.chart_titles={};w.by_id=id=>w.document.getElementById(id);
w.run_identifier=run=>run.dashboard_run_id;
w.render_plot=async()=>{};w.clear_plot=()=>{};
w.depth_card=key=>{
 const card=w.document.createElement('article');card.className='chart-card';card.dataset.chart=key;
 card.innerHTML=`<header class="chart-card-header"><h2></h2><div class="chart-card-actions"></div></header><p id="${key}_detail"></p><div id="${key}_placeholder"></div><div id="${key}_plot"></div>`;return card;
};
w.HTMLDialogElement.prototype.showModal=function(){this.open=true;};
w.HTMLDialogElement.prototype.close=function(){this.open=false;};
const family='attn_q_head_N';const urls=[];
function payload(){return {steps:[1,2],full_steps:[2],figures:{[family]:{data:[{x:[1,2],y:[.123456,2],name:'r0 c0 · step 2'}],layout:{}}},errors:{},snapshots:[{optimizer_update:2,families:{[family]:{coordinates:[[0,0]],values:{momentum:[[.123456,2]],second_moment:[[.4,.5]],rms:[[.2,.3]]},reference:{origin:0,values:{momentum:[[.1,1]],second_moment:[[.2,.3]],rms:[[.1,.2]]}}}}}]};}
w.fetch=async url=>{urls.push(url); return {ok:true,json:async()=>url.includes('-matrix?')?{values:[[.123456]],shape:[1000,1000],step:2,layer:1,kind:'thogopt',coefficient:null}:payload()};};
w.eval(fs.readFileSync(file,'utf8'));
const settle=()=>new Promise(resolve=>setTimeout(resolve,15));
(async()=>{
 await settle();
 assert.equal(w.document.querySelectorAll('.thogopt-group').length,2);
 assert.equal(w.document.querySelectorAll('.thogopt-card').length,12);
 assert.ok(urls.every(url=>url.includes('latest=true')));
 const quantity=w.document.querySelector('[aria-label="History quantity"]');quantity.value='raw_momentum';
 w.app.current_run_id='second';w.__instra_thogopt.refresh();await settle();
 assert.equal(quantity.value,'raw_momentum','run switch lost quantity');
 quantity.value='momentum';quantity.dispatchEvent(new w.Event('change'));await settle();
 w.__instra_thogopt.open_inspector('momentum',family);await settle();
 let dialog=w.document.getElementById('thogopt_inspector');
 assert.ok(dialog.textContent.includes('0.1235'),'default precision incorrect');
 const mode=dialog.querySelector('[aria-label="Inspection view"]');mode.value='difference';mode.dispatchEvent(new w.Event('change'));await settle();
 assert.ok(dialog.textContent.includes('0.0235'),'difference sign incorrect');
 mode.value='matrix';mode.dispatchEvent(new w.Event('change'));await settle();
 assert.ok(dialog.textContent.includes('1000 × 1000'));
 assert.ok(dialog.querySelectorAll('div').length<100,'matrix DOM grows with entire matrix');
 assert.ok(urls.at(-1).includes('row_count=40'));
 const card=w.document.querySelector('.thogopt-card');card.classList.add('maximized');
 card.querySelector('.thogopt-inspect').click();await settle();
 assert.ok(w.document.getElementById('thogopt_inspector').open,'maximized inspector unavailable');
 const precision=w.document.querySelector('#thogopt_inspector [aria-label="Decimal places"]');precision.value='6';precision.dispatchEvent(new w.Event('change'));await settle();
 assert.ok(w.document.getElementById('thogopt_inspector').textContent.includes('0.123456'));
 console.log('PASS thogopt history groups, run switching, latest requests, signed differences, precision, bounded matrix windows, standard/maximized inspector actions');
 dom.window.close();
})().catch(error=>{console.error(error);dom.window.close();process.exitCode=1;});
// ^^^ THOG
