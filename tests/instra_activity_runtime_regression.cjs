// vvv THOG actual status API, database writes and frontend polling recover existing loss-only jobs
"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs"),path=require("node:path"),vm=require("node:vm");
const {execFileSync:exec_file_sync}=require("node:child_process");
const address=process.env.INSTRA_TEST_URL || "http://127.0.0.1:8765";
const write_script=`
import json, sqlite3, sys, time
database, payload = sys.argv[1], json.loads(sys.argv[2])
connection = sqlite3.connect(database)
configuration = json.loads(connection.execute("SELECT value FROM metadata WHERE key='config_json'").fetchone()[0])
configuration['geometry_preset'] = 'width'
configuration['premat'] = 'disabled'
connection.execute("INSERT OR REPLACE INTO training_losses VALUES (?, ?, ?)", (payload['step'], payload['loss'], time.time()))
metadata = {'run_state': payload['state'], 'current_update': str(payload['step']),
            'heartbeat_at': payload['old_time'], 'data_updated_at': payload['data_time'],
            'updated_at': payload['data_time'], 'created_at': payload['old_time'],
            'config_json': json.dumps(configuration)}
connection.executemany("INSERT OR REPLACE INTO metadata VALUES (?, ?)", metadata.items())
connection.commit()
connection.close()
`;

async function check() {
  const catalog=await (await fetch(address+"/api/runs")).json();
  const fixture=catalog.runs.find(run=>run.dashboard_run_id==="fixture_00");
  assert.ok(fixture,"the isolated fixture run is missing");
  assert.equal(path.basename(fixture.run_directory),"fixture_00");
  const database=path.join(fixture.run_directory,"charts.sqlite3"),old_time=new Date(Date.now()-600000).toISOString();
  function write_activity(step,loss,data_time,state="recording") {
    exec_file_sync("python3",["-c",write_script,database,JSON.stringify({step,loss,data_time,state,old_time})]);
  }
  write_activity(5,5.5,old_time);
  let clock_offset=0,request_count=0,rendered_state=null,rendered_step=null;
  const elements=new Map();
  const context=vm.createContext({
    app:{runs:[],current_run_id:"fixture_00",timeout_minutes:1},
    document:{visibilityState:"visible"},window:{location:{search:""}},
    Date:class extends Date {static now(){return Date.now()+clock_offset;}},
    URLSearchParams,AbortController,setTimeout,clearTimeout,console,
    async fetch_json(url,options){request_count++;const response=await fetch(address+url,options);assert.equal(response.status,200);return response.json();},
    by_id(id){if(!elements.has(id))elements.set(id,{});return elements.get(id);},
    should_follow_recommendation:()=>false,render_run_heading:()=>{},render_empty_state:()=>{},
    render_runs(){const run=context.app.runs.find(run=>run.dashboard_run_id==="fixture_00");
      rendered_state=context.display_run_state(run);rendered_step=run.maximum_update;},
  });
  const source=fs.readFileSync(path.join(__dirname,"../sheet/local_dashboard_assets/dashboard.js"),"utf8");
  for(const [start,end] of [["function run_identifier(","function format_integer("],
    ["function format_run_duration(","function format_last_loss("],
    ["async function refresh_catalog(","function clear_plot("]]) {
    const begin=source.indexOf(start),finish=source.indexOf(end,begin);
    assert.ok(begin>=0 && finish>begin,"frontend function boundaries changed");
    vm.runInContext(source.slice(begin,finish),context);
  }
  async function wait_for(state,step) {
    const deadline=Date.now()+15000;
    while(Date.now()<deadline) {
      if(rendered_state===state && rendered_step===step)return;
      await new Promise(resolve=>setTimeout(resolve,50));
    }
    assert.fail(`expected ${state} at ${step}, got ${rendered_state} at ${rendered_step}`);
  }
  await context.refresh_catalog();
  assert.equal(rendered_state,"timed_out");
  // Keep the production 2.5-second catalogue polling cadence. Each successful
  // fetch runs the real display function when the table would render its badge.
  const poll=setInterval(()=>{void context.refresh_catalog();},2500);
  try {
    let started=Date.now();
    write_activity(6,5.3,new Date().toISOString());
    await wait_for("recording",6);
    const first_recovery_ms=Date.now()-started;
    assert.notEqual(context.format_run_duration(context.app.runs.find(run=>run.dashboard_run_id==="fixture_00")),"—");
    const observed=await (await fetch(address+"/api/status?run=fixture_00")).json();
    assert.ok(Date.parse(observed.heartbeat_at)>Date.parse(old_time));

    // Advance only the viewer clock. Read-only polling must not manufacture
    // model activity or prevent the next genuine timeout.
    clock_offset=120000;
    await wait_for("timed_out",6);
    started=Date.now();
    const resumed_time=new Date(Date.now()+clock_offset).toISOString();
    write_activity(7,5.2,resumed_time);
    await wait_for("recording",7);
    const second_recovery_ms=Date.now()-started;

    write_activity(50,4.2,resumed_time,"finished");
    await wait_for("finished",50);
    clock_offset=3600000;
    await context.refresh_catalog();
    assert.equal(rendered_state,"finished");
    console.log("PASS HTTP + frontend polling",JSON.stringify({first_recovery_ms,second_recovery_ms,request_count,
      checks:["stale run times out","fresh loss recovers automatically","silence times out again",
        "resumed progress recovers again","terminal state remains finished"]}));
  } finally {clearInterval(poll);}
}
check().catch(error=>{console.error(error);process.exitCode=1;});
// ^^^ THOG
