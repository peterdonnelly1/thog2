# vvv THOG isolated dashboard fixture for browser regression tests; no agents or GPUs
import json
from pathlib import Path
import sys
from http.server import ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
from types import SimpleNamespace

import run_thog2_dashboard as launcher
import run_thog2_local_dashboard as dashboard
runtime_assets = launcher._prepare_runtime_assets()
for attribute in ("_handler_for", "_ASSET_ROOT", "_ASSET_NAMES"):
    setattr(dashboard._base, attribute, vars(dashboard)[attribute])
import instra_runner
from sheet.local_chart_store import LocalChartStore

fixture_root = Path(sys.argv[1])
fixture_root.mkdir(parents=True, exist_ok=True)
parameters = {"--geometry-preset":"depth", "--optimizer":"adamw", "--n-layer":2, "DEPTH.order":1,
              "--warmup-iters":0, "--block-size":32, "--n-embd":64, "--n-head":4,
              "--gradient-accumulation-steps":1, "--checkpoint-segment-size":1,
              "--learning-rate":0.0009, "--min-lr":0.00009, "--max-iters":50, "--batch-size":1,
              "--premat":"enabled"}
gpu = {"gpu_key":"GPU-test", "gpu_id":"thog_host.scruffy.gpu.GPU-test", "uuid":"GPU-test",
       "ordinal":0,"model":"Fixture GPU","memory_mib":16384,"free_mib":12000,"compute_cap":"8.9"}
recipe = {"label":"Fixture Recipe","parameters":parameters,"gpu_pool":[gpu["gpu_id"]],"profilers":["none"]}
runs=[]
for index in range(12):
    run_id=f"fixture_{index:02d}"
    tag=f"G-{index//4+1:05d}"
    store=LocalChartStore(fixture_root/run_id/"charts.sqlite3",run_name=f"{tag}_fixture_{index:02d}",run_id=run_id,
                         config={"host_label":"scruffy","model_type":"sheet","premat":"enabled",
                                 "runner":{"grid_tag":tag,"run_id":run_id},"max_iters":50})
    for step in range(1,6):
        store.append_training_loss(step,5-step/10-index/100)
        store.append_processing_throughput(optimizer_update=step,tokens_per_second=10000+index*100+step)
    store.close()
    if index == 0:
        (fixture_root/run_id/"train.log").write_text("\x1b[32mCLI oldest\x1b[0m\nCLI newest\n")
    runs.append({"run_id":run_id,"grid_id":f"grid_{index//4}","state":"completed","host_label":"scruffy",
                 "host_id":"thog_host.scruffy","gpu":gpu,"profiler":"none","parameters":parameters,
                 "execution_profile":"current","dtype":"bfloat16","attention_backend":"flash2",
                 "required_mib":1000,"released":True,"attempts":[]})
grids=[{"grid_id":f"grid_{number}","grid_tag":f"G-{number+1:05d}","label":f"Grid {number+1}",
        "state":"completed" if number==0 else "failed","recipe_id":"recipe_fixture","recipe":recipe,
        "runs":[{**run,"state":"completed" if number==0 or number==1 and index==0 else "failed"}
                for index,run in enumerate(runs[number*4:(number+1)*4])],"created_at":"2026-09-30T00:00:00+00:00",
        "grid_script":"fixture.sh"} for number in range(3)]
grids.append({**grids[0],"grid_id":"active_grid","grid_tag":"G-99999","label":"Active fixture",
              "state":"running","runs":[{**runs[0],"run_id":"active_fixture","state":"running"}]})
network={"local_id":"thog_host.scruffy","master_id":None,"release_pending":False,"retry_interval":60,
         "restart_mode":"automatic","hosts":[{"thog_host_id":"thog_host.scruffy","display_name":"scruffy",
             "local":True,"execution_enabled":True,"monitoring_enabled":True,"state":"available",
             "last_discovered":{"execution_profiles":[{"execution_profile_id":"current"}],"gpus":[gpu]}}]}
snapshot={"recipes":[{"recipe_id":"recipe_fixture","recipe":recipe}],"grids":grids,
          "catalogue":instra_runner.CATALOGUE,"common":instra_runner.COMMON,"controller":True,"reconciled":True,
          "local_id":network["local_id"],"master_id":None}
# vvv THOG exercise actual file responses and download headers from isolated exports
file_paths = {}
for grid in grids:
    folder=fixture_root/"grid-scripts"/grid["grid_tag"]
    folder.mkdir(parents=True,exist_ok=True)
    grid["started_at"]="2026-09-30T00:01:00+00:00"
    if grid["state"] in {"completed","failed","cancelled"}:
        grid["finished_at"]="2026-09-30T00:02:05+00:00"
    grid["estimated_duration"]={"interval_seconds":[60,120],"seconds":90,"confidence":"high"}
    if grid["grid_id"]=="grid_0": grid["conversion"]={"from":"tight","to":"loose"}
    for name,filename in {"script":grid["grid_tag"]+".sh","classic":grid["grid_tag"]+"_grid_bash_runner_script.sh",
        "manifest":"manifest.json","placement":"placement.json","status":"status.json",
        "conversion":"conversion.json","log":"events.jsonl"}.items():
        path=folder/filename
        path.write_text("#!/bin/bash\n# " + grid["grid_tag"] + " " + name + "\n" if name in {"script","classic"}
                        else json.dumps({"grid_tag":grid["grid_tag"],"file":name})+"\n")
        file_paths[(grid["grid_id"],name)]=path

def fixture_file(grid_id,name):
    return file_paths[(grid_id,name)]

dashboard._runner_service=SimpleNamespace(snapshot=lambda:snapshot,file=fixture_file)
# ^^^ THOG
dashboard._network_service=SimpleNamespace(list_hosts=lambda:network)
catalog=dashboard._base.DashboardCatalog(root=fixture_root)
base_handler=dashboard._base._handler_for(catalog)
class FixtureHandler(base_handler):
    def do_GET(self):
        parsed=urlparse(self.path)
        query=parse_qs(parsed.query)
        run_id=query.get("run",["fixture_00"])[0]
        if parsed.path=="/api/chart-groups":
            return self._send_json({"available":True,"groups":[{"name":"train","chart_count":1,"revision":1}]})
        if parsed.path=="/api/chart-group":
            return self._send_json({"available":True,"group":{"name":query.get("group",["train"])[0],"revision":1,
                "charts":[{"id":"train/loss","title":"loss","x_title":"step","default_x_axis_mode":"step",
                "available_x_axis_modes":["step"],"series":[{"name":"loss","x":[1,2,3,4,5],"y":[5,4.8,4.6,4.4,4.2]}]}]}})
        return super().do_GET()
server=ThreadingHTTPServer(("127.0.0.1",int(sys.argv[2])),FixtureHandler)
print(f"fixture ready at {server.server_address}",flush=True)
server.serve_forever()
# ^^^ THOG
