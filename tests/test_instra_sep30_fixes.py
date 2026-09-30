# vvv THOG regression coverage for selected GPU caps, unavailable readbacks and batched deletion
import importlib.util
import io
import json
import os
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import instra_node_agent as agent
from tests import test_instra_runner_unittest as runner_test_support
gpu = runner_test_support.gpu

class SelectedPowerCapsTests(unittest.TestCase):
    setUp = runner_test_support.RunnerTests.setUp
    # Reuse the isolated Runner harness without duplicating its existing test suite.
    def test_unselected_and_unavailable_caps_do_not_block_selected_gpu(self):
        recipe={**self.recipe,"gpu_pool":[gpu(0)["gpu_id"]],
                "power_caps":{"thog_host.absent.gpu.GPU-missing":200,gpu(1)["gpu_id"]:200}}
        preview=self.service.preview(recipe)
        assert len(preview["gpu_pool"])==1
        assert preview["runs"][0]["requested_power_w"] is None

    def test_unticked_host_is_not_contacted_during_preflight(self):
        original=self.fake.list_hosts
        def hosts():
            state=original()
            state["master_id"]=state["local_id"]
            state["hosts"].append({"thog_host_id":"thog_host.offline","display_name":"offline","local":False,
                "execution_enabled":True,"last_discovered":{"gpus":[{"gpu_key":"GPU-away"}],
                "execution_profiles":[{"profile_key":"current"}]}})
            return state
        with patch.object(self.fake,"list_hosts",side_effect=hosts):
            preview=self.service.preview({**self.recipe,"gpu_pool":[gpu(0)["gpu_id"]]})
        assert preview["runs"][0]["host_id"]==self.fake.local_id


def power_helper():
    spec=importlib.util.spec_from_file_location("sep30_power_helper",Path("scripts/instra_power_control.py"))
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    return helper


def test_power_query_accepts_nvidia_unavailable_values():
    helper=power_helper()
    with patch.object(helper.subprocess,"run",return_value=SimpleNamespace(stdout="GPU-12345678, [N/A], 80, [N/A], [N/A]\n")):
        result=helper.query("GPU-12345678")
    assert result["current_w"] is None and result["default_w"]==80


def test_default_power_on_laptop_does_not_require_privileged_helper():
    with patch.object(agent,"_power_capability",side_effect=AssertionError("must not call sudo")):
        result=agent._set_gpu_power({"gpu_key":"GPU-laptop","uuid":"GPU-laptop","power_cap_w":None,"default_power_w":80},None)
    assert result["supported"] is False and result["changed"] is False


def test_default_helper_uses_driver_policy_when_readback_unavailable():
    helper=power_helper()
    before={"uuid":"GPU-12345678","current_w":None,"default_w":80,"minimum_w":None,"maximum_w":None}
    output=io.StringIO()
    with patch.object(helper,"query",return_value=before),patch.object(helper.sys,"argv",["power","GPU-12345678","default"]),redirect_stdout(output):
        helper.main()
    assert json.loads(output.getvalue())["supported"] is False


def test_explicit_laptop_cap_reports_driver_error_instead_of_conversion_error():
    helper=power_helper()
    before={"uuid":"GPU-12345678","current_w":None,"default_w":80,"minimum_w":None,"maximum_w":None}
    with patch.object(helper,"query",return_value=before),patch.object(helper.sys,"argv",["power","GPU-12345678","70"]),\
         patch.object(helper.subprocess,"run",return_value=SimpleNamespace(returncode=1,stderr="Changing power limit is not supported",stdout="")):
        try: helper.main()
        except RuntimeError as error: assert "not supported" in str(error)
        else: raise AssertionError("unsupported explicit cap must fail clearly")


def test_power_range_error_accepts_one_unavailable_bound():
    helper=power_helper()
    before={"uuid":"GPU-12345678","current_w":80,"default_w":80,"minimum_w":60,"maximum_w":None}
    with patch.object(helper,"query",return_value=before),patch.object(helper.sys,"argv",["power","GPU-12345678","50"]):
        try: helper.main()
        except ValueError as error: assert "60–unknown" in str(error)
        else: raise AssertionError("known minimum must be enforced")


def test_agent_advertises_optional_power_readback_for_upgrade():
    with patch.object(agent,"_power_capability",return_value={"ready":False}):
        assert agent._runner_operation({},"runner_capabilities",{})["optional_power_readback"] is True


def test_laptop_blank_cap_survives_reserve_launch_and_release(tmp_path):
    laptop={**gpu(0),"power_cap_w":None,"default_power_w":80}
    grid_id,attempt_id=uuid.uuid4().hex,uuid.uuid4().hex
    run={"run_id":uuid.uuid4().hex,"grid_tag":"G-00001","pairing_id":None,"profiler":"none",
         "parameters":{"--max-iters":2,"--warmup-iters":0},"dtype":"bfloat16",
         "attention_backend":"sdpa","gpu_uuid":laptop["uuid"]}
    with patch.object(agent,"STATE_DIR",tmp_path),patch.object(agent,"AGENT_STATE",tmp_path/"node.json"),\
         patch.object(agent,"_gpu_information",return_value=[laptop]),\
         patch.object(agent,"_known_thog_compute_pids",return_value=[]),\
         patch.object(agent,"_power_capability",side_effect=AssertionError("blank laptop cap must bypass privileged helper")),\
         patch.object(agent.subprocess,"Popen",return_value=SimpleNamespace(pid=os.getpid())) as spawn:
        agent._operation("runner_reserve",{"grid_id":grid_id,"gpu_key":laptop["gpu_key"],"required_mib":1024,"headroom_mib":512})
        accepted=agent._operation("runner_launch",{"grid_id":grid_id,"attempt_id":attempt_id,"gpu_key":laptop["gpu_key"],
            "run":run,"host_label":"scruffy","thog_host_id":"thog_host.test","execution_profile":"current"})
        assert accepted["state"]=="running" and accepted["observed_power_w"] is None
        spawn.assert_called_once()
        (tmp_path/f"attempt-{attempt_id}.exit").write_text("0\n")
        assert agent._operation("runner_release",{"grid_id":grid_id,"gpu_key":laptop["gpu_key"]})["released"] is True


def test_batch_deletion_populates_identity_cache_once(tmp_path):
    import run_thog2_local_dashboard as dashboard
    from sheet.local_dashboard_performance_patch import install
    install(dashboard)
    from sheet.local_chart_store import LocalChartStore
    for index in range(12):
        store=LocalChartStore(tmp_path/f"run_{index}"/"charts.sqlite3",run_name=f"run_{index}",run_id=f"id_{index}",config={})
        store.close()
    catalog=dashboard._base.DashboardCatalog(root=tmp_path)
    with patch.object(catalog,"_candidate_paths",wraps=catalog._candidate_paths) as scan:
        result=catalog.delete_runs([f"id_{index}" for index in range(12)])
    assert len(result["deleted_run_ids"])==12 and result["errors"]==[]
    assert scan.call_count==1
    assert not list(tmp_path.glob("**/charts.sqlite3"))
# ^^^ THOG

# vvv THOG HTTP guards must reject an active Grid before any chart database is removed
def test_grid_deletion_http_guard_and_terminal_batch(tmp_path):
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen
    import run_thog2_local_dashboard as dashboard
    from sheet.local_dashboard_performance_patch import install
    from sheet.local_chart_store import LocalChartStore
    install(dashboard)
    for index in range(3):
        store=LocalChartStore(tmp_path/f"run_{index}"/"charts.sqlite3",run_name=f"run_{index}",run_id=f"id_{index}",
                              config={"runner":{"grid_tag":"G-00001","run_id":f"id_{index}"}})
        store.close()
    catalog=dashboard._base.DashboardCatalog(root=tmp_path)
    grid={"grid_tag":"G-00001","state":"running","runs":[{"state":"running"}]}
    service=SimpleNamespace(snapshot=lambda:{"grids":[grid]})
    with patch.object(dashboard,"_runner_service",service):
        server=ThreadingHTTPServer(("127.0.0.1",0),dashboard._handler_for_with_runner(catalog))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        address=f"http://127.0.0.1:{server.server_address[1]}/api/runs"
        def request(payload):
            return urlopen(Request(address,data=json.dumps(payload).encode(),method="DELETE",
                                   headers={"Content-Type":"application/json"}),timeout=3)
        try:
            try: request({"grid_tag":"G-00001"})
            except HTTPError as error:
                assert error.code==400 and "Stop the Grid first" in error.read().decode()
            else: raise AssertionError("active Grid deletion was accepted")
            assert len(list(tmp_path.glob("**/charts.sqlite3")))==3
            grid.update(state="completed",runs=[{"state":"completed"}])
            with request({"grid_tag":"G-00001"}) as response: result=json.load(response)
            assert len(result["deleted_run_ids"])==3 and not result["errors"]
            assert not list(tmp_path.glob("**/charts.sqlite3"))
        finally:
            server.shutdown();server.server_close();thread.join(timeout=3)
# ^^^ THOG
