# vvv THOG exercise agent isolation, persisted network state, asynchronous failure and exclusivity without GPU dependencies
"""Run with python -m unittest tests/test_instra_network_unittest.py -v."""

import json
import ast
import io
import os
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import instra_network as network
import instra_node_agent as agent


class NetworkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.state_dir = Path(self.temporary.name)
        self.agent_patches = [
            patch.object(agent, "STATE_DIR", self.state_dir), patch.object(agent, "SOCKET_PATH", self.state_dir / "node.sock"),
            patch.object(agent, "AGENT_STATE", self.state_dir / "node.json"),
        ]
        self.network_patches = [
            patch.object(network, "STATE_DIR", self.state_dir), patch.object(network, "CONFIG_PATH", self.state_dir / "network.json"),
            patch.object(network, "LOCK_PATH", self.state_dir / "network.lock"), patch.object(network, "LOG_PATH", self.state_dir / "events.jsonl"),
            patch.object(network, "KNOWN_HOSTS", self.state_dir / "known_hosts"),
        ]
        for item in self.agent_patches + self.network_patches:
            item.start()
            self.addCleanup(item.stop)
        self.service = network.NetworkService(start_worker=False)
        self.addCleanup(self.service.close)

    def _await(self, job_id):
        for _ in range(100):
            outcome = self.service.job(job_id)
            if outcome["status"] != "working":
                return outcome
            time.sleep(0.02)
        self.fail("network operation did not finish")

    def test_local_discovery_and_stale_update(self):
        discovered = {"hostname": "scruffy", "instra_logs_root": "/logs", "wandb_root": "/wandb", "gpus": [],
                      "execution_profiles": [{"profile_key": "current", "python": "/python"}]}
        with patch.object(self.service, "_agent_request", return_value=discovered):
            self.assertEqual(self._await(self.service.submit("discover", self.service.local_id)["job_id"])["status"], "done")
        first = self.service._host(self.service.local_id)["last_discovered"]
        self.assertEqual(first["execution_profiles"][0]["execution_profile_id"],
                         f"{self.service.local_id}.execution_profile.current")
        with patch.object(self.service, "_agent_request", side_effect=network.NetworkError("agent availability", "agent unavailable")):
            failure = self._await(self.service.submit("discover", self.service.local_id)["job_id"])
        self.assertEqual(failure["category"], "agent availability")
        host = self.service._host(self.service.local_id)
        self.assertEqual(host["state"], "unavailable")
        self.assertEqual(host["last_discovered"], first)

    def test_add_requires_discovery_and_does_not_persist_password(self):
        record = {"instra_logs_root": "/tmp/logs", "wandb_root": "/tmp/wandb", "gpus": []}
        with patch.object(self.service, "_agent_request", return_value=record), patch.object(network, "_direct_route", return_value="192.168.1.2"):
            result = self._await(self.service.submit("add", address="192.168.1.2", ssh_port=22, ssh_user="peter", password="secret-value")["job_id"])
        self.assertEqual(result["status"], "done")
        host_id = result["result"]["thog_host_id"]
        self.assertEqual(host_id, "thog_host.ip_192_168_1_2")
        self.assertEqual(self.service._host(host_id)["state"], "disabled")
        self.assertNotIn("secret-value", network.CONFIG_PATH.read_text() + network.LOG_PATH.read_text())
        second = self._await(self.service.submit("add", address="192.168.1.2", ssh_port=22)["job_id"])
        self.assertEqual(second["category"], "validation")

    def test_disabling_local_host_stops_scheduled_discovery(self):
        disabled = self.service.update_host(self.service.local_id, monitoring_enabled=False, execution_enabled=False)
        self.assertEqual(disabled["state"], "disabled")
        self.assertFalse(disabled["monitoring_enabled"] or disabled["execution_enabled"])
        class OnePass:
            checks = 0
            def is_set(self):
                self.checks += 1
                return self.checks > 1
            def wait(self, _interval):
                pass
            def set(self):
                self.checks = 2
        self.service.stop_event = OnePass()
        with patch.object(self.service, "submit") as submit:
            self.service._retry_loop()
            submit.assert_not_called()

    def test_disabled_host_keeps_disabled_state_after_manual_refresh_and_reenables(self):
        self.service.update_host(self.service.local_id, monitoring_enabled=False, execution_enabled=False)
        discovered = {"instra_logs_root":"/logs", "wandb_root":"/wandb", "execution_profiles":[]}
        with patch.object(self.service, "_agent_request", return_value=discovered):
            outcome = self._await(self.service.submit("discover", self.service.local_id)["job_id"])
        self.assertEqual(outcome["status"], "done")
        self.assertEqual(self.service._host(self.service.local_id)["state"], "disabled")
        with patch.object(self.service, "_agent_request", return_value=discovered):
            enabled = self.service.update_host(self.service.local_id, monitoring_enabled=True)
            self.assertIn(enabled["state"], {"discovering", "available"})
            for _ in range(100):
                if self.service._host(self.service.local_id)["state"] == "available":
                    break
                time.sleep(0.02)
        self.assertEqual(self.service._host(self.service.local_id)["state"], "available")

    def test_master_release_pending_and_restart_settings(self):
        with patch.object(self.service, "_agent_request", return_value={"master_id":self.service.local_id}):
            self.service.designate(self.service.local_id)
            self.assertEqual(self.service.list_hosts()["master_id"], self.service.local_id)
            self.service.set_grid_activity(True, True)
            self.assertTrue(self.service.release()["release_pending"])
            self.assertEqual(self.service.list_hosts()["master_id"], self.service.local_id)
            self.service.set_grid_activity(False, False)
        self.assertIsNone(self.service.list_hosts()["master_id"])
        self.service.settings(restart_mode="remote", retry_interval=4)
        self.assertEqual(self.service.list_hosts()["retry_interval"], 4)
        self.assertEqual(network._read_config()["restart_mode"], "remote")

    def test_independent_jobs_and_agent_failure_category(self):
        def fake_request(host, operation, args=None, password=None, accepted_fingerprint=None):
            if operation == "discover":
                raise network.NetworkError("SSH transport", "Host offline")
            return {}
        with patch.object(self.service, "_agent_request", side_effect=fake_request):
            first = self.service.submit("discover", self.service.local_id)
            second = self.service.submit("settings", retry_interval=3)
            self.assertEqual(self._await(second["job_id"])["status"], "done")
            self.assertEqual(self._await(first["job_id"])["category"], "SSH transport")

    def test_monitoring_can_acquire_from_last_discovery_while_agent_unavailable(self):
        logs = self.state_dir / "logs"
        (logs / "run-1").mkdir(parents=True)
        (logs / "run-1" / "charts.sqlite3").write_bytes(b"example run data")
        self.service._record(self.service.local_id, state="unavailable", last_discovered={
            "instra_logs_root": str(logs), "wandb_root": str(self.state_dir / "wandb")})
        target = self.state_dir / "cache" / "charts.sqlite3"
        self.assertEqual(self.service.acquire_file(self.service.local_id, "logs", "run-1/charts.sqlite3", target), target)
        self.assertEqual(target.read_bytes(), b"example run data")
        with self.assertRaises(network.NetworkError):
            self.service.acquire_file(self.service.local_id, "logs", "../node.json", target)

    def test_node_agent_uses_named_operations_and_independent_socket(self):
        response = agent._operation("discover", {})
        self.assertEqual(response["thog"]["root"], str(agent.ROOT))
        self.assertIn("instra_logs_root", response)
        self.assertIn("execution_profiles", response)
        with self.assertRaisesRegex(ValueError, "unknown operation"):
            agent._operation("shell", {"command": "touch /tmp/forbidden"})
        # Some managed CI sandboxes prohibit Unix sockets; direct operation checks still run.
        try:
            import socket
            probe = socket.socket(socket.AF_UNIX)
            probe.close()
        except PermissionError:
            return
        environment = {**os.environ, "INSTRA_STATE_DIR": str(self.state_dir)}
        process = subprocess.Popen([sys.executable, str(Path(agent.__file__)), "serve"], cwd=agent.ROOT,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=environment)
        self.addCleanup(lambda: (process.terminate(), process.wait(timeout=5)) if process.poll() is None else None)
        for _ in range(100):
            if agent.SOCKET_PATH.exists():
                break
            time.sleep(0.02)
        self.assertTrue(agent.SOCKET_PATH.exists())
        response = agent.request("discover")
        with self.assertRaisesRegex(RuntimeError, "unknown operation"):
            agent.request("shell", {"command": "touch /tmp/forbidden"})
        self.assertFalse(agent._read_state().get("runs"))

    def test_remote_agent_absence_has_distinct_failure_category(self):
        host = self.service._new_host("dreedle", "peter", 22)
        response = SimpleNamespace(stdout=json.dumps({"ok": False, "category": "agent availability",
                                                       "error": "local socket unavailable"}), stderr="", returncode=1)
        with patch.object(network, "_verify_identity"), patch.object(network.subprocess, "run", return_value=response):
            with self.assertRaises(network.NetworkError) as failure:
                network._ssh_request(host, "discover", {})
        self.assertEqual(failure.exception.category, "agent availability")

    def test_manual_discovery_accepts_one_attempt_password_without_saving_it(self):
        host = self.service._new_host("dreedle", "peter", 22)
        host_id = host["thog_host_id"]
        host["monitoring_enabled"] = True
        with network._locked_config() as config:
            config["hosts"][host_id] = host
        discovered = {"instra_logs_root":"/logs", "wandb_root":"/wandb", "execution_profiles":[]}
        def remote_request(_host, operation, args=None, password=None, accepted_fingerprint=None):
            if password != "one-use-secret":
                raise network.NetworkError("authentication", "SSH authentication failed")
            return discovered
        with (patch.object(self.service, "_agent_request", side_effect=remote_request),
              patch.object(network, "_direct_route", return_value="192.168.1.3")):
            failure = self._await(self.service.submit("discover", host_id)["job_id"])
            self.assertEqual(failure["category"], "authentication")
            success = self._await(self.service.submit("discover", host_id, password="one-use-secret")["job_id"])
        self.assertEqual(success["status"], "done")
        self.assertEqual(self.service._host(host_id)["state"], "available")
        self.assertNotIn("one-use-secret", network.CONFIG_PATH.read_text() + network.LOG_PATH.read_text())

    def test_node_cli_distinguishes_socket_failure_from_operation_failure(self):
        for failure, category in ((OSError("socket unavailable"), "agent availability"),
                                  (ValueError("invalid operation"), "operation")):
            output = io.StringIO()
            with (patch.object(agent.sys, "stdin", io.StringIO('{"operation":"state","args":{}}')),
                  patch.object(agent, "request", side_effect=failure), redirect_stdout(output)):
                self.assertEqual(agent.main(["request"]), 1)
            self.assertEqual(json.loads(output.getvalue())["category"], category)

    def test_backend_restart_marks_only_intentional_exits(self):
        source = ast.parse((agent.ROOT / "run_thog2_dashboard.py").read_text())
        runner = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "_run_backend_with_agent")
        installed = {}
        cleaned = []
        class Assets:
            def cleanup(self):
                cleaned.append(True)
        backend = SimpleNamespace(main=lambda: 0)
        namespace = {"_set_process_name": lambda: None, "_start_node_agent": lambda: None,
                     "signal": SimpleNamespace(SIGTERM=15, signal=lambda _kind, callback: installed.update(callback=callback)),
                     "_prepare_runtime_assets": Assets, "_dashboard": backend, "os": os}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[runner], type_ignores=[])),
                     "run_thog2_dashboard.py", "exec"), namespace)
        launch = namespace["_run_backend_with_agent"]
        with patch.object(agent, "request") as request:
            self.assertEqual(launch(), 0)
            self.assertEqual(request.call_count, 1)
            request.reset_mock()
            def crash():
                raise RuntimeError("backend crashed")
            backend.main = crash
            with self.assertRaisesRegex(RuntimeError, "backend crashed"):
                launch()
            request.assert_not_called()
            backend.main = lambda: installed["callback"](15, None)
            with self.assertRaises(SystemExit):
                launch()
            self.assertEqual(request.call_count, 1)
        self.assertEqual(len(cleaned), 3)

    def test_network_post_wins_over_existing_post_handler(self):
        source = ast.parse((agent.ROOT / "run_thog2_dashboard.py").read_text())
        wrapper = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "_handler_for_with_network_post")
        calls = []
        class PreviousHandler:
            def do_POST(self):
                calls.append("existing handler")
        dashboard = SimpleNamespace(_network_do_post=lambda handler: calls.append("Network action"))
        namespace = {"_dashboard": dashboard, "_handler_for_before_network_post": lambda catalog: PreviousHandler}
        exec(compile(ast.Module(body=[wrapper], type_ignores=[]), "run_thog2_dashboard.py", "exec"), namespace)
        handler = namespace["_handler_for_with_network_post"](None)()
        handler.path = "/api/network/action"
        handler.do_POST()
        handler.path = "/api/weight-selection"
        handler.do_POST()
        self.assertEqual(calls, ["Network action", "existing handler"])

    def test_network_post_returns_json_job(self):
        source = ast.parse((agent.ROOT / "run_thog2_local_dashboard.py").read_text())
        operation = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "_network_do_post")
        service = SimpleNamespace(submit=lambda action, host_id, **args: {"job_id": "test-job"})
        namespace = {"_network_service": service, "_base": SimpleNamespace(HTTPStatus=SimpleNamespace(BAD_REQUEST=400, SERVICE_UNAVAILABLE=503)), "json": json}
        exec(compile(ast.Module(body=[operation], type_ignores=[]), "run_thog2_local_dashboard.py", "exec"), namespace)
        class Handler:
            headers = {"Content-Length": ""}
            def _send_json(self, result, **kwargs):
                self.result = result
        handler = Handler()
        body = json.dumps({"action": "prepare_host", "args": {"address": "dreedle"}}).encode()
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile = io.BytesIO(body)
        namespace["_network_do_post"](handler)
        self.assertEqual(handler.result, {"job_id": "test-job"})


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
