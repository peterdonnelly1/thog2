# vvv THOG exercise agent isolation, persisted network state, asynchronous failure and exclusivity without GPU dependencies
"""Run with python -m unittest tests/test_instra_network_unittest.py -v."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

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
        discovered = {"hostname": "scruffy", "instra_logs_root": "/logs", "wandb_root": "/wandb", "gpus": []}
        with patch.object(self.service, "_agent_request", return_value=discovered):
            self.assertEqual(self._await(self.service.submit("discover", self.service.local_id)["job_id"])["status"], "done")
        first = self.service._host(self.service.local_id)["last_discovered"]
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
        self.assertEqual(self.service._host(host_id)["state"], "available")
        self.assertNotIn("secret-value", network.CONFIG_PATH.read_text() + network.LOG_PATH.read_text())
        second = self._await(self.service.submit("add", address="192.168.1.2", ssh_port=22)["job_id"])
        self.assertEqual(second["category"], "validation")

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


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
