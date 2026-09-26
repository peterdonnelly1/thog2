# vvv THOG exercise multi-host catalogue identity, immutable acquisition, failure recovery and existing reader paths
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

import instra_monitoring
import instra_network
# The execution workspace has no GPU runtime; the dashboard readers under test use only the chart store.
if "sheet" not in sys.modules and "torch" not in sys.modules:
    sheet_package = types.ModuleType("sheet")
    sheet_package.__path__ = [str(Path(__file__).resolve().parents[1] / "sheet")]
    sys.modules["sheet"] = sheet_package
    for module_name in ("depth_weight_curves_v2_patch", "plastic_depth_wandb_probe_curves_patch"):
        sys.modules[f"sheet.{module_name}"] = types.ModuleType(f"sheet.{module_name}")
import run_thog2_local_dashboard as dashboard
from sheet.local_chart_store import LocalChartStore
from sheet.local_dashboard_wandb_charts_patch import _ScannerCatalog


class FakeNetwork:
    local_id = "thog_host.viewer"

    def __init__(self, root):
        self.root = root
        self.host_data = {}
        self.transfer_calls = []
        self.fail_transfer = False
        self.events = []

    def add_host(self, host_id, root):
        self.host_data[host_id] = {
            "thog_host_id": host_id, "display_name": host_id.split(".")[-1], "local": False,
            "address": "localhost", "ssh_port": 22, "ssh_user": "test_user",
            "monitoring_enabled": True, "state": "available", "monitoring_status": {},
            "last_discovered": {"instra_logs_root": str(root / "logs"), "wandb_root": str(root / "wandb"),
                                "gpus": [{"ordinal": 0, "uuid": "GPU-stable", "model": "Test GPU"}]}}

    def list_hosts(self):
        return {"hosts": [{"thog_host_id": self.local_id, "local": True, "monitoring_enabled": True,
                           "display_name": "viewer", "last_discovered": {"gpus": []}}] + list(self.host_data.values())}

    def _host(self, host_id):
        return self.host_data[host_id]

    def set_monitoring_provider(self, callback):
        self.provider = callback

    def update_monitoring_status(self, host_id, value):
        self.host_data[host_id]["monitoring_status"] = value

    def monitor_files(self, host_id, root_kind, relative_directory=""):
        root = Path(self._host(host_id)["last_discovered"]["instra_logs_root" if root_kind == "logs" else "wandb_root"])
        root = root / relative_directory
        if not root.exists():
            raise instra_network.NetworkError("missing", "Missing run folder", host_id)
        return [(str(path.relative_to(root)), path.stat().st_size, path.stat().st_mtime_ns)
                for path in root.rglob("*") if path.is_file() and not path.is_symlink()]

    def monitor_transfer(self, host_id, root_kind, relative, destination, *, database=False):
        if self.fail_transfer:
            raise instra_network.NetworkError("transfer", "Simulated failure", host_id)
        root = Path(self._host(host_id)["last_discovered"]["instra_logs_root" if root_kind == "logs" else "wandb_root"])
        source = root / relative
        self.transfer_calls.append((host_id, root_kind, relative, database))
        if database:
            with sqlite3.connect(source) as source_db, sqlite3.connect(destination) as target_db:
                source_db.backup(target_db)
        else:
            shutil.copyfile(source, destination)
        return Path(destination)


class MonitoringTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.network = FakeNetwork(self.root)
        event_patch = mock.patch.object(instra_network, "log_event",
                                        side_effect=lambda *args, **kwargs: self.network.events.append((args, kwargs)))
        event_patch.start()
        self.addCleanup(event_patch.stop)
        self.original = {name: (getattr(dashboard.DashboardCatalog, name)) for name in (
            "__init__", "_candidate_paths", "_state_for_path", "delete_run", "delete_local_file", "wandb_files")}
        self.original_status = dashboard.RunDashboardState.status
        self.catalog = dashboard.DashboardCatalog(root=self.root / "local")
        self.monitor = instra_monitoring.MonitoringService(self.network, self.catalog,
                                                            storage_root=self.root / "storage", start_worker=False)
        self.catalog.monitoring = self.monitor
        instra_monitoring.install(dashboard._base, self.network)
        self.addCleanup(self._restore)

    def _restore(self):
        self.monitor.close()
        for name, value in self.original.items():
            setattr(dashboard.DashboardCatalog, name, value)
        dashboard.RunDashboardState.status = self.original_status

    def _producer(self, name, *, gpu=0):
        host_id = "thog_host." + name
        root = self.root / name
        (root / "logs" / "run" / "same_id").mkdir(parents=True)
        (root / "wandb" / "run-same_id" / "files").mkdir(parents=True)
        self.network.add_host(host_id, root)
        path = root / "logs" / "run" / "same_id" / "charts.sqlite3"
        store = LocalChartStore(path, run_name="same run", run_id="same_id", wandb_run_id="same_id",
                                config={"host_label": "misleading", "gpu_index": gpu})
        store.connection.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)",
                                 ("wandb_run_directory", str(root / "wandb" / "run-same_id" / "files")))
        store.connection.commit()
        store.close()
        (path.parent / "processing").mkdir()
        (path.parent / "processing" / "processing_data.json").write_text('{"rows":[]}')
        (root / "wandb" / "run-same_id" / "run-same_id.wandb").write_bytes(b"wandb data")
        (root / "wandb" / "run-same_id" / "files" / "metadata.json").write_text('{"ok":true}')
        (root / "logs" / "run" / "train.log").write_text("T 42 loss=9.876 Δstep=4.0 s\n")
        return host_id, path

    def test_same_wandb_id_from_two_hosts_stays_separate_and_files_are_mapped(self):
        first, path_a = self._producer("scruffy")
        second, path_b = self._producer("dreedle", gpu=9)
        self.monitor._sync_host(first)
        self.monitor._sync_host(second)
        self.assertEqual([args[3] for args, _ in self.network.events], ["success", "success"])
        runs = self.catalog.runs()["runs"]
        self.assertEqual(len(runs), 2)
        self.assertEqual({run["thog_host_id"] for run in runs}, {first, second})
        self.assertEqual(len({run["dashboard_run_id"] for run in runs}), 2)
        self.assertTrue(all(run["host_label"] == "misleading" for run in runs))
        self.assertEqual({run["producing_host"] for run in runs}, {"scruffy", "dreedle"})
        self.assertTrue(all(run["last_loss"] == 9.876 for run in runs))
        self.assertEqual({run["gpu_assignment"]["status"] for run in runs}, {"verified", "unverified"})
        scanner = _ScannerCatalog(self.catalog)
        scanned_paths = {scanner._find_path("same_id", run) for run in runs}
        self.assertEqual(len(scanned_paths), 2)
        self.assertTrue(all(path.is_file() for path in scanned_paths))
        for run in runs:
            self.assertEqual(self.catalog.state_for_run(run["dashboard_run_id"]).database_path.name, "charts.sqlite3")
            self.assertTrue(Path(run["wandb_run_directory"]).is_dir())
            self.assertEqual(self.catalog.local_file(run["dashboard_run_id"], "processing/processing_data.json").read_text(), '{"rows":[]}')
            self.assertTrue(self.catalog.wandb_files(run["dashboard_run_id"])["available"])
            self.assertEqual(self.catalog.remote_wandb_path(run["dashboard_run_id"], "files/metadata.json").read_text(), '{"ok":true}')
            with self.assertRaises(PermissionError):
                self.catalog.delete_run(run["dashboard_run_id"])
            with self.assertRaises(PermissionError):
                self.catalog.delete_local_file(run["dashboard_run_id"], "processing/processing_data.json")
        self.assertTrue(path_a.is_file() and path_b.is_file())

    def test_latest_log_loss_updates_and_ordinary_transfer_reuses_previous_copy(self):
        host_id, path = self._producer("source")
        self.monitor._sync_host(host_id)
        log_path = path.parent.parent / "train.log"
        log_path.write_text("T 42 loss=9.876\nV 43 validation loss=6.3\nT 44 loss=10.24\n")
        ordinary_transfer = self.network.monitor_transfer
        reused = []
        def transfer(host_id, root_kind, relative, destination, *, database=False):
            if relative == "run/train.log":
                reused.append(Path(destination).read_text() if Path(destination).is_file() else "")
            return ordinary_transfer(host_id, root_kind, relative, destination, database=database)
        with mock.patch.object(self.network, "monitor_transfer", side_effect=transfer):
            self.monitor._sync_host(host_id)
        self.assertEqual(reused, ["T 42 loss=9.876 Δstep=4.0 s\n"])
        self.assertEqual(self.catalog.runs()["runs"][0]["last_loss"], 10.24)

    def test_failure_keeps_usable_copy_then_catches_up(self):
        host_id, path = self._producer("source")
        with sqlite3.connect(path) as connection:
            connection.execute("UPDATE metadata SET value='recording' WHERE key='run_state'")
        self.monitor._sync_host(host_id)
        first = self.catalog.runs()["runs"][0]
        self.assertEqual(first["acquisition_state"], "current")
        with sqlite3.connect(path) as connection:
            connection.execute("UPDATE metadata SET value='finished' WHERE key='run_state'")
        self.network.fail_transfer = True
        self.monitor._sync_host(host_id)
        self.monitor.host_snapshot = (0, {})
        stale = self.catalog.runs()["runs"][0]
        self.assertEqual(stale["acquisition_state"], "stale")
        self.assertEqual(stale["run_state"], first["run_state"])
        self.network.fail_transfer = False
        self.monitor._sync_host(host_id)
        self.monitor.host_snapshot = (0, {})
        self.assertEqual(self.catalog.runs()["runs"][0]["run_state"], "finished")
        calls = len(self.network.transfer_calls)
        self.monitor._sync_host(host_id)
        self.assertEqual(len(self.network.transfer_calls), calls)

    def test_source_disappearing_keeps_run_and_explains_stale_copy(self):
        host_id, path = self._producer("source")
        self.monitor._sync_host(host_id)
        path.unlink()
        self.monitor._sync_host(host_id)
        self.monitor.host_snapshot = (0, {})
        run = self.catalog.runs()["runs"][0]
        self.assertEqual(run["acquisition_state"], "stale")
        self.assertIn("no longer listed", run["acquisition_error"])
        self.assertEqual(self.network._host(host_id)["monitoring_status"]["activity"], "idle")

    def test_loss_falls_back_to_latest_probe_when_training_log_is_unavailable(self):
        host_id, path = self._producer("source")
        (path.parent.parent / "train.log").unlink()
        store = LocalChartStore(path, run_name="same run", run_id="same_id", wandb_run_id="same_id", config={})
        store.append_heatmap_records([{
            "optimizer_update": 42, "probe_id": "P42", "active_layers": 4,
            "selected_layers": 4, "shrink": (), "growth": (), "current_loss": 10.25,
        }])
        store.close()
        self.monitor._sync_host(host_id)
        self.assertEqual(self.catalog.runs()["runs"][0]["last_loss"], 10.25)

    def test_disabling_stops_scheduling_but_keeps_copy_and_interval_validates(self):
        host_id, _path = self._producer("source")
        self.monitor._sync_host(host_id)
        self.network._host(host_id)["monitoring_enabled"] = False
        self.monitor.host_snapshot = (0, {})
        self.assertEqual(self.catalog.runs()["runs"][0]["acquisition_state"], "stale")
        with self.assertRaises(instra_network.NetworkError):
            self.monitor.request_refresh(host_id, {}, refresh_interval=1)
        self.monitor.request_refresh(host_id, {}, refresh_interval=9)
        self.assertEqual(self.monitor.interval(host_id), 9)

    def test_agent_can_be_unavailable_while_ssh_copy_is_current_and_restart_keeps_data(self):
        host_id, _path = self._producer("source")
        self.network._host(host_id)["state"] = "unavailable"
        self.monitor._sync_host(host_id)
        self.monitor.host_snapshot = (0, {})
        self.assertEqual(self.catalog.runs()["runs"][0]["acquisition_state"], "current")
        replacement = instra_monitoring.MonitoringService(self.network, self.catalog,
                                                            storage_root=self.root / "storage", start_worker=False)
        self.addCleanup(replacement.close)
        self.catalog.monitoring = replacement
        self.assertEqual(len(self.catalog.runs()["runs"]), 1)

    def test_unreadable_first_database_does_not_enter_catalogue(self):
        host_id, _path = self._producer("source")
        self.network.fail_transfer = True
        self.monitor._sync_host(host_id)
        self.assertEqual(self.catalog.runs()["runs"], [])
        self.assertEqual(self.network._host(host_id)["monitoring_status"]["activity"], "failed")

    def test_missing_transfer_dependency_names_program_in_host_status(self):
        host_id, _path = self._producer("source")
        def missing_dependency(*_args, **_kwargs):
            raise instra_network.NetworkError("dependency", "sqlite3_rsync is missing on the monitoring host; install it on both hosts", host_id)
        self.network.monitor_transfer = missing_dependency
        self.monitor._sync_host(host_id)
        self.assertEqual(self.network._host(host_id)["monitoring_status"]["activity"], "failed")
        self.assertIn("sqlite3_rsync", self.network._host(host_id)["monitoring_status"]["latest_error"])
        self.assertEqual(self.catalog.runs()["runs"], [])
        self.assertEqual(self.network.events[-1][0][3], "dependency")

    def test_network_rejects_path_escape_before_running_ssh(self):
        host_id, _path = self._producer("source")
        service = instra_network.NetworkService(start_worker=False)
        self.addCleanup(service.close)
        with mock.patch.object(service, "_host", return_value=self.network._host(host_id)):
            with mock.patch.object(instra_network, "_is_known", return_value=True):
                for relative in ("../secret", "/etc/passwd", "run/../../secret", "a\\b"):
                    with self.subTest(relative=relative), self.assertRaises(instra_network.NetworkError):
                        service._monitor_source(host_id, "logs", relative)

    def test_existing_http_routes_and_remote_download_use_local_copies(self):
        host_id, _path = self._producer("source")
        self.monitor._sync_host(host_id)
        run_id = self.catalog.runs()["runs"][0]["dashboard_run_id"]
        server = ThreadingHTTPServer(("127.0.0.1", 0), dashboard._base._handler_for(self.catalog))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        address = f"http://127.0.0.1:{server.server_port}"
        def get(path):
            with urlopen(address + path, timeout=4) as response:
                return response.read()
        self.assertIn(run_id.encode(), get("/api/runs"))
        self.assertIn(host_id.encode(), get("/api/status?run=" + quote(run_id)))
        self.assertIn(b"processing_data.json", get("/api/local-files?run=" + quote(run_id) + "&path=processing"))
        self.assertEqual(get("/api/remote-wandb-file?run=" + quote(run_id) + "&path=files%2Fmetadata.json"), b'{"ok":true}')
        with self.assertRaises(HTTPError) as denied:
            urlopen(Request(address + "/api/run?run=" + quote(run_id), method="DELETE"), timeout=4)
        self.assertEqual(denied.exception.code, 403)

    def test_transport_uses_pull_only_commands_and_sqlite_protocol(self):
        host_id, _path = self._producer("source")
        service = instra_network.NetworkService(start_worker=False)
        self.addCleanup(service.close)
        captured = []
        def run(command, **kwargs):
            captured.append(command)
            return mock.Mock(returncode=0, stdout=b"")
        with mock.patch.object(service, "_host", return_value=self.network._host(host_id)):
            with mock.patch.object(instra_network, "_is_known", return_value=True):
                with mock.patch.object(instra_network.subprocess, "run", side_effect=run):
                    with mock.patch.object(instra_network.shutil, "which", return_value="/usr/bin/sqlite3_rsync"):
                        service.monitor_transfer(host_id, "logs", "run/same_id/charts.sqlite3", self.root / "copy.sqlite3", database=True)
                        service.monitor_transfer(host_id, "logs", "run/same_id/processing/processing_data.json", self.root / "copy.json")
        self.assertEqual(captured[0][0], "sqlite3_rsync")
        self.assertIn("test_user@localhost:", captured[0][-2])
        self.assertEqual(captured[1][0], "rsync")
        self.assertNotIn("--delete", captured[1])
        self.assertTrue(captured[1][-2].startswith("test_user@localhost:"))
        self.assertFalse(captured[1][-1].startswith("test_user@localhost:"))

    def test_transfer_reports_remote_missing_tool_and_useful_bounded_errors(self):
        host_id, _path = self._producer("source")
        service = instra_network.NetworkService(start_worker=False)
        self.addCleanup(service.close)
        destination = self.root / "copy.sqlite3"
        with mock.patch.object(service, "_host", return_value=self.network._host(host_id)):
            with mock.patch.object(instra_network, "_is_known", return_value=True):
                with mock.patch.object(instra_network.shutil, "which", return_value="/usr/local/bin/sqlite3_rsync"):
                    with mock.patch.object(instra_network.subprocess, "run",
                                           return_value=mock.Mock(returncode=127, stderr=b"sh: 1: sqlite3_rsync: not found")):
                        with self.assertRaises(instra_network.NetworkError) as missing:
                            service.monitor_transfer(host_id, "logs", "run/same_id/charts.sqlite3", destination, database=True)
                    self.assertEqual(missing.exception.category, "dependency")
                    self.assertIn("producing host", str(missing.exception))
                    with mock.patch.object(instra_network.subprocess, "run",
                                           return_value=mock.Mock(returncode=2, stderr=b"cannot read origin\n" + b"x" * 600)):
                        with self.assertRaises(instra_network.NetworkError) as failed:
                            service.monitor_transfer(host_id, "logs", "run/same_id/charts.sqlite3", destination, database=True)
                    self.assertEqual(failed.exception.category, "transfer")
                    self.assertIn("(exit 2): cannot read origin", str(failed.exception))
                    self.assertLess(len(str(failed.exception)), 260)

    def test_one_slow_host_does_not_block_another(self):
        slow, _ = self._producer("slow")
        fast, _ = self._producer("fast")
        entered = threading.Event()
        release = threading.Event()
        transfer = self.network.monitor_transfer
        def delayed(host_id, *args, **kwargs):
            if host_id == slow and kwargs.get("database"):
                entered.set()
                release.wait(4)
            return transfer(host_id, *args, **kwargs)
        self.network.monitor_transfer = delayed
        self.monitor.worker = threading.Thread(target=self.monitor._loop, daemon=True)
        self.monitor.worker.start()
        try:
            self.assertTrue(entered.wait(2))
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and fast not in self.monitor.manifests:
                time.sleep(0.02)
            self.assertIn(fast, self.monitor.manifests)
            self.assertNotIn(slow, self.monitor.manifests)
        finally:
            release.set()
# ^^^ THOG
