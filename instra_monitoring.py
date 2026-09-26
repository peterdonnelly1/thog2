# vvv THOG acquire remote Instra runs into isolated local copies and present them through the existing catalogue
"""Independent, read-only acquisition of participating thog_hosts' run data."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import mimetypes
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import sqlite3
import threading
import time
import uuid

import instra_network

DEFAULT_INTERVAL = 5
MIN_INTERVAL = 2
MAX_INTERVAL = 300
TERMINAL_STATES = {"finished", "stopped", "crashed"}
_active_service = None


def _time_now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.part")
    try:
        temporary.write_text(json.dumps(value, sort_keys=True))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _inside(path, root):
    try:
        target = PurePosixPath(path)
        if ".." in target.parts:
            return None
        return target.relative_to(PurePosixPath(root)).as_posix()
    except ValueError:
        return None


class MonitoringService:
    def __init__(self, network, catalog, *, storage_root=None, start_worker=True):
        global _active_service
        self.network = network
        self.catalog = catalog
        self.storage_root = Path(storage_root or instra_network.STATE_DIR / "monitoring").resolve()
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.settings_path = self.storage_root / "settings.json"
        try:
            self.settings = json.loads(self.settings_path.read_text())
        except (OSError, ValueError):
            self.settings = {}
        self.lock = threading.RLock()
        self.manifests = {}
        self.pending = set()
        self.active = set()
        self.last_checks = {}
        self.host_snapshot = (0.0, {})
        self.stop_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="instra-monitor")
        self._load_manifests()
        self.network.set_monitoring_provider(self.request_refresh)
        _active_service = self
        if start_worker:
            self.worker = threading.Thread(target=self._loop, name="instra-monitor-scheduler", daemon=True)
            self.worker.start()

    def _host_root(self, host_id):
        if not isinstance(host_id, str) or not host_id.startswith("thog_host.") or not all(
            character.isascii() and (character.isalnum() or character in "-_.") for character in host_id
        ):
            raise ValueError("Invalid thog_host_id")
        return self.storage_root / host_id

    def _manifest_path(self, host_id):
        return self._host_root(host_id) / "manifest.json"

    def _load_manifests(self):
        # Only currently registered producers may contribute runs, including disabled/disconnected ones.
        hosts = {host["thog_host_id"]: host for host in self.network.list_hosts()["hosts"] if not host["local"]}
        for host_id, host in hosts.items():
            try:
                manifest = json.loads(self._manifest_path(host_id).read_text())
                discovery = host.get("last_discovered") or {}
                if (manifest.get("logs_root") == discovery.get("instra_logs_root")
                        and manifest.get("wandb_root") == discovery.get("wandb_root")):
                    self.manifests[host_id] = manifest
            except (OSError, ValueError, TypeError):
                pass

    def interval(self, host_id):
        with self.lock:
            return self.settings.get(host_id, DEFAULT_INTERVAL)

    def request_refresh(self, host_id, discovery, refresh_interval=None):
        if refresh_interval is not None:
            if (isinstance(refresh_interval, bool) or not isinstance(refresh_interval, int)
                    or not MIN_INTERVAL <= refresh_interval <= MAX_INTERVAL):
                raise instra_network.NetworkError("validation", "Monitoring interval must be 2–300 seconds", host_id)
            with self.lock:
                self.settings[host_id] = refresh_interval
                _atomic_json(self.settings_path, self.settings)
        with self.lock:
            self.pending.add(host_id)
        return {"requested": host_id, "refresh_interval": self.interval(host_id)}

    def _loop(self):
        while not self.stop_event.is_set():
            hosts = self.network.list_hosts()["hosts"]
            for host in hosts:
                host_id = host["thog_host_id"]
                if host["local"] or not host["monitoring_enabled"] or not all(
                    (host.get("last_discovered") or {}).get(key) for key in ("instra_logs_root", "wandb_root")
                ):
                    continue
                with self.lock:
                    due = host_id in self.pending or time.monotonic() - self.last_checks.get(host_id, 0) >= self.interval(host_id)
                    if not due or host_id in self.active:
                        continue
                    self.pending.discard(host_id)
                    self.active.add(host_id)
                    self.last_checks[host_id] = time.monotonic()
                self.executor.submit(self._sync_host, host_id)
            self.stop_event.wait(0.5)

    def paths(self):
        with self.lock:
            manifests = {key: dict(value.get("runs", {})) for key, value in self.manifests.items()}
        paths = []
        for host_id, runs in manifests.items():
            for relative in runs:
                try:
                    path = self._host_root(host_id) / "logs" / relative
                    if path.is_file() and path.name == "charts.sqlite3" and not path.is_symlink():
                        paths.append(path)
                except (OSError, ValueError):
                    continue
        return tuple(paths)

    def hosts(self):
        with self.lock:
            checked_at, snapshot = self.host_snapshot
            if time.monotonic() - checked_at < 1:
                return snapshot
        snapshot = {item["thog_host_id"]: item for item in self.network.list_hosts()["hosts"]}
        with self.lock:
            self.host_snapshot = (time.monotonic(), snapshot)
        return snapshot

    def origin(self, path):
        path = Path(path)
        with self.lock:
            for host_id, manifest in self.manifests.items():
                relative = _inside(path, self._host_root(host_id) / "logs")
                if relative is not None and relative in manifest.get("runs", {}):
                    return host_id, relative, dict(manifest["runs"][relative]), manifest
        return None

    def _sync_database(self, host_id, relative, record, fingerprint):
        destination = self._host_root(host_id) / "logs" / relative
        fingerprint = [list(item) if item is not None else None for item in fingerprint]
        if destination.is_file() and record.get("database_fingerprint") == fingerprint:
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
        try:
            if destination.is_file():
                shutil.copyfile(destination, temporary)
            self.network.monitor_transfer(host_id, "logs", relative, temporary, database=True)
            # vvv THOG checkpoint the private replica before atomic replacement; otherwise its WAL is lost when the temporary name changes
            with sqlite3.connect(temporary) as connection:
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                connection.execute("PRAGMA journal_mode=DELETE")
                if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise ValueError("Acquired SQLite database failed integrity check")
            # ^^^ THOG
            os.replace(temporary, destination)
            record["database_fingerprint"] = fingerprint
            return True
        finally:
            temporary.unlink(missing_ok=True)
            Path(f"{temporary}-wal").unlink(missing_ok=True)
            Path(f"{temporary}-shm").unlink(missing_ok=True)

    def _sync_ordinary(self, host_id, root_kind, relative, size, modified, record):
        if Path(relative).name in {"charts.sqlite3", "charts.sqlite3-wal", "charts.sqlite3-shm"}:
            return False
        destination = self._host_root(host_id) / root_kind / relative
        key = f"{root_kind}/{relative}"
        fingerprint = [size, modified]
        if destination.is_file() and record.get("files", {}).get(key) == fingerprint:
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
        try:
            if destination.is_file():
                shutil.copyfile(destination, temporary)
            self.network.monitor_transfer(host_id, root_kind, relative, temporary)
            if temporary.stat().st_size != size:
                raise ValueError("Source file changed during acquisition")
            if root_kind == "wandb" and relative.endswith(".wandb"):
                parent = str(PurePosixPath(relative).parent)
                current = {name: (current_size, current_modified)
                           for name, current_size, current_modified in self.network.monitor_files(host_id, root_kind, parent)}
                if current.get(PurePosixPath(relative).name) != (size, modified):
                    raise ValueError("Growing W&B file changed during acquisition")
            os.replace(temporary, destination)
            record.setdefault("files", {})[key] = fingerprint
            return True
        finally:
            temporary.unlink(missing_ok=True)

    def _sync_host(self, host_id):
        began = time.monotonic()
        changed = 0
        error_category = None
        error_message = None
        try:
            host = self.network._host(host_id)
            discovery = host.get("last_discovered") or {}
            if not host["monitoring_enabled"] or not all(discovery.get(key) for key in ("instra_logs_root", "wandb_root")):
                return
            self.network.update_monitoring_status(host_id, {"refresh_interval": self.interval(host_id), "activity": "acquiring",
                                                          "last_success": host.get("monitoring_status", {}).get("last_success")})
            with self.lock:
                previous = self.manifests.get(host_id, {})
                if previous.get("logs_root") != discovery["instra_logs_root"] or previous.get("wandb_root") != discovery["wandb_root"]:
                    previous = {}
                manifest = json.loads(json.dumps(previous)) if previous else {
                    "logs_root": discovery["instra_logs_root"], "wandb_root": discovery["wandb_root"], "runs": {}}
            listing = self.network.monitor_files(host_id, "logs")
            files = {relative: (size, modified) for relative, size, modified in listing}
            databases = [relative for relative in files if Path(relative).name == "charts.sqlite3"]
            for missing in set(manifest["runs"]) - set(databases):
                manifest["runs"][missing]["error"] = "source unavailable"                                                                                # <<< THOG retain acquired results while marking a vanished source stale
                manifest["runs"][missing]["error_detail"] = "Run database is no longer listed on the producing host"
            for relative in sorted(databases):
                if not self.network._host(host_id)["monitoring_enabled"]:
                    break
                record = manifest["runs"].setdefault(relative, {"files": {}})
                db_stats = tuple(files.get(relative + suffix) for suffix in ("", "-wal"))
                try:
                    if self._sync_database(host_id, relative, record, db_stats):
                        changed += 1
                    path = self._host_root(host_id) / "logs" / relative
                    from sheet.local_chart_store import LocalChartReader
                    metadata = LocalChartReader(path).metadata()
                    record["run_state"] = metadata.get("run_state", "unknown")
                    parent = Path(relative).parent
                    for file_relative, (size, modified) in files.items():
                        is_run_log = Path(file_relative) == parent.parent / "train.log"
                        if file_relative != relative and (parent in Path(file_relative).parents or is_run_log):
                            if self._sync_ordinary(host_id, "logs", file_relative, size, modified, record):
                                changed += 1
                    recorded = metadata.get("wandb_run_directory", "")
                    wandb_relative = _inside(recorded, Path(discovery["wandb_root"])) if recorded else None
                    if wandb_relative is not None:
                        wandb_run = str(Path(wandb_relative).parent) if Path(wandb_relative).name == "files" else wandb_relative
                        if wandb_run != ".":
                            for filename, size, modified in self.network.monitor_files(host_id, "wandb", wandb_run):
                                file_relative = str(Path(wandb_run) / filename)
                                if self._sync_ordinary(host_id, "wandb", file_relative, size, modified, record):
                                    changed += 1
                        record["wandb_relative"] = wandb_relative
                    else:
                        record.pop("wandb_relative", None)                                                                                                   # <<< THOG never retain a mapping after the producer changes its W&B run path
                    record["acquired_at"] = _time_now()
                    record.pop("error", None)
                    record.pop("error_detail", None)
                except (instra_network.NetworkError, OSError, ValueError, sqlite3.DatabaseError) as error:
                    error_category = error.category if isinstance(error, instra_network.NetworkError) else "acquisition"
                    error_message = str(error) if isinstance(error, instra_network.NetworkError) else error_category
                    record["error"] = error_category
                    record["error_detail"] = (str(error) or error_category)[:200]
                    if not (self._host_root(host_id) / "logs" / relative).is_file():
                        manifest["runs"].pop(relative, None)
                    continue
            with self.lock:
                self.manifests[host_id] = manifest
                _atomic_json(self._manifest_path(host_id), manifest)
            if changed:
                instra_network.log_event("Monitoring", "acquire", host_id, "success",
                                         f"{changed} acquired files", duration_seconds=time.monotonic()-began)
        except (instra_network.NetworkError, OSError, ValueError) as error:
            error_category = error.category if isinstance(error, instra_network.NetworkError) else "acquisition"
            error_message = str(error) if isinstance(error, instra_network.NetworkError) else error_category
        finally:
            try:
                previous_status = self.network._host(host_id).get("monitoring_status") or {}
                success = error_category is None
                self.network.update_monitoring_status(host_id, {
                    "refresh_interval": self.interval(host_id), "activity": "idle" if success else "failed",
                    "last_success": _time_now() if success else previous_status.get("last_success"),
                    "latest_error": None if success else error_message})
                if error_category:
                    instra_network.log_event("Monitoring", "acquire", host_id, error_category,
                                             "Acquisition failed", duration_seconds=time.monotonic()-began)
            finally:
                with self.lock:
                    self.active.discard(host_id)

    def close(self):
        self.stop_event.set()
        if hasattr(self, "worker"):
            self.worker.join(timeout=3)
        self.executor.shutdown(wait=True)
        self.network.set_monitoring_provider(None)


def install(dashboard_module, network):
    """Attach the acquired paths to the original catalogue and all its established readers."""
    catalog_type = dashboard_module.DashboardCatalog
    state_type = dashboard_module.RunDashboardState
    original_init = catalog_type.__init__
    original_paths = catalog_type._candidate_paths
    original_state = catalog_type._state_for_path
    original_status = state_type.status
    original_delete_run = catalog_type.delete_run
    original_delete_file = catalog_type.delete_local_file
    original_wandb_files = catalog_type.wandb_files

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.monitoring = MonitoringService(network, self)

    def paths(self):
        return original_paths(self) + self.monitoring.paths()

    def state(self, path):
        result = original_state(self, path)
        result._instra_dashboard_catalog = self
        return result

    def status(self):
        value = original_status(self)
        catalog = self._instra_dashboard_catalog if hasattr(self, "_instra_dashboard_catalog") else None
        if catalog is None:
            return value
        origin = catalog.monitoring.origin(self.database_path)
        hosts = catalog.monitoring.hosts()
        if origin is None:
            host = hosts.get(network.local_id, {})
            host_id = network.local_id
            relative = None
            record = {}
            manifest = {}
        else:
            host_id, relative, record, manifest = origin
            host = hosts.get(host_id, {})
            value["dashboard_run_id"] = f"remote:{host_id}:{relative}"
            value["run_directory"] = str(self.database_path.parent.resolve())
            mapped = record.get("wandb_relative")
            value["wandb_run_directory"] = (str(catalog.monitoring._host_root(host_id) / "wandb" / mapped)
                                            if mapped is not None else "")                                                                               # <<< THOG never interpret an unmapped producer path on the monitoring host
        value["thog_host_id"] = host_id
        value["producing_host"] = host.get("display_name", host_id)
        value["source_chart_path"] = relative
        value["remote_copy"] = origin is not None
        value["acquired_at"] = record.get("acquired_at")
        monitor_status = host.get("monitoring_status") or {}
        last_success = monitor_status.get("last_success")
        fresh_ssh = False
        if last_success:
            try:
                fresh_ssh = (datetime.now(timezone.utc) - datetime.fromisoformat(last_success)).total_seconds() < max(
                    2 * catalog.monitoring.interval(host_id), 15)
            except ValueError:
                pass
        value["acquisition_state"] = ("local" if origin is None else
            "stale" if (not host.get("monitoring_enabled") or record.get("error") or monitor_status.get("activity") == "failed"
                        or (host.get("state") != "available" and not fresh_ssh)) else "current")
        value["acquisition_error"] = record.get("error_detail") or record.get("error") or host.get("monitoring_status", {}).get("latest_error")
        raw_gpu = value.get("gpu_index")
        gpus = (host.get("last_discovered") or {}).get("gpus", [])
        recorded_uuid = self.reader.metadata().get("gpu_uuid") or (value.get("configuration") or {}).get("gpu_uuid")
        matches = ([gpu for gpu in gpus if recorded_uuid and gpu.get("uuid") == recorded_uuid]
                   if recorded_uuid else [gpu for gpu in gpus if raw_gpu is not None and str(gpu.get("ordinal")) == str(raw_gpu)])
        value["gpu_assignment"] = ({"status": "verified", "ordinal": matches[0].get("ordinal"), "recorded_ordinal": raw_gpu,
                                    "model": matches[0].get("model"), "uuid": matches[0].get("uuid")}
                                   if len(matches) == 1 else {"status": "unverified" if raw_gpu is not None or recorded_uuid else "unknown",
                                                                "ordinal": raw_gpu})
        return value

    def guard_run(self, run_name):
        if self.monitoring.origin(self.state_for_run(run_name).database_path) is not None:
            raise PermissionError("Acquired remote runs cannot be deleted here")
        return original_delete_run(self, run_name)

    def guard_file(self, run_name, relative_path):
        if self.monitoring.origin(self.state_for_run(run_name).database_path) is not None:
            raise PermissionError("Acquired remote files cannot be deleted here")
        return original_delete_file(self, run_name, relative_path)

    def remote_wandb_path(self, run_name, relative_path):
        state = self.state_for_run(run_name)
        origin = self.monitoring.origin(state.database_path)
        if origin is None:
            raise PermissionError("Only acquired remote W&B files use this endpoint")
        host_id, _chart, record, _manifest = origin
        mapped = record.get("wandb_relative")
        if not mapped:
            raise FileNotFoundError("W&B run files have not been acquired")
        run_relative = PurePosixPath(mapped)
        if run_relative.name == "files":
            run_relative = run_relative.parent
        from run_thog2_local_dashboard_base import _normalise_relative_path
        relative = _normalise_relative_path(relative_path)
        root = (self.monitoring._host_root(host_id) / "wandb" / run_relative).resolve()
        candidate = root
        for part in relative.parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise PermissionError("Symbolic links cannot be opened from acquired runs")
        path = candidate.resolve(strict=True)
        if _inside(path, root) is None:
            raise PermissionError("W&B path is outside the acquired run")
        return path

    def wandb_files(self, run_name, relative_path="", *, refresh=False):
        state = self.state_for_run(run_name)
        if self.monitoring.origin(state.database_path) is None:
            return original_wandb_files(self, run_name, relative_path, refresh=refresh)
        from run_thog2_local_dashboard_base import _normalise_relative_path, _relative_path_text
        from urllib.parse import urlencode
        relative = _normalise_relative_path(relative_path)
        try:
            directory = self.remote_wandb_path(run_name, relative_path)
        except FileNotFoundError:
            return {"source": "wandb", "available": False, "entries": [], "error": "W&B run files not yet acquired",
                    "current_path": _relative_path_text(relative), "parent_path": None}
        if not directory.is_dir():
            raise NotADirectoryError(relative_path)
        entries = []
        for child in sorted(directory.iterdir()):
            if child.is_symlink():
                continue
            child_relative = _relative_path_text(relative / child.name)
            item = {"name": child.name, "path": child_relative, "kind": "folder" if child.is_dir() else "file",
                    "size": child.stat().st_size if child.is_file() else None,
                    "modified_at": datetime.fromtimestamp(child.stat().st_mtime, timezone.utc).isoformat(),
                    "mime_type": mimetypes.guess_type(child.name)[0] or ""}
            if child.is_file():
                item["download_url"] = "/api/remote-wandb-file?" + urlencode({"run": run_name, "path": child_relative})
            entries.append(item)
        return {"source": "wandb", "available": True, "entries": entries,
                "entry_count": len(entries), "current_path": _relative_path_text(relative),
                "parent_path": None if str(relative) == "." else _relative_path_text(relative.parent),
                "run_id": state.status()["dashboard_run_id"]}

    catalog_type.__init__ = init
    catalog_type._candidate_paths = paths
    catalog_type._state_for_path = state
    catalog_type.delete_run = guard_run
    catalog_type.delete_local_file = guard_file
    catalog_type.remote_wandb_path = remote_wandb_path
    catalog_type.wandb_files = wandb_files
    state_type.status = status
# ^^^ THOG
