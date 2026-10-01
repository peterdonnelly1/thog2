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
DEFAULT_DELETION_TIMEOUT_DAYS = 7                                                                                                                            # <<< THOG retain producer chart data while distributed deletion confirmations are outstanding
DELETION_SETTING_KEY = "__deletion_confirmation_timeout_days__"                                                                                              # <<< THOG persist the global deletion timeout beside Monitoring settings
_active_service = None


def _time_now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.part")
    try:
        with temporary.open("w") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
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


# vvv THOG validate deletion identities before resolving any receiver-side file path
def _chart_relative(value):
    return (isinstance(value, str) and bool(value) and not value.startswith("/")
            and not any(character in value for character in "\\\0\r\n")
            and all(part not in {"", ".", ".."} for part in value.split("/"))
            and PurePosixPath(value).name == "charts.sqlite3")


def _request_id_valid(value):
    return isinstance(value, str) and len(value) == 32 and all(character in "0123456789abcdef" for character in value)
# ^^^ THOG


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
        self.sync_locks = {}                                                                                                                                  # <<< THOG serialize each producer's acquisitions, deletion processing and force cleanup
        self.last_checks = {}
        self.host_snapshot = (0.0, {})
        self.deletions_path = self.storage_root / "deletions.json"                                                                                            # <<< THOG durable producer-side deletion requests and receipt state
        try:
            loaded_deletions = json.loads(self.deletions_path.read_text())
            self.deletions = loaded_deletions if isinstance(loaded_deletions, dict) else {}
        except (OSError, ValueError):
            self.deletions = {}
        self.force_delete_requests = {}                                                                                                                       # <<< THOG serialize local-force cleanup with in-flight acquisitions without persistent suppression
        self.receipt_retry_root = self.storage_root / "deletion_receipts"                                                                                     # <<< THOG durable receiver receipts survive restart and failed uploads
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
            # Publish the saved interval immediately; an in-flight acquisition may take minutes.
            previous = self.network._host(host_id).get("monitoring_status") or {}
            self.network.update_monitoring_status(host_id, {
                **previous, "refresh_interval": refresh_interval,
            })
        with self.lock:
            self.pending.add(host_id)
        return {"requested": host_id, "refresh_interval": self.interval(host_id)}

    # vvv THOG distributed deletion state, timeout capture, idempotent receiver cleanup and producer finalization
    def deletion_timeout_days(self):
        with self.lock:
            value = self.settings.get(DELETION_SETTING_KEY, DEFAULT_DELETION_TIMEOUT_DAYS)
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = DEFAULT_DELETION_TIMEOUT_DAYS
        return max(1, min(365, value))

    def set_deletion_timeout_days(self, days):
        if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= 365:
            raise ValueError("Deletion confirmation timeout must be between 1 and 365 days")
        with self.lock:
            self.settings[DELETION_SETTING_KEY] = days
            _atomic_json(self.settings_path, self.settings)
        return {"deletion_confirmation_timeout_days": days}

    def _persist_deletions(self):
        _atomic_json(self.deletions_path, self.deletions)

    def _pending_database_paths(self):
        with self.lock:
            return {str(item.get("database_path")) for item in self.deletions.values() if item.get("database_path")}

    def deletion_snapshot(self):
        hosts = {item["thog_host_id"]: item.get("display_name", item["thog_host_id"])
                 for item in self.network.list_hosts()["hosts"]}
        with self.lock:
            pending = []
            for request_id, item in self.deletions.items():
                expected = list(item.get("expected_hosts", []))
                received = set((item.get("receipts") or {}).keys())
                pending.append({
                    "request_id": request_id,
                    "run_id": item.get("run_id"),
                    "grid_tag": item.get("grid_tag"),
                    "requested_at": item.get("requested_at"),
                    "timeout_days": item.get("timeout_days"),
                    "outstanding_hosts": [hosts.get(host_id, host_id) for host_id in expected if host_id not in received],
                })
        return {"deletion_confirmation_timeout_days": self.deletion_timeout_days(), "pending": pending}

    def begin_authoritative_delete(self, run_name, grid_tag=None):
        state = self.catalog.state_for_run(run_name)
        database_path = state.database_path.resolve()
        if self.origin(database_path) is not None:
            raise PermissionError("Acquired remote runs require Force delete local copy")
        root = self.catalog.root.resolve()
        relative = _inside(database_path, root)
        if relative is None or database_path.name != "charts.sqlite3":
            raise PermissionError("refusing distributed deletion outside the configured Instra chart root")
        status = state.status()
        run_id = str(status["dashboard_run_id"])
        expected = sorted({host["thog_host_id"] for host in self.network.list_hosts()["hosts"]
                           if host["thog_host_id"] != self.network.local_id})
        request_id = uuid.uuid4().hex
        requested_at = _time_now()
        timeout_days = self.deletion_timeout_days()
        notice = {
            "schema": 1, "delete": True, "producer_host_id": self.network.local_id,
            "run_id": run_id, "source_chart_path": relative, "deletion_request_id": request_id,
            "requested_at": requested_at,
        }
        notice_dir = root / ".instra_deletions" / "notices"
        with self.lock:
            for previous_id, previous in self.deletions.items():
                if previous.get("database_path") == str(database_path):
                    return {"deleted_run_id": run_id, "deletion_request_id": previous_id, "pending": True}
            self.deletions[request_id] = {
                **notice, "database_path": str(database_path), "grid_tag": grid_tag,
                "expected_hosts": expected, "receipts": {}, "timeout_days": timeout_days,
            }
            self._persist_deletions()
        _atomic_json(notice_dir / f"{request_id}.json", notice)                                                                                                 # <<< THOG persist intent first; background recovery recreates a missing publication after interruption
        names = {host["thog_host_id"]: host.get("display_name", host["thog_host_id"]) for host in self.network.list_hosts()["hosts"]}
        outstanding = ", ".join(names.get(host_id, host_id) for host_id in expected) or "none"
        instra_network.log_event("Monitoring", "delete_pending", self.network.local_id, "pending",
                                 f"{run_id}: awaiting deletion confirmation from {outstanding}")
        return {"deleted_run_id": run_id, "deletion_request_id": request_id, "pending": True,
                "outstanding_hosts": [names.get(host_id, host_id) for host_id in expected]}

    def _clear_catalog_state(self, database_path):
        resolved = Path(database_path).resolve()
        with self.catalog.lock:
            state = self.catalog.states.pop(resolved, None)
            run_ids = [name for name, item in self.catalog.identity_states.items() if item is state]
            if state is not None:
                self.catalog.identity_states = {name: item for name, item in self.catalog.identity_states.items() if item is not state}
            self.catalog.wandb_file_cache.clear()
        for clear_cache in self.catalog.chart_cache_clearers if hasattr(self.catalog, "chart_cache_clearers") else ():
            clear_cache(resolved, run_ids)

    def _delete_chart_files(self, database_path):
        path = Path(database_path)
        for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
            if candidate.is_symlink():
                raise PermissionError("Refusing deletion through a symbolic link")
            candidate.unlink(missing_ok=True)                                                                                                                 # <<< THOG failed cleanup must remain pending and must never produce a successful receipt
        self._clear_catalog_state(path)

    def _finalize_deletions(self):
        now_epoch = time.time()
        receipts_root = self.catalog.root.resolve() / ".instra_deletions" / "receipts"
        completed = []
        with self.lock:
            items = list(self.deletions.items())
        for request_id, item in items:
            notice = self.catalog.root.resolve() / ".instra_deletions" / "notices" / f"{request_id}.json"
            if not notice.exists():
                _atomic_json(notice, {key: item[key] for key in ("schema", "delete", "producer_host_id", "run_id",
                                                               "source_chart_path", "deletion_request_id", "requested_at")})
            received = dict(item.get("receipts") or {})
            request_receipts = receipts_root / request_id
            if request_receipts.is_dir():
                for receipt_path in request_receipts.glob("*.json"):
                    try:
                        receipt = json.loads(receipt_path.read_text())
                    except (OSError, ValueError):
                        continue
                    if not isinstance(receipt, dict):
                        continue
                    host_id = receipt.get("responding_host_id")
                    if (receipt.get("schema") == 1 and receipt.get("deletion_processed") is True
                            and receipt.get("source_chart_path") == item.get("source_chart_path")
                            and receipt.get("producer_host_id") == self.network.local_id
                            and receipt.get("run_id") == item.get("run_id")
                            and receipt.get("deletion_request_id") == request_id
                            and host_id in item.get("expected_hosts", [])):
                        received[host_id] = receipt.get("processed_at") or _time_now()
            try:
                requested_epoch = datetime.fromisoformat(item["requested_at"]).timestamp()
            except (KeyError, TypeError, ValueError):
                requested_epoch = now_epoch
            timed_out = now_epoch - requested_epoch >= int(item.get("timeout_days", DEFAULT_DELETION_TIMEOUT_DAYS)) * 86400
            expected = set(item.get("expected_hosts", []))
            confirmed = expected.issubset(received)
            with self.lock:
                if request_id in self.deletions:
                    if self.deletions[request_id].get("receipts") != received:
                        self.deletions[request_id]["receipts"] = received
                        self._persist_deletions()
            if not confirmed and not timed_out:
                continue
            try:
                self._delete_chart_files(item["database_path"])
                notice.unlink(missing_ok=True)
                if request_receipts.exists():
                    shutil.rmtree(request_receipts)
            except OSError as error:
                instra_network.log_event("Monitoring", "delete_cleanup", self.network.local_id, "retry", str(error)[:200])
                continue
            with self.lock:
                self.deletions.pop(request_id, None)
                self._persist_deletions()
            outcome = "confirmed" if confirmed else "timeout"
            message = (f"{item.get('run_id')}: all deletion confirmations received"
                       if confirmed else f"{item.get('run_id')}: deletion confirmation timeout expired")
            instra_network.log_event("Monitoring", "delete_complete", self.network.local_id, outcome, message)
            completed.append(request_id)
        return completed

    def _remove_cached_remote_run(self, host_id, relative, manifest):
        if not _chart_relative(relative):
            raise ValueError("Invalid source chart path")
        record = manifest.get("runs", {}).get(relative)
        database = self._host_root(host_id) / "logs" / relative
        if _inside(database.resolve(), (self._host_root(host_id) / "logs").resolve()) is None:
            raise PermissionError("Refusing cached-run deletion outside the acquisition root")
        self._delete_chart_files(database)
        if record:
            shared_files = {key for source, other in manifest.get("runs", {}).items() if source != relative
                            for key in (other.get("files") or {})}                                                                                           # <<< THOG retain cached logs/W&B files still referenced by an unaffected acquired run
            for key in (record.get("files") or {}):
                if key in shared_files:
                    continue
                root_kind, _, file_relative = key.partition("/")
                if root_kind not in {"logs", "wandb"} or not file_relative:
                    continue
                path = self._host_root(host_id) / root_kind / file_relative
                if _inside(path.resolve(), (self._host_root(host_id) / root_kind).resolve()) is None:
                    raise PermissionError("Invalid acquired file path")
                if path.is_file() and not path.is_symlink():
                    path.unlink()
        manifest.get("runs", {}).pop(relative, None)
        return record is not None

    def _publish_manifest(self, host_id, manifest):
        _atomic_json(self._manifest_path(host_id), manifest)                                                                                                   # <<< THOG disk flushes run outside the lock used by Runs and Multiview catalogue reads
        with self.lock:
            self.manifests[host_id] = manifest

    def _process_deletion_notices(self, host_id, listing, manifest):
        notices = [relative for relative, _size, _modified in listing
                   if PurePosixPath(relative).parts[:2] == (".instra_deletions", "notices")
                   and len(PurePosixPath(relative).parts) == 3 and relative.endswith(".json")]
        suppressed, current_ids = set(), set()
        for relative in notices:
            request_id = PurePosixPath(relative).stem
            if not _request_id_valid(request_id):
                continue
            current_ids.add(request_id)
            local_receipt = self.receipt_retry_root / host_id / f"{request_id}.json"
            temporary = self.storage_root / "notices" / host_id / f"{request_id}.json"
            receipt = None
            try:
                temporary.parent.mkdir(parents=True, exist_ok=True)
                if local_receipt.is_file():
                    receipt = json.loads(local_receipt.read_text())
                    if (not isinstance(receipt, dict) or receipt.get("producer_host_id") != host_id
                            or receipt.get("deletion_request_id") != request_id
                            or receipt.get("responding_host_id") != self.network.local_id
                            or receipt.get("deletion_processed") is not True
                            or not _chart_relative(receipt.get("source_chart_path"))):
                        receipt = None
                if receipt is None:
                    self.network.monitor_transfer(host_id, "logs", relative, temporary)
                    if temporary.stat().st_size > 16384:
                        raise ValueError("Deletion notice is too large")
                    notice = json.loads(temporary.read_text())
                    if (not isinstance(notice, dict) or notice.get("schema") != 1 or notice.get("delete") is not True
                            or notice.get("producer_host_id") != host_id
                            or notice.get("deletion_request_id") != request_id
                            or not _chart_relative(notice.get("source_chart_path"))
                            or not isinstance(notice.get("run_id"), str) or not notice["run_id"]):
                        raise ValueError("Invalid deletion notice")
                    receipt = {key: notice[key] for key in ("schema", "producer_host_id", "run_id",
                                                            "source_chart_path", "deletion_request_id")}
                    receipt.update(responding_host_id=self.network.local_id, processed_at=_time_now(), deletion_processed=True)
                source_path = receipt["source_chart_path"]
                suppressed.add(source_path)
                self._remove_cached_remote_run(host_id, source_path, manifest)
                self._publish_manifest(host_id, manifest)
                _atomic_json(local_receipt, receipt)
                remote_receipt = f".instra_deletions/receipts/{request_id}/{self.network.local_id.replace('.', '_')}.json"
                self.network.monitor_upload_receipt(host_id, remote_receipt, receipt)
            except (OSError, ValueError, instra_network.NetworkError) as error:
                instra_network.log_event("Monitoring", "delete_receipt", host_id, "retry", str(error)[:200])
                if receipt is None and isinstance(error, (OSError, instra_network.NetworkError)):
                    raise                                                                                                                                    # <<< THOG an unread deletion notice must pause ordinary acquisition until its source identity is known
            finally:
                temporary.unlink(missing_ok=True)
        retry_dir = self.receipt_retry_root / host_id
        if retry_dir.is_dir():
            for receipt_path in retry_dir.glob("*.json"):
                if receipt_path.stem not in current_ids:
                    receipt_path.unlink(missing_ok=True)
        return suppressed

    def force_delete_local_copy(self, run_name):
        state = self.catalog.state_for_run(run_name)
        origin = self.origin(state.database_path)
        if origin is None:
            raise PermissionError("Force delete local copy applies only to acquired remote runs")
        host_id, relative, _record, _manifest = origin
        run_id = str(state.status()["dashboard_run_id"])
        with self.lock:
            self.force_delete_requests.setdefault(host_id, set()).add(relative)
            self.pending.add(host_id)
            if host_id not in self.active:
                self.active.add(host_id)
                self.executor.submit(self._sync_host, host_id)
        return {"queued": True, "run_id": run_id, "producing_host_id": host_id}

    def _apply_force_delete_requests(self, host_id, manifest):
        with self.lock:
            requested = set(self.force_delete_requests.get(host_id, set()))
        for relative in requested:
            self._remove_cached_remote_run(host_id, relative, manifest)
        if requested:
            self._publish_manifest(host_id, manifest)
            with self.lock:
                pending = self.force_delete_requests.get(host_id, set())
                pending.difference_update(requested)
                if not pending:
                    self.force_delete_requests.pop(host_id, None)

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                self._finalize_deletions()
            except (OSError, ValueError, KeyError) as error:
                instra_network.log_event("Monitoring", "delete_cleanup", self.network.local_id, "retry", str(error)[:200])
            hosts = self.network.list_hosts()["hosts"]
            for host in hosts:
                host_id = host["thog_host_id"]
                with self.lock:
                    force_pending = bool(self.force_delete_requests.get(host_id))
                if host["local"] or (not force_pending and (not host["monitoring_enabled"] or not all(
                    (host.get("last_discovered") or {}).get(key) for key in ("instra_logs_root", "wandb_root")
                ))):
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
            force_hidden = {(host_id, relative) for host_id, relatives in self.force_delete_requests.items() for relative in relatives}
        paths = []
        for host_id, runs in manifests.items():
            for relative in runs:
                if (host_id, relative) in force_hidden:
                    continue
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
        with self.lock:
            sync_lock = self.sync_locks.setdefault(host_id, threading.Lock())
        if not sync_lock.acquire(blocking=False):
            return
        with self.lock:
            self.active.add(host_id)
        manifest = None
        began = time.monotonic()
        changed = 0
        error_category = None
        error_message = None
        try:
            host = self.network._host(host_id)
            discovery = host.get("last_discovered") or {}
            with self.lock:
                force_pending = bool(self.force_delete_requests.get(host_id))
                if force_pending:
                    manifest = json.loads(json.dumps(self.manifests.get(host_id, {"runs": {}})))
            if force_pending:
                self._apply_force_delete_requests(host_id, manifest)                                                                                          # <<< THOG local force deletion never waits for SSH or reacquires the copy during its own cleanup cycle
                return
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
            suppressed = self._process_deletion_notices(host_id, listing, manifest)                                                                            # <<< THOG deletion notices win before any ordinary acquisition can restore a stale copy
            databases = [relative for relative in files if Path(relative).name == "charts.sqlite3" and relative not in suppressed]
            # vvv THOG index each local-log file to its run once; the old nested full-list scan became quadratic as monitored history grew
            database_parents = {str(PurePosixPath(relative).parent): relative for relative in databases}
            files_by_database = {relative: [] for relative in databases}
            train_log_to_databases = {}
            for parent_text, database in database_parents.items():
                train_log_to_databases.setdefault(str(PurePosixPath(parent_text).parent / "train.log"), []).append(database)
            for file_relative, stats in files.items():
                path_parts = PurePosixPath(file_relative)
                for parent in path_parts.parents:
                    database = database_parents.get(str(parent))
                    if database is not None:
                        files_by_database[database].append((file_relative, stats))
                        break
                for database in train_log_to_databases.get(file_relative, ()):
                    files_by_database[database].append((file_relative, stats))
            # ^^^ THOG
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
                    for file_relative, (size, modified) in files_by_database.get(relative, ()):                                                                  # <<< THOG consume the pre-indexed run slice instead of rescanning every remote file per run
                        if file_relative != relative:
                            if self._sync_ordinary(host_id, "logs", file_relative, size, modified, record):
                                changed += 1
                    recorded = metadata.get("wandb_run_directory", "")
                    wandb_relative = _inside(recorded, Path(discovery["wandb_root"])) if recorded else None
                    if wandb_relative is not None:
                        wandb_run = str(Path(wandb_relative).parent) if Path(wandb_relative).name == "files" else wandb_relative
                        # vvv THOG completed runs stop issuing one remote W&B find per five-second poll; live runs remain fresh
                        poll_wandb = record.get("run_state") not in TERMINAL_STATES or not record.get("wandb_complete")
                        if wandb_run != "." and poll_wandb:
                            for filename, size, modified in self.network.monitor_files(host_id, "wandb", wandb_run):
                                file_relative = str(Path(wandb_run) / filename)
                                if self._sync_ordinary(host_id, "wandb", file_relative, size, modified, record):
                                    changed += 1
                            if record.get("run_state") in TERMINAL_STATES:
                                record["wandb_complete"] = True
                        # ^^^ THOG
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
            self._apply_force_delete_requests(host_id, manifest)                                                                                              # <<< THOG a force-delete requested during acquisition wins before the refreshed manifest is published
            self._publish_manifest(host_id, manifest)
            if changed:
                instra_network.log_event("Monitoring", "acquire", host_id, "success",
                                         f"{changed} acquired files", duration_seconds=time.monotonic()-began)
        except (instra_network.NetworkError, OSError, ValueError) as error:
            error_category = error.category if isinstance(error, instra_network.NetworkError) else "acquisition"
            error_message = str(error) if isinstance(error, instra_network.NetworkError) else error_category
        finally:
            try:
                if manifest is not None:
                    self._publish_manifest(host_id, manifest)
                with self.lock:
                    force_pending = bool(self.force_delete_requests.get(host_id))
                    cleanup_manifest = json.loads(json.dumps(manifest if manifest is not None else
                                                             self.manifests.get(host_id, {"runs": {}})))
                if force_pending:
                    self._apply_force_delete_requests(host_id, cleanup_manifest)
                    self._publish_manifest(host_id, cleanup_manifest)
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
                sync_lock.release()

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
        hidden = self.monitoring._pending_database_paths()                                                                                                    # <<< THOG deletion-pending producer runs disappear immediately while chart files remain until confirmations/timeout
        local_paths = tuple(path for path in original_paths(self) if str(path.resolve()) not in hidden)
        return local_paths + self.monitoring.paths()

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
            raise PermissionError("Acquired remote runs require Force delete local copy")
        status = self.state_for_run(run_name).status()
        return self.monitoring.begin_authoritative_delete(run_name, grid_tag=status.get("runner_grid_tag"))                                                     # <<< THOG authoritative deletion enters receipt-backed pending state instead of immediately destroying chart data

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
    catalog_type.force_delete_local_copy = lambda self, run_name: self.monitoring.force_delete_local_copy(run_name)                                           # <<< THOG explicit disposable-cache action without producer suppression
    catalog_type.deletion_snapshot = lambda self: self.monitoring.deletion_snapshot()                                                                         # <<< THOG expose pending/outstanding deletion state to the UI
    catalog_type.set_deletion_timeout_days = lambda self, days: self.monitoring.set_deletion_timeout_days(days)                                               # <<< THOG persist the global timeout used by future deletion requests
    state_type.status = status
# ^^^ THOG
