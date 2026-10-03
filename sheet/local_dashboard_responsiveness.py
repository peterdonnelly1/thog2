# vvv THOG avoid repeated immutable database reads and unbounded SQLite waits in dashboard readers
"""Bound dashboard read latency without changing training database writers."""
from __future__ import annotations

import copy
from pathlib import Path
import threading


def file_signature(path):
    try:
        stat = Path(path).stat()
        return (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    except OSError:
        return None


def install(dashboard):
    state_class = dashboard.RunDashboardState
    if state_class.__dict__.get("_instra_read_cache_installed"):
        return
    original_init = state_class.__init__
    original_status = state_class.status

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._instra_status_read_lock = threading.Lock()
        self._instra_status_cache = None

    def status(self):
        path = self.database_path
        log_paths = (path.parent / "train.log", path.parent.parent / "train.log")
        from sheet.local_dashboard_logs_patch import _runner_attempt_log
        # Cache this path after the first metadata read; attempt IDs are immutable for a chart store.
        attempt_path = self.__dict__.get("_instra_attempt_log_path")
        signature = tuple(file_signature(item) for item in (path, Path(str(path) + "-wal"), *log_paths,
                                                            *([attempt_path] if attempt_path else [])))
        with self._instra_status_read_lock:
            cached = self._instra_status_cache
            if cached and cached[0] == signature:
                return copy.deepcopy(cached[1])
            self._thog2_status_cache = None  # Invalidate the older mtime-only cache when just the console log changes.
            result = original_status(self)
            if "_instra_attempt_log_path" not in self.__dict__:
                attempt_path = _runner_attempt_log(self)
                if attempt_path:
                    self._instra_attempt_log_path = attempt_path
                    signature += (file_signature(attempt_path),)
            self._instra_status_cache = (signature, copy.deepcopy(result))
            return result

    state_class.__init__ = initialize
    state_class.status = status
    state_class._instra_read_cache_installed = True
# ^^^ THOG
