# vvv THOG capture live stacks when a backend request stalls, without holding service locks
"""Low-cost diagnostics for long-lived Instra backends."""
import faulthandler
import json
import os
from pathlib import Path
import threading
import time
from urllib.parse import urlparse


def instrument_handler(handler, state_dir):
    active, lock = {}, threading.Lock()
    started_at = time.monotonic()
    watchdog_started = False

    def watchdog():
        while True:
            time.sleep(10)
            with lock:
                stalled = [item for item in active.values() if time.monotonic() - item["started"] > 30 and not item["reported"]]
                for item in stalled:
                    item["reported"] = True
            if not stalled:
                continue
            try:
                directory = Path(state_dir)
                directory.mkdir(parents=True, exist_ok=True, mode=0o700)
                path = directory / "backend-stalls.log"
                if path.exists() and path.stat().st_size > 4 * 1024 * 1024:
                    os.replace(path, directory / "backend-stalls.previous.log")
                with path.open("a") as stream:
                    os.chmod(path, 0o600)
                    stream.write(json.dumps({"time": time.time(), "pid": os.getpid(), "requests": [
                        {"path": item["path"], "seconds": round(time.monotonic() - item["started"], 1)} for item in stalled]}) + "\n")
                    stream.flush()
                    faulthandler.dump_traceback(file=stream, all_threads=True)
            except OSError:
                pass

    def wrap(original):
        def request(self):
            nonlocal watchdog_started
            path = urlparse(self.path).path
            request_id = id(self)
            with lock:
                active[request_id] = {"path": path, "started": time.monotonic(), "reported": False}
                if not watchdog_started:
                    watchdog_started = True
                    threading.Thread(target=watchdog, name="instra-stall-diagnostics", daemon=True).start()
            try:
                if path == "/api/health" and self.command == "GET":
                    with lock:
                        requests = [{"path": item["path"], "seconds": round(time.monotonic() - item["started"], 2)}
                                    for key, item in active.items() if key != request_id]
                    return self._send_json({"uptime_seconds": round(time.monotonic() - started_at, 1), "requests": requests})
                return original(self)
            finally:
                with lock:
                    active.pop(request_id, None)
        return request

    handler.do_GET = wrap(handler.do_GET)
    handler.do_POST = wrap(handler.do_POST)
    handler.do_DELETE = wrap(handler.do_DELETE)
    return handler
# ^^^ THOG
