# vvv THOG
"""Accelerate first-open catch-up of local W&B chart history without blocking HTTP requests."""

from __future__ import annotations

import threading
import queue
import time
from typing import Any


_CATCHUP_WORKERS = 2
_MAX_PENDING_SCANNERS = 16


def install(wandb_charts_module: Any) -> None:
    scanner_class = wandb_charts_module._WandbRunScanner
    if getattr(scanner_class, "_thog2_background_catchup_installed", False):
        return

    original_init = scanner_class.__init__
    pending = queue.Queue(maxsize=_MAX_PENDING_SCANNERS)

    def scanner_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        try:
            pending.put_nowait(self)
        except queue.Full:
            pass  # HTTP reads still advance this scanner in bounded parsing bursts.

    def _background_catchup() -> None:
        # Two shared workers and a bounded queue replace a thread per scanner.
        # Requeue after one burst so a mature history cannot starve other runs.
        while True:
            scanner = pending.get()
            try:
                if not getattr(scanner, "retired", False):
                    before = getattr(scanner, "good_offset", None)
                    scanner.refresh()
                    if bool(scanner.catching_up) and not getattr(scanner, "retired", False):
                        # A partially written record cannot advance until training appends bytes.
                        # Back off rather than spinning both shared workers on the same tail.
                        time.sleep(.25 if before is not None and before == scanner.good_offset else .02)
                        try:
                            pending.put_nowait(scanner)
                        except queue.Full:
                            pass
            except Exception:
                pass  # A broken file must not kill a shared worker.
            finally:
                pending.task_done()
                scanner = None  # Do not retain an evicted history while idle.

    scanner_class.__init__ = scanner_init
    scanner_class._thog2_background_catchup_installed = True
    workers = tuple(threading.Thread(target=_background_catchup, name="thog2-wandb-catchup", daemon=True)
                    for _ in range(_CATCHUP_WORKERS))
    scanner_class._thog2_background_workers = workers
    scanner_class._thog2_background_pending = pending
    for worker in workers:
        worker.start()
# ^^^ THOG
