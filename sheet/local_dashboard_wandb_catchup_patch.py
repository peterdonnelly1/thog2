# vvv THOG
"""Accelerate first-open catch-up of local W&B chart history without blocking HTTP requests."""

from __future__ import annotations

import threading
import time
from typing import Any


_CATCHUP_SLEEP_SECONDS = 0.02


def install(wandb_charts_module: Any) -> None:
    scanner_class = wandb_charts_module._WandbRunScanner
    if getattr(scanner_class, "_thog2_background_catchup_installed", False):
        return

    original_init = scanner_class.__init__

    def scanner_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        worker = threading.Thread(
            target=_background_catchup,
            args=(self,),
            name="thog2-wandb-catchup",
            daemon=True,
        )
        self._thog2_background_catchup_thread = worker
        worker.start()

    def _background_catchup(scanner: Any) -> None:
        # refresh() already owns the scanner lock and limits each parsing burst to
        # the established time budget. Running those bursts back-to-back avoids
        # making mature runs wait for many 2.5-second browser polling intervals.
        while True:
            scanner.refresh()
            if not bool(scanner.catching_up):
                return
            time.sleep(_CATCHUP_SLEEP_SECONDS)

    scanner_class.__init__ = scanner_init
    scanner_class._thog2_background_catchup_installed = True
# ^^^ THOG
