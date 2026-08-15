# vvv THOG
"""Launch the local THOG2 dashboard with an obvious Linux process name."""

from __future__ import annotations

import ctypes
import sys

from sheet import local_heatmap_loss_metadata_patch as _local_heatmap_loss_metadata_patch
from run_thog2_local_dashboard import main


_PROCESS_NAME = b"thog2-dashboard"
_PR_SET_NAME = 15


def _set_process_name() -> None:
    if not sys.platform.startswith("linux"):
        return
    try:
        libc = ctypes.CDLL(None)
        prctl = libc.prctl
        prctl.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_ulong,
        )
        prctl.restype = ctypes.c_int
        prctl(_PR_SET_NAME, _PROCESS_NAME, 0, 0, 0)
    except (AttributeError, OSError):
        return


if __name__ == "__main__":
    _set_process_name()
    raise SystemExit(main())
# ^^^ THOG
