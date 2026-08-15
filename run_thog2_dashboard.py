# vvv THOG
"""Launch the local THOG2 dashboard with an obvious Linux process name."""

from __future__ import annotations

import ctypes
import shutil
import sys
import tempfile
from pathlib import Path

from sheet import local_heatmap_loss_metadata_patch as _local_heatmap_loss_metadata_patch
import run_thog2_local_dashboard as _dashboard


_PROCESS_NAME = b"thog2-dashboard"
_PR_SET_NAME = 15
_EXTRA_ASSET_NAME = "dashboard_heatmap_loss_patch.js"


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


def _prepare_runtime_assets() -> tempfile.TemporaryDirectory[str]:
    source_root = Path(_dashboard._ASSET_ROOT)
    temporary = tempfile.TemporaryDirectory(prefix="thog2-dashboard-assets-")
    runtime_root = Path(temporary.name)
    for source in source_root.iterdir():
        if source.is_file():
            shutil.copy2(source, runtime_root / source.name)

    index_path = runtime_root / "index.html"
    index_html = index_path.read_text(encoding="utf-8")
    script_tag = f'  <script src="/assets/{_EXTRA_ASSET_NAME}" defer></script>\n'
    if script_tag not in index_html:
        index_html = index_html.replace("</head>", f"{script_tag}</head>", 1)
        index_path.write_text(index_html, encoding="utf-8")

    _dashboard._ASSET_ROOT = runtime_root
    _dashboard._ASSET_NAMES = frozenset((*_dashboard._ASSET_NAMES, _EXTRA_ASSET_NAME))
    return temporary


if __name__ == "__main__":
    _set_process_name()
    runtime_assets = _prepare_runtime_assets()
    try:
        raise SystemExit(_dashboard.main())
    finally:
        runtime_assets.cleanup()
# ^^^ THOG
