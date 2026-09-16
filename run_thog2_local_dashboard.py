# vvv THOG
"""Serve INSTRA through the preserved dashboard server with explicit Processing resource attribution assets."""

from __future__ import annotations

import atexit
from pathlib import Path
import shutil
import tempfile
from typing import Optional

import run_thog2_local_dashboard_base as _base
from run_thog2_local_dashboard_base import *  # noqa: F401,F403


_original_asset_root = Path(_base._ASSET_ROOT)
_overlay_asset_root = Path(tempfile.mkdtemp(prefix="thog2-instra-assets-"))
_dashboard_patch_names = (
    "dashboard_v058_repair_workspace_patch.js",
    "dashboard_sep16_workspace_ui_repair.js",
)
_processing_patch_names = (
    "dashboard_processing_resource_attribution.js",
    "dashboard_processing_ncu_2024_repair.js",
)

# vvv THOG build one explicit dashboard asset overlay at server startup; this avoids hidden import/read hooks and guarantees the resource view follows Processing
for _asset_name in (*sorted(_base._ASSET_NAMES), "index.html"):
    _source = _original_asset_root / _asset_name
    _payload = _source.read_bytes()
    if _asset_name == "dashboard.js":
        for _patch_name in _dashboard_patch_names:
            _payload += b"\n\n" + (_original_asset_root / _patch_name).read_bytes() + b"\n"
    if _asset_name == "dashboard_processing.js":
        for _patch_name in _processing_patch_names:
            _payload += b"\n\n" + (_original_asset_root / _patch_name).read_bytes() + b"\n"
    (_overlay_asset_root / _asset_name).write_bytes(_payload)
_base._ASSET_ROOT = _overlay_asset_root
# ^^^ THOG


def _cleanup_overlay() -> None:
    shutil.rmtree(_overlay_asset_root, ignore_errors=True)


atexit.register(_cleanup_overlay)


# vvv THOG preserve private attributes/functions from the established dashboard module for tests and callers that import them directly
def __getattr__(name: str):
    return getattr(_base, name)
# ^^^ THOG


def main(argv: Optional[list[str]] = None) -> int:
    return _base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
# ^^^ THOG
