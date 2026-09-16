# vvv THOG
"""Serve INSTRA through the preserved dashboard server with explicit Processing resource attribution assets."""

from __future__ import annotations

import atexit
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Optional

import run_thog2_local_dashboard_base as _base
from run_thog2_local_dashboard_base import *  # noqa: F401,F403


# vvv THOG expose the stored run configuration to the focused Runs-summary UI without changing the compact chart store
_original_run_status = _base.RunDashboardState.status


def _run_status_with_configuration(self):
    status = _original_run_status(self)
    metadata = self.reader.metadata()
    try:
        configuration = json.loads(metadata.get("config_json", "{}"))
    except json.JSONDecodeError:
        configuration = {}
    enriched = dict(status)
    enriched["configuration"] = configuration
    enriched["command"] = metadata.get("command", configuration.get("command", ""))
    return enriched


_base.RunDashboardState.status = _run_status_with_configuration
# ^^^ THOG


# vvv THOG Processing GPU Resource Compatibility pairs separate NSYS and NCU runs server-side.
# The artifact suffix after the triple underscore is the canonical encoded run
# configuration. Pair only identical suffixes on the same host, then choose the
# newest NCU compatibility artifact. This prevents unrelated profiler runs from
# being silently combined merely because their model shape looks similar.
_original_dashboard_state_for_path = _base.DashboardCatalog._state_for_path
_original_processing_payload = _base.RunDashboardState.processing


def _dashboard_state_for_path_with_catalog(self, path: Path):
    state = _original_dashboard_state_for_path(self, path)
    state._instra_dashboard_catalog = self
    return state


def _processing_pair_key(state: Any) -> tuple[str, str] | None:
    metadata = state.reader.metadata()
    try:
        configuration = json.loads(metadata.get("config_json", "{}"))
    except json.JSONDecodeError:
        configuration = {}
    artifact = str(
        metadata.get(
            "artifact_name",
            metadata.get("run_name", state.database_path.parent.name),
        )
    )
    _prefix, separator, encoded = artifact.partition("___")
    if not separator or not encoded:
        return None
    host = str(configuration.get("host_label", "")).strip()
    return host, encoded


def _matching_ncu_companion(state: Any):
    catalog = getattr(state, "_instra_dashboard_catalog", None)
    pair_key = _processing_pair_key(state)
    if catalog is None or pair_key is None:
        return None

    now = time.monotonic()
    cached = getattr(state, "_instra_ncu_companion_cache", None)
    if cached is not None and now - float(cached[0]) < 5.0:
        return cached[1]

    _host, encoded = pair_key
    if not catalog.root.is_dir():
        state._instra_ncu_companion_cache = (now, None)
        return None

    # Search the full catalog root even when Instra itself was launched with
    # --run.  The artifact-directory suffix lets us discard virtually every run
    # before opening its SQLite metadata.
    candidate_paths = tuple(catalog.root.glob(f"**/{_base.LOCAL_CHART_DATABASE_NAME}"))
    candidates = []
    for path in candidate_paths:
        if path.resolve() == state.database_path.resolve():
            continue
        artifact_hint = path.parent.parent.name if len(path.parents) >= 2 else ""
        if not str(artifact_hint).endswith(f"___{encoded}"):
            continue
        candidate = catalog._state_for_path(path)
        if _processing_pair_key(candidate) != pair_key:
            continue
        compatibility_path = (
            candidate.database_path.parent
            / "processing"
            / "processing_premat_compatibility.json"
        )
        if not compatibility_path.is_file():
            continue
        try:
            status = candidate.status()
        except Exception:
            continue
        candidates.append(
            (
                compatibility_path.stat().st_mtime_ns,
                str(status.get("created_at", "")),
                candidate,
                status,
                compatibility_path,
            )
        )
    result = max(candidates, key=lambda item: (item[0], item[1])) if candidates else None
    state._instra_ncu_companion_cache = (now, result)
    return result


def _processing_payload_with_ncu_companion(self):
    payload = _original_processing_payload(self)
    if not payload.get("available") or not payload.get("trace_available"):
        return payload
    data = payload.get("data")
    if not isinstance(data, dict) or data.get("premat_compatibility"):
        return payload

    companion = _matching_ncu_companion(self)
    if companion is None:
        return payload
    _mtime, _created_at, companion_state, companion_status, compatibility_path = companion
    companion_payload = _original_processing_payload(companion_state)
    companion_data = companion_payload.get("data") or {}
    compatibility = companion_data.get("premat_compatibility")
    if not compatibility:
        return payload

    merged_data = dict(data)
    merged_data["premat_compatibility"] = compatibility
    merged_data["premat_compatibility_files"] = dict(
        companion_data.get("premat_compatibility_files") or {}
    )
    merged_data["premat_compatibility_source"] = {
        "dashboard_run_id": str(companion_status.get("dashboard_run_id", "")),
        "artifact_name": str(companion_status.get("artifact_name", "")),
        "created_at": str(companion_status.get("created_at", "")),
        "host_label": str(companion_status.get("host_label", "")),
        "pair_key": pair_key_text(_processing_pair_key(self)),
    }
    result = dict(payload)
    result["data"] = merged_data
    result["revision"] = (
        f"{payload.get('revision', '')}:ncu-companion:"
        f"{companion_status.get('dashboard_run_id', '')}:"
        f"{compatibility_path.stat().st_mtime_ns}:{compatibility_path.stat().st_size}"
    )
    return result


def pair_key_text(value: tuple[str, str] | None) -> str:
    if value is None:
        return ""
    host, encoded = value
    return f"{host}|{encoded}"


_base.DashboardCatalog._state_for_path = _dashboard_state_for_path_with_catalog
_base.RunDashboardState.processing = _processing_payload_with_ncu_companion
# ^^^ THOG


_original_asset_root = Path(_base._ASSET_ROOT)
_overlay_asset_root = Path(tempfile.mkdtemp(prefix="thog2-instra-assets-"))
_dashboard_patch_names = (
    "dashboard_workspace_only.js",
    "dashboard_runs_table_restore.js",
    "dashboard_sep16_workspace_ui_repair.js",
)
_processing_patch_names = (
    "dashboard_processing_resource_attribution.js",
    "dashboard_processing_ncu_2024_repair.js",
    "dashboard_processing_compatibility_classification_patch.js",
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
