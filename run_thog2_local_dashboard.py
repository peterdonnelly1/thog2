# vvv THOG
"""Serve INSTRA through the preserved dashboard server with explicit Processing resource attribution assets."""

from __future__ import annotations

import atexit
import csv
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
    if cached is not None and now - float(cached[0]) < 30.0:
        return cached[1]

    _host, encoded = pair_key
    if not catalog.root.is_dir():
        state._instra_ncu_companion_cache = (now, None)
        return None

    # Search the full catalog root even when Instra itself was launched with
    # --run. The encoded artifact suffix discards almost every run before any
    # SQLite metadata is opened.
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


def _safe_int(value: Any) -> int:
    try:
        return int(round(float(str(value).strip())))
    except (TypeError, ValueError):
        return 0


def _hard_constraint_rows(
    processing_directory: Path,
    compatibility: Any,
) -> list[dict[str, Any]]:
    source = compatibility.get("rows", []) if isinstance(compatibility, dict) else compatibility
    if not isinstance(source, list) or not source:
        return []
    resource_path = processing_directory / "processing_ncu_kernel_resources.csv"
    if not resource_path.is_file():
        return []
    try:
        with resource_path.open(newline="") as handle:
            resources = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []
    if not resources:
        return []

    pair = source[0]
    main_family = str(pair.get("main_family", "")).upper()
    premat_family = str(pair.get("premat_family", "")).upper()
    main_layer = str(pair.get("main_layer", ""))
    premat_layer = str(pair.get("premat_layer", ""))

    def choose(role: str, family: str, layer: str):
        exact = [
            row for row in resources
            if str(row.get("role", "")).upper() == role
            and str(row.get("family", "")).upper() == family
            and str(row.get("layer", "")) == layer
        ]
        if exact:
            return exact[0]
        fallback = [row for row in resources if str(row.get("role", "")).upper() == role]
        return fallback[0] if fallback else None

    main = choose("MAIN", main_family, main_layer)
    premat = choose("PREMAT", premat_family, premat_layer)
    if main is None or premat is None:
        return []

    full_main_blocks = max(1, _safe_int(pair.get("main_theoretical_blocks_per_sm", 1)))
    definitions = (
        ("Registers", "registers_per_block", "sm_register_capacity", "registers"),
        ("Shared memory", "shared_mem_bytes", "sm_shared_mem_bytes", "bytes"),
        ("Warp slots", "warps_per_block", "sm_max_warps", "warps"),
        ("Thread slots", "threads_per_block", "sm_max_threads", "threads"),
        ("Block slots", None, "sm_max_blocks", "blocks"),
    )
    rows = []
    for label, footprint_key, capacity_key, unit in definitions:
        if footprint_key is None:
            main_per_block = 1
            premat_per_block = 1
        elif footprint_key == "shared_mem_bytes":
            main_per_block = _safe_int(pair.get("main_shared_mem_bytes", 0))
            premat_per_block = _safe_int(pair.get("premat_shared_mem_bytes", 0))
        else:
            main_per_block = _safe_int(pair.get(f"main_{footprint_key}", main.get(footprint_key, 0)))
            premat_per_block = _safe_int(pair.get(f"premat_{footprint_key}", premat.get(footprint_key, 0)))
        capacity = _safe_int(main.get(capacity_key, premat.get(capacity_key, 0)))
        pair_demand = main_per_block + premat_per_block
        full_main_plus_premat = full_main_blocks * main_per_block + premat_per_block
        rows.append({
            "resource": label,
            "unit": unit,
            "capacity": capacity,
            "main_per_block": main_per_block,
            "premat_per_block": premat_per_block,
            "main_full_residency_blocks": full_main_blocks,
            "pair_demand": pair_demand,
            "pair_capacity_pct": (100.0 * pair_demand / capacity) if capacity > 0 else None,
            "full_main_plus_one_premat": full_main_plus_premat,
            "full_main_plus_one_pct": (100.0 * full_main_plus_premat / capacity) if capacity > 0 else None,
            "limiting": str(pair.get("limiting_resource", "")).replace("_", " ").lower() in {
                label.lower(), unit.lower(), label.lower().replace(" slots", "s")
            },
        })
    return rows


def _attach_own_hard_constraints(state: Any, payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data")
    if not isinstance(data, dict):
        return payload
    compatibility = data.get("premat_compatibility")
    if not compatibility:
        return payload
    hard = _hard_constraint_rows(state.database_path.parent / "processing", compatibility)
    if not hard:
        return payload
    updated = dict(payload)
    updated_data = dict(data)
    updated_data["premat_hard_constraints"] = hard
    updated["data"] = updated_data
    return updated


def _processing_payload_with_ncu_companion(self):
    payload = _attach_own_hard_constraints(self, _original_processing_payload(self))
    if not payload.get("available") or not payload.get("trace_available"):
        return payload
    data = payload.get("data")
    if not isinstance(data, dict) or data.get("premat_compatibility"):
        return payload

    companion = _matching_ncu_companion(self)
    if companion is None:
        return payload
    _mtime, _created_at, companion_state, companion_status, compatibility_path = companion
    companion_payload = _attach_own_hard_constraints(
        companion_state,
        _original_processing_payload(companion_state),
    )
    companion_data = companion_payload.get("data") or {}
    compatibility = companion_data.get("premat_compatibility")
    if not compatibility:
        return payload

    merged_data = dict(data)
    merged_data["premat_compatibility"] = compatibility
    merged_data["premat_hard_constraints"] = list(companion_data.get("premat_hard_constraints") or [])
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
    "dashboard_processing_sep16_stability_polish.js",
    "dashboard_processing_sep16_companion_fallback.js",
    "dashboard_processing_sep16_quiescence.js",
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
