# vvv THOG
"""Serve INSTRA through the preserved dashboard server with explicit Processing resource attribution assets."""

from __future__ import annotations

import atexit
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Mapping, Optional
import zipfile

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
# viable NCU capture nearest in artifact timestamp to the selected NSYS capture.
_original_dashboard_state_for_path = _base.DashboardCatalog._state_for_path
_original_processing_payload = _base.RunDashboardState.processing


def _dashboard_state_for_path_with_catalog(self, path: Path):
    state = _original_dashboard_state_for_path(self, path)
    state._instra_dashboard_catalog = self
    return state


def _state_artifact_name(state: Any) -> str:
    metadata = state.reader.metadata()
    return str(
        metadata.get(
            "artifact_name",
            metadata.get("run_name", state.database_path.parent.name),
        )
    )


def _processing_pair_key(state: Any) -> tuple[str, str] | None:
    metadata = state.reader.metadata()
    try:
        configuration = json.loads(metadata.get("config_json", "{}"))
    except json.JSONDecodeError:
        configuration = {}
    artifact = _state_artifact_name(state)
    _prefix, separator, encoded = artifact.partition("___")
    if not separator or not encoded:
        return None
    host = str(configuration.get("host_label", "")).strip()
    return host, encoded


def _artifact_timestamp_seconds(artifact: Any, fallback: float = 0.0) -> float:
    prefix = str(artifact or "").split("_", 1)[0]
    try:
        parsed = datetime.strptime(prefix, "%y%m%d-%H%M").replace(tzinfo=timezone.utc)
        return float(parsed.timestamp())
    except (TypeError, ValueError, OverflowError):
        return float(fallback)


def _status_timestamp_seconds(status: Any, artifact: Any, fallback: float = 0.0) -> float:
    created_at = str((status or {}).get("created_at", "")).strip() if isinstance(status, dict) else ""
    if created_at:
        try:
            return datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    return _artifact_timestamp_seconds(artifact, fallback)


def _is_ncu_artifact(artifact: Any) -> bool:
    text = str(artifact or "").upper()
    return "_NCU_" in text or "NCU_PREMAT" in text


def _compatibility_file_has_rows(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    rows = payload.get("rows", []) if isinstance(payload, dict) else payload
    return isinstance(rows, list) and bool(rows)


def _matching_ncu_companion(
    state: Any,
    *,
    excluded_ncu_run_ids: Optional[set[str]] = None,
    preferred_ncu_run_id: Optional[str] = None,
):
    catalog = getattr(state, "_instra_dashboard_catalog", None)
    pair_key = _processing_pair_key(state)
    if catalog is None or pair_key is None:
        return None

    excluded = frozenset(str(value) for value in (excluded_ncu_run_ids or set()) if value)
    preferred = str(preferred_ncu_run_id or "")
    selection_key = (excluded, preferred)
    now = time.monotonic()
    cached = getattr(state, "_instra_ncu_companion_cache", None)
    if (
        cached is not None
        and len(cached) >= 3
        and cached[1] == selection_key
        and now - float(cached[0]) < 2.0
    ):
        return cached[2]

    selected_artifact = _state_artifact_name(state)
    if _is_ncu_artifact(selected_artifact):
        state._instra_ncu_companion_cache = (now, selection_key, None)
        state._instra_ncu_companion_diagnostics = {}
        return None
    selected_fallback = state.database_path.stat().st_mtime if state.database_path.exists() else 0.0
    try:
        selected_status = state.status()
    except Exception:
        selected_status = {}
    selected_time = _status_timestamp_seconds(selected_status, selected_artifact, selected_fallback)

    _host, encoded = pair_key
    if not catalog.root.is_dir():
        state._instra_ncu_companion_cache = (now, selection_key, None)
        state._instra_ncu_companion_diagnostics = {}
        return None

    viable = []
    invalid = []
    for path in catalog.root.glob(f"**/{_base.LOCAL_CHART_DATABASE_NAME}"):
        if path.resolve() == state.database_path.resolve():
            continue
        candidate = catalog._state_for_path(path)
        candidate_artifact = _state_artifact_name(candidate)
        if not _is_ncu_artifact(candidate_artifact):
            continue
        if _processing_pair_key(candidate) != pair_key:
            continue

        try:
            status = candidate.status()
        except Exception:
            status = {}
        candidate_fallback = path.stat().st_mtime if path.exists() else 0.0
        candidate_time = _status_timestamp_seconds(status, candidate_artifact, candidate_fallback)
        # NCU is the second half of a pair: never consume an older capture.
        if candidate_time <= selected_time:
            continue
        distance = candidate_time - selected_time
        compatibility_path = (
            candidate.database_path.parent
            / "processing"
            / "processing_premat_compatibility.json"
        )
        if not compatibility_path.is_file() or not _compatibility_file_has_rows(compatibility_path):
            invalid.append((distance, -candidate_time, candidate_artifact))
            continue

        if not status:
            invalid.append((distance, -candidate_time, candidate_artifact))
            continue

        candidate_run_id = str(status.get("dashboard_run_id", ""))
        if candidate_run_id in excluded and candidate_run_id != preferred:
            continue

        viable.append(
            (
                distance,
                candidate_time,
                candidate,
                status,
                compatibility_path,
            )
        )

    if preferred:
        preferred_candidates = [
            item for item in viable
            if str(item[3].get("dashboard_run_id", "")) == preferred
        ]
        result = preferred_candidates[0] if preferred_candidates else None
    else:
        result = min(viable, key=lambda item: (item[0], item[1])) if viable else None
    chosen_distance = result[0] if result is not None else float("inf")
    skipped = sorted(
        (item for item in invalid if item[0] < chosen_distance),
        key=lambda item: (item[0], item[1]),
    )
    state._instra_ncu_companion_diagnostics = {
        "selected_artifact": selected_artifact,
        "distance_minutes": (float(result[0]) / 60.0) if result is not None else None,
        "skipped_closer_invalid_count": len(skipped),
        "skipped_closer_invalid_artifacts": [item[2] for item in skipped],
        "viable_candidate_count": len(viable),
        "matching_invalid_candidate_count": len(invalid),
        "excluded_claimed_candidate_count": len(excluded),
        "preferred_ncu_run_id": preferred,
    }
    state._instra_ncu_companion_cache = (now, selection_key, result)
    return result


def _safe_int(value: Any) -> int:
    try:
        return int(round(float(str(value).strip())))
    except (TypeError, ValueError):
        return 0


# vvv THOG derive conservative MAIN/PREMAT contention evidence from correlated NSYS scheduling and NCU residency limits
_CONTENTION_RESOURCE = {
    "registers": ("R", "registers", "sm_register_capacity", "registers_per_block"),
    "shared": ("S", "shared memory", "sm_shared_mem_bytes", "shared_mem_bytes"),
    "shared_memory": ("S", "shared memory", "sm_shared_mem_bytes", "shared_mem_bytes"),
    "warps": ("W", "warp slots", "sm_max_warps", "warps_per_block"),
    "threads": ("W", "thread slots", "sm_max_threads", "threads_per_block"),
    "blocks": ("W", "resident block slots", "sm_max_blocks", "blocks"),
}


def _compatibility_rows(compatibility: Any) -> list[dict[str, Any]]:
    rows = compatibility.get("rows", []) if isinstance(compatibility, dict) else compatibility
    return [dict(row) for row in rows] if isinstance(rows, list) else []


def _compatibility_with_capacities(state: Any, compatibility: Any) -> Any:
    rows = _compatibility_rows(compatibility)
    resource_path = state.database_path.parent / "processing" / "processing_ncu_kernel_resources.csv"
    if not rows or not resource_path.is_file():
        return compatibility
    try:
        with resource_path.open(newline="") as handle:
            resources = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return compatibility
    capacity_keys = (
        "sm_register_capacity", "sm_shared_mem_bytes", "sm_max_warps",
        "sm_max_threads", "sm_max_blocks",
    )
    for row in rows:
        kernel_names = {
            str(row.get("main_cuda_kernel_name", "")),
            str(row.get("premat_cuda_kernel_name", "")),
        }
        resource = next(
            (
                item for item in resources
                if str(item.get("cuda_kernel_name", item.get("kernel_name", ""))) in kernel_names
            ),
            resources[0] if resources else None,
        )
        if resource is None:
            continue
        for key in capacity_keys:
            if row.get(key, "") in ("", None):
                row[key] = _safe_int(resource.get(key, 0))
    if isinstance(compatibility, dict):
        result = dict(compatibility)
        result["rows"] = rows
        return result
    return rows


def _compatibility_match(
    rows: list[dict[str, Any]],
    main: Mapping[str, Any],
    premat: Mapping[str, Any],
) -> dict[str, Any] | None:
    candidates = [
        row for row in rows
        if str(row.get("main_family", "")).upper() == str(main.get("family", "")).upper()
        and str(row.get("premat_family", "")).upper() == str(premat.get("family", "")).upper()
    ]
    for field, interval in (
        ("main_layer", main),
        ("premat_layer", premat),
    ):
        exact = [
            row for row in candidates
            if row.get(field, "") in ("", None)
            or str(row.get(field)) == str(interval.get("layer", ""))
        ]
        if exact:
            candidates = exact
    stage = _safe_int(premat.get("premat_stage_index", 0))
    if stage:
        exact_stage = [row for row in candidates if _safe_int(row.get("premat_stage_index", 0)) == stage]
        if exact_stage:
            candidates = exact_stage
    return candidates[0] if candidates else None


def _processing_contention_intervals(
    data: Mapping[str, Any],
    compatibility: Any = None,
) -> list[dict[str, Any]]:
    intervals = [dict(row) for row in data.get("intervals", []) if isinstance(row, Mapping)]
    compatibility_rows = _compatibility_rows(compatibility)
    premat_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in intervals:
        if str(row.get("owner", "")) != "PREMAT" or str(row.get("operation", "")) != "materialize":
            continue
        group_key = (
            row.get("op_id", ""), row.get("family", ""), row.get("layer", ""),
            row.get("context_id", ""), row.get("stream", ""),
        )
        premat_groups.setdefault(group_key, []).append(row)
    for group in premat_groups.values():
        ordered = sorted(group, key=lambda item: float(item.get("start_us", 0.0)))
        for index, row in enumerate(ordered, start=1):
            row.setdefault("premat_stage_index", index)
            row.setdefault("premat_stage_count", len(ordered))
    # Older normalized captures predate launch-correlation fields.  They can
    # still expose a conservative, explicitly inferred gap between consecutive
    # kernels in one PREMAT operation; such rows never become solid exclusions.
    stream_previous: dict[tuple[str, str], dict[str, Any]] = {}
    for row in sorted(intervals, key=lambda item: float(item.get("start_us", 0.0))):
        stream_key = (str(row.get("context_id", "")), str(row.get("stream", "")))
        previous = stream_previous.get(stream_key)
        if row.get("eligible_lower_bound_us", "") in ("", None) and previous is not None:
            same_premat_operation = (
                str(row.get("owner", "")) == "PREMAT"
                and str(previous.get("owner", "")) == "PREMAT"
                and str(row.get("operation", "")) == "materialize"
                and str(previous.get("operation", "")) == "materialize"
                and (
                    (row.get("op_id", "") not in ("", None) and row.get("op_id") == previous.get("op_id"))
                    or (
                        str(row.get("family", "")) == str(previous.get("family", ""))
                        and str(row.get("layer", "")) == str(previous.get("layer", ""))
                    )
                )
            )
            if same_premat_operation:
                row["eligible_lower_bound_us"] = float(previous.get("end_us", 0.0))
                row["admission_wait_lower_bound_us"] = max(
                    0.0, float(row.get("start_us", 0.0)) - float(previous.get("end_us", 0.0))
                )
                row["_contention_eligibility_basis"] = "same-stream predecessor only (legacy capture)"
        row.setdefault(
            "_contention_eligibility_basis",
            "correlated launch completion + same-stream predecessor"
            if row.get("submitted_us", "") not in ("", None)
            else "same-stream predecessor only",
        )
        stream_previous[stream_key] = row
    result: list[dict[str, Any]] = []
    for victim in intervals:
        victim_owner = str(victim.get("owner", "")).upper()
        if victim_owner not in {"MAIN", "PREMAT"}:
            continue
        try:
            eligible = float(victim["eligible_lower_bound_us"])
            start = float(victim["start_us"])
        except (KeyError, TypeError, ValueError):
            continue
        if start - eligible < 1.0:
            continue
        blocker_owner = "PREMAT" if victim_owner == "MAIN" else "MAIN"
        blockers = [
            row for row in intervals
            if str(row.get("owner", "")).upper() == blocker_owner
            and float(row.get("end_us", 0.0)) > eligible
            and float(row.get("start_us", 0.0)) < start
        ]
        for blocker in blockers:
            main, premat = (victim, blocker) if victim_owner == "MAIN" else (blocker, victim)
            match = _compatibility_match(compatibility_rows, main, premat)
            if match is None:
                continue
            limiter = str(match.get("limiting_resource", "")).strip().lower().replace(" ", "_")
            resource = _CONTENTION_RESOURCE.get(limiter)
            if resource is None:
                continue
            code, cause, capacity_key, footprint_key = resource
            capacity = _safe_int(match.get(capacity_key, 0))
            if footprint_key == "blocks":
                main_per_block = premat_per_block = 1
            else:
                main_per_block = _safe_int(match.get(f"main_{footprint_key}", 0))
                premat_per_block = _safe_int(match.get(f"premat_{footprint_key}", 0))
            if victim_owner == "PREMAT":
                blocker_blocks = max(1, _safe_int(match.get("main_theoretical_blocks_per_sm", 1)))
                available = max(0, capacity - blocker_blocks * main_per_block)
                required = premat_per_block
                full_residency_exclusion = _safe_int(
                    match.get("premat_blocks_with_full_main_residency", 0)
                ) == 0
            else:
                available = max(0, capacity - premat_per_block)
                required = main_per_block
                full_residency_exclusion = False
            pair_exclusion = not bool(match.get("pair_can_co_reside", False))
            correlated_eligibility = str(victim.get("_contention_eligibility_basis", "")).startswith(
                "correlated launch"
            )
            hard = (pair_exclusion or full_residency_exclusion) and correlated_eligibility
            result.append({
                "start_us": max(eligible, float(blocker.get("start_us", eligible))),
                "end_us": min(start, float(blocker.get("end_us", start))),
                "victim_owner": victim_owner,
                "victim_family": str(victim.get("family", "")),
                "victim_layer": victim.get("layer", ""),
                "victim_kernel_name": str(victim.get("kernel_name", "")),
                "victim_stage_index": victim.get("premat_stage_index", ""),
                "victim_stage_count": victim.get("premat_stage_count", ""),
                "blocker_owner": blocker_owner,
                "blocker_family": str(blocker.get("family", "")),
                "blocker_layer": blocker.get("layer", ""),
                "blocker_kernel_name": str(blocker.get("kernel_name", "")),
                "admission_wait_us": start - eligible,
                "resource_code": code,
                "resource": cause,
                "available": available,
                "required": required,
                "capacity": capacity,
                "confidence": "hard exclusion" if hard else "inferred admission delay",
                "hard_exclusion": hard,
                "evidence": (
                    "NSYS launch correlation + same-stream predecessor + NCU SM residency"
                    if hard else f"{victim.get('_contention_eligibility_basis')}; NCU structural limit"
                ),
            })

    pressure_specs = (
        ("tensor_active_pct", "T", "tensor/issue pipelines", 75.0),
        ("sm_issue_pct", "T", "tensor/issue pipelines", 90.0),
        ("dram_read_pct", "D", "DRAM bandwidth", 75.0),
        ("dram_write_pct", "D", "DRAM bandwidth", 75.0),
        ("l2_active_pct", "L2", "L2 bandwidth", 75.0),
    )
    for sample in data.get("stream_resources", []) or []:
        if str(sample.get("attribution_state", "")) != "MAIN_PREMAT_OVERLAP":
            continue
        selected = None
        for metric, code, cause, threshold in pressure_specs:
            try:
                value = float(sample.get(metric, ""))
            except (TypeError, ValueError):
                continue
            if value >= threshold and (selected is None or value > selected[-1]):
                selected = (metric, code, cause, threshold, value)
        if selected is None:
            continue
        metric, code, cause, threshold, value = selected
        result.append({
            "start_us": float(sample.get("sample_start_us", sample.get("time_us", 0.0))),
            "end_us": float(sample.get("sample_end_us", sample.get("time_us", 0.0))),
            "victim_owner": "BOTH",
            "victim_family": str(sample.get("main_families", "")),
            "victim_layer": str(sample.get("main_layers", "")),
            "victim_kernel_name": "",
            "blocker_owner": "BOTH",
            "blocker_family": str(sample.get("premat_families", "")),
            "blocker_layer": str(sample.get("premat_layers", "")),
            "blocker_kernel_name": "",
            "admission_wait_us": "",
            "resource_code": code,
            "resource": cause,
            "available": "",
            "required": "",
            "capacity": "",
            "pressure_metric": metric,
            "pressure_value_pct": value,
            "pressure_threshold_pct": threshold,
            "confidence": "concurrent pressure; causation not established",
            "hard_exclusion": False,
            "evidence": "device-wide NSYS sample during exact MAIN/PREMAT overlap",
        })
    return [row for row in result if float(row["end_us"]) > float(row["start_us"])]
# ^^^ THOG


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

    compatibility_rank = {"RED": 0, "ORANGE": 1, "YELLOW": 2, "GREEN": 3}
    pair = min(
        source,
        key=lambda row: (
            compatibility_rank.get(str(row.get("compatibility_class", "")).upper(), 99),
            _safe_int(row.get("premat_blocks_with_full_main_residency", 0)),
            -_safe_int(row.get("premat_registers_per_block", 0)),
        ),
    )
    main_family = str(pair.get("main_family", "")).upper()
    premat_family = str(pair.get("premat_family", "")).upper()
    main_layer = str(pair.get("main_layer", ""))
    premat_layer = str(pair.get("premat_layer", ""))

    def choose(role: str, family: str, layer: str, kernel_name: str):
        exact = [
            row for row in resources
            if str(row.get("role", "")).upper() == role
            and str(row.get("family", "")).upper() == family
            and str(row.get("layer", "")) == layer
        ]
        kernel_exact = [
            row for row in exact
            if str(row.get("cuda_kernel_name", row.get("kernel_name", ""))) == kernel_name
        ]
        if kernel_exact:
            return kernel_exact[0]
        if exact:
            return exact[0]
        fallback = [row for row in resources if str(row.get("role", "")).upper() == role]
        return fallback[0] if fallback else None

    main = choose("MAIN", main_family, main_layer, str(pair.get("main_cuda_kernel_name", "")))
    premat = choose("PREMAT", premat_family, premat_layer, str(pair.get("premat_cuda_kernel_name", "")))
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
            "premat_stage_index": _safe_int(pair.get("premat_stage_index", 1)),
            "premat_stage_count": _safe_int(pair.get("premat_stage_count", 1)),
            "premat_kernel_name": str(pair.get("premat_cuda_kernel_name", "")),
            "compatibility_class": str(pair.get("compatibility_class", "")),
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


def _ncu_processing_download_files(state: Any) -> dict[str, str]:
    processing_directory = state.database_path.parent / "processing"
    files: dict[str, str] = {}
    raw_reports = sorted(
        (
            path
            for pattern in ("processing_ncu_trace.ncu-rep", "processing_ncu_trace.ncu-repz")
            for path in processing_directory.glob(pattern)
            if path.is_file()
        ),
        key=lambda path: (path.suffix != ".ncu-rep", path.name),
    )
    if raw_reports:
        files["raw_ncu"] = raw_reports[0].name
    for key, filename in (
        ("ncu_raw_csv", "processing_ncu_raw.csv"),
        ("ncu_semantic_csv", "processing_ncu_semantic.csv"),
    ):
        path = processing_directory / filename
        if path.is_file():
            files[key] = filename
    return files


_LIFECYCLE_EVENTS = frozenset({
    "admission_considered", "admission_deferred", "materialising", "available",
    "critical_path_wait", "materialising_on_critical_path", "available_on_critical_path",
    "consuming", "consumed", "pass_end_release", "materialisation_failed",
    "main_materialisation_failed",
})


def _processing_capture_snapshot(state: Any, data: dict[str, Any]) -> dict[str, Any] | None:
    capture = (data.get("metadata") or {}).get("capture") or {}
    target_update = _safe_int(capture.get("optimizer_update"))
    snapshots = state.reader.premat_snapshots()
    if not snapshots:
        return None
    if target_update:
        matching = [row for row in snapshots if _safe_int(row.get("optimizer_update")) == target_update]
        if matching:
            return dict(matching[-1])
    return dict(snapshots[-1])


def _processing_lifecycle_rows(
    snapshot: dict[str, Any] | None,
    capture_host_start_ns: Any = None,
) -> list[dict[str, Any]]:
    if not snapshot:
        return []
    rows = []
    for source in snapshot.get("events", []) or []:
        event = str(source.get("event", source.get("event_type", "")))
        if event not in _LIFECYCLE_EVENTS and not event.startswith("deadline_"):
            continue
        exact = source.get("processing_capture_elapsed_ms")
        timing_basis = "capture_relative_host"
        if exact is None and capture_host_start_ns is not None:
            try:
                exact = max(
                    0.0,
                    (float(source["host_time_ns"]) - float(capture_host_start_ns))
                    / 1_000_000.0,
                )
            except (KeyError, TypeError, ValueError):
                exact = None
        fallback = source.get("elapsed_ms")
        if exact is None:
            timing_basis = "pass_relative_legacy"
        try:
            capture_ms = float(exact if exact is not None else fallback)
        except (TypeError, ValueError):
            continue
        candidate_sequence = source.get("candidate_sequence", "")
        layer = source.get("layer_index", "")
        family = str(source.get("family", ""))
        job_id = (
            f"p{source.get('pass_sequence', snapshot.get('pass_sequence', ''))}:c{candidate_sequence}"
            if candidate_sequence not in (None, "")
            else f"p{source.get('pass_sequence', snapshot.get('pass_sequence', ''))}:l{layer}:{family}"
        )
        rows.append({
            "job_id": job_id,
            "sequence": source.get("sequence", ""),
            "pass_sequence": source.get("pass_sequence", snapshot.get("pass_sequence", "")),
            "candidate_sequence": candidate_sequence,
            "capture_time_ms": capture_ms,
            "timing_basis": timing_basis,
            "event": event,
            "layer": layer,
            "family": family,
            "owner": source.get("owner", ""),
            "state": source.get("state", source.get("new_state", "")),
            "decision": source.get("decision", ""),
            "outcome": source.get("outcome", source.get("final_outcome", "")),
            "reason": source.get("reason", source.get("admission_reason", "")),
            "critical_path_miss": source.get("critical_path_miss", ""),
            "queue_depth": source.get("queue_depth", ""),
            "cumulative_charged_bytes": source.get("cumulative_charged_bytes", ""),
            "process_allocated_bytes": source.get("process_allocated_bytes", ""),
            "process_reserved_bytes": source.get("process_reserved_bytes", ""),
            "device_free_bytes": source.get("device_free_bytes", ""),
            "predicted_retained_bytes": source.get("predicted_retained_bytes", ""),
            "predicted_materialisation_peak_bytes": source.get("predicted_materialisation_peak_bytes", ""),
            "detail_json": json.dumps(source.get("detail", {}), separators=(",", ":"), sort_keys=True),
        })
    return rows


def _processing_lifecycle_summary(
    rows: list[dict[str, Any]],
    interval_rows: Any = (),
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["job_id"]), []).append(row)
    event_columns = {
        "requested_ms": {"admission_considered"},
        "admitted_ms": {"admission_considered"},
        "submitted_ms": {"materialising"},
        "available_ms": {"available", "available_on_critical_path"},
        "deadline_ms": set(),
        "wait_ms": {"critical_path_wait"},
        "consuming_ms": {"consuming"},
        "completed_ms": {"consumed"},
        "discarded_ms": {"pass_end_release"},
    }
    result = []
    for job_id, group in grouped.items():
        ordered = sorted(group, key=lambda row: (float(row["capture_time_ms"]), _safe_int(row["sequence"])))
        first = ordered[0]
        summary = {
            "job_id": job_id,
            "pass_sequence": first["pass_sequence"],
            "candidate_sequence": first["candidate_sequence"],
            "layer": first["layer"],
            "family": first["family"],
            "timing_basis": first["timing_basis"],
            "final_owner": next((row["owner"] for row in reversed(ordered) if row["owner"]), ""),
            "final_outcome": next((row["outcome"] for row in reversed(ordered) if row["outcome"]), ""),
            "critical_path_miss": any(str(row["critical_path_miss"]).lower() == "true" for row in ordered),
        }
        for column, names in event_columns.items():
            candidates = [
                float(row["capture_time_ms"]) for row in ordered
                if row["event"] in names or (column == "deadline_ms" and str(row["event"]).startswith("deadline_"))
            ]
            if column == "admitted_ms":
                candidates = [
                    float(row["capture_time_ms"]) for row in ordered
                    if row["event"] == "admission_considered" and str(row["decision"]) == "admit"
                ]
            summary[column] = min(candidates) if candidates else ""
        gpu_intervals = [
            interval for interval in (interval_rows or [])
            if str(interval.get("operation", "")) == "materialize"
            and str(interval.get("family", "")) == str(summary["family"])
            and str(interval.get("layer", "")) == str(summary["layer"])
        ]
        premat_intervals = [row for row in gpu_intervals if str(row.get("owner", "")) == "PREMAT"]
        chosen_intervals = premat_intervals or gpu_intervals
        summary["gpu_start_ms"] = (
            min(float(row["start_us"]) for row in chosen_intervals) / 1000.0
            if chosen_intervals else ""
        )
        summary["gpu_end_ms"] = (
            max(float(row["end_us"]) for row in chosen_intervals) / 1000.0
            if chosen_intervals else ""
        )
        result.append(summary)
    return result


def _csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes_if_changed(path: Path, payload: bytes) -> None:
    try:
        if path.is_file() and path.read_bytes() == payload:
            return
    except OSError:
        pass
    path.write_bytes(payload)


def _materialize_paired_analysis(
    state: Any,
    companion_state: Any,
    data: dict[str, Any],
    selected_files: dict[str, str],
    companion_files: dict[str, str],
    compatibility: Any = None,
) -> dict[str, str]:
    processing_directory = state.database_path.parent / "processing"
    processing_directory.mkdir(parents=True, exist_ok=True)
    snapshot = _processing_capture_snapshot(state, data)
    lifecycle_rows = _processing_lifecycle_rows(
        snapshot,
        ((data.get("metadata") or {}).get("capture") or {}).get("host_start_ns"),
    )
    lifecycle_summary = _processing_lifecycle_summary(lifecycle_rows, data.get("intervals", []))
    contention_rows = _processing_contention_intervals(data, compatibility)
    lifecycle_name = "processing_premat_lifecycle_events.csv"
    lifecycle_summary_name = "processing_premat_lifecycle_summary.csv"
    contention_name = "processing_contention_intervals.csv"
    lifecycle_fields = [
        "job_id", "sequence", "pass_sequence", "candidate_sequence", "capture_time_ms",
        "timing_basis", "event", "layer", "family", "owner", "state", "decision",
        "outcome", "reason", "critical_path_miss", "queue_depth",
        "cumulative_charged_bytes", "process_allocated_bytes", "process_reserved_bytes",
        "device_free_bytes", "predicted_retained_bytes",
        "predicted_materialisation_peak_bytes", "detail_json",
    ]
    lifecycle_summary_fields = [
        "job_id", "pass_sequence", "candidate_sequence", "layer", "family", "timing_basis",
        "final_owner", "final_outcome", "critical_path_miss", "requested_ms", "admitted_ms",
        "submitted_ms", "gpu_start_ms", "gpu_end_ms", "available_ms", "deadline_ms", "wait_ms", "consuming_ms",
        "completed_ms", "discarded_ms",
    ]
    contention_fields = [
        "start_us", "end_us", "victim_owner", "victim_family", "victim_layer",
        "victim_kernel_name", "victim_stage_index", "victim_stage_count", "blocker_owner",
        "blocker_family", "blocker_layer", "blocker_kernel_name", "admission_wait_us",
        "resource_code", "resource", "available", "required", "capacity", "pressure_metric",
        "pressure_value_pct", "pressure_threshold_pct", "confidence", "hard_exclusion", "evidence",
    ]
    generated_files = {
        "lifecycle_events": lifecycle_name,
        "lifecycle_summary": lifecycle_summary_name,
        "contention_intervals": contention_name,
        "pair_manifest": "processing_pair_manifest.json",
        "most": "processing_most.zip",
        "everything": "processing_everything.zip",
    }
    lifecycle_payload = _csv_bytes(lifecycle_rows, lifecycle_fields)
    lifecycle_summary_payload = _csv_bytes(lifecycle_summary, lifecycle_summary_fields)
    contention_payload = _csv_bytes(contention_rows, contention_fields)

    candidate_files: list[tuple[str, Path, str]] = []
    candidate_archive_names: set[str] = set()
    for namespace, owner_state, files in (
        ("nsys", state, selected_files),
        ("ncu", companion_state, companion_files),
    ):
        directory = owner_state.database_path.parent / "processing"
        for key, filename in sorted(files.items()):
            if key in {
                "everything", "paired_analysis", "pair_manifest",
                "lifecycle_events", "lifecycle_summary", "contention_intervals",
            }:
                continue
            path = directory / str(filename)
            archive_name = f"{namespace}/{path.name}"
            if path.is_file() and archive_name not in candidate_archive_names:
                candidate_files.append((archive_name, path, key))
                candidate_archive_names.add(archive_name)
    input_signature = (
        hashlib.sha256(lifecycle_payload).hexdigest(),
        hashlib.sha256(lifecycle_summary_payload).hexdigest(),
        hashlib.sha256(contention_payload).hexdigest(),
        tuple(
            (archive_name, path.stat().st_mtime_ns, path.stat().st_size)
            for archive_name, path, _key in candidate_files
        ),
    )
    if (
        getattr(state, "_instra_paired_analysis_input_signature", None) == input_signature
        and all((processing_directory / filename).is_file() for filename in generated_files.values())
    ):
        return generated_files
    _write_bytes_if_changed(
        processing_directory / lifecycle_name,
        lifecycle_payload,
    )
    _write_bytes_if_changed(
        processing_directory / lifecycle_summary_name,
        lifecycle_summary_payload,
    )
    _write_bytes_if_changed(processing_directory / contention_name, contention_payload)

    included: list[tuple[str, Path, str]] = list(candidate_files)
    included.extend((
        (f"nsys/{lifecycle_name}", processing_directory / lifecycle_name, "lifecycle_events"),
        (f"nsys/{lifecycle_summary_name}", processing_directory / lifecycle_summary_name, "lifecycle_summary"),
        (f"nsys/{contention_name}", processing_directory / contention_name, "contention_intervals"),
    ))
    manifest = {
        "schema_version": 1,
        "pair_key": pair_key_text(_processing_pair_key(state)),
        "nsys_artifact": _state_artifact_name(state),
        "ncu_artifact": _state_artifact_name(companion_state),
        "lifecycle_timing_basis": (
            lifecycle_rows[0]["timing_basis"] if lifecycle_rows else "unavailable"
        ),
        "files": [
            {
                "path": archive_name,
                "kind": key,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for archive_name, path, key in included
        ],
    }
    manifest_name = "processing_pair_manifest.json"
    manifest_path = processing_directory / manifest_name
    _write_bytes_if_changed(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
    )
    bundle_name = "processing_everything.zip"
    bundle_path = processing_directory / bundle_name
    most_name = "processing_most.zip"
    most_path = processing_directory / most_name
    signature = tuple((entry["path"], entry["sha256"]) for entry in manifest["files"])
    if (
        getattr(state, "_instra_paired_analysis_signature", None) != signature
        or not bundle_path.is_file()
        or not most_path.is_file()
    ):
        with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(manifest_path, arcname=manifest_name)
            for archive_name, path, _key in included:
                already_compressed = path.suffix.lower() in {
                    ".zip", ".nsys-rep", ".ncu-rep", ".ncu-repz",
                }
                archive.write(
                    path,
                    arcname=archive_name,
                    compress_type=zipfile.ZIP_STORED if already_compressed else zipfile.ZIP_DEFLATED,
                )
        with zipfile.ZipFile(most_path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(manifest_path, arcname=manifest_name)
            for archive_name, path, _key in included:
                if path.name in {
                    "processing_ncu_trace.ncu-rep",
                    "processing_trace.nsys-rep",
                }:
                    continue
                already_compressed = path.suffix.lower() in {".zip", ".ncu-repz"}
                archive.write(
                    path,
                    arcname=archive_name,
                    compress_type=zipfile.ZIP_STORED if already_compressed else zipfile.ZIP_DEFLATED,
                )
        state._instra_paired_analysis_signature = signature
    state._instra_paired_analysis_input_signature = input_signature
    return generated_files


def _processing_payload_with_ncu_companion(
    self,
    *,
    excluded_ncu_run_ids: Optional[set[str]] = None,
    preferred_ncu_run_id: Optional[str] = None,
):
    payload = _attach_own_hard_constraints(self, _original_processing_payload(self))
    if not payload.get("available") or not payload.get("trace_available"):
        return payload
    data = payload.get("data")
    if not isinstance(data, dict):
        return payload

    companion = _matching_ncu_companion(
        self,
        excluded_ncu_run_ids=excluded_ncu_run_ids,
        preferred_ncu_run_id=preferred_ncu_run_id,
    )
    if companion is None:
        return payload
    _distance, _neg_created_at, companion_state, companion_status, compatibility_path = companion
    companion_payload = _attach_own_hard_constraints(
        companion_state,
        _original_processing_payload(companion_state),
    )
    companion_data = companion_payload.get("data") or {}
    compatibility = companion_data.get("premat_compatibility")
    if not compatibility:
        return payload
    compatibility = _compatibility_with_capacities(companion_state, compatibility)

    diagnostics = dict(getattr(self, "_instra_ncu_companion_diagnostics", {}) or {})
    selected_status = self.status()
    selected_metadata = data.get("metadata") or {}
    companion_metadata = companion_data.get("metadata") or {}
    selected_files = dict(selected_metadata.get("files", {}) or {})
    companion_files = dict(companion_metadata.get("files", {}) or {})
    companion_files.update(dict(companion_data.get("premat_compatibility_files", {}) or {}))
    companion_files.update(_ncu_processing_download_files(companion_state))
    generated_files = _materialize_paired_analysis(
        self,
        companion_state,
        data,
        selected_files,
        companion_files,
        compatibility,
    )
    selected_files.update(generated_files)
    merged_data = dict(data)
    lifecycle_rows = _processing_lifecycle_rows(
        _processing_capture_snapshot(self, data),
        ((data.get("metadata") or {}).get("capture") or {}).get("host_start_ns"),
    )
    merged_data["premat_lifecycle"] = lifecycle_rows
    merged_data["premat_lifecycle_summary"] = _processing_lifecycle_summary(
        lifecycle_rows,
        data.get("intervals", []),
    )
    merged_data["premat_compatibility"] = compatibility
    merged_data["processing_contention_intervals"] = _processing_contention_intervals(
        data, compatibility
    )
    merged_data["processing_contention_method"] = {
        "admission": "launch completion and preceding same-stream completion form an eligibility lower bound; NCU proves only structural SM admission exclusions",
        "pressure": "device-wide NSYS samples at >=75% tensor/DRAM/L2 or >=90% issue during exact MAIN/PREMAT overlap; correlation, not causation",
        "slowdown": "shown only when an exact matched-control delta is present; this payload does not synthesize one",
    }
    merged_data["premat_hard_constraints"] = list(companion_data.get("premat_hard_constraints") or [])
    merged_data["premat_compatibility_files"] = dict(
        companion_data.get("premat_compatibility_files") or {}
    )
    merged_data["premat_compatibility_source"] = {
        "nsys_dashboard_run_id": str(selected_status.get("dashboard_run_id", "")),
        "nsys_artifact_name": str(selected_status.get("artifact_name", "")),
        "dashboard_run_id": str(companion_status.get("dashboard_run_id", "")),
        "artifact_name": str(companion_status.get("artifact_name", "")),
        "created_at": str(companion_status.get("created_at", "")),
        "host_label": str(companion_status.get("host_label", "")),
        "pair_key": pair_key_text(_processing_pair_key(self)),
        "selection": (
            "persisted_pair"
            if preferred_ncu_run_id
            else "nearest_forward_viable_unclaimed_ncu_by_artifact_time"
        ),
        "distance_minutes": diagnostics.get("distance_minutes"),
        "skipped_closer_invalid_count": diagnostics.get("skipped_closer_invalid_count", 0),
        "skipped_closer_invalid_artifacts": diagnostics.get("skipped_closer_invalid_artifacts", []),
        "viable_candidate_count": diagnostics.get("viable_candidate_count", 0),
    }
    pair_files = {
        key: selected_files[key]
        for key in ("everything", "most", "pair_manifest")
        if key in selected_files
    }
    nsys_files = {
        key: value for key, value in selected_files.items()
        if key not in pair_files
    }
    merged_data["paired_processing_downloads"] = {
        "pair": {
            "dashboard_run_id": str(selected_status.get("dashboard_run_id", "")),
            "artifact_name": (
                f"{selected_status.get('artifact_name', '')} + "
                f"{companion_status.get('artifact_name', '')}"
            ),
            "files": pair_files,
        },
        "nsys": {
            "dashboard_run_id": str(selected_status.get("dashboard_run_id", "")),
            "artifact_name": str(selected_status.get("artifact_name", "")),
            "files": nsys_files,
        },
        "ncu": {
            "dashboard_run_id": str(companion_status.get("dashboard_run_id", "")),
            "artifact_name": str(companion_status.get("artifact_name", "")),
            "files": companion_files,
        },
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


# vvv THOG Files-tab deletion is restricted to ordinary files inside the selected run root.
def _delete_local_file(self, run_name: str, relative_path: str) -> dict[str, Any]:
    _state, path, relative = self._resolved_local_path(run_name, relative_path)
    if not relative.parts:
        raise PermissionError("refusing to delete a run root")
    if not path.is_file() or path.is_symlink():
        raise ValueError("only ordinary files inside the selected run may be deleted")
    path.unlink()
    return {"deleted": str(path), "path": relative.as_posix()}


_base.DashboardCatalog.delete_local_file = _delete_local_file
_handler_for_before_file_delete = _base._handler_for


def _handler_for_with_file_delete(catalog):
    handler = _handler_for_before_file_delete(catalog)
    original_do_delete = handler.do_DELETE

    def do_delete(self):
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(self.path)
        if parsed.path != "/api/local-file":
            original_do_delete(self)
            return
        query = parse_qs(parsed.query)
        run_name = query.get("run", [""])[0]
        relative_path = query.get("path", [""])[0]
        if not run_name or not relative_path:
            self._send_json(
                {"error": "run and path query parameters are required"},
                status=_base.HTTPStatus.BAD_REQUEST,
            )
            return
        try:
            self._send_json(catalog.delete_local_file(run_name, relative_path))
        except (FileNotFoundError, KeyError) as error:
            self._send_json({"error": str(error)}, status=_base.HTTPStatus.NOT_FOUND)
        except (PermissionError, ValueError) as error:
            self._send_json({"error": str(error)}, status=_base.HTTPStatus.BAD_REQUEST)

    handler.do_DELETE = do_delete
    return handler


_base._handler_for = _handler_for_with_file_delete
# ^^^ THOG


_original_asset_root = Path(_base._ASSET_ROOT)
_overlay_asset_root = Path(tempfile.mkdtemp(prefix="thog2-instra-assets-"))
_dashboard_patch_names = (
    "dashboard_workspace_only.js",
    "dashboard_runs_table_restore.js",
    "dashboard_sep16_workspace_ui_repair.js",
    "dashboard_runs_trash_restore.js",
    "dashboard_feature_regression_restore.js",
)
_processing_patch_names = (
    "dashboard_processing_resource_attribution.js",
    "dashboard_processing_ncu_2024_repair.js",
    "dashboard_processing_compatibility_classification_patch.js",
    "dashboard_processing_sep16_stability_polish.js",
    "dashboard_processing_sep16_companion_fallback.js",
    "dashboard_processing_sep16_quiescence.js",
    "dashboard_processing_final_design.js",
    "dashboard_processing_operations_final.js",
    "dashboard_processing_pair_state_final.js",
    "dashboard_processing_nsys_eye_select.js",
    "dashboard_processing_user_fixes.js",
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
    # Runtime launchers install API wrappers and add late assets on this public
    # module.  The preserved server implementation resolves its globals in
    # ``_base``, so copy the final assembled owners across immediately before
    # starting it.  Without this bridge the UI can initially show core charts
    # and later appear Processing-only because chart-group routes/assets were
    # installed on a module the HTTP server never consulted.
    for name in ("_handler_for", "_ASSET_ROOT", "_ASSET_NAMES"):
        if name in globals():
            setattr(_base, name, globals()[name])
    return _base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
# ^^^ THOG
