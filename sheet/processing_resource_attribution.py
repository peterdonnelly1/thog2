# vvv THOG
"""Pure helpers for Processing stream/resource attribution.

The Nsight Systems GPU metrics used by Processing are device-wide. This module
therefore attributes a sample only when CUDA kernel ownership makes that
attribution exact. Samples containing real Main/PREMAT overlap remain combined
and are never numerically split between streams.
"""

from __future__ import annotations

from collections import defaultdict
from math import isfinite
from typing import Any, Dict, Iterable, Mapping, Sequence


PROCESSING_OWNERS = ("MAIN", "PREMAT", "OTHER", "UNKNOWN")
PROCESSING_OWNER_SOURCES = ("NVTX", "STREAM_INFERRED", "UNKNOWN")
PROCESSING_ATTRIBUTION_STATES = (
    "MAIN_ONLY",
    "PREMAT_ONLY",
    "MAIN_PREMAT_OVERLAP",
    "MIXED_SEQUENTIAL",
    "OTHER_OR_UNKNOWN",
    "IDLE",
)

PROCESSING_METRIC_SPECS: Dict[str, Dict[str, Any]] = {
    "sm_active_pct": {
        "patterns": ("SMs Active", "sm__cycles_active"),
        "unit": "%",
        "required": True,
    },
    "sm_issue_pct": {
        "patterns": ("SM Issue", "sm__inst_executed"),
        "unit": "%",
        "required": True,
    },
    "tensor_active_pct": {
        "patterns": ("Tensor Active", "sm__pipe_tensor"),
        "unit": "%",
        "required": True,
    },
    "active_sm_unused_warp_slots_pct": {
        "patterns": (
            "Active SM Unused Warp Slots",
            "Unallocated Warps in Active SM",
            "tpc__warps_inactive_sm_active",
        ),
        "unit": "%",
        "required": True,
    },
    "dram_read_pct": {
        "patterns": (
            "DRAM Read",
            "Memory Read",
            "dram__bytes_read",
            "dram__throughput_read",
        ),
        "unit": "%",
        "required": False,
    },
    "dram_write_pct": {
        "patterns": (
            "DRAM Write",
            "Memory Write",
            "dram__bytes_write",
            "dram__throughput_write",
        ),
        "unit": "%",
        "required": False,
    },
    "gr_active_pct": {
        "patterns": (
            "Graphics/Compute Active",
            "GR Active",
            "gr__cycles_active",
        ),
        "unit": "%",
        "required": False,
    },
    "l2_active_pct": {
        "patterns": (
            "L2 Active",
            "L2 Throughput",
            "lts__cycles_active",
        ),
        "unit": "%",
        "required": False,
    },
}

PROCESSING_METRIC_FIELDS = tuple(PROCESSING_METRIC_SPECS)

PROCESSING_STREAM_RESOURCE_FIELDS = (
    "time_us",
    "sample_start_us",
    "sample_end_us",
    "attribution_state",
    "main_coverage_pct",
    "premat_coverage_pct",
    "other_coverage_pct",
    "main_premat_simultaneous_pct",
    "main_stream_ids",
    "main_operations",
    "main_layers",
    "main_families",
    "premat_stream_ids",
    "premat_operations",
    "premat_layers",
    "premat_families",
    *PROCESSING_METRIC_FIELDS,
)


def canonical_owner(value: Any) -> str:
    owner = str(value or "").strip().upper()
    if owner in {"MAIN", "PREMAT"}:
        return owner
    if owner == "UNKNOWN" or not owner:
        return "UNKNOWN"
    return "OTHER"


def infer_kernel_owners(rows: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    """Resolve every kernel to an explicit owner without defaulting to Main."""
    explicit_by_stream: Dict[tuple[Any, int], set[str]] = defaultdict(set)
    normalized: list[Dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        explicit = bool(row.pop("_owner_explicit", False))
        context = row.get("context_id", "")
        stream = int(row.get("stream", 0))
        if explicit:
            owner = canonical_owner(row.get("owner"))
            row["owner"] = owner
            row["owner_source"] = "NVTX"
            if context not in (None, ""):
                explicit_by_stream[(context, stream)].add(owner)
        else:
            row["owner"] = "UNKNOWN"
            row["owner_source"] = "UNKNOWN"
        normalized.append(row)

    for row in normalized:
        if row["owner_source"] != "UNKNOWN":
            continue
        context = row.get("context_id", "")
        if context in (None, ""):
            continue
        key = (context, int(row.get("stream", 0)))
        owners = explicit_by_stream.get(key, set())
        if len(owners) == 1:
            row["owner"] = next(iter(owners))
            row["owner_source"] = "STREAM_INFERRED"
    return normalized


def merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for left, right in sorted(
        (float(left), float(right))
        for left, right in intervals
        if float(right) > float(left)
    ):
        if not merged or left > merged[-1][1]:
            merged.append((left, right))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
    return merged


def interval_duration(intervals: Iterable[tuple[float, float]]) -> float:
    return sum(right - left for left, right in merge_intervals(intervals))


def clipped_union_duration(
    intervals: Iterable[tuple[float, float]],
    start: float,
    end: float,
) -> float:
    if end <= start:
        return 0.0
    clipped = (
        (max(start, left), min(end, right))
        for left, right in intervals
        if right > start and left < end
    )
    return interval_duration(clipped)


def interval_overlap_duration(
    left_intervals: Iterable[tuple[float, float]],
    right_intervals: Iterable[tuple[float, float]],
    *,
    start: float | None = None,
    end: float | None = None,
) -> float:
    left = merge_intervals(left_intervals)
    right = merge_intervals(right_intervals)
    i = 0
    j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        overlap_start = max(left[i][0], right[j][0])
        overlap_end = min(left[i][1], right[j][1])
        if start is not None:
            overlap_start = max(overlap_start, float(start))
        if end is not None:
            overlap_end = min(overlap_end, float(end))
        if overlap_end > overlap_start:
            total += overlap_end - overlap_start
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def sample_bins(
    samples: Sequence[Mapping[str, Any]],
    capture_duration_us: float,
) -> list[Dict[str, Any]]:
    """Give each device-wide sample a midpoint-bounded interval."""
    duration = max(0.0, float(capture_duration_us))
    ordered = sorted(
        (dict(row) for row in samples if _finite_number(row.get("time_us"))),
        key=lambda row: float(row["time_us"]),
    )
    result: list[Dict[str, Any]] = []
    for index, row in enumerate(ordered):
        center = min(duration, max(0.0, float(row["time_us"])))
        previous = float(ordered[index - 1]["time_us"]) if index else None
        following = float(ordered[index + 1]["time_us"]) if index + 1 < len(ordered) else None
        start = 0.0 if previous is None else max(0.0, (previous + center) / 2.0)
        end = duration if following is None else min(duration, (center + following) / 2.0)
        if end < start:
            end = start
        row["time_us"] = center
        row["sample_start_us"] = start
        row["sample_end_us"] = end
        result.append(row)
    return result


def main_idle_intervals(
    interval_rows: Sequence[Mapping[str, Any]],
    capture_duration_us: float,
) -> list[Dict[str, float]]:
    duration = max(0.0, float(capture_duration_us))
    busy = merge_intervals(
        (
            max(0.0, float(row["start_us"])),
            min(duration, float(row["end_us"])),
        )
        for row in interval_rows
        if str(row.get("owner")) == "MAIN"
        and _finite_number(row.get("start_us"))
        and _finite_number(row.get("end_us"))
    )
    result: list[Dict[str, float]] = []
    cursor = 0.0
    for left, right in busy:
        if left > cursor:
            result.append({"start_us": cursor, "end_us": left, "duration_us": left - cursor})
        cursor = max(cursor, right)
    if cursor < duration:
        result.append({"start_us": cursor, "end_us": duration, "duration_us": duration - cursor})
    return result


def build_stream_resource_rows(
    samples: Sequence[Mapping[str, Any]],
    interval_rows: Sequence[Mapping[str, Any]],
    capture_duration_us: float,
) -> list[Dict[str, Any]]:
    intervals_by_owner: Dict[str, list[tuple[float, float]]] = {
        owner: [] for owner in PROCESSING_OWNERS
    }
    for row in interval_rows:
        if not (_finite_number(row.get("start_us")) and _finite_number(row.get("end_us"))):
            continue
        owner = canonical_owner(row.get("owner"))
        intervals_by_owner[owner].append((float(row["start_us"]), float(row["end_us"])))

    main_intervals = intervals_by_owner["MAIN"]
    premat_intervals = intervals_by_owner["PREMAT"]
    other_intervals = [
        *intervals_by_owner["OTHER"],
        *intervals_by_owner["UNKNOWN"],
    ]
    result: list[Dict[str, Any]] = []
    for sample in sample_bins(samples, capture_duration_us):
        start = float(sample["sample_start_us"])
        end = float(sample["sample_end_us"])
        width = end - start
        main_us = clipped_union_duration(main_intervals, start, end)
        premat_us = clipped_union_duration(premat_intervals, start, end)
        other_us = clipped_union_duration(other_intervals, start, end)
        simultaneous_us = interval_overlap_duration(
            main_intervals,
            premat_intervals,
            start=start,
            end=end,
        )
        state = _attribution_state(main_us, premat_us, other_us, simultaneous_us)
        row: Dict[str, Any] = dict(sample)
        row.update({
            "attribution_state": state,
            "main_coverage_pct": _percentage(main_us, width),
            "premat_coverage_pct": _percentage(premat_us, width),
            "other_coverage_pct": _percentage(other_us, width),
            "main_premat_simultaneous_pct": _percentage(simultaneous_us, width),
        })
        row.update(_semantic_context(interval_rows, start, end, "MAIN", "main"))
        row.update(_semantic_context(interval_rows, start, end, "PREMAT", "premat"))
        for key in PROCESSING_METRIC_FIELDS:
            row.setdefault(key, "")
        result.append(row)
    return result


def metric_catalog(metric_mapping: Mapping[str, str]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for key, spec in PROCESSING_METRIC_SPECS.items():
        mapped = str(metric_mapping.get(key, ""))
        result[key] = {
            "canonical_name": key,
            "nsight_name": mapped,
            "unit": str(spec["unit"]),
            "available": bool(mapped),
            "required": bool(spec["required"]),
        }
    return result


def _attribution_state(main_us: float, premat_us: float, other_us: float, simultaneous_us: float) -> str:
    epsilon = 1e-9
    if other_us > epsilon:
        return "OTHER_OR_UNKNOWN"
    if main_us > epsilon and premat_us > epsilon:
        return "MAIN_PREMAT_OVERLAP" if simultaneous_us > epsilon else "MIXED_SEQUENTIAL"
    if main_us > epsilon:
        return "MAIN_ONLY"
    if premat_us > epsilon:
        return "PREMAT_ONLY"
    return "IDLE"


def _semantic_context(
    interval_rows: Sequence[Mapping[str, Any]],
    start: float,
    end: float,
    owner: str,
    prefix: str,
) -> Dict[str, str]:
    rows = [
        row for row in interval_rows
        if str(row.get("owner")) == owner
        and _finite_number(row.get("start_us"))
        and _finite_number(row.get("end_us"))
        and float(row["end_us"]) > start
        and float(row["start_us"]) < end
    ]
    return {
        f"{prefix}_stream_ids": _joined_values(row.get("stream") for row in rows),
        f"{prefix}_operations": _joined_values(row.get("operation") for row in rows),
        f"{prefix}_layers": _joined_values((row.get("layer") for row in rows), numeric=True),
        f"{prefix}_families": _joined_values(row.get("family") for row in rows),
    }


def _joined_values(values: Iterable[Any], *, numeric: bool = False) -> str:
    normalized = set()
    for value in values:
        if value in (None, ""):
            continue
        if numeric:
            try:
                normalized.add(str(int(value) + 1))
            except (TypeError, ValueError):
                continue
        else:
            text = str(value).strip()
            if text:
                normalized.add(text)
    return ";".join(sorted(normalized, key=lambda value: (len(value), value)))


def _percentage(numerator: float, denominator: float) -> float:
    return 100.0 * numerator / denominator if denominator > 0.0 else 0.0


def _finite_number(value: Any) -> bool:
    try:
        return isfinite(float(value))
    except (TypeError, ValueError):
        return False


__all__ = [
    "PROCESSING_ATTRIBUTION_STATES",
    "PROCESSING_METRIC_FIELDS",
    "PROCESSING_METRIC_SPECS",
    "PROCESSING_OWNER_SOURCES",
    "PROCESSING_OWNERS",
    "PROCESSING_STREAM_RESOURCE_FIELDS",
    "build_stream_resource_rows",
    "canonical_owner",
    "infer_kernel_owners",
    "interval_duration",
    "interval_overlap_duration",
    "main_idle_intervals",
    "merge_intervals",
    "metric_catalog",
    "sample_bins",
]
# ^^^ THOG
