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
        "exact_names": (
            "SMs Active",
            "SMs Active [Throughput %]",
            "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
        ),
        "unit": "%",
        "required": True,
    },
    "sm_issue_pct": {
        "exact_names": (
            "SM Issue",
            "SM Issue [Throughput %]",
            "sm__inst_executed_realtime.avg.pct_of_peak_sustained_elapsed",
        ),
        "unit": "%",
        "required": True,
    },
    "tensor_active_pct": {
        "exact_names": (
            "Tensor Active",
            "Tensor Active [Throughput %]",
            "sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed",
        ),
        "unit": "%",
        "required": True,
    },
    "active_sm_unused_warp_slots_pct": {
        "exact_names": (
            "Active SM Unused Warp Slots",
            "Active SM Unused Warp Slots [Throughput %]",
            "Unallocated Warps in Active SM",
            "Unallocated Warps in Active SMs [Throughput %]",
            "tpc__warps_inactive_sm_active_realtime.avg.pct_of_peak_sustained_elapsed",
        ),
        "unit": "%",
        "required": True,
    },
    "dram_read_pct": {
        "exact_names": (
            "DRAM Read",
            "DRAM Read Throughput",
            "DRAM Read Bandwidth [Throughput %]",
            "dram__read_throughput.avg.pct_of_peak_sustained_elapsed",
            "dramc__read_throughput.avg.pct_of_peak_sustained_elapsed",
        ),
        "unit": "%",
        "required": False,
    },
    "dram_write_pct": {
        "exact_names": (
            "DRAM Write",
            "DRAM Write Throughput",
            "DRAM Write Bandwidth [Throughput %]",
            "dram__write_throughput.avg.pct_of_peak_sustained_elapsed",
            "dramc__write_throughput.avg.pct_of_peak_sustained_elapsed",
        ),
        "unit": "%",
        "required": False,
    },
    "gr_active_pct": {
        "exact_names": (
            "Graphics/Compute Active",
            "GR Active",
            "GR Active [Throughput %]",
            "gr__cycles_active.sum.pct_of_peak_sustained_elapsed",
        ),
        "unit": "%",
        "required": False,
    },
    "l2_active_pct": {
        "exact_names": (
            "L2 Active",
            "L2 Throughput",
            "L2 Active [Throughput %]",
            "lts__cycles_active.avg.pct_of_peak_sustained_elapsed",
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
            "matching": "exact",
            "transform": str(spec.get("transform", "identity")),
        }
    return result


def match_processing_metric_name(name: Any) -> str | None:
    """Return a metric key only for an explicitly enumerated exact name."""
    normalized = str(name or "").strip().casefold()
    if not normalized:
        return None
    for key, specification in PROCESSING_METRIC_SPECS.items():
        if normalized in {
            str(candidate).strip().casefold()
            for candidate in specification.get("exact_names", ())
        }:
            return key
    return None


def metric_display_value(key: str, value: Any) -> float | None:
    """Apply the one allowed display conversion; percentages remain raw."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(number):
        return None
    if PROCESSING_METRIC_SPECS.get(key, {}).get("transform") == "hz_to_mhz":
        return number / 1_000_000.0
    return number


def duration_weighted_metric_statistics(
    samples: Sequence[Mapping[str, Any]],
    key: str,
    intervals: Iterable[tuple[float, float]],
) -> Dict[str, float | int | None]:
    """Summarise midpoint bins by their exact overlap with an interval union."""
    target = merge_intervals(intervals)
    weighted: list[tuple[float, float]] = []
    for row in samples:
        try:
            value = float(row.get(key))
        except (TypeError, ValueError):
            continue
        if not isfinite(value):
            continue
        try:
            start = float(row["sample_start_us"])
            end = float(row["sample_end_us"])
        except (KeyError, TypeError, ValueError):
            continue
        weight = sum(
            max(0.0, min(end, right) - max(start, left))
            for left, right in target
            if right > start and left < end
        )
        if weight > 0.0:
            weighted.append((value, weight))
    if not weighted:
        return {
            "sample_count": 0,
            "covered_us": 0.0,
            "mean": None,
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    total = sum(weight for _value, weight in weighted)
    ordered = sorted(weighted)

    def quantile(fraction: float) -> float:
        threshold = total * fraction
        cumulative = 0.0
        for value, weight in ordered:
            cumulative += weight
            if cumulative >= threshold:
                return value
        return ordered[-1][0]

    values = [value for value, _weight in weighted]
    return {
        "sample_count": len(weighted),
        "covered_us": total,
        "mean": sum(value * weight for value, weight in weighted) / total,
        "min": min(values),
        "p50": quantile(0.50),
        "p90": quantile(0.90),
        "p95": quantile(0.95),
        "max": max(values),
    }


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
    "match_processing_metric_name",
    "metric_display_value",
    "duration_weighted_metric_statistics",
    "sample_bins",
]
# ^^^ THOG

# vvv THOG Processing resource tide metrics and operation subdivision v1
_previous_processing_metric_fields = PROCESSING_METRIC_FIELDS
PROCESSING_METRIC_SPECS.update({
    "compute_warps_in_flight_pct": {
        "exact_names": (
            "Compute Warps in Flight",
            "Compute Warps In Flight",
            "Compute Warps in Flight [Throughput %]",
            "tpc__warps_active_shader_cs_realtime.avg.pct_of_peak_sustained_elapsed",
        ),
        "unit": "%",
        "required": False,
    },
    "idle_sm_unused_warp_slots_pct": {
        "exact_names": (
            "Idle SM Unused Warp Slots",
            "Unallocated Warps in Idle SM",
            "Unallocated Warps in Idle SMs [Throughput %]",
            "tpc__warps_inactive_sm_idle_realtime.avg.pct_of_peak_sustained_elapsed",
        ),
        "unit": "%",
        "required": False,
    },
    "gpc_clock_mhz": {
        "exact_names": (
            "GPC Clock Frequency",
            "GPC Clock",
            "GPC Clock Frequency [MHz]",
            "gpc__cycles_elapsed.avg.per_second",
        ),
        "unit": "MHz",
        "required": False,
        "transform": "hz_to_mhz",
    },
})
PROCESSING_METRIC_FIELDS = tuple(PROCESSING_METRIC_SPECS)
_processing_resource_context_fields = tuple(
    field for field in PROCESSING_STREAM_RESOURCE_FIELDS
    if field not in _previous_processing_metric_fields
)
PROCESSING_STREAM_RESOURCE_FIELDS = (
    *_processing_resource_context_fields,
    *PROCESSING_METRIC_FIELDS,
)


def _operation_from_kernel_name(kernel_name: Any) -> str:
    lower = str(kernel_name or "").strip().lower()
    if not lower:
        return "misc"
    if any(token in lower for token in (
        "flashattention", "flash_attn", "fmha", "fused_attention", "attention_forward",
    )):
        return "attention"
    if any(token in lower for token in ("layer_norm", "layernorm", "rms_norm", "rmsnorm")):
        return "layernorm"
    if any(token in lower for token in ("gelu", "silu", "relu")):
        return "activation"
    if "residual" in lower:
        return "residual"
    if any(token in lower for token in ("cross_entropy", "nll_loss", "log_softmax")):
        return "loss"
    if any(token in lower for token in ("lm_head", "lmhead", "vocab_projection", "output_projection")):
        return "lm_head"
    return "misc"


def classify_processing_operations(rows: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    """Conservatively replace generic other with useful semantic classes.

    Explicit NVTX operations are preserved. Kernel-name classification is used
    only for otherwise generic work. A positional LM-head inference is made only
    for one dominant long post-stack MAIN kernel. Ambiguous work remains misc.
    """
    result: list[Dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        operation = str(row.get("operation") or "other")
        if operation != "other":
            row.setdefault("operation_source", "nvtx")
        else:
            classified = _operation_from_kernel_name(row.get("kernel_name"))
            row["operation"] = classified
            row["operation_source"] = "kernel_name" if classified != "misc" else "misc"
        result.append(row)

    layered_main = [
        row for row in result
        if str(row.get("owner")) == "MAIN"
        and row.get("layer") not in (None, "")
        and _finite_number(row.get("end_us"))
    ]
    if not layered_main:
        return result
    last_layer_end = max(float(row["end_us"]) for row in layered_main)

    for row in result:
        if (
            str(row.get("owner")) == "MAIN"
            and str(row.get("operation")) == "misc"
            and _finite_number(row.get("start_us"))
            and float(row["start_us"]) >= last_layer_end
        ):
            lower = str(row.get("kernel_name") or "").lower()
            if "softmax" in lower:
                row["operation"] = "loss"
                row["operation_source"] = "kernel_name_post_stack"

    opaque_post_stack = [
        row for row in result
        if str(row.get("owner")) == "MAIN"
        and str(row.get("operation")) == "misc"
        and _finite_number(row.get("start_us"))
        and _finite_number(row.get("duration_us"))
        and float(row["start_us"]) >= last_layer_end
        and float(row["duration_us"]) >= 5000.0
    ]
    opaque_post_stack.sort(key=lambda row: float(row["duration_us"]), reverse=True)
    if opaque_post_stack:
        dominant = opaque_post_stack[0]
        runner_up_us = float(opaque_post_stack[1]["duration_us"]) if len(opaque_post_stack) > 1 else 0.0
        if runner_up_us <= 0.0 or float(dominant["duration_us"]) >= 2.0 * runner_up_us:
            dominant["operation"] = "lm_head"
            dominant["operation_source"] = "position_post_stack_dominant"
    return result
# ^^^ THOG
