# vvv THOG
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one occurrence, found {count}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1))


def write_new(path: str, content: str) -> None:
    target = ROOT / path
    if target.exists():
        raise RuntimeError(f"refusing to overwrite existing file: {path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)


PREMAT_PROCESSING_PY = r'''# vvv THOG
"""Opt-in PREMAT processing capture and Nsight Systems normalization.

Nsight owns device-wide hardware-counter collection. THOG contributes only a
single bounded NVTX capture range plus semantic operation ranges, then converts
the exported SQLite trace into stable INSTRA/download formats.
"""

from __future__ import annotations

import csv
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence
import zipfile


PROCESSING_SWITCHES = ("enabled", "disabled")
PROCESSING_DEFAULT_CAPTURE_FREQUENCY_HZ = 10_000
PROCESSING_MIN_CAPTURE_FREQUENCY_HZ = 10
PROCESSING_MAX_CAPTURE_FREQUENCY_HZ = 200_000
PROCESSING_SCHEMA_VERSION = 1
PROCESSING_CAPTURE_RANGE = "THOG2_PREMAT_PROCESSING_CAPTURE"
PROCESSING_OPERATION_PREFIX = "THOG2_PROCESSING"
_PROCESSING_CHILD_ENV = "THOG2_PREMAT_PROCESSING_UNDER_NSYS"
_PROCESSING_HANDOFF_ENV = "THOG2_PREMAT_PROCESSING_HANDOFF"
_PROCESSING_CAPTURE_METADATA_ENV = "THOG2_PREMAT_PROCESSING_CAPTURE_METADATA"

_capture_active = False
_capture_done = False
_capture_stack_depth = 0


def validate_processing_configuration(logging: str, capture_frequency_hz: int, device: str) -> None:
    if logging not in PROCESSING_SWITCHES:
        raise ValueError(f"premat_processing_logging must be enabled or disabled; got {logging!r}")
    if isinstance(capture_frequency_hz, bool) or not isinstance(capture_frequency_hz, int):
        raise ValueError("premat_processing_logging_capture_frequency_hz must be an integer")
    if not PROCESSING_MIN_CAPTURE_FREQUENCY_HZ <= capture_frequency_hz <= PROCESSING_MAX_CAPTURE_FREQUENCY_HZ:
        raise ValueError(
            "premat_processing_logging_capture_frequency_hz must lie in "
            f"[{PROCESSING_MIN_CAPTURE_FREQUENCY_HZ}, {PROCESSING_MAX_CAPTURE_FREQUENCY_HZ}]"
        )
    if logging == "enabled" and not str(device).startswith("cuda"):
        raise ValueError("--premat_processing_logging enabled requires a CUDA device")


def _argv_value(arguments: Sequence[str], name: str, default: Optional[str] = None) -> Optional[str]:
    prefix = f"{name}="
    for index, argument in enumerate(arguments):
        if argument.startswith(prefix):
            return argument[len(prefix):]
        if argument == name and index + 1 < len(arguments):
            return arguments[index + 1]
    return default


def processing_requested_from_argv(arguments: Sequence[str]) -> tuple[bool, int]:
    logging = str(_argv_value(arguments, "--premat_processing_logging", "disabled"))
    raw_frequency = _argv_value(
        arguments,
        "--premat_processing_logging_capture_frequency_hz",
        str(PROCESSING_DEFAULT_CAPTURE_FREQUENCY_HZ),
    )
    try:
        frequency = int(str(raw_frequency))
    except ValueError as error:
        raise ValueError(
            "--premat_processing_logging_capture_frequency_hz must be an integer"
        ) from error
    validate_processing_configuration(logging, frequency, "cuda")
    return logging == "enabled", frequency


def _find_nsys() -> Optional[str]:
    resolved = shutil.which("nsys")
    if resolved:
        return resolved
    candidates = sorted(Path("/opt/nvidia/nsight-systems").glob("*/bin/nsys"), reverse=True)
    candidates.extend(Path("/usr/local/cuda/bin").glob("nsys"))
    return str(candidates[0]) if candidates else None


def register_processing_handoff(
    run_directory: Path,
    *,
    run_name: str,
    config: Mapping[str, Any],
) -> None:
    target = os.environ.get(_PROCESSING_HANDOFF_ENV, "").strip()
    if not target:
        return
    payload = {
        "run_directory": str(Path(run_directory).resolve()),
        "run_name": str(run_name),
        "config": {
            "premat": config.get("premat"),
            "premat_target_layer": config.get("premat_target_layer"),
            "premat_attention_mode": config.get("premat_attention_mode"),
            "premat_processing_logging": config.get("premat_processing_logging"),
            "premat_processing_logging_capture_frequency_hz": config.get(
                "premat_processing_logging_capture_frequency_hz"
            ),
            "n_layer": config.get("n_layer"),
            "n_embd": config.get("n_embd"),
            "n_head": config.get("n_head"),
            "batch_size": config.get("batch_size"),
            "block_size": config.get("block_size"),
            "gradient_accumulation_steps": config.get("gradient_accumulation_steps"),
            "dtype": config.get("dtype"),
            "device": config.get("device"),
        },
    }
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _capture_update_target(max_updates: int, log_interval: int) -> int:
    return min(max(1, int(max_updates)), max(1, int(log_interval)))


def should_capture_processing_forward(
    *,
    enabled: bool,
    completed_updates: int,
    max_updates: int,
    log_interval: int,
    micro_step: int,
) -> bool:
    global _capture_done
    if not enabled or _capture_done or int(micro_step) != 0:
        return False
    prospective_update = int(completed_updates) + 1
    return prospective_update >= _capture_update_target(max_updates, log_interval)


def _nvtx_push(label: str) -> None:
    import torch

    torch.cuda.nvtx.range_push(label)


def _nvtx_pop() -> None:
    import torch

    torch.cuda.nvtx.range_pop()


def _operation_label(owner: str, operation: str, family: Optional[str], layer_index: Optional[int]) -> str:
    fields = [PROCESSING_OPERATION_PREFIX, f"owner={owner}", f"operation={operation}"]
    if family is not None:
        fields.append(f"family={family}")
    if layer_index is not None:
        fields.append(f"layer={int(layer_index)}")
    return "|".join(fields)


def processing_operation_push(
    owner: str,
    operation: str,
    *,
    family: Optional[str] = None,
    layer_index: Optional[int] = None,
) -> bool:
    global _capture_stack_depth
    if not _capture_active:
        return False
    _nvtx_push(_operation_label(owner, operation, family, layer_index))
    _capture_stack_depth += 1
    return True


def processing_operation_pop(pushed: bool) -> None:
    global _capture_stack_depth
    if not pushed:
        return
    _nvtx_pop()
    _capture_stack_depth = max(0, _capture_stack_depth - 1)


@contextmanager
def processing_operation_range(
    owner: str,
    operation: str,
    *,
    family: Optional[str] = None,
    layer_index: Optional[int] = None,
) -> Iterator[None]:
    pushed = processing_operation_push(
        owner,
        operation,
        family=family,
        layer_index=layer_index,
    )
    try:
        yield
    finally:
        processing_operation_pop(pushed)


@contextmanager
def processing_capture_scope(
    *,
    enabled: bool,
    completed_updates: int,
    max_updates: int,
    log_interval: int,
    micro_step: int,
    device: Any,
) -> Iterator[None]:
    global _capture_active, _capture_done, _capture_stack_depth
    selected = should_capture_processing_forward(
        enabled=enabled,
        completed_updates=completed_updates,
        max_updates=max_updates,
        log_interval=log_interval,
        micro_step=micro_step,
    )
    if not selected:
        yield
        return

    import torch

    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise RuntimeError("PREMAT processing capture requires CUDA")
    torch.cuda.synchronize(device)
    metadata_path = os.environ.get(_PROCESSING_CAPTURE_METADATA_ENV, "").strip()
    if metadata_path:
        metadata = {
            "optimizer_update": int(completed_updates) + 1,
            "micro_step": int(micro_step) + 1,
            "log_interval": int(log_interval),
            "max_updates": int(max_updates),
        }
        destination = Path(metadata_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    _capture_active = True
    _capture_stack_depth = 0
    _nvtx_push(PROCESSING_CAPTURE_RANGE)
    try:
        yield
    finally:
        torch.cuda.synchronize(device)
        while _capture_stack_depth > 0:
            _nvtx_pop()
            _capture_stack_depth -= 1
        _nvtx_pop()
        _capture_active = False
        _capture_done = True


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')}


def _first_column(columns: set[str], *names: str) -> Optional[str]:
    for name in names:
        if name in columns:
            return name
    return None


def _string_ids(connection: sqlite3.Connection, tables: set[str]) -> Dict[int, str]:
    if "StringIds" not in tables:
        return {}
    columns = _columns(connection, "StringIds")
    id_column = _first_column(columns, "id", "stringId")
    value_column = _first_column(columns, "value", "text")
    if id_column is None or value_column is None:
        return {}
    return {
        int(row[0]): str(row[1])
        for row in connection.execute(
            f'SELECT "{id_column}", "{value_column}" FROM "StringIds"'
        )
    }


def _resolved_string(value: Any, strings: Mapping[int, str]) -> str:
    if value is None:
        return ""
    if isinstance(value, int) and value in strings:
        return strings[value]
    try:
        integer = int(value)
    except (TypeError, ValueError):
        integer = None
    if integer is not None and integer in strings:
        return strings[integer]
    return str(value)


def _nvtx_events(connection: sqlite3.Connection, tables: set[str], strings: Mapping[int, str]) -> list[Dict[str, Any]]:
    if "NVTX_EVENTS" not in tables:
        raise RuntimeError("Nsight SQLite export has no NVTX_EVENTS table")
    columns = _columns(connection, "NVTX_EVENTS")
    start_column = _first_column(columns, "start", "timestamp")
    end_column = _first_column(columns, "end")
    thread_column = _first_column(columns, "globalTid", "globalThreadId", "threadId")
    text_column = _first_column(columns, "text")
    text_id_column = _first_column(columns, "textId", "messageId")
    if start_column is None or end_column is None or (text_column is None and text_id_column is None):
        raise RuntimeError(f"unsupported NVTX_EVENTS schema: {sorted(columns)}")
    selected = [start_column, end_column]
    selected.append(thread_column or start_column)
    selected.append(text_column or text_id_column)  # type: ignore[arg-type]
    rows = connection.execute(
        "SELECT " + ", ".join(f'"{column}"' for column in selected) + ' FROM "NVTX_EVENTS"'
    )
    events = []
    for start, end, thread, raw_text in rows:
        if start is None or end is None:
            continue
        text = str(raw_text) if text_column is not None else _resolved_string(raw_text, strings)
        events.append({"start": int(start), "end": int(end), "thread": int(thread), "text": text})
    return events


def _parse_operation_label(text: str) -> Optional[Dict[str, Any]]:
    if not text.startswith(PROCESSING_OPERATION_PREFIX + "|"):
        return None
    values: Dict[str, str] = {}
    for field in text.split("|")[1:]:
        key, separator, value = field.partition("=")
        if separator:
            values[key] = value
    if "owner" not in values or "operation" not in values:
        return None
    return {
        "owner": values["owner"],
        "operation": values["operation"],
        "family": values.get("family", ""),
        "layer": int(values["layer"]) if "layer" in values else None,
    }


def _runtime_rows(connection: sqlite3.Connection, tables: set[str]) -> list[Dict[str, int]]:
    table = "CUPTI_ACTIVITY_KIND_RUNTIME"
    if table not in tables:
        raise RuntimeError("Nsight SQLite export has no CUDA runtime activity table")
    columns = _columns(connection, table)
    start_column = _first_column(columns, "start")
    end_column = _first_column(columns, "end")
    thread_column = _first_column(columns, "globalTid", "globalThreadId", "threadId")
    correlation_column = _first_column(columns, "correlationId")
    if None in (start_column, end_column, thread_column, correlation_column):
        raise RuntimeError(f"unsupported CUDA runtime schema: {sorted(columns)}")
    query = (
        f'SELECT "{start_column}", "{end_column}", "{thread_column}", "{correlation_column}" '
        f'FROM "{table}"'
    )
    return [
        {"start": int(row[0]), "end": int(row[1]), "thread": int(row[2]), "correlation": int(row[3])}
        for row in connection.execute(query)
        if None not in row
    ]


def _kernel_rows(
    connection: sqlite3.Connection,
    tables: set[str],
    strings: Mapping[int, str],
) -> list[Dict[str, Any]]:
    table = next(
        (name for name in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL") if name in tables),
        None,
    )
    if table is None:
        raise RuntimeError("Nsight SQLite export has no CUDA kernel activity table")
    columns = _columns(connection, table)
    start_column = _first_column(columns, "start")
    end_column = _first_column(columns, "end")
    stream_column = _first_column(columns, "streamId", "stream")
    correlation_column = _first_column(columns, "correlationId")
    name_column = _first_column(columns, "shortName", "demangledName", "name")
    if None in (start_column, end_column, stream_column, correlation_column):
        raise RuntimeError(f"unsupported CUDA kernel schema: {sorted(columns)}")
    selected = [start_column, end_column, stream_column, correlation_column]
    if name_column is not None:
        selected.append(name_column)
    query = "SELECT " + ", ".join(f'"{column}"' for column in selected) + f' FROM "{table}"'
    kernels = []
    for row in connection.execute(query):
        start, end, stream, correlation = row[:4]
        if None in (start, end, stream, correlation):
            continue
        kernels.append({
            "start": int(start),
            "end": int(end),
            "stream": int(stream),
            "correlation": int(correlation),
            "kernel_name": _resolved_string(row[4], strings) if name_column is not None else "",
        })
    return kernels


def _metric_rows(
    connection: sqlite3.Connection,
    tables: set[str],
    capture_start: int,
    capture_end: int,
) -> tuple[list[Dict[str, Any]], Dict[str, str]]:
    required = {"GPU_METRICS", "TARGET_INFO_GPU_METRICS"}
    if not required.issubset(tables):
        return [], {}
    info_columns = _columns(connection, "TARGET_INFO_GPU_METRICS")
    metric_columns = _columns(connection, "GPU_METRICS")
    metric_id = _first_column(info_columns, "metricId", "id")
    metric_name = _first_column(info_columns, "metricName", "name")
    gpu_metric_id = _first_column(metric_columns, "metricId", "id")
    timestamp = _first_column(metric_columns, "timestamp", "start")
    value = _first_column(metric_columns, "value")
    if None in (metric_id, metric_name, gpu_metric_id, timestamp, value):
        return [], {}
    names = {
        int(row[0]): str(row[1])
        for row in connection.execute(
            f'SELECT "{metric_id}", "{metric_name}" FROM "TARGET_INFO_GPU_METRICS"'
        )
    }
    desired_patterns = {
        "sm_active_pct": ("SMs Active", "sm__cycles_active"),
        "sm_issue_pct": ("SM Issue", "sm__inst_executed"),
        "tensor_active_pct": ("Tensor Active", "sm__pipe_tensor"),
        "active_sm_unused_warp_slots_pct": ("Active SM Unused Warp Slots", "tpc__warps_inactive_sm_active"),
    }
    selected: Dict[int, str] = {}
    mapping: Dict[str, str] = {}
    for identifier, name in names.items():
        lower = name.lower()
        for key, patterns in desired_patterns.items():
            if key in mapping:
                continue
            if any(pattern.lower() in lower for pattern in patterns):
                selected[identifier] = key
                mapping[key] = name
                break
    if not selected:
        return [], mapping
    placeholders = ",".join("?" for _ in selected)
    query = (
        f'SELECT "{timestamp}", "{gpu_metric_id}", "{value}" FROM "GPU_METRICS" '
        f'WHERE "{timestamp}" >= ? AND "{timestamp}" <= ? AND "{gpu_metric_id}" IN ({placeholders}) '
        f'ORDER BY "{timestamp}"'
    )
    samples_by_time: Dict[int, Dict[str, Any]] = {}
    parameters = [capture_start, capture_end, *selected.keys()]
    for sample_time, identifier, raw_value in connection.execute(query, parameters):
        key = selected.get(int(identifier))
        if key is None:
            continue
        row = samples_by_time.setdefault(int(sample_time), {"timestamp": int(sample_time)})
        try:
            row[key] = float(raw_value)
        except (TypeError, ValueError):
            continue
    return list(samples_by_time.values()), mapping


def _union_overlap(intervals: Iterable[tuple[float, float]], start: float, end: float) -> float:
    clipped = sorted(
        (max(start, left), min(end, right))
        for left, right in intervals
        if right > start and left < end
    )
    total = 0.0
    cursor: Optional[float] = None
    stop = 0.0
    for left, right in clipped:
        if right <= left:
            continue
        if cursor is None:
            cursor, stop = left, right
            continue
        if left <= stop:
            stop = max(stop, right)
        else:
            total += stop - cursor
            cursor, stop = left, right
    if cursor is not None:
        total += stop - cursor
    return total


def _mean_metric(samples: Sequence[Mapping[str, Any]], key: str, start_us: float, end_us: float) -> Optional[float]:
    values = [
        float(row[key])
        for row in samples
        if key in row and start_us <= float(row["time_us"]) <= end_us and math.isfinite(float(row[key]))
    ]
    return sum(values) / len(values) if values else None


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def normalize_nsys_sqlite(
    sqlite_path: Path,
    output_directory: Path,
    *,
    capture_frequency_hz: int,
    handoff: Optional[Mapping[str, Any]] = None,
    capture_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    with sqlite3.connect(sqlite_path) as connection:
        tables = _table_names(connection)
        strings = _string_ids(connection, tables)
        nvtx = _nvtx_events(connection, tables, strings)
        captures = [event for event in nvtx if event["text"] == PROCESSING_CAPTURE_RANGE]
        if not captures:
            raise RuntimeError("Nsight trace does not contain the PREMAT processing capture range")
        capture = captures[0]
        capture_start = int(capture["start"])
        capture_end = int(capture["end"])
        operations = []
        for index, event in enumerate(nvtx):
            parsed = _parse_operation_label(str(event["text"]))
            if parsed is None:
                continue
            if int(event["end"]) < capture_start or int(event["start"]) > capture_end:
                continue
            operations.append({"op_id": len(operations) + 1, **event, **parsed})

        runtimes = _runtime_rows(connection, tables)
        kernels = _kernel_rows(connection, tables, strings)
        correlation_to_operation: Dict[int, Dict[str, Any]] = {}
        for operation in operations:
            matching = [
                runtime
                for runtime in runtimes
                if runtime["thread"] == operation["thread"]
                and runtime["start"] >= operation["start"]
                and runtime["end"] <= operation["end"]
            ]
            for runtime in matching:
                correlation_to_operation[int(runtime["correlation"])] = operation

        interval_rows = []
        operation_kernel_intervals: Dict[int, list[tuple[float, float]]] = {}
        for kernel in kernels:
            if kernel["end"] < capture_start or kernel["start"] > capture_end:
                continue
            operation = correlation_to_operation.get(int(kernel["correlation"]))
            start_us = (max(kernel["start"], capture_start) - capture_start) / 1000.0
            end_us = (min(kernel["end"], capture_end) - capture_start) / 1000.0
            if end_us <= start_us:
                continue
            owner = str(operation["owner"]) if operation is not None else "MAIN"
            op_name = str(operation["operation"]) if operation is not None else "other"
            family = str(operation["family"]) if operation is not None else ""
            layer = operation["layer"] if operation is not None else None
            op_id = int(operation["op_id"]) if operation is not None else None
            row = {
                "start_us": start_us,
                "end_us": end_us,
                "duration_us": end_us - start_us,
                "stream": int(kernel["stream"]),
                "owner": owner,
                "layer": "" if layer is None else int(layer),
                "family": family,
                "operation": op_name,
                "kernel_name": str(kernel["kernel_name"]),
                "op_id": "" if op_id is None else op_id,
            }
            interval_rows.append(row)
            if op_id is not None:
                operation_kernel_intervals.setdefault(op_id, []).append((start_us, end_us))

        raw_samples, metric_mapping = _metric_rows(connection, tables, capture_start, capture_end)
        sample_rows = []
        for sample in raw_samples:
            sample_rows.append({
                "time_us": (int(sample["timestamp"]) - capture_start) / 1000.0,
                "sm_active_pct": sample.get("sm_active_pct", ""),
                "sm_issue_pct": sample.get("sm_issue_pct", ""),
                "tensor_active_pct": sample.get("tensor_active_pct", ""),
                "active_sm_unused_warp_slots_pct": sample.get("active_sm_unused_warp_slots_pct", ""),
            })
        if not sample_rows:
            warnings.append("No requested GPU Metrics samples were found in the Nsight export")

    premat_intervals = [
        (float(row["start_us"]), float(row["end_us"]))
        for row in interval_rows
        if row["owner"] == "PREMAT" and row["operation"] == "materialize"
    ]
    summary_rows = []
    for operation in operations:
        if operation["owner"] != "MAIN" or operation["operation"] != "consume":
            continue
        kernel_intervals = operation_kernel_intervals.get(int(operation["op_id"]), [])
        if not kernel_intervals:
            warnings.append(
                f"No CUDA kernels correlated to MAIN consume op {operation['op_id']} "
                f"{operation.get('family', '')} L{operation.get('layer', '')}"
            )
            continue
        start_us = min(left for left, _ in kernel_intervals)
        end_us = max(right for _, right in kernel_intervals)
        duration_us = end_us - start_us
        overlap_us = _union_overlap(premat_intervals, start_us, end_us)
        summary_rows.append({
            "op_id": int(operation["op_id"]),
            "layer": "" if operation["layer"] is None else int(operation["layer"]),
            "family": str(operation["family"]),
            "start_us": start_us,
            "end_us": end_us,
            "duration_ms": duration_us / 1000.0,
            "premat_overlap_ms": overlap_us / 1000.0,
            "premat_overlap_pct": 100.0 * overlap_us / duration_us if duration_us > 0.0 else 0.0,
            "sm_active_pct_mean": _mean_metric(sample_rows, "sm_active_pct", start_us, end_us),
            "sm_issue_pct_mean": _mean_metric(sample_rows, "sm_issue_pct", start_us, end_us),
            "tensor_active_pct_mean": _mean_metric(sample_rows, "tensor_active_pct", start_us, end_us),
            "active_sm_unused_warp_slots_pct_mean": _mean_metric(
                sample_rows, "active_sm_unused_warp_slots_pct", start_us, end_us
            ),
        })

    sample_fields = (
        "time_us", "sm_active_pct", "sm_issue_pct", "tensor_active_pct",
        "active_sm_unused_warp_slots_pct",
    )
    interval_fields = (
        "start_us", "end_us", "duration_us", "stream", "owner", "layer", "family",
        "operation", "kernel_name", "op_id",
    )
    summary_fields = (
        "op_id", "layer", "family", "start_us", "end_us", "duration_ms",
        "premat_overlap_ms", "premat_overlap_pct", "sm_active_pct_mean", "sm_issue_pct_mean",
        "tensor_active_pct_mean", "active_sm_unused_warp_slots_pct_mean",
    )
    _write_csv(output_directory / "processing_samples.csv", sample_rows, sample_fields)
    _write_csv(output_directory / "processing_intervals.csv", interval_rows, interval_fields)
    _write_csv(output_directory / "processing_summary.csv", summary_rows, summary_fields)

    metadata = {
        "schema_version": PROCESSING_SCHEMA_VERSION,
        "capture_frequency_hz": int(capture_frequency_hz),
        "capture_duration_ms": (capture_end - capture_start) / 1_000_000.0,
        "metric_mapping": metric_mapping,
        "warnings": warnings,
        "capture": dict(capture_metadata or {}),
        "run": dict(handoff or {}),
        "files": {
            "samples": "processing_samples.csv",
            "intervals": "processing_intervals.csv",
            "summary": "processing_summary.csv",
            "metadata": "processing_metadata.json",
            "bundle": "processing_bundle.zip",
            "raw_trace": "processing_trace.nsys-rep",
        },
    }
    (output_directory / "processing_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    processing_data = {
        "metadata": metadata,
        "samples": sample_rows,
        "intervals": interval_rows,
        "summary": summary_rows,
    }
    (output_directory / "processing_data.json").write_text(
        json.dumps(processing_data, separators=(",", ":"), allow_nan=False)
    )
    with zipfile.ZipFile(output_directory / "processing_bundle.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in (
            "processing_samples.csv",
            "processing_intervals.csv",
            "processing_summary.csv",
            "processing_metadata.json",
            "processing_data.json",
        ):
            archive.write(output_directory / name, arcname=name)
    return processing_data


def maybe_reexec_under_nsys(arguments: Sequence[str], *, entrypoint: Path) -> Optional[int]:
    requested, frequency = processing_requested_from_argv(arguments)
    if not requested or os.environ.get(_PROCESSING_CHILD_ENV) == "1":
        return None
    nsys = _find_nsys()
    if nsys is None:
        raise RuntimeError(
            "--premat_processing_logging enabled requires NVIDIA Nsight Systems (nsys) on PATH"
        )
    temporary_root = Path(tempfile.mkdtemp(prefix="thog2-premat-processing-"))
    report_base = temporary_root / "processing_trace"
    handoff_path = temporary_root / "handoff.json"
    capture_metadata_path = temporary_root / "capture.json"
    environment = dict(os.environ)
    environment[_PROCESSING_CHILD_ENV] = "1"
    environment[_PROCESSING_HANDOFF_ENV] = str(handoff_path)
    environment[_PROCESSING_CAPTURE_METADATA_ENV] = str(capture_metadata_path)
    command = [
        nsys,
        "profile",
        "--trace=cuda,nvtx",
        "--capture-range=nvtx",
        "--capture-range-end=stop",
        f"--nvtx-capture={PROCESSING_CAPTURE_RANGE}",
        "--gpu-metrics-devices=cuda-visible",
        f"--gpu-metrics-frequency={frequency}",
        "--force-overwrite=true",
        f"--output={report_base}",
        sys.executable,
        str(Path(entrypoint).resolve()),
        *arguments,
    ]
    print(
        f"THOG2 PREMAT processing capture: Nsight Systems @ {frequency} Hz; "
        "capturing one forward microstep",
        flush=True,
    )
    completed = subprocess.run(command, env=environment)
    if completed.returncode != 0:
        return int(completed.returncode)
    if not handoff_path.exists():
        raise RuntimeError("PREMAT processing capture completed but no INSTRA run handoff was written")
    handoff = json.loads(handoff_path.read_text())
    run_directory = Path(handoff["run_directory"])
    processing_directory = run_directory / "processing"
    processing_directory.mkdir(parents=True, exist_ok=True)
    report_path = report_base.with_suffix(".nsys-rep")
    if not report_path.exists():
        reports = list(temporary_root.glob("*.nsys-rep"))
        if len(reports) != 1:
            raise RuntimeError("Nsight Systems did not produce exactly one .nsys-rep trace")
        report_path = reports[0]
    sqlite_path = temporary_root / "processing_trace.sqlite"
    export = subprocess.run(
        [
            nsys,
            "export",
            "--type=sqlite",
            "--force-overwrite=true",
            f"--output={sqlite_path}",
            str(report_path),
        ]
    )
    if export.returncode != 0 or not sqlite_path.exists():
        raise RuntimeError("Nsight Systems SQLite export failed for PREMAT processing capture")
    capture_metadata = (
        json.loads(capture_metadata_path.read_text())
        if capture_metadata_path.exists()
        else {}
    )
    normalize_nsys_sqlite(
        sqlite_path,
        processing_directory,
        capture_frequency_hz=frequency,
        handoff=handoff,
        capture_metadata=capture_metadata,
    )
    shutil.copy2(report_path, processing_directory / "processing_trace.nsys-rep")
    print(
        "THOG2 PREMAT processing data: "
        f"{processing_directory / 'processing_bundle.zip'}",
        flush=True,
    )
    return int(completed.returncode)


__all__ = [
    "PROCESSING_DEFAULT_CAPTURE_FREQUENCY_HZ",
    "maybe_reexec_under_nsys",
    "normalize_nsys_sqlite",
    "processing_capture_scope",
    "processing_operation_pop",
    "processing_operation_push",
    "processing_operation_range",
    "processing_requested_from_argv",
    "register_processing_handoff",
    "should_capture_processing_forward",
    "validate_processing_configuration",
]
# ^^^ THOG
'''


PROCESSING_JS = r'''// vvv THOG
"use strict";

const processing_view = {
  run_id: null,
  revision: null,
  timer: null,
};

function processing_escape(value) {
  return String(value ?? "").replace(/[&<>"']/g, character => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[character]));
}

function processing_current_run() {
  return String(app.current_run_id || "");
}

function processing_download_url(path) {
  return `/api/local-file?run=${encodeURIComponent(processing_current_run())}&path=${encodeURIComponent(`processing/${path}`)}&download=1`;
}

function processing_set_downloads(metadata) {
  const files = metadata?.files || {};
  const links = {
    processing_download_bundle: files.bundle,
    processing_download_samples: files.samples,
    processing_download_intervals: files.intervals,
    processing_download_summary: files.summary,
    processing_download_metadata: files.metadata,
    processing_download_raw: files.raw_trace,
  };
  for (const [id, filename] of Object.entries(links)) {
    const element = by_id(id);
    if (!element) continue;
    if (!filename) {
      element.hidden = true;
      continue;
    }
    element.href = processing_download_url(filename);
    element.hidden = false;
  }
}

function processing_trace_groups(intervals) {
  const order = ["MAIN:consume", "MAIN:materialize", "MAIN:other", "PREMAT:materialize"];
  const groups = new Map();
  for (const row of intervals || []) {
    const owner = String(row.owner || "MAIN");
    const operation = String(row.operation || "other");
    const key = `${owner}:${operation}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }
  return [...groups.entries()].sort((left, right) => {
    const a = order.indexOf(left[0]);
    const b = order.indexOf(right[0]);
    return (a < 0 ? 99 : a) - (b < 0 ? 99 : b);
  });
}

function processing_render_timeline(payload) {
  const traces = [];
  for (const [key, rows] of processing_trace_groups(payload.intervals)) {
    const x = [];
    const y = [];
    const hover = [];
    const lane = key.startsWith("PREMAT:") ? 0 : 1;
    for (const row of rows) {
      const start = Number(row.start_us) / 1000.0;
      const end = Number(row.end_us) / 1000.0;
      const detail = `${row.owner} · ${row.operation}${row.family ? ` · ${row.family}` : ""}${row.layer !== "" && row.layer !== null ? ` · L${Number(row.layer) + 1}` : ""}<br>${processing_escape(row.kernel_name)}<br>${(end - start).toFixed(4)} ms`;
      x.push(start, end, null);
      y.push(lane, lane, null);
      hover.push(detail, detail, "");
    }
    traces.push({
      type: "scattergl",
      mode: "lines",
      name: key.replace(":", " "),
      x,
      y,
      hovertext: hover,
      hoverinfo: "text",
      line: {width: key.startsWith("PREMAT:") ? 7 : 6},
      yaxis: "y",
    });
  }
  const metric_specs = [
    ["sm_active_pct", "SM Active"],
    ["sm_issue_pct", "SM Issue"],
    ["tensor_active_pct", "Tensor Active"],
    ["active_sm_unused_warp_slots_pct", "Unused warp slots"],
  ];
  for (const [key, label] of metric_specs) {
    const rows = (payload.samples || []).filter(row => Number.isFinite(Number(row[key])));
    if (!rows.length) continue;
    traces.push({
      type: "scattergl",
      mode: "lines",
      name: label,
      x: rows.map(row => Number(row.time_us) / 1000.0),
      y: rows.map(row => Number(row[key])),
      hovertemplate: `${label}: %{y:.1f}%<br>%{x:.3f} ms<extra></extra>`,
      yaxis: "y2",
    });
  }
  const capture_ms = Number(payload.metadata?.capture_duration_ms || 0);
  Plotly.react("processing_timeline_plot", traces, {
    margin: {l: 70, r: 34, t: 12, b: 48},
    hovermode: "x unified",
    legend: {orientation: "h", y: 1.08},
    xaxis: {title: "capture time (ms)", range: capture_ms > 0 ? [0, capture_ms] : undefined},
    yaxis: {
      domain: [0.0, 0.24],
      tickmode: "array",
      tickvals: [0, 1],
      ticktext: ["PREMAT", "MAIN"],
      range: [-0.5, 1.5],
      fixedrange: true,
    },
    yaxis2: {
      domain: [0.34, 1.0],
      title: "processing (%)",
      range: [0, 100],
      fixedrange: true,
    },
  }, plot_config);
}

function processing_render_contention(payload) {
  const families = [...new Set((payload.summary || []).map(row => String(row.family || "?")))];
  const traces = families.map(family => {
    const rows = payload.summary.filter(row => String(row.family || "?") === family);
    return {
      type: "scatter",
      mode: "markers",
      name: family,
      x: rows.map(row => Number(row.premat_overlap_pct)),
      y: rows.map(row => Number(row.duration_ms)),
      text: rows.map(row => `L${Number(row.layer) + 1}`),
      hovertemplate: `${family} %{text}<br>PREMAT overlap %{x:.1f}%<br>Main duration %{y:.4f} ms<extra></extra>`,
    };
  });
  Plotly.react("processing_contention_plot", traces, {
    margin: {l: 70, r: 24, t: 12, b: 58},
    xaxis: {title: "Main operation overlapped by PREMAT (%)", range: [0, 100]},
    yaxis: {title: "Main operation duration (ms)"},
    legend: {orientation: "h", y: 1.08},
  }, plot_config);
}

function processing_mean(rows, key) {
  const values = rows.map(row => Number(row[key])).filter(Number.isFinite);
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

function processing_format(value, digits = 2, suffix = "") {
  return Number.isFinite(value) ? `${value.toFixed(digits)}${suffix}` : "—";
}

function processing_render_summary(payload) {
  const body = by_id("processing_summary_body");
  const families = [...new Set((payload.summary || []).map(row => String(row.family || "?")))];
  body.innerHTML = families.map(family => {
    const rows = payload.summary.filter(row => String(row.family || "?") === family);
    return `<tr><td><strong>${processing_escape(family)}</strong></td><td>${rows.length}</td><td>${processing_format(processing_mean(rows, "duration_ms"), 4, " ms")}</td><td>${processing_format(processing_mean(rows, "premat_overlap_pct"), 1, "%")}</td><td>${processing_format(processing_mean(rows, "sm_active_pct_mean"), 1, "%")}</td><td>${processing_format(processing_mean(rows, "sm_issue_pct_mean"), 1, "%")}</td><td>${processing_format(processing_mean(rows, "tensor_active_pct_mean"), 1, "%")}</td><td>${processing_format(processing_mean(rows, "active_sm_unused_warp_slots_pct_mean"), 1, "%")}</td></tr>`;
  }).join("") || '<tr><td colspan="8">No labelled Main consuming operations in this capture.</td></tr>';
}

function processing_render(payload) {
  const group = by_id("processing_chart_group");
  group.hidden = false;
  const capture = payload.metadata?.capture || {};
  by_id("processing_step").textContent = String(capture.optimizer_update ?? "—");
  const warning_count = (payload.metadata?.warnings || []).length;
  by_id("processing_status").textContent = `${Number(payload.metadata?.capture_frequency_hz || 0).toLocaleString()} Hz · ${Number(payload.metadata?.capture_duration_ms || 0).toFixed(2)} ms capture${warning_count ? ` · ${warning_count} warning${warning_count === 1 ? "" : "s"}` : ""}`;
  processing_set_downloads(payload.metadata);
  processing_render_timeline(payload);
  processing_render_contention(payload);
  processing_render_summary(payload);
}

async function processing_refresh() {
  const run_id = processing_current_run();
  if (!run_id) {
    by_id("processing_chart_group").hidden = true;
    processing_view.run_id = null;
    processing_view.revision = null;
    return;
  }
  try {
    const payload = await fetch_json(`/api/processing?run=${encodeURIComponent(run_id)}`);
    if (!payload.available) {
      by_id("processing_chart_group").hidden = true;
      processing_view.run_id = run_id;
      processing_view.revision = null;
      return;
    }
    if (processing_view.run_id === run_id && processing_view.revision === payload.revision) return;
    processing_view.run_id = run_id;
    processing_view.revision = payload.revision;
    processing_render(payload.data);
  } catch (error) {
    console.warn("Processing data refresh failed", error);
  }
}

processing_view.timer = window.setInterval(processing_refresh, 1500);
window.addEventListener("load", processing_refresh);
// ^^^ THOG
'''


PROCESSING_CSS = r'''/* vvv THOG */
.processing-group { padding-bottom: 18px; }
.processing-header { align-items: center; gap: 14px; }
.processing-header-copy { display: flex; flex-direction: column; gap: 3px; }
.processing-status { color: var(--muted, #77808d); font-size: 12px; }
.processing-downloads { margin-left: auto; display: flex; flex-wrap: wrap; gap: 7px; }
.processing-downloads a { border: 1px solid rgba(127,127,127,.28); border-radius: 6px; padding: 5px 8px; text-decoration: none; color: inherit; font-size: 12px; }
.processing-grid { display: grid; grid-template-columns: minmax(0, 1.6fr) minmax(360px, 1fr); gap: 12px; padding: 12px; }
.processing-card { min-width: 0; border: 1px solid rgba(127,127,127,.18); border-radius: 8px; padding: 10px; }
.processing-card h2 { margin: 0 0 8px; font-size: 14px; }
.processing-plot { width: 100%; height: 420px; }
.processing-contention-plot { height: 300px; }
.processing-summary-wrap { overflow-x: auto; margin-top: 10px; }
.processing-summary { width: 100%; border-collapse: collapse; font-size: 11px; }
.processing-summary th, .processing-summary td { padding: 5px 7px; border-bottom: 1px solid rgba(127,127,127,.16); text-align: right; white-space: nowrap; }
.processing-summary th:first-child, .processing-summary td:first-child { text-align: left; }
@media (max-width: 1180px) { .processing-grid { grid-template-columns: 1fr; } }
/* ^^^ THOG */
'''


PROCESSING_TEST = r'''# vvv THOG
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import pytest

from sheet.premat_processing import (
    normalize_nsys_sqlite,
    processing_requested_from_argv,
    validate_processing_configuration,
)


def test_processing_cli_surface_exact_names() -> None:
    enabled, frequency = processing_requested_from_argv([
        "--premat_processing_logging", "enabled",
        "--premat_processing_logging_capture_frequency_hz", "12345",
    ])
    assert enabled is True
    assert frequency == 12345


def test_processing_configuration_is_cuda_but_not_premat_dependent() -> None:
    validate_processing_configuration("enabled", 10000, "cuda")
    validate_processing_configuration("disabled", 10, "cpu")
    with pytest.raises(ValueError, match="CUDA"):
        validate_processing_configuration("enabled", 10000, "cpu")
    with pytest.raises(ValueError, match="capture_frequency_hz"):
        validate_processing_configuration("enabled", 9, "cuda")


def _synthetic_nsys_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, globalTid INTEGER, text TEXT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, streamId INTEGER, correlationId INTEGER, shortName INTEGER);
            CREATE TABLE TARGET_INFO_GPU_METRICS (metricId INTEGER, metricName TEXT);
            CREATE TABLE GPU_METRICS (timestamp INTEGER, metricId INTEGER, value REAL);
            """
        )
        connection.executemany("INSERT INTO StringIds VALUES (?, ?)", [(1, "main_kernel"), (2, "premat_kernel")])
        connection.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?)",
            [
                (1000, 10000, 7, "THOG2_PREMAT_PROCESSING_CAPTURE"),
                (1800, 2400, 7, "THOG2_PROCESSING|owner=MAIN|operation=consume|family=QKV|layer=0"),
                (2000, 2600, 7, "THOG2_PROCESSING|owner=PREMAT|operation=materialize|family=DOWN|layer=1"),
            ],
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?)",
            [(1900, 1950, 7, 11), (2100, 2150, 7, 12)],
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?)",
            [(2500, 4500, 3, 11, 1), (3500, 4300, 9, 12, 2)],
        )
        connection.executemany(
            "INSERT INTO TARGET_INFO_GPU_METRICS VALUES (?, ?)",
            [
                (1, "SMs Active %"),
                (2, "SM Issue %"),
                (3, "Tensor Active %"),
                (4, "Active SM Unused Warp Slots %"),
            ],
        )
        for timestamp, values in ((3000, (80, 70, 90, 20)), (4000, (90, 75, 95, 15))):
            connection.executemany(
                "INSERT INTO GPU_METRICS VALUES (?, ?, ?)",
                [(timestamp, metric_id, value) for metric_id, value in enumerate(values, start=1)],
            )
        connection.commit()


def test_processing_normalizer_emits_graph_and_download_data(tmp_path: Path) -> None:
    database = tmp_path / "trace.sqlite"
    output = tmp_path / "processing"
    _synthetic_nsys_database(database)
    payload = normalize_nsys_sqlite(
        database,
        output,
        capture_frequency_hz=10000,
        handoff={"run_name": "fixture"},
        capture_metadata={"optimizer_update": 10, "micro_step": 1},
    )
    assert payload["metadata"]["capture_frequency_hz"] == 10000
    assert payload["samples"][0]["sm_active_pct"] == 80.0
    assert {row["owner"] for row in payload["intervals"]} == {"MAIN", "PREMAT"}
    assert len(payload["summary"]) == 1
    summary = payload["summary"][0]
    assert summary["family"] == "QKV"
    assert summary["duration_ms"] == pytest.approx(0.002)
    assert summary["premat_overlap_ms"] == pytest.approx(0.0008)
    assert summary["premat_overlap_pct"] == pytest.approx(40.0)
    expected = {
        "processing_samples.csv",
        "processing_intervals.csv",
        "processing_summary.csv",
        "processing_metadata.json",
        "processing_data.json",
        "processing_bundle.zip",
    }
    assert expected.issubset({path.name for path in output.iterdir()})
    with (output / "processing_summary.csv").open() as source:
        rows = list(csv.DictReader(source))
    assert rows[0]["family"] == "QKV"
# ^^^ THOG
'''


# Configuration: public run config, shared trainer config and exact CLI names.
replace_once(
    "sheet/run_config.py",
    '    premat_logging: str = "disabled"\n    premat_instra: str = "disabled"\n    premat_retain_detailed_premat_history: bool = False\n',
    '    premat_logging: str = "disabled"\n    premat_instra: str = "disabled"\n    # vvv THOG device-processing capture is execution instrumentation and remains independent of PREMAT scheduling\n    premat_processing_logging: str = "disabled"\n    premat_processing_logging_capture_frequency_hz: int = 10000\n    # ^^^ THOG\n    premat_retain_detailed_premat_history: bool = False\n',
)
replace_once(
    "sheet/run_config.py",
    '        if self.premat == "enabled" and not str(self.device).startswith("cuda"):\n            raise ValueError("--premat enabled requires a CUDA device")\n',
    '        if self.premat == "enabled" and not str(self.device).startswith("cuda"):\n            raise ValueError("--premat enabled requires a CUDA device")\n        # vvv THOG validate opt-in whole-GPU processing capture without requiring PREMAT itself\n        from .premat_processing import validate_processing_configuration\n        validate_processing_configuration(\n            self.premat_processing_logging,\n            self.premat_processing_logging_capture_frequency_hz,\n            self.device,\n        )\n        # ^^^ THOG\n',
)
replace_once(
    "sheet/run_config.py",
    '            premat_logging=self.premat_logging,\n            premat_instra=self.premat_instra,\n            premat_retain_detailed_premat_history=self.premat_retain_detailed_premat_history,\n',
    '            premat_logging=self.premat_logging,\n            premat_instra=self.premat_instra,\n            premat_processing_logging=self.premat_processing_logging,\n            premat_processing_logging_capture_frequency_hz=self.premat_processing_logging_capture_frequency_hz,\n            premat_retain_detailed_premat_history=self.premat_retain_detailed_premat_history,\n',
)
replace_once(
    "sheet/run_config.py",
    '            or self.premat_logging != "disabled"\n            or self.premat_instra != "disabled"\n        ):\n',
    '            or self.premat_logging != "disabled"\n            or self.premat_instra != "disabled"\n            or self.premat_processing_logging != "disabled"\n        ):\n',
)
replace_once(
    "sheet/run_config.py",
    '            shadow_fragment = "SHADOW_" if self.premat_enable_shadow_mode else ""\n            premat_fragment = (\n',
    '            shadow_fragment = "SHADOW_" if self.premat_enable_shadow_mode else ""\n            processing_fragment = (\n                f"_PROC{self.premat_processing_logging_capture_frequency_hz}"\n                if self.premat_processing_logging == "enabled"\n                else ""\n            )\n            premat_fragment = (\n',
)
replace_once(
    "sheet/run_config.py",
    '                f"L{self.premat_logging[0].upper()}_"\n                f"I{self.premat_instra[0].upper()}"\n',
    '                f"L{self.premat_logging[0].upper()}_"\n                f"I{self.premat_instra[0].upper()}"\n                f"{processing_fragment}"\n',
)

replace_once(
    "sheet/training_config.py",
    'EXECUTION_OVERRIDE_FIELDS = {"instrumentation__optimizer_histories__full_matrix_every_n_steps", "device", "dtype", "max_updates", "max_wall_minutes", "eval_interval", "eval_batches", "checkpoint_interval", "checkpoint_segment_size", "out_dir", "log_interval", "nonfinite_update_policy", "max_nonfinite_update_skips", "premat_enable_gpu_timing_diagnostic"}\n',
    'EXECUTION_OVERRIDE_FIELDS = {"instrumentation__optimizer_histories__full_matrix_every_n_steps", "device", "dtype", "max_updates", "max_wall_minutes", "eval_interval", "eval_batches", "checkpoint_interval", "checkpoint_segment_size", "out_dir", "log_interval", "nonfinite_update_policy", "max_nonfinite_update_skips", "premat_enable_gpu_timing_diagnostic", "premat_processing_logging", "premat_processing_logging_capture_frequency_hz"}                         # <<< THOG processing capture is diagnostic execution state, never checkpoint model identity\n',
)
replace_once(
    "sheet/training_config.py",
    '    premat_logging: str = "disabled"\n    premat_instra: str = "disabled"\n    premat_retain_detailed_premat_history: bool = False\n',
    '    premat_logging: str = "disabled"\n    premat_instra: str = "disabled"\n    # vvv THOG Nsight-backed processing capture is execution instrumentation and works with NOMAT\n    premat_processing_logging: str = "disabled"\n    premat_processing_logging_capture_frequency_hz: int = 10000\n    # ^^^ THOG\n    premat_retain_detailed_premat_history: bool = False\n',
)
replace_once(
    "sheet/training_config.py",
    '        if self.premat == "enabled" and not str(self.device).startswith("cuda"):\n            raise ValueError("--premat enabled requires a CUDA device")\n',
    '        if self.premat == "enabled" and not str(self.device).startswith("cuda"):\n            raise ValueError("--premat enabled requires a CUDA device")\n        # vvv THOG validate processing capture independently of PREMAT scheduling\n        from .premat_processing import validate_processing_configuration\n        validate_processing_configuration(\n            self.premat_processing_logging,\n            self.premat_processing_logging_capture_frequency_hz,\n            self.device,\n        )\n        # ^^^ THOG\n',
)

replace_once(
    "run_thog2_owt_core.py",
    '    parser.add_argument("--premat_logging", choices=("enabled", "disabled"), default="disabled")\n    parser.add_argument("--premat_instra", choices=("enabled", "disabled"), default="disabled")\n    parser.add_argument("--premat_retain_detailed_premat_history", type=_true_false, default=False, metavar="true|false")\n',
    '    parser.add_argument("--premat_logging", choices=("enabled", "disabled"), default="disabled")\n    parser.add_argument("--premat_instra", choices=("enabled", "disabled"), default="disabled")\n    # vvv THOG exact public processing-diagnostic names; disabled/default frequency preserves existing runs\n    parser.add_argument("--premat_processing_logging", choices=("enabled", "disabled"), default="disabled")\n    parser.add_argument("--premat_processing_logging_capture_frequency_hz", type=int, default=10000)\n    # ^^^ THOG\n    parser.add_argument("--premat_retain_detailed_premat_history", type=_true_false, default=False, metavar="true|false")\n',
)
replace_once(
    "run_thog2_owt_core.py",
    '        premat_logging=arguments.premat_logging,\n        premat_instra=arguments.premat_instra,\n        premat_retain_detailed_premat_history=arguments.premat_retain_detailed_premat_history,\n',
    '        premat_logging=arguments.premat_logging,\n        premat_instra=arguments.premat_instra,\n        premat_processing_logging=arguments.premat_processing_logging,\n        premat_processing_logging_capture_frequency_hz=arguments.premat_processing_logging_capture_frequency_hz,\n        premat_retain_detailed_premat_history=arguments.premat_retain_detailed_premat_history,\n',
)

# Public runner self-wraps in nsys before importing torch-heavy runner modules.
replace_once(
    "run_thog2_owt.py",
    'from typing import Any, Mapping, Optional, Sequence\n\n\n# vvv THOG make --print-geometry-registry',
    'from typing import Any, Mapping, Optional, Sequence\n\n# vvv THOG opt-in processing capture re-execs this public runner under Nsight before torch/CUDA initialization\nfrom sheet.premat_processing import maybe_reexec_under_nsys\n_processing_exit_code = maybe_reexec_under_nsys(sys.argv[1:], entrypoint=Path(__file__))\nif _processing_exit_code is not None:\n    raise SystemExit(_processing_exit_code)\n# ^^^ THOG\n\n\n# vvv THOG make --print-geometry-registry',
)

# Trainer capture bounds one forward microstep; the ending synchronize is outside the measured forward schedule.
replace_once(
    "sheet/trainer_step.py",
    'from .training_model import TrainingModel\n',
    'from .training_model import TrainingModel\n# vvv THOG opt-in bounded Nsight processing capture\nfrom .premat_processing import processing_capture_scope\n# ^^^ THOG\n',
)
replace_once(
    "sheet/trainer_step.py",
    '            with self.distributed.no_sync_context(self.model, enabled=not should_sync):\n                with self.autocast_context():\n',
    '            with self.distributed.no_sync_context(self.model, enabled=not should_sync):\n                # vvv THOG capture exactly one first accumulation forward; backward/checkpoint replay remains outside the Nsight range\n                with processing_capture_scope(\n                    enabled=(str(getattr(self.config, "premat_processing_logging", "disabled")) == "enabled"),\n                    completed_updates=self.state.completed_updates,\n                    max_updates=self.config.max_updates,\n                    log_interval=self.config.log_interval,\n                    micro_step=micro_step,\n                    device=self.device,\n                ):\n                    with self.autocast_context():\n                # ^^^ THOG\n',
)
# The inserted extra nesting requires indenting the existing autocast body until the first post-autocast statement.
trainer_path = ROOT / "sheet/trainer_step.py"
trainer_text = trainer_path.read_text()
needle = '                    with self.autocast_context():\n                # ^^^ THOG\n'
start = trainer_text.index(needle) + len(needle)
end_marker = '                total_loss += float(loss.detach().item())\n'
end = trainer_text.index(end_marker, start)
body = trainer_text[start:end]
if not body.strip():
    raise RuntimeError("trainer processing capture insertion found empty autocast body")
body = ''.join(('    ' + line if line.strip() else line) for line in body.splitlines(keepends=True))
trainer_text = trainer_text[:start] + body + trainer_text[end:]
trainer_path.write_text(trainer_text)

# Model operation NVTX markers work for both NOMAT and PREMAT; existing CUDA-event forensics remain untouched.
replace_once(
    "sheet/model.py",
    'from .premat import PrematRuntime, validate_premat_configuration\n',
    'from .premat import PrematRuntime, validate_premat_configuration\nfrom .premat_processing import processing_operation_pop, processing_operation_push, processing_operation_range                             # <<< THOG semantic NVTX labels for INSTRA Processing\n',
)
replace_once(
    "sheet/model.py",
    '    def _premat_weight(self, family: str, layer_index: int) -> Tensor:\n        if self._premat_runtime is None or not self._premat_runtime.active:\n            if self._premat_runtime is None:\n                return self._premat_materialize_candidate(family, layer_index)\n',
    '    def _premat_weight(self, family: str, layer_index: int) -> Tensor:\n        if self._premat_runtime is None or not self._premat_runtime.active:\n            if self._premat_runtime is None:\n                with processing_operation_range("MAIN", "materialize", family=family, layer_index=layer_index):\n                    return self._premat_materialize_candidate(family, layer_index)\n',
)
replace_once(
    "sheet/model.py",
    '    def _premat_forensic_main_work_start(self, family: str, layer_index: int) -> None:\n        runtime = self._premat_runtime\n',
    '    def _premat_forensic_main_work_start(self, family: str, layer_index: int) -> None:\n        self._processing_main_work_marker = processing_operation_push("MAIN", "consume", family=family, layer_index=layer_index)\n        runtime = self._premat_runtime\n',
)
replace_once(
    "sheet/model.py",
    '    def _premat_forensic_main_work_end(self, family: str, layer_index: int) -> None:\n        runtime = self._premat_runtime\n',
    '    def _premat_forensic_main_work_end(self, family: str, layer_index: int) -> None:\n        processing_operation_pop(bool(getattr(self, "_processing_main_work_marker", False)))\n        self._processing_main_work_marker = False\n        runtime = self._premat_runtime\n',
)

# Real PREMAT and Main fallback materialisation ranges identify reconstruction kernels by stream/owner.
replace_once(
    "sheet/premat.py",
    'import torch\n',
    'import torch\nfrom .premat_processing import processing_operation_range                                                     # <<< THOG semantic processing ranges are inert outside the selected capture\n',
)
replace_once(
    "sheet/premat.py",
    '                candidate.tensor = self._materialize(candidate.family, candidate.layer_index)\n',
    '                with processing_operation_range("MAIN", "materialize", family=candidate.family, layer_index=candidate.layer_index):\n                    candidate.tensor = self._materialize(candidate.family, candidate.layer_index)\n',
)
replace_once(
    "sheet/premat.py",
    '                            with torch.no_grad():\n                                candidate.tensor = self._materialize(\n                                    candidate.family,\n                                    candidate.layer_index,\n                                )\n',
    '                            with torch.no_grad():\n                                with processing_operation_range("PREMAT", "materialize", family=candidate.family, layer_index=candidate.layer_index):\n                                    candidate.tensor = self._materialize(\n                                        candidate.family,\n                                        candidate.layer_index,\n                                    )\n',
)
replace_once(
    "sheet/premat.py",
    '        return self._materialize(family, layer_index)\n\n    def consumed(self, family: str, layer_index: int) -> None:\n',
    '        with processing_operation_range("MAIN", "materialize", family=family, layer_index=layer_index):\n            return self._materialize(family, layer_index)\n\n    def consumed(self, family: str, layer_index: int) -> None:\n',
)

# Telemetry writes the run-directory handoff even when Premat Recapitulation itself is disabled.
replace_once(
    "sheet/wandb_telemetry.py",
    'from .stage6_source import (\n',
    'from .premat_processing import register_processing_handoff                                             # <<< THOG parent nsys wrapper receives the concrete INSTRA run directory\nfrom .stage6_source import (\n',
)
replace_once(
    "sheet/wandb_telemetry.py",
    '    # vvv THOG detailed Premat Instra captures exactly the first accumulation\n',
    '    # vvv THOG processing diagnostics need the local run directory regardless of Premat Recapitulation or scalar backend\n    if (\n        trainer.distributed.is_primary\n        and str(getattr(trainer.config, "premat_processing_logging", "disabled")) == "enabled"\n    ):\n        processing_store = ensure_local_chart_store(telemetry)\n        register_processing_handoff(\n            processing_store.path.parent,\n            run_name=telemetry.name,\n            config=telemetry.config,\n        )\n    # ^^^ THOG\n\n    # vvv THOG detailed Premat Instra captures exactly the first accumulation\n',
)

# Dashboard backend serves one stable JSON payload and relies on the existing secure local-file download endpoint.
replace_once(
    "run_thog2_local_dashboard.py",
    '    "dashboard_premat.css",\n    "dashboard_premat.js",\n',
    '    "dashboard_premat.css",\n    "dashboard_premat.js",\n    "dashboard_processing.css",\n    "dashboard_processing.js",\n',
)
replace_once(
    "run_thog2_local_dashboard.py",
    '    def figures(self) -> Dict[str, Any]:\n',
    '    # vvv THOG Nsight-normalized processing data lives beside charts.sqlite3 and is immutable after capture\n    def processing(self) -> Dict[str, Any]:\n        path = self.database_path.parent / "processing" / "processing_data.json"\n        if not path.is_file():\n            return {"available": False, "revision": None, "data": None}\n        stat_result = path.stat()\n        return {\n            "available": True,\n            "revision": f"{stat_result.st_mtime_ns}:{stat_result.st_size}",\n            "data": json.loads(path.read_text()),\n        }\n    # ^^^ THOG\n\n    def figures(self) -> Dict[str, Any]:\n',
)
replace_once(
    "run_thog2_local_dashboard.py",
    '                if path in {"/api/status", "/api/figures", "/api/premat"}:\n',
    '                if path in {"/api/status", "/api/figures", "/api/premat", "/api/processing"}:\n',
)
replace_once(
    "run_thog2_local_dashboard.py",
    '                    elif path == "/api/figures":\n                        value = state.figures()\n                    else:\n',
    '                    elif path == "/api/figures":\n                        value = state.figures()\n                    elif path == "/api/processing":\n                        value = state.processing()\n                    else:\n',
)

# INSTRA group and assets.
replace_once(
    "sheet/local_dashboard_assets/index.html",
    '  <link rel="stylesheet" href="/assets/dashboard_premat.css">                                                                                           <!-- THOG dedicated Premat view -->\n',
    '  <link rel="stylesheet" href="/assets/dashboard_premat.css">                                                                                           <!-- THOG dedicated Premat view -->\n  <link rel="stylesheet" href="/assets/dashboard_processing.css">                                                                                        <!-- THOG Processing view -->\n',
)
replace_once(
    "sheet/local_dashboard_assets/index.html",
    '  <script src="/assets/dashboard_premat.js" defer></script>                                                                                              <!-- THOG bounded live Premat polling -->\n',
    '  <script src="/assets/dashboard_premat.js" defer></script>                                                                                              <!-- THOG bounded live Premat polling -->\n  <script src="/assets/dashboard_processing.js" defer></script>                                                                                           <!-- THOG Nsight-normalized processing plots -->\n',
)
processing_html = '''        <!-- vvv THOG whole-GPU processing diagnostics aligned to semantic MAIN/PREMAT kernel intervals -->\n        <section class="chart-group processing-group" id="processing_chart_group" data-chart-group="processing" hidden>\n          <header class="chart-group-header processing-header">\n            <div class="processing-header-copy"><strong>Processing - Step <span id="processing_step">—</span></strong><span class="processing-status" id="processing_status">Waiting for capture</span></div>\n            <div class="processing-downloads">\n              <a id="processing_download_bundle" hidden>Bundle ZIP</a><a id="processing_download_samples" hidden>Samples CSV</a><a id="processing_download_intervals" hidden>Intervals CSV</a><a id="processing_download_summary" hidden>Summary CSV</a><a id="processing_download_metadata" hidden>Metadata JSON</a><a id="processing_download_raw" hidden>Raw nsys</a>\n            </div>\n          </header>\n          <div class="processing-grid">\n            <article class="processing-card"><h2>GPU processing timeline</h2><div class="processing-plot" id="processing_timeline_plot"></div></article>\n            <article class="processing-card"><h2>Main duration vs PREMAT overlap</h2><div class="processing-plot processing-contention-plot" id="processing_contention_plot"></div><div class="processing-summary-wrap"><table class="processing-summary"><thead><tr><th>Matrix</th><th>N</th><th>Main mean</th><th>PREMAT overlap</th><th>SM active</th><th>SM issue</th><th>Tensor active</th><th>Unused warp slots</th></tr></thead><tbody id="processing_summary_body"></tbody></table></div></article>\n          </div>\n        </section>\n        <!-- ^^^ THOG -->\n'''
replace_once(
    "sheet/local_dashboard_assets/index.html",
    '        <!-- ^^^ THOG -->\n        <section class="chart-group" id="depth_chart_group" data-chart-group="depth">\n',
    '        <!-- ^^^ THOG -->\n' + processing_html + '        <section class="chart-group" id="depth_chart_group" data-chart-group="depth">\n',
)

write_new("sheet/premat_processing.py", PREMAT_PROCESSING_PY)
write_new("sheet/local_dashboard_assets/dashboard_processing.js", PROCESSING_JS)
write_new("sheet/local_dashboard_assets/dashboard_processing.css", PROCESSING_CSS)
write_new("tests/test_premat_processing.py", PROCESSING_TEST)

print("PREMAT processing tranche applied")
# ^^^ THOG
