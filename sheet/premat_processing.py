# vvv THOG
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


def rewrite_processing_cli_for_core(arguments: Sequence[str]) -> list[str]:
    rewritten: list[str] = []
    index = 0
    replacements = {
        "--premat_processing_logging": "--processing_logging_internal",
        "--premat_processing_logging_capture_frequency_hz": "--processing_logging_capture_frequency_hz_internal",
    }
    while index < len(arguments):
        argument = str(arguments[index])
        matched = False
        for public_name, internal_name in replacements.items():
            if argument == public_name:
                if index + 1 >= len(arguments):
                    raise ValueError(f"{public_name} requires a value")
                rewritten.extend((internal_name, str(arguments[index + 1])))
                index += 2
                matched = True
                break
            prefix = public_name + "="
            if argument.startswith(prefix):
                rewritten.append(internal_name + "=" + argument[len(prefix):])
                index += 1
                matched = True
                break
        if matched:
            continue
        rewritten.append(argument)
        index += 1
    return rewritten


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
    if os.environ.get(_PROCESSING_CHILD_ENV) == "1":
        return None
    requested, frequency = processing_requested_from_argv(arguments)
    rewritten_arguments = rewrite_processing_cli_for_core(arguments)
    if not requested:
        sys.argv[:] = [sys.argv[0], *rewritten_arguments]
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
        *rewritten_arguments,
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
    "rewrite_processing_cli_for_core",
    "register_processing_handoff",
    "should_capture_processing_forward",
    "validate_processing_configuration",
]
# ^^^ THOG
