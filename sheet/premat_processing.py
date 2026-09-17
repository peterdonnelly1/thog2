# vvv THOG
"""Opt-in PREMAT Processing capture and Nsight Systems normalization."""

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
import time
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence
import zipfile

from sheet.processing_resource_attribution import (
    PROCESSING_METRIC_FIELDS,
    PROCESSING_METRIC_SPECS,
    PROCESSING_STREAM_RESOURCE_FIELDS,
    build_stream_resource_rows,
    classify_processing_operations,
    duration_weighted_metric_statistics,
    infer_kernel_owners,
    main_idle_intervals,
    merge_intervals,
    metric_catalog,
    match_processing_metric_name,
    metric_display_value,
)


PROCESSING_SWITCHES = ("enabled", "disabled")
PROCESSING_PROFILERS = ("nsys", "ncu")
PROCESSING_DEFAULT_PROFILER = "nsys"
PROCESSING_DEFAULT_CAPTURE_FREQUENCY_HZ = 10_000
PROCESSING_DEFAULT_CAPTURE_UPDATE = 1
PROCESSING_MIN_CAPTURE_FREQUENCY_HZ = 10
PROCESSING_MAX_CAPTURE_FREQUENCY_HZ = 200_000
PROCESSING_SCHEMA_VERSION = 3
PROCESSING_CAPTURE_RANGE = "THOG2_PREMAT_PROCESSING_CAPTURE"
PROCESSING_OPERATION_PREFIX = "THOG2_PROCESSING"
_PROCESSING_CHILD_ENV = "THOG2_PREMAT_PROCESSING_UNDER_NSYS"
_PROCESSING_HANDOFF_ENV = "THOG2_PREMAT_PROCESSING_HANDOFF"
_PROCESSING_CAPTURE_METADATA_ENV = "THOG2_PREMAT_PROCESSING_CAPTURE_METADATA"

_capture_active = False
_capture_done = False
_capture_stack_depth = 0
_capture_start_ns: Optional[int] = None


def validate_processing_configuration(
    logging: str,
    capture_frequency_hz: int,
    device: str,
    capture_update: int = PROCESSING_DEFAULT_CAPTURE_UPDATE,
    max_updates: Optional[int] = None,
) -> None:
    if logging not in PROCESSING_SWITCHES:
        raise ValueError(f"premat_processing_logging must be enabled or disabled; got {logging!r}")
    if isinstance(capture_frequency_hz, bool) or not isinstance(capture_frequency_hz, int):
        raise ValueError("premat_processing_logging_capture_frequency_hz must be an integer")
    if not PROCESSING_MIN_CAPTURE_FREQUENCY_HZ <= capture_frequency_hz <= PROCESSING_MAX_CAPTURE_FREQUENCY_HZ:
        raise ValueError(
            "premat_processing_logging_capture_frequency_hz must lie in "
            f"[{PROCESSING_MIN_CAPTURE_FREQUENCY_HZ}, {PROCESSING_MAX_CAPTURE_FREQUENCY_HZ}]"
        )
    if isinstance(capture_update, bool) or not isinstance(capture_update, int) or capture_update < 1:
        raise ValueError("premat_processing_logging_capture_update must be a positive integer")
    if logging == "enabled" and max_updates is not None and capture_update > int(max_updates):
        raise ValueError(
            "premat_processing_logging_capture_update must not exceed max_updates; "
            f"got capture_update={capture_update}, max_updates={max_updates}"
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


def processing_profiler_from_argv(arguments: Sequence[str]) -> str:
    profiler = str(_argv_value(arguments, "--premat_processing_profiler", PROCESSING_DEFAULT_PROFILER)).strip().lower()
    if profiler not in PROCESSING_PROFILERS:
        raise ValueError(
            "--premat_processing_profiler must be one of "
            + ", ".join(PROCESSING_PROFILERS)
            + f"; got {profiler!r}"
        )
    return profiler


def processing_requested_from_argv(arguments: Sequence[str]) -> tuple[bool, int, int]:
    logging = str(_argv_value(arguments, "--premat_processing_logging", "disabled"))
    raw_frequency = _argv_value(
        arguments,
        "--premat_processing_logging_capture_frequency_hz",
        str(PROCESSING_DEFAULT_CAPTURE_FREQUENCY_HZ),
    )
    raw_capture_update = _argv_value(
        arguments,
        "--premat_processing_logging_capture_update",
        str(PROCESSING_DEFAULT_CAPTURE_UPDATE),
    )
    try:
        frequency = int(str(raw_frequency))
    except ValueError as error:
        raise ValueError("--premat_processing_logging_capture_frequency_hz must be an integer") from error
    try:
        capture_update = int(str(raw_capture_update))
    except ValueError as error:
        raise ValueError("--premat_processing_logging_capture_update must be an integer") from error
    validate_processing_configuration(logging, frequency, "cuda", capture_update)
    return logging == "enabled", frequency, capture_update


def rewrite_processing_cli_for_core(arguments: Sequence[str]) -> list[str]:
    rewritten: list[str] = []
    index = 0
    replacements = {
        "--premat_processing_logging": "--processing_logging_internal",
        "--premat_processing_logging_capture_frequency_hz": "--processing_logging_capture_frequency_hz_internal",
        "--premat_processing_logging_capture_update": "--processing_logging_capture_update_internal",
    }
    outer_only_value_options = {"--premat_processing_profiler", "--ncu_probe_layer", "--ncu-probe-layer"}
    while index < len(arguments):
        argument = str(arguments[index])
        if argument in outer_only_value_options:
            if index + 1 >= len(arguments):
                raise ValueError(f"{argument} requires a value")
            index += 2
            continue
        if any(argument.startswith(option + "=") for option in outer_only_value_options):
            index += 1
            continue
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


_PROCESSING_NON_TRAINING_FLAGS = frozenset({
    "-h", "--help", "--dry-run", "--explain-geometry", "--print-artifact-name",
    "--print-geometry-registry", "--print-resolved-json",
})


def processing_invocation_is_non_training(arguments: Sequence[str]) -> bool:
    return any(str(argument).split("=", 1)[0] in _PROCESSING_NON_TRAINING_FLAGS for argument in arguments)


def _find_nsys() -> Optional[str]:
    resolved = shutil.which("nsys")
    if resolved:
        return resolved
    candidates = sorted(Path("/opt/nvidia/nsight-systems").glob("*/bin/nsys"), reverse=True)
    candidates.extend(Path("/usr/local/cuda/bin").glob("nsys"))
    return str(candidates[0]) if candidates else None


def _find_ncu() -> Optional[str]:
    resolved = shutil.which("ncu")
    if resolved:
        return resolved
    candidates = sorted(Path("/opt/nvidia/nsight-compute").glob("*/ncu"), reverse=True)
    candidates.extend(sorted(Path("/opt/nvidia/nsight-compute").glob("*/target/linux-desktop-glibc_*/ncu"), reverse=True))
    candidates.extend(Path("/usr/local/cuda/bin").glob("ncu"))
    return str(candidates[0]) if candidates else None


_PROCESSING_TARGET_FAMILIES = {1: "QKV", 2: "O", 3: "UP", 4: "DOWN"}


def _processing_ncu_target_from_argv(arguments: Sequence[str]) -> tuple[int, str]:
    raw_probe_layer = _argv_value(arguments, "--ncu-probe-layer")
    if raw_probe_layer is None:
        raw_probe_layer = _argv_value(arguments, "--ncu_probe_layer")
    raw_matrix = _argv_value(arguments, "--premat_target_matrix")
    if raw_probe_layer is None or raw_matrix is None:
        raise ValueError(
            "--premat_processing_profiler ncu requires explicit "
            "--ncu-probe-layer and --premat_target_matrix"
        )
    try:
        probe_layer = int(str(raw_probe_layer))
        matrix = int(str(raw_matrix))
    except ValueError as error:
        raise ValueError("NCU probe layer/matrix must be integers") from error
    if probe_layer < 0:
        raise ValueError("--ncu-probe-layer must be non-negative")
    if matrix not in _PROCESSING_TARGET_FAMILIES:
        raise ValueError("--premat_target_matrix must be 1(QKV), 2(O), 3(UP), or 4(DOWN) for NCU capture")
    return probe_layer, _PROCESSING_TARGET_FAMILIES[matrix]


def _ncu_profile_command(
    ncu: str,
    *,
    report_base: Path,
    entrypoint: Path,
    arguments: Sequence[str],
    layer: int,
    family: str,
) -> list[str]:
    main_label = _operation_label("MAIN", "consume", family, layer)
    premat_label = _operation_label("PREMAT", "materialize", family, layer)
    return [
        ncu,
        "--target-processes", "all",
        "--nvtx",
        "--nvtx-include", main_label + "/",
        "--nvtx-include", premat_label + "/",
        "--section", "LaunchStats",
        "--section", "Occupancy",
        "--metrics", "gpu__time_duration.sum",
        "--replay-mode", "kernel",
        "--force-overwrite",
        "--export", str(report_base),
        sys.executable,
        str(Path(entrypoint).resolve()),
        *arguments,
    ]


def _export_ncu_raw_csv(
    ncu: str,
    report_path: Path,
    destination: Path,
    *,
    nvtx_rename: bool,
) -> Path:
    command = [
        ncu,
        "--import", str(report_path),
        "--csv",
        "--page", "raw",
        "--print-units", "base",
    ]
    if nvtx_rename:
        command.extend(("--print-nvtx-rename", "kernel"))
    completed = subprocess.run(command, stdout=subprocess.PIPE, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Nsight Compute CSV export failed for {report_path}")
    destination.write_text(completed.stdout)
    return destination


def _nsys_profile_command(
    nsys: str,
    *,
    report_base: Path,
    frequency: int,
    entrypoint: Path,
    arguments: Sequence[str],
) -> list[str]:
    return [
        nsys, "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
        "--show-output=true", "--inherit-environment=true", "--wait=primary",
        "--capture-range=nvtx", "--capture-range-end=stop",
        f"--nvtx-capture={PROCESSING_CAPTURE_RANGE}",
        "--gpu-metrics-devices=cuda-visible", f"--gpu-metrics-frequency={frequency}",
        "--force-overwrite=true", f"--output={report_base}", sys.executable,
        str(Path(entrypoint).resolve()), *arguments,
    ]


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
            key: config.get(key)
            for key in (
                "premat", "premat_target_layer", "premat_target_matrix", "premat_attention_mode",
                "premat_processing_logging", "premat_processing_logging_capture_frequency_hz",
                "premat_processing_logging_capture_update", "n_layer", "n_embd", "n_head",
                "batch_size", "block_size", "gradient_accumulation_steps", "dtype", "device",
            )
        },
    }
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True))


def should_capture_processing_forward(
    *,
    enabled: bool,
    completed_updates: int,
    max_updates: int,
    log_interval: int,
    capture_update: int = PROCESSING_DEFAULT_CAPTURE_UPDATE,
    micro_step: int,
) -> bool:
    global _capture_done
    del log_interval
    if not enabled or _capture_done or int(micro_step) != 0:
        return False
    if int(capture_update) > int(max_updates):
        return False
    return int(completed_updates) + 1 == int(capture_update)


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
    if pushed:
        _nvtx_pop()
        _capture_stack_depth = max(0, _capture_stack_depth - 1)


def processing_capture_elapsed_ms() -> Optional[float]:
    """Return host time relative to the active NSYS capture range."""
    if not _capture_active or _capture_start_ns is None:
        return None
    return max(0.0, (time.perf_counter_ns() - _capture_start_ns) / 1_000_000.0)


@contextmanager
def processing_operation_range(
    owner: str,
    operation: str,
    *,
    family: Optional[str] = None,
    layer_index: Optional[int] = None,
) -> Iterator[None]:
    pushed = processing_operation_push(owner, operation, family=family, layer_index=layer_index)
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
    capture_update: int = PROCESSING_DEFAULT_CAPTURE_UPDATE,
    micro_step: int,
    device: Any,
) -> Iterator[None]:
    global _capture_active, _capture_done, _capture_stack_depth, _capture_start_ns
    selected = should_capture_processing_forward(
        enabled=enabled,
        completed_updates=completed_updates,
        max_updates=max_updates,
        log_interval=log_interval,
        capture_update=capture_update,
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
    metadata = None
    if metadata_path:
        metadata = {
            "optimizer_update": int(completed_updates) + 1,
            "micro_step": int(micro_step) + 1,
            "log_interval": int(log_interval),
            "requested_capture_update": int(capture_update),
            "max_updates": int(max_updates),
        }
    _capture_stack_depth = 0
    _nvtx_push(PROCESSING_CAPTURE_RANGE)
    # The NSYS capture begins at the NVTX push, so establish the host timing
    # origin immediately afterwards rather than including profiler setup time.
    _capture_start_ns = time.perf_counter_ns()
    if metadata is not None:
        metadata["host_start_ns"] = int(_capture_start_ns)
    _capture_active = True
    try:
        yield
    finally:
        torch.cuda.synchronize(device)
        while _capture_stack_depth > 0:
            _nvtx_pop()
            _capture_stack_depth -= 1
        _nvtx_pop()
        _capture_active = False
        _capture_start_ns = None
        _capture_done = True
        if metadata is not None:
            destination = Path(metadata_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(metadata, indent=2, sort_keys=True))


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')}


def _first_column(columns: set[str], *names: str) -> Optional[str]:
    return next((name for name in names if name in columns), None)


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
        for row in connection.execute(f'SELECT "{id_column}", "{value_column}" FROM "StringIds"')
    }


def _resolved_string(value: Any, strings: Mapping[int, str]) -> str:
    if value is None:
        return ""
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
    selected = [start_column, end_column, thread_column or start_column, text_column or text_id_column]
    events = []
    query = "SELECT " + ", ".join(f'"{column}"' for column in selected) + ' FROM "NVTX_EVENTS"'
    for start, end, thread, raw_text in connection.execute(query):
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
        for row in connection.execute(query) if None not in row
    ]


def _kernel_rows(connection: sqlite3.Connection, tables: set[str], strings: Mapping[int, str]) -> list[Dict[str, Any]]:
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
    context_column = _first_column(columns, "contextId", "context", "context_id")
    name_column = _first_column(columns, "shortName", "demangledName", "name")
    if None in (start_column, end_column, stream_column, correlation_column):
        raise RuntimeError(f"unsupported CUDA kernel schema: {sorted(columns)}")
    selected = [start_column, end_column, stream_column, correlation_column]
    if context_column is not None:
        selected.append(context_column)
    if name_column is not None:
        selected.append(name_column)
    query = "SELECT " + ", ".join(f'"{column}"' for column in selected) + f' FROM "{table}"'
    kernels = []
    for row in connection.execute(query):
        start, end, stream, correlation = row[:4]
        if None in (start, end, stream, correlation):
            continue
        offset = 4
        context_id: Any = ""
        if context_column is not None:
            context_id = "" if row[offset] is None else int(row[offset])
            offset += 1
        kernel_name = _resolved_string(row[offset], strings) if name_column is not None else ""
        kernels.append({
            "start": int(start), "end": int(end), "stream": int(stream),
            "correlation": int(correlation), "context_id": context_id, "kernel_name": kernel_name,
        })
    return kernels


def _metric_rows(
    connection: sqlite3.Connection,
    tables: set[str],
    capture_start: int,
    capture_end: int,
) -> tuple[list[Dict[str, Any]], Dict[str, str], list[str]]:
    if not {"GPU_METRICS", "TARGET_INFO_GPU_METRICS"}.issubset(tables):
        return [], {}, []
    info_columns = _columns(connection, "TARGET_INFO_GPU_METRICS")
    metric_columns = _columns(connection, "GPU_METRICS")
    metric_id = _first_column(info_columns, "metricId", "id")
    metric_name = _first_column(info_columns, "metricName", "name")
    gpu_metric_id = _first_column(metric_columns, "metricId", "id")
    timestamp = _first_column(metric_columns, "timestamp", "start")
    value = _first_column(metric_columns, "value")
    if None in (metric_id, metric_name, gpu_metric_id, timestamp, value):
        return [], {}, []
    names = {
        int(row[0]): str(row[1])
        for row in connection.execute(f'SELECT "{metric_id}", "{metric_name}" FROM "TARGET_INFO_GPU_METRICS"')
    }
    selected: Dict[int, str] = {}
    mapping: Dict[str, str] = {}
    for identifier, name in names.items():
        key = match_processing_metric_name(name)
        if key is None or key in mapping:
            continue
        selected[identifier] = key
        mapping[key] = name
    available_metric_names = sorted(set(names.values()))
    if not selected:
        return [], mapping, available_metric_names
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
            converted = metric_display_value(key, raw_value)
            if converted is not None:
                row[key] = converted
        except (TypeError, ValueError):
            continue
    return list(samples_by_time.values()), mapping, available_metric_names


def _merged_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    return merge_intervals(intervals)


def _interval_duration(intervals: Iterable[tuple[float, float]]) -> float:
    return sum(right - left for left, right in _merged_intervals(intervals))


def _union_overlap(intervals: Iterable[tuple[float, float]], start: float, end: float) -> float:
    return _interval_duration(
        (max(start, left), min(end, right))
        for left, right in intervals
        if right > start and left < end
    )


def _kernel_union_overlap(
    main_intervals: Iterable[tuple[float, float]],
    premat_intervals: Iterable[tuple[float, float]],
) -> float:
    premat = list(premat_intervals)
    return sum(_union_overlap(premat, left, right) for left, right in _merged_intervals(main_intervals))


_PROCESSING_MATRIX_FAMILIES = ("QKV", "O", "UP", "DOWN")


def _processing_matrix_summary(interval_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Optional[Dict[str, float | int]]]:
    main_intervals = [
        (float(row["start_us"]), float(row["end_us"])) for row in interval_rows if row.get("owner") == "MAIN"
    ]
    main_consume_intervals = [
        (float(row["start_us"]), float(row["end_us"]))
        for row in interval_rows if row.get("owner") == "MAIN" and row.get("operation") == "consume"
    ]
    main_busy_us = _interval_duration(main_intervals)
    result: Dict[str, Optional[Dict[str, float | int]]] = {family: None for family in _PROCESSING_MATRIX_FAMILIES}
    for family in _PROCESSING_MATRIX_FAMILIES:
        premat_rows = [
            row for row in interval_rows
            if row.get("owner") == "PREMAT" and row.get("operation") == "materialize" and row.get("family") == family
        ]
        if not premat_rows:
            continue
        premat_intervals = [(float(row["start_us"]), float(row["end_us"])) for row in premat_rows]
        premat_busy_us = _interval_duration(premat_intervals)
        by_op: Dict[int, list[tuple[float, float]]] = {}
        for row in premat_rows:
            raw_op_id = row.get("op_id")
            if raw_op_id in (None, ""):
                continue
            by_op.setdefault(int(raw_op_id), []).append((float(row["start_us"]), float(row["end_us"])))
        op_durations_us = [_interval_duration(rows) for rows in by_op.values()]
        family_consumes = [
            row for row in interval_rows
            if row.get("owner") == "MAIN" and row.get("operation") == "consume" and row.get("family") == family
        ]
        consume_op_ids = {int(row["op_id"]) for row in family_consumes if row.get("op_id") not in (None, "")}
        ready_leads_us: list[float] = []
        premat_layers = {int(row["layer"]) for row in premat_rows if row.get("layer") not in (None, "")}
        for layer in sorted(premat_layers):
            premat_layer = [row for row in premat_rows if row.get("layer") not in (None, "") and int(row["layer"]) == layer]
            consume_layer = [row for row in family_consumes if row.get("layer") not in (None, "") and int(row["layer"]) == layer]
            if premat_layer and consume_layer:
                ready_leads_us.append(
                    min(float(row["start_us"]) for row in consume_layer)
                    - max(float(row["end_us"]) for row in premat_layer)
                )
        consume_overlap_us = _kernel_union_overlap(premat_intervals, main_consume_intervals)
        any_main_overlap_us = _kernel_union_overlap(premat_intervals, main_intervals)
        result[family] = {
            "premat_materialisations": len(by_op),
            "main_consume_operations": len(consume_op_ids),
            "premat_gpu_ms_total": premat_busy_us / 1000.0,
            "mean_reconstruction_ms": (sum(op_durations_us) / len(op_durations_us) / 1000.0) if op_durations_us else 0.0,
            "main_consume_overlap_ms": consume_overlap_us / 1000.0,
            "any_main_overlap_ms": any_main_overlap_us / 1000.0,
            "premat_concurrent_with_main_pct": 100.0 * any_main_overlap_us / premat_busy_us if premat_busy_us > 0.0 else 0.0,
            "main_busy_concurrent_with_premat_pct": 100.0 * any_main_overlap_us / main_busy_us if main_busy_us > 0.0 else 0.0,
            "mean_ready_lead_ms": (sum(ready_leads_us) / len(ready_leads_us) / 1000.0) if ready_leads_us else 0.0,
        }
    return result


def _mean_metric_intervals(
    samples: Sequence[Mapping[str, Any]],
    key: str,
    intervals: Iterable[tuple[float, float]],
) -> Optional[float]:
    merged = _merged_intervals(intervals)
    values: list[float] = []
    for row in samples:
        timestamp = float(row["time_us"])
        if not any(left <= timestamp <= right for left, right in merged):
            continue
        raw_value = row.get(key)
        if raw_value in (None, ""):
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return sum(values) / len(values) if values else None


def _operation_resource_statistics(
    stream_rows: Sequence[Mapping[str, Any]],
    interval_rows: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    groups: Dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for index, row in enumerate(interval_rows):
        raw_op_id = row.get("op_id")
        identity = (
            str(row.get("owner", "UNKNOWN")),
            str(row.get("operation", "misc")),
            str(row.get("family", "")),
            row.get("layer", ""),
            raw_op_id if raw_op_id not in (None, "") else f"kernel-{index + 1}",
        )
        groups.setdefault(identity, []).append(row)
    result: list[Dict[str, Any]] = []
    for identity, rows in groups.items():
        owner, operation, family, layer, op_id = identity
        intervals = [(float(row["start_us"]), float(row["end_us"])) for row in rows]
        for metric_key in PROCESSING_METRIC_FIELDS:
            stats = duration_weighted_metric_statistics(stream_rows, metric_key, intervals)
            if not stats["sample_count"]:
                continue
            result.append({
                "owner": owner,
                "operation": operation,
                "family": family,
                "layer": layer,
                "op_id": op_id,
                "start_us": min(left for left, _right in intervals),
                "end_us": max(right for _left, right in intervals),
                "duration_us": _interval_duration(intervals),
                "metric": metric_key,
                **stats,
            })
    return result


def _attribution_resource_statistics(
    stream_rows: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    result: list[Dict[str, Any]] = []
    states = sorted({str(row.get("attribution_state", "")) for row in stream_rows})
    for state in states:
        rows = [row for row in stream_rows if str(row.get("attribution_state", "")) == state]
        intervals = [
            (float(row["sample_start_us"]), float(row["sample_end_us"]))
            for row in rows
        ]
        for metric_key in PROCESSING_METRIC_FIELDS:
            stats = duration_weighted_metric_statistics(rows, metric_key, intervals)
            if stats["sample_count"]:
                result.append({"attribution_state": state, "metric": metric_key, **stats})
    return result


def _metric_audit_rows(
    sample_rows: Sequence[Mapping[str, Any]],
    metric_mapping: Mapping[str, str],
) -> list[Dict[str, Any]]:
    result: list[Dict[str, Any]] = []
    for key, specification in PROCESSING_METRIC_SPECS.items():
        values = []
        for row in sample_rows:
            try:
                value = float(row.get(key))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        mapped = str(metric_mapping.get(key, ""))
        result.append({
            "canonical_name": key,
            "nsight_name": mapped,
            "unit": str(specification["unit"]),
            "required": bool(specification["required"]),
            "available": bool(mapped and values),
            "status": "available" if mapped and values else ("non_finite" if mapped else "not_captured"),
            "matching": "exact",
            "transform": str(specification.get("transform", "identity")),
            "finite_sample_count": len(values),
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
            "accepted_exact_names": ";".join(specification.get("exact_names", ())),
        })
    return result


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
        capture_duration_us = (capture_end - capture_start) / 1000.0
        operations = []
        for event in nvtx:
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
            for runtime in runtimes:
                if (
                    runtime["thread"] == operation["thread"]
                    and runtime["start"] >= operation["start"]
                    and runtime["end"] <= operation["end"]
                ):
                    correlation_to_operation[int(runtime["correlation"])] = operation

        candidates = []
        operation_kernel_intervals: Dict[int, list[tuple[float, float]]] = {}
        for kernel in kernels:
            if kernel["end"] < capture_start or kernel["start"] > capture_end:
                continue
            start_us = (max(kernel["start"], capture_start) - capture_start) / 1000.0
            end_us = (min(kernel["end"], capture_end) - capture_start) / 1000.0
            if end_us <= start_us:
                continue
            operation = correlation_to_operation.get(int(kernel["correlation"]))
            op_id = int(operation["op_id"]) if operation is not None else None
            candidates.append({
                "start_us": start_us,
                "end_us": end_us,
                "duration_us": end_us - start_us,
                "stream": int(kernel["stream"]),
                "context_id": kernel.get("context_id", ""),
                "owner": str(operation["owner"]) if operation is not None else "",
                "_owner_explicit": operation is not None,
                "layer": "" if operation is None or operation["layer"] is None else int(operation["layer"]),
                "family": "" if operation is None else str(operation["family"]),
                "operation": "other" if operation is None else str(operation["operation"]),
                "kernel_name": str(kernel["kernel_name"]),
                "op_id": "" if op_id is None else op_id,
            })
            if op_id is not None:
                operation_kernel_intervals.setdefault(op_id, []).append((start_us, end_us))
        # vvv THOG Processing resource tide export v1
        interval_rows = classify_processing_operations(infer_kernel_owners(candidates))
        # ^^^ THOG

        raw_samples, metric_mapping, available_metric_names = _metric_rows(
            connection, tables, capture_start, capture_end
        )
        sample_rows = []
        for sample in raw_samples:
            row = {"time_us": (int(sample["timestamp"]) - capture_start) / 1000.0}
            for key in PROCESSING_METRIC_FIELDS:
                row[key] = sample.get(key, "")
            sample_rows.append(row)

    if not sample_rows:
        warnings.append("No requested GPU Metrics samples were found in the Nsight export")
    missing_required = [
        key for key, specification in PROCESSING_METRIC_SPECS.items()
        if specification["required"] and key not in metric_mapping
    ]
    if missing_required:
        warnings.append("Required GPU metrics not mapped: " + ", ".join(missing_required))

    unknown_kernels = sum(1 for row in interval_rows if row["owner"] == "UNKNOWN")
    other_kernels = sum(1 for row in interval_rows if row["owner"] == "OTHER")
    if unknown_kernels:
        warnings.append(f"{unknown_kernels} CUDA kernel(s) could not be attributed to a known stream owner")
    if other_kernels:
        warnings.append(f"{other_kernels} CUDA kernel(s) belong to explicitly non-Main/non-PREMAT work")

    stream_resource_rows = build_stream_resource_rows(sample_rows, interval_rows, capture_duration_us)
    sample_rows = [
        {
            "time_us": row.get("time_us", ""),
            "sample_start_us": row.get("sample_start_us", ""),
            "sample_end_us": row.get("sample_end_us", ""),
            **{key: row.get(key, "") for key in PROCESSING_METRIC_FIELDS},
        }
        for row in stream_resource_rows
    ]
    idle_rows = main_idle_intervals(interval_rows, capture_duration_us)
    premat_intervals = [
        (float(row["start_us"]), float(row["end_us"]))
        for row in interval_rows if row["owner"] == "PREMAT" and row["operation"] == "materialize"
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
        merged_main_intervals = _merged_intervals(kernel_intervals)
        start_us = min(left for left, _ in merged_main_intervals)
        end_us = max(right for _, right in merged_main_intervals)
        duration_us = _interval_duration(merged_main_intervals)
        overlap_us = _kernel_union_overlap(merged_main_intervals, premat_intervals)
        summary_rows.append({
            "op_id": int(operation["op_id"]),
            "layer": "" if operation["layer"] is None else int(operation["layer"]),
            "family": str(operation["family"]),
            "start_us": start_us,
            "end_us": end_us,
            "duration_ms": duration_us / 1000.0,
            "premat_overlap_ms": overlap_us / 1000.0,
            "premat_overlap_pct": 100.0 * overlap_us / duration_us if duration_us > 0.0 else 0.0,
            "sm_active_pct_mean": duration_weighted_metric_statistics(sample_rows, "sm_active_pct", merged_main_intervals)["mean"],
            "sm_issue_pct_mean": duration_weighted_metric_statistics(sample_rows, "sm_issue_pct", merged_main_intervals)["mean"],
            "tensor_active_pct_mean": duration_weighted_metric_statistics(sample_rows, "tensor_active_pct", merged_main_intervals)["mean"],
            "active_sm_unused_warp_slots_pct_mean": duration_weighted_metric_statistics(
                sample_rows, "active_sm_unused_warp_slots_pct", merged_main_intervals
            )["mean"],
            "compute_warps_in_flight_pct_mean": duration_weighted_metric_statistics(
                sample_rows, "compute_warps_in_flight_pct", merged_main_intervals
            )["mean"],
            "gpc_clock_mhz_mean": duration_weighted_metric_statistics(
                sample_rows, "gpc_clock_mhz", merged_main_intervals
            )["mean"],
        })

    matrix_summary = _processing_matrix_summary(interval_rows)
    operation_resource_stats = _operation_resource_statistics(stream_resource_rows, interval_rows)
    attribution_resource_stats = _attribution_resource_statistics(stream_resource_rows)
    metric_audit = _metric_audit_rows(sample_rows, metric_mapping)
    sample_fields = ("time_us", "sample_start_us", "sample_end_us", *PROCESSING_METRIC_FIELDS)
    interval_fields = (
        "start_us", "end_us", "duration_us", "stream", "context_id", "owner", "owner_source",
        "layer", "family", "operation", "operation_source", "kernel_name", "op_id",
    )
    summary_fields = (
        "op_id", "layer", "family", "start_us", "end_us", "duration_ms",
        "premat_overlap_ms", "premat_overlap_pct", "sm_active_pct_mean", "sm_issue_pct_mean",
        "tensor_active_pct_mean", "active_sm_unused_warp_slots_pct_mean",
        "compute_warps_in_flight_pct_mean", "gpc_clock_mhz_mean",
    )
    resource_stat_fields = (
        "owner", "operation", "family", "layer", "op_id", "start_us", "end_us",
        "duration_us", "metric", "sample_count", "covered_us", "mean", "min",
        "p50", "p90", "p95", "max",
    )
    attribution_stat_fields = (
        "attribution_state", "metric", "sample_count", "covered_us", "mean", "min",
        "p50", "p90", "p95", "max",
    )
    metric_audit_fields = (
        "canonical_name", "nsight_name", "unit", "required", "available", "status",
        "matching", "transform", "finite_sample_count", "minimum", "maximum",
        "accepted_exact_names",
    )
    run_artifact = str((handoff or {}).get("run_name", "")).strip()
    if "/" in run_artifact or "\\" in run_artifact:
        run_artifact = Path(run_artifact).name
    prefix = f"{run_artifact}_" if run_artifact else ""
    processing_files = {
        "samples": f"{prefix}processing_samples.csv",
        "intervals": f"{prefix}processing_intervals.csv",
        "stream_resources": f"{prefix}processing_stream_resources.csv",
        "summary": f"{prefix}processing_summary.csv",
        "operation_resource_stats": f"{prefix}processing_operation_resource_stats.csv",
        "attribution_resource_stats": f"{prefix}processing_attribution_resource_stats.csv",
        "metric_audit": f"{prefix}processing_metric_audit.csv",
        "metadata": f"{prefix}processing_metadata.json",
        "bundle": f"{prefix}processing_bundle.zip",
        "raw_trace": f"{prefix}processing_trace.nsys-rep",
    }
    _write_csv(output_directory / processing_files["samples"], sample_rows, sample_fields)
    _write_csv(output_directory / processing_files["intervals"], interval_rows, interval_fields)
    _write_csv(
        output_directory / processing_files["stream_resources"],
        stream_resource_rows,
        PROCESSING_STREAM_RESOURCE_FIELDS,
    )
    _write_csv(output_directory / processing_files["summary"], summary_rows, summary_fields)
    _write_csv(
        output_directory / processing_files["operation_resource_stats"],
        operation_resource_stats,
        resource_stat_fields,
    )
    _write_csv(
        output_directory / processing_files["attribution_resource_stats"],
        attribution_resource_stats,
        attribution_stat_fields,
    )
    _write_csv(
        output_directory / processing_files["metric_audit"],
        metric_audit,
        metric_audit_fields,
    )

    attribution_counts: Dict[str, int] = {}
    for row in stream_resource_rows:
        state = str(row["attribution_state"])
        attribution_counts[state] = attribution_counts.get(state, 0) + 1
    metadata = {
        "schema_version": PROCESSING_SCHEMA_VERSION,
        "capture_frequency_hz": int(capture_frequency_hz),
        "capture_duration_ms": (capture_end - capture_start) / 1_000_000.0,
        "metric_mapping": metric_mapping,
        "metric_catalog": metric_catalog(metric_mapping),
        "available_gpu_metric_names": available_metric_names,
        "kernel_owner_counts": {
            owner: sum(1 for row in interval_rows if row["owner"] == owner)
            for owner in ("MAIN", "PREMAT", "OTHER", "UNKNOWN")
        },
        "attribution_counts": attribution_counts,
        "warnings": warnings,
        "capture": dict(capture_metadata or {}),
        "run": dict(handoff or {}),
        "files": dict(processing_files),
    }
    (output_directory / processing_files["metadata"]).write_text(json.dumps(metadata, indent=2, sort_keys=True))
    processing_data = {
        "metadata": metadata,
        "samples": sample_rows,
        "intervals": interval_rows,
        "stream_resources": stream_resource_rows,
        "main_idle_intervals": idle_rows,
        "summary": summary_rows,
        "operation_resource_stats": operation_resource_stats,
        "attribution_resource_stats": attribution_resource_stats,
        "metric_audit": metric_audit,
        "matrix_summary": matrix_summary,
    }
    (output_directory / "processing_data.json").write_text(
        json.dumps(processing_data, separators=(",", ":"), allow_nan=False)
    )
    with zipfile.ZipFile(output_directory / processing_files["bundle"], "w", zipfile.ZIP_DEFLATED) as archive:
        for name in (
            processing_files["samples"], processing_files["intervals"],
            processing_files["stream_resources"], processing_files["summary"],
            processing_files["operation_resource_stats"],
            processing_files["attribution_resource_stats"],
            processing_files["metric_audit"],
            processing_files["metadata"], "processing_data.json",
        ):
            archive.write(output_directory / name, arcname=name)
    return processing_data


def maybe_reexec_under_nsys(arguments: Sequence[str], *, entrypoint: Path) -> Optional[int]:
    if os.environ.get(_PROCESSING_CHILD_ENV) == "1":
        return None
    requested, frequency, capture_update = processing_requested_from_argv(arguments)
    profiler = processing_profiler_from_argv(arguments)
    rewritten_arguments = rewrite_processing_cli_for_core(arguments)
    if not requested or processing_invocation_is_non_training(arguments):
        sys.argv[:] = [sys.argv[0], *rewritten_arguments]
        return None

    temporary_root = Path(tempfile.mkdtemp(prefix=f"thog2-premat-processing-{profiler}-"))
    handoff_path = temporary_root / "handoff.json"
    capture_metadata_path = temporary_root / "capture.json"
    environment = dict(os.environ)
    environment[_PROCESSING_CHILD_ENV] = "1"
    environment[_PROCESSING_HANDOFF_ENV] = str(handoff_path)
    environment[_PROCESSING_CAPTURE_METADATA_ENV] = str(capture_metadata_path)

    if profiler == "nsys":
        nsys = _find_nsys()
        if nsys is None:
            raise RuntimeError(
                "--premat_processing_logging enabled with --premat_processing_profiler nsys "
                "requires NVIDIA Nsight Systems (nsys) on PATH"
            )
        report_base = temporary_root / "processing_trace"
        environment["NSYS_NVTX_PROFILER_REGISTER_ONLY"] = "0"
        command = _nsys_profile_command(
            nsys,
            report_base=report_base,
            frequency=frequency,
            entrypoint=entrypoint,
            arguments=rewritten_arguments,
        )
        print(
            f"THOG2 PREMAT processing capture: Nsight Systems @ {frequency} Hz; "
            f"capturing update {capture_update}, first forward microstep",
            flush=True,
        )
        completed = subprocess.run(command, env=environment)
        if completed.returncode != 0:
            return int(completed.returncode)
        if not handoff_path.exists():
            raise RuntimeError(
                "PREMAT processing capture completed but no INSTRA run handoff was written; "
                "the profiled child did not reach telemetry attachment or did not inherit the THOG processing environment"
            )
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
        export = subprocess.run([
            nsys, "export", "--type=sqlite", "--force-overwrite=true",
            f"--output={sqlite_path}", str(report_path),
        ])
        if export.returncode != 0 or not sqlite_path.exists():
            raise RuntimeError("Nsight Systems SQLite export failed for PREMAT processing capture")
        capture_metadata = json.loads(capture_metadata_path.read_text()) if capture_metadata_path.exists() else {}
        processing_data = normalize_nsys_sqlite(
            sqlite_path,
            processing_directory,
            capture_frequency_hz=frequency,
            handoff=handoff,
            capture_metadata=capture_metadata,
        )
        processing_files = processing_data["metadata"]["files"]
        shutil.copy2(report_path, processing_directory / processing_files["raw_trace"])
        print(f"THOG2 PREMAT processing data: {processing_directory / processing_files['bundle']}", flush=True)
        return int(completed.returncode)

    ncu = _find_ncu()
    if ncu is None:
        raise RuntimeError(
            "--premat_processing_logging enabled with --premat_processing_profiler ncu "
            "requires NVIDIA Nsight Compute CLI (ncu) on PATH"
        )
    layer, family = _processing_ncu_target_from_argv(arguments)
    report_base = temporary_root / "processing_ncu_trace"
    command = _ncu_profile_command(
        ncu,
        report_base=report_base,
        entrypoint=entrypoint,
        arguments=rewritten_arguments,
        layer=layer,
        family=family,
    )
    print(
        "THOG2 PREMAT processing capture: Nsight Compute; "
        f"capturing update {capture_update}, layer {layer} {family}; "
        "MAIN consume + PREMAT materialize",
        flush=True,
    )
    completed = subprocess.run(command, env=environment)
    if completed.returncode != 0:
        return int(completed.returncode)
    if not handoff_path.exists():
        raise RuntimeError(
            "NCU processing capture completed but no INSTRA run handoff was written; "
            "the profiled child did not reach telemetry attachment"
        )
    handoff = json.loads(handoff_path.read_text())
    run_directory = Path(handoff["run_directory"])
    processing_directory = run_directory / "processing"
    processing_directory.mkdir(parents=True, exist_ok=True)

    reports = [
        path for pattern in ("*.ncu-rep", "*.ncu-repz")
        for path in temporary_root.glob(pattern)
    ]
    preferred = [path for path in reports if path.stem.startswith(report_base.name)]
    if len(preferred) == 1:
        report_path = preferred[0]
    elif len(reports) == 1:
        report_path = reports[0]
    else:
        raise RuntimeError(
            f"Nsight Compute did not produce exactly one report; found {len(reports)}"
        )

    raw_csv = _export_ncu_raw_csv(
        ncu, report_path, temporary_root / "processing_ncu_raw.csv", nvtx_rename=False
    )
    semantic_csv = _export_ncu_raw_csv(
        ncu, report_path, temporary_root / "processing_ncu_semantic.csv", nvtx_rename=True
    )
    from sheet.processing_ncu_compatibility import (
        normalize_semantic_ncu_exports,
        select_representative_kernel_resources,
        write_outputs,
    )
    resource_rows = normalize_semantic_ncu_exports(raw_csv, semantic_csv)
    representative_rows = select_representative_kernel_resources(resource_rows)
    paths = write_outputs(representative_rows, processing_directory)
    report_destination = processing_directory / f"processing_ncu_trace{report_path.suffix}"
    shutil.copy2(report_path, report_destination)
    shutil.copy2(raw_csv, processing_directory / "processing_ncu_raw.csv")
    shutil.copy2(semantic_csv, processing_directory / "processing_ncu_semantic.csv")
    print(
        "THOG2 PREMAT NCU compatibility data: "
        f"{paths['csv']} ({family} layer {layer}; dominant MAIN/PREMAT kernels)",
        flush=True,
    )
    return int(completed.returncode)


__all__ = [
    "PROCESSING_DEFAULT_CAPTURE_FREQUENCY_HZ",
    "PROCESSING_DEFAULT_CAPTURE_UPDATE",
    "PROCESSING_DEFAULT_PROFILER",
    "PROCESSING_PROFILERS",
    "_ncu_profile_command",
    "_processing_ncu_target_from_argv",
    "maybe_reexec_under_nsys",
    "normalize_nsys_sqlite",
    "processing_capture_scope",
    "processing_capture_elapsed_ms",
    "processing_operation_pop",
    "processing_operation_push",
    "processing_operation_range",
    "processing_invocation_is_non_training",
    "processing_profiler_from_argv",
    "processing_requested_from_argv",
    "rewrite_processing_cli_for_core",
    "register_processing_handoff",
    "should_capture_processing_forward",
    "validate_processing_configuration",
]
# ^^^ THOG
