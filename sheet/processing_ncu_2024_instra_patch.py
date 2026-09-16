# vvv THOG
"""NCU 2024.3 compatibility and visible INSTRA post-run processing progress.

This overlay keeps the established Processing implementation authoritative while:
- accepting both legacy long-form and modern wide-form ``ncu --csv --page raw`` exports;
- joining semantic NVTX labels without relying on one particular NCU rename column;
- preserving byte/block units emitted by modern NCU; and
- making the post-profile normalization/copy phase visible on the training CLI.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence

from . import premat_processing as _processing
from . import processing_ncu_compatibility as _compat


_original_raw_ncu_reader = _compat._raw_ncu_reader
_original_byte_value = _compat._byte_value

_WIDE_METRICS = frozenset({
    "gpu__time_duration.sum",
    "launch__registers_per_thread_allocated",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_allocated",
    "launch__shared_mem_per_block_static",
    "launch__shared_mem_per_block_dynamic",
    "launch__stream_id",
})


def _csv_fields(line: str) -> list[str]:
    try:
        return next(csv.reader([line]))
    except (csv.Error, StopIteration):
        return []


def _wide_units_row(headers: Sequence[str], values: Sequence[str]) -> bool:
    by_name = {name: values[index] if index < len(values) else "" for index, name in enumerate(headers)}
    identity_names = ("ID", "Process ID", "Kernel Name", "Context", "Stream", "Block Size", "CC")
    identities_empty = all(not str(by_name.get(name, "")).strip() for name in identity_names)
    metric_units_present = any(
        str(by_name.get(name, "")).strip()
        for name in _WIDE_METRICS
        if name in by_name
    )
    return identities_empty and metric_units_present


def _raw_ncu_reader_2024(path: Path) -> list[Dict[str, str]]:
    """Read both legacy long-form and NCU 2024.3 wide-form raw CSV."""

    lines = Path(path).read_text(errors="replace").splitlines()
    header_index: Optional[int] = None
    wide = False
    for index, line in enumerate(lines):
        fields = _csv_fields(line)
        if "Metric Name" in fields and "Metric Value" in fields:
            header_index = index
            break
        if (
            "ID" in fields
            and "Kernel Name" in fields
            and "gpu__time_duration.sum" in fields
            and any(name.startswith("launch__") for name in fields)
        ):
            header_index = index
            wide = True
            break

    if header_index is None:
        raise ValueError(
            f"{path} does not look like `ncu --csv --page raw` output "
            "(neither long-form Metric Name/Metric Value nor wide-form metric columns were found)"
        )

    if not wide:
        return _original_raw_ncu_reader(path)

    table = list(csv.reader(io.StringIO("\n".join(lines[header_index:]))))
    if not table:
        return []
    headers = [str(value) for value in table[0]]
    width = len(headers)
    rows = [list(row[:width]) + [""] * max(0, width - len(row)) for row in table[1:]]

    units = [""] * width
    if rows and _wide_units_row(headers, rows[0]):
        units = rows.pop(0)

    metric_indices = [
        index for index, name in enumerate(headers)
        if name in _WIDE_METRICS
    ]
    metric_like_index_set = {
        index for index, name in enumerate(headers)
        if "__" in name
    }
    exploded: list[Dict[str, str]] = []
    for values in rows:
        if not any(str(value).strip() for value in values):
            continue
        base = {
            name: str(values[index])
            for index, name in enumerate(headers)
            if index not in metric_like_index_set
        }
        for index in metric_indices:
            value = str(values[index]).strip()
            if not value:
                continue
            row = dict(base)
            row["Metric Name"] = headers[index]
            row["Metric Unit"] = str(units[index]).strip()
            row["Metric Value"] = str(values[index])
            exploded.append(row)
    return exploded


def _byte_value_2024(value: Any, unit: Any) -> int:
    normalized = str(unit or "").strip()
    lower = normalized.lower()
    for suffix in ("/block", "/thread"):
        if lower.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return _original_byte_value(value, normalized)


def _semantic_launch_map_2024(path: Path) -> Dict[tuple[str, str, str, str], Dict[str, Any]]:
    rows = _compat._raw_ncu_reader(path)

    def first(row: Mapping[str, Any], *names: str) -> str:
        for name in names:
            value = row.get(name)
            if value not in (None, ""):
                return str(value)
        return ""

    mapping: Dict[tuple[str, str, str, str], Dict[str, Any]] = {}
    for row in rows:
        semantic = None
        preferred = (
            first(row, "Kernel Name", "Kernel", "Name"),
            *(
                str(value)
                for value in row.values()
                if "THOG2_PROCESSING|" in str(value or "")
            ),
        )
        for candidate in preferred:
            semantic = _compat.parse_processing_nvtx_label(candidate)
            if semantic is not None:
                break
        if semantic is None:
            continue
        key = (
            first(row, "ID", "Kernel ID", "Launch ID"),
            first(row, "Process ID", "PID"),
            first(row, "Context", "Context ID"),
            first(row, "Stream", "Stream ID"),
        )
        existing = mapping.get(key)
        if existing is not None and existing != semantic:
            raise ValueError(f"conflicting semantic NCU labels for launch {key}")
        mapping[key] = semantic
    return mapping


def _postrun_progress(step: int, total: int, message: str) -> None:
    percent = int(round(100.0 * float(step) / float(total)))
    print(
        f"THOG2 INSTRA post-run [{step}/{total} · {percent:3d}%] {message}",
        flush=True,
    )


def maybe_reexec_under_nsys(
    arguments: Sequence[str],
    *,
    entrypoint: Path,
) -> Optional[int]:
    """Established Processing launcher with explicit post-run progress stages."""

    if os.environ.get(_processing._PROCESSING_CHILD_ENV) == "1":
        return None
    requested, frequency, capture_update = _processing.processing_requested_from_argv(arguments)
    profiler = _processing.processing_profiler_from_argv(arguments)
    rewritten_arguments = _processing.rewrite_processing_cli_for_core(arguments)
    if not requested or _processing.processing_invocation_is_non_training(arguments):
        sys.argv[:] = [sys.argv[0], *rewritten_arguments]
        return None

    temporary_root = Path(tempfile.mkdtemp(prefix=f"thog2-premat-processing-{profiler}-"))
    handoff_path = temporary_root / "handoff.json"
    capture_metadata_path = temporary_root / "capture.json"
    environment = dict(os.environ)
    environment[_processing._PROCESSING_CHILD_ENV] = "1"
    environment[_processing._PROCESSING_HANDOFF_ENV] = str(handoff_path)
    environment[_processing._PROCESSING_CAPTURE_METADATA_ENV] = str(capture_metadata_path)

    if profiler == "nsys":
        nsys = _processing._find_nsys()
        if nsys is None:
            raise RuntimeError(
                "--premat_processing_logging enabled with --premat_processing_profiler nsys "
                "requires NVIDIA Nsight Systems (nsys) on PATH"
            )
        report_base = temporary_root / "processing_trace"
        environment["NSYS_NVTX_PROFILER_REGISTER_ONLY"] = "0"
        command = _processing._nsys_profile_command(
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

        total = 7
        _postrun_progress(1, total, "profiled training child complete; resolving INSTRA handoff")
        if not handoff_path.exists():
            raise RuntimeError(
                "PREMAT processing capture completed but no INSTRA run handoff was written; "
                "the profiled child did not reach telemetry attachment or did not inherit the THOG processing environment"
            )
        handoff = json.loads(handoff_path.read_text())
        run_directory = Path(handoff["run_directory"])
        processing_directory = run_directory / "processing"
        processing_directory.mkdir(parents=True, exist_ok=True)
        _postrun_progress(2, total, f"INSTRA handoff resolved; processing directory {processing_directory}")

        report_path = report_base.with_suffix(".nsys-rep")
        if not report_path.exists():
            reports = list(temporary_root.glob("*.nsys-rep"))
            if len(reports) != 1:
                raise RuntimeError("Nsight Systems did not produce exactly one .nsys-rep trace")
            report_path = reports[0]
        _postrun_progress(3, total, "Nsight Systems report located; exporting SQLite")

        sqlite_path = temporary_root / "processing_trace.sqlite"
        export = subprocess.run([
            nsys, "export", "--type=sqlite", "--force-overwrite=true",
            f"--output={sqlite_path}", str(report_path),
        ])
        if export.returncode != 0 or not sqlite_path.exists():
            raise RuntimeError("Nsight Systems SQLite export failed for PREMAT processing capture")
        _postrun_progress(4, total, "SQLite export complete; normalizing Processing telemetry")

        capture_metadata = (
            json.loads(capture_metadata_path.read_text())
            if capture_metadata_path.exists()
            else {}
        )
        processing_data = _processing.normalize_nsys_sqlite(
            sqlite_path,
            processing_directory,
            capture_frequency_hz=frequency,
            handoff=handoff,
            capture_metadata=capture_metadata,
        )
        _postrun_progress(5, total, "GPU samples, kernel intervals and resource attribution normalized")

        processing_files = processing_data["metadata"]["files"]
        shutil.copy2(report_path, processing_directory / processing_files["raw_trace"])
        _postrun_progress(6, total, "raw Nsight report copied into the run Processing artifacts")

        bundle_path = processing_directory / processing_files["bundle"]
        _postrun_progress(7, total, f"INSTRA Processing artifacts ready: {bundle_path}")
        print(f"THOG2 PREMAT processing data: {bundle_path}", flush=True)
        return int(completed.returncode)

    ncu = _processing._find_ncu()
    if ncu is None:
        raise RuntimeError(
            "--premat_processing_logging enabled with --premat_processing_profiler ncu "
            "requires NVIDIA Nsight Compute CLI (ncu) on PATH"
        )
    layer, family = _processing._processing_ncu_target_from_argv(arguments)
    report_base = temporary_root / "processing_ncu_trace"
    command = _processing._ncu_profile_command(
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

    total = 7
    _postrun_progress(1, total, "profiled training child complete; resolving INSTRA handoff")
    if not handoff_path.exists():
        raise RuntimeError(
            "NCU processing capture completed but no INSTRA run handoff was written; "
            "the profiled child did not reach telemetry attachment"
        )
    handoff = json.loads(handoff_path.read_text())
    run_directory = Path(handoff["run_directory"])
    processing_directory = run_directory / "processing"
    processing_directory.mkdir(parents=True, exist_ok=True)
    _postrun_progress(2, total, f"INSTRA handoff resolved; processing directory {processing_directory}")

    reports = [
        path
        for pattern in ("*.ncu-rep", "*.ncu-repz")
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
    _postrun_progress(3, total, "Nsight Compute report located; exporting raw kernel-resource CSV")

    raw_csv = _processing._export_ncu_raw_csv(
        ncu,
        report_path,
        temporary_root / "processing_ncu_raw.csv",
        nvtx_rename=False,
    )
    _postrun_progress(4, total, "raw NCU CSV export complete; exporting semantic NVTX view")

    semantic_csv = _processing._export_ncu_raw_csv(
        ncu,
        report_path,
        temporary_root / "processing_ncu_semantic.csv",
        nvtx_rename=True,
    )
    _postrun_progress(5, total, "semantic NCU CSV export complete; normalizing representative kernels")

    resource_rows = _compat.normalize_semantic_ncu_exports(raw_csv, semantic_csv)
    representative_rows = _compat.select_representative_kernel_resources(resource_rows)
    paths = _compat.write_outputs(representative_rows, processing_directory)
    _postrun_progress(6, total, "NCU resources normalized and MAIN/PREMAT compatibility data written")

    report_destination = processing_directory / f"processing_ncu_trace{report_path.suffix}"
    shutil.copy2(report_path, report_destination)
    shutil.copy2(raw_csv, processing_directory / "processing_ncu_raw.csv")
    shutil.copy2(semantic_csv, processing_directory / "processing_ncu_semantic.csv")
    _postrun_progress(
        7,
        total,
        f"INSTRA NCU artifacts ready: {paths['csv']} ({family} layer {layer})",
    )
    print(
        "THOG2 PREMAT NCU compatibility data: "
        f"{paths['csv']} ({family} layer {layer}; dominant MAIN/PREMAT kernels)",
        flush=True,
    )
    return int(completed.returncode)


_compat._raw_ncu_reader = _raw_ncu_reader_2024
_compat._byte_value = _byte_value_2024
_compat._semantic_launch_map = _semantic_launch_map_2024
_processing.maybe_reexec_under_nsys = maybe_reexec_under_nsys
# ^^^ THOG
