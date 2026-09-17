# vvv THOG
"""Build theoretical MAIN/PREMAT block co-residency data from Nsight Compute data.

Two inputs are supported:

1. A normalized ``processing_ncu_kernel_resources.csv`` containing role=MAIN /
   role=PREMAT rows.
2. Raw ``ncu --csv --page raw --print-units base`` CSV exports, supplied as repeated
   ``--main FAMILY=path.csv`` / ``--premat FAMILY=path.csv`` arguments.  The
   raw importer deliberately requires each file to represent the requested
   semantic family; it never guesses MAIN/PREMAT ownership from kernel names.

The compatibility result is an aggregate SM-budget test.  It answers whether
blocks *can fit* together from registers/shared-memory/warps/threads/block
slots.  It does not claim that CUDA actually scheduled them concurrently.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


_REQUIRED_RESOURCE_FIELDS = (
    "threads_per_block",
    "registers_per_block",
    "static_shared_mem_bytes",
    "dynamic_shared_mem_bytes",
    "theoretical_blocks_per_sm",
    "sm_register_capacity",
    "sm_shared_mem_bytes",
    "sm_max_warps",
    "sm_max_threads",
    "sm_max_blocks",
)

# Physical limits used only when normalising raw NCU CSV.  8.9 is the Ada
# target used by scruffy.  7.5 is retained for the TITAN RTX host.  Unknown
# architectures must be supplied explicitly instead of being guessed.
_SM_LIMITS: Dict[str, Dict[str, int]] = {
    "8.9": {
        "sm_register_capacity": 65536,
        "sm_shared_mem_bytes": 102400,
        "sm_max_warps": 48,
        "sm_max_threads": 1536,
        "sm_max_blocks": 24,
    },
    "7.5": {
        "sm_register_capacity": 65536,
        "sm_shared_mem_bytes": 65536,
        "sm_max_warps": 32,
        "sm_max_threads": 1024,
        "sm_max_blocks": 16,
    },
}


def _number(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key, "")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid or missing {key!r}: {value!r}") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"invalid or missing {key!r}: {value!r}")
    return number


def _integer(row: Mapping[str, Any], key: str) -> int:
    return int(round(_number(row, key)))


def _warps_per_block(row: Mapping[str, Any]) -> int:
    raw = row.get("warps_per_block", "")
    if raw not in (None, ""):
        return max(1, int(round(_number(row, "warps_per_block"))))
    return max(1, math.ceil(_integer(row, "threads_per_block") / 32))


def _shared_per_block(row: Mapping[str, Any]) -> int:
    return _integer(row, "static_shared_mem_bytes") + _integer(row, "dynamic_shared_mem_bytes")


def _capacity(row: Mapping[str, Any], key: str) -> int:
    value = _integer(row, key)
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _blocks_limited_by_resource(remaining: int, per_block: int) -> int:
    if per_block <= 0:
        return 1 << 30
    return max(0, remaining // per_block)


def _kernel_footprint(row: Mapping[str, Any]) -> Dict[str, int]:
    for field in _REQUIRED_RESOURCE_FIELDS:
        _number(row, field)
    return {
        "threads": _integer(row, "threads_per_block"),
        "warps": _warps_per_block(row),
        "registers": _integer(row, "registers_per_block"),
        "shared": _shared_per_block(row),
        "blocks": 1,
    }


def _sm_capacities(row: Mapping[str, Any]) -> Dict[str, int]:
    return {
        "threads": _capacity(row, "sm_max_threads"),
        "warps": _capacity(row, "sm_max_warps"),
        "registers": _capacity(row, "sm_register_capacity"),
        "shared": _capacity(row, "sm_shared_mem_bytes"),
        "blocks": _capacity(row, "sm_max_blocks"),
    }


def _compatible_capacities(main_row: Mapping[str, Any], premat_row: Mapping[str, Any]) -> Dict[str, int]:
    main_capacity = _sm_capacities(main_row)
    premat_capacity = _sm_capacities(premat_row)
    for key in main_capacity:
        if main_capacity[key] != premat_capacity[key]:
            raise ValueError(
                f"MAIN/PREMAT NCU rows disagree on SM {key} capacity: "
                f"{main_capacity[key]} != {premat_capacity[key]}"
            )
    return main_capacity


def _fits(capacity: Mapping[str, int], footprints: Iterable[Mapping[str, int]]) -> tuple[bool, str]:
    totals = {key: 0 for key in capacity}
    for footprint in footprints:
        for key in totals:
            totals[key] += int(footprint[key])
    for key in ("registers", "shared", "warps", "threads", "blocks"):
        if totals[key] > capacity[key]:
            return False, key
    return True, ""


def _premat_blocks_after_main(
    capacity: Mapping[str, int],
    main: Mapping[str, int],
    premat: Mapping[str, int],
    main_blocks: int,
) -> tuple[int, str]:
    limits: Dict[str, int] = {}
    for key in ("registers", "shared", "warps", "threads", "blocks"):
        remaining = int(capacity[key]) - int(main_blocks) * int(main[key])
        limits[key] = _blocks_limited_by_resource(max(0, remaining), int(premat[key]))
    count = min(limits.values())
    limiter = min(limits, key=lambda key: (limits[key], key))
    return max(0, int(count)), limiter


def compatibility_row(main_row: Mapping[str, Any], premat_row: Mapping[str, Any]) -> Dict[str, Any]:
    main = _kernel_footprint(main_row)
    premat = _kernel_footprint(premat_row)
    capacity = _compatible_capacities(main_row, premat_row)
    pair_can_co_reside, pair_limiter = _fits(capacity, (main, premat))
    main_blocks = max(1, _integer(main_row, "theoretical_blocks_per_sm"))
    premat_with_one_main, one_main_limiter = _premat_blocks_after_main(capacity, main, premat, 1)
    premat_with_full_main, full_main_limiter = _premat_blocks_after_main(
        capacity, main, premat, main_blocks
    )

    if not pair_can_co_reside:
        compatibility_class = "RED"
        limiting_resource = pair_limiter
    elif premat_with_full_main <= 0:
        compatibility_class = "YELLOW"
        limiting_resource = full_main_limiter
    elif premat_with_full_main == 1:
        compatibility_class = "ORANGE"
        limiting_resource = full_main_limiter
    else:
        compatibility_class = "GREEN"
        limiting_resource = full_main_limiter

    main_layer = main_row.get("layer", "")
    premat_layer = premat_row.get("layer", "")
    return {
        "main_operation": str(main_row.get("operation", "")),
        "main_family": str(main_row.get("family", "")),
        "main_layer": "" if main_layer in (None, "") else int(float(main_layer)),
        "main_cuda_kernel_name": str(main_row.get("cuda_kernel_name", main_row.get("kernel_name", ""))),
        "main_threads_per_block": int(main["threads"]),
        "main_warps_per_block": int(main["warps"]),
        "main_registers_per_thread": (
            "" if main_row.get("registers_per_thread", "") in (None, "")
            else float(main_row["registers_per_thread"])
        ),
        "main_registers_per_block": int(main["registers"]),
        "main_shared_mem_bytes": int(main["shared"]),
        "premat_operation": str(premat_row.get("operation", "materialize")),
        "premat_family": str(premat_row.get("family", "")),
        "premat_layer": "" if premat_layer in (None, "") else int(float(premat_layer)),
        "premat_cuda_kernel_name": str(premat_row.get("cuda_kernel_name", premat_row.get("kernel_name", ""))),
        "premat_threads_per_block": int(premat["threads"]),
        "premat_warps_per_block": int(premat["warps"]),
        "premat_registers_per_thread": (
            "" if premat_row.get("registers_per_thread", "") in (None, "")
            else float(premat_row["registers_per_thread"])
        ),
        "premat_registers_per_block": int(premat["registers"]),
        "premat_shared_mem_bytes": int(premat["shared"]),
        "pair_can_co_reside": bool(pair_can_co_reside),
        "main_theoretical_blocks_per_sm": int(main_blocks),
        "premat_blocks_with_one_main_block": int(premat_with_one_main),
        "premat_blocks_with_full_main_residency": int(premat_with_full_main),
        "limiting_resource": str(limiting_resource or one_main_limiter),
        "compatibility_class": compatibility_class,
        "interpretation": "aggregate_sm_budget_only",
    }


def build_compatibility_rows(rows: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    main_rows = [row for row in rows if str(row.get("role", "")).strip().upper() == "MAIN"]
    premat_rows = [row for row in rows if str(row.get("role", "")).strip().upper() == "PREMAT"]
    if not main_rows:
        raise ValueError("NCU resource CSV has no role=MAIN rows")
    if not premat_rows:
        raise ValueError("NCU resource CSV has no role=PREMAT rows")
    return [compatibility_row(main_row, premat_row) for main_row in main_rows for premat_row in premat_rows]


def read_kernel_resources(path: Path) -> list[Dict[str, Any]]:
    with Path(path).open(newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"NCU resource CSV is empty: {path}")
    return rows


def _clean_number(value: Any) -> float:
    text = str(value or "").strip().replace(",", "")
    if not text or text.lower() in {"n/a", "nan", "-"}:
        raise ValueError(f"not a numeric NCU value: {value!r}")
    return float(text)


def _byte_value(value: Any, unit: Any) -> int:
    number = _clean_number(value)
    normalized = str(unit or "").strip().lower().replace("bytes", "byte")
    factors = {
        "": 1, "byte": 1, "b": 1,
        "kbyte": 1000, "kb": 1000,
        "mbyte": 1000 ** 2, "mb": 1000 ** 2,
        "gbyte": 1000 ** 3, "gb": 1000 ** 3,
        "kib": 1024, "mib": 1024 ** 2, "gib": 1024 ** 3,
    }
    return int(round(number * factors.get(normalized, 1)))


def _duration_ns(value: Any, unit: Any) -> int:
    number = _clean_number(value)
    normalized = str(unit or "").strip().lower().replace(" ", "")
    factors = {
        "": 1.0,
        "ns": 1.0, "nsecond": 1.0, "nanosecond": 1.0, "nanoseconds": 1.0,
        "us": 1_000.0, "usecond": 1_000.0, "microsecond": 1_000.0, "microseconds": 1_000.0,
        "ms": 1_000_000.0, "msecond": 1_000_000.0, "millisecond": 1_000_000.0, "milliseconds": 1_000_000.0,
        "s": 1_000_000_000.0, "second": 1_000_000_000.0, "seconds": 1_000_000_000.0,
    }
    if normalized not in factors:
        raise ValueError(f"unsupported NCU duration unit: {unit!r}")
    return int(round(number * factors[normalized]))


def _block_threads(value: Any) -> int:
    # Typical NCU CSV forms include "256,1,1", "(256, 1, 1)" and "256".
    numbers = [int(part) for part in re.findall(r"\d+", str(value or ""))]
    if not numbers:
        raise ValueError(f"cannot parse NCU Block Size: {value!r}")
    product = 1
    for number in numbers[:3]:
        product *= number
    return product


def _canonical_cc(value: Any, explicit: str | None) -> str:
    candidate = str(explicit or value or "").strip().lower().replace("sm_", "")
    if candidate.isdigit() and len(candidate) == 2:
        candidate = f"{candidate[0]}.{candidate[1]}"
    if candidate not in _SM_LIMITS:
        raise ValueError(
            f"unsupported/unknown compute capability {candidate!r}; "
            f"supported automatic limits: {', '.join(sorted(_SM_LIMITS))}"
        )
    return candidate


def _raw_ncu_reader(path: Path) -> list[Dict[str, str]]:
    """Read NCU long- or wide-form CSV into long-form metric rows."""
    lines = Path(path).read_text(errors="replace").splitlines()
    long_header_index = None
    for index, line in enumerate(lines):
        if "Metric Name" in line and "Metric Value" in line:
            long_header_index = index
            break

    if long_header_index is not None:
        reader = csv.DictReader(io.StringIO("\n".join(lines[long_header_index:])))
        return [dict(row) for row in reader if any(str(value or "").strip() for value in row.values())]

    wide_header_index = None
    for index, line in enumerate(lines):
        try:
            columns = next(csv.reader([line]))
        except csv.Error:
            continue
        if "ID" in columns and "Kernel Name" in columns and any(
            name.startswith("launch__") for name in columns
        ):
            wide_header_index = index
            break
    if wide_header_index is None:
        raise ValueError(
            f"{path} does not look like `ncu --csv --page raw` output "
            "(neither long-form Metric Name/Metric Value columns nor a wide-form "
            "LaunchStats header was found)"
        )
    wide_rows = [
        dict(row)
        for row in csv.DictReader(io.StringIO("\n".join(lines[wide_header_index:])))
        if any(str(value or "").strip() for value in row.values())
    ]
    if not wide_rows:
        return []

    identity_columns = {
        "ID", "Process ID", "Process Name", "Host Name", "Kernel Name",
        "Context", "Stream", "Block Size", "Grid Size", "Device", "CC",
    }
    unit_row: Dict[str, str] = {}
    first_row = wide_rows[0]
    if not str(first_row.get("ID", "")).strip() and not str(
        first_row.get("Kernel Name", "")
    ).strip():
        unit_row = wide_rows.pop(0)

    result: list[Dict[str, str]] = []
    for row in wide_rows:
        if not str(row.get("ID", "")).strip() or not str(row.get("Kernel Name", "")).strip():
            continue
        identity = {name: str(row.get(name, "") or "") for name in identity_columns}
        for metric_name, metric_value in row.items():
            if metric_name in identity_columns or not str(metric_value or "").strip():
                continue
            result.append({
                **identity,
                "Metric Name": str(metric_name),
                "Metric Value": str(metric_value),
                "Metric Unit": str(unit_row.get(metric_name, "") or ""),
            })
    return result


def normalize_raw_ncu_csv(
    path: Path,
    *,
    role: str,
    family: str,
    operation: str,
    layer: int | None = None,
    compute_capability: str | None = None,
) -> list[Dict[str, Any]]:
    """Normalize per-launch NCU raw-page rows into co-residency resource rows."""
    source_rows = _raw_ncu_reader(path)
    if not source_rows:
        raise ValueError(f"NCU raw CSV contains no metric rows: {path}")

    def first(row: Mapping[str, Any], *names: str) -> str:
        for name in names:
            value = row.get(name)
            if value not in (None, ""):
                return str(value)
        return ""

    groups: Dict[tuple[str, ...], list[Dict[str, str]]] = {}
    for row in source_rows:
        key = (
            first(row, "ID", "Kernel ID", "Launch ID"),
            first(row, "Kernel Name", "Kernel", "Name"),
            first(row, "Context", "Context ID"),
            first(row, "Stream", "Stream ID"),
        )
        groups.setdefault(key, []).append(row)

    normalized: list[Dict[str, Any]] = []
    for key, rows in groups.items():
        representative = rows[0]
        metrics: Dict[str, tuple[str, str]] = {}
        for row in rows:
            name = first(row, "Metric Name", "Metric").strip()
            if name:
                metrics[name] = (
                    first(row, "Metric Value", "Value"),
                    first(row, "Metric Unit", "Unit"),
                )

        def metric(*names: str) -> tuple[str, str] | None:
            for name in names:
                if name in metrics:
                    return metrics[name]
            return None

        block_text = first(representative, "Block Size", "Block")
        threads = _block_threads(block_text)

        registers_per_thread_metric = metric("launch__registers_per_thread")
        allocated_registers_per_block_metric = metric(
            "launch__registers_per_thread_allocated"
        )
        if registers_per_thread_metric is None:
            raise ValueError(f"{path}: launch is missing registers-per-thread metric; collect LaunchStats")
        registers_per_thread = int(round(_clean_number(registers_per_thread_metric[0])))
        # Despite its metric name and reported unit, NCU's
        # launch__registers_per_thread_allocated value is the allocation for the
        # entire block.  It may exceed registers_per_thread * threads because
        # register allocation is granular.  Treating it as per-thread and
        # multiplying again creates impossible multi-million-register blocks.
        registers_per_block = (
            int(round(_clean_number(allocated_registers_per_block_metric[0])))
            if allocated_registers_per_block_metric is not None
            else registers_per_thread * threads
        )
        if registers_per_block < registers_per_thread * threads:
            raise ValueError(
                f"{path}: allocated registers per block ({registers_per_block}) "
                f"is below registers-per-thread * threads "
                f"({registers_per_thread * threads})"
            )

        allocated_shared = metric("launch__shared_mem_per_block_allocated")
        static_shared = metric("launch__shared_mem_per_block_static")
        dynamic_shared = metric("launch__shared_mem_per_block_dynamic")
        if allocated_shared is not None:
            shared_total = _byte_value(*allocated_shared)
            static_bytes = 0
            dynamic_bytes = shared_total
        else:
            static_bytes = 0 if static_shared is None else _byte_value(*static_shared)
            dynamic_bytes = 0 if dynamic_shared is None else _byte_value(*dynamic_shared)

        cc = _canonical_cc(first(representative, "CC", "Compute Capability"), compute_capability)
        limits = dict(_SM_LIMITS[cc])
        warps = max(1, math.ceil(threads / 32))
        independent_limits = [
            limits["sm_max_threads"] // threads,
            limits["sm_max_warps"] // warps,
            limits["sm_register_capacity"] // registers_per_block if registers_per_block else limits["sm_max_blocks"],
            limits["sm_shared_mem_bytes"] // (static_bytes + dynamic_bytes) if (static_bytes + dynamic_bytes) else limits["sm_max_blocks"],
            limits["sm_max_blocks"],
        ]
        theoretical_blocks = max(1, min(independent_limits))
        stream_metric = metric("launch__stream_id")
        stream_id = first(representative, "Stream", "Stream ID")
        if stream_metric is not None:
            try:
                stream_id = str(int(round(_clean_number(stream_metric[0]))))
            except ValueError:
                pass
        duration_metric = metric("gpu__time_duration.sum")
        duration_ns = 0 if duration_metric is None else _duration_ns(*duration_metric)

        normalized.append({
            "role": role.upper(),
            "operation": operation,
            "family": family.upper(),
            "layer": "" if layer is None else int(layer),
            "launch_id": key[0],
            "process_id": first(representative, "Process ID", "PID"),
            "context_id": key[2],
            "stream_id": stream_id,
            "cuda_kernel_name": key[1],
            "duration_ns": duration_ns,
            "compute_capability": cc,
            "threads_per_block": threads,
            "warps_per_block": warps,
            "registers_per_thread": registers_per_thread,
            "registers_per_block": registers_per_block,
            "static_shared_mem_bytes": static_bytes,
            "dynamic_shared_mem_bytes": dynamic_bytes,
            "theoretical_blocks_per_sm": theoretical_blocks,
            **limits,
            "resource_source": "ncu_raw_csv",
        })
    return normalized


def parse_processing_nvtx_label(value: Any) -> Dict[str, Any] | None:
    text = str(value or "")
    marker = "THOG2_PROCESSING|"
    offset = text.find(marker)
    if offset < 0:
        return None
    # With `ncu --print-nvtx-rename kernel`, NCU appends the original CUDA
    # kernel name after the semantic range as `...|layer=N/<kernel>`.  Parse
    # only the range label so the layer remains an integer.
    label = text[offset:].split("/", 1)[0]
    fields: Dict[str, Any] = {}
    for part in label.split("|")[1:]:
        if "=" not in part:
            break
        key, raw = part.split("=", 1)
        fields[key.strip()] = raw.strip()
    owner = str(fields.get("owner", "")).upper()
    operation = str(fields.get("operation", "")).lower()
    family = str(fields.get("family", "")).upper()
    if owner not in {"MAIN", "PREMAT"} or not operation or not family:
        return None
    try:
        layer = int(fields.get("layer", ""))
    except (TypeError, ValueError):
        return None
    return {"role": owner, "operation": operation, "family": family, "layer": layer}


def _semantic_launch_map(path: Path) -> Dict[tuple[str, str, str, str], Dict[str, Any]]:
    rows = _raw_ncu_reader(path)

    def first(row: Mapping[str, Any], *names: str) -> str:
        for name in names:
            value = row.get(name)
            if value not in (None, ""):
                return str(value)
        return ""

    mapping: Dict[tuple[str, str, str, str], Dict[str, Any]] = {}
    for row in rows:
        semantic = parse_processing_nvtx_label(first(row, "Kernel Name", "Kernel", "Name"))
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


def normalize_semantic_ncu_exports(
    raw_path: Path,
    semantic_path: Path,
    *,
    compute_capability: str | None = None,
) -> list[Dict[str, Any]]:
    rows = normalize_raw_ncu_csv(
        raw_path,
        role="UNKNOWN",
        family="UNKNOWN",
        operation="unknown",
        layer=None,
        compute_capability=compute_capability,
    )
    semantic = _semantic_launch_map(semantic_path)
    normalized: list[Dict[str, Any]] = []
    for row in rows:
        key = (
            str(row.get("launch_id", "")),
            str(row.get("process_id", "")),
            str(row.get("context_id", "")),
            str(row.get("stream_id", "")),
        )
        label = semantic.get(key)
        if label is None:
            continue
        updated = dict(row)
        updated.update(label)
        normalized.append(updated)
    roles = {str(row.get("role", "")).upper() for row in normalized}
    if "MAIN" not in roles or "PREMAT" not in roles:
        raise ValueError(
            "NCU semantic export did not contain both MAIN consume and PREMAT materialize launches"
        )
    return normalized


def select_representative_kernel_resources(
    rows: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    groups: Dict[tuple[str, str, str, Any], list[Dict[str, Any]]] = {}
    for source in rows:
        row = dict(source)
        key = (
            str(row.get("role", "")).upper(),
            str(row.get("operation", "")).lower(),
            str(row.get("family", "")).upper(),
            row.get("layer", ""),
        )
        groups.setdefault(key, []).append(row)
    selected: list[Dict[str, Any]] = []
    for key, candidates in groups.items():
        timed = []
        for row in candidates:
            try:
                duration = int(float(row.get("duration_ns", 0)))
            except (TypeError, ValueError):
                duration = 0
            if duration > 0:
                timed.append((duration, row))
        if not timed:
            raise ValueError(
                "NCU representative selection requires gpu__time_duration.sum; "
                f"no duration was recorded for {key}"
            )
        selected.append(max(timed, key=lambda item: item[0])[1])
    selected.sort(
        key=lambda row: (
            0 if str(row.get("role", "")).upper() == "MAIN" else 1,
            int(row.get("layer", 0) or 0),
            str(row.get("family", "")),
        )
    )
    return selected


def write_kernel_resources(rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no normalized NCU kernel-resource rows to write")
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_outputs(rows: Sequence[Mapping[str, Any]], output_directory: Path) -> Dict[str, Path]:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    kernel_path = write_kernel_resources(rows, output_directory / "processing_ncu_kernel_resources.csv")
    compatibility = build_compatibility_rows(rows)
    json_path = output_directory / "processing_premat_compatibility.json"
    csv_path = output_directory / "processing_premat_compatibility.csv"
    json_path.write_text(json.dumps({
        "schema_version": 1,
        "interpretation": "aggregate_sm_budget_only",
        "rows": compatibility,
    }, indent=2, sort_keys=True))
    fieldnames = list(compatibility[0]) if compatibility else []
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(compatibility)
    return {"kernel_resources": kernel_path, "json": json_path, "csv": csv_path}


def _raw_spec(value: str) -> tuple[str, Path, int | None]:
    # FAMILY=path.csv or FAMILY:LAYER=path.csv. Layer is zero-based, matching
    # Processing's stored layer convention.
    label, separator, raw_path = str(value).partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("expected FAMILY=path.csv or FAMILY:LAYER=path.csv")
    family, colon, raw_layer = label.partition(":")
    layer = None
    if colon:
        try:
            layer = int(raw_layer)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"invalid zero-based layer in {value!r}") from error
    return family.upper(), Path(raw_path), layer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build theoretical MAIN/PREMAT co-residency data from normalized or raw "
            "Nsight Compute kernel-resource data."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="existing processing_ncu_kernel_resources.csv")
    source.add_argument(
        "--raw-ncu", action="store_true",
        help="normalize repeated --main/--premat ncu --csv --page raw exports first",
    )
    parser.add_argument(
        "--main", action="append", default=[], type=_raw_spec, metavar="FAMILY[:LAYER]=CSV",
        help="raw NCU CSV for one representative MAIN consume family; repeat as needed",
    )
    parser.add_argument(
        "--premat", action="append", default=[], type=_raw_spec, metavar="FAMILY[:LAYER]=CSV",
        help="raw NCU CSV for one representative PREMAT materialisation family; repeat as needed",
    )
    parser.add_argument(
        "--compute-capability", default=None,
        help="override/inject compute capability for raw CSV (e.g. 8.9)",
    )
    parser.add_argument("--output-directory", required=True, type=Path, help="run processing directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.raw_ncu:
        if not args.main or not args.premat:
            raise SystemExit("--raw-ncu requires at least one --main and one --premat CSV")
        rows: list[Dict[str, Any]] = []
        for family, path, layer in args.main:
            rows.extend(normalize_raw_ncu_csv(
                path, role="MAIN", family=family, operation="consume", layer=layer,
                compute_capability=args.compute_capability,
            ))
        for family, path, layer in args.premat:
            rows.extend(normalize_raw_ncu_csv(
                path, role="PREMAT", family=family, operation="materialize", layer=layer,
                compute_capability=args.compute_capability,
            ))
    else:
        if args.main or args.premat:
            raise SystemExit("--main/--premat are used only with --raw-ncu")
        rows = read_kernel_resources(args.input)

    paths = write_outputs(rows, args.output_directory)
    print(paths["kernel_resources"])
    print(paths["json"])
    print(paths["csv"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# ^^^ THOG
