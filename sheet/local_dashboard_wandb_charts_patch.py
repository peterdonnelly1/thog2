# vvv THOG
"""Expose W&B's already-local history/system records to the THOG2 dashboard.

The dashboard remains local: this reads the run-<id>.wandb file written by the
W&B SDK and incrementally scans only newly appended protobuf records.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional
from urllib.parse import parse_qs, urlparse


_MAX_POINTS_PER_SERIES = 1600
_SCAN_TIME_BUDGET_SECONDS = 0.22
_X_AXIS_MODE_ORDER = ("step", "relative_wall", "relative_process", "wall_time")
_GPU_METRIC_PATTERN = re.compile(r"^(?:system[./])?gpu[./](?P<index>\d+)[./](?P<metric>.+)$", re.IGNORECASE)
_GPU_PROCESS_METRIC_PATTERN = re.compile(r"^(?:system[./])?gpu[./]process[./](?P<index>\d+)[./](?P<metric>.+)$", re.IGNORECASE)

_SYSTEM_TITLES = {
    "gpu.fanspeed": "GPU Fan Speed (%)",
    "gpu.memoryclock": "GPU Memory Clock Speed (MHz)",
    "gpu.smclock": "GPU Streaming Multiprocessor (SM) Clock Speed (MHz)",
    "gpu.powerwatts": "GPU Power Usage (W)",
    "gpu.powerpercent": "GPU Power Usage (%)",
    "gpu.gpu": "GPU Usage (%)",
    "gpu.memory": "GPU Memory Usage (%)",
    "gpu.temp": "GPU Temperature (°C)",
    "cpu": "CPU Utilization (%)",
    "memory": "System Memory Usage (%)",
    "proc.cpu.threads": "Process CPU Threads",
    "proc.memory.rssmb": "Process Memory RSS (MB)",
}


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float) and math.isfinite(value):
        return float(value)
    return None


def _json_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _flatten_numeric(prefix: str, value: Any) -> Iterable[tuple[str, float]]:
    numeric = _finite_number(value)
    if numeric is not None:
        yield prefix, numeric
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten_numeric(child, nested)


def _metric_key(item: Any) -> str:
    key = str(getattr(item, "key", "") or "")
    if key:
        return key
    nested = tuple(str(value) for value in getattr(item, "nested_key", ()) if str(value))
    return "/".join(nested)


def _group_name(metric_name: str) -> str:
    clean = metric_name.strip("/.")
    if not clean:
        return "other"
    for separator in ("/", "."):
        if separator in clean:
            return clean.split(separator, 1)[0]
    return "charts"


def _pretty_metric_name(metric_name: str) -> str:
    clean = metric_name.strip("/.")
    if "/" in clean:
        clean = clean.split("/", 1)[1]
    clean = clean.replace("_", " ").replace(".", " ")
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", clean).split()
    return " ".join(word.upper() if word.lower() in {"gpu", "cpu", "sm", "rss", "mb", "gb"} else word.capitalize() for word in words)


def _system_chart_identity(metric_name: str) -> tuple[str, str, str]:
    clean = metric_name.strip("/.")
    if clean.lower().startswith("system."):
        clean = clean[len("system."):]
    elif clean.lower().startswith("system/"):
        clean = clean[len("system/"):]

    process_match = _GPU_PROCESS_METRIC_PATTERN.match(clean)
    if process_match:
        metric = process_match.group("metric")
        index = process_match.group("index")
        chart_id = f"gpu.process.{metric}"
        title = _SYSTEM_TITLES.get(chart_id.lower(), _pretty_metric_name(chart_id))
        return chart_id, title, f"GPU process {index}"

    gpu_match = _GPU_METRIC_PATTERN.match(clean)
    if gpu_match:
        metric = gpu_match.group("metric")
        index = gpu_match.group("index")
        chart_id = f"gpu.{metric}"
        title = _SYSTEM_TITLES.get(chart_id.lower(), _pretty_metric_name(chart_id))
        return chart_id, title, f"GPU {index}"

    chart_id = clean or metric_name
    title = _SYSTEM_TITLES.get(chart_id.lower(), _pretty_metric_name(chart_id))
    return chart_id, title, metric_name


def _history_chart_identity(metric_name: str) -> tuple[str, str, str, str]:
    group = _group_name(metric_name)
    title = metric_name.split("/", 1)[1] if "/" in metric_name else metric_name
    return group, metric_name, _pretty_metric_name(title), metric_name


def _downsample_points(
    points: list[tuple[float, float, Optional[float], Optional[float], Optional[float], Optional[float]]],
    limit: int = _MAX_POINTS_PER_SERIES,
) -> list[tuple[float, float, Optional[float], Optional[float], Optional[float], Optional[float]]]:
    if len(points) <= limit:
        return list(points)
    if limit <= 2:
        return [points[0], points[-1]]

    # Preserve endpoints plus local extrema from each bucket. This retains the
    # spikes that matter in system metrics better than uniform point skipping.
    bucket_count = max(1, (limit - 2) // 2)
    span = len(points) - 2
    selected = [points[0]]
    for bucket in range(bucket_count):
        start = 1 + (bucket * span) // bucket_count
        stop = 1 + ((bucket + 1) * span) // bucket_count
        chunk = points[start:stop]
        if not chunk:
            continue
        minimum = min(chunk, key=lambda point: point[1])
        maximum = max(chunk, key=lambda point: point[1])
        if minimum[0] <= maximum[0]:
            selected.extend((minimum, maximum))
        else:
            selected.extend((maximum, minimum))
    selected.append(points[-1])
    if len(selected) > limit:
        stride = max(1, math.ceil(len(selected) / limit))
        selected = selected[::stride]
        if selected[-1] != points[-1]:
            selected.append(points[-1])
    return selected


class _WandbRunScanner:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.lock = threading.Lock()
        self.store: Any = None
        self.record_class: Any = None
        self.good_offset = 0
        self.record_count = 0
        self.history_step = 0.0
        self.first_history_timestamp: Optional[float] = None
        self.first_stats_timestamp: Optional[float] = None
        self.process_anchor_timestamp: Optional[float] = None
        self.process_anchor_runtime: Optional[float] = None
        self.series: Dict[
            str,
            Dict[
                str,
                list[
                    tuple[
                        float,
                        float,
                        Optional[float],
                        Optional[float],
                        Optional[float],
                        Optional[float],
                    ]
                ],
            ],
        ] = defaultdict(lambda: defaultdict(list))
        self.chart_titles: Dict[str, Dict[str, str]] = defaultdict(dict)
        self.x_titles: Dict[str, Dict[str, str]] = defaultdict(dict)
        self.default_x_axis_modes: Dict[str, Dict[str, str]] = defaultdict(dict)
        self.group_revisions: Dict[str, int] = defaultdict(int)
        self.error = ""
        self.catching_up = True

    def _open(self) -> None:
        from wandb.proto import wandb_internal_pb2
        from wandb.sdk.internal.datastore import DataStore

        self.store = DataStore()
        self.store.open_for_scan(str(self.path))
        self.record_class = wandb_internal_pb2.Record
        self.good_offset = int(self.store.get_offset())

    def _reset(self) -> None:
        try:
            if self.store is not None and getattr(self.store, "_fp", None) is not None:
                self.store._fp.close()
        except Exception:
            pass
        self.store = None
        self.good_offset = 0
        self.record_count = 0
        self.history_step = 0.0
        self.first_history_timestamp = None
        self.first_stats_timestamp = None
        self.process_anchor_timestamp = None
        self.process_anchor_runtime = None
        self.series.clear()
        self.chart_titles.clear()
        self.x_titles.clear()
        self.default_x_axis_modes.clear()
        self.group_revisions.clear()
        self.error = ""
        self.catching_up = True

    def _append(
        self,
        group: str,
        chart_id: str,
        title: str,
        series_name: str,
        x: float,
        y: float,
        x_title: str,
        *,
        step: Optional[float],
        relative_wall_seconds: Optional[float],
        relative_process_seconds: Optional[float],
        wall_time_epoch_seconds: Optional[float],
        default_x_axis_mode: str,
    ) -> None:
        points = self.series[group][f"{chart_id}\0{series_name}"]
        points.append((
            float(x),
            float(y),
            step,
            relative_wall_seconds,
            relative_process_seconds,
            wall_time_epoch_seconds,
        ))
        self.chart_titles[group][chart_id] = title
        self.x_titles[group][chart_id] = x_title
        self.default_x_axis_modes[group][chart_id] = default_x_axis_mode
        self.group_revisions[group] += 1

    def _consume_history(self, history: Any) -> None:
        explicit_step = float(getattr(getattr(history, "step", None), "num", self.history_step) or self.history_step)
        values: Dict[str, Any] = {}
        for item in getattr(history, "item", ()):
            key = _metric_key(item)
            if not key:
                continue
            values[key] = _json_value(getattr(item, "value_json", "null"))
        if "_step" in values:
            numeric_step = _finite_number(values["_step"])
            if numeric_step is not None:
                explicit_step = numeric_step
        self.history_step = explicit_step
        timestamp = _finite_number(values.get("_timestamp"))
        runtime = _finite_number(values.get("_runtime"))
        if timestamp is not None and self.first_history_timestamp is None:
            self.first_history_timestamp = timestamp
        relative_wall_seconds = (
            max(0.0, timestamp - self.first_history_timestamp)
            if timestamp is not None and self.first_history_timestamp is not None
            else None
        )
        if timestamp is not None and runtime is not None:
            self.process_anchor_timestamp = timestamp
            self.process_anchor_runtime = runtime

        for metric_name, value in values.items():
            if metric_name.startswith("_"):
                continue
            for flattened_name, numeric in _flatten_numeric(metric_name, value):
                group, chart_id, title, series_name = _history_chart_identity(flattened_name)
                self._append(
                    group,
                    chart_id,
                    title,
                    series_name,
                    explicit_step,
                    numeric,
                    "step",
                    step=explicit_step,
                    relative_wall_seconds=relative_wall_seconds,
                    relative_process_seconds=runtime,
                    wall_time_epoch_seconds=timestamp,
                    default_x_axis_mode="step",
                )

    def _consume_stats(self, stats: Any) -> None:
        timestamp = getattr(stats, "timestamp", None)
        seconds = float(getattr(timestamp, "seconds", 0) or 0)
        nanos = float(getattr(timestamp, "nanos", 0) or 0)
        epoch = seconds + nanos / 1_000_000_000.0
        if self.first_stats_timestamp is None:
            self.first_stats_timestamp = epoch
        relative_wall_seconds = max(0.0, epoch - self.first_stats_timestamp)
        elapsed_minutes = relative_wall_seconds / 60.0
        relative_process_seconds = (
            max(0.0, self.process_anchor_runtime + epoch - self.process_anchor_timestamp)
            if self.process_anchor_timestamp is not None and self.process_anchor_runtime is not None
            else None
        )

        for item in getattr(stats, "item", ()):
            key = _metric_key(item)
            if not key:
                continue
            value = _json_value(getattr(item, "value_json", "null"))
            for flattened_name, numeric in _flatten_numeric(key, value):
                chart_id, title, series_name = _system_chart_identity(flattened_name)
                self._append(
                    "system",
                    chart_id,
                    title,
                    series_name,
                    elapsed_minutes,
                    numeric,
                    "Time (minutes)",
                    step=None,
                    relative_wall_seconds=relative_wall_seconds,
                    relative_process_seconds=relative_process_seconds,
                    wall_time_epoch_seconds=epoch,
                    default_x_axis_mode="relative_wall",
                )

    def _consume_record(self, record: Any) -> None:
        record_type = record.WhichOneof("record_type")
        if record_type == "history":
            self._consume_history(record.history)
        elif record_type == "stats":
            self._consume_stats(record.stats)

    def refresh(self) -> None:
        with self.lock:
            try:
                if self.store is None:
                    self._open()
                current_size = int(self.path.stat().st_size)
                if current_size < self.good_offset:
                    self._reset()
                    self._open()

                deadline = time.monotonic() + _SCAN_TIME_BUDGET_SECONDS
                reached_eof = False
                while time.monotonic() < deadline:
                    before = int(self.store.get_offset())
                    try:
                        payload = self.store.scan_data()
                    except (AssertionError, EOFError, OSError, ValueError):
                        self.store.seek(before)
                        break
                    if payload is None:
                        self.store.seek(before)
                        reached_eof = True
                        break
                    record = self.record_class()
                    record.ParseFromString(payload)
                    self._consume_record(record)
                    self.record_count += 1
                    self.good_offset = int(self.store.get_offset())
                self.catching_up = not reached_eof
                self.error = ""
            except Exception as error:
                self.error = str(error)
                self.catching_up = False

    def group_summaries(self) -> list[dict[str, Any]]:
        self.refresh()
        with self.lock:
            groups = []
            for group in sorted(self.series, key=lambda name: (name not in {"train", "val", "system"}, {"train": 0, "val": 1, "system": 2}.get(name, 99), name)):
                chart_ids = {key.split("\0", 1)[0] for key in self.series[group]}
                if not chart_ids:
                    continue
                groups.append({
                    "name": group,
                    "chart_count": len(chart_ids),
                    "revision": int(self.group_revisions[group]),
                })
            return groups

    def group_payload(self, group: str) -> dict[str, Any]:
        self.refresh()
        with self.lock:
            chart_series: Dict[str, list[dict[str, Any]]] = defaultdict(list)
            for encoded, points in self.series.get(group, {}).items():
                chart_id, series_name = encoded.split("\0", 1)
                selected = _downsample_points(points)
                x_variants = {}
                for mode_index, mode_name in enumerate(_X_AXIS_MODE_ORDER, start=2):
                    values = [point[mode_index] for point in selected]
                    if any(value is not None for value in values):
                        x_variants[mode_name] = values
                chart_series[chart_id].append({
                    "name": series_name,
                    "x": [point[0] for point in selected],
                    "x_variants": x_variants,
                    "y": [point[1] for point in selected],
                    "points": len(points),
                })
            charts = []
            for chart_id in sorted(chart_series, key=lambda value: self.chart_titles[group].get(value, value).lower()):
                series = chart_series[chart_id]
                available_modes = [
                    mode_name
                    for mode_name in _X_AXIS_MODE_ORDER
                    if all(mode_name in item["x_variants"] for item in series)
                ]
                charts.append({
                    "id": chart_id,
                    "title": self.chart_titles[group].get(chart_id, _pretty_metric_name(chart_id)),
                    "x_title": self.x_titles[group].get(chart_id, "step"),
                    "default_x_axis_mode": self.default_x_axis_modes[group].get(chart_id, "step"),
                    "available_x_axis_modes": available_modes,
                    "series": series,
                })
            return {
                "name": group,
                "revision": int(self.group_revisions[group]),
                "charts": charts,
            }


class _ScannerCatalog:
    def __init__(self, catalog: Any) -> None:
        self.catalog = catalog
        self.lock = threading.Lock()
        self.paths: Dict[str, Optional[Path]] = {}
        self.scanners: Dict[Path, _WandbRunScanner] = {}

    def _find_path(self, run_id: str, status: Optional[dict[str, Any]] = None) -> Optional[Path]:
        cached = self.paths.get(run_id)
        if cached is not None and cached.exists():
            return cached
        if run_id in self.paths and self.paths[run_id] is None:
            # Recheck missing live runs; W&B may create the file after the dashboard starts.
            pass

        candidates = []
        status = status or {}
        recorded_directory = status.get("wandb_run_directory")
        if recorded_directory:
            directory = Path(recorded_directory)
            # W&B run.dir usually ends in /files; accept the run directory too.
            for parent in (directory, directory.parent):
                recorded_path = parent / f"run-{run_id}.wandb"
                if recorded_path.is_file():
                    self.paths[run_id] = recorded_path.resolve()
                    return self.paths[run_id]
        project_root = Path(self.catalog.root).resolve().parent
        roots = [Path.cwd() / "wandb", Path.cwd(), project_root / "wandb"]
        configured_root = (status.get("configuration") or {}).get("wandb_root")
        if configured_root:
            configured_path = Path(configured_root)
            for anchor in (Path.cwd(), project_root):
                roots.extend((anchor / configured_path, anchor / configured_path / "wandb"))
        seen_roots = set()
        for root in roots:
            try:
                resolved = root.resolve()
            except OSError:
                continue
            if resolved in seen_roots or not resolved.exists():
                continue
            seen_roots.add(resolved)
            for pattern in (f"run-*-{run_id}/run-{run_id}.wandb", f"offline-run-*-{run_id}/run-{run_id}.wandb"):
                candidates.extend(path for path in resolved.glob(pattern) if path.is_file())
            if resolved == Path.cwd().resolve():
                candidates.extend(path for path in resolved.glob(f"**/run-{run_id}.wandb") if path.is_file())
        unique = {path.resolve(): path.resolve() for path in candidates}
        chosen = max(unique.values(), key=lambda path: path.stat().st_mtime, default=None)
        self.paths[run_id] = chosen
        return chosen

    def scanner_for(self, state: Any) -> Optional[_WandbRunScanner]:
        status = state.status()
        run_id = str(status.get("wandb_run_id") or "").strip()
        if not run_id:
            return None
        path = self._find_path(run_id, status)
        if path is None:
            return None
        with self.lock:
            scanner = self.scanners.get(path)
            if scanner is None:
                scanner = _WandbRunScanner(path)
                self.scanners[path] = scanner
            return scanner


def install(dashboard_module: Any) -> None:
    original_handler_for = dashboard_module._handler_for

    def handler_for(catalog: Any):
        scanner_catalog = _ScannerCatalog(catalog)
        handler = original_handler_for(catalog)
        original_do_get = handler.do_GET

        def do_get(self: Any) -> None:
            parsed = urlparse(self.path)
            if parsed.path not in {"/api/chart-groups", "/api/chart-group"}:
                original_do_get(self)
                return
            query = parse_qs(parsed.query)
            run_name = query.get("run", [""])[0]
            if not run_name:
                self._send_json({"error": "run query parameter is required"}, status=dashboard_module.HTTPStatus.BAD_REQUEST)
                return
            try:
                state = catalog.state_for_run(run_name)
                scanner = scanner_catalog.scanner_for(state)
                if scanner is None:
                    self._send_json({
                        "available": False,
                        "groups": [],
                        "reason": "no local W&B run file found for this run",
                    })
                    return
                if parsed.path == "/api/chart-groups":
                    groups = scanner.group_summaries()
                    self._send_json({
                        "available": True,
                        "source": str(scanner.path.resolve()),
                        "record_count": scanner.record_count,
                        "catching_up": scanner.catching_up,
                        "error": scanner.error,
                        "groups": groups,
                    })
                    return
                group = query.get("group", [""])[0]
                if not group:
                    self._send_json({"error": "group query parameter is required"}, status=dashboard_module.HTTPStatus.BAD_REQUEST)
                    return
                self._send_json({
                    "available": True,
                    "source": str(scanner.path.resolve()),
                    "record_count": scanner.record_count,
                    "catching_up": scanner.catching_up,
                    "error": scanner.error,
                    "group": scanner.group_payload(group),
                })
            except (FileNotFoundError, KeyError) as error:
                self._send_json({"error": str(error)}, status=dashboard_module.HTTPStatus.NOT_FOUND)
            except Exception as error:
                self._send_json({"error": str(error)}, status=dashboard_module.HTTPStatus.INTERNAL_SERVER_ERROR)

        handler.do_GET = do_get
        return handler

    dashboard_module._handler_for = handler_for
# ^^^ THOG
