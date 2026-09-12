# vvv THOG
"""Early console losses while W&B's local history is buffered.

Console values retain printed precision. Committed W&B values replace console
values through its last step; training and validation remain separate.
"""
from __future__ import annotations

import math
import re
import threading
from pathlib import Path
from typing import Any

from sheet.local_dashboard_logs_patch import _resolve_train_log

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_ROW = re.compile(r"^\s*(?P<kind>[TV])\s+(?P<step>\d+)\s+")
_NUMBER = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
_LOSS = re.compile(r"(?<![\w/])loss\s*=\s*" + _NUMBER + r"(?![\w.])")
_VAL_LOSS = re.compile(r"validation\s+loss\s*=\s*" + _NUMBER + r"(?![\w.])")
_STEP_SECONDS = re.compile(r"Δstep\s*=\s*" + _NUMBER + r"\s*s")
_TIME_AXIS_MODES = ("relative_wall", "relative_process", "wall_time")
# vvv THOG keep restart/backlog reconstruction anchored to the newest train.log tail rather than stale chunks
_READ_CHUNK_BYTES = 1024 * 1024
_LIVE_TAIL_BYTES = 4 * 1024 * 1024
# ^^^ THOG


class LiveLossReader:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.path: Path | None = None
        self.identity: tuple[int, int] | None = None
        self.offset = 0
        self.pending = b""
        self.revision = 0
        self.values: dict[str, dict[int, float]] = {"train": {}, "val": {}}
        # vvv THOG provisional live-tail timestamps bridge W&B history buffering on all Instra time axes
        self.wall_times: dict[str, dict[int, float]] = {"train": {}, "val": {}}
        self.first_wall_time: float | None = None
        # ^^^ THOG

    def refresh(self, catalog: Any, state: Any, dashboard: Any) -> None:
        with self.lock:
            path = self.path if self.path and self.path.is_file() else _resolve_train_log(catalog, state, dashboard)
            if path is None:
                return
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            reset_reader = identity != self.identity or stat.st_size < self.offset
            if reset_reader:
                self.offset = 0
                self.pending = b""
                self.values = {"train": {}, "val": {}}
                # vvv THOG a rotated/restarted train.log starts a fresh provisional timing origin
                self.wall_times = {"train": {}, "val": {}}
                self.first_wall_time = None
                # ^^^ THOG
                self.revision += 1
            self.path, self.identity = path, identity
            # vvv THOG restart from the newest bounded tail and drain any modest backlog to EOF before assigning mtime-derived timestamps
            backlog_bytes = max(0, int(stat.st_size) - int(self.offset))
            drain_to_eof = reset_reader or backlog_bytes > _READ_CHUNK_BYTES
            discard_partial_prefix = False
            if reset_reader:
                self.offset = max(0, int(stat.st_size) - _LIVE_TAIL_BYTES)
                discard_partial_prefix = self.offset > 0
            elif drain_to_eof:
                tail_offset = max(int(self.offset), int(stat.st_size) - _LIVE_TAIL_BYTES)
                if tail_offset > self.offset:
                    self.offset = tail_offset
                    self.pending = b""
                    discard_partial_prefix = True
            with path.open("rb") as handle:
                handle.seek(self.offset)
                content = handle.read() if drain_to_eof else handle.read(_READ_CHUNK_BYTES)
                self.offset = handle.tell()
            if discard_partial_prefix:
                newline = content.find(b"\n")
                content = content[newline + 1:] if newline >= 0 else b""
            # ^^^ THOG
            lines = (self.pending + content).split(b"\n")
            self.pending = lines.pop()[-65536:]
            # vvv THOG train.log mtime anchors the newest live row; exact Δstep values backfill earlier rows in the same read
            accepted: list[tuple[str, int, float, float | None]] = []
            for raw in lines:
                line = _ANSI.sub("", raw.decode("utf-8", errors="replace"))
                row = _ROW.match(line)
                if not row:
                    continue
                group = "train" if row["kind"] == "T" else "val"
                match = (_LOSS if group == "train" else _VAL_LOSS).search(line)
                if not match:
                    continue
                value, step = float(match[1]), int(row["step"])
                step_seconds_match = _STEP_SECONDS.search(line)
                step_seconds = (
                    float(step_seconds_match[1])
                    if step_seconds_match is not None
                    else None
                )
                if math.isfinite(value) and self.values[group].get(step) != value:
                    accepted.append((group, step, value, step_seconds))
            # vvv THOG restat after the read so the anchor corresponds to the newest bytes actually consumed
            wall_cursor = float(path.stat().st_mtime)
            # ^^^ THOG
            for group, step, value, step_seconds in reversed(accepted):
                self.values[group][step] = value
                self.wall_times[group][step] = wall_cursor
                self.first_wall_time = (
                    wall_cursor
                    if self.first_wall_time is None
                    else min(self.first_wall_time, wall_cursor)
                )
                self.revision += 1
                if (
                    group == "train"
                    and step_seconds is not None
                    and math.isfinite(step_seconds)
                    and step_seconds >= 0.0
                ):
                    wall_cursor = max(0.0, wall_cursor - step_seconds)
            # ^^^ THOG
            for group, values in self.values.items():
                if len(values) > 3200:
                    kept_steps = {step for step, _value in sorted(values.items())[-3200:]}
                    self.values[group] = {
                        step: value
                        for step, value in values.items()
                        if step in kept_steps
                    }
                    # vvv THOG bound provisional timing state with its corresponding live loss rows
                    self.wall_times[group] = {
                        step: wall_time
                        for step, wall_time in self.wall_times[group].items()
                        if step in kept_steps
                    }
                    # ^^^ THOG

    def summaries(self, groups: list[dict[str, Any]], scanner: Any) -> list[dict[str, Any]]:
        # No repeated scans or chart serialization during the one-second discovery poll.
        result = [dict(group) for group in groups]
        with self.lock:
            for name in ("train", "val"):
                if not self.values[name]:
                    continue
                metric = "train/loss" if name == "train" else "val/val_loss"
                existing = next((group for group in result if group["name"] == name), None)
                has_chart = False
                if scanner is not None:
                    with scanner.lock:
                        has_chart = any(key.split("\0", 1)[0] == metric for key in scanner.series.get(name, {}))
                if existing is None:
                    existing = {"name": name, "chart_count": 0, "revision": 0}
                    result.append(existing)
                existing["chart_count"] += int(not has_chart)
                existing["revision"] += self.revision
        return result

    # vvv THOG project provisional train.log timestamps from the latest exact W&B timing anchor without changing W&B axis semantics
    @staticmethod
    def _last_exact_time_anchor(series: dict[str, Any]) -> tuple[float, float, float] | None:
        variants = series.get("x_variants", {})
        wall_values = variants.get("wall_time", [])
        relative_wall_values = variants.get("relative_wall", [])
        relative_process_values = variants.get("relative_process", [])
        count = min(len(wall_values), len(relative_wall_values), len(relative_process_values))
        for index in range(count - 1, -1, -1):
            values = (
                wall_values[index],
                relative_wall_values[index],
                relative_process_values[index],
            )
            if all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
                return tuple(float(value) for value in values)
        return None

    def _tail_time_variants(
        self,
        group: str,
        tail: list[tuple[int, float]],
        series: dict[str, Any],
    ) -> dict[str, list[float]]:
        anchor = self._last_exact_time_anchor(series)
        raw_times = [self.wall_times[group].get(step) for step, _value in tail]
        finite_raw_times = [
            float(value)
            for value in raw_times
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        ]
        if anchor is not None:
            anchor_wall, anchor_relative_wall, anchor_relative_process = anchor
            origin = anchor_wall
            previous_wall = anchor_wall
        else:
            origin = (
                float(self.first_wall_time)
                if self.first_wall_time is not None
                else (min(finite_raw_times) if finite_raw_times else 0.0)
            )
            anchor_wall = origin
            anchor_relative_wall = 0.0
            anchor_relative_process = 0.0
            previous_wall = origin
        wall_time_values: list[float] = []
        relative_wall_values: list[float] = []
        relative_process_values: list[float] = []
        for raw_time in raw_times:
            wall_time = (
                float(raw_time)
                if isinstance(raw_time, (int, float)) and math.isfinite(float(raw_time))
                else previous_wall
            )
            wall_time = max(previous_wall, wall_time)
            previous_wall = wall_time
            delta = max(0.0, wall_time - anchor_wall)
            wall_time_values.append(wall_time)
            relative_wall_values.append(anchor_relative_wall + delta)
            relative_process_values.append(anchor_relative_process + delta)
        return {
            "relative_wall": relative_wall_values,
            "relative_process": relative_process_values,
            "wall_time": wall_time_values,
        }
    # ^^^ THOG

    def merge(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            group = payload["name"]
            points = self.values.get(group, {})
            if not points:
                return payload
            metric = "train/loss" if group == "train" else "val/val_loss"
            charts = list(payload.get("charts", []))
            chart = next((item for item in charts if item["id"] == metric), None)
            if chart is None:
                chart = {"id": metric, "title": "Loss" if group == "train" else "Val Loss",
                         "x_title": "step", "default_x_axis_mode": "step",
                         "available_x_axis_modes": ["step", *_TIME_AXIS_MODES], "series": []}
                charts.append(chart)
            else:
                chart = dict(chart)
                charts = [chart if item["id"] == metric else item for item in charts]
            series = dict(chart["series"][0]) if chart["series"] else {
                "name": metric, "x": [], "y": [], "x_variants": {"step": []}}
            last_step = max(series["x"], default=-1)
            tail = sorted((step, value) for step, value in points.items() if step > last_step)
            if tail:
                count = len(series["x"])
                tail_steps = [step for step, _value in tail]
                tail_time_variants = self._tail_time_variants(group, tail, series)
                series["x"] = list(series["x"]) + tail_steps
                series["y"] = list(series["y"]) + [value for _step, value in tail]
                existing_variants = {
                    key: list(values)
                    for key, values in series.get("x_variants", {}).items()
                }
                existing_variants["step"] = list(series["x"])
                # vvv THOG exact W&B histories remain authoritative; only append provisional times when the complete axis can be rendered
                if count == 0:
                    for mode in _TIME_AXIS_MODES:
                        existing_variants[mode] = list(tail_time_variants[mode])
                    chart["available_x_axis_modes"] = ["step", *_TIME_AXIS_MODES]
                else:
                    available_modes = list(chart.get("available_x_axis_modes", ["step"]))
                    for mode in _TIME_AXIS_MODES:
                        existing = existing_variants.get(mode)
                        if existing is None or len(existing) != count:
                            continue
                        existing_variants[mode] = existing + tail_time_variants[mode]
                        if mode not in available_modes:
                            available_modes.append(mode)
                    chart["available_x_axis_modes"] = available_modes
                # ^^^ THOG
                series["x_variants"] = existing_variants
                series["point_sources"] = ["W&B"] * count + ["train.log (printed precision)"] * len(tail)
                series["points"] = len(series["x"])
                chart["series"] = [series, *chart["series"][1:]]
            return {**payload, "charts": charts, "revision": int(payload.get("revision", 0)) + self.revision}
# ^^^ THOG
