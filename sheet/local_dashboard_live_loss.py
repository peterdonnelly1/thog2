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


class LiveLossReader:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.path: Path | None = None
        self.identity: tuple[int, int] | None = None
        self.offset = 0
        self.pending = b""
        self.revision = 0
        self.values: dict[str, dict[int, float]] = {"train": {}, "val": {}}

    def refresh(self, catalog: Any, state: Any, dashboard: Any) -> None:
        with self.lock:
            path = self.path if self.path and self.path.is_file() else _resolve_train_log(catalog, state, dashboard)
            if path is None:
                return
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity != self.identity or stat.st_size < self.offset:
                self.offset = 0
                self.pending = b""
                self.values = {"train": {}, "val": {}}
                self.revision += 1
            self.path, self.identity = path, identity
            with path.open("rb") as handle:
                handle.seek(self.offset)
                content = handle.read(1024 * 1024)
                self.offset = handle.tell()
            lines = (self.pending + content).split(b"\n")
            self.pending = lines.pop()[-65536:]
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
                if math.isfinite(value) and self.values[group].get(step) != value:
                    self.values[group][step] = value
                    self.revision += 1
            for group, values in self.values.items():
                if len(values) > 3200:
                    self.values[group] = dict(sorted(values.items())[-3200:])

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
                         "available_x_axis_modes": ["step"], "series": []}
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
                series["x"] = list(series["x"]) + [step for step, _ in tail]
                series["y"] = list(series["y"]) + [value for _, value in tail]
                series["x_variants"] = {
                    key: list(values) + ([step for step, _ in tail] if key == "step" else [None] * len(tail))
                    for key, values in series.get("x_variants", {}).items()}
                series["x_variants"]["step"] = list(series["x"])
                series["point_sources"] = ["W&B"] * count + ["train.log (printed precision)"] * len(tail)
                series["points"] = len(series["x"])
                chart["series"] = [series, *chart["series"][1:]]
            return {**payload, "charts": charts, "revision": int(payload.get("revision", 0)) + self.revision}
# ^^^ THOG
