# vvv THOG
"""Compact concurrent local storage for live THOG chart data."""

from __future__ import annotations

import json
import math
import os
import sqlite3
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


CHART_DESTINATIONS = ("wandb", "local", "none")
LOCAL_CHART_DATABASE_NAME = "charts.sqlite3"
LOCAL_CHART_SCHEMA_VERSION = 1


def normalize_chart_destination(value: Any, *, label: str) -> str:
    destination = str(value).strip().lower()
    if destination not in CHART_DESTINATIONS:
        allowed = ", ".join(CHART_DESTINATIONS)
        raise ValueError(f"{label} must be one of {allowed}; got {value!r}")
    return destination


def local_chart_root() -> Path:
    configured = os.environ.get("THOG2_INSTRUMENTATION_LOCAL_ROOT", "logs").strip()
    if not configured:
        raise ValueError("THOG2_INSTRUMENTATION_LOCAL_ROOT must not be empty")
    return Path(configured)


def local_chart_database_path(run_name: str, *, root: Optional[Path] = None) -> Path:
    normalized_name = Path(str(run_name)).name
    if normalized_name in {"", ".", ".."}:
        raise ValueError(f"invalid local chart run name: {run_name!r}")
    resolved_root = local_chart_root() if root is None else Path(root)
    return resolved_root / normalized_name / LOCAL_CHART_DATABASE_NAME


def _encode_payload(value: Mapping[str, Any]) -> bytes:
    raw = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return zlib.compress(raw, level=6)


def _decode_payload(value: bytes) -> Dict[str, Any]:
    decoded = json.loads(zlib.decompress(value).decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("local chart payload must decode to an object")
    return decoded


def _json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return str(value)


def _open_database(path: Path, *, readonly: bool) -> sqlite3.Connection:
    if readonly:
        connection = sqlite3.connect(
            f"file:{path.resolve()}?mode=ro",
            timeout=30.0,
            uri=True,
        )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=30.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA busy_timeout=30000")
    connection.row_factory = sqlite3.Row
    return connection


class LocalChartStore:
    def __init__(
        self,
        path: Path,
        *,
        run_name: str,
        config: Mapping[str, Any],
    ) -> None:
        self.path = Path(path)
        self.connection = _open_database(self.path, readonly=False)
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS heatmap_records (
                optimizer_update INTEGER PRIMARY KEY,
                probe_id TEXT NOT NULL,
                active_layers INTEGER NOT NULL,
                selected_layers INTEGER NOT NULL,
                payload BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS depth_weight_snapshots (
                optimizer_update INTEGER PRIMARY KEY,
                payload BLOB NOT NULL
            );
            """
        )
        metadata = {
            "schema_version": str(LOCAL_CHART_SCHEMA_VERSION),
            "run_name": str(run_name),
            "config_json": json.dumps(
                _json_compatible(config),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        }
        self.connection.executemany(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            tuple(metadata.items()),
        )
        self.connection.commit()

    def append_heatmap_records(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        maximum_step: Optional[int] = None,
    ) -> int:
        rows = []
        for record in records:
            optimizer_update = int(record["optimizer_update"])
            if maximum_step is not None and optimizer_update > int(maximum_step):
                continue
            point_by_candidate: Dict[int, float] = {}
            for side in ("shrink", "growth"):
                for _distance, delta, candidate, _offset in record.get(side, ()):
                    point_by_candidate[int(candidate)] = float(delta)
            payload = {
                "candidate_layers": sorted(point_by_candidate),
                "delta_losses": [
                    point_by_candidate[candidate]
                    for candidate in sorted(point_by_candidate)
                ],
            }
            rows.append(
                (
                    optimizer_update,
                    str(record["probe_id"]),
                    int(record["active_layers"]),
                    int(record.get("selected_layers", record["active_layers"])),
                    _encode_payload(payload),
                )
            )
        if not rows:
            return 0
        self.connection.executemany(
            """
            INSERT OR REPLACE INTO heatmap_records(
                optimizer_update,
                probe_id,
                active_layers,
                selected_layers,
                payload
            ) VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.connection.commit()
        return len(rows)

    def append_depth_weight_snapshot(
        self,
        snapshot: Mapping[str, Any],
        *,
        history_length: int,
    ) -> None:
        optimizer_update = int(snapshot["optimizer_update"])
        self.connection.execute(
            """
            INSERT OR REPLACE INTO depth_weight_snapshots(optimizer_update, payload)
            VALUES (?, ?)
            """,
            (optimizer_update, _encode_payload(_json_compatible(snapshot))),
        )
        self.connection.execute(
            """
            DELETE FROM depth_weight_snapshots
            WHERE optimizer_update NOT IN (
                SELECT optimizer_update
                FROM depth_weight_snapshots
                ORDER BY optimizer_update DESC
                LIMIT ?
            )
            """,
            (max(1, int(history_length)),),
        )
        self.connection.commit()

    def close(self) -> None:
        if self.connection is None:
            return
        self.connection.execute("PRAGMA wal_checkpoint(PASSIVE)")
        self.connection.close()
        self.connection = None


class LocalChartReader:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _connection(self) -> sqlite3.Connection:
        return _open_database(self.path, readonly=True)

    def metadata(self) -> Dict[str, str]:
        connection = self._connection()
        try:
            rows = connection.execute(
                "SELECT key, value FROM metadata ORDER BY key"
            ).fetchall()
        finally:
            connection.close()
        return {str(row["key"]): str(row["value"]) for row in rows}

    def status(self) -> Dict[str, Any]:
        connection = self._connection()
        try:
            heatmap = connection.execute(
                """
                SELECT COUNT(*) AS count, MAX(optimizer_update) AS maximum_update
                FROM heatmap_records
                """
            ).fetchone()
            depth = connection.execute(
                """
                SELECT COUNT(*) AS count, MAX(optimizer_update) AS maximum_update
                FROM depth_weight_snapshots
                """
            ).fetchone()
        finally:
            connection.close()
        return {
            "heatmap_count": int(heatmap["count"]),
            "heatmap_maximum_update": (
                None if heatmap["maximum_update"] is None else int(heatmap["maximum_update"])
            ),
            "depth_snapshot_count": int(depth["count"]),
            "depth_maximum_update": (
                None if depth["maximum_update"] is None else int(depth["maximum_update"])
            ),
        }

    def heatmap_history(self) -> Tuple[Dict[str, Any], ...]:
        connection = self._connection()
        try:
            rows = connection.execute(
                """
                SELECT optimizer_update, probe_id, active_layers, selected_layers, payload
                FROM heatmap_records
                ORDER BY optimizer_update
                """
            ).fetchall()
        finally:
            connection.close()
        decoded = []
        maximum_layers = 1
        payloads = []
        for row in rows:
            payload = _decode_payload(row["payload"])
            candidates = tuple(int(value) for value in payload["candidate_layers"])
            deltas = tuple(float(value) for value in payload["delta_losses"])
            if candidates:
                maximum_layers = max(maximum_layers, max(candidates))
            payloads.append((row, candidates, deltas))
        for row, candidates, deltas in payloads:
            values = [math.nan] * maximum_layers
            for candidate, delta in zip(candidates, deltas):
                values[candidate - 1] = delta
            decoded.append(
                {
                    "probe_id": str(row["probe_id"]),
                    "optimizer_update": int(row["optimizer_update"]),
                    "active_layers": int(row["active_layers"]),
                    "selected_layers": int(row["selected_layers"]),
                    "values": values,
                }
            )
        return tuple(decoded)

    def maximum_heatmap_layer(self) -> int:
        history = self.heatmap_history()
        return max((len(record["values"]) for record in history), default=1)

    def depth_weight_snapshots(self) -> Tuple[Dict[str, Any], ...]:
        connection = self._connection()
        try:
            rows = connection.execute(
                """
                SELECT payload
                FROM depth_weight_snapshots
                ORDER BY optimizer_update
                """
            ).fetchall()
        finally:
            connection.close()
        return tuple(_decode_payload(row["payload"]) for row in rows)


def ensure_local_chart_store(telemetry: Any) -> LocalChartStore:
    existing = getattr(telemetry, "_thog_local_chart_store", None)
    if existing is not None:
        return existing
    path = local_chart_database_path(str(telemetry.name))
    store = LocalChartStore(
        path,
        run_name=str(telemetry.name),
        config=getattr(telemetry, "config", {}),
    )
    setattr(telemetry, "_thog_local_chart_store", store)
    if not bool(getattr(telemetry, "_thog_local_chart_store_announced", False)):
        print(
            f"THOG2 local chart data: {path.resolve()}",
            flush=True,
        )
        print(
            "THOG2 local chart viewer: "
            f"python -m run_thog2_local_dashboard --run {telemetry.name}",
            flush=True,
        )
        setattr(telemetry, "_thog_local_chart_store_announced", True)
    return store


def close_local_chart_store(telemetry: Any) -> None:
    store = getattr(telemetry, "_thog_local_chart_store", None)
    if store is None:
        return
    store.close()
    setattr(telemetry, "_thog_local_chart_store", None)


__all__ = [
    "CHART_DESTINATIONS",
    "LOCAL_CHART_DATABASE_NAME",
    "LocalChartReader",
    "LocalChartStore",
    "close_local_chart_store",
    "ensure_local_chart_store",
    "local_chart_database_path",
    "normalize_chart_destination",
]
# ^^^ THOG
