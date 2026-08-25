# vvv THOG
"""Compact concurrent local storage for live THOG chart data."""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
import uuid
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


CHART_DESTINATIONS = ("wandb", "local", "none")
LOCAL_CHART_DATABASE_NAME = "charts.sqlite3"
LOCAL_CHART_SCHEMA_VERSION = 2
LOCAL_CHART_ACTIVE_STATES = frozenset(("preparing", "recording", "monitoring", "running"))
LOCAL_CHART_TERMINAL_STATES = frozenset(("finished", "stopped", "crashed"))


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _safe_path_component(value: Any, *, label: str) -> str:
    normalized = Path(str(value)).name.strip()
    if normalized in {"", ".", ".."}:
        raise ValueError(f"invalid local chart {label}: {value!r}")
    return normalized


def local_chart_database_path(
    run_name: str,
    *,
    run_id: Optional[str] = None,
    root: Optional[Path] = None,
) -> Path:
    normalized_name = _safe_path_component(run_name, label="run name")
    resolved_root = local_chart_root() if root is None else Path(root)
    if run_id is not None:
        normalized_id = _safe_path_component(run_id, label="run ID")
        return resolved_root / normalized_name / normalized_id / LOCAL_CHART_DATABASE_NAME
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
        run_id: Optional[str] = None,
        wandb_run_id: Optional[str] = None,
        wandb_url: Optional[str] = None,
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
        now = _utc_timestamp()
        metadata = {
            "schema_version": str(LOCAL_CHART_SCHEMA_VERSION),
            "run_name": str(run_name),
            "artifact_name": str(run_name),
            "local_run_id": str(run_id or run_name),
            "wandb_run_id": str(wandb_run_id or ""),
            "wandb_url": str(wandb_url or ""),
            "run_state": "preparing",
            "current_update": "0",
            "heartbeat_at": now,
            "data_updated_at": now,
            "updated_at": now,
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
        self.connection.execute(
            "INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)",
            ("created_at", _utc_timestamp()),
        )
        self.connection.commit()
        self._last_heartbeat_monotonic = 0.0
        self._last_heartbeat_update = 0
        self._last_heartbeat_state = "preparing"
        self._has_heatmap_records = bool(
            self.connection.execute("SELECT EXISTS(SELECT 1 FROM heatmap_records)").fetchone()[0]
        )
        self._has_depth_records = bool(
            self.connection.execute("SELECT EXISTS(SELECT 1 FROM depth_weight_snapshots)").fetchone()[0]
        )
        self._has_recorded_data = self._has_heatmap_records or self._has_depth_records

    def _touch(self) -> None:
        now = _utc_timestamp()
        self.connection.executemany(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            (
                ("run_state", "recording"),
                ("data_updated_at", now),
                ("updated_at", now),
            ),
        )
        self._last_heartbeat_state = "recording"
        self._has_recorded_data = True

    def configure_weight_capture(
        self,
        *,
        start_step: Optional[int],
        end_step: Optional[int],
        cadence: int,
        history_length: int,
    ) -> None:
        values = {
            "weight_capture_expected": "true",
            "weight_capture_start_step": "" if start_step is None else str(int(start_step)),
            "weight_capture_end_step": "" if end_step is None else str(int(end_step)),
            "weight_capture_cadence": str(int(cadence)),
            "weight_capture_history_length": str(int(history_length)),
        }
        self.connection.executemany(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            tuple(values.items()),
        )
        self.connection.commit()

    def heartbeat(
        self,
        optimizer_update: int,
        *,
        run_state: str,
        force: bool = False,
    ) -> None:
        state = str(run_state)
        if state not in LOCAL_CHART_ACTIVE_STATES:
            raise ValueError(f"invalid active local chart state: {state!r}")
        update = max(0, int(optimizer_update))
        monotonic_now = time.monotonic()
        phase_changed = state != self._last_heartbeat_state
        if (
            not force
            and not phase_changed
            and monotonic_now - self._last_heartbeat_monotonic < 1.0
        ):
            return
        now = _utc_timestamp()
        self.connection.executemany(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            (
                ("run_state", state),
                ("current_update", str(update)),
                ("heartbeat_at", now),
            ),
        )
        self.connection.commit()
        self._last_heartbeat_monotonic = monotonic_now
        self._last_heartbeat_update = update
        self._last_heartbeat_state = state

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
        self._touch()
        self._has_heatmap_records = True
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
        self._touch()
        self._has_depth_records = True
        self.connection.commit()

    def close(self, *, final_state: str = "finished") -> None:
        if self.connection is None:
            return
        state = str(final_state)
        if state not in LOCAL_CHART_TERMINAL_STATES:
            raise ValueError(f"invalid terminal local chart state: {state!r}")
        now = _utc_timestamp()
        self.connection.executemany(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            (
                ("run_state", state),
                ("heartbeat_at", now),
                ("updated_at", now),
            ),
        )
        self.connection.commit()
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
            # vvv THOG expose the already-recorded full run configuration to local dashboard consumers without duplicating storage
            config_row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'config_json'"
            ).fetchone()
            configuration = (
                json.loads(str(config_row["value"]))
                if config_row is not None
                else {}
            )
            # ^^^ THOG
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
            # vvv THOG consumed by the W&B-like local artifact Overview tab
            "configuration": configuration,
            # ^^^ THOG
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
    run = getattr(telemetry, "run", None)
    wandb_run_id = str(getattr(run, "id", "") or "").strip()
    local_run_id = wandb_run_id or str(
        getattr(telemetry, "_thog_local_chart_run_id", "") or ""
    ).strip()
    if not local_run_id:
        local_run_id = f"local-{uuid.uuid4().hex[:8]}"
        setattr(telemetry, "_thog_local_chart_run_id", local_run_id)
    path = local_chart_database_path(
        str(telemetry.name),
        run_id=local_run_id,
    )
    store = LocalChartStore(
        path,
        run_name=str(telemetry.name),
        run_id=local_run_id,
        wandb_run_id=wandb_run_id or None,
        wandb_url=str(getattr(run, "url", "") or "") or None,
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
            f"python -m run_thog2_local_dashboard --run {local_run_id}",
            flush=True,
        )
        setattr(telemetry, "_thog_local_chart_store_announced", True)
    return store


def close_local_chart_store(
    telemetry: Any,
    *,
    final_state: str = "finished",
) -> None:
    store = getattr(telemetry, "_thog_local_chart_store", None)
    if store is None:
        return
    store.close(final_state=final_state)
    setattr(telemetry, "_thog_local_chart_store", None)


__all__ = [
    "CHART_DESTINATIONS",
    "LOCAL_CHART_ACTIVE_STATES",
    "LOCAL_CHART_DATABASE_NAME",
    "LOCAL_CHART_TERMINAL_STATES",
    "LocalChartReader",
    "LocalChartStore",
    "close_local_chart_store",
    "ensure_local_chart_store",
    "local_chart_database_path",
    "normalize_chart_destination",
]
# ^^^ THOG
