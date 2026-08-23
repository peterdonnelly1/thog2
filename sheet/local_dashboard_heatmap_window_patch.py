# vvv THOG
"""Read only the heatmap probe window INSTRA is actually going to render."""

from __future__ import annotations

import math
from typing import Any

from . import local_chart_store as _store


def _heatmap_history_window(
    self: Any,
    *,
    probe_count: int,
    window_mode: str,
) -> tuple[dict[str, Any], ...]:
    resolved_count = max(1, min(512, int(probe_count)))
    resolved_window = str(window_mode).strip().lower()
    if resolved_window not in {"from_zero", "rolling"}:
        raise ValueError("heatmap window_mode must be from_zero or rolling")

    connection = _store._open_database(self.path, readonly=True)
    try:
        if resolved_window == "from_zero":
            rows = connection.execute(
                """
                SELECT optimizer_update, probe_id, active_layers, selected_layers, payload
                FROM heatmap_records
                ORDER BY optimizer_update ASC
                LIMIT ?
                """,
                (resolved_count,),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT optimizer_update, probe_id, active_layers, selected_layers, payload
                FROM heatmap_records
                ORDER BY optimizer_update DESC
                LIMIT ?
                """,
                (resolved_count,),
            ).fetchall()
            rows = list(reversed(rows))
    finally:
        connection.close()

    decoded_payloads: list[tuple[Any, dict[str, Any], tuple[int, ...], tuple[float, ...]]] = []
    maximum_layers = 1
    for row in rows:
        payload = _store._decode_payload(row["payload"])
        candidates = tuple(int(value) for value in payload["candidate_layers"])
        deltas = tuple(float(value) for value in payload["delta_losses"])
        if candidates:
            maximum_layers = max(maximum_layers, max(candidates))
        decoded_payloads.append((row, payload, candidates, deltas))

    decoded: list[dict[str, Any]] = []
    for row, payload, candidates, deltas in decoded_payloads:
        values = [math.nan] * maximum_layers
        for candidate, delta in zip(candidates, deltas):
            values[candidate - 1] = delta
        current_loss = payload.get("current_loss")
        decoded.append(
            {
                "probe_id": str(row["probe_id"]),
                "optimizer_update": int(row["optimizer_update"]),
                "active_layers": int(row["active_layers"]),
                "selected_layers": int(row["selected_layers"]),
                "brake_active": bool(payload.get("brake_active", False)),
                "decision_committed": bool(
                    payload.get(
                        "decision_committed",
                        int(row["selected_layers"]) != int(row["active_layers"]),
                    )
                ),
                "chaos_bump": payload.get("chaos_bump"),
                "current_loss": (
                    None if current_loss is None else float(current_loss)
                ),
                "values": values,
            }
        )
    return tuple(decoded)


def install() -> None:
    if getattr(_store.LocalChartReader, "_thog2_heatmap_window_patch_installed", False):
        return
    _store.LocalChartReader._thog2_heatmap_window_patch_installed = True
    _store.LocalChartReader.heatmap_history_window = _heatmap_history_window


__all__ = ["install"]
# ^^^ THOG
