# vvv THOG
"""Read only the heatmap probe window INSTRA is actually going to render."""

from __future__ import annotations

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

    return _store._decode_heatmap_rows(rows)


def install() -> None:
    if getattr(_store.LocalChartReader, "_thog2_heatmap_window_patch_installed", False):
        return
    _store.LocalChartReader._thog2_heatmap_window_patch_installed = True
    _store.LocalChartReader.heatmap_history_window = _heatmap_history_window


__all__ = ["install"]
# ^^^ THOG
