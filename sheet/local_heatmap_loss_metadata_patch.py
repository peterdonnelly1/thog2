# vvv THOG
"""Retain the absolute centre/L probe loss needed by local heatmap relative display modes."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from . import local_chart_store as _local_store
from . import plastic_depth_wandb_probe_curves_patch as _probe_curves


_ORIGINAL_PROBE_RECORD_FROM_EVENT = _probe_curves._probe_record_from_event
_ORIGINAL_HEATMAP_RENDER_DATA = _probe_curves._delta_loss_heatmap_render_data
_ORIGINAL_HEATMAP_FIGURE = _probe_curves._delta_loss_heatmap_figure


def _probe_record_from_event_with_current_loss(event: Any) -> Optional[Dict[str, Any]]:
    record = _ORIGINAL_PROBE_RECORD_FROM_EVENT(event)
    if record is None:
        return None
    payload = getattr(event, "payload", None)
    if not isinstance(payload, Mapping):
        return record
    current = int(record["active_layers"])
    for candidate in payload.get("candidates", ()):
        if not isinstance(candidate, Mapping):
            continue
        try:
            if int(candidate["active_layers"]) != current:
                continue
            current_loss = float(candidate["validation_loss"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(current_loss):
            return {**record, "current_loss": current_loss}
    return record


def _append_heatmap_records_with_current_loss(
    self: Any,
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
        current_loss = record.get("current_loss")
        resolved_current_loss = (
            float(current_loss)
            if current_loss is not None and math.isfinite(float(current_loss))
            else None
        )
        payload = {
            "candidate_layers": sorted(point_by_candidate),
            "delta_losses": [
                point_by_candidate[candidate]
                for candidate in sorted(point_by_candidate)
            ],
            "current_loss": resolved_current_loss,
        }
        rows.append(
            (
                optimizer_update,
                str(record["probe_id"]),
                int(record["active_layers"]),
                int(record.get("selected_layers", record["active_layers"])),
                _local_store._encode_payload(payload),
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
    self.connection.commit()
    return len(rows)


def _heatmap_history_with_current_loss(self: Any) -> Tuple[Dict[str, Any], ...]:
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
        payload = _local_store._decode_payload(row["payload"])
        candidates = tuple(int(value) for value in payload["candidate_layers"])
        deltas = tuple(float(value) for value in payload["delta_losses"])
        if candidates:
            maximum_layers = max(maximum_layers, max(candidates))
        payloads.append((row, payload, candidates, deltas))
    for row, payload, candidates, deltas in payloads:
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
                "current_loss": (
                    None if current_loss is None else float(current_loss)
                ),
                "values": values,
            }
        )
    return tuple(decoded)


def _heatmap_render_data_with_current_loss(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    rendered = _ORIGINAL_HEATMAP_RENDER_DATA(*args, **kwargs)
    history = args[0] if args else kwargs["history"]
    rendered_row_limit = int(
        kwargs.get(
            "rendered_row_limit",
            _probe_curves._DELTA_LOSS_HEATMAP_MAX_RENDERED_ROWS,
        )
    )
    indices = _probe_curves._evenly_spaced_record_indices(
        len(history),
        rendered_row_limit,
    )
    rendered["current_losses"] = tuple(
        None
        if history[index].get("current_loss") is None
        else float(history[index]["current_loss"])
        for index in indices
    )
    return rendered


def _heatmap_figure_with_current_loss(
    history: Any,
    *,
    maximum_layers: int,
    abs_limit: float,
    go_module: Any = None,
) -> Any:
    figure = _ORIGINAL_HEATMAP_FIGURE(
        history,
        maximum_layers=maximum_layers,
        abs_limit=abs_limit,
        go_module=go_module,
    )
    rendered = _heatmap_render_data_with_current_loss(
        history,
        maximum_layers=maximum_layers,
    )
    figure.update_layout(
        meta={
            **(dict(figure.layout.meta) if isinstance(figure.layout.meta, Mapping) else {}),
            "thog2_current_losses": list(rendered["current_losses"]),
        }
    )
    return figure


_probe_curves._probe_record_from_event = _probe_record_from_event_with_current_loss
_local_store.LocalChartStore.append_heatmap_records = _append_heatmap_records_with_current_loss
_local_store.LocalChartReader.heatmap_history = _heatmap_history_with_current_loss
_probe_curves._delta_loss_heatmap_render_data = _heatmap_render_data_with_current_loss
_probe_curves._delta_loss_heatmap_figure = _heatmap_figure_with_current_loss
# ^^^ THOG
