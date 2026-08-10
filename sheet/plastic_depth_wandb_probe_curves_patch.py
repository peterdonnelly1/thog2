# vvv THOG
"""W&B-only rolling PLASTIC probe-loss spaghetti curves over the most recent probes."""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import wandb_telemetry as _wandb


_WINDOW_PROBES = 300
_REFRESH_EVERY_PROBES = 25
_EARLY_REFRESH_PROBES = 10
_MAX_TABLE_ROWS = 9_999
_CHART_COLUMNS = (
    "distance",
    "delta_loss",
    "probe_id",
    "optimizer_update",
    "active_layers",
    "selected_layers",
    "candidate_layers",
    "offset",
)


def _probe_record_from_event(event: Any) -> Optional[Dict[str, Any]]:
    if getattr(event, "name", None) != "plastic_depth_count_decision":
        return None
    payload = getattr(event, "payload", None)
    if not isinstance(payload, Mapping):
        return None
    try:
        current = int(payload["previous_active_layers"])
        selected = int(payload["selected_active_layers"])
    except (KeyError, TypeError, ValueError):
        return None

    losses: Dict[int, float] = {}
    for item in payload.get("candidates", ()):
        if not isinstance(item, Mapping):
            continue
        try:
            count = int(item["active_layers"])
            loss = float(item["validation_loss"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(loss):
            losses[count] = loss
    current_loss = losses.get(current)
    if current_loss is None:
        return None

    update = int(getattr(event, "completed_updates", 0)) + 1
    probe_id = f"U{update}"
    shrink = [(0, 0.0, current, 0)]
    growth = [(0, 0.0, current, 0)]
    for count in sorted(losses):
        offset = count - current
        if offset == 0:
            continue
        delta = float(losses[count] - current_loss)
        point = (abs(offset), delta, count, offset)
        if offset < 0:
            shrink.append(point)
        else:
            growth.append(point)
    return {
        "probe_id": probe_id,
        "optimizer_update": update,
        "active_layers": current,
        "selected_layers": selected,
        "shrink": tuple(shrink),
        "growth": tuple(growth),
    }


def _ensure_curve_state(telemetry: Any) -> Deque[Dict[str, Any]]:
    history = getattr(telemetry, "_plastic_probe_curve_history", None)
    if history is None:
        history = deque(maxlen=_WINDOW_PROBES)
        setattr(telemetry, "_plastic_probe_curve_history", history)
        setattr(telemetry, "_plastic_probe_curve_last_event_index", -1)
        setattr(telemetry, "_plastic_probe_curve_total", 0)
    return history


def _consume_new_probe_records(trainer: Any, telemetry: Any) -> Tuple[Dict[str, Any], ...]:
    history = _ensure_curve_state(telemetry)
    events = tuple(getattr(trainer, "events", ()))
    start = int(getattr(telemetry, "_plastic_probe_curve_last_event_index", -1)) + 1
    records: List[Dict[str, Any]] = []
    for event in events[start:]:
        record = _probe_record_from_event(event)
        if record is None:
            continue
        history.append(record)
        records.append(record)
        telemetry._plastic_probe_curve_total = int(telemetry._plastic_probe_curve_total) + 1
    telemetry._plastic_probe_curve_last_event_index = len(events) - 1
    return tuple(records)


def _rows_for_side(history: Iterable[Mapping[str, Any]], side: str) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for record in history:
        probe_id = str(record["probe_id"])
        update = int(record["optimizer_update"])
        active = int(record["active_layers"])
        selected = int(record["selected_layers"])
        for distance, delta, candidate, offset in record.get(side, ()):
            rows.append(
                [
                    int(distance),
                    float(delta),
                    probe_id,
                    update,
                    active,
                    selected,
                    int(candidate),
                    int(offset),
                ]
            )
    return rows


def _bounded_rows_for_side(
    history: Iterable[Mapping[str, Any]],
    side: str,
) -> List[List[Any]]:
    records = tuple(history)
    groups: List[List[List[Any]]] = []
    row_count = 0
    for record in reversed(records):
        group = _rows_for_side((record,), side)
        if not group:
            continue
        if len(group) > _MAX_TABLE_ROWS:
            group = group[-_MAX_TABLE_ROWS:]
        if row_count + len(group) > _MAX_TABLE_ROWS:
            break
        groups.append(group)
        row_count += len(group)
    rows: List[List[Any]] = []
    for group in reversed(groups):
        rows.extend(group)
    return rows


def _should_refresh_charts(
    telemetry: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    evaluation: bool,
) -> bool:
    history = _ensure_curve_state(telemetry)
    if not history:
        return False
    if evaluation:
        return True
    if any(int(row["selected_layers"]) != int(row["active_layers"]) for row in records):
        return True
    total = int(getattr(telemetry, "_plastic_probe_curve_total", 0))
    return bool(records) and (total <= _EARLY_REFRESH_PROBES or total % _REFRESH_EVERY_PROBES == 0)


def _log_rolling_probe_charts(telemetry: Any, *, step: int) -> None:
    if telemetry.run is None or telemetry.module is None:
        return
    history = tuple(_ensure_curve_state(telemetry))
    if not history:
        return

    payload: Dict[str, Any] = {}
    for side, title, key in (
        (
            "shrink",
            f"PLASTIC probe shrink/interpolation side — recent {len(history)} probes",
            "fine/plastic_probe_shrink_curves",
        ),
        (
            "growth",
            f"PLASTIC probe growth/extrapolation side — recent {len(history)} probes",
            "fine/plastic_probe_growth_curves",
        ),
    ):
        rows = _bounded_rows_for_side(history, side)
        if not rows:
            continue
        table = telemetry.module.Table(data=rows, columns=list(_CHART_COLUMNS))
        payload[key] = telemetry.module.plot.line(
            table=table,
            x="distance",
            y="delta_loss",
            stroke="probe_id",
            title=title,
        )
    if not payload:
        return
    try:
        telemetry.run.log(payload, step=int(step))
    except TypeError:
        telemetry.run.log(payload)


# vvv THOG attach the rolling charts after the established scalar telemetry wrapper; TensorBoard receives no tables, plots or figures from this path
_ORIGINAL_ATTACH_TELEMETRY = _wandb.attach_telemetry


def attach_telemetry_with_plastic_probe_curves(trainer: Any, telemetry: Any) -> None:
    _ORIGINAL_ATTACH_TELEMETRY(trainer, telemetry)
    original_progress = trainer._print_progress

    def progress(run_id: str, event: str, **payload: Any) -> None:
        original_progress(run_id, event, **payload)
        if not trainer.distributed.is_primary:
            return
        if telemetry.run is None or telemetry.module is None:
            return
        if not bool(getattr(trainer.config, "plastic__enabled", False)):
            return
        if event not in {"optimizer_progress", "evaluation_completed"}:
            return
        records = _consume_new_probe_records(trainer, telemetry)
        if not _should_refresh_charts(
            telemetry,
            records,
            evaluation=event == "evaluation_completed",
        ):
            return
        try:
            step = int(str(payload.get("completed_updates", trainer.state.completed_updates)).strip().replace(",", ""))
            _log_rolling_probe_charts(telemetry, step=step)
        except Exception as error:
            print(
                "THOG2 WARNING: W&B PLASTIC probe-curve logging failed; "
                f"continuing without refreshed probe curves: {error}",
                flush=True,
            )

    trainer._print_progress = progress


_wandb.attach_telemetry = attach_telemetry_with_plastic_probe_curves
# ^^^ THOG


__all__ = [
    "_bounded_rows_for_side",
    "_consume_new_probe_records",
    "_log_rolling_probe_charts",
    "_probe_record_from_event",
    "_rows_for_side",
    "attach_telemetry_with_plastic_probe_curves",
]
# ^^^ THOG
