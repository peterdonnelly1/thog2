# vvv THOG
"""W&B-only rolling PLASTIC probe-loss and sampled-coefficient curves."""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from . import wandb_telemetry as _wandb
from .semantic_materializer import ATTENTION_QUERY_WEIGHT


_WINDOW_PROBES = 300
_WINDOW_STEPS = 300
_REFRESH_EVERY_PROBES = 25
_EARLY_REFRESH_PROBES = 10
_REFRESH_EVERY_STEPS = 25
_EARLY_REFRESH_STEPS = 10
_MAX_TABLE_ROWS = 9_999
_ZERO_LOSS_REFERENCE_ID = "Δloss = 0 reference"
_ZERO_LOSS_REFERENCE_MAX_ROWS = 2
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
_COEFFICIENT_CHART_COLUMNS = (
    "capacity_layer_sample_number",
    "sampled_coefficient",
    "step_id",
    "optimizer_update",
    "active_layers",
    "maximum_layers",
)


# vvv THOG PLASTIC W&B history charts have their own visibility threshold; the broader forensic scalar surface remains DEBUG>9
def _plastic_wandb_charts_enabled() -> bool:
    return int(getattr(_wandb._constants, "DEBUG", 0)) > 2
# ^^^ THOG


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


# vvv THOG retain one fixed generated attention-query coefficient sample for every successful optimiser step
def _ensure_coefficient_curve_state(telemetry: Any) -> Deque[Dict[str, Any]]:
    history = getattr(telemetry, "_plastic_coefficient_curve_history", None)
    if history is None:
        history = deque(maxlen=_WINDOW_STEPS)
        setattr(telemetry, "_plastic_coefficient_curve_history", history)
        setattr(telemetry, "_plastic_coefficient_curve_total", 0)
        setattr(telemetry, "_plastic_coefficient_curve_last_logged_total", 0)
    return history


@torch.no_grad()
def _coefficient_record_from_trainer(
    trainer: Any,
    *,
    optimizer_update: int,
) -> Dict[str, Any]:
    lattice = trainer._plastic_depth_lattice()
    if lattice is None:
        raise RuntimeError("PLASTIC DEPTH lattice unexpectedly absent while sampling W&B coefficients")
    active_layers = int(lattice.current_active_layers)
    maximum_layers = int(lattice.maximum_layers)
    report = lattice.interval_report()
    capacity_layer_sample_numbers = tuple(
        float(value)
        for value in report["active_sample_layer_coordinates"]
    )
    if len(capacity_layer_sample_numbers) != active_layers:
        raise RuntimeError(
            "PLASTIC capacity-layer sample coordinates do not match the active layer count"
        )
    sampled = torch.stack(
        tuple(
            trainer.raw_model.semantic_materializer.direct_matrix_value(
                ATTENTION_QUERY_WEIGHT,
                layer_index,
                0,
                0,
            ).detach()
            for layer_index in range(active_layers)
        )
    ).to(device="cpu", dtype=torch.float64)
    coefficients = tuple(float(value) for value in sampled.tolist())
    if not all(math.isfinite(value) for value in coefficients):
        raise RuntimeError("PLASTIC sampled coefficients contain a non-finite value")
    return {
        "step_id": f"U{int(optimizer_update)}",
        "optimizer_update": int(optimizer_update),
        "active_layers": active_layers,
        "maximum_layers": maximum_layers,
        "capacity_layer_sample_numbers": capacity_layer_sample_numbers,
        "coefficients": coefficients,
    }


def _capture_coefficient_record(
    trainer: Any,
    telemetry: Any,
    *,
    optimizer_update: int,
) -> Dict[str, Any]:
    history = _ensure_coefficient_curve_state(telemetry)
    record = _coefficient_record_from_trainer(
        trainer,
        optimizer_update=int(optimizer_update),
    )
    history.append(record)
    telemetry._plastic_coefficient_curve_total = (
        int(telemetry._plastic_coefficient_curve_total) + 1
    )
    return record
# ^^^ THOG


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


# vvv THOG concatenate the established interpolation/extrapolation probe sides into one signed-offset landscape while retaining L exactly once
def _rows_for_combined(history: Iterable[Mapping[str, Any]]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for record in history:
        probe_id = str(record["probe_id"])
        update = int(record["optimizer_update"])
        active = int(record["active_layers"])
        selected = int(record["selected_layers"])
        point_by_offset: Dict[int, Tuple[int, float, int, int]] = {}
        for side in ("shrink", "growth"):
            for distance, delta, candidate, offset in record.get(side, ()):
                point_by_offset[int(offset)] = (
                    int(distance),
                    float(delta),
                    int(candidate),
                    int(offset),
                )
        for offset in sorted(point_by_offset):
            distance, delta, candidate, signed_offset = point_by_offset[offset]
            rows.append(
                [
                    distance,
                    delta,
                    probe_id,
                    update,
                    active,
                    selected,
                    candidate,
                    signed_offset,
                ]
            )
    return rows
# ^^^ THOG


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
        data_row_limit = _MAX_TABLE_ROWS - _ZERO_LOSS_REFERENCE_MAX_ROWS
        if len(group) > data_row_limit:
            group = group[-data_row_limit:]
        if row_count + len(group) > data_row_limit:
            break
        groups.append(group)
        row_count += len(group)
    rows: List[List[Any]] = []
    for group in reversed(groups):
        rows.extend(group)
    return rows


# vvv THOG give the wider combined landscape its own whole-probe row cap so very wide probes reduce only this panel's retained probe count

def _bounded_rows_for_combined(history: Iterable[Mapping[str, Any]]) -> List[List[Any]]:
    records = tuple(history)
    groups: List[List[List[Any]]] = []
    row_count = 0
    for record in reversed(records):
        group = _rows_for_combined((record,))
        if not group:
            continue
        data_row_limit = _MAX_TABLE_ROWS - _ZERO_LOSS_REFERENCE_MAX_ROWS
        if len(group) > data_row_limit:
            group = group[-data_row_limit:]
        if row_count + len(group) > data_row_limit:
            break
        groups.append(group)
        row_count += len(group)
    rows: List[List[Any]] = []
    for group in reversed(groups):
        rows.extend(group)
    return rows
# ^^^ THOG


# vvv THOG cap the coefficient chart by dropping oldest complete step-curves, never individual capacity-layer samples
def _rows_for_coefficients(history: Iterable[Mapping[str, Any]]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for record in history:
        step_id = str(record["step_id"])
        update = int(record["optimizer_update"])
        active = int(record["active_layers"])
        maximum = int(record["maximum_layers"])
        for layer_sample_number, coefficient in zip(
            record.get("capacity_layer_sample_numbers", ()),
            record.get("coefficients", ()),
        ):
            rows.append(
                [
                    float(layer_sample_number),
                    float(coefficient),
                    step_id,
                    update,
                    active,
                    maximum,
                ]
            )
    return rows


def _bounded_rows_for_coefficients(
    history: Iterable[Mapping[str, Any]],
) -> List[List[Any]]:
    groups: List[List[List[Any]]] = []
    row_count = 0
    for record in reversed(tuple(history)):
        group = _rows_for_coefficients((record,))
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
# ^^^ THOG


# vvv THOG make the zero-delta x axis unmistakable in W&B's fixed built-in line preset by logging it as an explicit full-width reference series
def _rows_with_zero_loss_reference(
    rows: Sequence[Sequence[Any]],
    *,
    x: str,
) -> List[List[Any]]:
    rendered = [list(row) for row in rows]
    if not rendered:
        return rendered
    x_index = _CHART_COLUMNS.index(str(x))
    coordinates = sorted({float(row[x_index]) for row in rendered})
    endpoints = (coordinates[0],) if coordinates[0] == coordinates[-1] else (
        coordinates[0],
        coordinates[-1],
    )
    template = list(rendered[-1])
    active_layers = int(template[4])
    for coordinate in endpoints:
        reference = list(template)
        reference[0] = abs(coordinate)
        reference[1] = 0.0
        reference[2] = _ZERO_LOSS_REFERENCE_ID
        reference[6] = active_layers + int(coordinate) if x == "offset" else active_layers
        reference[7] = int(coordinate) if x == "offset" else 0
        rendered.append(reference)
    return rendered
# ^^^ THOG


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


def _should_refresh_coefficient_chart(
    telemetry: Any,
    *,
    evaluation: bool,
) -> bool:
    history = _ensure_coefficient_curve_state(telemetry)
    if not history:
        return False
    total = int(getattr(telemetry, "_plastic_coefficient_curve_total", 0))
    last_logged = int(
        getattr(telemetry, "_plastic_coefficient_curve_last_logged_total", 0)
    )
    if evaluation:
        return total > 0
    if total <= _EARLY_REFRESH_STEPS:
        return total > last_logged
    return total - last_logged >= _REFRESH_EVERY_STEPS


def _log_rolling_probe_charts(
    telemetry: Any,
    *,
    step: int,
    include_probe_charts: bool = True,
    include_coefficient_chart: bool = True,
) -> None:
    if not _plastic_wandb_charts_enabled():
        return
    if telemetry.run is None or telemetry.module is None:
        return

    payload: Dict[str, Any] = {}
    history = tuple(_ensure_curve_state(telemetry))
    if include_probe_charts:
        for side, title, key in (
            (
                "shrink",
                f"PLASTIC probe shrink/interpolation side — recent {len(history)} probes",
                "plastic/probe_shrink_curves",
            ),
            (
                "growth",
                f"PLASTIC probe grow/extrapolation side — recent {len(history)} probes",
                "plastic/probe_growth_curves",
            ),
        ):
            rows = _rows_with_zero_loss_reference(
                _bounded_rows_for_side(history, side),
                x="distance",
            )
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

    # vvv THOG add a third objective-neutral visual joining both existing sides around signed offset zero without changing either established split chart
    if include_probe_charts:
        combined_rows = _rows_with_zero_loss_reference(
            _bounded_rows_for_combined(history),
            x="offset",
        )
        if combined_rows:
            combined_table = telemetry.module.Table(
                data=combined_rows,
                columns=list(_CHART_COLUMNS),
            )
            payload["plastic/probe_combined_curves"] = telemetry.module.plot.line(
                table=combined_table,
                x="offset",
                y="delta_loss",
                stroke="probe_id",
                title=(
                    "PLASTIC probe shrink/interpolation : grow/extrapolation — "
                    f"recent {len(history)} probes"
                ),
            )
    # ^^^ THOG

    # vvv THOG show one recent-step curve per fixed generated coefficient sample over stable maximum-capacity layer identities
    if include_coefficient_chart:
        coefficient_rows = _bounded_rows_for_coefficients(
            tuple(_ensure_coefficient_curve_state(telemetry))
        )
        if coefficient_rows:
            coefficient_steps = len({str(row[2]) for row in coefficient_rows})
            coefficient_table = telemetry.module.Table(
                data=coefficient_rows,
                columns=list(_COEFFICIENT_CHART_COLUMNS),
            )
            payload["plastic/sampled_coefficients_curves"] = telemetry.module.plot.line(
                table=coefficient_table,
                x="capacity_layer_sample_number",
                y="sampled_coefficient",
                stroke="step_id",
                title=(
                    "PLASTIC sampled coefficients by capacity layer sample number — "
                    f"recent {coefficient_steps} steps"
                ),
            )
    # ^^^ THOG

    if not payload:
        return
    try:
        telemetry.run.log(payload, step=int(step))
    except TypeError:
        telemetry.run.log(payload)
    if include_coefficient_chart and "plastic/sampled_coefficients_curves" in payload:
        telemetry._plastic_coefficient_curve_last_logged_total = int(
            getattr(telemetry, "_plastic_coefficient_curve_total", 0)
        )


# vvv THOG attach the rolling charts after the established scalar telemetry wrapper; TensorBoard receives no tables, plots or figures from this path
_ORIGINAL_ATTACH_TELEMETRY = _wandb.attach_telemetry


def attach_telemetry_with_plastic_probe_curves(trainer: Any, telemetry: Any) -> None:
    _ORIGINAL_ATTACH_TELEMETRY(trainer, telemetry)
    # vvv THOG DEBUG<=2 never installs event scanning, per-step coefficient sampling or W&B chart construction
    if not _plastic_wandb_charts_enabled():
        return
    # ^^^ THOG
    original_timed = trainer._timed
    train_one_update = trainer.train_one_update
    original_progress = trainer._print_progress

    # vvv THOG capture every successful optimiser step independently of the console/log interval and outside the clean optimiser-update timer
    def timed(function: Any):
        metrics, elapsed = original_timed(function)
        if function != train_one_update:
            return metrics, elapsed
        if not trainer.distributed.is_primary:
            return metrics, elapsed
        if telemetry.run is None or telemetry.module is None:
            return metrics, elapsed
        if not bool(getattr(trainer.config, "plastic__enabled", False)):
            return metrics, elapsed
        if bool(float(metrics.get("skipped_update", 0.0))):
            return metrics, elapsed
        if bool(getattr(telemetry, "_plastic_coefficient_curve_disabled", False)):
            return metrics, elapsed
        try:
            _capture_coefficient_record(
                trainer,
                telemetry,
                optimizer_update=int(trainer.state.completed_updates),
            )
        except Exception as error:
            telemetry._plastic_coefficient_curve_disabled = True
            print(
                "THOG2 WARNING: W&B PLASTIC coefficient sampling failed; "
                f"continuing without coefficient curves: {error}",
                flush=True,
            )
        return metrics, elapsed

    trainer._timed = timed
    # ^^^ THOG

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
        evaluation = event == "evaluation_completed"
        probe_charts_due = _should_refresh_charts(
            telemetry,
            records,
            evaluation=evaluation,
        )
        coefficient_chart_due = _should_refresh_coefficient_chart(
            telemetry,
            evaluation=evaluation,
        )
        if not probe_charts_due and not coefficient_chart_due:
            return
        try:
            step = int(str(payload.get("completed_updates", trainer.state.completed_updates)).strip().replace(",", ""))
            _log_rolling_probe_charts(
                telemetry,
                step=step,
                include_probe_charts=probe_charts_due,
                include_coefficient_chart=coefficient_chart_due,
            )
        except Exception as error:
            print(
                "THOG2 WARNING: W&B PLASTIC chart logging failed; "
                f"continuing without refreshed PLASTIC charts: {error}",
                flush=True,
            )

    trainer._print_progress = progress


_wandb.attach_telemetry = attach_telemetry_with_plastic_probe_curves
# ^^^ THOG


__all__ = [
    "_bounded_rows_for_coefficients",
    "_bounded_rows_for_combined",
    "_bounded_rows_for_side",
    "_capture_coefficient_record",
    "_coefficient_record_from_trainer",
    "_consume_new_probe_records",
    "_ensure_coefficient_curve_state",
    "_log_rolling_probe_charts",
    "_plastic_wandb_charts_enabled",
    "_probe_record_from_event",
    "_rows_for_coefficients",
    "_rows_for_combined",
    "_rows_for_side",
    "_rows_with_zero_loss_reference",
    "attach_telemetry_with_plastic_probe_curves",
]
# ^^^ THOG
