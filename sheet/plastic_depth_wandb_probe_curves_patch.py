# vvv THOG
"""W&B-only rolling PLASTIC probe-loss and sampled-coefficient curves."""

from __future__ import annotations

import math
from array import array
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
_DELTA_LOSS_HEATMAP_KEY = "depth/delta_loss_v_layer_heatmap"
_DELTA_LOSS_HEATMAP_MAX_RENDERED_ROWS = 512
_DELTA_LOSS_HEATMAP_EARLY_REFRESH_PROBES = frozenset((1, 2, 3, 5, 10, 25, 50, 100))
_DELTA_LOSS_HEATMAP_COLOUR_SCALE = (
    (0.0, "rgb(0,255,0)"),
    (0.5, "rgb(88,88,88)"),
    (1.0, "rgb(255,0,0)"),
)
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
    raw_probe_sequence = payload.get("probe_sequence")
    probe_id = (
        f"P{int(raw_probe_sequence)}"
        if raw_probe_sequence is not None and int(raw_probe_sequence) > 0
        else f"U{update}"
    )
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


# vvv THOG retain every probe compactly, render at most 512 exact rows, and upload either sparsely or once per probe
def _delta_loss_heatmap_enabled(telemetry: Any) -> bool:
    config = getattr(telemetry, "config", {})
    return bool(
        isinstance(config, Mapping)
        and config.get("instrumentation__delta_loss_v_layer_heatmap") in {"log", "linear"}
        and not bool(getattr(telemetry, "_delta_loss_heatmap_disabled", False))
    )


def _ensure_delta_loss_heatmap_state(telemetry: Any) -> List[Dict[str, Any]]:
    history = getattr(telemetry, "_delta_loss_heatmap_history", None)
    if history is None:
        history = []
        setattr(telemetry, "_delta_loss_heatmap_history", history)
        setattr(telemetry, "_delta_loss_heatmap_last_event_index", -1)
        setattr(telemetry, "_delta_loss_heatmap_last_logged_total", 0)
        setattr(telemetry, "_delta_loss_heatmap_maximum_layers", None)
    return history


def _consume_new_delta_loss_heatmap_records(
    trainer: Any,
    telemetry: Any,
) -> Tuple[Dict[str, Any], ...]:
    _ensure_delta_loss_heatmap_state(telemetry)
    events = tuple(getattr(trainer, "events", ()))
    start = int(getattr(telemetry, "_delta_loss_heatmap_last_event_index", -1)) + 1
    records = tuple(
        record
        for event in events[start:]
        if (record := _probe_record_from_event(event)) is not None
    )
    telemetry._delta_loss_heatmap_last_event_index = len(events) - 1
    return records


def _maximum_candidate_layer(
    records: Sequence[Mapping[str, Any]],
    *,
    minimum: int,
) -> int:
    maximum = max(1, int(minimum))
    for record in records:
        for side in ("shrink", "growth"):
            for _distance, _delta, candidate, _offset in record.get(side, ()):
                maximum = max(maximum, int(candidate))
    return maximum


def _delta_loss_heatmap_record(
    record: Mapping[str, Any],
    *,
    maximum_layers: int,
) -> Dict[str, Any]:
    if maximum_layers < 1:
        raise ValueError("delta-loss heatmap maximum_layers must be positive")
    values = array("d", (math.nan for _ in range(maximum_layers)))
    for side in ("shrink", "growth"):
        for _distance, delta, candidate, _offset in record.get(side, ()):
            candidate_index = int(candidate) - 1
            if 0 <= candidate_index < maximum_layers:
                values[candidate_index] = float(delta)
    return {
        "probe_id": str(record["probe_id"]),
        "optimizer_update": int(record["optimizer_update"]),
        "active_layers": int(record["active_layers"]),
        "values": values,
    }


def _append_delta_loss_heatmap_records(
    telemetry: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    maximum_layers: int,
) -> None:
    history = _ensure_delta_loss_heatmap_state(telemetry)
    established_maximum = getattr(telemetry, "_delta_loss_heatmap_maximum_layers", None)
    resolved_maximum = int(maximum_layers)
    if established_maximum is None:
        telemetry._delta_loss_heatmap_maximum_layers = resolved_maximum
    elif resolved_maximum > int(established_maximum):
        extension = resolved_maximum - int(established_maximum)
        for record in history:
            record["values"].extend(math.nan for _ in range(extension))
        telemetry._delta_loss_heatmap_maximum_layers = resolved_maximum
    else:
        resolved_maximum = int(established_maximum)
    history.extend(
        _delta_loss_heatmap_record(record, maximum_layers=resolved_maximum)
        for record in records
    )


def _evenly_spaced_record_indices(count: int, limit: int) -> Tuple[int, ...]:
    if count <= limit:
        return tuple(range(count))
    if limit < 2:
        return (count - 1,)
    denominator = limit - 1
    return tuple(
        (position * (count - 1) + denominator // 2) // denominator
        for position in range(limit)
    )


def _delta_loss_heatmap_render_data(
    history: Sequence[Mapping[str, Any]],
    *,
    maximum_layers: int,
    rendered_row_limit: int = _DELTA_LOSS_HEATMAP_MAX_RENDERED_ROWS,
) -> Dict[str, Any]:
    indices = _evenly_spaced_record_indices(len(history), int(rendered_row_limit))
    selected = tuple(history[index] for index in indices)
    probe_labels = tuple(
        f"{record['probe_id']} | update {int(record['optimizer_update'])} | active {int(record['active_layers'])}"
        for record in selected
    )
    probe_rows = tuple(
        tuple(
            None if not math.isfinite(float(value)) else float(value)
            for value in record["values"]
        )
        for record in selected
    )
    # vvv THOG steps run left-to-right and absolute layer counts bottom-to-top; unit-spaced numeric coordinates permit a true square-cell aspect lock
    z_by_layer = tuple(
        tuple(probe_rows[probe_index][layer_index] for probe_index in range(len(selected)))
        for layer_index in range(int(maximum_layers))
    )
    return {
        "x": tuple(range(1, len(selected) + 1)),
        "x_steps": tuple(int(record["optimizer_update"]) for record in selected),
        "y": tuple(range(1, int(maximum_layers) + 1)),
        "z": z_by_layer,
        "probe_labels": probe_labels,
        "active_layers": tuple(int(record["active_layers"]) for record in selected),
        "optimizer_updates": tuple(int(record["optimizer_update"]) for record in selected),
        "source_rows": len(history),
        "rendered_rows": len(selected),
    }
    # ^^^ THOG


def _delta_loss_heatmap_figure(
    history: Sequence[Mapping[str, Any]],
    *,
    maximum_layers: int,
    abs_limit: float,
    go_module: Any = None,
) -> Any:
    if go_module is None:
        import plotly.graph_objects as go_module

    rendered = _delta_loss_heatmap_render_data(
        history,
        maximum_layers=int(maximum_layers),
    )
    figure = go_module.Figure()
    figure.add_trace(
        go_module.Heatmap(
            x=rendered["x"],
            y=rendered["y"],
            z=rendered["z"],
            zmin=-float(abs_limit),
            zmax=float(abs_limit),
            zmid=0.0,
            colorscale=list(_DELTA_LOSS_HEATMAP_COLOUR_SCALE),
            colorbar={"title": "candidate loss − current loss"},
            connectgaps=False,
            hovertemplate=(
                "step=%{customdata}<br>candidate layers=%{y}<br>"
                "Δloss=%{z:.8f}<extra></extra>"
            ),
            customdata=[
                list(rendered["x_steps"])
                for _layer in rendered["y"]
            ],
            xgap=0,
            ygap=0,
        )
    )
    figure.add_trace(
        go_module.Scatter(
            x=rendered["x"],
            y=rendered["active_layers"],
            mode="lines",
            line={"color": "black", "width": 1.25},
            name="active layer count",
            customdata=rendered["x_steps"],
            hovertemplate="step=%{customdata}<br>active layers=%{y}<extra></extra>",
        )
    )
    tick_count = min(12, int(rendered["rendered_rows"]))
    tick_indices = _evenly_spaced_record_indices(
        int(rendered["rendered_rows"]),
        tick_count,
    )
    figure.update_layout(
        title=(
            "DEPTH Δloss by step and absolute candidate layer — "
            f"{rendered['source_rows']} probes captured; "
            f"{rendered['rendered_rows']} exact rows shown"
        ),
        # vvv THOG white canvas and equal numeric axis scales keep unprobed cells white and every rendered heatmap cell square
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": "rgb(32,32,32)"},
        height=760,
        margin={"l": 85, "r": 45, "t": 75, "b": 75},
        showlegend=True,
        uirevision="depth_delta_loss_v_layer_heatmap_step_x_square_cells_v2",
        xaxis={
            "title": "step",
            "range": (0.5, int(rendered["rendered_rows"]) + 0.5),
            "tickmode": "array",
            "tickvals": [int(rendered["x"][index]) for index in tick_indices],
            "ticktext": [str(rendered["x_steps"][index]) for index in tick_indices],
            "constrain": "domain",
        },
        yaxis={
            "title": "absolute candidate layer count",
            "range": (0.5, int(maximum_layers) + 0.5),
            "tickmode": "linear",
            "tick0": 1,
            "dtick": max(1, int(maximum_layers) // 12),
            "scaleanchor": "x",
            "scaleratio": 1,
            "constrain": "domain",
        },
        # ^^^ THOG
    )
    return figure


def _should_refresh_delta_loss_heatmap(
    telemetry: Any,
    *,
    force: bool = False,
) -> bool:
    history = _ensure_delta_loss_heatmap_state(telemetry)
    total = len(history)
    last_logged = int(getattr(telemetry, "_delta_loss_heatmap_last_logged_total", 0))
    if total <= last_logged:
        return False
    config = getattr(telemetry, "config", {})
    mode = config.get("instrumentation__delta_loss_v_layer_heatmap")
    if mode == "linear":
        maximum_step = config.get(
            "instrumentation__delta_loss_v_layer_heatmap_linear"
        )
        latest_probe_step = int(history[-1]["optimizer_update"])
        if maximum_step is not None and latest_probe_step > int(maximum_step):
            return False
        return True
    if force or total in _DELTA_LOSS_HEATMAP_EARLY_REFRESH_PROBES:
        return True
    cadence = int(
        config.get(
            "instrumentation__delta_loss_v_layer_heatmap_log_every_n_probes",
            250,
        )
    )
    return total // cadence > last_logged // cadence
# ^^^ THOG


def _log_rolling_probe_charts(
    telemetry: Any,
    *,
    step: int,
    include_probe_charts: bool = True,
    include_coefficient_chart: bool = True,
    include_delta_loss_heatmap: bool = False,
) -> None:
    if not _plastic_wandb_charts_enabled() and not include_delta_loss_heatmap:
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

    # vvv THOG log the explicitly enabled absolute-layer heatmap independently of PLASTIC mutation state
    if include_delta_loss_heatmap:
        heatmap_history = tuple(_ensure_delta_loss_heatmap_state(telemetry))
        maximum_layers = int(telemetry._delta_loss_heatmap_maximum_layers)
        abs_limit = float(
            telemetry.config[
                "instrumentation__delta_loss_v_layer_heatmap_abs_limit"
            ]
        )
        payload[_DELTA_LOSS_HEATMAP_KEY] = _delta_loss_heatmap_figure(
            heatmap_history,
            maximum_layers=maximum_layers,
            abs_limit=abs_limit,
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
    if include_delta_loss_heatmap and _DELTA_LOSS_HEATMAP_KEY in payload:
        telemetry._delta_loss_heatmap_last_logged_total = len(
            _ensure_delta_loss_heatmap_state(telemetry)
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
    "_consume_new_delta_loss_heatmap_records",
    "_delta_loss_heatmap_enabled",
    "_delta_loss_heatmap_figure",
    "_delta_loss_heatmap_record",
    "_delta_loss_heatmap_render_data",
    "_append_delta_loss_heatmap_records",
    "_ensure_coefficient_curve_state",
    "_ensure_delta_loss_heatmap_state",
    "_log_rolling_probe_charts",
    "_maximum_candidate_layer",
    "_plastic_wandb_charts_enabled",
    "_probe_record_from_event",
    "_rows_for_coefficients",
    "_rows_for_combined",
    "_rows_for_side",
    "_rows_with_zero_loss_reference",
    "_should_refresh_delta_loss_heatmap",
    "attach_telemetry_with_plastic_probe_curves",
]
# ^^^ THOG
