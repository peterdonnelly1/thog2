# vvv THOG
"""Render conventional DENSE weights in the six established depth-weight charts."""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Mapping, Sequence

import torch

import constants as _constants

from . import depth_weight_curves_and_observational_probes_patch as _depth
from . import depth_weight_curves_v2_patch as _v2
from . import wandb_telemetry as _wandb
from .training_model import TrainingDenseGPT


_DENSE_KIND = "dense_discrete_weights"


def _dense_model(raw_model: Any) -> bool:
    return isinstance(raw_model, TrainingDenseGPT)


def _dense_selection(trainer: Any, telemetry: Any) -> Dict[str, Any]:
    cached = getattr(telemetry, "_thog_depth_weight_curve_selection", None)
    required = set(_v2._CHART_FAMILIES) | {"seed", "attention_head"}
    if isinstance(cached, Mapping) and required.issubset(cached.keys()):
        return dict(cached)
    model = trainer.raw_model
    if not _dense_model(model):
        return {}
    width = int(model.config.n_embd)
    n_head = int(model.config.n_head)
    if width % n_head != 0:
        raise ValueError("n_embd must be divisible by n_head for DENSE weight diagnostics")
    seed = int(_depth._selection_seed(trainer, telemetry))
    head = random.Random(seed ^ 0xA771).randrange(n_head)
    head_dim = width // n_head
    head_start = head * head_dim
    head_stop = (head + 1) * head_dim
    count = int(_depth._scalar_weights_per_matrix())
    selection = {
        "seed": seed,
        "attention_head": head,
        "attn_q_head_N": _v2._sample_matrix_rectangle(
            seed=seed ^ 0x11,
            row_start=head_start,
            row_stop=head_stop,
            column_start=0,
            column_stop=width,
            count=count,
        ),
        "attn_k_head_N": _v2._sample_matrix_rectangle(
            seed=seed ^ 0x12,
            row_start=head_start,
            row_stop=head_stop,
            column_start=0,
            column_stop=width,
            count=count,
        ),
        "attn_v_head_N": _v2._sample_matrix_rectangle(
            seed=seed ^ 0x13,
            row_start=head_start,
            row_stop=head_stop,
            column_start=0,
            column_stop=width,
            count=count,
        ),
        "attn_out_head_N": _v2._sample_matrix_rectangle(
            seed=seed ^ 0x14,
            row_start=0,
            row_stop=width,
            column_start=head_start,
            column_stop=head_stop,
            count=count,
        ),
        "mlp_up": _v2._sample_matrix_rectangle(
            seed=seed ^ 0x22,
            row_start=0,
            row_stop=4 * width,
            column_start=0,
            column_stop=width,
            count=count,
        ),
        "mlp_down": _v2._sample_matrix_rectangle(
            seed=seed ^ 0x33,
            row_start=0,
            row_stop=width,
            column_start=0,
            column_stop=4 * width,
            count=count,
        ),
    }
    setattr(telemetry, "_thog_depth_weight_curve_selection", selection)
    return selection


def _dense_family_weight(block: Any, chart_name: str, width: int) -> tuple[torch.Tensor, int]:
    if chart_name == "attn_q_head_N":
        return block.attn.c_attn.weight, 0
    if chart_name == "attn_k_head_N":
        return block.attn.c_attn.weight, width
    if chart_name == "attn_v_head_N":
        return block.attn.c_attn.weight, 2 * width
    if chart_name == "attn_out_head_N":
        return block.attn.c_proj.weight, 0
    if chart_name == "mlp_up":
        return block.mlp.c_fc.weight, 0
    if chart_name == "mlp_down":
        return block.mlp.c_proj.weight, 0
    raise KeyError(chart_name)


@torch.no_grad()
def _dense_weight_snapshot(
    trainer: Any,
    telemetry: Any,
    *,
    optimizer_update: int,
) -> Dict[str, Any]:
    model = trainer.raw_model
    if not _dense_model(model):
        return {}
    selection = _dense_selection(trainer, telemetry)
    if not selection:
        return {}
    blocks = tuple(model.transformer.h)
    layer_axis = tuple(float(index) for index in range(1, len(blocks) + 1))
    width = int(model.config.n_embd)
    snapshot: Dict[str, Any] = {
        "optimizer_update": int(optimizer_update),
        "attention_head": int(selection["attention_head"]),
        "seed": int(selection["seed"]),
        "model_type": "dense",
        "trajectory_kind": _DENSE_KIND,
        "families": {},
    }
    for chart_name, semantic_family in _v2._CHART_FAMILIES.items():
        curves = []
        for output_row, row_index in selection[chart_name]:
            values = []
            for block in blocks:
                weight, row_offset = _dense_family_weight(block, chart_name, width)
                values.append(
                    float(
                        weight[int(output_row) + int(row_offset), int(row_index)]
                        .detach()
                        .to(device="cpu", dtype=torch.float64)
                        .item()
                    )
                )
            curves.append(
                {
                    "scalar_id": f"r{int(output_row)}_c{int(row_index)}",
                    "output_row": int(output_row),
                    "row_index": int(row_index),
                    "values": tuple(values),
                    "executed_values": tuple(values),
                }
            )
        snapshot["families"][chart_name] = {
            "semantic_family": semantic_family,
            "depth_coordinates": layer_axis,
            "executed_layer_coordinates": layer_axis,
            "curves": tuple(curves),
        }
    return snapshot


_ORIGINAL_DEPTH_WEIGHT_SNAPSHOT = _depth._depth_weight_snapshot


def _weight_snapshot_with_dense(
    trainer: Any,
    telemetry: Any,
    *,
    optimizer_update: int,
) -> Dict[str, Any]:
    if _dense_model(trainer.raw_model):
        return _dense_weight_snapshot(
            trainer,
            telemetry,
            optimizer_update=optimizer_update,
        )
    return _ORIGINAL_DEPTH_WEIGHT_SNAPSHOT(
        trainer,
        telemetry,
        optimizer_update=optimizer_update,
    )


_depth._depth_weight_snapshot = _weight_snapshot_with_dense
# ^^^ THOG


def _snapshots_are_dense(snapshots: Sequence[Mapping[str, Any]]) -> bool:
    return bool(snapshots) and all(
        str(snapshot.get("trajectory_kind", "")) == _DENSE_KIND
        for snapshot in snapshots
    )


def _dense_step_colour(optimizer_update: int) -> str:
    """Return a stable, random-looking colour for one optimizer update."""

    seed = (int(optimizer_update) * 0x9E3779B1) & 0xFFFFFFFF
    hue = seed % 360
    saturation = 62 + ((seed >> 8) % 17)
    lightness = 43 + ((seed >> 16) % 12)
    return f"hsl({hue}, {saturation}%, {lightness}%)"


def _build_dense_plotly_figure(
    snapshots: Sequence[Mapping[str, Any]],
    chart_name: str,
):
    import plotly.graph_objects as go

    retained = tuple(snapshots)
    if not retained:
        return go.Figure()
    oldest_update = int(retained[0]["optimizer_update"])
    newest_update = int(retained[-1]["optimizer_update"])
    history_count = len(retained)
    figure = go.Figure()
    for history_index, snapshot in enumerate(retained):
        update = int(snapshot["optimizer_update"])
        family = snapshot["families"][chart_name]
        x_values = tuple(float(value) for value in family["executed_layer_coordinates"])
        age_fraction = 1.0 if history_count == 1 else history_index / float(history_count - 1)
        opacity = 0.22 + 0.78 * age_fraction
        marker_size = 4.0 + 2.0 * age_fraction
        colour = _dense_step_colour(update)
        for scalar_index, curve in enumerate(family["curves"]):
            scalar_id = str(curve["scalar_id"])
            figure.add_trace(
                go.Scatter(
                    x=x_values,
                    y=tuple(float(value) for value in curve["values"]),
                    mode="lines+markers",
                    name=f"step {update}",
                    legendgroup=f"dense-step-{update}",
                    showlegend=scalar_index == 0,
                    line={
                        "color": colour,
                        "width": 0.45,
                        "shape": "linear",
                    },
                    marker={
                        "color": colour,
                        "size": marker_size,
                        "symbol": "x",
                        "line": {"width": 0.35, "color": colour},
                    },
                    meta={
                        "instra_dense_weight": True,
                        "instra_dense_optimizer_update": update,
                        "instra_dense_step_legend": scalar_index == 0,
                        "instra_dense_scalar_id": scalar_id,
                    },
                    opacity=opacity,
                    hovertemplate=(
                        f"{scalar_id}<br>step {update}"
                        "<br>layer=%{x:.0f}<br>weight=%{y:.7g}<extra></extra>"
                    ),
                )
            )
    title_name = _v2._chart_title(chart_name, retained[-1])
    step_title = (
        f"steps {oldest_update} → {newest_update}"
        if history_count > 1
        else f"step {newest_update}"
    )
    layer_count = len(
        retained[-1]["families"][chart_name]["executed_layer_coordinates"]
    )
    figure.update_layout(
        title=(
            f"DENSE learned scalar weights — {title_name}"
            f"<br><sup>{step_title}; × = discrete materialised layer weight; faint lines group one step</sup>"
        ),
        xaxis_title="layer index",
        yaxis_title="weight value",
        hovermode="closest",
        legend={"title": {"text": ""}},
        template="plotly_white",
    )
    if layer_count >= 1:
        options: Dict[str, Any] = {
            "range": [0.5, float(layer_count) + 0.5],
            "tickangle": 0,
            "tickformat": ".0f",
            "automargin": True,
        }
        if layer_count <= 18:
            options.update({"tickmode": "linear", "tick0": 1, "dtick": 1})
        else:
            options.update({"tickmode": "auto", "nticks": 18})
        figure.update_xaxes(**options)
    return figure


_ORIGINAL_BUILD_DEPTH_FIGURE = _v2._build_depth_plotly_figure


def _build_weight_plotly_figure(
    snapshots: Sequence[Mapping[str, Any]],
    chart_name: str,
):
    if _snapshots_are_dense(snapshots):
        return _build_dense_plotly_figure(snapshots, chart_name)
    return _ORIGINAL_BUILD_DEPTH_FIGURE(snapshots, chart_name)


_v2._build_depth_plotly_figure = _build_weight_plotly_figure
_depth._build_depth_plotly_figure = _build_weight_plotly_figure
# ^^^ THOG


# vvv THOG the retained telemetry hook installs only for DEPTH; add the identical
# cadence/destination lifecycle for DENSE without touching the DEPTH path.
_ORIGINAL_ATTACH_TELEMETRY = _wandb.attach_telemetry


def _attach_telemetry_with_dense_weight_curves(trainer: Any, telemetry: Any) -> None:
    _ORIGINAL_ATTACH_TELEMETRY(trainer, telemetry)
    if int(getattr(_constants, "DEBUG", 0)) <= 2 or not _dense_model(trainer.raw_model):
        return
    original_timed = trainer._timed
    train_one_update = trainer.train_one_update

    def timed(function: Any):
        metrics, elapsed = original_timed(function)
        if function != train_one_update:
            return metrics, elapsed
        if not trainer.distributed.is_primary:
            return metrics, elapsed
        if bool(float(metrics.get("skipped_update", 0.0))):
            return metrics, elapsed
        update = int(trainer.state.completed_updates)
        if update < 1 or (update != 1 and update % _depth._log_every_n_steps() != 0):
            return metrics, elapsed
        try:
            _depth._log_depth_weight_snapshot(
                trainer,
                telemetry,
                optimizer_update=update,
            )
        except Exception as error:
            print(
                "THOG2 WARNING: DENSE discrete weight-chart logging failed; "
                f"continuing without this refresh: {error}",
                flush=True,
            )
        return metrics, elapsed

    trainer._timed = timed


_wandb.attach_telemetry = _attach_telemetry_with_dense_weight_curves
# ^^^ THOG


__all__ = [
    "_build_dense_plotly_figure",
    "_dense_selection",
    "_dense_weight_snapshot",
    "_weight_snapshot_with_dense",
]
# ^^^ THOG

# vvv THOG install matched logical weight selection after DENSE has installed its final snapshot and figure dispatch
from . import matched_weight_selection_patch as _matched_weight_selection_patch
# ^^^ THOG
