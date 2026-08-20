# vvv THOG
"""Second-stage DEPTH weight-curve diagnostics: layer-index ruler, full attention families, and interactive Plotly history."""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch

import constants as _constants

from . import depth_weight_curves_and_observational_probes_patch as _depth
from .basis import chebyshev_first_kind_basis, stabilized_chebyshev_basis_at_coordinates
from .plastic_depth import public_to_internal_depth
from .semantic_materializer import (
    ATTENTION_KEY_WEIGHT,
    ATTENTION_OUTPUT_WEIGHT,
    ATTENTION_QUERY_WEIGHT,
    ATTENTION_VALUE_WEIGHT,
    MLP_CONTRACTION_WEIGHT,
    MLP_EXPANSION_WEIGHT,
)


_CHART_FAMILIES = {
    "attn_q_head_N": ATTENTION_QUERY_WEIGHT,
    "attn_k_head_N": ATTENTION_KEY_WEIGHT,
    "attn_v_head_N": ATTENTION_VALUE_WEIGHT,
    "attn_out_head_N": ATTENTION_OUTPUT_WEIGHT,
    "mlp_up": MLP_EXPANSION_WEIGHT,
    "mlp_down": MLP_CONTRACTION_WEIGHT,
}
_SCALAR_COLOURS = (
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
)


# vvv THOG sample a deterministic rectangular matrix region so attention-output diagnostics can select the chosen head by input-column slice rather than output-row slice
def _sample_matrix_rectangle(
    *,
    seed: int,
    row_start: int,
    row_stop: int,
    column_start: int,
    column_stop: int,
    count: int,
) -> Tuple[Tuple[int, int], ...]:
    row_count = int(row_stop) - int(row_start)
    column_count = int(column_stop) - int(column_start)
    population = row_count * column_count
    if population < 1:
        raise ValueError("depth-weight diagnostic matrix region is empty")
    requested = min(int(count), population)
    generator = random.Random(int(seed))
    flat_indices = generator.sample(range(population), requested)
    return tuple(
        (
            int(row_start) + flat_index // column_count,
            int(column_start) + flat_index % column_count,
        )
        for flat_index in flat_indices
    )
# ^^^ THOG


# vvv THOG one deterministic attention head N is shared by Q/K/V/OUT; the fixed cross-run seed preserves N and scalar identities when requested
def _selected_scalar_coordinates_v2(trainer: Any, telemetry: Any) -> Dict[str, Any]:
    cached = getattr(telemetry, "_thog_depth_weight_curve_selection", None)
    required = set(_CHART_FAMILIES) | {"seed", "attention_head"}
    if isinstance(cached, Mapping) and required.issubset(cached.keys()):
        return dict(cached)

    trajectory = _depth._depth_trajectory_from_model(trainer.raw_model)
    if trajectory is None:
        return {}
    width = int(trajectory.config.n_embd)
    n_head = int(trajectory.config.n_head)
    if width % n_head != 0:
        raise ValueError("n_embd must be divisible by n_head for DEPTH attention diagnostics")

    seed = int(_depth._selection_seed(trainer, telemetry))
    head = random.Random(seed ^ 0xA771).randrange(n_head)
    head_dim = width // n_head
    count = int(_depth._scalar_weights_per_matrix())
    head_start = head * head_dim
    head_stop = (head + 1) * head_dim

    selection = {
        "seed": seed,
        "attention_head": head,
        "attn_q_head_N": _sample_matrix_rectangle(
            seed=seed ^ 0x11,
            row_start=head_start,
            row_stop=head_stop,
            column_start=0,
            column_stop=width,
            count=count,
        ),
        "attn_k_head_N": _sample_matrix_rectangle(
            seed=seed ^ 0x12,
            row_start=head_start,
            row_stop=head_stop,
            column_start=0,
            column_stop=width,
            count=count,
        ),
        "attn_v_head_N": _sample_matrix_rectangle(
            seed=seed ^ 0x13,
            row_start=head_start,
            row_stop=head_stop,
            column_start=0,
            column_stop=width,
            count=count,
        ),
        "attn_out_head_N": _sample_matrix_rectangle(
            seed=seed ^ 0x14,
            row_start=0,
            row_stop=width,
            column_start=head_start,
            column_stop=head_stop,
            count=count,
        ),
        "mlp_up": _sample_matrix_rectangle(
            seed=seed ^ 0x22,
            row_start=0,
            row_stop=4 * width,
            column_start=0,
            column_stop=width,
            count=count,
        ),
        "mlp_down": _sample_matrix_rectangle(
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
# ^^^ THOG


# vvv THOG actual executed DEPTH samples define integer x coordinates 1..L; PLASTIC public coordinates are retained only as the hidden coefficient-evaluation chart beneath that ruler
def _executed_public_coordinates(trajectory: Any, reference: torch.Tensor) -> torch.Tensor:
    if bool(getattr(trajectory, "plastic_enabled", False)) and trajectory.plastic_sampling is not None:
        return trajectory.plastic_sampling.active_public_coordinates().detach().to(
            device=reference.device,
            dtype=torch.float64,
        )
    return torch.linspace(
        1.0,
        100.0,
        int(trajectory.config.n_layer),
        device=reference.device,
        dtype=torch.float64,
    )


def _public_coordinates_from_layer_axis(
    layer_axis: torch.Tensor,
    executed_public: torch.Tensor,
) -> torch.Tensor:
    count = int(executed_public.numel())
    if count < 1:
        raise ValueError("DEPTH diagnostic requires at least one executed layer")
    if count == 1:
        return torch.full_like(layer_axis, float(executed_public[0].item()))

    zero_based = layer_axis - 1.0
    left = torch.floor(zero_based).to(dtype=torch.long).clamp(0, count - 1)
    right = (left + 1).clamp(0, count - 1)
    fraction = (zero_based - left.to(dtype=zero_based.dtype)).clamp(0.0, 1.0)
    return executed_public.index_select(0, left) + fraction * (
        executed_public.index_select(0, right) - executed_public.index_select(0, left)
    )


def _basis_at_public_coordinates(
    trajectory: Any,
    reference: torch.Tensor,
    public: torch.Tensor,
) -> torch.Tensor:
    internal = public_to_internal_depth(public).to(dtype=torch.float64)
    if bool(getattr(trajectory, "plastic_enabled", False)):
        raw = chebyshev_first_kind_basis(internal, trajectory.config.depth_order)
        basis = raw @ trajectory.plastic_depth_inverse_r.to(
            device=internal.device,
            dtype=internal.dtype,
        )
        return basis.to(dtype=reference.dtype)
    return stabilized_chebyshev_basis_at_coordinates(
        internal,
        reference_sample_count=int(trajectory.config.n_layer),
        order=int(trajectory.config.depth_order),
        runtime_dtype=reference.dtype,
        version=trajectory.basis_version,
    )


def _continuous_layer_basis(
    trajectory: Any,
    reference: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    executed_public = _executed_public_coordinates(trajectory, reference)
    layer_count = int(executed_public.numel())
    layer_axis = torch.linspace(
        1.0,
        float(layer_count),
        int(_depth._depth_evaluation_points()),
        device=reference.device,
        dtype=torch.float64,
    )
    public = _public_coordinates_from_layer_axis(layer_axis, executed_public)
    continuous_basis = _basis_at_public_coordinates(trajectory, reference, public)
    executed_layer_axis = torch.arange(
        1,
        layer_count + 1,
        device=reference.device,
        dtype=torch.float64,
    )
    executed_basis = _basis_at_public_coordinates(
        trajectory,
        reference,
        executed_public,
    )
    return layer_axis, continuous_basis, executed_layer_axis, executed_basis
# ^^^ THOG


# vvv THOG snapshot all six requested semantic families on the layer-index ruler and retain exact executed-layer values for marker overlays
@torch.no_grad()
def _depth_weight_snapshot_v2(
    trainer: Any,
    telemetry: Any,
    *,
    optimizer_update: int,
) -> Dict[str, Any]:
    trajectory = _depth._depth_trajectory_from_model(trainer.raw_model)
    if trajectory is None:
        return {}
    selection = _selected_scalar_coordinates_v2(trainer, telemetry)
    if not selection:
        return {}

    snapshot: Dict[str, Any] = {
        "optimizer_update": int(optimizer_update),
        "attention_head": int(selection["attention_head"]),
        "seed": int(selection["seed"]),
        "families": {},
    }
    for chart_name, family in _CHART_FAMILIES.items():
        parameter = trajectory.coefficients[family]
        layer_axis, basis, executed_layer_axis, executed_basis = _continuous_layer_basis(
            trajectory,
            parameter,
        )
        curves = []
        for output_row, row_index in selection[chart_name]:
            coefficient = parameter[int(output_row), int(row_index)]
            runtime_basis = basis.to(device=coefficient.device, dtype=coefficient.dtype)
            runtime_executed_basis = executed_basis.to(
                device=coefficient.device,
                dtype=coefficient.dtype,
            )
            values = runtime_basis @ coefficient
            executed_values = runtime_executed_basis @ coefficient
            curves.append(
                {
                    "scalar_id": f"r{int(output_row)}_c{int(row_index)}",
                    "output_row": int(output_row),
                    "row_index": int(row_index),
                    "values": tuple(
                        float(value)
                        for value in values.detach().to(device="cpu", dtype=torch.float64).tolist()
                    ),
                    "executed_values": tuple(
                        float(value)
                        for value in executed_values.detach().to(device="cpu", dtype=torch.float64).tolist()
                    ),
                }
            )
        snapshot["families"][chart_name] = {
            "semantic_family": family,
            "depth_coordinates": tuple(
                float(value) for value in layer_axis.detach().to(device="cpu").tolist()
            ),
            "executed_layer_coordinates": tuple(
                float(value)
                for value in executed_layer_axis.detach().to(device="cpu").tolist()
            ),
            "curves": tuple(curves),
        }
    return snapshot
# ^^^ THOG


# vvv THOG direct Plotly logging makes age and exact update identity visible: newer curves are darker/thicker, oldest/newest are labelled, and hover identifies every intermediate snapshot
def _chart_title(chart_name: str, snapshot: Mapping[str, Any]) -> str:
    head = int(snapshot["attention_head"])
    if chart_name.startswith("attn_"):
        return chart_name.replace("_N", f"_{head}")
    return chart_name


def _build_depth_plotly_figure(
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
    latest_family = retained[-1]["families"][chart_name]
    executed_layers = tuple(float(value) for value in latest_family["executed_layer_coordinates"])
    layer_count = len(executed_layers)

    figure = go.Figure()
    for history_index, snapshot in enumerate(retained):
        update = int(snapshot["optimizer_update"])
        family = snapshot["families"][chart_name]
        x_values = tuple(float(value) for value in family["depth_coordinates"])
        if history_count == 1:
            age_fraction = 1.0
        else:
            age_fraction = history_index / float(history_count - 1)
        opacity = 0.18 + 0.82 * age_fraction
        width = 1.0 + 2.6 * age_fraction
        is_oldest = history_index == 0
        is_newest = history_index == history_count - 1
        age_label = "newest" if is_newest else ("oldest" if is_oldest else f"history {history_index + 1}/{history_count}")

        for scalar_index, curve in enumerate(family["curves"]):
            scalar_id = str(curve["scalar_id"])
            colour = _SCALAR_COLOURS[scalar_index % len(_SCALAR_COLOURS)]
            legend_name = f"{scalar_id} · {age_label} U{update}"
            figure.add_trace(
                go.Scatter(
                    x=x_values,
                    y=tuple(float(value) for value in curve["values"]),
                    mode="lines",
                    name=legend_name,
                    showlegend=is_oldest or is_newest,
                    line={"color": colour, "width": width},
                    opacity=opacity,
                    meta={
                        "instra_thog_weight": True,
                        "instra_thog_optimizer_update": update,
                        "instra_thog_scalar_id": scalar_id,
                        "instra_thog_integer_x": tuple(
                            float(value)
                            for value in family["executed_layer_coordinates"]
                        ),
                        "instra_thog_integer_y": tuple(
                            float(value) for value in curve["executed_values"]
                        ),
                    },
                    hovertemplate=(
                        f"{scalar_id}<br>U{update} · {age_label}"
                        "<br>layer=%{x:.3f}<br>weight=%{y:.7g}<extra></extra>"
                    ),
                )
            )
            if is_newest:
                figure.add_trace(
                    go.Scatter(
                        x=tuple(float(value) for value in family["executed_layer_coordinates"]),
                        y=tuple(float(value) for value in curve["executed_values"]),
                        mode="markers",
                        name=f"{scalar_id} · executed layers",
                        showlegend=False,
                        meta={
                            "instra_thog_weight": True,
                            "instra_thog_executed_overlay": True,
                            "instra_thog_optimizer_update": update,
                            "instra_thog_scalar_id": scalar_id,
                        },
                        marker={
                            "color": colour,
                            "size": 7,
                            "symbol": "circle-open",
                            "line": {"width": 1.5, "color": colour},
                        },
                        hovertemplate=(
                            f"{scalar_id}<br>U{update} · executed layer"
                            "<br>layer=%{x:.0f}<br>weight=%{y:.7g}<extra></extra>"
                        ),
                    )
                )

    title_name = _chart_title(chart_name, retained[-1])
    age_title = (
        f"U{oldest_update} oldest → U{newest_update} newest"
        if history_count > 1
        else f"U{newest_update} newest"
    )
    figure.update_layout(
        title=(
            f"DEPTH generated scalar trajectories — {title_name}"
            f"<br><sup>{age_title}; newer = darker/thicker; ○ = executed layer on newest curve</sup>"
        ),
        xaxis_title="layer index",
        yaxis_title="weight value",
        hovermode="closest",
        legend={"title": {"text": "oldest / newest"}},
        template="plotly_white",
    )
    if layer_count >= 1:
        xaxis_options = {
            "range": [1.0, float(layer_count)],
            "tickangle": 0,
            "tickformat": ".0f",
            "automargin": True,
        }
        if layer_count <= 18:
            xaxis_options.update({"tickmode": "linear", "tick0": 1, "dtick": 1})
        else:
            xaxis_options.update({"tickmode": "auto", "nticks": 18})
        figure.update_xaxes(**xaxis_options)
    return figure
# ^^^ THOG


# vvv THOG replace only the DEPTH diagnostic logger; training, checkpoint, probe, and existing telemetry state remain untouched
def _log_depth_weight_snapshot_v2(
    trainer: Any,
    telemetry: Any,
    *,
    optimizer_update: int,
) -> None:
    if int(getattr(_constants, "DEBUG", 0)) <= 2:
        return
    if telemetry.run is None:
        return

    snapshot = _depth_weight_snapshot_v2(
        trainer,
        telemetry,
        optimizer_update=optimizer_update,
    )
    if not snapshot:
        return
    history = _depth._depth_weight_history(telemetry)
    history.append(snapshot)
    snapshots = (snapshot,) if _depth._time_mode() == "latest" else tuple(history)

    payload: Dict[str, Any] = {}
    for chart_name in _CHART_FAMILIES:
        payload[f"depth/{chart_name}"] = _build_depth_plotly_figure(snapshots, chart_name)
    try:
        telemetry.run.log(payload, step=int(optimizer_update))
    except TypeError:
        telemetry.run.log(payload)

    # vvv THOG W&B receives only the six requested depth figures; selection metadata remains in run configuration and must not create scalar panels
    # if not bool(getattr(telemetry, "_thog_depth_weight_curve_selection_logged", False)):
    #     selection = _selected_scalar_coordinates_v2(trainer, telemetry)
    #     metadata = {
    #         "depth/selection_seed": int(selection["seed"]),
    #         "depth/attention_head": int(selection["attention_head"]),
    #         "depth/same_coordinates_all_runs": bool(_depth._same_coordinates_all_runs()),
    #         "depth/scalar_weights_per_matrix": int(_depth._scalar_weights_per_matrix()),
    #         "depth/evaluation_points": int(_depth._depth_evaluation_points()),
    #         "depth/x_axis": "executed_layer_index",
    #         "depth/chart_renderer": "plotly",
    #     }
    #     try:
    #         telemetry.run.log(metadata, step=int(optimizer_update))
    #     except TypeError:
    #         telemetry.run.log(metadata)
    #     telemetry._thog_depth_weight_curve_selection_logged = True
    # ^^^ THOG
# ^^^ THOG


_depth._selected_scalar_coordinates = _selected_scalar_coordinates_v2
_depth._depth_weight_snapshot = _depth_weight_snapshot_v2
_depth._log_depth_weight_snapshot = _log_depth_weight_snapshot_v2
_depth._build_depth_plotly_figure = _build_depth_plotly_figure
_depth._DEPTH_WEIGHT_CHART_FAMILIES = tuple(_CHART_FAMILIES)


__all__ = [
    "_build_depth_plotly_figure",
    "_depth_weight_snapshot_v2",
    "_log_depth_weight_snapshot_v2",
    "_selected_scalar_coordinates_v2",
]
# ^^^ THOG
