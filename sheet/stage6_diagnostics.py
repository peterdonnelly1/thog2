# vvv THOG
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch import Tensor

from .training_model import TrainingSheetGPT


def finite_float(value: Tensor) -> float:
    result = float(value.detach().to(dtype=torch.float64, device="cpu").item())
    if not math.isfinite(result):
        raise FloatingPointError("non-finite Stage 6 diagnostic")
    return result


def normalized_energy(values: Tensor) -> Tuple[float, ...]:
    energy = values.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
    total = float(energy.sum().item())
    if total <= 0.0:
        return tuple(0.0 for _ in range(energy.numel()))
    return tuple(float(value) / total for value in energy.tolist())


def high_order_fraction(values: Tuple[float, ...]) -> float:
    if not values:
        return 0.0
    count = max(1, math.ceil(len(values) / 4))
    return float(sum(values[-count:]))


def _coefficient_axis_plan(coefficient: Tensor, depth_order: int) -> Tuple[int, Tuple[int, ...]]:
    shape = tuple(coefficient.shape)
    if coefficient.ndim == 3 and shape[1] == depth_order:
        return 1, (2,)
    if coefficient.ndim == 3 and shape[0] == depth_order:
        return 0, (1, 2)
    if coefficient.ndim == 3 and shape[2] == depth_order:
        return 2, ()
    if coefficient.ndim == 4 and shape[1] == depth_order:
        return 1, (2, 3)
    raise ValueError(f"unsupported coefficient diagnostic shape {shape} for depth_order={depth_order}")


def _axis_energy_fraction(coefficient: Tensor, axis: int) -> Tuple[float, ...]:
    reduce_dims = tuple(index for index in range(coefficient.ndim) if index != axis)
    return normalized_energy(coefficient.square().sum(dim=reduce_dims))


def _combined_axis_energy_fraction(coefficient: Tensor, axes: Tuple[int, ...]) -> Tuple[float, ...]:
    if not axes:
        return tuple()
    reduce_dims = tuple(index for index in range(coefficient.ndim) if index not in axes)
    return normalized_energy(coefficient.square().sum(dim=reduce_dims))


def gradient_report(model: TrainingSheetGPT) -> Dict[str, Dict[str, float]]:
    rows: Dict[str, Dict[str, float]] = {}
    for metadata in model.trajectory.metadata:
        parameter = model.trajectory.coefficients[metadata.name]
        gradient = parameter.grad
        if gradient is None:
            rows[metadata.name] = {
                "gradient_present": 0.0,
                "gradient_l2_norm": 0.0,
                "gradient_rms": 0.0,
                "gradient_max_abs": 0.0,
            }
            continue
        detached = gradient.detach().float()
        rows[metadata.name] = {
            "gradient_present": 1.0,
            "gradient_l2_norm": finite_float(torch.linalg.vector_norm(detached)),
            "gradient_rms": finite_float(torch.sqrt(torch.mean(detached.square()))),
            "gradient_max_abs": finite_float(detached.abs().max()),
        }
    return rows


@torch.no_grad()
def coefficient_utilization_report(model: TrainingSheetGPT) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    depth_order = int(model.config.depth_order)
    for metadata in model.trajectory.metadata:
        coefficient = model.trajectory.coefficients[metadata.name].detach().float()
        depth_axis, order_axes = _coefficient_axis_plan(coefficient, depth_order)
        depth_fractions = _axis_energy_fraction(coefficient, depth_axis)
        row_fractions = _combined_axis_energy_fraction(coefficient, order_axes)
        rows[metadata.name] = {
            "semantic_type": metadata.semantic_type,
            "shape": list(coefficient.shape),
            "coefficient_rms": finite_float(torch.sqrt(torch.mean(coefficient.square()))),
            "coefficient_l2_norm": finite_float(torch.linalg.vector_norm(coefficient)),
            "coefficient_max_abs": finite_float(coefficient.abs().max()),
            "nonzero_fraction": finite_float(
                torch.count_nonzero(coefficient).float() / coefficient.numel()
            ),
            "depth_order_energy_fraction": list(depth_fractions),
            "row_order_energy_fraction": list(row_fractions),
            "high_depth_order_energy_fraction": high_order_fraction(depth_fractions),
            "high_row_order_energy_fraction": high_order_fraction(row_fractions),
        }
    return rows


@torch.no_grad()
def generated_weight_report(
    model: TrainingSheetGPT,
    *,
    layer_indices: Optional[Iterable[int]] = None,
    families: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    if layer_indices is None:
        layer_indices = (0, model.config.n_layer // 2, model.config.n_layer - 1)
    layers = tuple(dict.fromkeys(int(index) for index in layer_indices))
    if any(index < 0 or index >= model.config.n_layer for index in layers):
        raise IndexError("generated-weight diagnostic layer is out of range")
    if families is None:
        families = tuple(metadata.name for metadata in model.trajectory.metadata)
    report: Dict[str, Dict[str, Dict[str, float]]] = {}
    for layer_index in layers:
        family_rows: Dict[str, Dict[str, float]] = {}
        for family in tuple(families):
            generated = model.trajectory.materialize(family, layer_index).detach().float()
            family_rows[family] = {
                "mean": finite_float(generated.mean()),
                "standard_deviation": finite_float(generated.std(unbiased=False)),
                "rms": finite_float(torch.sqrt(torch.mean(generated.square()))),
                "maximum_absolute_value": finite_float(generated.abs().max()),
            }
        report[str(layer_index)] = family_rows
    return report


@torch.no_grad()
def stage6_sheet_diagnostics(model: TrainingSheetGPT) -> Dict[str, Any]:
    return {
        "coefficient_utilization": coefficient_utilization_report(model),
        "generated_weights": generated_weight_report(model),
        "compact_state_violations": list(model.compact_state_violations()),
    }


__all__ = [
    "coefficient_utilization_report",
    "generated_weight_report",
    "gradient_report",
    "stage6_sheet_diagnostics",
]
# ^^^ THOG
