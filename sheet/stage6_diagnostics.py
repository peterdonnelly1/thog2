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
    if coefficient.ndim == 4 and shape[0] == depth_order:
        return 0, (1,)                                                                                                                                        # <<< THOG JPEG-like coefficients are DEPTH x retained-order x physical-group-position x group-instance
    raise ValueError(f"unsupported coefficient diagnostic shape {shape} for depth_order={depth_order}")


def _axis_energy_fraction(coefficient: Tensor, axis: int) -> Tuple[float, ...]:
    reduce_dims = tuple(index for index in range(coefficient.ndim) if index != axis)
    return normalized_energy(coefficient.square().sum(dim=reduce_dims))


def _combined_axis_energy_fraction(coefficient: Tensor, axes: Tuple[int, ...]) -> Tuple[float, ...]:
    if not axes:
        return tuple()
    reduce_dims = tuple(index for index in range(coefficient.ndim) if index not in axes)
    return normalized_energy(coefficient.square().sum(dim=reduce_dims))


# vvv THOG report the actual persistent parameter regions when operational matrix-family metadata no longer maps one-to-one onto coefficients

def _persistent_parameter_rows(
    model: TrainingSheetGPT,
    *,
    include_vectors: bool = True,
) -> Tuple[Tuple[str, str, Tensor], ...]:
    trajectory = model.trajectory
    metadata_by_name = {
        metadata.name: metadata
        for metadata in trajectory.metadata
    }
    metadata_names = tuple(metadata_by_name)
    if all(name in trajectory.coefficients for name in metadata_names):
        return tuple(
            (
                metadata.name,
                metadata.semantic_type,
                trajectory.coefficients[metadata.name],
            )
            for metadata in trajectory.metadata
        )

    rows = [
        (name, "coupled_coefficient_region", parameter)
        for name, parameter in trajectory.coefficients.items()
    ]
    vector_parameters = getattr(trajectory, "vector_parameters", None)
    if include_vectors and vector_parameters is not None:
        for name, parameter in vector_parameters.items():
            metadata = metadata_by_name.get(name)
            semantic_type = (
                metadata.semantic_type
                if metadata is not None
                else "vector_parameter"
            )
            rows.append((name, semantic_type, parameter))
    return tuple(rows)


def _diagnostic_depth_order(model: TrainingSheetGPT) -> int:
    plan = getattr(model.trajectory, "plan", None)
    orders = getattr(plan, "orders", None)
    if orders is not None and hasattr(orders, "depth"):
        return int(orders.depth)
    return int(model.config.depth_order)

# vvv THOG preserve the exact replaced executable lines for the nanoGPT source-history contract
# parameter = model.trajectory.coefficients[metadata.name]
# coefficient = model.trajectory.coefficients[metadata.name].detach().float()
# "semantic_type": metadata.semantic_type,
# rows[metadata.name]["depth_order_energy_fraction"] = None
# rows[metadata.name]["row_order_energy_fraction"] = None
# rows[metadata.name]["high_depth_order_energy_fraction"] = None
# rows[metadata.name]["high_row_order_energy_fraction"] = None
# rows[metadata.name]["order_axis_diagnostics_error"] = order_axis_diagnostics_error
# ^^^ THOG

# ^^^ THOG


def gradient_report(model: TrainingSheetGPT) -> Dict[str, Dict[str, float]]:
    rows: Dict[str, Dict[str, float]] = {}
    # vvv THOG HYPERBLOCK operational families share common/attention/MLP parameters, so diagnose those persistent regions rather than indexing them by family name
    # for metadata in model.trajectory.metadata:
    #     parameter = model.trajectory.coefficients[metadata.name]
    #     gradient = parameter.grad
    #     if gradient is None:
    #         rows[metadata.name] = {
    #             "gradient_present": 0.0,
    #             "gradient_l2_norm": 0.0,
    #             "gradient_rms": 0.0,
    #             "gradient_max_abs": 0.0,
    #         }
    #         continue
    #     detached = gradient.detach().float()
    #     rows[metadata.name] = {
    #         "gradient_present": 1.0,
    #         "gradient_l2_norm": finite_float(torch.linalg.vector_norm(detached)),
    #         "gradient_rms": finite_float(torch.sqrt(torch.mean(detached.square()))),
    #         "gradient_max_abs": finite_float(detached.abs().max()),
    #     }
    for name, _, parameter in _persistent_parameter_rows(model):
        gradient = parameter.grad
        if gradient is None:
            rows[name] = {
                "gradient_present": 0.0,
                "gradient_l2_norm": 0.0,
                "gradient_rms": 0.0,
                "gradient_max_abs": 0.0,
            }
            continue
        detached = gradient.detach().float()
        rows[name] = {
            "gradient_present": 1.0,
            "gradient_l2_norm": finite_float(torch.linalg.vector_norm(detached)),
            "gradient_rms": finite_float(torch.sqrt(torch.mean(detached.square()))),
            "gradient_max_abs": finite_float(detached.abs().max()),
        }
    # ^^^ THOG
    return rows


@torch.no_grad()
def coefficient_utilization_report(model: TrainingSheetGPT) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    # vvv THOG use the HYPERBLOCK retained DEPTH order and persistent coefficient regions while preserving the legacy metadata-driven path through the shared helper
    # depth_order = int(model.config.depth_order)
    # for metadata in model.trajectory.metadata:
    #     coefficient = model.trajectory.coefficients[metadata.name].detach().float()
    depth_order = _diagnostic_depth_order(model)
    for name, semantic_type, parameter in _persistent_parameter_rows(
        model,
        include_vectors=False,
    ):
        coefficient = parameter.detach().float()
        # ^^^ THOG
        # vvv THOG unsupported coefficient layouts still report generic utilization instead of aborting a completed run
        # depth_axis, order_axes = _coefficient_axis_plan(coefficient, depth_order)
        # depth_fractions = _axis_energy_fraction(coefficient, depth_axis)
        # row_fractions = _combined_axis_energy_fraction(coefficient, order_axes)
        order_axis_diagnostics_error: Optional[str] = None
        try:
            depth_axis, order_axes = _coefficient_axis_plan(coefficient, depth_order)
            depth_fractions = _axis_energy_fraction(coefficient, depth_axis)
            row_fractions = _combined_axis_energy_fraction(coefficient, order_axes)
        except ValueError as error:
            depth_fractions = tuple()
            row_fractions = tuple()
            order_axis_diagnostics_error = str(error)
        # ^^^ THOG
        # vvv THOG key utilization by the persistent parameter name; operational family names remain reserved for generated-weight diagnostics
        # rows[metadata.name] = {
        #     "semantic_type": metadata.semantic_type,
        rows[name] = {
            "semantic_type": semantic_type,
            # ^^^ THOG
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
        # vvv THOG never encode an unsupported order-axis diagnostic as a misleading zero-energy result
        # rows[metadata.name]["order_axis_diagnostics_supported"] = order_axis_diagnostics_error is None
        # if order_axis_diagnostics_error is not None:
        #     rows[metadata.name]["depth_order_energy_fraction"] = None
        #     rows[metadata.name]["row_order_energy_fraction"] = None
        #     rows[metadata.name]["high_depth_order_energy_fraction"] = None
        #     rows[metadata.name]["high_row_order_energy_fraction"] = None
        #     rows[metadata.name]["order_axis_diagnostics_error"] = order_axis_diagnostics_error
        rows[name]["order_axis_diagnostics_supported"] = order_axis_diagnostics_error is None
        if order_axis_diagnostics_error is not None:
            rows[name]["depth_order_energy_fraction"] = None
            rows[name]["row_order_energy_fraction"] = None
            rows[name]["high_depth_order_energy_fraction"] = None
            rows[name]["high_row_order_energy_fraction"] = None
            rows[name]["order_axis_diagnostics_error"] = order_axis_diagnostics_error
        # ^^^ THOG
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
