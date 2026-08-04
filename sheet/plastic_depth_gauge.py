# vvv THOG
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from .basis import differentiable_chebyshev_first_kind_basis


@dataclass(frozen=True)
class PlasticDepthGaugeError:
    max_absolute_error: float
    max_relative_error: float


def _validate_square_float64_matrix(name: str, value: Tensor) -> None:
    if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] < 1:
        raise ValueError(
            f"{name} must be a non-empty square matrix; got shape {tuple(value.shape)}"
        )
    if value.dtype != torch.float64:
        raise ValueError(f"{name} must use torch.float64; got {value.dtype}")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")


def chebyshev_gauss_nodes(
    order: int,
    *,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Return P interior Chebyshev nodes for a degree-(P-1) transform solve."""

    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValueError(f"order must be a positive integer; got {order!r}")
    indices = torch.arange(order, dtype=torch.float64, device=device)
    return torch.cos(math.pi * (indices + 0.5) / order)


def stabilized_chebyshev_affine_change_of_chart(
    inverse_r: Tensor,
    *,
    old_from_new_scale: float,
    old_from_new_shift: float,
) -> Tensor:
    """Map fixed QR-stabilised coefficients from a new affine chart to an old field.

    The returned matrix M satisfies c_new = M @ c_old when
    z_old = old_from_new_scale * z_new + old_from_new_shift and both charts use
    the same fixed QR-stabilised coefficient coordinate system.
    """

    _validate_square_float64_matrix("inverse_r", inverse_r)
    scale = float(old_from_new_scale)
    shift = float(old_from_new_shift)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            "old_from_new_scale must be finite and positive; "
            f"got {old_from_new_scale!r}"
        )
    if not math.isfinite(shift):
        raise ValueError(
            "old_from_new_shift must be finite; "
            f"got {old_from_new_shift!r}"
        )

    order = int(inverse_r.shape[0])

    # vvv THOG build the affine composition exactly in the ordinary Chebyshev
    # coefficient basis; this avoids a Vandermonde solve and its avoidable
    # extrapolation error when the new chart extends beyond the old endpoint.
    standard_transform = torch.zeros(
        (order, order),
        dtype=torch.float64,
        device=inverse_r.device,
    )
    standard_transform[0, 0] = 1.0
    if order > 1:
        standard_transform[0, 1] = shift
        standard_transform[1, 1] = scale

    def multiply_by_x(series: Tensor) -> Tensor:
        result = torch.zeros_like(series)
        if order > 1:
            result[1] += series[0]
        for index in range(1, order):
            result[index - 1] += 0.5 * series[index]
            if index + 1 < order:
                result[index + 1] += 0.5 * series[index]
        return result

    for degree in range(2, order):
        previous = standard_transform[:, degree - 1]
        previous_previous = standard_transform[:, degree - 2]
        affine_times_previous = (
            scale * multiply_by_x(previous) + shift * previous
        )
        standard_transform[:, degree] = (
            2.0 * affine_times_previous - previous_previous
        )

    # c_standard = inverse_r @ c_stabilised.  Convert the exact standard-basis
    # composition back into the fixed QR-stabilised coefficient coordinates.
    matrix = torch.linalg.solve(
        inverse_r,
        standard_transform @ inverse_r,
    )
    # ^^^ THOG
    if not bool(torch.isfinite(matrix).all().item()):
        raise RuntimeError("affine Chebyshev change-of-chart matrix is non-finite")
    return matrix


def apply_depth_coefficient_transform(
    coefficients: Tensor,
    transform: Tensor,
    *,
    depth_axis: int = -1,
) -> Tensor:
    """Apply a P-by-P coefficient transform along one tensor axis."""

    _validate_square_float64_matrix("transform", transform)
    if not coefficients.is_floating_point():
        raise ValueError(
            f"coefficients must use a floating dtype; got {coefficients.dtype}"
        )
    resolved_axis = depth_axis if depth_axis >= 0 else coefficients.ndim + depth_axis
    if resolved_axis < 0 or resolved_axis >= coefficients.ndim:
        raise ValueError(
            f"depth_axis {depth_axis} is invalid for shape {tuple(coefficients.shape)}"
        )
    order = int(transform.shape[0])
    if coefficients.shape[resolved_axis] != order:
        raise ValueError(
            "coefficient depth axis must match transform order; "
            f"got shape={tuple(coefficients.shape)}, depth_axis={depth_axis}, "
            f"order={order}"
        )

    moved = coefficients.movedim(resolved_axis, -1)
    transformed = torch.matmul(
        moved.to(dtype=torch.float64),
        transform.transpose(0, 1),
    )
    return transformed.movedim(-1, resolved_axis)



# vvv THOG chunked stored-dtype transform avoids a whole-parameter float64 temporary
def apply_depth_coefficient_transform_chunked(
    coefficients: Tensor,
    transform: Tensor,
    *,
    depth_axis: int = -1,
    output_dtype: Optional[torch.dtype] = None,
    maximum_series_per_chunk: int = 65536,
) -> Tensor:
    _validate_square_float64_matrix("transform", transform)
    if not coefficients.is_floating_point():
        raise ValueError(
            f"coefficients must use a floating dtype; got {coefficients.dtype}"
        )
    if isinstance(maximum_series_per_chunk, bool) or maximum_series_per_chunk <= 0:
        raise ValueError(
            "maximum_series_per_chunk must be a positive integer; "
            f"got {maximum_series_per_chunk!r}"
        )
    resolved_axis = depth_axis if depth_axis >= 0 else coefficients.ndim + depth_axis
    if resolved_axis < 0 or resolved_axis >= coefficients.ndim:
        raise ValueError(
            f"depth_axis {depth_axis} is invalid for shape {tuple(coefficients.shape)}"
        )
    order = int(transform.shape[0])
    if coefficients.shape[resolved_axis] != order:
        raise ValueError(
            "coefficient depth axis must match transform order; "
            f"got shape={tuple(coefficients.shape)}, depth_axis={depth_axis}, "
            f"order={order}"
        )

    target_dtype = coefficients.dtype if output_dtype is None else output_dtype
    moved = coefficients.movedim(resolved_axis, -1)
    flattened = moved.reshape(-1, order)
    transformed_flattened = torch.empty(
        flattened.shape,
        dtype=target_dtype,
        device=coefficients.device,
    )
    runtime_transform = transform.to(device=coefficients.device, dtype=torch.float64)
    for start in range(0, flattened.shape[0], maximum_series_per_chunk):
        stop = min(start + maximum_series_per_chunk, flattened.shape[0])
        transformed = (
            flattened[start:stop].to(dtype=torch.float64)
            @ runtime_transform.transpose(0, 1)
        )
        transformed_flattened[start:stop].copy_(transformed.to(dtype=target_dtype))
    return transformed_flattened.reshape(moved.shape).movedim(-1, resolved_axis)
# ^^^ THOG

def gauge_error(reference: Tensor, candidate: Tensor) -> PlasticDepthGaugeError:
    if reference.shape != candidate.shape:
        raise ValueError(
            "reference and candidate must have identical shapes; "
            f"got {tuple(reference.shape)} and {tuple(candidate.shape)}"
        )
    reference64 = reference.to(dtype=torch.float64)
    candidate64 = candidate.to(dtype=torch.float64)
    difference = torch.abs(candidate64 - reference64)
    max_absolute = float(difference.max().item()) if difference.numel() else 0.0
    denominator = torch.clamp(torch.abs(reference64), min=torch.finfo(torch.float64).eps)
    max_relative = (
        float((difference / denominator).max().item()) if difference.numel() else 0.0
    )
    return PlasticDepthGaugeError(
        max_absolute_error=max_absolute,
        max_relative_error=max_relative,
    )


__all__ = [
    "PlasticDepthGaugeError",
    "apply_depth_coefficient_transform",
    "apply_depth_coefficient_transform_chunked",
    "chebyshev_gauss_nodes",
    "gauge_error",
    "stabilized_chebyshev_affine_change_of_chart",
]
# ^^^ THOG
