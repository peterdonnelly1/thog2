# vvv THOG
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class ProjectionError:
    max_abs: float
    rmse: float
    frobenius: float
    relative_frobenius: float


def _canonical_sampled_sheets(sampled_sheets: Tensor) -> Tuple[Tensor, bool]:
    if sampled_sheets.ndim == 2:
        return sampled_sheets.unsqueeze(1), True
    if sampled_sheets.ndim == 3:
        return sampled_sheets, False
    raise ValueError(
        "sampled_sheets must have shape [layers, row_width] or "
        f"[layers, output_rows, row_width]; got {tuple(sampled_sheets.shape)}"
    )


def _validate_projection_geometry(
    sampled_sheets: Tensor,
    depth_basis: Tensor,
    row_basis: Tensor,
) -> Tuple[Tensor, bool]:
    canonical, squeezed = _canonical_sampled_sheets(sampled_sheets)
    if depth_basis.ndim != 2:
        raise ValueError(f"depth_basis must be two-dimensional; got {tuple(depth_basis.shape)}")
    if row_basis.ndim != 2:
        raise ValueError(f"row_basis must be two-dimensional; got {tuple(row_basis.shape)}")
    if canonical.shape[0] != depth_basis.shape[0]:
        raise ValueError(
            f"layer count mismatch: sampled={canonical.shape[0]}, basis={depth_basis.shape[0]}"
        )
    if canonical.shape[2] != row_basis.shape[0]:
        raise ValueError(
            f"row width mismatch: sampled={canonical.shape[2]}, basis={row_basis.shape[0]}"
        )
    if canonical.device != depth_basis.device or canonical.device != row_basis.device:
        raise ValueError("sampled sheets and bases must be on the same device")
    if canonical.dtype != depth_basis.dtype or canonical.dtype != row_basis.dtype:
        raise ValueError("sampled sheets and bases must use the same dtype")
    if not canonical.is_floating_point():
        raise ValueError("sampled_sheets must use a floating dtype")
    if not torch.isfinite(canonical).all():
        raise ValueError("sampled_sheets must be finite")
    return canonical, squeezed


def fit_sampled_sheets(
    sampled_sheets: Tensor,
    depth_basis: Tensor,
    row_basis: Tensor,
) -> Tensor:
    """Return the orthogonal least-squares coefficient tensor [rows, P, Q].

    With discrete orthonormal bases this is the best approximation in sampled
    Frobenius norm. Saturated bases (P=L and Q=C) reconstruct every sampled
    sheet up to floating-point error. Sub-saturated bases cannot guarantee a
    requested epsilon for arbitrary weights; the residual must be measured or
    smoothness assumptions supplied.
    """

    canonical, _ = _validate_projection_geometry(sampled_sheets, depth_basis, row_basis)
    return torch.einsum("lp,lrc,cq->rpq", depth_basis, canonical, row_basis)


def reconstruct_sampled_sheets(
    coefficients: Tensor,
    depth_basis: Tensor,
    row_basis: Tensor,
    *,
    squeeze_single_row: bool = False,
) -> Tensor:
    if coefficients.ndim != 3:
        raise ValueError(f"coefficients must have shape [rows, P, Q]; got {tuple(coefficients.shape)}")
    if coefficients.shape[1] != depth_basis.shape[1]:
        raise ValueError("coefficient depth order does not match depth basis")
    if coefficients.shape[2] != row_basis.shape[1]:
        raise ValueError("coefficient row order does not match row basis")
    if coefficients.device != depth_basis.device or coefficients.device != row_basis.device:
        raise ValueError("coefficients and bases must be on the same device")
    if coefficients.dtype != depth_basis.dtype or coefficients.dtype != row_basis.dtype:
        raise ValueError("coefficients and bases must use the same dtype")
    reconstructed = torch.einsum("lp,rpq,cq->lrc", depth_basis, coefficients, row_basis)
    if squeeze_single_row:
        if reconstructed.shape[1] != 1:
            raise ValueError("squeeze_single_row requires exactly one output row")
        return reconstructed[:, 0, :]
    return reconstructed


def project_sampled_sheets(
    sampled_sheets: Tensor,
    depth_basis: Tensor,
    row_basis: Tensor,
) -> Tensor:
    canonical, squeezed = _validate_projection_geometry(sampled_sheets, depth_basis, row_basis)
    coefficients = fit_sampled_sheets(canonical, depth_basis, row_basis)
    return reconstruct_sampled_sheets(
        coefficients,
        depth_basis,
        row_basis,
        squeeze_single_row=squeezed,
    )


def projection_error(reference: Tensor, approximation: Tensor) -> ProjectionError:
    if reference.shape != approximation.shape:
        raise ValueError(
            f"shape mismatch: reference={tuple(reference.shape)}, approximation={tuple(approximation.shape)}"
        )
    difference = approximation - reference
    frobenius = float(torch.linalg.vector_norm(difference).item())
    reference_frobenius = float(torch.linalg.vector_norm(reference).item())
    relative = frobenius / reference_frobenius if reference_frobenius > 0.0 else frobenius
    return ProjectionError(
        max_abs=float(torch.max(torch.abs(difference)).item()),
        rmse=float(torch.sqrt(torch.mean(difference.square())).item()),
        frobenius=frobenius,
        relative_frobenius=relative,
    )


def is_within_epsilon(
    reference: Tensor,
    approximation: Tensor,
    epsilon: float,
    *,
    metric: str = "max_abs",
) -> bool:
    if epsilon < 0.0:
        raise ValueError(f"epsilon must be non-negative; got {epsilon}")
    errors = projection_error(reference, approximation)
    if metric == "max_abs":
        value = errors.max_abs
    elif metric == "rmse":
        value = errors.rmse
    elif metric == "relative_frobenius":
        value = errors.relative_frobenius
    else:
        raise ValueError(f"unsupported epsilon metric: {metric}")
    return value <= epsilon
# ^^^ THOG
