# vvv THOG
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from .protocol import BasisDefinition, BasisKernel, DeviceLike, validate_floating_dtype, validate_positive_integer


BASIS_FAMILY_CHEBYSHEV = "chebyshev"
CHEBYSHEV_BASIS_VERSION = "chebyshev_first_kind_roots_v1"
BASIS_ARTIFACT_TAG_CHEBYSHEV = "CHEBY"
SINGLE_POINT_COORDINATE = 0.0


def chebyshev_coordinates(sample_count: int, *, dtype: torch.dtype = torch.float64, device: Optional[DeviceLike] = None) -> Tensor:
    validate_positive_integer("sample_count", sample_count)
    validate_floating_dtype(dtype)
    target_device = torch.device("cpu" if device is None else device)
    if sample_count == 1:
        return torch.tensor([SINGLE_POINT_COORDINATE], dtype=dtype, device=target_device)
    # Ascending roots: T_k(x_i) = (-1)^k cos(pi * (i + 1/2) * k / N).
    # Construct in float64 before casting so low precision cannot distort the nodes.
    angles = math.pi * (torch.arange(sample_count, dtype=torch.float64, device=target_device) + 0.5) / sample_count
    return (-torch.cos(angles)).to(dtype=dtype)


def chebyshev_normalization(sample_count: int, order: int, *, dtype: torch.dtype = torch.float64, device: Optional[DeviceLike] = None) -> Tensor:
    """Column scales for first-kind terms sampled at the N roots of T_N."""
    validate_positive_integer("sample_count", sample_count)
    validate_positive_integer("order", order)
    validate_floating_dtype(dtype)
    if order > sample_count:
        raise ValueError("order must not exceed sample_count")
    scales = torch.full((order,), math.sqrt(2.0 / sample_count), dtype=dtype, device=device)
    scales[0] = math.sqrt(1.0 / sample_count)
    return scales


def chebyshev_raw_basis(coordinates: Tensor, order: int) -> Tensor:
    validate_positive_integer("order", order)
    if coordinates.ndim != 1:
        raise ValueError(f"coordinates must be one-dimensional; got shape {tuple(coordinates.shape)}")
    if coordinates.numel() == 0:
        raise ValueError("coordinates must contain at least one sample")
    if not coordinates.is_floating_point():
        raise ValueError(f"coordinates must use a floating dtype; got {coordinates.dtype}")
    if not torch.isfinite(coordinates).all():
        raise ValueError("coordinates must be finite")
    sample_count = coordinates.numel()
    basis = torch.empty((sample_count, order), dtype=coordinates.dtype, device=coordinates.device)
    basis[:, 0] = 1.0
    if order == 1:
        return basis
    basis[:, 1] = coordinates
    for term_index in range(2, order):
        basis[:, term_index] = 2.0 * coordinates * basis[:, term_index - 1] - basis[:, term_index - 2]
    return basis


class ChebyshevRootBasisKernel(BasisKernel):
    def __init__(self) -> None:
        super().__init__(
            basis_family=BASIS_FAMILY_CHEBYSHEV,
            basis_version=CHEBYSHEV_BASIS_VERSION,
            coordinate_policy="ascending_chebyshev_roots_single_point_zero",
            stabilization_policy="analytic_column_normalization_no_qr",
        )

    def coordinates(self, sample_count: int, *, dtype: torch.dtype = torch.float64, device: Optional[DeviceLike] = None) -> Tensor:
        return chebyshev_coordinates(sample_count, dtype=dtype, device=device)

    def raw_basis(self, coordinates: Tensor, order: int) -> Tensor:
        return chebyshev_raw_basis(coordinates, order)

    def stabilize(self, raw_basis: Tensor) -> Tensor:
        if raw_basis.ndim != 2 or not raw_basis.is_floating_point():
            raise ValueError("raw_basis must be a two-dimensional floating tensor")
        if not torch.isfinite(raw_basis).all():
            raise FloatingPointError("Chebyshev basis contains a non-finite value")
        scales = chebyshev_normalization(*raw_basis.shape, dtype=raw_basis.dtype, device=raw_basis.device)
        basis = raw_basis * scales
        gram = basis.T @ basis
        error = (gram - torch.eye(basis.shape[1], dtype=basis.dtype, device=basis.device)).abs().max()
        tolerance = max(1.0e-10, 8 * torch.finfo(basis.dtype).eps * basis.shape[0])
        if not torch.isfinite(error) or float(error) > tolerance:
            raise ValueError(f"Chebyshev root basis failed orthogonality validation: error={float(error):.6g}")
        return basis


BASIS_DEFINITION = BasisDefinition(
    family=BASIS_FAMILY_CHEBYSHEV,
    aliases=("cheby", "chebyshev_first_kind_roots"),
    version=CHEBYSHEV_BASIS_VERSION,
    artifact_tag=BASIS_ARTIFACT_TAG_CHEBYSHEV,
    supports_weight_basis=True,
    supports_native_products=False,
    kernel=ChebyshevRootBasisKernel(),
)
# ^^^ THOG
