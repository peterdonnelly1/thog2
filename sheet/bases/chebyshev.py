# vvv THOG
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .protocol import BasisDefinition, BasisKernel, DeviceLike, deterministic_reduced_qr_positive_diagonal, validate_floating_dtype, validate_positive_integer


BASIS_FAMILY_CHEBYSHEV = "chebyshev"
CHEBYSHEV_BASIS_VERSION = "chebyshev_first_kind_qr_v1"
BASIS_ARTIFACT_TAG_CHEBYSHEV = "CHEBY"
SINGLE_POINT_COORDINATE = 0.0


def chebyshev_coordinates(sample_count: int, *, dtype: torch.dtype = torch.float64, device: Optional[DeviceLike] = None) -> Tensor:
    validate_positive_integer("sample_count", sample_count)
    validate_floating_dtype(dtype)
    target_device = torch.device("cpu" if device is None else device)
    if sample_count == 1:
        return torch.tensor([SINGLE_POINT_COORDINATE], dtype=dtype, device=target_device)
    return torch.linspace(-1.0, 1.0, sample_count, dtype=dtype, device=target_device)


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


class ChebyshevQrBasisKernel(BasisKernel):
    def __init__(self) -> None:
        super().__init__(
            basis_family=BASIS_FAMILY_CHEBYSHEV,
            basis_version=CHEBYSHEV_BASIS_VERSION,
            coordinate_policy="linear_minus_one_to_one_single_point_zero_v1",
            stabilization_policy="deterministic_reduced_qr_positive_diagonal_v1",
        )

    def coordinates(self, sample_count: int, *, dtype: torch.dtype = torch.float64, device: Optional[DeviceLike] = None) -> Tensor:
        return chebyshev_coordinates(sample_count, dtype=dtype, device=device)

    def raw_basis(self, coordinates: Tensor, order: int) -> Tensor:
        return chebyshev_raw_basis(coordinates, order)

    def stabilize(self, raw_basis: Tensor) -> Tensor:
        stabilized_basis, _ = deterministic_reduced_qr_positive_diagonal(raw_basis)
        return stabilized_basis


BASIS_DEFINITION = BasisDefinition(
    family=BASIS_FAMILY_CHEBYSHEV,
    aliases=("cheby", "chebyshev_first_kind_qr"),
    version=CHEBYSHEV_BASIS_VERSION,
    artifact_tag=BASIS_ARTIFACT_TAG_CHEBYSHEV,
    supports_weight_basis=True,
    supports_native_products=False,
    kernel=ChebyshevQrBasisKernel(),
)
# ^^^ THOG
