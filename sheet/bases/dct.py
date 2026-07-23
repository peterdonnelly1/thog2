# vvv THOG
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from .protocol import BasisDefinition, BasisKernel, DeviceLike, validate_floating_dtype, validate_positive_integer


BASIS_FAMILY_DCT = "dct"
DCT_BASIS_VERSION = "dct_ii_orthonormal_v1"
BASIS_ARTIFACT_TAG_DCT = "DCT"


def dct_sample_indices(sample_count: int, *, dtype: torch.dtype = torch.float64, device: Optional[DeviceLike] = None) -> Tensor:
    validate_positive_integer("sample_count", sample_count)
    validate_floating_dtype(dtype)
    target_device = torch.device("cpu" if device is None else device)
    return torch.arange(sample_count, dtype=dtype, device=target_device)


def dct_ii_orthonormal_raw_basis(sample_indices: Tensor, order: int) -> Tensor:
    validate_positive_integer("order", order)
    if sample_indices.ndim != 1:
        raise ValueError(f"sample_indices must be one-dimensional; got shape {tuple(sample_indices.shape)}")
    if sample_indices.numel() == 0:
        raise ValueError("sample_indices must contain at least one sample")
    if not sample_indices.is_floating_point():
        raise ValueError(f"sample_indices must use a floating dtype; got {sample_indices.dtype}")
    if not torch.isfinite(sample_indices).all():
        raise ValueError("sample_indices must be finite")
    sample_count = sample_indices.numel()
    column_indices = torch.arange(order, dtype=sample_indices.dtype, device=sample_indices.device)
    angles = math.pi * (sample_indices.unsqueeze(1) + 0.5) * column_indices.unsqueeze(0) / float(sample_count)
    basis = torch.cos(angles)
    basis[:, 0] *= math.sqrt(1.0 / float(sample_count))
    if order > 1:
        basis[:, 1:] *= math.sqrt(2.0 / float(sample_count))
    return basis


class DctIiOrthonormalBasisKernel(BasisKernel):
    def __init__(self) -> None:
        super().__init__(
            basis_family=BASIS_FAMILY_DCT,
            basis_version=DCT_BASIS_VERSION,
            coordinate_policy="integer_sample_index_half_shifted_dct_ii_v1",
            stabilization_policy="closed_form_dct_ii_orthonormal_no_qr_v1",
        )

    def coordinates(self, sample_count: int, *, dtype: torch.dtype = torch.float64, device: Optional[DeviceLike] = None) -> Tensor:
        return dct_sample_indices(sample_count, dtype=dtype, device=device)

    def raw_basis(self, coordinates: Tensor, order: int) -> Tensor:
        return dct_ii_orthonormal_raw_basis(coordinates, order)

    def stabilize(self, raw_basis: Tensor) -> Tensor:
        if raw_basis.ndim != 2:
            raise ValueError(f"raw_basis must be two-dimensional; got shape {tuple(raw_basis.shape)}")
        if not raw_basis.is_floating_point():
            raise ValueError(f"raw_basis must use a floating dtype; got {raw_basis.dtype}")
        if not torch.isfinite(raw_basis).all():
            raise FloatingPointError("DCT basis contains a non-finite value")
        return raw_basis


BASIS_DEFINITION = BasisDefinition(
    family=BASIS_FAMILY_DCT,
    aliases=("dct_ii", "dct_ii_orthonormal"),
    version=DCT_BASIS_VERSION,
    artifact_tag=BASIS_ARTIFACT_TAG_DCT,
    supports_weight_basis=True,
    supports_native_products=False,
    kernel=DctIiOrthonormalBasisKernel(),
)
# ^^^ THOG
