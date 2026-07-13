# vvv THOG
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Union

import torch
from torch import Tensor


BASIS_FAMILY_CHEBYSHEV = "chebyshev"
BASIS_FAMILY_DCT = "dct"
CHEBYSHEV_BASIS_VERSION = "chebyshev_first_kind_qr_v1"
DCT_BASIS_VERSION = "dct_ii_orthonormal_v1"
SINGLE_POINT_COORDINATE = 0.0

DeviceLike = Union[str, torch.device]


def validate_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")


def validate_floating_dtype(dtype: torch.dtype) -> None:
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError(f"dtype must be floating point; got {dtype}")


def normalize_basis_family(basis_family: str) -> str:
    if not isinstance(basis_family, str) or not basis_family.strip():
        raise ValueError(f"basis_family must be a non-empty string; got {basis_family!r}")
    normalized = basis_family.strip().lower()
    aliases = {
        BASIS_FAMILY_CHEBYSHEV: BASIS_FAMILY_CHEBYSHEV,
        "cheby": BASIS_FAMILY_CHEBYSHEV,
        "chebyshev_first_kind_qr": BASIS_FAMILY_CHEBYSHEV,
        CHEBYSHEV_BASIS_VERSION: BASIS_FAMILY_CHEBYSHEV,
        BASIS_FAMILY_DCT: BASIS_FAMILY_DCT,
        "dct_ii": BASIS_FAMILY_DCT,
        "dct_ii_orthonormal": BASIS_FAMILY_DCT,
        DCT_BASIS_VERSION: BASIS_FAMILY_DCT,
    }
    return aliases.get(normalized, normalized)


def chebyshev_coordinates(sample_count: int, *, dtype: torch.dtype = torch.float64, device: Optional[DeviceLike] = None) -> Tensor:
    validate_positive_integer("sample_count", sample_count)
    validate_floating_dtype(dtype)
    target_device = torch.device("cpu" if device is None else device)
    if sample_count == 1:
        return torch.tensor([SINGLE_POINT_COORDINATE], dtype=dtype, device=target_device)
    return torch.linspace(-1.0, 1.0, sample_count, dtype=dtype, device=target_device)


def dct_sample_indices(sample_count: int, *, dtype: torch.dtype = torch.float64, device: Optional[DeviceLike] = None) -> Tensor:
    validate_positive_integer("sample_count", sample_count)
    validate_floating_dtype(dtype)
    target_device = torch.device("cpu" if device is None else device)
    return torch.arange(sample_count, dtype=dtype, device=target_device)


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


def deterministic_reduced_qr_positive_diagonal(raw_basis: Tensor) -> tuple[Tensor, Tensor]:
    if raw_basis.ndim != 2:
        raise ValueError(f"raw_basis must be two-dimensional; got shape {tuple(raw_basis.shape)}")
    row_count, column_count = raw_basis.shape
    if row_count < column_count:
        raise ValueError(f"reduced QR requires rows >= columns; got rows={row_count}, columns={column_count}")
    if not raw_basis.is_floating_point():
        raise ValueError(f"raw_basis must use a floating dtype; got {raw_basis.dtype}")
    if not torch.isfinite(raw_basis).all():
        raise ValueError("raw_basis must be finite")
    q_matrix, r_matrix = torch.linalg.qr(raw_basis, mode="reduced")
    diagonal = torch.diagonal(r_matrix)
    signs = torch.where(diagonal < 0.0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
    q_matrix = q_matrix * signs.unsqueeze(0)
    r_matrix = r_matrix * signs.unsqueeze(1)
    if not torch.isfinite(q_matrix).all() or not torch.isfinite(r_matrix).all():
        raise FloatingPointError("QR stabilization produced a non-finite value")
    return q_matrix, r_matrix


@dataclass(frozen=True)
class BasisKernel:
    basis_family: str
    basis_version: str
    coordinate_policy: str
    stabilization_policy: str

    def coordinates(self, sample_count: int, *, dtype: torch.dtype = torch.float64, device: Optional[DeviceLike] = None) -> Tensor:
        raise NotImplementedError

    def raw_basis(self, coordinates: Tensor, order: int) -> Tensor:
        raise NotImplementedError

    def stabilize(self, raw_basis: Tensor) -> Tensor:
        raise NotImplementedError

    def build(self, sample_count: int, order: int, *, runtime_dtype: torch.dtype = torch.float64, device: Optional[DeviceLike] = None, version: Optional[str] = None) -> Tensor:
        validate_positive_integer("sample_count", sample_count)
        validate_positive_integer("order", order)
        if order > sample_count:
            raise ValueError(f"order must not exceed sample_count; got order={order}, sample_count={sample_count}")
        validate_floating_dtype(runtime_dtype)
        basis_version = self.basis_version if version is None else version
        if not isinstance(basis_version, str) or not basis_version.strip():
            raise ValueError("version must be a non-empty string")
        if basis_version != self.basis_version:
            raise ValueError(f"basis_version mismatch for {self.basis_family}: expected {self.basis_version!r}, got {basis_version!r}")
        coordinates = self.coordinates(sample_count, dtype=torch.float64, device="cpu")
        raw_basis = self.raw_basis(coordinates, order)
        stabilized_basis = self.stabilize(raw_basis)
        target_device = torch.device("cpu" if device is None else device)
        runtime_basis = stabilized_basis.to(device=target_device, dtype=runtime_dtype)
        runtime_basis.requires_grad_(False)
        if runtime_basis.shape != (sample_count, order):
            raise RuntimeError(f"unexpected basis shape {tuple(runtime_basis.shape)}; expected {(sample_count, order)}")
        if not torch.isfinite(runtime_basis).all():
            raise FloatingPointError("runtime basis contains a non-finite value")
        return runtime_basis

    def metadata(self) -> Dict[str, str]:
        return {"basis_family": self.basis_family, "basis_version": self.basis_version, "coordinate_policy": self.coordinate_policy, "stabilization_policy": self.stabilization_policy}


class ChebyshevQrBasisKernel(BasisKernel):
    def __init__(self) -> None:
        super().__init__(basis_family=BASIS_FAMILY_CHEBYSHEV, basis_version=CHEBYSHEV_BASIS_VERSION, coordinate_policy="linear_minus_one_to_one_single_point_zero_v1", stabilization_policy="deterministic_reduced_qr_positive_diagonal_v1")

    def coordinates(self, sample_count: int, *, dtype: torch.dtype = torch.float64, device: Optional[DeviceLike] = None) -> Tensor:
        return chebyshev_coordinates(sample_count, dtype=dtype, device=device)

    def raw_basis(self, coordinates: Tensor, order: int) -> Tensor:
        return chebyshev_raw_basis(coordinates, order)

    def stabilize(self, raw_basis: Tensor) -> Tensor:
        stabilized_basis, _ = deterministic_reduced_qr_positive_diagonal(raw_basis)
        return stabilized_basis


class DctIiOrthonormalBasisKernel(BasisKernel):
    def __init__(self) -> None:
        super().__init__(basis_family=BASIS_FAMILY_DCT, basis_version=DCT_BASIS_VERSION, coordinate_policy="integer_sample_index_half_shifted_dct_ii_v1", stabilization_policy="closed_form_dct_ii_orthonormal_no_qr_v1")

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


_BASIS_KERNELS: Mapping[str, BasisKernel] = {
    BASIS_FAMILY_CHEBYSHEV: ChebyshevQrBasisKernel(),
    BASIS_FAMILY_DCT: DctIiOrthonormalBasisKernel(),
}


def get_basis_kernel(basis_family: str = BASIS_FAMILY_CHEBYSHEV) -> BasisKernel:
    normalized = normalize_basis_family(basis_family)
    try:
        return _BASIS_KERNELS[normalized]
    except KeyError as error:
        raise ValueError(f"unknown basis_family: {basis_family!r}") from error


def basis_kernel_metadata(basis_family: str = BASIS_FAMILY_CHEBYSHEV) -> Dict[str, str]:
    return get_basis_kernel(basis_family).metadata()


def basis_version_for_family(basis_family: str = BASIS_FAMILY_CHEBYSHEV) -> str:
    return get_basis_kernel(basis_family).basis_version
# ^^^ THOG
