# vvv THOG
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from .basis_kernel import (
    BASIS_FAMILY_CHEBYSHEV,
    CHEBYSHEV_BASIS_VERSION,
    SINGLE_POINT_COORDINATE,
    chebyshev_coordinates,
    chebyshev_raw_basis,
    deterministic_reduced_qr_positive_diagonal,
    validate_floating_dtype,
    validate_positive_integer,
)
from .basis_registry import (
    basis_version_for_family,
    build_registered_basis,
    normalize_registered_basis_family,
)


BASIS_VERSION = CHEBYSHEV_BASIS_VERSION

DeviceLike = Union[str, torch.device]


def _validate_positive_integer(name: str, value: int) -> None:
    validate_positive_integer(name, value)


def _validate_floating_dtype(dtype: torch.dtype) -> None:
    validate_floating_dtype(dtype)


def normalized_coordinates(sample_count: int, *, dtype: torch.dtype = torch.float64, device: Optional[DeviceLike] = None) -> Tensor:
    """Return fixed monotone sample coordinates in [-1, 1].

    A one-point axis is placed at 0.0. This avoids arbitrarily privileging
    either endpoint when the geometry has no direction to express.
    """

    return chebyshev_coordinates(sample_count, dtype=dtype, device=device)


def chebyshev_first_kind_basis(coordinates: Tensor, order: int) -> Tensor:
    """Construct sampled first-kind Chebyshev terms by recurrence."""

    return chebyshev_raw_basis(coordinates, order)


def deterministic_reduced_qr(raw_basis: Tensor) -> Tuple[Tensor, Tensor]:
    """Return reduced QR with a positive diagonal in R.

    Positive diagonal normalization removes the otherwise arbitrary column
    signs returned by QR and makes reconstruction stable across repeated runs.
    """

    return deterministic_reduced_qr_positive_diagonal(raw_basis)


def build_stabilized_basis(sample_count: int, order: int, *, runtime_dtype: torch.dtype = torch.float64, device: Optional[DeviceLike] = None, version: str = BASIS_VERSION, basis_family: str = BASIS_FAMILY_CHEBYSHEV) -> Tensor:
    """Build a registry-selected stabilized basis in float64 on CPU, then cast for runtime use."""

    return build_registered_basis(sample_count, order, runtime_dtype=runtime_dtype, device=device, version=version, basis_family=basis_family)


@dataclass(frozen=True)
class BasisCacheKey:
    sample_count: int
    order: int
    basis_family: str
    version: str
    device_type: str
    device_index: Optional[int]
    dtype: torch.dtype


class BasisCache:
    """In-memory cache keyed by family, geometry, version, device, and runtime dtype."""

    def __init__(self) -> None:
        self._cache: Dict[BasisCacheKey, Tensor] = {}

    @staticmethod
    def make_key(sample_count: int, order: int, *, runtime_dtype: torch.dtype, device: Optional[DeviceLike], version: str, basis_family: str = BASIS_FAMILY_CHEBYSHEV) -> BasisCacheKey:
        target_device = torch.device("cpu" if device is None else device)
        canonical_family = normalize_registered_basis_family(basis_family)
        expected_version = basis_version_for_family(canonical_family)
        if version != expected_version:
            raise ValueError(f"basis_version mismatch for {canonical_family}: expected {expected_version!r}, got {version!r}")
        return BasisCacheKey(sample_count=sample_count, order=order, basis_family=canonical_family, version=version, device_type=target_device.type, device_index=target_device.index, dtype=runtime_dtype)

    def get(self, sample_count: int, order: int, *, runtime_dtype: torch.dtype = torch.float64, device: Optional[DeviceLike] = None, version: str = BASIS_VERSION, basis_family: str = BASIS_FAMILY_CHEBYSHEV) -> Tensor:
        key = self.make_key(sample_count, order, runtime_dtype=runtime_dtype, device=device, version=version, basis_family=basis_family)
        if key not in self._cache:
            self._cache[key] = build_stabilized_basis(sample_count, order, runtime_dtype=runtime_dtype, device=device, version=version, basis_family=basis_family)
        return self._cache[key]

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


class BasisOwner(nn.Module):
    """Own fixed bases as non-persistent, non-trainable module buffers."""

    def __init__(self, cache: Optional[BasisCache] = None) -> None:
        super().__init__()
        self._basis_cache = BasisCache() if cache is None else cache

    def add_basis(self, name: str, sample_count: int, order: int, *, runtime_dtype: torch.dtype = torch.float64, device: Optional[DeviceLike] = None, version: str = BASIS_VERSION, basis_family: str = BASIS_FAMILY_CHEBYSHEV) -> Tensor:
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"basis name must be a valid identifier; got {name!r}")
        if hasattr(self, name):
            raise ValueError(f"basis name already exists: {name}")
        basis = self._basis_cache.get(sample_count, order, runtime_dtype=runtime_dtype, device=device, version=version, basis_family=basis_family)
        self.register_buffer(name, basis, persistent=False)
        return basis


def orthonormality_max_error(basis: Tensor) -> float:
    if basis.ndim != 2:
        raise ValueError(f"basis must be two-dimensional; got shape {tuple(basis.shape)}")
    identity = torch.eye(basis.shape[1], dtype=basis.dtype, device=basis.device)
    gram = basis.transpose(0, 1) @ basis
    return float(torch.max(torch.abs(gram - identity)).item())


def basis_sha256(basis: Tensor) -> str:
    contiguous = basis.detach().to(device="cpu").contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def estimated_peak_tensor_bytes(sample_count: int, order: int, *, runtime_dtype: torch.dtype = torch.float64) -> int:
    """Conservative tensor-only estimate for coordinates, QR, cast, and Gram work."""

    _validate_positive_integer("sample_count", sample_count)
    _validate_positive_integer("order", order)
    if order > sample_count:
        raise ValueError(f"order must not exceed sample_count; got order={order}, sample_count={sample_count}")
    _validate_floating_dtype(runtime_dtype)
    float64_bytes = torch.tensor([], dtype=torch.float64).element_size()
    runtime_bytes = torch.tensor([], dtype=runtime_dtype).element_size()
    coordinate_bytes = sample_count * float64_bytes
    raw_bytes = sample_count * order * float64_bytes
    q_bytes = sample_count * order * float64_bytes
    r_bytes = order * order * float64_bytes
    runtime_basis_bytes = sample_count * order * runtime_bytes
    gram_bytes = order * order * runtime_bytes
    return coordinate_bytes + raw_bytes + q_bytes + r_bytes + runtime_basis_bytes + gram_bytes
# ^^^ THOG
