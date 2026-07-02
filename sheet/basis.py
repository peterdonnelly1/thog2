# vvv THOG
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn


BASIS_VERSION = "chebyshev_first_kind_qr_v1"
SINGLE_POINT_COORDINATE = 0.0

DeviceLike = Union[str, torch.device]


def _validate_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")


def _validate_floating_dtype(dtype: torch.dtype) -> None:
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError(f"dtype must be floating point; got {dtype}")


def normalized_coordinates(
    sample_count: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: Optional[DeviceLike] = None,
) -> Tensor:
    """Return fixed monotone sample coordinates in [-1, 1].

    A one-point axis is placed at 0.0. This avoids arbitrarily privileging
    either endpoint when the geometry has no direction to express.
    """

    _validate_positive_integer("sample_count", sample_count)
    _validate_floating_dtype(dtype)
    target_device = torch.device("cpu" if device is None else device)
    if sample_count == 1:
        return torch.tensor([SINGLE_POINT_COORDINATE], dtype=dtype, device=target_device)
    return torch.linspace(-1.0, 1.0, sample_count, dtype=dtype, device=target_device)


def chebyshev_first_kind_basis(coordinates: Tensor, order: int) -> Tensor:
    """Construct sampled first-kind Chebyshev terms by recurrence."""

    _validate_positive_integer("order", order)
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
        basis[:, term_index] = (
            2.0 * coordinates * basis[:, term_index - 1] - basis[:, term_index - 2]
        )
    return basis


def deterministic_reduced_qr(raw_basis: Tensor) -> Tuple[Tensor, Tensor]:
    """Return reduced QR with a positive diagonal in R.

    Positive diagonal normalization removes the otherwise arbitrary column
    signs returned by QR and makes reconstruction stable across repeated runs.
    """

    if raw_basis.ndim != 2:
        raise ValueError(f"raw_basis must be two-dimensional; got shape {tuple(raw_basis.shape)}")
    row_count, column_count = raw_basis.shape
    if row_count < column_count:
        raise ValueError(
            f"reduced QR requires rows >= columns; got rows={row_count}, columns={column_count}"
        )
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


def build_stabilized_basis(
    sample_count: int,
    order: int,
    *,
    runtime_dtype: torch.dtype = torch.float64,
    device: Optional[DeviceLike] = None,
    version: str = BASIS_VERSION,
) -> Tensor:
    """Build a stabilized basis in float64 on CPU, then cast for runtime use."""

    _validate_positive_integer("sample_count", sample_count)
    _validate_positive_integer("order", order)
    if order > sample_count:
        raise ValueError(
            f"order must not exceed sample_count; got order={order}, sample_count={sample_count}"
        )
    _validate_floating_dtype(runtime_dtype)
    if not isinstance(version, str) or not version.strip():
        raise ValueError("version must be a non-empty string")

    coordinates = normalized_coordinates(sample_count, dtype=torch.float64, device="cpu")
    raw_basis = chebyshev_first_kind_basis(coordinates, order)
    stabilized_basis, _ = deterministic_reduced_qr(raw_basis)

    target_device = torch.device("cpu" if device is None else device)
    runtime_basis = stabilized_basis.to(device=target_device, dtype=runtime_dtype)
    runtime_basis.requires_grad_(False)
    if runtime_basis.shape != (sample_count, order):
        raise RuntimeError(
            f"unexpected basis shape {tuple(runtime_basis.shape)}; expected {(sample_count, order)}"
        )
    if not torch.isfinite(runtime_basis).all():
        raise FloatingPointError("runtime basis contains a non-finite value")
    return runtime_basis


@dataclass(frozen=True)
class BasisCacheKey:
    sample_count: int
    order: int
    version: str
    device_type: str
    device_index: Optional[int]
    dtype: torch.dtype


class BasisCache:
    """In-memory cache keyed by geometry, version, device, and runtime dtype."""

    def __init__(self) -> None:
        self._cache: Dict[BasisCacheKey, Tensor] = {}

    @staticmethod
    def make_key(
        sample_count: int,
        order: int,
        *,
        runtime_dtype: torch.dtype,
        device: Optional[DeviceLike],
        version: str,
    ) -> BasisCacheKey:
        target_device = torch.device("cpu" if device is None else device)
        return BasisCacheKey(
            sample_count=sample_count,
            order=order,
            version=version,
            device_type=target_device.type,
            device_index=target_device.index,
            dtype=runtime_dtype,
        )

    def get(
        self,
        sample_count: int,
        order: int,
        *,
        runtime_dtype: torch.dtype = torch.float64,
        device: Optional[DeviceLike] = None,
        version: str = BASIS_VERSION,
    ) -> Tensor:
        key = self.make_key(
            sample_count,
            order,
            runtime_dtype=runtime_dtype,
            device=device,
            version=version,
        )
        if key not in self._cache:
            self._cache[key] = build_stabilized_basis(
                sample_count,
                order,
                runtime_dtype=runtime_dtype,
                device=device,
                version=version,
            )
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

    def add_basis(
        self,
        name: str,
        sample_count: int,
        order: int,
        *,
        runtime_dtype: torch.dtype = torch.float64,
        device: Optional[DeviceLike] = None,
        version: str = BASIS_VERSION,
    ) -> Tensor:
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"basis name must be a valid identifier; got {name!r}")
        if hasattr(self, name):
            raise ValueError(f"basis name already exists: {name}")
        basis = self._basis_cache.get(
            sample_count,
            order,
            runtime_dtype=runtime_dtype,
            device=device,
            version=version,
        )
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


def estimated_peak_tensor_bytes(
    sample_count: int,
    order: int,
    *,
    runtime_dtype: torch.dtype = torch.float64,
) -> int:
    """Conservative tensor-only estimate for coordinates, QR, cast, and Gram work."""

    _validate_positive_integer("sample_count", sample_count)
    _validate_positive_integer("order", order)
    if order > sample_count:
        raise ValueError(
            f"order must not exceed sample_count; got order={order}, sample_count={sample_count}"
        )
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
