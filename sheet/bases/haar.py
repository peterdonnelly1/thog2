# vvv THOG
from __future__ import annotations

import math
from collections import deque
from typing import Deque, Optional, Tuple

import torch
from torch import Tensor

from .protocol import BasisDefinition, BasisKernel, DeviceLike, validate_floating_dtype, validate_positive_integer


BASIS_FAMILY_HAAR = "haar"
HAAR_BASIS_VERSION = "haar_balanced_binary_orthonormal_v1"
BASIS_ARTIFACT_TAG_HAAR = "HAAR"


def haar_sample_indices(
    sample_count: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: Optional[DeviceLike] = None,
) -> Tensor:
    validate_positive_integer("sample_count", sample_count)
    validate_floating_dtype(dtype)
    target_device = torch.device("cpu" if device is None else device)
    return torch.arange(sample_count, dtype=dtype, device=target_device)


def _validate_haar_sample_indices(sample_indices: Tensor) -> int:
    if sample_indices.ndim != 1:
        raise ValueError(f"sample_indices must be one-dimensional; got shape {tuple(sample_indices.shape)}")
    if sample_indices.numel() == 0:
        raise ValueError("sample_indices must contain at least one sample")
    if not sample_indices.is_floating_point():
        raise ValueError(f"sample_indices must use a floating dtype; got {sample_indices.dtype}")
    if not torch.isfinite(sample_indices).all():
        raise ValueError("sample_indices must be finite")
    sample_count = sample_indices.numel()
    expected = torch.arange(sample_count, dtype=sample_indices.dtype, device=sample_indices.device)
    if not torch.equal(sample_indices, expected):
        raise ValueError("sample_indices must be the contiguous sequence 0..sample_count-1")
    return sample_count


def haar_balanced_orthonormal_raw_basis(sample_indices: Tensor, order: int) -> Tensor:
    validate_positive_integer("order", order)
    sample_count = _validate_haar_sample_indices(sample_indices)
    if order > sample_count:
        raise ValueError(f"order must not exceed sample_count; got order={order}, sample_count={sample_count}")

    basis = torch.zeros(
        (sample_count, order),
        dtype=sample_indices.dtype,
        device=sample_indices.device,
    )
    basis[:, 0] = 1.0 / math.sqrt(float(sample_count))
    if order == 1:
        return basis

    intervals: Deque[Tuple[int, int]] = deque([(0, sample_count)])
    column_index = 1
    while intervals and column_index < order:
        start, end = intervals.popleft()
        interval_count = end - start
        if interval_count < 2:
            continue

        left_count = interval_count // 2
        right_count = interval_count - left_count
        split = start + left_count

        left_value = math.sqrt(float(right_count) / float(left_count * interval_count))
        right_value = -math.sqrt(float(left_count) / float(right_count * interval_count))
        basis[start:split, column_index] = left_value
        basis[split:end, column_index] = right_value
        column_index += 1

        if left_count > 1:
            intervals.append((start, split))
        if right_count > 1:
            intervals.append((split, end))

    if column_index != order:
        raise RuntimeError(f"balanced Haar construction produced {column_index} columns; expected {order}")
    return basis


class BalancedHaarOrthonormalBasisKernel(BasisKernel):
    def __init__(self) -> None:
        super().__init__(
            basis_family=BASIS_FAMILY_HAAR,
            basis_version=HAAR_BASIS_VERSION,
            coordinate_policy="integer_sample_index_balanced_binary_partition_v1",
            stabilization_policy="closed_form_balanced_haar_orthonormal_no_qr_v1",
        )

    def coordinates(
        self,
        sample_count: int,
        *,
        dtype: torch.dtype = torch.float64,
        device: Optional[DeviceLike] = None,
    ) -> Tensor:
        return haar_sample_indices(sample_count, dtype=dtype, device=device)

    def raw_basis(self, coordinates: Tensor, order: int) -> Tensor:
        return haar_balanced_orthonormal_raw_basis(coordinates, order)

    def stabilize(self, raw_basis: Tensor) -> Tensor:
        if raw_basis.ndim != 2:
            raise ValueError(f"raw_basis must be two-dimensional; got shape {tuple(raw_basis.shape)}")
        if not raw_basis.is_floating_point():
            raise ValueError(f"raw_basis must use a floating dtype; got {raw_basis.dtype}")
        if not torch.isfinite(raw_basis).all():
            raise FloatingPointError("balanced Haar basis contains a non-finite value")
        return raw_basis


BASIS_DEFINITION = BasisDefinition(
    family=BASIS_FAMILY_HAAR,
    aliases=("balanced_haar", "haar_balanced"),
    version=HAAR_BASIS_VERSION,
    artifact_tag=BASIS_ARTIFACT_TAG_HAAR,
    supports_weight_basis=True,
    supports_native_products=False,
    kernel=BalancedHaarOrthonormalBasisKernel(),
)
# ^^^ THOG
