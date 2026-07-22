# vvv THOG
from __future__ import annotations

import math
import re
from collections import deque
from typing import Deque, Dict, Optional, Tuple

import torch
from torch import Tensor

from .protocol import BasisDefinition, BasisKernel, DeviceLike, validate_floating_dtype, validate_positive_integer


BASIS_FAMILY_LAPPED_COSINE = "lapped_cosine"
LAPPED_COSINE_BASIS_VERSION = "lapped_cosine_dc_preserving_orthonormal_v1"
BASIS_ARTIFACT_TAG_LAPPED_COSINE = "LAPPED_COSINE"
DEFAULT_LAPPED_COSINE_WINDOW_LENGTH = 36
DEFAULT_LAPPED_COSINE_OVERLAP_FRACTION = 0.5
_BOUNDARY_PREFILTER_ANGLE = math.pi / 4.0
_VERSION_PATTERN = re.compile(
    rf"^{re.escape(LAPPED_COSINE_BASIS_VERSION)}_w(?P<window>[1-9][0-9]*)_o(?P<overlap_millis>[0-9]{{3}})$"
)


ColumnDescriptor = Tuple[str, int, int]
BlockInterval = Tuple[int, int]


def validate_lapped_cosine_controls(window_length: int, overlap_fraction: float) -> None:
    validate_positive_integer("lapped_cosine_window_length", window_length)
    if window_length < 2 or window_length % 2 != 0:
        raise ValueError(
            "lapped_cosine_window_length must be an even integer >= 2; "
            f"got {window_length!r}"
        )
    if isinstance(overlap_fraction, bool) or not isinstance(overlap_fraction, (int, float)):
        raise ValueError(
            "lapped_cosine_overlap_fraction must be numeric; "
            f"got {overlap_fraction!r}"
        )
    if not math.isfinite(float(overlap_fraction)):
        raise ValueError("lapped_cosine_overlap_fraction must be finite")
    if not math.isclose(
        float(overlap_fraction),
        DEFAULT_LAPPED_COSINE_OVERLAP_FRACTION,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "lapped_cosine_overlap_fraction currently supports only 0.5; "
            f"got {overlap_fraction!r}"
        )


def lapped_cosine_basis_version(window_length: int, overlap_fraction: float) -> str:
    validate_lapped_cosine_controls(window_length, overlap_fraction)
    if (
        window_length == DEFAULT_LAPPED_COSINE_WINDOW_LENGTH
        and math.isclose(
            float(overlap_fraction),
            DEFAULT_LAPPED_COSINE_OVERLAP_FRACTION,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        return LAPPED_COSINE_BASIS_VERSION
    overlap_millis = int(round(float(overlap_fraction) * 1000.0))
    return f"{LAPPED_COSINE_BASIS_VERSION}_w{window_length}_o{overlap_millis:03d}"


def parse_lapped_cosine_basis_version(version: str) -> Tuple[int, float]:
    if not isinstance(version, str) or not version.strip():
        raise ValueError("basis_version must be a non-empty string")
    normalized = version.strip().lower()
    if normalized == LAPPED_COSINE_BASIS_VERSION:
        return (
            DEFAULT_LAPPED_COSINE_WINDOW_LENGTH,
            DEFAULT_LAPPED_COSINE_OVERLAP_FRACTION,
        )
    match = _VERSION_PATTERN.fullmatch(normalized)
    if match is None:
        raise ValueError(
            f"basis_version mismatch for {BASIS_FAMILY_LAPPED_COSINE}: "
            f"expected {LAPPED_COSINE_BASIS_VERSION!r} or a parameterised v1 version, "
            f"got {version!r}"
        )
    window_length = int(match.group("window"))
    overlap_fraction = int(match.group("overlap_millis")) / 1000.0
    validate_lapped_cosine_controls(window_length, overlap_fraction)
    canonical = lapped_cosine_basis_version(window_length, overlap_fraction)
    if normalized != canonical:
        raise ValueError(
            f"non-canonical lapped cosine basis_version: expected {canonical!r}, "
            f"got {version!r}"
        )
    return window_length, overlap_fraction


def normalize_lapped_cosine_basis_version(version: str) -> str:
    window_length, overlap_fraction = parse_lapped_cosine_basis_version(version)
    return lapped_cosine_basis_version(window_length, overlap_fraction)


def lapped_cosine_sample_indices(
    sample_count: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: Optional[DeviceLike] = None,
) -> Tensor:
    validate_positive_integer("sample_count", sample_count)
    validate_floating_dtype(dtype)
    target_device = torch.device("cpu" if device is None else device)
    return torch.arange(sample_count, dtype=dtype, device=target_device)


def _validate_sample_indices(sample_indices: Tensor) -> int:
    if sample_indices.ndim != 1:
        raise ValueError(
            f"sample_indices must be one-dimensional; got shape {tuple(sample_indices.shape)}"
        )
    if sample_indices.numel() == 0:
        raise ValueError("sample_indices must contain at least one sample")
    if not sample_indices.is_floating_point():
        raise ValueError(
            f"sample_indices must use a floating dtype; got {sample_indices.dtype}"
        )
    if not torch.isfinite(sample_indices).all():
        raise ValueError("sample_indices must be finite")
    sample_count = sample_indices.numel()
    expected = torch.arange(
        sample_count,
        dtype=sample_indices.dtype,
        device=sample_indices.device,
    )
    if not torch.equal(sample_indices, expected):
        raise ValueError("sample_indices must be the contiguous sequence 0..sample_count-1")
    return sample_count


def _dct_ii_orthonormal_basis(
    sample_count: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    validate_positive_integer("sample_count", sample_count)
    sample_indices = torch.arange(sample_count, dtype=dtype, device=device).unsqueeze(1)
    mode_indices = torch.arange(sample_count, dtype=dtype, device=device).unsqueeze(0)
    angles = math.pi * (sample_indices + 0.5) * mode_indices / float(sample_count)
    basis = torch.cos(angles)
    basis[:, 0] *= math.sqrt(1.0 / float(sample_count))
    if sample_count > 1:
        basis[:, 1:] *= math.sqrt(2.0 / float(sample_count))
    return basis


def lapped_cosine_block_layout(
    sample_count: int,
    window_length: int,
) -> Tuple[BlockInterval, ...]:
    validate_positive_integer("sample_count", sample_count)
    validate_lapped_cosine_controls(
        window_length,
        DEFAULT_LAPPED_COSINE_OVERLAP_FRACTION,
    )
    block_length = min(sample_count, max(1, window_length // 2))
    blocks = []
    start = 0
    while start < sample_count:
        end = min(sample_count, start + block_length)
        blocks.append((start, end))
        start = end
    return tuple(blocks)


def lapped_cosine_column_schedule(
    sample_count: int,
    order: int,
    window_length: int,
) -> Tuple[ColumnDescriptor, ...]:
    validate_positive_integer("order", order)
    if order > sample_count:
        raise ValueError(
            f"order must not exceed sample_count; got order={order}, sample_count={sample_count}"
        )
    blocks = lapped_cosine_block_layout(sample_count, window_length)
    schedule = [
        ("coarse", coarse_index, 0)
        for coarse_index in range(min(order, len(blocks)))
    ]
    if len(schedule) == order:
        return tuple(schedule)

    maximum_block_length = max(end - start for start, end in blocks)
    for local_mode in range(1, maximum_block_length):
        for block_index, (start, end) in enumerate(blocks):
            if local_mode >= end - start:
                continue
            schedule.append(("detail", block_index, local_mode))
            if len(schedule) == order:
                return tuple(schedule)
    raise RuntimeError(
        f"lapped cosine schedule produced {len(schedule)} columns; expected {order}"
    )


def _weighted_balanced_block_mean_basis(
    sample_count: int,
    blocks: Tuple[BlockInterval, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    block_count = len(blocks)
    basis = torch.zeros((sample_count, block_count), dtype=dtype, device=device)
    basis[:, 0] = 1.0 / math.sqrt(float(sample_count))
    if block_count == 1:
        return basis

    intervals: Deque[Tuple[int, int]] = deque([(0, block_count)])
    column_index = 1
    while intervals and column_index < block_count:
        first_block, final_block = intervals.popleft()
        interval_block_count = final_block - first_block
        if interval_block_count < 2:
            continue
        split_block = first_block + interval_block_count // 2
        left_start = blocks[first_block][0]
        left_end = blocks[split_block - 1][1]
        right_start = blocks[split_block][0]
        right_end = blocks[final_block - 1][1]
        left_count = left_end - left_start
        right_count = right_end - right_start
        interval_count = left_count + right_count
        left_value = math.sqrt(float(right_count) / float(left_count * interval_count))
        right_value = -math.sqrt(float(left_count) / float(right_count * interval_count))
        basis[left_start:left_end, column_index] = left_value
        basis[right_start:right_end, column_index] = right_value
        column_index += 1
        if split_block - first_block > 1:
            intervals.append((first_block, split_block))
        if final_block - split_block > 1:
            intervals.append((split_block, final_block))

    if column_index != block_count:
        raise RuntimeError(
            f"weighted block-mean construction produced {column_index} columns; "
            f"expected {block_count}"
        )
    return basis


def _dc_preserving_boundary_prefilter(
    sample_count: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    cosine_basis = _dct_ii_orthonormal_basis(
        sample_count,
        dtype=dtype,
        device=device,
    )
    mode_rotation = torch.eye(sample_count, dtype=dtype, device=device)
    cosine = math.cos(_BOUNDARY_PREFILTER_ANGLE)
    sine = math.sin(_BOUNDARY_PREFILTER_ANGLE)
    for first_mode in range(1, sample_count - 1, 2):
        second_mode = first_mode + 1
        mode_rotation[first_mode, first_mode] = cosine
        mode_rotation[first_mode, second_mode] = -sine
        mode_rotation[second_mode, first_mode] = sine
        mode_rotation[second_mode, second_mode] = cosine
    return cosine_basis @ mode_rotation @ cosine_basis.transpose(0, 1)


def lapped_cosine_orthonormal_raw_basis(
    sample_indices: Tensor,
    order: int,
    *,
    window_length: int = DEFAULT_LAPPED_COSINE_WINDOW_LENGTH,
    overlap_fraction: float = DEFAULT_LAPPED_COSINE_OVERLAP_FRACTION,
) -> Tensor:
    validate_positive_integer("order", order)
    sample_count = _validate_sample_indices(sample_indices)
    if order > sample_count:
        raise ValueError(
            f"order must not exceed sample_count; got order={order}, sample_count={sample_count}"
        )
    validate_lapped_cosine_controls(window_length, overlap_fraction)
    blocks = lapped_cosine_block_layout(sample_count, window_length)
    schedule = lapped_cosine_column_schedule(sample_count, order, window_length)
    coarse_basis = _weighted_balanced_block_mean_basis(
        sample_count,
        blocks,
        dtype=sample_indices.dtype,
        device=sample_indices.device,
    )
    local_bases = {
        block_index: _dct_ii_orthonormal_basis(
            end - start,
            dtype=sample_indices.dtype,
            device=sample_indices.device,
        )
        for block_index, (start, end) in enumerate(blocks)
    }
    basis = torch.zeros(
        (sample_count, order),
        dtype=sample_indices.dtype,
        device=sample_indices.device,
    )

    for column_index, (column_kind, block_or_coarse_index, local_mode) in enumerate(schedule):
        if column_kind == "coarse":
            basis[:, column_index] = coarse_basis[:, block_or_coarse_index]
            continue
        if column_kind != "detail":
            raise RuntimeError(f"unknown lapped cosine column kind: {column_kind!r}")
        block_start, block_end = blocks[block_or_coarse_index]
        basis[block_start:block_end, column_index] = local_bases[block_or_coarse_index][
            :, local_mode
        ]

    for boundary_index in range(len(blocks) - 1):
        left_start, left_end = blocks[boundary_index]
        right_start, right_end = blocks[boundary_index + 1]
        lap_count = min(left_end - left_start, right_end - right_start) // 2
        if lap_count == 0:
            continue
        boundary_indices = torch.tensor(
            [
                *range(left_end - lap_count, left_end),
                *range(right_start, right_start + lap_count),
            ],
            dtype=torch.long,
            device=sample_indices.device,
        )
        prefilter = _dc_preserving_boundary_prefilter(
            2 * lap_count,
            dtype=sample_indices.dtype,
            device=sample_indices.device,
        )
        basis[boundary_indices] = prefilter @ basis[boundary_indices]

    basis[:, 0] = 1.0 / math.sqrt(float(sample_count))
    return basis


class LappedCosineOrthonormalBasisKernel(BasisKernel):
    def __init__(self) -> None:
        super().__init__(
            basis_family=BASIS_FAMILY_LAPPED_COSINE,
            basis_version=LAPPED_COSINE_BASIS_VERSION,
            coordinate_policy="integer_sample_index_balanced_local_blocks_v1",
            stabilization_policy="weighted_block_means_dct_ii_dc_preserving_boundary_prefilter_no_qr_v1",
        )

    def normalize_version(self, version: Optional[str]) -> str:
        candidate = self.basis_version if version is None else version
        return normalize_lapped_cosine_basis_version(candidate)

    def coordinates(
        self,
        sample_count: int,
        *,
        dtype: torch.dtype = torch.float64,
        device: Optional[DeviceLike] = None,
    ) -> Tensor:
        return lapped_cosine_sample_indices(sample_count, dtype=dtype, device=device)

    def raw_basis(self, coordinates: Tensor, order: int) -> Tensor:
        return lapped_cosine_orthonormal_raw_basis(coordinates, order)

    def stabilize(self, raw_basis: Tensor) -> Tensor:
        if raw_basis.ndim != 2:
            raise ValueError(
                f"raw_basis must be two-dimensional; got shape {tuple(raw_basis.shape)}"
            )
        if not raw_basis.is_floating_point():
            raise ValueError(
                f"raw_basis must use a floating dtype; got {raw_basis.dtype}"
            )
        if not torch.isfinite(raw_basis).all():
            raise FloatingPointError("lapped cosine basis contains a non-finite value")
        return raw_basis

    def build(
        self,
        sample_count: int,
        order: int,
        *,
        runtime_dtype: torch.dtype = torch.float64,
        device: Optional[DeviceLike] = None,
        version: Optional[str] = None,
    ) -> Tensor:
        validate_positive_integer("sample_count", sample_count)
        validate_positive_integer("order", order)
        if order > sample_count:
            raise ValueError(
                f"order must not exceed sample_count; got order={order}, sample_count={sample_count}"
            )
        validate_floating_dtype(runtime_dtype)
        normalized_version = self.normalize_version(version)
        window_length, overlap_fraction = parse_lapped_cosine_basis_version(
            normalized_version
        )
        coordinates = self.coordinates(sample_count, dtype=torch.float64, device="cpu")
        raw_basis = lapped_cosine_orthonormal_raw_basis(
            coordinates,
            order,
            window_length=window_length,
            overlap_fraction=overlap_fraction,
        )
        stabilized_basis = self.stabilize(raw_basis)
        target_device = torch.device("cpu" if device is None else device)
        runtime_basis = stabilized_basis.to(
            device=target_device,
            dtype=runtime_dtype,
        )
        runtime_basis.requires_grad_(False)
        if runtime_basis.shape != (sample_count, order):
            raise RuntimeError(
                f"unexpected basis shape {tuple(runtime_basis.shape)}; "
                f"expected {(sample_count, order)}"
            )
        if not torch.isfinite(runtime_basis).all():
            raise FloatingPointError("runtime basis contains a non-finite value")
        return runtime_basis

    def metadata(self) -> Dict[str, object]:
        return {
            **super().metadata(),
            "default_window_length": DEFAULT_LAPPED_COSINE_WINDOW_LENGTH,
            "default_overlap_fraction": DEFAULT_LAPPED_COSINE_OVERLAP_FRACTION,
            "boundary_policy": "finite_non_circular_v1",
            "coefficient_ordering": "global_dc_block_means_then_local_mode_by_block_v1",
            "initialization_contract": "normalized_global_dc_first_column_v1",
        }


BASIS_DEFINITION = BasisDefinition(
    family=BASIS_FAMILY_LAPPED_COSINE,
    aliases=("lapped", "local_cosine", "lapped_local_cosine"),
    version=LAPPED_COSINE_BASIS_VERSION,
    artifact_tag=BASIS_ARTIFACT_TAG_LAPPED_COSINE,
    supports_weight_basis=True,
    supports_native_products=False,
    kernel=LappedCosineOrthonormalBasisKernel(),
)
# ^^^ THOG
