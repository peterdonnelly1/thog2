from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    print(f"wrote {path}")


def replace_once(path: str, old: str, new: str) -> None:
    text = read(path)
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement target, found {count}: {old[:120]!r}")
    write(path, text.replace(old, new, 1))


write(
    "sheet/bases/lapped_cosine.py",
    """# vvv THOG
from __future__ import annotations

import math
import re
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from .protocol import BasisDefinition, BasisKernel, DeviceLike, validate_floating_dtype, validate_positive_integer


BASIS_FAMILY_LAPPED_COSINE = "lapped_cosine"
LAPPED_COSINE_BASIS_VERSION = "lapped_cosine_balanced_orthonormal_v1"
BASIS_ARTIFACT_TAG_LAPPED_COSINE = "LAPPED_COSINE"
DEFAULT_LAPPED_COSINE_WINDOW_LENGTH = 36
DEFAULT_LAPPED_COSINE_OVERLAP_FRACTION = 0.5
_VERSION_PATTERN = re.compile(
    rf"^{re.escape(LAPPED_COSINE_BASIS_VERSION)}_w(?P<window>[1-9][0-9]*)_o(?P<overlap_millis>[0-9]{{3}})$"
)


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


def lapped_cosine_block_layout(
    sample_count: int,
    window_length: int,
) -> Tuple[Tuple[int, int], ...]:
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
) -> Tuple[Tuple[int, int], ...]:
    validate_positive_integer("order", order)
    if order > sample_count:
        raise ValueError(
            f"order must not exceed sample_count; got order={order}, sample_count={sample_count}"
        )
    blocks = lapped_cosine_block_layout(sample_count, window_length)
    maximum_block_length = max(end - start for start, end in blocks)
    schedule = []
    for local_mode in range(maximum_block_length):
        for block_index, (start, end) in enumerate(blocks):
            if local_mode >= end - start:
                continue
            schedule.append((block_index, local_mode))
            if len(schedule) == order:
                return tuple(schedule)
    raise RuntimeError(
        f"lapped cosine schedule produced {len(schedule)} columns; expected {order}"
    )


def _dct_iv_column(
    sample_count: int,
    local_mode: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    local_indices = torch.arange(sample_count, dtype=dtype, device=device)
    angles = (
        math.pi
        * (local_indices + 0.5)
        * (float(local_mode) + 0.5)
        / float(sample_count)
    )
    return math.sqrt(2.0 / float(sample_count)) * torch.cos(angles)


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
    basis = torch.zeros(
        (sample_count, order),
        dtype=sample_indices.dtype,
        device=sample_indices.device,
    )

    for column_index, (block_index, local_mode) in enumerate(schedule):
        start, end = blocks[block_index]
        basis[start:end, column_index] = _dct_iv_column(
            end - start,
            local_mode,
            dtype=sample_indices.dtype,
            device=sample_indices.device,
        )

    for boundary_index in range(len(blocks) - 1):
        left_start, left_end = blocks[boundary_index]
        right_start, right_end = blocks[boundary_index + 1]
        lap_count = min(left_end - left_start, right_end - right_start) // 2
        if lap_count == 0:
            continue
        for lap_index in range(lap_count):
            left_index = left_end - lap_count + lap_index
            right_index = right_start + lap_index
            angle = math.pi * (float(lap_index) + 0.5) / (2.0 * float(lap_count))
            cosine = math.cos(angle)
            sine = math.sin(angle)
            left_values = basis[left_index].clone()
            right_values = basis[right_index].clone()
            basis[left_index] = cosine * left_values + sine * right_values
            basis[right_index] = -sine * left_values + cosine * right_values

    return basis


class LappedCosineOrthonormalBasisKernel(BasisKernel):
    def __init__(self) -> None:
        super().__init__(
            basis_family=BASIS_FAMILY_LAPPED_COSINE,
            basis_version=LAPPED_COSINE_BASIS_VERSION,
            coordinate_policy="integer_sample_index_balanced_local_blocks_v1",
            stabilization_policy="closed_form_dct_iv_with_orthogonal_boundary_rotations_no_qr_v1",
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
            "coefficient_ordering": "local_mode_then_block_v1",
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
""",
)

replace_once(
    "sheet/bases/protocol.py",
    """    def coordinates(self, sample_count: int, *, dtype: torch.dtype = torch.float64, device: Optional[DeviceLike] = None) -> Tensor:
        raise NotImplementedError

    def raw_basis(self, coordinates: Tensor, order: int) -> Tensor:
        raise NotImplementedError
""",
    """    # vvv THOG permit basis families with parameterised but canonical version strings
    def normalize_version(self, version: Optional[str]) -> str:
        candidate = self.basis_version if version is None else version
        if not isinstance(candidate, str) or not candidate.strip():
            raise ValueError("basis_version must be a non-empty string")
        normalized = candidate.strip().lower()
        if normalized != self.basis_version:
            raise ValueError(
                f"basis_version mismatch for {self.basis_family}: "
                f"expected {self.basis_version!r}, got {candidate!r}"
            )
        return normalized
    # ^^^ THOG

    def coordinates(self, sample_count: int, *, dtype: torch.dtype = torch.float64, device: Optional[DeviceLike] = None) -> Tensor:
        raise NotImplementedError

    def raw_basis(self, coordinates: Tensor, order: int) -> Tensor:
        raise NotImplementedError
""",
)
replace_once(
    "sheet/bases/protocol.py",
    """        basis_version = self.basis_version if version is None else version
        if not isinstance(basis_version, str) or not basis_version.strip():
            raise ValueError("version must be a non-empty string")
        if basis_version != self.basis_version:
            raise ValueError(f"basis_version mismatch for {self.basis_family}: expected {self.basis_version!r}, got {basis_version!r}")
""",
    """        # vvv THOG version validation is delegated to the basis kernel
        basis_version = self.normalize_version(version)
        # ^^^ THOG
""",
)

replace_once(
    "sheet/bases/registry.py",
    """BUILTIN_BASIS_MODULES = (
    "chebyshev",
    "dct",
    "haar",
)
""",
    """BUILTIN_BASIS_MODULES = (
    "chebyshev",
    "dct",
    "haar",
    "lapped_cosine",                                                                                                                                    # <<< THOG append the local lapped basis without reordering existing registry entries
)
""",
)
replace_once(
    "sheet/bases/registry.py",
    """def normalize_basis_version(basis_family: str, basis_version: str, *, legacy_default_version: Optional[str] = None) -> str:
    expected_version = basis_version_for_family(basis_family)
    if basis_version == "auto":
        return expected_version
    if legacy_default_version is not None and basis_version == legacy_default_version and expected_version != legacy_default_version:
        return expected_version
    if basis_version != expected_version:
        raise ValueError(f"basis_version mismatch for basis_family={normalize_registered_basis_family(basis_family)!r}: expected {expected_version!r}, got {basis_version!r}")
    return expected_version
""",
    """def normalize_basis_version(basis_family: str, basis_version: str, *, legacy_default_version: Optional[str] = None) -> str:
    expected_version = basis_version_for_family(basis_family)
    if basis_version == "auto":
        return expected_version
    if legacy_default_version is not None and basis_version == legacy_default_version and expected_version != legacy_default_version:
        return expected_version
    # vvv THOG allow a kernel to validate and canonicalise a parameterised version
    return get_basis_kernel(basis_family).normalize_version(basis_version)
    # ^^^ THOG
""",
)

replace_once(
    "sheet/bases/__init__.py",
    """from .dct import BASIS_ARTIFACT_TAG_DCT, BASIS_FAMILY_DCT, DCT_BASIS_VERSION, DctIiOrthonormalBasisKernel, dct_ii_orthonormal_raw_basis, dct_sample_indices
from .registry import BASIS_FAMILIES, BASIS_REGISTRY, BUILTIN_BASIS_MODULES, BasisRegistry, basis_artifact_tag_for_family, basis_kernel_metadata, basis_registry_metadata, basis_version_for_family, build_registered_basis, get_basis_definition, get_basis_kernel, get_basis_spec, normalize_basis_family, normalize_basis_version, normalize_registered_basis_family, registered_basis_families
""",
    """from .dct import BASIS_ARTIFACT_TAG_DCT, BASIS_FAMILY_DCT, DCT_BASIS_VERSION, DctIiOrthonormalBasisKernel, dct_ii_orthonormal_raw_basis, dct_sample_indices
# vvv THOG export the lapped cosine plugin and its explicit controls
from .lapped_cosine import (
    BASIS_ARTIFACT_TAG_LAPPED_COSINE,
    BASIS_FAMILY_LAPPED_COSINE,
    DEFAULT_LAPPED_COSINE_OVERLAP_FRACTION,
    DEFAULT_LAPPED_COSINE_WINDOW_LENGTH,
    LAPPED_COSINE_BASIS_VERSION,
    LappedCosineOrthonormalBasisKernel,
    lapped_cosine_basis_version,
    lapped_cosine_block_layout,
    lapped_cosine_column_schedule,
    lapped_cosine_orthonormal_raw_basis,
    lapped_cosine_sample_indices,
    normalize_lapped_cosine_basis_version,
    parse_lapped_cosine_basis_version,
    validate_lapped_cosine_controls,
)
# ^^^ THOG
from .registry import BASIS_FAMILIES, BASIS_REGISTRY, BUILTIN_BASIS_MODULES, BasisRegistry, basis_artifact_tag_for_family, basis_kernel_metadata, basis_registry_metadata, basis_version_for_family, build_registered_basis, get_basis_definition, get_basis_kernel, get_basis_spec, normalize_basis_family, normalize_basis_version, normalize_registered_basis_family, registered_basis_families
""",
)
replace_once(
    "sheet/bases/__init__.py",
    """    "BASIS_ARTIFACT_TAG_DCT",
    "BASIS_FAMILIES",
""",
    """    "BASIS_ARTIFACT_TAG_DCT",
    "BASIS_ARTIFACT_TAG_LAPPED_COSINE",                                                                                                                  # <<< THOG public lapped cosine artifact tag
    "BASIS_FAMILIES",
""",
)
replace_once(
    "sheet/bases/__init__.py",
    """    "BASIS_FAMILY_DCT",
    "BASIS_REGISTRY",
""",
    """    "BASIS_FAMILY_DCT",
    "BASIS_FAMILY_LAPPED_COSINE",                                                                                                                        # <<< THOG public lapped cosine family
    "BASIS_REGISTRY",
""",
)
replace_once(
    "sheet/bases/__init__.py",
    """    "DCT_BASIS_VERSION",
    "SINGLE_POINT_COORDINATE",
""",
    """    "DCT_BASIS_VERSION",
    "LAPPED_COSINE_BASIS_VERSION",                                                                                                                       # <<< THOG public lapped cosine version
    "DEFAULT_LAPPED_COSINE_OVERLAP_FRACTION",                                                                                                            # <<< THOG public lapped overlap default
    "DEFAULT_LAPPED_COSINE_WINDOW_LENGTH",                                                                                                               # <<< THOG public lapped locality default
    "SINGLE_POINT_COORDINATE",
""",
)
replace_once(
    "sheet/bases/__init__.py",
    """    "DctIiOrthonormalBasisKernel",
    "DeviceLike",
""",
    """    "DctIiOrthonormalBasisKernel",
    "LappedCosineOrthonormalBasisKernel",                                                                                                                # <<< THOG public lapped kernel
    "DeviceLike",
""",
)
replace_once(
    "sheet/bases/__init__.py",
    """    "dct_sample_indices",
    "deterministic_reduced_qr_positive_diagonal",
""",
    """    "dct_sample_indices",
    "lapped_cosine_basis_version",                                                                                                                       # <<< THOG lapped version builder
    "lapped_cosine_block_layout",                                                                                                                        # <<< THOG lapped block diagnostics
    "lapped_cosine_column_schedule",                                                                                                                     # <<< THOG balanced prefix diagnostics
    "lapped_cosine_orthonormal_raw_basis",                                                                                                               # <<< THOG lapped raw basis
    "lapped_cosine_sample_indices",                                                                                                                      # <<< THOG lapped coordinates
    "normalize_lapped_cosine_basis_version",                                                                                                             # <<< THOG lapped version normalisation
    "parse_lapped_cosine_basis_version",                                                                                                                 # <<< THOG lapped version parsing
    "validate_lapped_cosine_controls",                                                                                                                   # <<< THOG lapped control validation
    "deterministic_reduced_qr_positive_diagonal",
""",
)

replace_once(
    "sheet/basis.py",
    """        expected_version = basis_version_for_family(canonical_family)
        if version != expected_version:
            raise ValueError(f"basis_version mismatch for {canonical_family}: expected {expected_version!r}, got {version!r}")
        return BasisCacheKey(sample_count=sample_count, order=order, basis_family=canonical_family, version=version, device_type=target_device.type, device_index=target_device.index, dtype=runtime_dtype)
""",
    """        # vvv THOG cache parameterised basis versions by their canonical kernel-normalised identity
        normalized_version = get_basis_kernel(canonical_family).normalize_version(version)
        return BasisCacheKey(sample_count=sample_count, order=order, basis_family=canonical_family, version=normalized_version, device_type=target_device.type, device_index=target_device.index, dtype=runtime_dtype)
        # ^^^ THOG
""",
)

replace_once(
    "sheet/compact_identity.py",
    """from .bases import BASIS_FAMILIES as REGISTERED_BASIS_FAMILIES, BASIS_FAMILY_CHEBYSHEV, BASIS_FAMILY_DCT, basis_version_for_family, normalize_basis_version, normalize_registered_basis_family
""",
    """from .bases import BASIS_FAMILIES as REGISTERED_BASIS_FAMILIES, BASIS_FAMILY_CHEBYSHEV, BASIS_FAMILY_DCT, basis_version_for_family, normalize_basis_version, normalize_registered_basis_family
# vvv THOG lapped cosine controls are part of compact identity
from .bases.lapped_cosine import (
    BASIS_FAMILY_LAPPED_COSINE,
    DEFAULT_LAPPED_COSINE_OVERLAP_FRACTION,
    DEFAULT_LAPPED_COSINE_WINDOW_LENGTH,
    lapped_cosine_basis_version,
    validate_lapped_cosine_controls,
)
# ^^^ THOG
""",
)
replace_once(
    "sheet/compact_identity.py",
    """    basis_family: str
    basis_version: str
    materialization_version: str
""",
    """    basis_family: str
    basis_version: str
    lapped_cosine_window_length: int                                                                                                                     # <<< THOG explicit locality identity
    lapped_cosine_overlap_fraction: float                                                                                                                # <<< THOG explicit overlap identity
    materialization_version: str
""",
)
replace_once(
    "sheet/compact_identity.py",
    """    basis_version: str = BASIS_VERSION,
    row_order_scaling_rule: str,
""",
    """    basis_version: str = BASIS_VERSION,
    lapped_cosine_window_length: int = DEFAULT_LAPPED_COSINE_WINDOW_LENGTH,                                                                               # <<< THOG locality control
    lapped_cosine_overlap_fraction: float = DEFAULT_LAPPED_COSINE_OVERLAP_FRACTION,                                                                        # <<< THOG overlap control
    row_order_scaling_rule: str,
""",
)
replace_once(
    "sheet/compact_identity.py",
    """    basis_version = normalize_compact_basis_version(selectors, basis_version)
    heads = head_metadata(n_embd, n_head)
""",
    """    basis_version = normalize_compact_basis_version(selectors, basis_version)
    # vvv THOG bind the lapped controls to the canonical basis version and reject leakage into other families
    if selectors.basis_family == BASIS_FAMILY_LAPPED_COSINE:
        validate_lapped_cosine_controls(
            lapped_cosine_window_length,
            lapped_cosine_overlap_fraction,
        )
        expected_lapped_version = lapped_cosine_basis_version(
            lapped_cosine_window_length,
            lapped_cosine_overlap_fraction,
        )
        if basis_version != expected_lapped_version:
            raise ValueError(
                "lapped cosine controls do not match basis_version: "
                f"controls imply {expected_lapped_version!r}, got {basis_version!r}"
            )
    elif (
        lapped_cosine_window_length != DEFAULT_LAPPED_COSINE_WINDOW_LENGTH
        or abs(
            float(lapped_cosine_overlap_fraction)
            - DEFAULT_LAPPED_COSINE_OVERLAP_FRACTION
        )
        > 1.0e-12
    ):
        raise ValueError(
            "lapped cosine controls may be changed only when "
            "basis_family='lapped_cosine'"
        )
    # ^^^ THOG
    heads = head_metadata(n_embd, n_head)
""",
)
replace_once(
    "sheet/compact_identity.py",
    """        basis_family=selectors.basis_family,
        basis_version=basis_version,
        materialization_version=compact_materialization_version(selectors),
""",
    """        basis_family=selectors.basis_family,
        basis_version=basis_version,
        lapped_cosine_window_length=lapped_cosine_window_length,                                                                                          # <<< THOG persist locality identity
        lapped_cosine_overlap_fraction=float(lapped_cosine_overlap_fraction),                                                                              # <<< THOG persist overlap identity
        materialization_version=compact_materialization_version(selectors),
""",
)

replace_once(
    "sheet/training_config.py",
    """from .basis import BASIS_VERSION
from .checkpointing import validate_checkpoint_segment_size
""",
    """from .basis import BASIS_VERSION
# vvv THOG lapped cosine controls survive training config and checkpoints
from .bases import normalize_registered_basis_family
from .bases.lapped_cosine import (
    BASIS_FAMILY_LAPPED_COSINE,
    DEFAULT_LAPPED_COSINE_OVERLAP_FRACTION,
    DEFAULT_LAPPED_COSINE_WINDOW_LENGTH,
    LAPPED_COSINE_BASIS_VERSION,
    lapped_cosine_basis_version,
)
# ^^^ THOG
from .checkpointing import validate_checkpoint_segment_size
""",
)
replace_once(
    "sheet/training_config.py",
    """    "basis_version",
    "row_order_scaling_rule",
""",
    """    "basis_version",
    "lapped_cosine_window_length",                                                                                                                        # <<< THOG checkpoint compatibility locality control
    "lapped_cosine_overlap_fraction",                                                                                                                     # <<< THOG checkpoint compatibility overlap control
    "row_order_scaling_rule",
""",
)
replace_once(
    "sheet/training_config.py",
    """    basis_version: str = BASIS_VERSION
    row_order_scaling_rule: str = ROW_ORDER_SCALING_RULE
""",
    """    basis_version: str = BASIS_VERSION
    lapped_cosine_window_length: int = DEFAULT_LAPPED_COSINE_WINDOW_LENGTH                                                                                 # <<< THOG explicit locality control
    lapped_cosine_overlap_fraction: float = DEFAULT_LAPPED_COSINE_OVERLAP_FRACTION                                                                          # <<< THOG explicit overlap control
    row_order_scaling_rule: str = ROW_ORDER_SCALING_RULE
""",
)
replace_once(
    "sheet/training_config.py",
    """        else:
            identity = self.compact_identity_metadata()
            self.basis_version = str(identity["basis_version"])
""",
    """        else:
            # vvv THOG derive the parameterised lapped version before compact identity validation
            canonical_basis_family = normalize_registered_basis_family(self.basis_family or "chebyshev")
            if canonical_basis_family == BASIS_FAMILY_LAPPED_COSINE and self.basis_version in ("auto", BASIS_VERSION, LAPPED_COSINE_BASIS_VERSION):
                self.basis_version = lapped_cosine_basis_version(
                    self.lapped_cosine_window_length,
                    self.lapped_cosine_overlap_fraction,
                )
            identity = self.compact_identity_metadata()
            self.basis_version = str(identity["basis_version"])
            # ^^^ THOG
""",
)
replace_once(
    "sheet/training_config.py",
    """            basis_version=self.basis_version,
            row_order_scaling_rule=self.row_order_scaling_rule,
""",
    """            basis_version=self.basis_version,
            lapped_cosine_window_length=self.lapped_cosine_window_length,                                                                                  # <<< THOG compact identity locality control
            lapped_cosine_overlap_fraction=self.lapped_cosine_overlap_fraction,                                                                            # <<< THOG compact identity overlap control
            row_order_scaling_rule=self.row_order_scaling_rule,
""",
)

replace_once(
    "sheet/run_config.py",
    """from .bases import basis_artifact_tag_for_family, basis_version_for_family
""",
    """from .bases import basis_artifact_tag_for_family, basis_version_for_family, normalize_registered_basis_family
# vvv THOG lapped cosine run controls and version identity
from .bases.lapped_cosine import (
    BASIS_FAMILY_LAPPED_COSINE,
    DEFAULT_LAPPED_COSINE_OVERLAP_FRACTION,
    DEFAULT_LAPPED_COSINE_WINDOW_LENGTH,
    LAPPED_COSINE_BASIS_VERSION,
    lapped_cosine_basis_version,
    normalize_lapped_cosine_basis_version,
)
# ^^^ THOG
""",
)
replace_once(
    "sheet/run_config.py",
    """    basis_family: Optional[str] = BASIS_FAMILY_CHEBYSHEV
    basis_version: str = BASIS_VERSION
    attention_backend: str = "auto"
""",
    """    basis_family: Optional[str] = BASIS_FAMILY_CHEBYSHEV
    basis_version: str = BASIS_VERSION
    lapped_cosine_window_length: int = DEFAULT_LAPPED_COSINE_WINDOW_LENGTH                                                                                 # <<< THOG locality control
    lapped_cosine_overlap_fraction: float = DEFAULT_LAPPED_COSINE_OVERLAP_FRACTION                                                                          # <<< THOG overlap control
    attention_backend: str = "auto"
""",
)
replace_once(
    "sheet/run_config.py",
    """        if self.basis_version == "auto":
            requested_family = self.basis_family or BASIS_FAMILY_CHEBYSHEV
            resolved_version = BASIS_VERSION if requested_family == BASIS_FAMILY_CONVENTIONAL else basis_version_for_family(requested_family)
            object.__setattr__(self, "basis_version", resolved_version)
""",
    """        # vvv THOG derive and validate the parameterised lapped basis version from explicit controls
        requested_family = self.basis_family or BASIS_FAMILY_CHEBYSHEV
        canonical_family = (
            requested_family
            if requested_family == BASIS_FAMILY_CONVENTIONAL
            else normalize_registered_basis_family(requested_family)
        )
        if canonical_family == BASIS_FAMILY_LAPPED_COSINE:
            control_version = lapped_cosine_basis_version(
                self.lapped_cosine_window_length,
                self.lapped_cosine_overlap_fraction,
            )
            if self.basis_version in ("auto", BASIS_VERSION, LAPPED_COSINE_BASIS_VERSION):
                object.__setattr__(self, "basis_version", control_version)
            else:
                explicit_version = normalize_lapped_cosine_basis_version(self.basis_version)
                if explicit_version != control_version:
                    raise ValueError(
                        "lapped cosine controls do not match explicit basis_version: "
                        f"controls imply {control_version!r}, got {explicit_version!r}"
                    )
                object.__setattr__(self, "basis_version", explicit_version)
        else:
            if (
                self.lapped_cosine_window_length != DEFAULT_LAPPED_COSINE_WINDOW_LENGTH
                or abs(
                    float(self.lapped_cosine_overlap_fraction)
                    - DEFAULT_LAPPED_COSINE_OVERLAP_FRACTION
                )
                > 1.0e-12
            ):
                raise ValueError(
                    "lapped cosine controls may be changed only when "
                    "basis_family='lapped_cosine'"
                )
            if self.basis_version == "auto":
                resolved_version = BASIS_VERSION if canonical_family == BASIS_FAMILY_CONVENTIONAL else basis_version_for_family(canonical_family)
                object.__setattr__(self, "basis_version", resolved_version)
        # ^^^ THOG
""",
)
replace_once(
    "sheet/run_config.py",
    """            basis_version=self.basis_version,
            row_order_scaling_rule=ROW_ORDER_SCALING_RULE,
""",
    """            basis_version=self.basis_version,
            lapped_cosine_window_length=self.lapped_cosine_window_length,                                                                                  # <<< THOG compact identity locality control
            lapped_cosine_overlap_fraction=self.lapped_cosine_overlap_fraction,                                                                            # <<< THOG compact identity overlap control
            row_order_scaling_rule=ROW_ORDER_SCALING_RULE,
""",
)
replace_once(
    "sheet/run_config.py",
    """                "basis_version": self.basis_version,
                "geometry_preset": self.geometry_preset,
""",
    """                "basis_version": self.basis_version,
                "lapped_cosine_window_length": self.lapped_cosine_window_length,                                                                           # <<< THOG checkpoint locality control
                "lapped_cosine_overlap_fraction": self.lapped_cosine_overlap_fraction,                                                                      # <<< THOG checkpoint overlap control
                "geometry_preset": self.geometry_preset,
""",
)
replace_once(
    "sheet/run_config.py",
    """            fields.extend([
                f"P_{self.o_depth}",
                f"Q_{self.o_attn_d_model}",
                f"J_{self.o_attn_qkv_per_channel}",
                f"O_{self.o_attn_out_per_channel}",
                f"X_{self.o_mlp_d_model}",
                f"Y_{self.o_mlp_hidden}",
            ])
""",
    """            fields.extend([
                f"P_{self.o_depth}",
                f"Q_{self.o_attn_d_model}",
                f"J_{self.o_attn_qkv_per_channel}",
                f"O_{self.o_attn_out_per_channel}",
                f"X_{self.o_mlp_d_model}",
                f"Y_{self.o_mlp_hidden}",
            ])
            # vvv THOG lapped locality and overlap are visible in run identity
            if self.compact_identity()["basis_family"] == BASIS_FAMILY_LAPPED_COSINE:
                overlap_percent = int(round(self.lapped_cosine_overlap_fraction * 100.0))
                fields.extend(
                    [
                        f"LCW_{self.lapped_cosine_window_length}",
                        f"LCO_{overlap_percent}",
                    ]
                )
            # ^^^ THOG
""",
)

replace_once(
    "run_thog2_owt.py",
    """from sheet.bases import BASIS_FAMILIES
""",
    """from sheet.bases import BASIS_FAMILIES
from sheet.bases.lapped_cosine import DEFAULT_LAPPED_COSINE_OVERLAP_FRACTION, DEFAULT_LAPPED_COSINE_WINDOW_LENGTH                                             # <<< THOG lapped CLI defaults
""",
)
replace_once(
    "run_thog2_owt.py",
    """    parser.add_argument("--basis-version", default="auto")
    parser.add_argument("--attention-backend", choices=("auto", "flash2", "sdpa", "math"), default="auto")
""",
    """    parser.add_argument("--basis-version", default="auto")
    # vvv THOG explicit lapped cosine controls
    parser.add_argument("--lapped-cosine-window-length", type=int, default=DEFAULT_LAPPED_COSINE_WINDOW_LENGTH)
    parser.add_argument("--lapped-cosine-overlap-fraction", type=float, default=DEFAULT_LAPPED_COSINE_OVERLAP_FRACTION)
    # ^^^ THOG
    parser.add_argument("--attention-backend", choices=("auto", "flash2", "sdpa", "math"), default="auto")
""",
)
replace_once(
    "run_thog2_owt.py",
    """        basis_family=arguments.basis_family,
        basis_version=basis_version,
        attention_backend=arguments.attention_backend,
""",
    """        basis_family=arguments.basis_family,
        basis_version=basis_version,
        lapped_cosine_window_length=arguments.lapped_cosine_window_length,                                                                                 # <<< THOG CLI locality control
        lapped_cosine_overlap_fraction=arguments.lapped_cosine_overlap_fraction,                                                                           # <<< THOG CLI overlap control
        attention_backend=arguments.attention_backend,
""",
)
replace_once(
    "run_thog2_owt.py",
    """def config_from_arguments(arguments: argparse.Namespace) -> OwtRunConfig:
    basis_version = BASIS_VERSION if arguments.basis_version == "auto" else arguments.basis_version
""",
    """def config_from_arguments(arguments: argparse.Namespace) -> OwtRunConfig:
    # vvv THOG OwtRunConfig resolves auto after seeing basis-specific controls
    basis_version = arguments.basis_version
    # ^^^ THOG
""",
)

for wrapper_path in ("current_scruffy_train_OWT.sh", "current_dreedle_train_OWT.sh"):
    replace_once(
        wrapper_path,
        """BASIS_VERSION="auto"
ATTENTION_GEOMETRY=""
""",
        """BASIS_VERSION="auto"
LAPPED_COSINE_WINDOW_LENGTH=36                                                                                                                             # <<< THOG default lapped locality scale
LAPPED_COSINE_OVERLAP_FRACTION="0.5"                                                                                                                       # <<< THOG v1 fixed overlap
ATTENTION_GEOMETRY=""
""",
    )
    replace_once(
        wrapper_path,
        """Haar aliases: balanced_haar | haar_balanced""",
        """Haar aliases: balanced_haar | haar_balanced
                                                     Lapped cosine aliases: lapped | local_cosine | lapped_local_cosine""",
    )
    replace_once(
        wrapper_path,
        """                                                     haar_balanced_binary_orthonormal_v1
  -a ATTENTION_GEOMETRY=${ATTENTION_GEOMETRY:-preset default}
""",
        """                                                     haar_balanced_binary_orthonormal_v1
                                                     lapped_cosine_balanced_orthonormal_v1
  -W LAPPED_COSINE_WINDOW_LENGTH=${LAPPED_COSINE_WINDOW_LENGTH}
  -i LAPPED_COSINE_OVERLAP_FRACTION=${LAPPED_COSINE_OVERLAP_FRACTION}  currently 0.5 only
  -a ATTENTION_GEOMETRY=${ATTENTION_GEOMETRY:-preset default}
""",
    )
    replace_once(
        wrapper_path,
        """  -B BASIS_FAMILY=${BASIS_FAMILY}                   canonical: chebyshev | dct | haar; single, comma, or quoted space list
""",
        """  -B BASIS_FAMILY=${BASIS_FAMILY}                   canonical: chebyshev | dct | haar | lapped_cosine; single, comma, or quoted space list
""",
    )
    replace_once(
        wrapper_path,
        'while getopts ":q:g:n:b:c:f:A:G:u:e:l:w:k:I:F:N:U:V:p:B:v:a:m:L:H:D:C:P:Q:J:O:X:Y:S:E:T:K:r:z:Z:d:t:o:j:R:x:h" option; do',
        'while getopts ":q:g:n:b:c:f:A:G:u:e:l:w:k:I:F:N:U:V:p:B:v:W:i:a:m:L:H:D:C:P:Q:J:O:X:Y:S:E:T:K:r:z:Z:d:t:o:j:R:x:h" option; do',
    )
    replace_once(
        wrapper_path,
        """    p) GEOMETRY_PRESET="$OPTARG" ;; B) BASIS_FAMILY="$OPTARG" ;; v) BASIS_VERSION="$OPTARG" ;; a) ATTENTION_GEOMETRY="$OPTARG" ;; m) MLP_GEOMETRY="$OPTARG" ;;
""",
        """    p) GEOMETRY_PRESET="$OPTARG" ;; B) BASIS_FAMILY="$OPTARG" ;; v) BASIS_VERSION="$OPTARG" ;; W) LAPPED_COSINE_WINDOW_LENGTH="$OPTARG" ;; i) LAPPED_COSINE_OVERLAP_FRACTION="$OPTARG" ;; a) ATTENTION_GEOMETRY="$OPTARG" ;; m) MLP_GEOMETRY="$OPTARG" ;;
""",
    )
    replace_once(
        wrapper_path,
        """for setting in "$STEPS" "$GRADIENT_ACCUMULATION_STEPS" "$NUM_GPUS" "$EVAL_ITERS" "$EVAL_INTERVAL" "$LOG_INTERVAL" "$N_LAYER" "$N_HEAD" "$N_EMBD" "$BLOCK_SIZE" "$O_ATTN_D_MODEL" "$O_ATTN_QKV_PER_CHANNEL" "$O_ATTN_OUT_PER_CHANNEL" "$O_MLP_D_MODEL" "$O_MLP_HIDDEN" "$CHECKPOINT_SEGMENT_SIZE" "$RESIDUAL_INIT_DEPTH_VALUE" "$DEPTH_CURVE_SAMPLE_ELEMENTS"; do validate_positive_uint "$setting" "numeric setting"; done
""",
        """for setting in "$STEPS" "$GRADIENT_ACCUMULATION_STEPS" "$NUM_GPUS" "$EVAL_ITERS" "$EVAL_INTERVAL" "$LOG_INTERVAL" "$N_LAYER" "$N_HEAD" "$N_EMBD" "$BLOCK_SIZE" "$O_ATTN_D_MODEL" "$O_ATTN_QKV_PER_CHANNEL" "$O_ATTN_OUT_PER_CHANNEL" "$O_MLP_D_MODEL" "$O_MLP_HIDDEN" "$CHECKPOINT_SEGMENT_SIZE" "$RESIDUAL_INIT_DEPTH_VALUE" "$DEPTH_CURVE_SAMPLE_ELEMENTS" "$LAPPED_COSINE_WINDOW_LENGTH"; do validate_positive_uint "$setting" "numeric setting"; done
# vvv THOG lapped cosine v1 accepts exactly 50 percent overlap
case "$LAPPED_COSINE_OVERLAP_FRACTION" in
  .5|0.5|0.50|0.500) LAPPED_COSINE_OVERLAP_FRACTION="0.5" ;;
  *) echo "LAPPED_COSINE_OVERLAP_FRACTION currently supports only 0.5." >&2; exit 2 ;;
esac
(( LAPPED_COSINE_WINDOW_LENGTH >= 2 && LAPPED_COSINE_WINDOW_LENGTH % 2 == 0 )) || { echo "LAPPED_COSINE_WINDOW_LENGTH must be an even integer >= 2." >&2; exit 2; }
# ^^^ THOG
""",
    )
    replace_once(
        wrapper_path,
        """    compact_args=(--geometry-preset "$geometry_preset_value" --basis-family "$basis_family_value" --basis-version "$BASIS_VERSION")
""",
        """    compact_args=(--geometry-preset "$geometry_preset_value" --basis-family "$basis_family_value" --basis-version "$BASIS_VERSION" --lapped-cosine-window-length "$LAPPED_COSINE_WINDOW_LENGTH" --lapped-cosine-overlap-fraction "$LAPPED_COSINE_OVERLAP_FRACTION")
""",
    )
    replace_once(
        wrapper_path,
        """  model/preset/basis: $display_model_type / $geometry_preset_value / $basis_family_value
  backend/dtype:      $ATTENTION_BACKEND / $DTYPE
""",
        """  model/preset/basis: $display_model_type / $geometry_preset_value / $basis_family_value
  lapped cosine:      window=$LAPPED_COSINE_WINDOW_LENGTH overlap=$LAPPED_COSINE_OVERLAP_FRACTION
  backend/dtype:      $ATTENTION_BACKEND / $DTYPE
""",
    )

replace_once(
    "tests/test_stage8_training_instrumentation_selector.py",
    """        assert "M:" not in text.split("getopts", 1)[1].split(" option", 1)[0]
        assert "W:" not in text.split("getopts", 1)[1].split(" option", 1)[0]
""",
    """        assert "M:" not in text.split("getopts", 1)[1].split(" option", 1)[0]
        assert "-W LAPPED_COSINE_WINDOW_LENGTH=" in text
        assert "W:" in text.split("getopts", 1)[1].split(" option", 1)[0]
""",
)

write(
    "tests/test_lapped_cosine_basis_kernel.py",
    """# vvv THOG
from __future__ import annotations

import unittest
from pathlib import Path

import torch

from sheet.bases import (
    BASIS_ARTIFACT_TAG_LAPPED_COSINE,
    BASIS_FAMILY_LAPPED_COSINE,
    DEFAULT_LAPPED_COSINE_OVERLAP_FRACTION,
    DEFAULT_LAPPED_COSINE_WINDOW_LENGTH,
    LAPPED_COSINE_BASIS_VERSION,
    basis_artifact_tag_for_family,
    basis_version_for_family,
    build_registered_basis,
    get_basis_definition,
    lapped_cosine_basis_version,
    lapped_cosine_block_layout,
    lapped_cosine_column_schedule,
    lapped_cosine_orthonormal_raw_basis,
    lapped_cosine_sample_indices,
    normalize_registered_basis_family,
)


class LappedCosineBasisKernelTests(unittest.TestCase):
    def test_01_full_bases_are_orthonormal_for_small_odd_even_and_depth_lengths(self) -> None:
        for sample_count in (1, 2, 3, 5, 8, 9, 17, 36, 72, 144):
            with self.subTest(sample_count=sample_count):
                basis = build_registered_basis(
                    sample_count,
                    sample_count,
                    basis_family=BASIS_FAMILY_LAPPED_COSINE,
                )
                identity = torch.eye(sample_count, dtype=torch.float64)
                torch.testing.assert_close(
                    basis.transpose(0, 1) @ basis,
                    identity,
                    rtol=0.0,
                    atol=5.0e-12,
                )
                torch.testing.assert_close(
                    basis @ basis.transpose(0, 1),
                    identity,
                    rtol=0.0,
                    atol=5.0e-12,
                )

    def test_02_large_thog_axes_have_orthonormal_retained_columns(self) -> None:
        for sample_count, order in ((768, 96), (3072, 128)):
            with self.subTest(sample_count=sample_count, order=order):
                basis = build_registered_basis(
                    sample_count,
                    order,
                    basis_family=BASIS_FAMILY_LAPPED_COSINE,
                )
                identity = torch.eye(order, dtype=torch.float64)
                torch.testing.assert_close(
                    basis.transpose(0, 1) @ basis,
                    identity,
                    rtol=0.0,
                    atol=5.0e-12,
                )

    def test_03_basis_order_is_prefix_stable_and_balanced_across_blocks(self) -> None:
        full_basis = build_registered_basis(
            144,
            144,
            basis_family=BASIS_FAMILY_LAPPED_COSINE,
        )
        for order in (1, 7, 8, 9, 32, 72):
            with self.subTest(order=order):
                prefix = build_registered_basis(
                    144,
                    order,
                    basis_family=BASIS_FAMILY_LAPPED_COSINE,
                )
                self.assertTrue(torch.equal(prefix, full_basis[:, :order]))
        blocks = lapped_cosine_block_layout(
            144,
            DEFAULT_LAPPED_COSINE_WINDOW_LENGTH,
        )
        schedule = lapped_cosine_column_schedule(
            144,
            len(blocks),
            DEFAULT_LAPPED_COSINE_WINDOW_LENGTH,
        )
        self.assertEqual(tuple(block_index for block_index, _ in schedule), tuple(range(len(blocks))))
        self.assertTrue(all(local_mode == 0 for _, local_mode in schedule))

    def test_04_atoms_have_bounded_contiguous_support_and_no_circular_wrap(self) -> None:
        basis = build_registered_basis(
            144,
            72,
            basis_family=BASIS_FAMILY_LAPPED_COSINE,
        )
        tolerance = 1.0e-14
        for column_index in range(basis.shape[1]):
            indices = torch.nonzero(
                torch.abs(basis[:, column_index]) > tolerance,
                as_tuple=False,
            ).flatten()
            self.assertGreater(indices.numel(), 0)
            support_length = int(indices[-1] - indices[0] + 1)
            self.assertLessEqual(
                support_length,
                DEFAULT_LAPPED_COSINE_WINDOW_LENGTH,
            )
            self.assertFalse(
                bool(torch.abs(basis[0, column_index]) > tolerance)
                and bool(torch.abs(basis[-1, column_index]) > tolerance)
            )

    def test_05_full_basis_reconstructs_vectors(self) -> None:
        basis = build_registered_basis(
            72,
            72,
            basis_family=BASIS_FAMILY_LAPPED_COSINE,
        )
        generator = torch.Generator().manual_seed(1234)
        vector = torch.randn(72, generator=generator, dtype=torch.float64)
        reconstructed = basis @ (basis.transpose(0, 1) @ vector)
        torch.testing.assert_close(reconstructed, vector, rtol=0.0, atol=5.0e-12)

    def test_06_parameterised_version_changes_locality_and_is_canonical(self) -> None:
        version = lapped_cosine_basis_version(48, 0.5)
        self.assertEqual(
            version,
            f"{LAPPED_COSINE_BASIS_VERSION}_w48_o500",
        )
        basis = build_registered_basis(
            144,
            48,
            basis_family=BASIS_FAMILY_LAPPED_COSINE,
            version=version,
        )
        self.assertEqual(basis.shape, (144, 48))
        self.assertLessEqual(
            max(
                int(indices[-1] - indices[0] + 1)
                for indices in (
                    torch.nonzero(torch.abs(basis[:, column]) > 1.0e-14, as_tuple=False).flatten()
                    for column in range(basis.shape[1])
                )
            ),
            48,
        )

    def test_07_registry_alias_metadata_and_artifact_contract(self) -> None:
        for alias in (
            BASIS_FAMILY_LAPPED_COSINE,
            "lapped",
            "local_cosine",
            "lapped_local_cosine",
            LAPPED_COSINE_BASIS_VERSION,
        ):
            with self.subTest(alias=alias):
                self.assertEqual(
                    normalize_registered_basis_family(alias),
                    BASIS_FAMILY_LAPPED_COSINE,
                )
        self.assertEqual(
            basis_version_for_family(BASIS_FAMILY_LAPPED_COSINE),
            LAPPED_COSINE_BASIS_VERSION,
        )
        self.assertEqual(
            basis_artifact_tag_for_family(BASIS_FAMILY_LAPPED_COSINE),
            BASIS_ARTIFACT_TAG_LAPPED_COSINE,
        )
        metadata = get_basis_definition(BASIS_FAMILY_LAPPED_COSINE).metadata()
        self.assertEqual(
            metadata["default_window_length"],
            DEFAULT_LAPPED_COSINE_WINDOW_LENGTH,
        )
        self.assertEqual(
            metadata["default_overlap_fraction"],
            DEFAULT_LAPPED_COSINE_OVERLAP_FRACTION,
        )
        self.assertEqual(metadata["boundary_policy"], "finite_non_circular_v1")

    def test_08_construction_is_deterministic_and_runtime_cast_is_non_trainable(self) -> None:
        version = lapped_cosine_basis_version(24, 0.5)
        first = build_registered_basis(
            144,
            40,
            basis_family=BASIS_FAMILY_LAPPED_COSINE,
            version=version,
            runtime_dtype=torch.float32,
        )
        second = build_registered_basis(
            144,
            40,
            basis_family=BASIS_FAMILY_LAPPED_COSINE,
            version=version,
            runtime_dtype=torch.float32,
        )
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.dtype, torch.float32)
        self.assertFalse(first.requires_grad)
        self.assertTrue(torch.isfinite(first).all())

    def test_09_invalid_controls_versions_and_indices_fail_explicitly(self) -> None:
        indices = lapped_cosine_sample_indices(8)
        with self.assertRaisesRegex(ValueError, "even integer"):
            lapped_cosine_orthonormal_raw_basis(indices, 4, window_length=5)
        with self.assertRaisesRegex(ValueError, "only 0.5"):
            lapped_cosine_orthonormal_raw_basis(indices, 4, overlap_fraction=0.25)
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            lapped_cosine_orthonormal_raw_basis(indices, 9)
        with self.assertRaisesRegex(ValueError, "contiguous sequence"):
            lapped_cosine_orthonormal_raw_basis(torch.tensor([0.0, 2.0]), 2)
        with self.assertRaisesRegex(ValueError, "basis_version mismatch"):
            build_registered_basis(
                8,
                4,
                basis_family=BASIS_FAMILY_LAPPED_COSINE,
                version="wrong_version",
            )
        with self.assertRaisesRegex(ValueError, "floating"):
            build_registered_basis(
                8,
                4,
                basis_family=BASIS_FAMILY_LAPPED_COSINE,
                runtime_dtype=torch.int64,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
""",
)

write(
    "tests/test_lapped_cosine_training_and_checkpoint.py",
    """# vvv THOG
from __future__ import annotations

import unittest
from pathlib import Path

import torch

from sheet.bases import BASIS_FAMILY_LAPPED_COSINE, lapped_cosine_basis_version
from sheet.run_config import OwtRunConfig
from sheet.training_config import TrainingConfig
from sheet.training_model_factory import build_training_model


class LappedCosineTrainingAndCheckpointTests(unittest.TestCase):
    def training_config(self, *, window_length: int = 8) -> TrainingConfig:
        return TrainingConfig(
            model_type="thog2_sheet",
            block_size=8,
            vocab_size=64,
            n_layer=8,
            n_head=2,
            n_embd=16,
            depth_order=4,
            base_row_order=8,
            mlp_channel_order=32,
            o_attn_d_model=8,
            o_attn_qkv_per_channel=4,
            o_attn_out_per_channel=4,
            o_mlp_d_model=8,
            o_mlp_hidden=32,
            basis_family=BASIS_FAMILY_LAPPED_COSINE,
            basis_version=lapped_cosine_basis_version(window_length, 0.5),
            lapped_cosine_window_length=window_length,
            lapped_cosine_overlap_fraction=0.5,
            geometry_preset="depth",
            batch_size=2,
            gradient_accumulation_steps=1,
            max_updates=2,
            warmup_updates=0,
            decay_updates=2,
            eval_interval=0,
            checkpoint_interval=0,
            checkpoint_segment_size=0,
            device="cpu",
            dtype="float32",
        )

    def test_01_tiny_depth_model_forward_backward_and_update_are_finite(self) -> None:
        config = self.training_config()
        model = build_training_model(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        generator = torch.Generator().manual_seed(2026)
        inputs = torch.randint(0, config.vocab_size, (2, 8), generator=generator)
        targets = torch.randint(0, config.vocab_size, (2, 8), generator=generator)
        logits, loss = model(inputs, targets)
        self.assertEqual(logits.shape, (2, 8, config.vocab_size))
        self.assertIsNotNone(loss)
        assert loss is not None
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        optimizer.step()

    def test_02_controls_are_checkpoint_compatibility_fields(self) -> None:
        first = self.training_config(window_length=8)
        second = self.training_config(window_length=12)
        first_signature = first.compatibility_signature()
        second_signature = second.compatibility_signature()
        self.assertEqual(first_signature["lapped_cosine_window_length"], 8)
        self.assertEqual(first_signature["lapped_cosine_overlap_fraction"], 0.5)
        self.assertNotEqual(first_signature, second_signature)
        self.assertNotEqual(first.basis_version, second.basis_version)

    def test_03_run_identity_and_training_config_preserve_controls(self) -> None:
        run = OwtRunConfig(
            model_type="sheet",
            run_name="LAPPED_TEST",
            max_iters=2,
            warmup_iters=1,
            n_layer=8,
            n_head=2,
            n_embd=16,
            block_size=8,
            o_depth=4,
            o_attn_d_model=8,
            o_attn_qkv_per_channel=4,
            o_attn_out_per_channel=4,
            o_mlp_d_model=8,
            o_mlp_hidden=32,
            basis_family=BASIS_FAMILY_LAPPED_COSINE,
            basis_version="auto",
            lapped_cosine_window_length=8,
            lapped_cosine_overlap_fraction=0.5,
            device="cpu",
            dtype="float32",
            wandb_enabled=False,
            wandb_mode="disabled",
        )
        self.assertIn("LAPPED_COSINE_DEPTH", run.artifact_name)
        self.assertIn("LCW_8", run.artifact_name)
        self.assertIn("LCO_50", run.artifact_name)
        training = run.to_training_config(
            vocab_size=64,
            world_size=1,
            out_dir=Path("."),
        )
        self.assertEqual(training.lapped_cosine_window_length, 8)
        self.assertEqual(training.lapped_cosine_overlap_fraction, 0.5)
        self.assertEqual(
            training.basis_version,
            lapped_cosine_basis_version(8, 0.5),
        )

    def test_04_non_lapped_basis_rejects_nondefault_lapped_controls(self) -> None:
        with self.assertRaisesRegex(ValueError, "only when"):
            OwtRunConfig(
                model_type="sheet",
                max_iters=2,
                warmup_iters=1,
                basis_family="haar",
                basis_version="auto",
                lapped_cosine_window_length=24,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
""",
)

write(
    "tests/test_lapped_cosine_wrappers.py",
    """# vvv THOG
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


class LappedCosineWrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.environment = os.environ.copy()
        cls.environment["THOG2_PYTHON"] = sys.executable

    def dry_run(self, wrapper_name: str) -> subprocess.CompletedProcess[str]:
        command = [
            "bash",
            str(self.root / wrapper_name),
            "-p", "depth",
            "-B", "lapped_cosine",
            "-W", "8",
            "-i", "0.5",
            "-x", "true",
            "-g", "LAPPED_TEST",
            "-n", "2",
            "-w", "1",
            "-b", "1",
            "-A", "1",
            "-G", "1",
            "-u", "1",
            "-e", "1",
            "-l", "1",
            "-L", "8",
            "-H", "2",
            "-D", "16",
            "-C", "8",
            "-P", "4",
            "-Q", "8",
            "-J", "4",
            "-O", "4",
            "-X", "8",
            "-Y", "32",
            "-S", "1",
            "-I", "none",
            "-F", "none",
            "-T", "float32",
            "-K", "math",
        ]
        return subprocess.run(
            command,
            cwd=self.root,
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def test_01_primary_wrappers_propagate_controls_and_identity(self) -> None:
        for wrapper_name in (
            "current_scruffy_train_OWT.sh",
            "current_dreedle_train_OWT.sh",
        ):
            with self.subTest(wrapper_name=wrapper_name):
                result = self.dry_run(wrapper_name)
                combined = result.stdout + result.stderr
                self.assertEqual(result.returncode, 0, combined)
                self.assertIn("LAPPED_COSINE_DEPTH", combined)
                self.assertIn("LCW_8", combined)
                self.assertIn("LCO_50", combined)
                self.assertIn("--basis-family lapped_cosine", combined)
                self.assertIn("--lapped-cosine-window-length 8", combined)
                self.assertIn("--lapped-cosine-overlap-fraction 0.5", combined)

    def test_02_invalid_overlap_fails_before_runner_launch(self) -> None:
        result = subprocess.run(
            [
                "bash",
                str(self.root / "current_scruffy_train_OWT.sh"),
                "-B", "lapped_cosine",
                "-i", "0.25",
                "-x", "true",
            ],
            cwd=self.root,
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        combined = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, combined)
        self.assertIn("supports only 0.5", combined)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
""",
)

replace_once(
    "tests/test_basis_family_plugin_registry.py",
    """from sheet.bases.haar import BASIS_ARTIFACT_TAG_HAAR, BASIS_FAMILY_HAAR, HAAR_BASIS_VERSION
""",
    """from sheet.bases.haar import BASIS_ARTIFACT_TAG_HAAR, BASIS_FAMILY_HAAR, HAAR_BASIS_VERSION
from sheet.bases.lapped_cosine import BASIS_ARTIFACT_TAG_LAPPED_COSINE, BASIS_FAMILY_LAPPED_COSINE, LAPPED_COSINE_BASIS_VERSION                                  # <<< THOG appended plugin contract
""",
)
replace_once(
    "tests/test_basis_family_plugin_registry.py",
    """        self.assertEqual(BASIS_FAMILIES, (BASIS_FAMILY_CHEBYSHEV, BASIS_FAMILY_DCT, BASIS_FAMILY_HAAR))
""",
    """        self.assertEqual(BASIS_FAMILIES, (BASIS_FAMILY_CHEBYSHEV, BASIS_FAMILY_DCT, BASIS_FAMILY_HAAR, BASIS_FAMILY_LAPPED_COSINE))
""",
)
replace_once(
    "tests/test_basis_family_plugin_registry.py",
    """        self.assertEqual(normalize_registered_basis_family(HAAR_BASIS_VERSION), BASIS_FAMILY_HAAR)
        self.assertEqual(basis_version_for_family(BASIS_FAMILY_CHEBYSHEV), CHEBYSHEV_BASIS_VERSION)
""",
    """        self.assertEqual(normalize_registered_basis_family(HAAR_BASIS_VERSION), BASIS_FAMILY_HAAR)
        self.assertEqual(normalize_registered_basis_family("lapped"), BASIS_FAMILY_LAPPED_COSINE)
        self.assertEqual(normalize_registered_basis_family(LAPPED_COSINE_BASIS_VERSION), BASIS_FAMILY_LAPPED_COSINE)
        self.assertEqual(basis_version_for_family(BASIS_FAMILY_CHEBYSHEV), CHEBYSHEV_BASIS_VERSION)
""",
)
replace_once(
    "tests/test_basis_family_plugin_registry.py",
    """        self.assertEqual(basis_version_for_family(BASIS_FAMILY_HAAR), HAAR_BASIS_VERSION)
        self.assertEqual(basis_artifact_tag_for_family(BASIS_FAMILY_CHEBYSHEV), BASIS_ARTIFACT_TAG_CHEBYSHEV)
""",
    """        self.assertEqual(basis_version_for_family(BASIS_FAMILY_HAAR), HAAR_BASIS_VERSION)
        self.assertEqual(basis_version_for_family(BASIS_FAMILY_LAPPED_COSINE), LAPPED_COSINE_BASIS_VERSION)
        self.assertEqual(basis_artifact_tag_for_family(BASIS_FAMILY_CHEBYSHEV), BASIS_ARTIFACT_TAG_CHEBYSHEV)
""",
)
replace_once(
    "tests/test_basis_family_plugin_registry.py",
    """        self.assertEqual(basis_artifact_tag_for_family(BASIS_FAMILY_HAAR), BASIS_ARTIFACT_TAG_HAAR)
        self.assertEqual(get_basis_definition("cheby").family, BASIS_FAMILY_CHEBYSHEV)
""",
    """        self.assertEqual(basis_artifact_tag_for_family(BASIS_FAMILY_HAAR), BASIS_ARTIFACT_TAG_HAAR)
        self.assertEqual(basis_artifact_tag_for_family(BASIS_FAMILY_LAPPED_COSINE), BASIS_ARTIFACT_TAG_LAPPED_COSINE)
        self.assertEqual(get_basis_definition("cheby").family, BASIS_FAMILY_CHEBYSHEV)
""",
)
replace_once(
    "tests/test_basis_family_plugin_registry.py",
    """        haar = OwtRunConfig(model_type="sheet", basis_family=BASIS_FAMILY_HAAR, basis_version="auto")
        self.assertEqual(cheby.compact_artifact_fragment(), "CHEBY_DEPTH")
""",
    """        haar = OwtRunConfig(model_type="sheet", basis_family=BASIS_FAMILY_HAAR, basis_version="auto")
        lapped = OwtRunConfig(model_type="sheet", basis_family=BASIS_FAMILY_LAPPED_COSINE, basis_version="auto")
        self.assertEqual(cheby.compact_artifact_fragment(), "CHEBY_DEPTH")
""",
)
replace_once(
    "tests/test_basis_family_plugin_registry.py",
    """        self.assertEqual(haar.compact_artifact_fragment(), "HAAR_DEPTH")
        self.assertEqual(cheby.basis_version, CHEBYSHEV_BASIS_VERSION)
""",
    """        self.assertEqual(haar.compact_artifact_fragment(), "HAAR_DEPTH")
        self.assertEqual(lapped.compact_artifact_fragment(), "LAPPED_COSINE_DEPTH")
        self.assertEqual(cheby.basis_version, CHEBYSHEV_BASIS_VERSION)
""",
)
replace_once(
    "tests/test_basis_family_plugin_registry.py",
    """        self.assertEqual(haar.basis_version, HAAR_BASIS_VERSION)
""",
    """        self.assertEqual(haar.basis_version, HAAR_BASIS_VERSION)
        self.assertEqual(lapped.basis_version, LAPPED_COSINE_BASIS_VERSION)
""",
)

write(".thog_commit_message", "Implement lapped cosine basis and controls\n")
print("lapped cosine enhancement prepared")