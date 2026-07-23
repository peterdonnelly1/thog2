# vvv THOG
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor


DeviceLike = Union[str, torch.device]
_FAMILY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_ARTIFACT_TAG_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")


def validate_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")


def validate_floating_dtype(dtype: torch.dtype) -> None:
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError(f"dtype must be floating point; got {dtype}")


def deterministic_reduced_qr_positive_diagonal(raw_basis: Tensor) -> Tuple[Tensor, Tensor]:
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

    # vvv THOG permit basis families with parameterised but canonical version strings
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

    def stabilize(self, raw_basis: Tensor) -> Tensor:
        raise NotImplementedError

    def build(self, sample_count: int, order: int, *, runtime_dtype: torch.dtype = torch.float64, device: Optional[DeviceLike] = None, version: Optional[str] = None) -> Tensor:
        validate_positive_integer("sample_count", sample_count)
        validate_positive_integer("order", order)
        if order > sample_count:
            raise ValueError(f"order must not exceed sample_count; got order={order}, sample_count={sample_count}")
        validate_floating_dtype(runtime_dtype)
        # vvv THOG version validation is delegated to the basis kernel
        basis_version = self.normalize_version(version)
        # ^^^ THOG
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
        return {
            "basis_family": self.basis_family,
            "basis_version": self.basis_version,
            "coordinate_policy": self.coordinate_policy,
            "stabilization_policy": self.stabilization_policy,
        }


@dataclass(frozen=True)
class BasisDefinition:
    family: str
    aliases: Tuple[str, ...]
    version: str
    artifact_tag: str
    supports_weight_basis: bool
    supports_native_products: bool
    kernel: BasisKernel

    def __post_init__(self) -> None:
        family = self.family.strip().lower()
        aliases = tuple(alias.strip().lower() for alias in self.aliases)
        version = self.version.strip().lower()
        artifact_tag = self.artifact_tag.strip().upper()
        if not _FAMILY_PATTERN.fullmatch(family):
            raise ValueError(f"invalid basis family: {self.family!r}")
        if not version:
            raise ValueError("basis version must be non-empty")
        if not _ARTIFACT_TAG_PATTERN.fullmatch(artifact_tag):
            raise ValueError(f"invalid basis artifact tag: {self.artifact_tag!r}")
        if any(not alias for alias in aliases):
            raise ValueError("basis aliases must be non-empty")
        if len(set(aliases)) != len(aliases):
            raise ValueError(f"duplicate aliases within basis definition {family!r}")
        if self.kernel.basis_family != family:
            raise ValueError(f"kernel family mismatch: definition={family!r}, kernel={self.kernel.basis_family!r}")
        if self.kernel.basis_version != version:
            raise ValueError(f"kernel version mismatch: definition={version!r}, kernel={self.kernel.basis_version!r}")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "artifact_tag", artifact_tag)

    def build(self, sample_count: int, order: int, *, runtime_dtype: torch.dtype = torch.float64, device: Optional[DeviceLike] = None, version: Optional[str] = None) -> Tensor:
        return self.kernel.build(sample_count, order, runtime_dtype=runtime_dtype, device=device, version=version or self.version)

    def metadata(self) -> Dict[str, object]:
        return {
            "basis_family": self.family,
            "basis_aliases": self.aliases,
            "basis_version": self.version,
            "artifact_tag": self.artifact_tag,
            "supports_weight_basis": self.supports_weight_basis,
            "supports_native_products": self.supports_native_products,
            **self.kernel.metadata(),
        }
# ^^^ THOG
