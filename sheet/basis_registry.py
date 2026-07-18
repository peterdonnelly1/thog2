# vvv THOG
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple, Union

import torch
from torch import Tensor

from .basis_kernel import (
    BASIS_FAMILY_CHEBYSHEV,
    BASIS_FAMILY_DCT,
    CHEBYSHEV_BASIS_VERSION,
    DCT_BASIS_VERSION,
    BasisKernel,
    get_basis_kernel,
    normalize_basis_family,
)


DeviceLike = Union[str, torch.device]
BASIS_ARTIFACT_TAG_CHEBYSHEV = "CHEBY"
BASIS_ARTIFACT_TAG_DCT = "DCT"


@dataclass(frozen=True)
class BasisSpec:
    basis_family: str
    artifact_tag: str
    basis_version: str
    supports_weight_basis: bool
    supports_native_products: bool
    kernel: BasisKernel

    def basis_matrix(
        self,
        sample_count: int,
        order: int,
        *,
        runtime_dtype: torch.dtype = torch.float64,
        device: Optional[DeviceLike] = None,
        version: Optional[str] = None,
    ) -> Tensor:
        return self.kernel.build(sample_count, order, runtime_dtype=runtime_dtype, device=device, version=version or self.basis_version)

    def metadata(self) -> Dict[str, object]:
        return {
            "basis_family": self.basis_family,
            "artifact_tag": self.artifact_tag,
            "basis_version": self.basis_version,
            "supports_weight_basis": self.supports_weight_basis,
            "supports_native_products": self.supports_native_products,
            **self.kernel.metadata(),
        }


BASIS_REGISTRY: Mapping[str, BasisSpec] = {
    BASIS_FAMILY_CHEBYSHEV: BasisSpec(
        basis_family=BASIS_FAMILY_CHEBYSHEV,
        artifact_tag=BASIS_ARTIFACT_TAG_CHEBYSHEV,
        basis_version=CHEBYSHEV_BASIS_VERSION,
        supports_weight_basis=True,
        supports_native_products=False,
        kernel=get_basis_kernel(BASIS_FAMILY_CHEBYSHEV),
    ),
    BASIS_FAMILY_DCT: BasisSpec(
        basis_family=BASIS_FAMILY_DCT,
        artifact_tag=BASIS_ARTIFACT_TAG_DCT,
        basis_version=DCT_BASIS_VERSION,
        supports_weight_basis=True,
        supports_native_products=False,
        kernel=get_basis_kernel(BASIS_FAMILY_DCT),
    ),
}

BASIS_FAMILIES: Tuple[str, ...] = tuple(BASIS_REGISTRY.keys())


def normalize_registered_basis_family(basis_family: str) -> str:
    normalized = normalize_basis_family(basis_family)
    if normalized not in BASIS_REGISTRY:
        raise ValueError(f"unknown basis_family: {basis_family!r}")
    return normalized


def get_basis_spec(basis_family: str = BASIS_FAMILY_CHEBYSHEV) -> BasisSpec:
    return BASIS_REGISTRY[normalize_registered_basis_family(basis_family)]


def basis_version_for_family(basis_family: str = BASIS_FAMILY_CHEBYSHEV) -> str:
    return get_basis_spec(basis_family).basis_version


def basis_artifact_tag_for_family(basis_family: str = BASIS_FAMILY_CHEBYSHEV) -> str:
    return get_basis_spec(basis_family).artifact_tag


def basis_registry_metadata(basis_family: str = BASIS_FAMILY_CHEBYSHEV) -> Dict[str, object]:
    return get_basis_spec(basis_family).metadata()


def build_registered_basis(
    sample_count: int,
    order: int,
    *,
    runtime_dtype: torch.dtype = torch.float64,
    device: Optional[DeviceLike] = None,
    version: Optional[str] = None,
    basis_family: str = BASIS_FAMILY_CHEBYSHEV,
) -> Tensor:
    return get_basis_spec(basis_family).basis_matrix(sample_count, order, runtime_dtype=runtime_dtype, device=device, version=version)


__all__ = [
    "BASIS_ARTIFACT_TAG_CHEBYSHEV",
    "BASIS_ARTIFACT_TAG_DCT",
    "BASIS_FAMILIES",
    "BASIS_FAMILY_CHEBYSHEV",
    "BASIS_FAMILY_DCT",
    "BASIS_REGISTRY",
    "CHEBYSHEV_BASIS_VERSION",
    "DCT_BASIS_VERSION",
    "BasisSpec",
    "basis_artifact_tag_for_family",
    "basis_registry_metadata",
    "basis_version_for_family",
    "build_registered_basis",
    "get_basis_spec",
    "normalize_registered_basis_family",
]
# ^^^ THOG
