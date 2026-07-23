# vvv THOG
from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from .bases import (
    BASIS_FAMILIES,
    BASIS_FAMILY_CHEBYSHEV,
    BASIS_FAMILY_DCT,
    CHEBYSHEV_BASIS_VERSION,
    DCT_BASIS_VERSION,
    SINGLE_POINT_COORDINATE,
    BasisKernel,
    ChebyshevQrBasisKernel,
    DctIiOrthonormalBasisKernel,
    basis_artifact_tag_for_family,
    basis_kernel_metadata,
    basis_registry_metadata,
    basis_version_for_family,
    build_registered_basis,
    chebyshev_coordinates,
    chebyshev_raw_basis,
    dct_ii_orthonormal_raw_basis,
    dct_sample_indices,
    deterministic_reduced_qr_positive_diagonal,
    get_basis_definition,
    get_basis_kernel,
    get_basis_spec,
    normalize_basis_family,
    normalize_basis_version,
    normalize_registered_basis_family,
    registered_basis_families,
    validate_floating_dtype,
    validate_positive_integer,
)


_BASIS_KERNELS: Mapping[str, BasisKernel] = MappingProxyType({family: get_basis_kernel(family) for family in BASIS_FAMILIES})


__all__ = [
    "BASIS_FAMILIES",
    "BASIS_FAMILY_CHEBYSHEV",
    "BASIS_FAMILY_DCT",
    "CHEBYSHEV_BASIS_VERSION",
    "DCT_BASIS_VERSION",
    "SINGLE_POINT_COORDINATE",
    "BasisKernel",
    "ChebyshevQrBasisKernel",
    "DctIiOrthonormalBasisKernel",
    "basis_artifact_tag_for_family",
    "basis_kernel_metadata",
    "basis_registry_metadata",
    "basis_version_for_family",
    "build_registered_basis",
    "chebyshev_coordinates",
    "chebyshev_raw_basis",
    "dct_ii_orthonormal_raw_basis",
    "dct_sample_indices",
    "deterministic_reduced_qr_positive_diagonal",
    "get_basis_definition",
    "get_basis_kernel",
    "get_basis_spec",
    "normalize_basis_family",
    "normalize_basis_version",
    "normalize_registered_basis_family",
    "registered_basis_families",
    "validate_floating_dtype",
    "validate_positive_integer",
]
# ^^^ THOG
