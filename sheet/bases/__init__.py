# vvv THOG
from .protocol import BasisDefinition, BasisKernel, DeviceLike, deterministic_reduced_qr_positive_diagonal, validate_floating_dtype, validate_positive_integer
from .chebyshev import BASIS_ARTIFACT_TAG_CHEBYSHEV, BASIS_FAMILY_CHEBYSHEV, CHEBYSHEV_BASIS_VERSION, SINGLE_POINT_COORDINATE, ChebyshevQrBasisKernel, chebyshev_coordinates, chebyshev_raw_basis
from .dct import BASIS_ARTIFACT_TAG_DCT, BASIS_FAMILY_DCT, DCT_BASIS_VERSION, DctIiOrthonormalBasisKernel, dct_ii_orthonormal_raw_basis, dct_sample_indices
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


BasisSpec = BasisDefinition


__all__ = [
    "BASIS_ARTIFACT_TAG_CHEBYSHEV",
    "BASIS_ARTIFACT_TAG_DCT",
    "BASIS_ARTIFACT_TAG_LAPPED_COSINE",                                                                                                                  # <<< THOG public lapped cosine artifact tag
    "BASIS_FAMILIES",
    "BASIS_FAMILY_CHEBYSHEV",
    "BASIS_FAMILY_DCT",
    "BASIS_FAMILY_LAPPED_COSINE",                                                                                                                        # <<< THOG public lapped cosine family
    "BASIS_REGISTRY",
    "BUILTIN_BASIS_MODULES",
    "CHEBYSHEV_BASIS_VERSION",
    "DCT_BASIS_VERSION",
    "LAPPED_COSINE_BASIS_VERSION",                                                                                                                       # <<< THOG public lapped cosine version
    "DEFAULT_LAPPED_COSINE_OVERLAP_FRACTION",                                                                                                            # <<< THOG public lapped overlap default
    "DEFAULT_LAPPED_COSINE_WINDOW_LENGTH",                                                                                                               # <<< THOG public lapped locality default
    "SINGLE_POINT_COORDINATE",
    "BasisDefinition",
    "BasisKernel",
    "BasisRegistry",
    "BasisSpec",
    "ChebyshevQrBasisKernel",
    "DctIiOrthonormalBasisKernel",
    "LappedCosineOrthonormalBasisKernel",                                                                                                                # <<< THOG public lapped kernel
    "DeviceLike",
    "basis_artifact_tag_for_family",
    "basis_kernel_metadata",
    "basis_registry_metadata",
    "basis_version_for_family",
    "build_registered_basis",
    "chebyshev_coordinates",
    "chebyshev_raw_basis",
    "dct_ii_orthonormal_raw_basis",
    "dct_sample_indices",
    "lapped_cosine_basis_version",                                                                                                                       # <<< THOG lapped version builder
    "lapped_cosine_block_layout",                                                                                                                        # <<< THOG lapped block diagnostics
    "lapped_cosine_column_schedule",                                                                                                                     # <<< THOG balanced prefix diagnostics
    "lapped_cosine_orthonormal_raw_basis",                                                                                                               # <<< THOG lapped raw basis
    "lapped_cosine_sample_indices",                                                                                                                      # <<< THOG lapped coordinates
    "normalize_lapped_cosine_basis_version",                                                                                                             # <<< THOG lapped version normalisation
    "parse_lapped_cosine_basis_version",                                                                                                                 # <<< THOG lapped version parsing
    "validate_lapped_cosine_controls",                                                                                                                   # <<< THOG lapped control validation
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
