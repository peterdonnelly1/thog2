# vvv THOG
from __future__ import annotations

import unittest

import torch

from sheet.basis import BasisCache
from sheet.basis_kernel import CHEBYSHEV_BASIS_VERSION, DCT_BASIS_VERSION, get_basis_kernel
from sheet.basis_registry import (
    BASIS_ARTIFACT_TAG_CHEBYSHEV,
    BASIS_ARTIFACT_TAG_DCT,
    BASIS_FAMILIES,
    BASIS_FAMILY_CHEBYSHEV,
    BASIS_FAMILY_DCT,
    basis_artifact_tag_for_family,
    basis_registry_metadata,
    basis_version_for_family,
    get_basis_spec,
    normalize_registered_basis_family,
)
from sheet.compact_identity import (
    BASIS_FAMILIES as COMPACT_IDENTITY_BASIS_FAMILIES,
    BASIS_FAMILY_CONVENTIONAL,
    normalize_compact_basis_version,
    resolve_compact_selectors,
)


class BasisRegistryShimTests(unittest.TestCase):
    def test_basis_registry_lists_only_current_weight_basis_families(self) -> None:
        self.assertEqual(BASIS_FAMILIES, (BASIS_FAMILY_CHEBYSHEV, BASIS_FAMILY_DCT))
        self.assertNotIn(BASIS_FAMILY_CONVENTIONAL, BASIS_FAMILIES)
        for family in BASIS_FAMILIES:
            self.assertTrue(get_basis_spec(family).supports_weight_basis)

    def test_basis_registry_exposes_stable_artifact_tags_and_versions(self) -> None:
        self.assertEqual(basis_artifact_tag_for_family(BASIS_FAMILY_CHEBYSHEV), BASIS_ARTIFACT_TAG_CHEBYSHEV)
        self.assertEqual(basis_artifact_tag_for_family(BASIS_FAMILY_DCT), BASIS_ARTIFACT_TAG_DCT)
        self.assertEqual(basis_version_for_family(BASIS_FAMILY_CHEBYSHEV), CHEBYSHEV_BASIS_VERSION)
        self.assertEqual(basis_version_for_family(BASIS_FAMILY_DCT), DCT_BASIS_VERSION)
        self.assertEqual(basis_registry_metadata(BASIS_FAMILY_CHEBYSHEV)["artifact_tag"], "CHEBY")
        self.assertEqual(basis_registry_metadata(BASIS_FAMILY_DCT)["artifact_tag"], "DCT")

    def test_basis_registry_normalizes_legacy_aliases_without_changing_family_identity(self) -> None:
        cases = {
            "cheby": BASIS_FAMILY_CHEBYSHEV,
            CHEBYSHEV_BASIS_VERSION: BASIS_FAMILY_CHEBYSHEV,
            "dct_ii": BASIS_FAMILY_DCT,
            DCT_BASIS_VERSION: BASIS_FAMILY_DCT,
        }
        for alias, expected in cases.items():
            with self.subTest(alias=alias):
                self.assertEqual(normalize_registered_basis_family(alias), expected)

    def test_basis_registry_build_basis_matches_existing_basis_kernel_build(self) -> None:
        for family, version in ((BASIS_FAMILY_CHEBYSHEV, CHEBYSHEV_BASIS_VERSION), (BASIS_FAMILY_DCT, DCT_BASIS_VERSION)):
            with self.subTest(family=family):
                expected = get_basis_kernel(family).build(8, 4, runtime_dtype=torch.float32, device="cpu", version=version)
                actual = get_basis_spec(family).basis_matrix(8, 4, runtime_dtype=torch.float32, device="cpu", version=version)
                self.assertTrue(torch.equal(actual, expected))

    def test_basis_registry_cache_key_uses_canonical_family_and_version(self) -> None:
        key = BasisCache.make_key(8, 4, runtime_dtype=torch.float32, device="cpu", version=CHEBYSHEV_BASIS_VERSION, basis_family="cheby")
        self.assertEqual(key.basis_family, BASIS_FAMILY_CHEBYSHEV)
        with self.assertRaisesRegex(ValueError, "basis_version mismatch"):
            BasisCache.make_key(8, 4, runtime_dtype=torch.float32, device="cpu", version=DCT_BASIS_VERSION, basis_family="cheby")

    def test_compact_identity_uses_registry_basis_families_and_versions(self) -> None:
        self.assertEqual(COMPACT_IDENTITY_BASIS_FAMILIES, (*BASIS_FAMILIES, BASIS_FAMILY_CONVENTIONAL))
        dct_selectors = resolve_compact_selectors(geometry_preset="depth", basis_family=BASIS_FAMILY_DCT)
        self.assertEqual(normalize_compact_basis_version(dct_selectors, CHEBYSHEV_BASIS_VERSION), DCT_BASIS_VERSION)
        cheb_selectors = resolve_compact_selectors(geometry_preset="depth", basis_family=BASIS_FAMILY_CHEBYSHEV)
        self.assertEqual(normalize_compact_basis_version(cheb_selectors, CHEBYSHEV_BASIS_VERSION), CHEBYSHEV_BASIS_VERSION)

    def test_basis_registry_marks_native_products_as_deferred(self) -> None:
        self.assertFalse(get_basis_spec(BASIS_FAMILY_CHEBYSHEV).supports_native_products)
        self.assertFalse(get_basis_spec(BASIS_FAMILY_DCT).supports_native_products)

    def test_unknown_basis_family_fails_at_registry_boundary(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown basis_family"):
            get_basis_spec("legendre")
        with self.assertRaisesRegex(ValueError, "unknown basis_family"):
            normalize_registered_basis_family("fourier")


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
