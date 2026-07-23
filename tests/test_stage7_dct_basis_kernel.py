# vvv THOG
from __future__ import annotations

import math
import unittest

import torch

from sheet.basis import BasisCache, build_stabilized_basis, orthonormality_max_error
from sheet.basis_kernel import (
    BASIS_FAMILY_CHEBYSHEV,
    BASIS_FAMILY_DCT,
    CHEBYSHEV_BASIS_VERSION,
    DCT_BASIS_VERSION,
    basis_kernel_metadata,
    get_basis_kernel,
)


class Stage7DctBasisKernelTests(unittest.TestCase):
    def reference_dct_basis(self, sample_count: int, order: int) -> torch.Tensor:
        rows = torch.arange(sample_count, dtype=torch.float64).unsqueeze(1)
        cols = torch.arange(order, dtype=torch.float64).unsqueeze(0)
        basis = torch.cos(math.pi * (rows + 0.5) * cols / float(sample_count))
        basis[:, 0] *= math.sqrt(1.0 / float(sample_count))
        if order > 1:
            basis[:, 1:] *= math.sqrt(2.0 / float(sample_count))
        return basis

    def test_01_dct_ii_orthonormal_kernel_matches_closed_form_formula_without_qr_sign_ambiguity(self) -> None:
        basis = build_stabilized_basis(8, 5, runtime_dtype=torch.float64, version=DCT_BASIS_VERSION, basis_family=BASIS_FAMILY_DCT)
        expected = self.reference_dct_basis(8, 5)
        self.assertEqual(tuple(basis.shape), (8, 5))
        torch.testing.assert_close(basis, expected, rtol=0.0, atol=1.0e-14)
        self.assertLess(orthonormality_max_error(basis), 1.0e-14)

    def test_02_dct_kernel_metadata_records_dct_family_version_and_no_qr_stabilization_policy(self) -> None:
        metadata = basis_kernel_metadata(BASIS_FAMILY_DCT)
        self.assertEqual(metadata["basis_family"], BASIS_FAMILY_DCT)
        self.assertEqual(metadata["basis_version"], DCT_BASIS_VERSION)
        self.assertEqual(metadata["stabilization_policy"], "closed_form_dct_ii_orthonormal_no_qr_v1")
        self.assertEqual(get_basis_kernel("dct_ii_orthonormal").basis_version, DCT_BASIS_VERSION)

    def test_03_chebyshev_and_dct_cache_entries_are_separated_by_family_version_dtype_and_device_even_for_the_same_shape(self) -> None:
        cache = BasisCache()
        cheby = cache.get(8, 4, runtime_dtype=torch.float32, version=CHEBYSHEV_BASIS_VERSION, basis_family=BASIS_FAMILY_CHEBYSHEV)
        dct = cache.get(8, 4, runtime_dtype=torch.float32, version=DCT_BASIS_VERSION, basis_family=BASIS_FAMILY_DCT)
        self.assertEqual(len(cache), 2)
        self.assertEqual(tuple(cheby.shape), tuple(dct.shape))
        self.assertFalse(torch.allclose(cheby, dct))
        self.assertIs(dct, cache.get(8, 4, runtime_dtype=torch.float32, version=DCT_BASIS_VERSION, basis_family=BASIS_FAMILY_DCT))

    def test_04_dct_kernel_rejects_wrong_basis_version_invalid_order_unknown_family_and_nonfloating_runtime_dtype(self) -> None:
        with self.assertRaisesRegex(ValueError, "basis_version"):
            build_stabilized_basis(8, 4, version=CHEBYSHEV_BASIS_VERSION, basis_family=BASIS_FAMILY_DCT)
        with self.assertRaisesRegex(ValueError, "order must not exceed sample_count"):
            build_stabilized_basis(3, 4, version=DCT_BASIS_VERSION, basis_family=BASIS_FAMILY_DCT)
        with self.assertRaisesRegex(ValueError, "unknown basis_family"):
            build_stabilized_basis(8, 4, basis_family="not_a_basis")
        with self.assertRaisesRegex(ValueError, "dtype must be floating point"):
            build_stabilized_basis(8, 4, runtime_dtype=torch.int64, version=DCT_BASIS_VERSION, basis_family=BASIS_FAMILY_DCT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
