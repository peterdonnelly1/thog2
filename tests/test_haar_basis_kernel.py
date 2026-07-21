# vvv THOG
from __future__ import annotations

import math
import unittest

import torch

from sheet.bases import (
    BASIS_FAMILIES,
    basis_artifact_tag_for_family,
    basis_version_for_family,
    build_registered_basis,
    get_basis_definition,
    normalize_registered_basis_family,
)
from sheet.bases.haar import (
    BASIS_ARTIFACT_TAG_HAAR,
    BASIS_FAMILY_HAAR,
    HAAR_BASIS_VERSION,
    BalancedHaarOrthonormalBasisKernel,
    haar_balanced_orthonormal_raw_basis,
    haar_sample_indices,
)


class BalancedHaarBasisKernelTests(unittest.TestCase):
    def test_01_closed_form_small_bases(self) -> None:
        basis_1 = build_registered_basis(1, 1, basis_family=BASIS_FAMILY_HAAR)
        torch.testing.assert_close(basis_1, torch.ones((1, 1), dtype=torch.float64), rtol=0.0, atol=0.0)

        inverse_sqrt_2 = 1.0 / math.sqrt(2.0)
        float64_ulp = torch.finfo(torch.float64).eps
        expected_2 = torch.tensor(
            [
                [inverse_sqrt_2, inverse_sqrt_2],
                [inverse_sqrt_2, -inverse_sqrt_2],
            ],
            dtype=torch.float64,
        )
        basis_2 = build_registered_basis(2, 2, basis_family=BASIS_FAMILY_HAAR)
        torch.testing.assert_close(basis_2, expected_2, rtol=0.0, atol=float64_ulp)

        expected_4 = torch.tensor(
            [
                [0.5, 0.5, inverse_sqrt_2, 0.0],
                [0.5, 0.5, -inverse_sqrt_2, 0.0],
                [0.5, -0.5, 0.0, inverse_sqrt_2],
                [0.5, -0.5, 0.0, -inverse_sqrt_2],
            ],
            dtype=torch.float64,
        )
        basis_4 = build_registered_basis(4, 4, basis_family=BASIS_FAMILY_HAAR)
        torch.testing.assert_close(basis_4, expected_4, rtol=0.0, atol=float64_ulp)

    def test_02_full_bases_are_orthonormal_for_odd_even_and_thog_axis_lengths(self) -> None:
        for sample_count in (1, 2, 3, 5, 8, 9, 16, 31, 144):
            with self.subTest(sample_count=sample_count):
                basis = build_registered_basis(sample_count, sample_count, basis_family=BASIS_FAMILY_HAAR)
                identity = torch.eye(sample_count, dtype=torch.float64)
                torch.testing.assert_close(basis.transpose(0, 1) @ basis, identity, rtol=0.0, atol=1.0e-12)
                torch.testing.assert_close(basis @ basis.transpose(0, 1), identity, rtol=0.0, atol=1.0e-12)
                if sample_count > 1:
                    torch.testing.assert_close(
                        basis[:, 1:].sum(dim=0),
                        torch.zeros(sample_count - 1, dtype=torch.float64),
                        rtol=0.0,
                        atol=1.0e-12,
                    )

    def test_03_basis_order_is_prefix_stable(self) -> None:
        for sample_count in (3, 5, 8, 9, 17, 144):
            full_basis = build_registered_basis(sample_count, sample_count, basis_family=BASIS_FAMILY_HAAR)
            orders = sorted({1, min(2, sample_count), min(3, sample_count), min(8, sample_count), sample_count})
            for order in orders:
                with self.subTest(sample_count=sample_count, order=order):
                    prefix = build_registered_basis(sample_count, order, basis_family=BASIS_FAMILY_HAAR)
                    self.assertTrue(torch.equal(prefix, full_basis[:, :order]))

    def test_04_construction_is_deterministic_and_preserves_runtime_contract(self) -> None:
        first = build_registered_basis(17, 11, basis_family=BASIS_FAMILY_HAAR, runtime_dtype=torch.float32, device="cpu")
        second = build_registered_basis(17, 11, basis_family=BASIS_FAMILY_HAAR, runtime_dtype=torch.float32, device=torch.device("cpu"))
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.shape, (17, 11))
        self.assertEqual(first.dtype, torch.float32)
        self.assertEqual(first.device.type, "cpu")
        self.assertFalse(first.requires_grad)
        self.assertTrue(torch.isfinite(first).all())

    def test_05_registry_metadata_aliases_and_version_are_exact(self) -> None:
        self.assertIn(BASIS_FAMILY_HAAR, BASIS_FAMILIES)
        for alias in (BASIS_FAMILY_HAAR, "balanced_haar", "haar_balanced", HAAR_BASIS_VERSION):
            with self.subTest(alias=alias):
                self.assertEqual(normalize_registered_basis_family(alias), BASIS_FAMILY_HAAR)
        self.assertEqual(basis_version_for_family(BASIS_FAMILY_HAAR), HAAR_BASIS_VERSION)
        self.assertEqual(basis_artifact_tag_for_family(BASIS_FAMILY_HAAR), BASIS_ARTIFACT_TAG_HAAR)
        definition = get_basis_definition(BASIS_FAMILY_HAAR)
        self.assertIsInstance(definition.kernel, BalancedHaarOrthonormalBasisKernel)
        self.assertEqual(definition.metadata()["coordinate_policy"], "integer_sample_index_balanced_binary_partition_v1")
        self.assertEqual(definition.metadata()["stabilization_policy"], "closed_form_balanced_haar_orthonormal_no_qr_v1")

    def test_06_invalid_inputs_fail_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample_count"):
            haar_sample_indices(0)
        with self.assertRaisesRegex(ValueError, "order"):
            haar_balanced_orthonormal_raw_basis(haar_sample_indices(4), 0)
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            haar_balanced_orthonormal_raw_basis(haar_sample_indices(4), 5)
        with self.assertRaisesRegex(ValueError, "contiguous sequence"):
            haar_balanced_orthonormal_raw_basis(torch.tensor([0.0, 2.0]), 2)
        with self.assertRaisesRegex(ValueError, "floating"):
            build_registered_basis(4, 2, basis_family=BASIS_FAMILY_HAAR, runtime_dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, "basis_version mismatch"):
            build_registered_basis(4, 2, basis_family=BASIS_FAMILY_HAAR, version="wrong_version")


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
