# vvv THOG
from __future__ import annotations

import math
import unittest

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

    def test_02_first_column_is_the_exact_global_dc_initialization_direction(self) -> None:
        for sample_count, window_length in (
            (1, 36),
            (8, 8),
            (17, 12),
            (72, 24),
            (144, 36),
            (768, 48),
        ):
            with self.subTest(sample_count=sample_count, window_length=window_length):
                version = lapped_cosine_basis_version(window_length, 0.5)
                basis = build_registered_basis(
                    sample_count,
                    min(sample_count, 16),
                    basis_family=BASIS_FAMILY_LAPPED_COSINE,
                    version=version,
                )
                expected = torch.full(
                    (sample_count,),
                    1.0 / math.sqrt(float(sample_count)),
                    dtype=torch.float64,
                )
                torch.testing.assert_close(
                    basis[:, 0],
                    expected,
                    rtol=0.0,
                    atol=2.0e-15,
                )

    def test_03_large_thog_axes_have_orthonormal_retained_columns(self) -> None:
        for sample_count, order in ((768, 96), (3072, 256)):
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

    def test_04_basis_order_is_prefix_stable_and_balanced_across_blocks(self) -> None:
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
                torch.testing.assert_close(
                    prefix,
                    full_basis[:, :order],
                    rtol=0.0,
                    atol=1.0e-15,
                )

        blocks = lapped_cosine_block_layout(
            144,
            DEFAULT_LAPPED_COSINE_WINDOW_LENGTH,
        )
        schedule = lapped_cosine_column_schedule(
            144,
            2 * len(blocks),
            DEFAULT_LAPPED_COSINE_WINDOW_LENGTH,
        )
        coarse = schedule[: len(blocks)]
        first_detail_round = schedule[len(blocks) :]
        self.assertEqual(
            coarse,
            tuple(("coarse", index, 0) for index in range(len(blocks))),
        )
        self.assertTrue(all(kind == "detail" for kind, _, _ in first_detail_round))
        self.assertEqual(
            {block_index for _, block_index, _ in first_detail_round},
            set(range(len(blocks))),
        )
        self.assertTrue(all(local_mode == 1 for _, _, local_mode in first_detail_round))

    def test_05_detail_atoms_have_bounded_contiguous_support_and_no_circular_wrap(self) -> None:
        basis = build_registered_basis(
            144,
            72,
            basis_family=BASIS_FAMILY_LAPPED_COSINE,
        )
        blocks = lapped_cosine_block_layout(
            144,
            DEFAULT_LAPPED_COSINE_WINDOW_LENGTH,
        )
        tolerance = 1.0e-14
        for column_index in range(len(blocks), basis.shape[1]):
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

    def test_06_full_basis_reconstructs_vectors(self) -> None:
        basis = build_registered_basis(
            72,
            72,
            basis_family=BASIS_FAMILY_LAPPED_COSINE,
        )
        generator = torch.Generator().manual_seed(1234)
        vector = torch.randn(72, generator=generator, dtype=torch.float64)
        reconstructed = basis @ (basis.transpose(0, 1) @ vector)
        torch.testing.assert_close(reconstructed, vector, rtol=0.0, atol=5.0e-12)

    def test_07_parameterised_version_changes_locality_and_is_canonical(self) -> None:
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
        blocks = lapped_cosine_block_layout(144, 48)
        self.assertEqual(basis.shape, (144, 48))
        self.assertLessEqual(
            max(
                int(indices[-1] - indices[0] + 1)
                for indices in (
                    torch.nonzero(
                        torch.abs(basis[:, column]) > 1.0e-14,
                        as_tuple=False,
                    ).flatten()
                    for column in range(len(blocks), basis.shape[1])
                )
            ),
            48,
        )

    def test_08_registry_alias_metadata_and_artifact_contract(self) -> None:
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
        self.assertEqual(
            metadata["initialization_contract"],
            "normalized_global_dc_first_column_v1",
        )

    def test_09_construction_is_deterministic_and_runtime_cast_is_non_trainable(self) -> None:
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

    def test_10_invalid_controls_versions_and_indices_fail_explicitly(self) -> None:
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
