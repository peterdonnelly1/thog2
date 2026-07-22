# vvv THOG
from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any, Dict

import torch

from sheet.basis import (
    BASIS_FAMILY_CHEBYSHEV,
    BASIS_FAMILY_DCT,
    BASIS_VERSION,
    DCT_BASIS_VERSION,
    BasisCache,
    BasisOwner,
    basis_kernel_metadata,
    basis_sha256,
    build_stabilized_basis,
    chebyshev_first_kind_basis,
    deterministic_reduced_qr,
    get_basis_kernel,
    normalized_coordinates,
    orthonormality_max_error,
)
from sheet.model import SheetGPTConfig
from sheet.trajectory import SheetTrajectory


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "stage0_legacy_sheet_col_fixture.json"


class Stage2BasisKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with FIXTURE_PATH.open("r", encoding="utf-8") as handle:
            cls.fixture: Dict[str, Any] = json.load(handle)

    def config(self) -> SheetGPTConfig:
        return SheetGPTConfig(**self.fixture["config"])

    def trajectory(self) -> SheetTrajectory:
        torch.manual_seed(self.fixture["seed"])
        config = self.config()
        return SheetTrajectory(config.sheet_geometry(), runtime_dtype=torch.float32, basis_version=config.basis_version, basis_family=BASIS_FAMILY_CHEBYSHEV)

    def test_kernel_registry_and_metadata(self) -> None:
        kernel = get_basis_kernel("chebyshev")
        self.assertIs(kernel, get_basis_kernel("cheby"))
        self.assertIs(kernel, get_basis_kernel(BASIS_VERSION))
        self.assertEqual(kernel.basis_family, BASIS_FAMILY_CHEBYSHEV)
        self.assertEqual(kernel.basis_version, BASIS_VERSION)
        metadata = basis_kernel_metadata("chebyshev")
        self.assertEqual(metadata["basis_family"], BASIS_FAMILY_CHEBYSHEV)
        self.assertEqual(metadata["basis_version"], BASIS_VERSION)
        self.assertIn("single_point_zero", metadata["coordinate_policy"])
        self.assertIn("positive_diagonal", metadata["stabilization_policy"])
        dct_kernel = get_basis_kernel("dct")
        self.assertIs(dct_kernel, get_basis_kernel(DCT_BASIS_VERSION))
        self.assertEqual(dct_kernel.basis_family, BASIS_FAMILY_DCT)
        self.assertEqual(dct_kernel.basis_version, DCT_BASIS_VERSION)
        with self.assertRaisesRegex(ValueError, "unknown"):
            get_basis_kernel("made_up_basis")

    def test_cheby_kernel_matches_legacy_build_and_stage0_hashes(self) -> None:
        kernel = get_basis_kernel("chebyshev")
        for sample_count, order in ((1, 1), (4, 3), (16, 8), (48, 24), (64, 32)):
            with self.subTest(sample_count=sample_count, order=order):
                legacy = build_stabilized_basis(sample_count, order, runtime_dtype=torch.float32)
                through_kernel = kernel.build(sample_count, order, runtime_dtype=torch.float32)
                torch.testing.assert_close(through_kernel, legacy, rtol=0.0, atol=0.0)
        trajectory = self.trajectory()
        expected_hashes = self.fixture["basis_hashes"]
        self.assertEqual(basis_sha256(trajectory.depth_basis), expected_hashes["depth_4_order_3_float32"])
        self.assertEqual(basis_sha256(trajectory.row_basis("attention_input_weight")), expected_hashes["row_16_order_8_float32"])
        self.assertEqual(basis_sha256(trajectory.row_basis("attention_input_bias")), expected_hashes["row_48_order_24_float32"])
        self.assertEqual(basis_sha256(trajectory.row_basis("mlp_contraction_weight")), expected_hashes["row_64_order_32_float32"])

    def test_coordinate_recurrence_qr_and_orthonormality_behaviour_is_unchanged(self) -> None:
        one = normalized_coordinates(1)
        self.assertEqual(float(one[0]), 0.0)
        coordinates = normalized_coordinates(7)
        self.assertEqual(float(coordinates[0]), -1.0)
        self.assertEqual(float(coordinates[-1]), 1.0)
        raw = chebyshev_first_kind_basis(coordinates, 6)
        analytic = torch.cos(torch.acos(coordinates).unsqueeze(1) * torch.arange(6, dtype=torch.float64).unsqueeze(0))
        torch.testing.assert_close(raw, analytic, rtol=0.0, atol=2.0e-14)
        stabilized, r_matrix = deterministic_reduced_qr(raw)
        self.assertTrue(torch.all(torch.diagonal(r_matrix) > 0.0))
        self.assertLessEqual(orthonormality_max_error(stabilized), 2.0e-12)

    def test_cache_key_includes_family_version_dtype_and_device(self) -> None:
        cache = BasisCache()
        first = cache.get(16, 8, runtime_dtype=torch.float32, basis_family="chebyshev")
        second = cache.get(16, 8, runtime_dtype=torch.float32, basis_family="cheby")
        self.assertIs(first, second)
        dtype_variant = cache.get(16, 8, runtime_dtype=torch.float64, basis_family="chebyshev")
        dct_variant = cache.get(16, 8, runtime_dtype=torch.float32, version=DCT_BASIS_VERSION, basis_family="dct")
        self.assertIsNot(first, dtype_variant)
        self.assertIsNot(first, dct_variant)
        self.assertEqual(len(cache), 3)
        key = cache.make_key(16, 8, runtime_dtype=torch.float32, device="cpu", version=BASIS_VERSION, basis_family="chebyshev")
        self.assertEqual(key.basis_family, BASIS_FAMILY_CHEBYSHEV)
        self.assertEqual(key.version, BASIS_VERSION)
        dct_key = cache.make_key(16, 8, runtime_dtype=torch.float32, device="cpu", version=DCT_BASIS_VERSION, basis_family="dct")
        self.assertEqual(dct_key.basis_family, BASIS_FAMILY_DCT)
        self.assertEqual(dct_key.version, DCT_BASIS_VERSION)
        cuda_key = cache.make_key(16, 8, runtime_dtype=torch.float32, device="cuda:0", version=BASIS_VERSION, basis_family="chebyshev")
        self.assertNotEqual(key, cuda_key)

    def test_basis_owner_registers_non_persistent_non_trainable_buffers(self) -> None:
        cache = BasisCache()
        owner = BasisOwner(cache)
        depth = owner.add_basis("depth_basis", 4, 3, runtime_dtype=torch.float32, basis_family="chebyshev")
        row = owner.add_basis("row_basis", 16, 8, runtime_dtype=torch.float32, basis_family="cheby")
        self.assertFalse(depth.requires_grad)
        self.assertFalse(row.requires_grad)
        self.assertEqual(list(owner.parameters()), [])
        self.assertEqual(dict(owner.state_dict()), {})
        self.assertEqual(set(dict(owner.named_buffers())), {"depth_basis", "row_basis"})
        with self.assertRaisesRegex(ValueError, "already exists"):
            owner.add_basis("depth_basis", 4, 3)

    def test_sheet_trajectory_materialization_matches_stage0_fixture_through_kernel_seam(self) -> None:
        trajectory = self.trajectory()
        self.assertEqual(trajectory.basis_family, BASIS_FAMILY_CHEBYSHEV)
        self.assertEqual(trajectory.persistent_basis_keys(), ())
        for name, expected in self.fixture["selected_materializations"].items():
            materialized = trajectory.materialize(name, expected["layer_index"])
            self.assertEqual(list(materialized.shape), expected["shape"])
            first_values = torch.tensor(expected["first_values"], dtype=materialized.dtype)
            torch.testing.assert_close(materialized.flatten()[: first_values.numel()], first_values, rtol=5.0e-6, atol=5.0e-7)
            torch.testing.assert_close(materialized.sum(), torch.tensor(expected["sum"], dtype=materialized.dtype), rtol=5.0e-6, atol=5.0e-7)
            torch.testing.assert_close(materialized.norm(), torch.tensor(expected["norm"], dtype=materialized.dtype), rtol=5.0e-6, atol=5.0e-7)
        direct = trajectory.direct_value("attention_input_weight", 2, 17, 9)
        materialized = trajectory.materialize("attention_input_weight", 2)
        torch.testing.assert_close(direct, materialized[17, 9], rtol=1.0e-6, atol=1.0e-7)

    def test_negative_validation_paths(self) -> None:
        invalid_builds = (
            lambda: build_stabilized_basis(0, 1),
            lambda: build_stabilized_basis(4, 0),
            lambda: build_stabilized_basis(4, 5),
            lambda: build_stabilized_basis(4, 2, runtime_dtype=torch.int64),
            lambda: build_stabilized_basis(4, 2, version=""),
            lambda: build_stabilized_basis(4, 2, basis_family="dct"),
            lambda: build_stabilized_basis(4, 2, basis_family="nope"),
        )
        for index, operation in enumerate(invalid_builds):
            with self.subTest(index=index):
                with self.assertRaises((ValueError, TypeError)):
                    operation()


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
