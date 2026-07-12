# vvv THOG
from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import torch

from sheet.basis import (
    BASIS_VERSION,
    BasisCache,
    BasisOwner,
    basis_sha256,
    build_stabilized_basis,
    chebyshev_first_kind_basis,
    deterministic_reduced_qr,
    normalized_coordinates,
    orthonormality_max_error,
)
from sheet.geometry import (
    MATRIX_FAMILY_NAMES,
    SheetGeometryConfig,
    derive_row_order,
    family_geometry_map,
    total_dense_equivalent_count,
    total_sheet_parameter_count,
    transformer_family_geometries,
)


class Stage1MathematicalCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        calibration_path_value = os.environ.get("THOG2_STAGE1_CALIBRATION")
        if not calibration_path_value:
            raise RuntimeError(
                "THOG2_STAGE1_CALIBRATION is required; use tests/run_sheet_stage1_tests.py"
            )
        calibration_path = Path(calibration_path_value)
        cls.calibration = json.loads(calibration_path.read_text(encoding="utf-8"))

    def test_s1_01_coordinate_endpoints(self) -> None:
        for sample_count in (1, 2, 144, 768, 3072):
            with self.subTest(sample_count=sample_count):
                coordinates = normalized_coordinates(sample_count)
                self.assertEqual(tuple(coordinates.shape), (sample_count,))
                self.assertTrue(torch.isfinite(coordinates).all())
                if sample_count == 1:
                    self.assertEqual(float(coordinates[0]), 0.0)
                else:
                    self.assertEqual(float(coordinates[0]), -1.0)
                    self.assertEqual(float(coordinates[-1]), 1.0)

    def test_s1_02_coordinate_monotonicity(self) -> None:
        for sample_count in (2, 3, 144, 768, 3072):
            with self.subTest(sample_count=sample_count):
                first = normalized_coordinates(sample_count)
                second = normalized_coordinates(sample_count)
                self.assertTrue(torch.equal(first, second))
                self.assertTrue(torch.all(first[1:] > first[:-1]))

    def test_s1_03_chebyshev_recurrence_reference(self) -> None:
        coordinates = torch.tensor(
            [-1.0, -0.75, -0.2, 0.0, 0.3, 0.8, 1.0], dtype=torch.float64
        )
        order = 12
        recurrence_basis = chebyshev_first_kind_basis(coordinates, order)
        term_indices = torch.arange(order, dtype=torch.float64)
        analytic_basis = torch.cos(torch.acos(coordinates).unsqueeze(1) * term_indices.unsqueeze(0))
        torch.testing.assert_close(recurrence_basis, analytic_basis, rtol=0.0, atol=2.0e-14)

    def test_s1_04_raw_basis_shape_and_finiteness(self) -> None:
        for sample_count, order in ((16, 4), (144, 16), (768, 128), (3072, 512)):
            with self.subTest(sample_count=sample_count, order=order):
                coordinates = normalized_coordinates(sample_count)
                raw_basis = chebyshev_first_kind_basis(coordinates, order)
                self.assertEqual(tuple(raw_basis.shape), (sample_count, order))
                self.assertTrue(torch.isfinite(raw_basis).all())
                self.assertLessEqual(float(torch.max(torch.abs(raw_basis))), 1.0 + 1.0e-12)

    def test_s1_05_qr_orthonormality(self) -> None:
        basis_float64 = build_stabilized_basis(256, 64, runtime_dtype=torch.float64)
        basis_float32 = build_stabilized_basis(256, 64, runtime_dtype=torch.float32)
        self.assertLessEqual(orthonormality_max_error(basis_float64), 2.0e-12)
        self.assertLessEqual(orthonormality_max_error(basis_float32), 2.0e-5)

        thresholds = self.calibration["thresholds"]
        for result in self.calibration["results"]:
            self.assertLessEqual(
                result["orthonormality_max_error_float64"],
                thresholds["float64_orthonormality_tolerance"],
            )
            self.assertLessEqual(
                result["orthonormality_max_error_float32"],
                thresholds["float32_orthonormality_tolerance"],
            )

    def test_s1_06_full_column_rank(self) -> None:
        coordinates = normalized_coordinates(128)
        raw_basis = chebyshev_first_kind_basis(coordinates, 32)
        stabilized_basis, r_matrix = deterministic_reduced_qr(raw_basis)
        self.assertEqual(int(torch.linalg.matrix_rank(raw_basis)), 32)
        self.assertEqual(int(torch.linalg.matrix_rank(stabilized_basis)), 32)
        self.assertTrue(torch.all(torch.diagonal(r_matrix) > 0.0))

        for result in self.calibration["results"]:
            self.assertTrue(result["full_rank_qr_proxy"])
            self.assertGreater(result["minimum_abs_r_diagonal"], 0.0)

    def test_s1_07_discrete_span_preservation(self) -> None:
        coordinates = normalized_coordinates(48)
        raw_basis = chebyshev_first_kind_basis(coordinates, 12)
        stabilized_basis, _ = deterministic_reduced_qr(raw_basis)
        raw_projector = raw_basis @ torch.linalg.pinv(raw_basis)
        stabilized_projector = stabilized_basis @ stabilized_basis.transpose(0, 1)
        torch.testing.assert_close(raw_projector, stabilized_projector, rtol=0.0, atol=2.0e-11)

    def test_s1_08_deterministic_column_signs(self) -> None:
        first = build_stabilized_basis(96, 24)
        second = build_stabilized_basis(96, 24)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(basis_sha256(first), basis_sha256(second))

        command = (
            "import torch; "
            "torch.set_num_threads(1); "
            "from sheet.basis import basis_sha256, build_stabilized_basis; "
            "print(basis_sha256(build_stabilized_basis(96, 24)))"
        )
        environment = os.environ.copy()
        environment["OMP_NUM_THREADS"] = "1"
        environment["MKL_NUM_THREADS"] = "1"
        repository_root = Path(__file__).resolve().parents[1]
        hashes = []
        for _ in range(2):
            completed = subprocess.run(
                [sys.executable, "-c", command],
                cwd=repository_root,
                env=environment,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            hashes.append(completed.stdout.strip())
        self.assertEqual(hashes[0], hashes[1])

    def test_s1_09_basis_cache_identity(self) -> None:
        cache = BasisCache()
        first = cache.get(64, 16, runtime_dtype=torch.float64)
        second = cache.get(64, 16, runtime_dtype=torch.float64)
        self.assertIs(first, second)
        self.assertEqual(len(cache), 1)

        with self.assertRaisesRegex(ValueError, "basis_version mismatch|unsupported"):
            cache.get(
                64,
                16,
                runtime_dtype=torch.float64,
                version=BASIS_VERSION + "_test_variant",
            )
        dtype_variant = cache.get(64, 16, runtime_dtype=torch.float32)
        self.assertIsNot(first, dtype_variant)
        self.assertEqual(len(cache), 2)

        cpu_key = cache.make_key(
            64,
            16,
            runtime_dtype=torch.float32,
            device="cpu",
            version=BASIS_VERSION,
        )
        cuda_key = cache.make_key(
            64,
            16,
            runtime_dtype=torch.float32,
            device="cuda:0",
            version=BASIS_VERSION,
        )
        self.assertNotEqual(cpu_key, cuda_key)

    def test_s1_10_row_order_derivation(self) -> None:
        self.assertEqual(derive_row_order(768, 768, 128), 128)
        self.assertEqual(derive_row_order(3072, 768, 128), 512)
        self.assertEqual(derive_row_order(2304, 768, 128), 384)
        self.assertEqual(derive_row_order(1, 768, 128), 1)
        self.assertEqual(derive_row_order(3, 8, 3), 2)
        self.assertEqual(derive_row_order(8, 8, 8), 8)

    def test_s1_11_configuration_rejection(self) -> None:
        invalid_cases = (
            (lambda: SheetGeometryConfig(0, 768, 12, 16, 128), "n_layer"),
            (lambda: SheetGeometryConfig(12, 0, 12, 8, 16), "n_embd"),
            (lambda: SheetGeometryConfig(12, 768, 0, 8, 16), "n_head"),
            (lambda: SheetGeometryConfig(12, 768, 12, 13, 16), "depth_order"),
            (lambda: SheetGeometryConfig(12, 768, 12, 8, 769), "base_row_order"),
            (lambda: SheetGeometryConfig(12, 770, 12, 8, 16), "divisible"),
            (lambda: derive_row_order(0, 768, 128), "row_width"),
            (lambda: derive_row_order(768, 0, 128), "model_width"),
            (lambda: derive_row_order(768, 768, 769), "base_row_order"),
            (lambda: build_stabilized_basis(8, 9), "must not exceed"),
        )
        for operation, diagnostic in invalid_cases:
            with self.subTest(diagnostic=diagnostic):
                with self.assertRaisesRegex(ValueError, diagnostic):
                    operation()

    def test_s1_12_high_order_construction(self) -> None:
        expected_orders = {128, 256, 512, 1024}
        observed_orders = {int(result["order"]) for result in self.calibration["results"]}
        self.assertEqual(observed_orders, expected_orders)
        self.assertTrue(self.calibration["accepted"])
        thresholds = self.calibration["thresholds"]
        for result in self.calibration["results"]:
            self.assertTrue(result["accepted"])
            self.assertTrue(result["raw_finite"])
            self.assertTrue(result["basis_float64_finite"])
            self.assertTrue(result["basis_float32_finite"])
            self.assertLessEqual(
                result["elapsed_seconds"], thresholds["max_seconds_per_geometry"]
            )
            self.assertLessEqual(
                result["estimated_peak_tensor_bytes_float64"],
                thresholds["max_estimated_peak_bytes"],
            )

    def test_s1_13_family_geometry(self) -> None:
        config = SheetGeometryConfig(
            n_layer=144,
            n_embd=768,
            n_head=12,
            depth_order=16,
            base_row_order=128,
            bias=True,
        )
        geometry = family_geometry_map(config)
        self.assertEqual(set(MATRIX_FAMILY_NAMES), {
            name for name, family in geometry.items() if family.family_kind == "matrix"
        })
        expected = {
            "attention_input_weight": (2304, 768, 128),
            "attention_output_weight": (768, 768, 128),
            "mlp_expansion_weight": (3072, 768, 128),
            "mlp_contraction_weight": (768, 3072, 512),
            "ln_1_weight": (1, 768, 128),
            "ln_2_weight": (1, 768, 128),
            "attention_input_bias": (1, 2304, 384),
            "mlp_expansion_bias": (1, 3072, 512),
        }
        for name, expected_geometry in expected.items():
            family = geometry[name]
            self.assertEqual(
                (family.output_rows, family.row_width, family.row_order),
                expected_geometry,
            )

        no_bias = SheetGeometryConfig(12, 96, 6, 8, 16, bias=False)
        no_bias_names = {family.name for family in transformer_family_geometries(no_bias)}
        self.assertIn("ln_1_weight", no_bias_names)
        self.assertNotIn("ln_1_bias", no_bias_names)
        self.assertNotIn("attention_input_bias", no_bias_names)

    def test_s1_14_analytical_parameter_counts(self) -> None:
        principal = SheetGeometryConfig(144, 768, 12, 16, 128, bias=True)
        principal_matrices = transformer_family_geometries(principal, include_vectors=False)
        explicit_sheet_count = sum(
            family.output_rows * principal.depth_order * family.row_order
            for family in principal_matrices
        )
        explicit_dense_count = sum(
            principal.n_layer * family.output_rows * family.row_width
            for family in principal_matrices
        )
        self.assertEqual(explicit_sheet_count, 18_874_368)
        self.assertEqual(explicit_dense_count, 1_019_215_872)
        self.assertEqual(
            total_sheet_parameter_count(principal_matrices, principal.depth_order),
            explicit_sheet_count,
        )
        self.assertEqual(
            total_dense_equivalent_count(principal_matrices, principal.n_layer),
            explicit_dense_count,
        )

        tiny = SheetGeometryConfig(3, 8, 2, 2, 3, bias=False)
        tiny_families = transformer_family_geometries(tiny, include_vectors=False)
        for family in tiny_families:
            shape = family.coefficient_shape(tiny.depth_order)
            self.assertEqual(family.sheet_parameter_count(tiny.depth_order), shape[0] * shape[1] * shape[2])
            self.assertEqual(
                family.dense_equivalent_count(tiny.n_layer),
                tiny.n_layer * family.output_rows * family.row_width,
            )

    def test_s1_15_non_persistent_basis_schema(self) -> None:
        cache = BasisCache()
        owner = BasisOwner(cache)
        depth_basis = owner.add_basis("depth_basis", 16, 4, runtime_dtype=torch.float64)
        row_basis = owner.add_basis("row_basis", 32, 8, runtime_dtype=torch.float32)
        self.assertFalse(depth_basis.requires_grad)
        self.assertFalse(row_basis.requires_grad)
        self.assertEqual(list(owner.parameters()), [])
        self.assertEqual(dict(owner.state_dict()), {})
        self.assertEqual(set(dict(owner.named_buffers()).keys()), {"depth_basis", "row_basis"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
