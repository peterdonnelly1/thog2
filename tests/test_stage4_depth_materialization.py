# vvv THOG
from __future__ import annotations

import unittest

import torch

from sheet.basis_kernel import DCT_BASIS_VERSION
from sheet.compact_identity import (
    ATTENTION_GEOMETRY_DEPTH,
    BASIS_FAMILY_CHEBYSHEV,
    BASIS_FAMILY_DCT,
    DEPTH_MATERIALIZATION_VERSION,
    GEOMETRY_PRESET_DEPTH,
    GEOMETRY_PRESET_FULL_BLOCK,
    MLP_GEOMETRY_DEPTH,
)
from sheet.depth_trajectory import DEPTH_MATRIX_FAMILIES, DepthTrajectory
from sheet.model import SheetGPT, SheetGPTConfig
from sheet.semantic_materializer import (
    ATTENTION_KEY_WEIGHT,
    ATTENTION_OUTPUT_WEIGHT,
    ATTENTION_QUERY_WEIGHT,
    ATTENTION_VALUE_WEIGHT,
    LEGACY_ATTENTION_INPUT_BIAS,
    LEGACY_ATTENTION_INPUT_WEIGHT,
    MLP_CONTRACTION_WEIGHT,
    MLP_EXPANSION_WEIGHT,
)
from tests.stage4_test_support import stage4_training_config


class PictonStage4DepthMaterializationTests(unittest.TestCase):
    def depth_config(self) -> SheetGPTConfig:
        return SheetGPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=4,
            n_head=2,
            n_embd=16,
            dropout=0.0,
            bias=True,
            depth_order=3,
            base_row_order=8,
            geometry_preset=GEOMETRY_PRESET_DEPTH,
        )

    def test_picton_01_depth_config_identity_is_accepted_for_chebyshev_and_dct(self) -> None:
        depth = stage4_training_config(geometry_preset=GEOMETRY_PRESET_DEPTH)
        identity = depth.compact_identity_metadata()
        self.assertEqual(identity["geometry_preset"], GEOMETRY_PRESET_DEPTH)
        self.assertEqual(identity["attention_geometry"], ATTENTION_GEOMETRY_DEPTH)
        self.assertEqual(identity["mlp_geometry"], MLP_GEOMETRY_DEPTH)
        self.assertEqual(identity["basis_family"], BASIS_FAMILY_CHEBYSHEV)
        self.assertEqual(identity["materialization_version"], DEPTH_MATERIALIZATION_VERSION)
        explicit = stage4_training_config(
            attention_geometry=ATTENTION_GEOMETRY_DEPTH,
            mlp_geometry=MLP_GEOMETRY_DEPTH,
        )
        self.assertEqual(explicit.compact_identity_metadata()["geometry_preset"], GEOMETRY_PRESET_DEPTH)
        self.assertEqual(
            stage4_training_config(geometry_preset=GEOMETRY_PRESET_FULL_BLOCK).compact_identity_metadata()["geometry_preset"],
            GEOMETRY_PRESET_FULL_BLOCK,
        )
        dct_depth = stage4_training_config(geometry_preset=GEOMETRY_PRESET_DEPTH, basis_family=BASIS_FAMILY_DCT)
        self.assertEqual(dct_depth.compact_identity_metadata()["basis_version"], DCT_BASIS_VERSION)

    def test_picton_02_depth_trajectory_has_depth_only_matrix_coefficients_and_no_packed_qkv_parameter(self) -> None:
        config = self.depth_config()
        trajectory = DepthTrajectory(config.sheet_geometry(), runtime_dtype=torch.float32, basis_version=config.basis_version)
        expected_shapes = {
            ATTENTION_QUERY_WEIGHT: (16, 16, 3),
            ATTENTION_KEY_WEIGHT: (16, 16, 3),
            ATTENTION_VALUE_WEIGHT: (16, 16, 3),
            ATTENTION_OUTPUT_WEIGHT: (16, 16, 3),
            MLP_EXPANSION_WEIGHT: (64, 16, 3),
            MLP_CONTRACTION_WEIGHT: (16, 64, 3),
        }
        self.assertEqual(DEPTH_MATRIX_FAMILIES, tuple(expected_shapes))
        for name, shape in expected_shapes.items():
            with self.subTest(name=name):
                self.assertEqual(tuple(trajectory.coefficients[name].shape), shape)
        self.assertNotIn(LEGACY_ATTENTION_INPUT_WEIGHT, trajectory.coefficients)

    def test_picton_03_depth_contraction_direct_value_and_packed_qkv_boundary_are_exact(self) -> None:
        config = self.depth_config()
        trajectory = DepthTrajectory(config.sheet_geometry(), runtime_dtype=torch.float32, basis_version=config.basis_version)
        with torch.no_grad():
            for family in DEPTH_MATRIX_FAMILIES:
                trajectory.coefficients[family].zero_()
            trajectory.coefficients[ATTENTION_QUERY_WEIGHT][2, 3, :] = torch.tensor([1.0, 2.0, 3.0])
            trajectory.coefficients[ATTENTION_KEY_WEIGHT][4, 5, :] = torch.tensor([2.0, 0.0, -1.0])
            trajectory.coefficients[ATTENTION_VALUE_WEIGHT][6, 7, :] = torch.tensor([-3.0, 1.0, 0.5])
        layer_index = 1
        depth = trajectory.depth_basis[layer_index]
        query = trajectory.materialize(ATTENTION_QUERY_WEIGHT, layer_index)
        manual_query = torch.dot(depth, trajectory.coefficients[ATTENTION_QUERY_WEIGHT][2, 3])
        torch.testing.assert_close(query[2, 3], manual_query, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            trajectory.direct_value(ATTENTION_QUERY_WEIGHT, layer_index, 2, 3),
            query[2, 3],
            rtol=0.0,
            atol=0.0,
        )
        packed = trajectory.materialize(LEGACY_ATTENTION_INPUT_WEIGHT, layer_index)
        reconstructed = torch.cat(
            (
                trajectory.materialize(ATTENTION_QUERY_WEIGHT, layer_index),
                trajectory.materialize(ATTENTION_KEY_WEIGHT, layer_index),
                trajectory.materialize(ATTENTION_VALUE_WEIGHT, layer_index),
            ),
            dim=0,
        )
        torch.testing.assert_close(packed, reconstructed, rtol=0.0, atol=0.0)
        self.assertEqual(tuple(trajectory.materialize_vector(LEGACY_ATTENTION_INPUT_BIAS, layer_index).shape), (3 * config.n_embd,))

    def test_picton_04_depth_parameter_report_counts_only_depth_matrices_plus_legacy_vectors(self) -> None:
        torch.manual_seed(4101)
        model = SheetGPT(self.depth_config())
        report = model.parameter_report()
        width = model.config.n_embd
        depth_order = model.config.depth_order
        depth_matrix_coefficients = depth_order * (
            width * width
            + width * width
            + width * width
            + width * width
            + (4 * width) * width
            + width * (4 * width)
        )
        legacy_vector_coefficients = 312
        conventional_parameters = 672
        self.assertEqual(depth_matrix_coefficients, 4608)
        self.assertEqual(report["matrix_sheet_coefficients"], depth_matrix_coefficients)
        self.assertEqual(report["sheet_coefficients"], depth_matrix_coefficients + legacy_vector_coefficients)
        self.assertEqual(report["persistent_parameters"], depth_matrix_coefficients + legacy_vector_coefficients + conventional_parameters)
        parameter_names = tuple(name for group in model.optimizer_parameter_groups(0.1) for name in group["parameter_names"])
        self.assertIn("trajectory.coefficients.attention_query_weight", parameter_names)
        self.assertIn("trajectory.coefficients.mlp_contraction_weight", parameter_names)
        self.assertNotIn("trajectory.coefficients.attention_input_weight", parameter_names)
        self.assertEqual(model.compact_state_violations(), ())

    def test_picton_05_depth_model_forward_backward_sends_gradients_to_qkv_coefficients(self) -> None:
        torch.manual_seed(4101)
        model = SheetGPT(self.depth_config())
        idx = torch.randint(0, model.config.vocab_size, (2, 4))
        targets = torch.randint(0, model.config.vocab_size, (2, 4))
        logits, loss = model(idx, targets)
        self.assertEqual(tuple(logits.shape), (2, 4, model.config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name in (ATTENTION_QUERY_WEIGHT, ATTENTION_KEY_WEIGHT, ATTENTION_VALUE_WEIGHT):
            with self.subTest(name=name):
                gradient = model.trajectory.coefficients[name].grad
                self.assertIsNotNone(gradient)
                self.assertGreater(float(gradient.abs().sum().item()), 0.0)
        self.assertIsNotNone(model.trajectory.coefficients[LEGACY_ATTENTION_INPUT_BIAS].grad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
