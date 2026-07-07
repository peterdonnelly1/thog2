# vvv THOG
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from sheet.checkpoints import load_payload, validate_compatibility
from sheet.compact_identity import (
    ATTENTION_GEOMETRY_CURVE,
    BASIS_FAMILY_CHEBYSHEV,
    BASIS_FAMILY_DCT,
    CURVE_MATERIALIZATION_VERSION,
    GEOMETRY_PRESET_BLOCK,
    GEOMETRY_PRESET_CURVE,
    GEOMETRY_PRESET_MLP_BLOCK,
    MLP_GEOMETRY_CURVE,
)
from sheet.curve_trajectory import CURVE_MATRIX_FAMILIES, CurveTrajectory
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
from sheet.stage4_trainer import Stage4Trainer
from tests.stage4_test_support import stage4_tokens, stage4_training_config


class Stage4CurveMaterializationTests(unittest.TestCase):
    def curve_config(self) -> SheetGPTConfig:
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
            geometry_preset=GEOMETRY_PRESET_CURVE,
        )

    def test_01_curve_config_identity_is_accepted_and_future_block_modes_still_fail(self) -> None:
        curve = stage4_training_config(geometry_preset=GEOMETRY_PRESET_CURVE)
        identity = curve.compact_identity_metadata()
        self.assertEqual(identity["geometry_preset"], GEOMETRY_PRESET_CURVE)
        self.assertEqual(identity["attention_geometry"], ATTENTION_GEOMETRY_CURVE)
        self.assertEqual(identity["mlp_geometry"], MLP_GEOMETRY_CURVE)
        self.assertEqual(identity["basis_family"], BASIS_FAMILY_CHEBYSHEV)
        self.assertEqual(identity["materialization_version"], CURVE_MATERIALIZATION_VERSION)

        explicit = stage4_training_config(attention_geometry=ATTENTION_GEOMETRY_CURVE, mlp_geometry=MLP_GEOMETRY_CURVE)
        self.assertEqual(explicit.compact_identity_metadata()["geometry_preset"], GEOMETRY_PRESET_CURVE)

        for overrides in (
            {"geometry_preset": GEOMETRY_PRESET_CURVE, "basis_family": BASIS_FAMILY_DCT},
            {"geometry_preset": GEOMETRY_PRESET_MLP_BLOCK},
            {"geometry_preset": GEOMETRY_PRESET_BLOCK},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, "Stage 4 supports only"):
                    stage4_training_config(**overrides)

    def test_02_curve_trajectory_has_depth_only_matrix_coefficients_and_no_packed_qkv_parameter(self) -> None:
        config = self.curve_config()
        trajectory = CurveTrajectory(config.sheet_geometry(), runtime_dtype=torch.float32, basis_version=config.basis_version)
        expected_shapes = {
            ATTENTION_QUERY_WEIGHT: (16, 16, 3),
            ATTENTION_KEY_WEIGHT: (16, 16, 3),
            ATTENTION_VALUE_WEIGHT: (16, 16, 3),
            ATTENTION_OUTPUT_WEIGHT: (16, 16, 3),
            MLP_EXPANSION_WEIGHT: (64, 16, 3),
            MLP_CONTRACTION_WEIGHT: (16, 64, 3),
        }
        self.assertEqual(CURVE_MATRIX_FAMILIES, tuple(expected_shapes))
        for name, shape in expected_shapes.items():
            with self.subTest(name=name):
                self.assertEqual(tuple(trajectory.coefficients[name].shape), shape)
        self.assertNotIn(LEGACY_ATTENTION_INPUT_WEIGHT, trajectory.coefficients)

    def test_03_curve_depth_contraction_direct_value_and_packed_qkv_boundary_are_exact(self) -> None:
        config = self.curve_config()
        trajectory = CurveTrajectory(config.sheet_geometry(), runtime_dtype=torch.float32, basis_version=config.basis_version)
        with torch.no_grad():
            for family in CURVE_MATRIX_FAMILIES:
                trajectory.coefficients[family].zero_()
            trajectory.coefficients[ATTENTION_QUERY_WEIGHT][2, 3, :] = torch.tensor([1.0, 2.0, 3.0])
            trajectory.coefficients[ATTENTION_KEY_WEIGHT][4, 5, :] = torch.tensor([2.0, 0.0, -1.0])
            trajectory.coefficients[ATTENTION_VALUE_WEIGHT][6, 7, :] = torch.tensor([-3.0, 1.0, 0.5])
        layer_index = 1
        depth = trajectory.depth_basis[layer_index]
        query = trajectory.materialize(ATTENTION_QUERY_WEIGHT, layer_index)
        manual_query = torch.dot(depth, trajectory.coefficients[ATTENTION_QUERY_WEIGHT][2, 3])
        torch.testing.assert_close(query[2, 3], manual_query, rtol=0.0, atol=0.0)
        torch.testing.assert_close(trajectory.direct_value(ATTENTION_QUERY_WEIGHT, layer_index, 2, 3), query[2, 3], rtol=0.0, atol=0.0)
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

    def test_04_curve_parameter_report_counts_only_curve_matrices_plus_legacy_vectors(self) -> None:
        torch.manual_seed(4101)
        model = SheetGPT(self.curve_config())
        report = model.parameter_report()
        width = model.config.n_embd
        depth_order = model.config.depth_order
        curve_matrix_coefficients = depth_order * (
            width * width
            + width * width
            + width * width
            + width * width
            + (4 * width) * width
            + width * (4 * width)
        )
        legacy_vector_coefficients = 312
        conventional_parameters = 672
        self.assertEqual(report["matrix_sheet_coefficients"], curve_matrix_coefficients)
        self.assertGreater(curve_matrix_coefficients, 4608)
        self.assertEqual(report["sheet_coefficients"], curve_matrix_coefficients + legacy_vector_coefficients)
        self.assertEqual(report["persistent_parameters"], curve_matrix_coefficients + legacy_vector_coefficients + conventional_parameters)
        parameter_names = tuple(name for group in model.optimizer_parameter_groups(0.1) for name in group["parameter_names"])
        self.assertIn("trajectory.coefficients.attention_query_weight", parameter_names)
        self.assertIn("trajectory.coefficients.mlp_contraction_weight", parameter_names)
        self.assertNotIn("trajectory.coefficients.attention_input_weight", parameter_names)
        self.assertEqual(model.compact_state_violations(), ())

    def test_05_curve_model_forward_backward_sends_gradients_to_qkv_curve_coefficients(self) -> None:
        torch.manual_seed(4101)
        model = SheetGPT(self.curve_config())
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

    def test_06_curve_checkpoint_identity_round_trips_and_rejects_legacy_cross_loads(self) -> None:
        train_tokens, validation_tokens = stage4_tokens(128)
        curve_config = stage4_training_config(geometry_preset=GEOMETRY_PRESET_CURVE, max_updates=1)
        curve_trainer = Stage4Trainer(curve_config, train_tokens, validation_tokens)
        self.assertIsInstance(curve_trainer.raw_model.trajectory, CurveTrajectory)
        legacy_config = stage4_training_config(max_updates=1)
        legacy_trainer = Stage4Trainer(legacy_config, train_tokens, validation_tokens)
        with tempfile.TemporaryDirectory() as directory:
            curve_path = Path(directory) / "curve.pt"
            legacy_path = Path(directory) / "legacy.pt"
            curve_trainer.save_checkpoint(curve_path)
            legacy_trainer.save_checkpoint(legacy_path)
            curve_payload = load_payload(curve_path)
            legacy_payload = load_payload(legacy_path)
            self.assertEqual(curve_payload["compact_identity"]["materialization_version"], CURVE_MATERIALIZATION_VERSION)
            resumed = Stage4Trainer.from_checkpoint(curve_path, train_tokens, validation_tokens)
            self.assertIsInstance(resumed.raw_model.trajectory, CurveTrajectory)
            self.assertEqual(resumed.config.compact_identity_metadata()["materialization_version"], CURVE_MATERIALIZATION_VERSION)
            with self.assertRaisesRegex(ValueError, "compact_identity"):
                validate_compatibility(curve_payload, legacy_config)
            with self.assertRaisesRegex(ValueError, "compact_identity"):
                validate_compatibility(legacy_payload, curve_config)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
