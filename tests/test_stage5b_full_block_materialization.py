# vvv THOG
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from sheet.block_trajectory import BLOCK_MATERIALIZATION_VERSION, BLOCK_MATRIX_FAMILIES, BlockTrajectory
from sheet.checkpoints import load_payload, validate_compatibility
from sheet.compact_identity import (
    ATTENTION_GEOMETRY_CURVE,
    ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK,
    BASIS_FAMILY_CHEBYSHEV,
    BASIS_FAMILY_DCT,
    GEOMETRY_PRESET_BLOCK,
    GEOMETRY_PRESET_CURVE,
    GEOMETRY_PRESET_MLP_BLOCK,
    MLP_GEOMETRY_MLP_BLOCK,
)
from sheet.curve_trajectory import CurveTrajectory
from sheet.mlp_block_trajectory import MlpBlockTrajectory
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
from sheet.trajectory import SheetTrajectory
from tests.stage4_test_support import stage4_tokens, stage4_training_config


class Stage5bFullBlockMaterializationTests(unittest.TestCase):
    def full_block_config(self) -> SheetGPTConfig:
        return SheetGPTConfig(block_size=8, vocab_size=32, n_layer=4, n_head=2, n_embd=16, dropout=0.0, bias=True, depth_order=3, base_row_order=8, geometry_preset=GEOMETRY_PRESET_BLOCK)

    def test_01_full_block_config_identity_is_accepted_and_dct_or_mixed_partial_block_modes_still_fail(self) -> None:
        preset = stage4_training_config(geometry_preset=GEOMETRY_PRESET_BLOCK)
        identity = preset.compact_identity_metadata()
        self.assertEqual(identity["geometry_preset"], GEOMETRY_PRESET_BLOCK)
        self.assertEqual(identity["attention_geometry"], ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK)
        self.assertEqual(identity["mlp_geometry"], MLP_GEOMETRY_MLP_BLOCK)
        self.assertEqual(identity["basis_family"], BASIS_FAMILY_CHEBYSHEV)
        self.assertEqual(identity["materialization_version"], BLOCK_MATERIALIZATION_VERSION)
        explicit = stage4_training_config(attention_geometry=ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK, mlp_geometry=MLP_GEOMETRY_MLP_BLOCK)
        self.assertEqual(explicit.compact_identity_metadata()["geometry_preset"], GEOMETRY_PRESET_BLOCK)
        for overrides in ({"geometry_preset": GEOMETRY_PRESET_BLOCK, "basis_family": BASIS_FAMILY_DCT}, {"geometry_preset": GEOMETRY_PRESET_BLOCK, "attention_geometry": ATTENTION_GEOMETRY_CURVE}):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, "Stage 5 supports only"):
                    stage4_training_config(**overrides)

    def test_02_full_block_trajectory_replaces_every_repeated_matrix_family_with_head_aware_attention_or_mlp_block_coefficients(self) -> None:
        trajectory = BlockTrajectory(self.full_block_config().sheet_geometry(), runtime_dtype=torch.float32)
        expected_shapes = {
            ATTENTION_QUERY_WEIGHT: (2, 3, 4, 8),
            ATTENTION_KEY_WEIGHT: (2, 3, 4, 8),
            ATTENTION_VALUE_WEIGHT: (2, 3, 4, 8),
            ATTENTION_OUTPUT_WEIGHT: (2, 3, 8, 4),
            MLP_EXPANSION_WEIGHT: (3, 32, 8),
            MLP_CONTRACTION_WEIGHT: (3, 8, 32),
        }
        self.assertEqual(BLOCK_MATRIX_FAMILIES, tuple(expected_shapes))
        for name, shape in expected_shapes.items():
            with self.subTest(name=name):
                self.assertEqual(tuple(trajectory.coefficients[name].shape), shape)
        self.assertNotIn(LEGACY_ATTENTION_INPUT_WEIGHT, trajectory.coefficients)

    def test_03_full_block_materialization_matches_head_aware_attention_and_mlp_block_contracts(self) -> None:
        trajectory = BlockTrajectory(self.full_block_config().sheet_geometry(), runtime_dtype=torch.float32)
        with torch.no_grad():
            for name in BLOCK_MATRIX_FAMILIES:
                trajectory.coefficients[name].zero_()
            trajectory.coefficients[ATTENTION_QUERY_WEIGHT][0, :, 2, 3] = torch.tensor([1.0, 2.0, 3.0])
            trajectory.coefficients[ATTENTION_OUTPUT_WEIGHT][1, :, 4, 2] = torch.tensor([-1.0, 0.5, 2.0])
            trajectory.coefficients[MLP_EXPANSION_WEIGHT][:, 6, 7] = torch.tensor([0.25, -0.5, 1.5])
        layer_index = 1
        depth = trajectory.depth_basis[layer_index]
        query_mixed = torch.einsum("p,pab->ab", depth, trajectory.coefficients[ATTENTION_QUERY_WEIGHT][0])
        manual_query_head = trajectory.output_basis(ATTENTION_QUERY_WEIGHT) @ query_mixed @ trajectory.input_basis(ATTENTION_QUERY_WEIGHT).transpose(0, 1)
        query = trajectory.materialize(ATTENTION_QUERY_WEIGHT, layer_index)
        torch.testing.assert_close(query[:8], manual_query_head, rtol=0.0, atol=0.0)
        torch.testing.assert_close(query[8:], torch.zeros_like(query[8:]), rtol=0.0, atol=0.0)
        output_mixed = torch.einsum("p,pab->ab", depth, trajectory.coefficients[ATTENTION_OUTPUT_WEIGHT][1])
        manual_output_head = trajectory.output_basis(ATTENTION_OUTPUT_WEIGHT) @ output_mixed @ trajectory.input_basis(ATTENTION_OUTPUT_WEIGHT).transpose(0, 1)
        output = trajectory.materialize(ATTENTION_OUTPUT_WEIGHT, layer_index)
        torch.testing.assert_close(output[:, :8], torch.zeros_like(output[:, :8]), rtol=0.0, atol=0.0)
        torch.testing.assert_close(output[:, 8:], manual_output_head, rtol=0.0, atol=0.0)
        expansion_mixed = torch.einsum("p,pab->ab", depth, trajectory.coefficients[MLP_EXPANSION_WEIGHT])
        manual_expansion = trajectory.output_basis(MLP_EXPANSION_WEIGHT) @ expansion_mixed @ trajectory.input_basis(MLP_EXPANSION_WEIGHT).transpose(0, 1)
        torch.testing.assert_close(trajectory.materialize(MLP_EXPANSION_WEIGHT, layer_index), manual_expansion, rtol=0.0, atol=0.0)
        packed = trajectory.materialize(LEGACY_ATTENTION_INPUT_WEIGHT, layer_index)
        reconstructed = torch.cat((trajectory.materialize(ATTENTION_QUERY_WEIGHT, layer_index), trajectory.materialize(ATTENTION_KEY_WEIGHT, layer_index), trajectory.materialize(ATTENTION_VALUE_WEIGHT, layer_index)), dim=0)
        torch.testing.assert_close(packed, reconstructed, rtol=0.0, atol=0.0)

    def test_04_full_block_parameter_report_counts_all_six_matrix_families_as_block_coefficients_and_is_smaller_than_mlp_block(self) -> None:
        torch.manual_seed(4101)
        full_block_model = SheetGPT(self.full_block_config())
        mlp_block_config = self.full_block_config().to_dict()
        mlp_block_config["geometry_preset"] = GEOMETRY_PRESET_MLP_BLOCK
        mlp_block_model = SheetGPT(SheetGPTConfig(**mlp_block_config))
        report = full_block_model.parameter_report()
        mlp_block_report = mlp_block_model.parameter_report()
        depth_order = full_block_model.config.depth_order
        attention_block_coefficients = 4 * 2 * depth_order * 4 * 8
        mlp_block_coefficients = depth_order * 32 * 8 + depth_order * 8 * 32
        self.assertEqual(report["matrix_sheet_coefficients"], attention_block_coefficients + mlp_block_coefficients)
        self.assertLess(report["matrix_sheet_coefficients"], mlp_block_report["matrix_sheet_coefficients"])
        self.assertEqual(full_block_model.compact_state_violations(), ())
        report_by_name = {row["name"]: row for row in report["families"]}
        self.assertEqual(report_by_name[ATTENTION_QUERY_WEIGHT]["coefficient_shape"], (2, 3, 4, 8))
        self.assertEqual(report_by_name[MLP_EXPANSION_WEIGHT]["coefficient_shape"], (3, 32, 8))

    def test_05_full_block_model_forward_backward_sends_gradients_to_attention_and_mlp_block_coefficients(self) -> None:
        torch.manual_seed(4101)
        model = SheetGPT(self.full_block_config())
        self.assertIsInstance(model.trajectory, BlockTrajectory)
        idx = torch.randint(0, model.config.vocab_size, (2, 4))
        targets = torch.randint(0, model.config.vocab_size, (2, 4))
        logits, loss = model(idx, targets)
        self.assertEqual(tuple(logits.shape), (2, 4, model.config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name in BLOCK_MATRIX_FAMILIES:
            with self.subTest(name=name):
                gradient = model.trajectory.coefficients[name].grad
                self.assertIsNotNone(gradient)
                self.assertGreater(float(gradient.abs().sum().item()), 0.0)
        self.assertIsNotNone(model.trajectory.coefficients[LEGACY_ATTENTION_INPUT_BIAS].grad)

    def test_06_full_block_checkpoint_identity_round_trips_and_rejects_mlp_block_curve_or_legacy_cross_loads(self) -> None:
        train_tokens, validation_tokens = stage4_tokens(128)
        block_config = stage4_training_config(geometry_preset=GEOMETRY_PRESET_BLOCK, max_updates=1)
        mlp_block_config = stage4_training_config(geometry_preset=GEOMETRY_PRESET_MLP_BLOCK, max_updates=1)
        curve_config = stage4_training_config(geometry_preset=GEOMETRY_PRESET_CURVE, max_updates=1)
        legacy_config = stage4_training_config(max_updates=1)
        block_trainer = Stage4Trainer(block_config, train_tokens, validation_tokens)
        mlp_block_trainer = Stage4Trainer(mlp_block_config, train_tokens, validation_tokens)
        curve_trainer = Stage4Trainer(curve_config, train_tokens, validation_tokens)
        legacy_trainer = Stage4Trainer(legacy_config, train_tokens, validation_tokens)
        self.assertIsInstance(block_trainer.raw_model.trajectory, BlockTrajectory)
        self.assertIsInstance(mlp_block_trainer.raw_model.trajectory, MlpBlockTrajectory)
        self.assertIsInstance(curve_trainer.raw_model.trajectory, CurveTrajectory)
        self.assertIsInstance(legacy_trainer.raw_model.trajectory, SheetTrajectory)
        with tempfile.TemporaryDirectory() as directory:
            block_path = Path(directory) / "block.pt"
            mlp_block_path = Path(directory) / "mlp_block.pt"
            curve_path = Path(directory) / "curve.pt"
            legacy_path = Path(directory) / "legacy.pt"
            block_trainer.save_checkpoint(block_path)
            mlp_block_trainer.save_checkpoint(mlp_block_path)
            curve_trainer.save_checkpoint(curve_path)
            legacy_trainer.save_checkpoint(legacy_path)
            block_payload = load_payload(block_path)
            mlp_block_payload = load_payload(mlp_block_path)
            curve_payload = load_payload(curve_path)
            legacy_payload = load_payload(legacy_path)
            self.assertEqual(block_payload["compact_identity"]["materialization_version"], BLOCK_MATERIALIZATION_VERSION)
            resumed = Stage4Trainer.from_checkpoint(block_path, train_tokens, validation_tokens)
            self.assertIsInstance(resumed.raw_model.trajectory, BlockTrajectory)
            for payload, config in ((block_payload, mlp_block_config), (mlp_block_payload, block_config), (curve_payload, block_config), (legacy_payload, block_config)):
                with self.subTest(materialization_version=payload["compact_identity"]["materialization_version"]):
                    with self.assertRaisesRegex(ValueError, "compact_identity"):
                        validate_compatibility(payload, config)

    def test_07_full_block_dtype_movement_keeps_fixed_bases_nonpersistent_and_materializes_on_the_requested_dtype(self) -> None:
        trajectory = BlockTrajectory(self.full_block_config().sheet_geometry(), runtime_dtype=torch.float32)
        trajectory = trajectory.to(dtype=torch.float64)
        materialized = trajectory.materialize(ATTENTION_QUERY_WEIGHT, 0)
        self.assertEqual(materialized.dtype, torch.float64)
        self.assertEqual(trajectory.persistent_basis_keys(), ())


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
