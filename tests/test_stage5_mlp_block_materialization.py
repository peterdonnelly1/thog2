# vvv THOG
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from sheet.basis_kernel import DCT_BASIS_VERSION
from sheet.checkpoints import load_payload, validate_compatibility
from sheet.compact_identity import (
    ATTENTION_GEOMETRY_DEPTH,
    BASIS_FAMILY_CHEBYSHEV,
    BASIS_FAMILY_DCT,
    GEOMETRY_PRESET_DEPTH,
    GEOMETRY_PRESET_FULL_BLOCK,
    GEOMETRY_PRESET_MLP_BLOCK,
    MLP_GEOMETRY_MLP_BLOCK,
)
from sheet.depth_trajectory import DepthTrajectory
from sheet.mlp_block_trajectory import MLP_BLOCK_MATERIALIZATION_VERSION, MLP_BLOCK_MATRIX_FAMILIES, MlpBlockTrajectory
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


class Stage5MlpBlockMaterializationTests(unittest.TestCase):
    def mlp_block_config(self) -> SheetGPTConfig:
        return SheetGPTConfig(block_size=8, vocab_size=32, n_layer=4, n_head=2, n_embd=16, dropout=0.0, bias=True, depth_order=3, base_row_order=8, geometry_preset=GEOMETRY_PRESET_MLP_BLOCK)

    def test_01_mlp_block_config_identity_is_accepted_for_chebyshev_and_dct(self) -> None:
        preset = stage4_training_config(geometry_preset=GEOMETRY_PRESET_MLP_BLOCK)
        identity = preset.compact_identity_metadata()
        self.assertEqual(identity["geometry_preset"], GEOMETRY_PRESET_MLP_BLOCK)
        self.assertEqual(identity["attention_geometry"], ATTENTION_GEOMETRY_DEPTH)
        self.assertEqual(identity["mlp_geometry"], MLP_GEOMETRY_MLP_BLOCK)
        self.assertEqual(identity["basis_family"], BASIS_FAMILY_CHEBYSHEV)
        self.assertEqual(identity["materialization_version"], MLP_BLOCK_MATERIALIZATION_VERSION)
        explicit = stage4_training_config(attention_geometry=ATTENTION_GEOMETRY_DEPTH, mlp_geometry=MLP_GEOMETRY_MLP_BLOCK)
        self.assertEqual(explicit.compact_identity_metadata()["geometry_preset"], GEOMETRY_PRESET_MLP_BLOCK)
        self.assertEqual(stage4_training_config(geometry_preset=GEOMETRY_PRESET_FULL_BLOCK).compact_identity_metadata()["geometry_preset"], GEOMETRY_PRESET_FULL_BLOCK)
        dct_mlp_block = stage4_training_config(geometry_preset=GEOMETRY_PRESET_MLP_BLOCK, basis_family=BASIS_FAMILY_DCT)
        self.assertEqual(dct_mlp_block.compact_identity_metadata()["basis_version"], DCT_BASIS_VERSION)

    def test_02_mlp_block_trajectory_keeps_attention_depth_coefficients_and_replaces_only_mlp_coefficients_with_blocks(self) -> None:
        config = self.mlp_block_config()
        trajectory = MlpBlockTrajectory(config.sheet_geometry(), runtime_dtype=torch.float32, basis_version=config.basis_version)
        expected_shapes = {
            ATTENTION_QUERY_WEIGHT: (16, 16, 3),
            ATTENTION_KEY_WEIGHT: (16, 16, 3),
            ATTENTION_VALUE_WEIGHT: (16, 16, 3),
            ATTENTION_OUTPUT_WEIGHT: (16, 16, 3),
            MLP_EXPANSION_WEIGHT: (3, 32, 8),
            MLP_CONTRACTION_WEIGHT: (3, 8, 32),
        }
        self.assertEqual(MLP_BLOCK_MATRIX_FAMILIES, tuple(expected_shapes))
        for name, shape in expected_shapes.items():
            with self.subTest(name=name):
                self.assertEqual(tuple(trajectory.coefficients[name].shape), shape)
        self.assertNotIn(LEGACY_ATTENTION_INPUT_WEIGHT, trajectory.coefficients)

    def test_03_mlp_block_materialization_matches_manual_depth_output_and_input_basis_contraction(self) -> None:
        config = self.mlp_block_config()
        trajectory = MlpBlockTrajectory(config.sheet_geometry(), runtime_dtype=torch.float32, basis_version=config.basis_version)
        with torch.no_grad():
            trajectory.coefficients[MLP_EXPANSION_WEIGHT].zero_()
            trajectory.coefficients[MLP_CONTRACTION_WEIGHT].zero_()
            trajectory.coefficients[MLP_EXPANSION_WEIGHT][:, 2, 3] = torch.tensor([1.0, 2.0, 3.0])
            trajectory.coefficients[MLP_CONTRACTION_WEIGHT][:, 4, 5] = torch.tensor([-1.0, 0.5, 2.0])
        layer_index = 1
        depth = trajectory.depth_basis[layer_index]
        expansion = trajectory.materialize(MLP_EXPANSION_WEIGHT, layer_index)
        collapsed = torch.einsum("p,pab->ab", depth, trajectory.coefficients[MLP_EXPANSION_WEIGHT])
        manual = trajectory.output_basis(MLP_EXPANSION_WEIGHT) @ collapsed @ trajectory.input_basis(MLP_EXPANSION_WEIGHT).transpose(0, 1)
        torch.testing.assert_close(expansion, manual, rtol=0.0, atol=0.0)
        torch.testing.assert_close(trajectory.direct_value(MLP_EXPANSION_WEIGHT, layer_index, 7, 5), expansion[7, 5], rtol=0.0, atol=0.0)
        packed = trajectory.materialize(LEGACY_ATTENTION_INPUT_WEIGHT, layer_index)
        reconstructed = torch.cat((trajectory.materialize(ATTENTION_QUERY_WEIGHT, layer_index), trajectory.materialize(ATTENTION_KEY_WEIGHT, layer_index), trajectory.materialize(ATTENTION_VALUE_WEIGHT, layer_index)), dim=0)
        torch.testing.assert_close(packed, reconstructed, rtol=0.0, atol=0.0)

    def test_04_mlp_block_parameter_report_is_smaller_than_depth_for_mlp_matrices_but_keeps_attention_depth_only(self) -> None:
        torch.manual_seed(4101)
        mlp_block_model = SheetGPT(self.mlp_block_config())
        depth_config = self.mlp_block_config().to_dict()
        depth_config["geometry_preset"] = GEOMETRY_PRESET_DEPTH
        depth_model = SheetGPT(SheetGPTConfig(**depth_config))
        report = mlp_block_model.parameter_report()
        depth_report = depth_model.parameter_report()
        width = mlp_block_model.config.n_embd
        depth_order = mlp_block_model.config.depth_order
        attention_depth_coefficients = depth_order * 4 * width * width
        mlp_block_coefficients = depth_order * 32 * 8 + depth_order * 8 * 32
        self.assertEqual(report["matrix_sheet_coefficients"], attention_depth_coefficients + mlp_block_coefficients)
        self.assertLess(report["matrix_sheet_coefficients"], depth_report["matrix_sheet_coefficients"])
        self.assertEqual(mlp_block_model.compact_state_violations(), ())

    def test_05_mlp_block_model_forward_backward_sends_gradients_to_attention_depth_and_mlp_block_coefficients(self) -> None:
        torch.manual_seed(4101)
        model = SheetGPT(self.mlp_block_config())
        idx = torch.randint(0, model.config.vocab_size, (2, 4))
        targets = torch.randint(0, model.config.vocab_size, (2, 4))
        logits, loss = model(idx, targets)
        self.assertEqual(tuple(logits.shape), (2, 4, model.config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name in (ATTENTION_QUERY_WEIGHT, ATTENTION_KEY_WEIGHT, ATTENTION_VALUE_WEIGHT, MLP_EXPANSION_WEIGHT, MLP_CONTRACTION_WEIGHT):
            with self.subTest(name=name):
                gradient = model.trajectory.coefficients[name].grad
                self.assertIsNotNone(gradient)
                self.assertGreater(float(gradient.abs().sum().item()), 0.0)
        self.assertIsNotNone(model.trajectory.coefficients[LEGACY_ATTENTION_INPUT_BIAS].grad)

    def test_06_mlp_block_checkpoint_identity_round_trips_and_rejects_depth_or_legacy_cross_loads(self) -> None:
        train_tokens, validation_tokens = stage4_tokens(128)
        mlp_config = stage4_training_config(geometry_preset=GEOMETRY_PRESET_MLP_BLOCK, max_updates=1)
        depth_config = stage4_training_config(geometry_preset=GEOMETRY_PRESET_DEPTH, max_updates=1)
        legacy_config = stage4_training_config(max_updates=1)
        mlp_trainer = Stage4Trainer(mlp_config, train_tokens, validation_tokens)
        depth_trainer = Stage4Trainer(depth_config, train_tokens, validation_tokens)
        legacy_trainer = Stage4Trainer(legacy_config, train_tokens, validation_tokens)
        self.assertIsInstance(mlp_trainer.raw_model.trajectory, MlpBlockTrajectory)
        self.assertIsInstance(depth_trainer.raw_model.trajectory, DepthTrajectory)
        self.assertIsInstance(legacy_trainer.raw_model.trajectory, SheetTrajectory)
        with tempfile.TemporaryDirectory() as directory:
            mlp_path = Path(directory) / "mlp_block.pt"
            depth_path = Path(directory) / "depth.pt"
            legacy_path = Path(directory) / "legacy.pt"
            mlp_trainer.save_checkpoint(mlp_path)
            depth_trainer.save_checkpoint(depth_path)
            legacy_trainer.save_checkpoint(legacy_path)
            mlp_payload = load_payload(mlp_path)
            depth_payload = load_payload(depth_path)
            legacy_payload = load_payload(legacy_path)
            self.assertEqual(mlp_payload["compact_identity"]["materialization_version"], MLP_BLOCK_MATERIALIZATION_VERSION)
            resumed = Stage4Trainer.from_checkpoint(mlp_path, train_tokens, validation_tokens)
            self.assertIsInstance(resumed.raw_model.trajectory, MlpBlockTrajectory)
            with self.assertRaisesRegex(ValueError, "compact_identity"):
                validate_compatibility(mlp_payload, depth_config)
            with self.assertRaisesRegex(ValueError, "compact_identity"):
                validate_compatibility(depth_payload, mlp_config)
            with self.assertRaisesRegex(ValueError, "compact_identity"):
                validate_compatibility(legacy_payload, mlp_config)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
