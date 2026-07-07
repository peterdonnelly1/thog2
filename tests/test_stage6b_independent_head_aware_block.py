# vvv THOG
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from sheet.block_trajectory import HEAD_AWARE_BLOCK_MATERIALIZATION_VERSION, BlockTrajectory
from sheet.checkpoints import load_payload, validate_compatibility
from sheet.compact_identity import (
    ATTENTION_GEOMETRY_CURVE,
    ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK,
    GEOMETRY_PRESET_BLOCK,
    GEOMETRY_PRESET_HEAD_AWARE_BLOCK,
    GEOMETRY_PRESET_MLP_BLOCK,
    MLP_GEOMETRY_CURVE,
    MLP_GEOMETRY_MLP_BLOCK,
)
from sheet.model import SheetGPT, SheetGPTConfig
from sheet.semantic_materializer import (
    ATTENTION_KEY_WEIGHT,
    ATTENTION_OUTPUT_WEIGHT,
    ATTENTION_QUERY_WEIGHT,
    ATTENTION_VALUE_WEIGHT,
    MLP_CONTRACTION_WEIGHT,
    MLP_EXPANSION_WEIGHT,
)
from sheet.stage4_trainer import Stage4Trainer
from tests.stage4_test_support import stage4_tokens, stage4_training_config


class Stage6bIndependentHeadAwareBlockTests(unittest.TestCase):
    def attention_only_config(self) -> SheetGPTConfig:
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
            attention_geometry=ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK,
            mlp_geometry=MLP_GEOMETRY_CURVE,
        )

    def test_01_explicit_head_aware_block_attention_with_curve_mlp_resolves_to_attention_only_ablation_identity_before_block_preset_is_considered(self) -> None:
        config = stage4_training_config(attention_geometry=ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK, mlp_geometry=MLP_GEOMETRY_CURVE)
        identity = config.compact_identity_metadata()
        self.assertEqual(identity["geometry_preset"], GEOMETRY_PRESET_HEAD_AWARE_BLOCK)
        self.assertEqual(identity["attention_geometry"], ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK)
        self.assertEqual(identity["mlp_geometry"], MLP_GEOMETRY_CURVE)
        self.assertEqual(identity["materialization_version"], HEAD_AWARE_BLOCK_MATERIALIZATION_VERSION)
        block_identity = stage4_training_config(geometry_preset=GEOMETRY_PRESET_BLOCK).compact_identity_metadata()
        self.assertEqual(block_identity["attention_geometry"], ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK)
        self.assertEqual(block_identity["mlp_geometry"], MLP_GEOMETRY_MLP_BLOCK)

    def test_02_independent_head_aware_block_trajectory_compacts_attention_per_head_but_leaves_mlp_on_curve_coefficients(self) -> None:
        model = SheetGPT(self.attention_only_config())
        trajectory = model.trajectory
        self.assertIsInstance(trajectory, BlockTrajectory)
        expected_shapes = {
            ATTENTION_QUERY_WEIGHT: (2, 3, 4, 8),
            ATTENTION_KEY_WEIGHT: (2, 3, 4, 8),
            ATTENTION_VALUE_WEIGHT: (2, 3, 4, 8),
            ATTENTION_OUTPUT_WEIGHT: (2, 3, 8, 4),
            MLP_EXPANSION_WEIGHT: (64, 16, 3),
            MLP_CONTRACTION_WEIGHT: (16, 64, 3),
        }
        for name, expected_shape in expected_shapes.items():
            with self.subTest(name=name):
                self.assertEqual(tuple(trajectory.coefficients[name].shape), expected_shape)
        report = {row["name"]: row for row in trajectory.family_report()}
        self.assertEqual(report[ATTENTION_QUERY_WEIGHT]["attention_head_axis"], "output")
        self.assertFalse(report[ATTENTION_QUERY_WEIGHT]["basis_crosses_attention_head_boundary"])
        self.assertEqual(report[MLP_EXPANSION_WEIGHT]["coefficient_shape"], (64, 16, 3))
        self.assertNotIn("attention_head_axis", report[MLP_EXPANSION_WEIGHT])

    def test_03_independent_head_aware_block_parameter_report_is_between_curve_and_full_block_because_only_attention_is_blocked(self) -> None:
        torch.manual_seed(4101)
        attention_only = SheetGPT(self.attention_only_config())
        curve = SheetGPT(SheetGPTConfig(**{**self.attention_only_config().to_dict(), "geometry_preset": GEOMETRY_PRESET_HEAD_AWARE_BLOCK, "attention_geometry": ATTENTION_GEOMETRY_CURVE, "mlp_geometry": MLP_GEOMETRY_CURVE}))
        full_block = SheetGPT(SheetGPTConfig(**{**self.attention_only_config().to_dict(), "geometry_preset": GEOMETRY_PRESET_BLOCK, "attention_geometry": None, "mlp_geometry": None}))
        attention_only_count = attention_only.parameter_report()["matrix_sheet_coefficients"]
        curve_count = curve.parameter_report()["matrix_sheet_coefficients"]
        full_block_count = full_block.parameter_report()["matrix_sheet_coefficients"]
        self.assertLess(attention_only_count, curve_count)
        self.assertGreater(attention_only_count, full_block_count)

    def test_04_independent_head_aware_block_checkpoint_round_trips_and_rejects_full_block_or_mlp_block_cross_loads(self) -> None:
        train_tokens, validation_tokens = stage4_tokens(128)
        attention_only_config = stage4_training_config(attention_geometry=ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK, mlp_geometry=MLP_GEOMETRY_CURVE, max_updates=1)
        full_block_config = stage4_training_config(geometry_preset=GEOMETRY_PRESET_BLOCK, max_updates=1)
        mlp_block_config = stage4_training_config(geometry_preset=GEOMETRY_PRESET_MLP_BLOCK, max_updates=1)
        attention_only_trainer = Stage4Trainer(attention_only_config, train_tokens, validation_tokens)
        full_block_trainer = Stage4Trainer(full_block_config, train_tokens, validation_tokens)
        mlp_block_trainer = Stage4Trainer(mlp_block_config, train_tokens, validation_tokens)
        with tempfile.TemporaryDirectory() as directory:
            attention_only_path = Path(directory) / "head_aware_block.pt"
            full_block_path = Path(directory) / "block.pt"
            mlp_block_path = Path(directory) / "mlp_block.pt"
            attention_only_trainer.save_checkpoint(attention_only_path)
            full_block_trainer.save_checkpoint(full_block_path)
            mlp_block_trainer.save_checkpoint(mlp_block_path)
            attention_only_payload = load_payload(attention_only_path)
            full_block_payload = load_payload(full_block_path)
            mlp_block_payload = load_payload(mlp_block_path)
            self.assertEqual(attention_only_payload["compact_identity"]["materialization_version"], HEAD_AWARE_BLOCK_MATERIALIZATION_VERSION)
            resumed = Stage4Trainer.from_checkpoint(attention_only_path, train_tokens, validation_tokens)
            self.assertEqual(resumed.config.compact_identity_metadata()["materialization_version"], HEAD_AWARE_BLOCK_MATERIALIZATION_VERSION)
            for payload, config in ((attention_only_payload, full_block_config), (full_block_payload, attention_only_config), (mlp_block_payload, attention_only_config)):
                with self.subTest(materialization_version=payload["compact_identity"]["materialization_version"]):
                    with self.assertRaisesRegex(ValueError, "compact_identity"):
                        validate_compatibility(payload, config)

    def test_05_independent_head_aware_block_tiny_train_step_updates_attention_block_and_mlp_curve_coefficients(self) -> None:
        train_tokens, validation_tokens = stage4_tokens(128)
        config = stage4_training_config(attention_geometry=ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK, mlp_geometry=MLP_GEOMETRY_CURVE, max_updates=1)
        trainer = Stage4Trainer(config, train_tokens, validation_tokens)
        metrics = trainer.train_one_update()
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
        self.assertEqual(trainer.state.completed_updates, 1)
        for name in (ATTENTION_QUERY_WEIGHT, ATTENTION_OUTPUT_WEIGHT, MLP_EXPANSION_WEIGHT, MLP_CONTRACTION_WEIGHT):
            with self.subTest(name=name):
                gradient = trainer.raw_model.model.trajectory.coefficients[name].grad
                self.assertIsNotNone(gradient)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
