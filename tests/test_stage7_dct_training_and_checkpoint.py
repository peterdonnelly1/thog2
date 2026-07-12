# vvv THOG
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from sheet.basis_kernel import DCT_BASIS_VERSION
from sheet.checkpoints import load_payload, validate_compatibility
from sheet.compact_identity import BASIS_FAMILY_CHEBYSHEV, BASIS_FAMILY_DCT, GEOMETRY_PRESET_DEPTH, GEOMETRY_PRESET_FULL_BLOCK, GEOMETRY_PRESET_HEAD_AWARE_BLOCK, GEOMETRY_PRESET_MLP_BLOCK
from sheet.stage4_trainer import Stage4Trainer
from tests.stage4_test_support import stage4_tokens, stage4_training_config


class Stage7DctTrainingAndCheckpointTests(unittest.TestCase):
    def run_one_update_for_preset(self, geometry_preset: str) -> Stage4Trainer:
        train_tokens, validation_tokens = stage4_tokens(256)
        config = stage4_training_config(geometry_preset=geometry_preset, basis_family=BASIS_FAMILY_DCT, max_updates=1, decay_updates=1)
        self.assertEqual(config.basis_version, DCT_BASIS_VERSION)
        trainer = Stage4Trainer(config, train_tokens, validation_tokens)
        metrics = trainer.train_one_update()
        self.assertTrue(torch.isfinite(torch.tensor(metrics["training_loss"])))
        self.assertEqual(trainer.state.completed_updates, 1)
        self.assertEqual(trainer.config.compact_identity_metadata()["basis_family"], BASIS_FAMILY_DCT)
        self.assertEqual(trainer.config.compact_identity_metadata()["basis_version"], DCT_BASIS_VERSION)
        return trainer

    def test_01_dct_depth_head_aware_mlp_block_and_full_block_each_run_a_tiny_train_step_without_geometry_specific_code_changes(self) -> None:
        for geometry_preset in (GEOMETRY_PRESET_DEPTH, GEOMETRY_PRESET_HEAD_AWARE_BLOCK, GEOMETRY_PRESET_MLP_BLOCK, GEOMETRY_PRESET_FULL_BLOCK):
            with self.subTest(geometry_preset=geometry_preset):
                trainer = self.run_one_update_for_preset(geometry_preset)
                self.assertGreater(trainer.parameter_report["sheet_coefficients"], 0)

    def test_02_dct_checkpoint_payload_records_dct_basis_version_and_resumes_with_the_same_identity(self) -> None:
        train_tokens, validation_tokens = stage4_tokens(256)
        config = stage4_training_config(geometry_preset=GEOMETRY_PRESET_FULL_BLOCK, basis_family=BASIS_FAMILY_DCT, max_updates=2, decay_updates=2)
        trainer = Stage4Trainer(config, train_tokens, validation_tokens)
        trainer.train_one_update()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dct_full_block.pt"
            trainer.save_checkpoint(path)
            payload = load_payload(path)
            self.assertEqual(payload["compact_identity"]["basis_family"], BASIS_FAMILY_DCT)
            self.assertEqual(payload["compact_identity"]["basis_version"], DCT_BASIS_VERSION)
            resumed = Stage4Trainer.from_checkpoint(path, train_tokens, validation_tokens, overrides={"max_updates": 2})
            self.assertEqual(resumed.config.basis_family, BASIS_FAMILY_DCT)
            self.assertEqual(resumed.config.basis_version, DCT_BASIS_VERSION)
            resumed.train_one_update()
            self.assertEqual(resumed.state.completed_updates, 2)

    def test_03_chebyshev_and_dct_checkpoint_cross_loads_hard_fail_even_when_geometry_and_shapes_match(self) -> None:
        train_tokens, validation_tokens = stage4_tokens(256)
        dct_config = stage4_training_config(geometry_preset=GEOMETRY_PRESET_DEPTH, basis_family=BASIS_FAMILY_DCT, max_updates=1)
        cheby_config = stage4_training_config(geometry_preset=GEOMETRY_PRESET_DEPTH, basis_family=BASIS_FAMILY_CHEBYSHEV, max_updates=1)
        dct_trainer = Stage4Trainer(dct_config, train_tokens, validation_tokens)
        cheby_trainer = Stage4Trainer(cheby_config, train_tokens, validation_tokens)
        with tempfile.TemporaryDirectory() as directory:
            dct_path = Path(directory) / "dct_depth.pt"
            cheby_path = Path(directory) / "cheby_depth.pt"
            dct_trainer.save_checkpoint(dct_path)
            cheby_trainer.save_checkpoint(cheby_path)
            dct_payload = load_payload(dct_path)
            cheby_payload = load_payload(cheby_path)
            with self.assertRaisesRegex(ValueError, "compact_identity"):
                validate_compatibility(dct_payload, cheby_config)
            with self.assertRaisesRegex(ValueError, "compact_identity"):
                validate_compatibility(cheby_payload, dct_config)

    def test_04_dct_basis_family_rejects_non_default_wrong_basis_version_to_avoid_silent_cache_poisoning(self) -> None:
        with self.assertRaisesRegex(ValueError, "basis_version"):
            stage4_training_config(geometry_preset=GEOMETRY_PRESET_FULL_BLOCK, basis_family=BASIS_FAMILY_DCT, basis_version="not_the_dct_version")


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
