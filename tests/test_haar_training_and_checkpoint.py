# vvv THOG
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from sheet.bases.haar import BASIS_FAMILY_HAAR, HAAR_BASIS_VERSION
from sheet.checkpoints import load_payload, validate_compatibility
from sheet.compact_identity import (
    BASIS_FAMILY_CHEBYSHEV,
    BASIS_FAMILY_DCT,
    GEOMETRY_PRESET_DEPTH,
    GEOMETRY_PRESET_FULL_BLOCK,
    GEOMETRY_PRESET_HEAD_AWARE_BLOCK,
    GEOMETRY_PRESET_LEGACY_SHEET_COL,
    GEOMETRY_PRESET_MLP_BLOCK,
)
from sheet.run_config import OwtRunConfig
from sheet.stage4_trainer import Stage4Trainer
from tests.stage4_test_support import stage4_tokens, stage4_training_config


class BalancedHaarTrainingAndCheckpointTests(unittest.TestCase):
    def run_one_update_for_preset(self, geometry_preset: str) -> Stage4Trainer:
        train_tokens, validation_tokens = stage4_tokens(256)
        config = stage4_training_config(
            geometry_preset=geometry_preset,
            basis_family=BASIS_FAMILY_HAAR,
            max_updates=1,
            decay_updates=1,
        )
        self.assertEqual(config.basis_version, HAAR_BASIS_VERSION)
        trainer = Stage4Trainer(config, train_tokens, validation_tokens)
        metrics = trainer.train_one_update()
        self.assertTrue(torch.isfinite(torch.tensor(metrics["training_loss"])))
        self.assertEqual(trainer.state.completed_updates, 1)
        identity = trainer.config.compact_identity_metadata()
        self.assertEqual(identity["basis_family"], BASIS_FAMILY_HAAR)
        self.assertEqual(identity["basis_version"], HAAR_BASIS_VERSION)
        return trainer

    def test_01_every_existing_compact_geometry_runs_a_tiny_haar_train_step(self) -> None:
        presets = (
            GEOMETRY_PRESET_LEGACY_SHEET_COL,
            GEOMETRY_PRESET_DEPTH,
            GEOMETRY_PRESET_MLP_BLOCK,
            GEOMETRY_PRESET_HEAD_AWARE_BLOCK,
            GEOMETRY_PRESET_FULL_BLOCK,
        )
        for geometry_preset in presets:
            with self.subTest(geometry_preset=geometry_preset):
                trainer = self.run_one_update_for_preset(geometry_preset)
                self.assertGreater(trainer.parameter_report["sheet_coefficients"], 0)

    def test_02_checkpoint_records_haar_identity_and_resumes(self) -> None:
        train_tokens, validation_tokens = stage4_tokens(256)
        config = stage4_training_config(
            geometry_preset=GEOMETRY_PRESET_FULL_BLOCK,
            basis_family=BASIS_FAMILY_HAAR,
            max_updates=2,
            decay_updates=2,
        )
        trainer = Stage4Trainer(config, train_tokens, validation_tokens)
        trainer.train_one_update()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "haar_full_block.pt"
            trainer.save_checkpoint(path)
            payload = load_payload(path)
            self.assertEqual(payload["compact_identity"]["basis_family"], BASIS_FAMILY_HAAR)
            self.assertEqual(payload["compact_identity"]["basis_version"], HAAR_BASIS_VERSION)
            resumed = Stage4Trainer.from_checkpoint(
                path,
                train_tokens,
                validation_tokens,
                overrides={"max_updates": 2},
            )
            self.assertEqual(resumed.config.basis_family, BASIS_FAMILY_HAAR)
            self.assertEqual(resumed.config.basis_version, HAAR_BASIS_VERSION)
            resumed.train_one_update()
            self.assertEqual(resumed.state.completed_updates, 2)

    def test_03_cross_family_checkpoint_compatibility_fails(self) -> None:
        train_tokens, validation_tokens = stage4_tokens(256)
        configs = {
            BASIS_FAMILY_HAAR: stage4_training_config(
                geometry_preset=GEOMETRY_PRESET_DEPTH,
                basis_family=BASIS_FAMILY_HAAR,
                max_updates=1,
            ),
            BASIS_FAMILY_CHEBYSHEV: stage4_training_config(
                geometry_preset=GEOMETRY_PRESET_DEPTH,
                basis_family=BASIS_FAMILY_CHEBYSHEV,
                max_updates=1,
            ),
            BASIS_FAMILY_DCT: stage4_training_config(
                geometry_preset=GEOMETRY_PRESET_DEPTH,
                basis_family=BASIS_FAMILY_DCT,
                max_updates=1,
            ),
        }
        trainers = {
            family: Stage4Trainer(config, train_tokens, validation_tokens)
            for family, config in configs.items()
        }
        with tempfile.TemporaryDirectory() as directory:
            payloads = {}
            for family, trainer in trainers.items():
                path = Path(directory) / f"{family}.pt"
                trainer.save_checkpoint(path)
                payloads[family] = load_payload(path)
            for source_family, payload in payloads.items():
                for target_family, target_config in configs.items():
                    if source_family == target_family:
                        validate_compatibility(payload, target_config)
                    else:
                        with self.assertRaisesRegex(ValueError, "compact_identity"):
                            validate_compatibility(payload, target_config)

    def test_04_run_configuration_uses_registry_version_and_haar_artifact_tag(self) -> None:
        config = OwtRunConfig(
            model_type="sheet",
            basis_family=BASIS_FAMILY_HAAR,
            basis_version="auto",
        )
        self.assertEqual(config.basis_version, HAAR_BASIS_VERSION)
        self.assertEqual(config.compact_artifact_fragment(), "HAAR_DEPTH")


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
