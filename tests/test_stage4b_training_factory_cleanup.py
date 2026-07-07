# vvv THOG
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from sheet.curve_trajectory import CurveTrajectory
from sheet.semantic_materializer import LEGACY_ATTENTION_INPUT_WEIGHT
from sheet.stage4_trainer import Stage4Trainer
from sheet.trajectory import SheetTrajectory
from sheet.training_model import TrainingSheetGPT
from sheet.training_model_factory import build_training_model
from tests.stage4_test_support import stage4_tokens, stage4_training_config


class Stage4bTrainingFactoryCleanupTests(unittest.TestCase):
    def test_01_main_training_model_factory_builds_curve_sheetgpt_from_curve_training_config(self) -> None:
        config = stage4_training_config(geometry_preset="curve")
        model = build_training_model(config)
        self.assertIsInstance(model, TrainingSheetGPT)
        self.assertIsInstance(model.trajectory, CurveTrajectory)
        self.assertNotIn(LEGACY_ATTENTION_INPUT_WEIGHT, model.trajectory.coefficients)
        self.assertEqual(config.compact_identity_metadata()["materialization_version"], "curve_v1")

    def test_02_main_training_model_factory_preserves_legacy_sheet_col_training_config(self) -> None:
        config = stage4_training_config()
        model = build_training_model(config)
        self.assertIsInstance(model, TrainingSheetGPT)
        self.assertIsInstance(model.trajectory, SheetTrajectory)
        self.assertIn(LEGACY_ATTENTION_INPUT_WEIGHT, model.trajectory.coefficients)
        self.assertEqual(config.compact_identity_metadata()["materialization_version"], "legacy_sheet_col_v1")

    def test_03_stage4_trainer_uses_main_factory_for_curve_and_legacy_without_special_case_module(self) -> None:
        train_tokens, validation_tokens = stage4_tokens(128)
        curve_trainer = Stage4Trainer(stage4_training_config(geometry_preset="curve", max_updates=1), train_tokens, validation_tokens)
        legacy_trainer = Stage4Trainer(stage4_training_config(max_updates=1), train_tokens, validation_tokens)
        self.assertIsInstance(curve_trainer.raw_model.trajectory, CurveTrajectory)
        self.assertIsInstance(legacy_trainer.raw_model.trajectory, SheetTrajectory)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "curve_factory_cleanup.pt"
            curve_trainer.save_checkpoint(checkpoint_path)
            resumed = Stage4Trainer.from_checkpoint(checkpoint_path, train_tokens, validation_tokens)
        self.assertIsInstance(resumed.raw_model.trajectory, CurveTrajectory)
        self.assertEqual(resumed.config.compact_identity_metadata()["materialization_version"], "curve_v1")

    def test_04_temporary_stage4_training_model_factory_module_has_been_removed(self) -> None:
        self.assertIsNone(importlib.util.find_spec("sheet.stage4_training_model_factory"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
