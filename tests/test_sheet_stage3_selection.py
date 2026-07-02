# vvv THOG
from __future__ import annotations

import unittest

from model import GPT
from sheet.model import SheetGPT
from sheet.trainer import SharedTrainer
from sheet.training_config import TrainingConfig
from tests.stage3_test_support import stage3_config, token_splits


class Stage3SelectionTests(unittest.TestCase):
    def test_s3_01_model_selection(self) -> None:
        train_tokens, validation_tokens = token_splits()
        dense = SharedTrainer(
            stage3_config("dense"),
            train_tokens,
            validation_tokens,
        )
        sheet = SharedTrainer(
            stage3_config("thog2_sheet"),
            train_tokens,
            validation_tokens,
        )
        self.assertIsInstance(dense.model, GPT)
        self.assertIsInstance(sheet.model, SheetGPT)
        with self.assertRaisesRegex(ValueError, "model_type"):
            TrainingConfig(model_type="invalid")

    def test_s3_04_data_rng_separation(self) -> None:
        train_tokens, validation_tokens = token_splits()
        dense = SharedTrainer(
            stage3_config("dense", model_seed=1),
            train_tokens,
            validation_tokens,
        )
        sheet = SharedTrainer(
            stage3_config("thog2_sheet", model_seed=999),
            train_tokens,
            validation_tokens,
        )
        dense_batches = [
            dense.batch_source.get_batch("train", device="cpu").starts
            for _ in range(5)
        ]
        sheet_batches = [
            sheet.batch_source.get_batch("train", device="cpu").starts
            for _ in range(5)
        ]
        self.assertEqual(dense_batches, sheet_batches)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
