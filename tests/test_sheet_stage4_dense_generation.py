# vvv THOG
from __future__ import annotations

import unittest

import torch

from sheet.generation import generate_tokens
from sheet.stage4_trainer import Stage4Trainer
from tests.stage4_test_support import stage4_tokens, stage4_training_config


class Stage4DenseGenerationTests(unittest.TestCase):
    def test_s4_14_dense_generation_regression(self) -> None:
        train_tokens, validation_tokens = stage4_tokens()
        trainer = Stage4Trainer(
            stage4_training_config(model_type="dense"),
            train_tokens,
            validation_tokens,
        )
        trainer.model.eval()
        output = generate_tokens(
            trainer.model,
            torch.tensor([[1, 2, 3]], dtype=torch.long),
            device=trainer.device,
            dtype=trainer.config.dtype,
            max_new_tokens=2,
            top_k=8,
            seed=8,
        )
        self.assertEqual(tuple(output.shape), (1, 5))


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
