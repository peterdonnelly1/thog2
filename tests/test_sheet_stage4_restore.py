# vvv THOG
from __future__ import annotations

import unittest

import torch

from sheet.compact_state import model_from_compact_state
from sheet.stage4_trainer import Stage4Trainer
from tests.stage4_test_support import stage4_tokens, stage4_training_config


class Stage4RestoreTests(unittest.TestCase):
    def test_s4_08_compact_model_restoration(self) -> None:
        train_tokens, validation_tokens = stage4_tokens()
        trainer = Stage4Trainer(
            stage4_training_config(checkpoint_segment_size=2),
            train_tokens,
            validation_tokens,
        )
        trainer.train_one_update()
        trainer.model.eval()
        inputs = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        with torch.no_grad():
            expected, _ = trainer.model(inputs)
        model, config = model_from_compact_state(trainer.checkpoint_payload())
        with torch.no_grad():
            actual, _ = model(inputs)
        self.assertEqual(config.model_type, "thog2_sheet")
        self.assertEqual(config.n_layer, trainer.config.n_layer)
        self.assertEqual(config.depth_order, trainer.config.depth_order)
        self.assertEqual(config.base_row_order, trainer.config.base_row_order)
        self.assertEqual(config.checkpoint_segment_size, 0)
        self.assertFalse(model.training)
        self.assertTrue(torch.allclose(expected, actual, rtol=1.0e-6, atol=1.0e-7))


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
