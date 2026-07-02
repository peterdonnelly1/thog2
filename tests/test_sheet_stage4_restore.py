# vvv THOG
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            trainer.save_checkpoint(path)
            restored = Stage4Trainer.resume_for_inference(path, train_tokens)
        self.assertEqual(restored.config.model_type, "thog2_sheet")
        self.assertEqual(restored.config.n_layer, trainer.config.n_layer)
        self.assertEqual(restored.config.depth_order, trainer.config.depth_order)
        self.assertEqual(restored.config.base_row_order, trainer.config.base_row_order)
        self.assertEqual(restored.config.checkpoint_segment_size, 0)
        self.assertFalse(restored.model.training)


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
