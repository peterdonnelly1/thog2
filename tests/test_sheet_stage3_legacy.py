# vvv THOG
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from sheet.trainer import SharedTrainer
from tests.stage3_test_support import stage3_config, token_splits


class Stage3LegacyCheckpointTests(unittest.TestCase):
    def test_s3_10_legacy_dense_checkpoint(self) -> None:
        train_tokens, validation_tokens = token_splits()
        dense_config = stage3_config("dense", max_updates=2)
        trainer = SharedTrainer(
            dense_config,
            train_tokens,
            validation_tokens,
        )
        trainer.run(target_updates=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save(
                {
                    "model": trainer.model.state_dict(),
                    "optimizer": trainer.optimizer.state_dict(),
                    "model_args": dense_config.model_arguments(),
                    "iter_num": trainer.state.completed_updates,
                    "best_val_loss": trainer.state.best_validation_loss,
                },
                path,
            )
            loaded = SharedTrainer.from_checkpoint(
                path,
                train_tokens,
                validation_tokens,
                expected_config=dense_config,
            )
            self.assertEqual(loaded.state.completed_updates, 1)
            for key, value in trainer.model.state_dict().items():
                self.assertTrue(
                    torch.equal(value, loaded.model.state_dict()[key])
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
