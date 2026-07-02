# vvv THOG
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from sheet.checkpoints import optimizer_group_names
from sheet.trainer import SharedTrainer
from tests.stage3_test_support import (
    assert_nested_equal,
    stage3_config,
    token_splits,
)


class Stage3ResumeTests(unittest.TestCase):
    def test_s3_07_resume_optimizer_state(self) -> None:
        train_tokens, validation_tokens = token_splits()
        with tempfile.TemporaryDirectory() as directory:
            trainer = SharedTrainer(
                stage3_config("thog2_sheet", max_updates=3),
                train_tokens,
                validation_tokens,
            )
            trainer.run(target_updates=2)
            path = trainer.save_checkpoint(Path(directory) / "ckpt.pt")
            resumed = SharedTrainer.from_checkpoint(
                path,
                train_tokens,
                validation_tokens,
            )
            self.assertEqual(
                optimizer_group_names(trainer.optimizer),
                optimizer_group_names(resumed.optimizer),
            )
            assert_nested_equal(
                self,
                trainer.optimizer.state_dict(),
                resumed.optimizer.state_dict(),
            )

    def test_s3_08_resume_update_and_scheduler(self) -> None:
        train_tokens, validation_tokens = token_splits()
        with tempfile.TemporaryDirectory() as directory:
            original = SharedTrainer(
                stage3_config("thog2_sheet", max_updates=4),
                train_tokens,
                validation_tokens,
            )
            original.run(target_updates=2)
            path = original.save_checkpoint(Path(directory) / "ckpt.pt")
            resumed = SharedTrainer.from_checkpoint(
                path,
                train_tokens,
                validation_tokens,
            )
            self.assertEqual(
                original.state.completed_updates,
                resumed.state.completed_updates,
            )
            self.assertEqual(
                original.learning_rate_for_update(2),
                resumed.learning_rate_for_update(2),
            )
            original.train_one_update()
            resumed.train_one_update()
            for key, value in original.model.state_dict().items():
                torch.testing.assert_close(
                    value,
                    resumed.model.state_dict()[key],
                    rtol=0.0,
                    atol=0.0,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
