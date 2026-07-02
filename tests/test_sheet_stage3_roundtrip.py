# vvv THOG
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from sheet.trainer import SharedTrainer
from tests.stage3_test_support import stage3_config, token_splits


class Stage3RoundTripTests(unittest.TestCase):
    def test_s3_06_checkpoint_round_trip(self) -> None:
        train_tokens, validation_tokens = token_splits()
        fixed = torch.arange(16, dtype=torch.long).view(2, 8) % 32
        with tempfile.TemporaryDirectory() as directory:
            trainer = SharedTrainer(
                stage3_config("thog2_sheet", max_updates=2),
                train_tokens,
                validation_tokens,
            )
            trainer.run(target_updates=1)
            trainer.model.eval()
            with torch.no_grad():
                before, _ = trainer.model(fixed)
            path = trainer.save_checkpoint(Path(directory) / "ckpt.pt")
            resumed = SharedTrainer.from_checkpoint(
                path,
                train_tokens,
                validation_tokens,
            )
            resumed.model.eval()
            with torch.no_grad():
                after, _ = resumed.model(fixed)
            for key, value in trainer.model.state_dict().items():
                self.assertTrue(
                    torch.equal(value, resumed.model.state_dict()[key]),
                    key,
                )
            torch.testing.assert_close(before, after, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
