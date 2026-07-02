# vvv THOG
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from sheet.stage4_trainer import Stage4Trainer
from sheet.trainer import SharedTrainer
from tests.stage4_test_support import stage4_tokens, stage4_training_config


class Stage4SaveTimelineTests(unittest.TestCase):
    def test_s4_07_checkpoint_phase_hook(self) -> None:
        train_tokens, validation_tokens = stage4_tokens()
        trainer = Stage4Trainer(
            stage4_training_config(checkpoint_segment_size=2),
            train_tokens,
            validation_tokens,
        )
        with patch.object(
            SharedTrainer,
            "save_checkpoint",
            return_value=Path("state.pt"),
        ):
            trainer.save_checkpoint(Path("state.pt"))
        self.assertEqual(
            trainer.memory_telemetry.report()["samples"][-1]["phase"],
            "checkpoint",
        )


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
