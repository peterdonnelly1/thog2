# vvv THOG
from __future__ import annotations

import unittest

from sheet.stage4_trainer import Stage4Trainer
from tests.stage4_test_support import stage4_tokens, stage4_training_config


class Stage4TimelineTests(unittest.TestCase):
    def test_s4_07_distinct_allocation_and_update_phases(self) -> None:
        train_tokens, validation_tokens = stage4_tokens()
        trainer = Stage4Trainer(
            stage4_training_config(checkpoint_segment_size=2),
            train_tokens,
            validation_tokens,
        )
        trainer.train_one_update()
        trainer.train_one_update()
        trainer.evaluate()
        phases = [
            sample["phase"]
            for sample in trainer.memory_telemetry.report()["samples"]
        ]
        self.assertEqual(
            phases,
            [
                "trainer_start",
                "model_construction",
                "optimizer_allocation",
                "first_optimizer_state",
                "steady_update",
                "evaluation",
            ],
        )


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
