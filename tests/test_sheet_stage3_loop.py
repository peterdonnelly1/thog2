# vvv THOG
from __future__ import annotations

import unittest

from sheet.trainer import SharedTrainer
from tests.stage3_test_support import stage3_config, token_splits


class Stage3LoopTests(unittest.TestCase):
    def test_s3_02_shared_loop_parity(self) -> None:
        train_tokens, validation_tokens = token_splits()
        trainers = [
            SharedTrainer(
                stage3_config(model_type, max_updates=2),
                train_tokens,
                validation_tokens,
            )
            for model_type in ("dense", "thog2_sheet")
        ]
        for trainer in trainers:
            trainer.run()
        event_names = [
            tuple(event.name for event in trainer.events)
            for trainer in trainers
        ]
        self.assertEqual(event_names[0], event_names[1])
        self.assertEqual(
            trainers[0].batch_source.training_trace(),
            trainers[1].batch_source.training_trace(),
        )

    def test_s3_03_completed_update_semantics(self) -> None:
        train_tokens, validation_tokens = token_splits()
        trainer = SharedTrainer(
            stage3_config(
                "thog2_sheet",
                max_updates=3,
                eval_interval=2,
            ),
            train_tokens,
            validation_tokens,
        )
        trainer.run()
        self.assertEqual(trainer.state.completed_updates, 3)
        completed = [
            event for event in trainer.events
            if event.name == "optimizer_step_completed"
        ]
        self.assertEqual(
            [event.completed_updates for event in completed],
            [1, 2, 3],
        )
        microbatches = [
            event for event in trainer.events
            if event.name == "microbatch"
        ]
        self.assertEqual(
            len(microbatches),
            3 * trainer.config.gradient_accumulation_steps,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
