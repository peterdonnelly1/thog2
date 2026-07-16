# vvv THOG
from __future__ import annotations

import io
import math
import unittest
from contextlib import redirect_stdout
from dataclasses import replace

from run_thog2_owt import print_schedule_startup
from sheet.lr_schedule import COSINE_SCHEDULE, learning_rate_for_config
from sheet.run_config import OwtRunConfig
from sheet.training_config import TrainingConfig


class ResumeScheduleCalculationTests(unittest.TestCase):
    def config(self, **changes) -> TrainingConfig:
        values = dict(
            model_type="dense", block_size=8, vocab_size=32, n_layer=1, n_head=1, n_embd=8,
            depth_order=1, base_row_order=1, batch_size=1, gradient_accumulation_steps=1,
            max_updates=10, learning_rate=1.0e-3, min_learning_rate=1.0e-4,
            warmup_updates=2, decay_updates=10, eval_batches=1, log_interval=1,
            residual_init_depth_source="true_layer_depth", device="cpu", dtype="float32",
        )
        values.update(changes)
        return TrainingConfig(**values)

    def test_resume_before_warmup_end_matches_original_schedule(self) -> None:
        config = self.config()
        self.assertEqual(learning_rate_for_config(config, 0), 1.0e-3 / 3.0)
        self.assertEqual(learning_rate_for_config(config, 1), 2.0e-3 / 3.0)

    def test_resume_mid_cosine_matches_original_schedule(self) -> None:
        config = self.config()
        expected = 1.0e-4 + 0.5 * (1.0 + math.cos(math.pi * 0.5)) * 9.0e-4
        self.assertTrue(math.isclose(learning_rate_for_config(config, 6), expected))

    def test_larger_total_steps_does_not_change_original_decay_endpoint_or_lr(self) -> None:
        original = self.config(max_updates=10)
        extended = replace(original, max_updates=50)
        self.assertEqual(extended.decay_updates, 10)
        for step in (0, 1, 2, 6, 10, 11, 49):
            self.assertEqual(learning_rate_for_config(original, step), learning_rate_for_config(extended, step))

    def test_extension_after_original_decay_end_holds_minimum_lr(self) -> None:
        config = self.config(max_updates=50)
        self.assertEqual(learning_rate_for_config(config, 11), config.min_learning_rate)
        self.assertEqual(learning_rate_for_config(config, 49), config.min_learning_rate)

    def test_startup_console_uses_first_resumed_step_lr_and_no_restored_position(self) -> None:
        run = OwtRunConfig(model_type="dense", max_iters=50, decay_iters=10, warmup_iters=2, learning_rate=1.0e-3, min_lr=1.0e-4, residual_init_depth_source="true_layer_depth", device="cpu", dtype="float32")
        config = self.config(max_updates=50)
        output = io.StringIO()
        with redirect_stdout(output):
            print_schedule_startup("resume", run, 10, config)
        text = output.getvalue()
        self.assertIn("first resumed step LR:", text)
        self.assertIn("original decay end:          10", text)
        self.assertIn("hold minimum LR", text)
        self.assertNotIn("restored position", text)
        self.assertNotIn("next learning rate", text)

    def test_resume_schedule_kind_remains_original_cosine(self) -> None:
        self.assertEqual(self.config().lr_schedule_kind, COSINE_SCHEDULE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
