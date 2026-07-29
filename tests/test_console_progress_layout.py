# vvv THOG
from __future__ import annotations

import re
import unittest
from types import SimpleNamespace

from sheet.stage6_trainer import Stage6Trainer, format_progress_line


class ConsoleProgressLayoutTests(unittest.TestCase):
    def test_optimizer_payload_uses_interval_mean_and_compact_numeric_fields(self) -> None:
        trainer = object.__new__(Stage6Trainer)
        trainer.state = SimpleNamespace(completed_updates=90)
        trainer._console_previous_completed_updates = 90
        trainer._console_previous_training_seconds = 0.0
        trainer._console_latest_mean_step_seconds = None

        values = trainer._prepare_console_progress_payload(
            "optimizer_progress",
            {
                "completed_updates": "   100",
                "cumulative_training_seconds": "   120",
                "tok/s": "        8189",
                "consumed_tokens": "    1234567890",
                "training_loss": "   3.2000",
                "learning_rate": " 9.000e-05",
                "gradient_norm": "   0.250",
            },
        )

        self.assertEqual(values["mean_step_seconds"], "12")
        self.assertEqual(values["tok/s"], "  8189")
        self.assertEqual(values["consumed_tokens"], " 1,234,567,890")
        self.assertRegex(values["timestamp"], r"^\d{6}-\d{4}$")

    def test_validation_reuses_latest_step_mean_and_highlights_only_validation_loss(self) -> None:
        trainer = object.__new__(Stage6Trainer)
        trainer.state = SimpleNamespace(completed_updates=100)
        trainer._console_previous_completed_updates = 100
        trainer._console_previous_training_seconds = 120.0
        trainer._console_latest_mean_step_seconds = 12.0

        values = trainer._prepare_console_progress_payload(
            "evaluation_completed",
            {
                "completed_updates": "   100",
                "cumulative_training_seconds": "   120",
                "tok/s": "        8189",
                "consumed_tokens": "    1234567890",
                "training_loss": "   3.2000",
                "validation_loss": "   3.1990",
            },
        )
        values["timestamp"] = "300726-0840"
        line = format_progress_line("unchanging-run-id", "evaluation_completed", values)

        self.assertIn("V     100  300726-0840", line)
        self.assertIn("Δ step=12s", line)
        self.assertIn("tok/s=  8189", line)
        self.assertIn("tokens= 1,234,567,890", line)
        self.assertIn("training loss  =   3.2000", line)
        self.assertIn("\033[1;93mvalidation loss=   3.1990\033[0m", line)
        self.assertNotIn("unchanging-run-id", line)
        self.assertFalse(line.startswith("\033["), "only validation loss should be coloured")
        self.assertEqual(len(re.findall(r"\033\[", line)), 2)


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
