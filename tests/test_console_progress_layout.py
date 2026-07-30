# vvv THOG
from __future__ import annotations

import re
import unittest
from types import SimpleNamespace

import run_thog2_owt  # noqa: F401  # <<< THOG apply the public terminal-colour and progress-format policy before formatting rows
from sheet.stage6_trainer import Stage6Trainer, format_progress_line


class ConsoleProgressLayoutTests(unittest.TestCase):
    def test_optimizer_payload_uses_exact_interval_mean_and_compact_numeric_fields(self) -> None:
        trainer = object.__new__(Stage6Trainer)
        trainer.state = SimpleNamespace(completed_updates=90)
        trainer._console_previous_completed_updates = 90
        trainer._console_previous_training_seconds = 0.0
        trainer._console_exact_training_seconds = 0.25
        trainer._console_latest_mean_step_seconds = None

        values = trainer._prepare_console_progress_payload(
            "optimizer_progress",
            {
                "completed_updates": "   100",
                "cumulative_training_seconds": "     0",
                "tok/s": "        8189",
                "consumed_tokens": "     999999999",
                "training_loss": "   3.2000",
                "learning_rate": " 9.000e-05",
                "gradient_norm": "   0.250",
            },
        )

        self.assertEqual(values["mean_step_seconds"], "  0.0250")
        self.assertEqual(values["tok/s"], "  8189")
        self.assertEqual(values["consumed_tokens"], "999,999,999")
        self.assertRegex(values["timestamp"], r"^\d{6}:\d{4}$")

    def test_validation_reuses_latest_step_mean_and_uses_two_level_yellow_emphasis(self) -> None:
        trainer = object.__new__(Stage6Trainer)
        trainer.state = SimpleNamespace(completed_updates=100)
        trainer._console_previous_completed_updates = 100
        trainer._console_previous_training_seconds = 120.0
        trainer._console_exact_training_seconds = 120.0
        trainer._console_latest_mean_step_seconds = 12.0

        values = trainer._prepare_console_progress_payload(
            "evaluation_completed",
            {
                "completed_updates": "   100",
                "cumulative_training_seconds": "   120",
                "tok/s": "        8189",
                "consumed_tokens": "      999999999",
                "training_loss": "   3.2000",
                "validation_loss": "   3.1990",
            },
        )
        values["timestamp"] = "260730:0840"
        line = format_progress_line("unchanging-run-id", "evaluation_completed", values)

        self.assertTrue(line.startswith("\033[33mV     100  260730:0840"))
        self.assertIn("Δstep= 12.0000s", line)
        self.assertIn("tok/s=  8189", line)
        self.assertIn("tokens=999,999,999", line)
        self.assertIn("training loss  =   3.2000", line)
        self.assertIn(
            "\033[1;33mvalidation loss=   3.1990\033[33m\033[0m",
            line,
        )
        self.assertNotIn("unchanging-run-id", line)
        self.assertEqual(len(re.findall(r"\033\[", line)), 4)


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
