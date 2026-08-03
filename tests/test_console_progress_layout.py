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
        trainer._console_previous_reported_training_loss = 3.3

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
        self.assertEqual(values["training_loss_delta"], "  -0.100")
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
            "\033[1;38;2;255;255;0mvalidation loss=   3.1990\033[33m\033[0m",
            line,
        )
        self.assertNotIn("\033[1;93m", line)
        self.assertNotIn("unchanging-run-id", line)
        self.assertEqual(len(re.findall(r"\033\[", line)), 4)

    # vvv THOG show current layer count only on PLASTIC DEPTH training and validation rows
    def test_current_layer_count_is_appended_only_when_plastic_is_enabled(self) -> None:
        trainer = object.__new__(Stage6Trainer)
        trainer.state = SimpleNamespace(completed_updates=8)
        trainer._console_previous_completed_updates = 8
        trainer._console_previous_training_seconds = 7.0
        trainer._console_exact_training_seconds = 7.0
        trainer._console_latest_mean_step_seconds = 0.875
        trainer._console_previous_reported_training_loss = 10.1
        trainer._plastic_depth_lattice = lambda: SimpleNamespace(current_active_layers=4)

        base_payload = {
            "completed_updates": "     8",
            "cumulative_training_seconds": "     7",
            "tok/s": "        2300",
            "consumed_tokens": "          16384",
            "training_loss": "  10.0000",
            "learning_rate": " 1.000e-05",
            "gradient_norm": "   2.000",
        }

        trainer.config = SimpleNamespace(plastic__enabled=False)
        ordinary_values = trainer._prepare_console_progress_payload(
            "optimizer_progress",
            base_payload,
        )
        ordinary_line = format_progress_line(
            "ordinary",
            "optimizer_progress",
            ordinary_values,
        )
        self.assertNotIn("current_layer_count", ordinary_line)

        trainer.config = SimpleNamespace(plastic__enabled=True)
        lattice = SimpleNamespace(current_active_layers=6)
        trainer._plastic_depth_lattice = lambda: lattice
        plastic_training_values = trainer._prepare_console_progress_payload(
            "optimizer_progress",
            base_payload,
        )
        plastic_training_line = format_progress_line(
            "plastic",
            "optimizer_progress",
            plastic_training_values,
        )
        self.assertTrue(plastic_training_line.endswith("current_layer_count = 6"))

        lattice.current_active_layers = 5
        plastic_validation_values = trainer._prepare_console_progress_payload(
            "evaluation_completed",
            {
                **base_payload,
                "validation_loss": "   9.9000",
            },
        )
        plastic_validation_line = format_progress_line(
            "plastic",
            "evaluation_completed",
            plastic_validation_values,
        )
        self.assertIn("current_layer_count = 5", plastic_validation_line)
        self.assertTrue(plastic_validation_line.endswith("current_layer_count = 5\033[0m"))
    # ^^^ THOG

    # vvv THOG keep semantic-QKV bypass reporting for non-DEPTH geometries while suppressing it for DEPTH
    def test_semantic_qkv_bypass_is_suppressed_only_for_depth_reporting(self) -> None:
        model_config = SimpleNamespace(
            fast_discard=True,
            bypass_semantic_qkv_adapter=True,
            vectorise_per_head_materialisation=False,
            direct_factorised_mlp=False,
            depth_compress_layer_norm_and_bias=False,
        )
        model = SimpleNamespace(
            config=model_config,
            trajectory=SimpleNamespace(),
            _supports_direct_factorised_mlp=lambda: False,
        )
        trainer = SimpleNamespace(raw_model=model)

        depth_config = SimpleNamespace(
            model_type="sheet",
            geometry_preset="depth",
            activation_checkpointing=True,
        )
        depth_fields = run_thog2_owt._optimisation_fields(depth_config, trainer)
        self.assertNotIn("semantic_qkv_bypass=True", depth_fields)

        non_depth_config = SimpleNamespace(
            model_type="sheet",
            geometry_preset="jpeg_like_v1",
            activation_checkpointing=True,
        )
        non_depth_fields = run_thog2_owt._optimisation_fields(non_depth_config, trainer)
        self.assertIn("semantic_qkv_bypass=True", non_depth_fields)
    # ^^^ THOG


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG