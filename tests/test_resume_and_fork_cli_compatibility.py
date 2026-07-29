# vvv THOG
from __future__ import annotations

import contextlib
import io
import os
import re
import subprocess
import unittest
from pathlib import Path
from unittest import mock

import run_thog2_owt
import sheet.owt_lifecycle_cli as lifecycle_cli
from sheet.owt_lifecycle_cli import normalize_lifecycle_wrapper_argv
from sheet.wandb_telemetry import _evaluation_metrics, _final_metrics, _training_metrics


class ResumeAndForkCliCompatibilityTests(unittest.TestCase):
    def test_lifecycle_short_option_coverage_matches_preserved_master_wrapper(self) -> None:
        wrapper = (Path(__file__).resolve().parents[1] / "train_OWT_core.sh").read_text(encoding="utf-8")
        match = re.search(r'while getopts "(?P<spec>:[^"]+)" option; do', wrapper)
        self.assertIsNotNone(match)
        spec = match.group("spec")[1:]
        master_options = {character for character in spec if character != ":"}
        self.assertEqual(master_options, lifecycle_cli._SHORT_VALUE_OPTIONS | {"h"})

    def test_established_eval_and_checkpoint_short_options_reach_lifecycle_parser(self) -> None:
        normalized = normalize_lifecycle_wrapper_argv(
            ["--resume", "260727-1934", "-e", "100", "-u50", "-k", "500"]
        )
        self.assertEqual(
            normalized.argv,
            (
                "--resume", "260727-1934",
                "--eval-interval", "100",
                "--eval-iters", "50",
                "--checkpoint-interval", "500",
            ),
        )

    def test_established_model_short_options_are_assertions_not_wrapper_defaults(self) -> None:
        normalized = normalize_lifecycle_wrapper_argv(
            ["--resume", "run", "-p", "depth", "-Bcheby", "-L", "32", "-D1024", "-P16"]
        )
        self.assertEqual(
            normalized.argv,
            (
                "--resume", "run",
                "--model-type", "sheet",
                "--geometry-preset", "depth",
                "--basis-family", "chebyshev",
                "--n-layer", "32",
                "--n-embd", "1024",
                "--o-depth", "16",
            ),
        )
        self.assertNotIn("--batch-size", normalized.argv)
        self.assertNotIn("--learning-rate", normalized.argv)

    def test_wrapper_learning_rate_codes_keep_existing_meaning(self) -> None:
        normalized = normalize_lifecycle_wrapper_argv(["--resume", "run", "-c90", "-f", "9"])
        self.assertEqual(
            normalized.argv,
            ("--resume", "run", "--learning-rate", "90e-5", "--min-lr", "9e-5"),
        )

    def test_run_name_keeps_existing_experiment_prefix_semantics(self) -> None:
        normalized = normalize_lifecycle_wrapper_argv(["--fork", "run", "-g", "CHILD"])
        self.assertEqual(
            normalized.argv,
            ("--fork", "run", "--run-name", "CHILD", "--experiment-prefix", "CHILD"),
        )

    def test_dataset_short_option_keeps_existing_data_directory_side_effect(self) -> None:
        normalized = normalize_lifecycle_wrapper_argv(["--resume", "run", "-d", "owt"])
        self.assertEqual(
            normalized.argv,
            ("--resume", "run", "--dataset", "owt", "--data-dir", "data/owt"),
        )

    def test_wrapper_only_runtime_options_become_process_environment(self) -> None:
        normalized = normalize_lifecycle_wrapper_argv(
            ["--resume", "run", "-F", "eval", "-N16384", "-U", "plotly", "-Vtrue", "-E", "false"]
        )
        self.assertEqual(normalized.argv, ("--resume", "run"))
        self.assertEqual(
            normalized.environment,
            {
                "THOG2_DEPTH_CURVE_PLOTS": "eval",
                "THOG2_DEPTH_CURVE_SAMPLE_ELEMENTS": "16384",
                "THOG2_DEPTH_CURVE_RENDERER": "plotly",
                "THOG2_DEPTH_CURVE_LOCAL_HTML": "true",
                "THOG2_FAST_DISCARD": "false",
            },
        )

    def test_grid_values_fail_clearly_in_single_checkpoint_lifecycle_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "grid values are fresh-run only"):
            normalize_lifecycle_wrapper_argv(["--resume", "run", "-b", "16,32"])

    def test_public_entry_normalizes_before_calling_lifecycle(self) -> None:
        with mock.patch.object(run_thog2_owt, "_call_lifecycle_main", return_value=17) as call:
            status = run_thog2_owt.main(["--resume", "run", "-e100", "-u", "50", "-k500"])
        self.assertEqual(status, 17)
        forwarded_argv, forwarded_environment = call.call_args.args
        self.assertEqual(
            forwarded_argv,
            ("--resume", "run", "--eval-interval", "100", "--eval-iters", "50", "--checkpoint-interval", "500"),
        )
        self.assertEqual(forwarded_environment, {})

    def test_lifecycle_environment_is_restored_after_programmatic_call(self) -> None:
        original = os.environ.get("THOG2_DEPTH_CURVE_PLOTS")
        try:
            os.environ["THOG2_DEPTH_CURVE_PLOTS"] = "none"
            with mock.patch.object(run_thog2_owt, "_lifecycle_main", return_value=0):
                status = run_thog2_owt._call_lifecycle_main(
                    ("--resume", "run"),
                    {"THOG2_DEPTH_CURVE_PLOTS": "eval"},
                )
            self.assertEqual(status, 0)
            self.assertEqual(os.environ["THOG2_DEPTH_CURVE_PLOTS"], "none")
        finally:
            if original is None:
                os.environ.pop("THOG2_DEPTH_CURVE_PLOTS", None)
            else:
                os.environ["THOG2_DEPTH_CURVE_PLOTS"] = original

    def test_schedule_console_values_start_in_model_options_value_column(self) -> None:
        lifecycle = {
            "target_updates": 50000,
            "lr_phases": [{"phase_type": "cosine"}],
            "active_lr_phase_index": 0,
        }
        training_config = type("TrainingConfigStub", (), {"decay_updates": 10000})()
        stream = io.StringIO()
        with mock.patch.object(run_thog2_owt._lifecycle, "learning_rate_for_lifecycle", return_value=9.0e-5):
            with contextlib.redirect_stdout(stream):
                run_thog2_owt._print_lifecycle_schedule("resume", lifecycle, training_config, 13000)
        lines = stream.getvalue().splitlines()
        self.assertEqual(lines[0], "")
        self.assertEqual(lines[1], "resume schedule")
        for line in lines[2:9]:
            if not line:
                continue
            value = line[27:]
            self.assertTrue(value, line)
            self.assertEqual(line[:2], "  ")

    def test_lifecycle_shell_remains_syntactically_valid(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        subprocess.run(
            ["bash", "-n", str(repository_root / "resume_and_fork_OWT.sh")],
            check=True,
            cwd=repository_root,
        )


class CanonicalTelemetryMetricNameTests(unittest.TestCase):
    def test_training_loss_has_one_canonical_label(self) -> None:
        metrics = _training_metrics(
            {
                "completed_updates": 10,
                "consumed_tokens": 1000,
                "cumulative_training_seconds": 5.0,
                "training_loss": 3.25,
                "learning_rate": 9.0e-5,
                "gradient_norm": 0.2,
            }
        )
        self.assertEqual(metrics["train/loss"], 3.25)
        self.assertNotIn("train/step_loss", metrics)

    def test_validation_pass_uses_train_and_validation_split_labels(self) -> None:
        metrics = _evaluation_metrics(
            {
                "completed_updates": 100,
                "consumed_tokens": 10000,
                "training_loss": 3.10,
                "validation_loss": 3.30,
            }
        )
        self.assertEqual(metrics["val/train_loss"], 3.10)
        self.assertEqual(metrics["val/val_loss"], 3.30)
        self.assertNotIn("eval/loss", metrics)
        self.assertNotIn("eval/val_loss", metrics)
        self.assertNotIn("test/loss", metrics)

    def test_final_validation_metrics_do_not_use_test_namespace(self) -> None:
        metrics = _final_metrics(
            {
                "budget": {"completed_updates": 200, "consumed_tokens": 20000},
                "timing": {"training_seconds": 10.0, "tokens_per_training_second": 2000.0},
                "parameter_report": {
                    "persistent_parameters": 10,
                    "dense_equivalent_total_parameters": 20,
                },
                "checkpoint": {"bytes": 1234},
                "evaluations": [{"val": 3.4}, {"val": 3.2}],
            }
        )
        self.assertEqual(metrics["val/final_loss"], 3.2)
        self.assertEqual(metrics["val/best_loss"], 3.2)
        self.assertFalse(any(name.startswith("test/") or name.startswith("eval/") for name in metrics))


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
