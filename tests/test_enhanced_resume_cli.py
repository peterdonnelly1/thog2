# vvv THOG
from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from run_thog2_owt import _configure_instrumentation_environment, _prepare_context, build_parser, explicit_destinations, inherited_config
from sheet.checkpoint_resolver import ResolvedCheckpoint
from sheet.checkpoints import load_payload
from tests.enhanced_resume_test_support import make_run_config, write_checkpoint_stub


class EnhancedResumeCliTests(unittest.TestCase):
    def parse(self, values):
        parser = build_parser()
        return parser, parser.parse_args(values), explicit_destinations(parser, values)

    def test_help_describes_n_as_total_optimizer_steps_not_additional_steps(self) -> None:
        help_text = " ".join(build_parser().format_help().split())
        self.assertIn("total optimizer steps for the run (not additional steps)", help_text)

    def test_explicit_destination_detection_distinguishes_omitted_defaults(self) -> None:
        parser, arguments, explicit = self.parse(["--run-mode", "resume", "--resume-from", "260715-1200", "--max-iters", "20"])
        self.assertEqual(arguments.batch_size, 12)
        self.assertNotIn("batch_size", explicit)
        self.assertIn("max_iters", explicit)
        self.assertIn("resume_from", explicit)

    def test_fresh_rejects_resume_from(self) -> None:
        _, arguments, explicit = self.parse(["--model-type", "dense", "--resume-from", "x"])
        with self.assertRaisesRegex(ValueError, "forbids --resume-from"):
            _prepare_context(arguments, explicit, 1)

    def test_fresh_rejects_fork_only_options(self) -> None:
        _, arguments, explicit = self.parse(["--model-type", "dense", "--fork-lr-mode", "restart_cosine"])
        with self.assertRaisesRegex(ValueError, "forbids fork-only"):
            _prepare_context(arguments, explicit, 1)

    def test_resume_without_n_prints_required_info_before_checkpoint_resolution(self) -> None:
        _, arguments, explicit = self.parse(["--run-mode", "resume", "--resume-from", "missing"])
        output = io.StringIO()
        with patch("run_thog2_owt.resolve_checkpoint") as resolver, redirect_stderr(output), self.assertRaises(SystemExit):
            _prepare_context(arguments, explicit, 1)
        resolver.assert_not_called()
        self.assertIn("THOG2 INFO: -n was omitted during resume!", output.getvalue())
        self.assertIn("greater than the number of steps already completed", output.getvalue())

    def test_resume_n_less_than_completed_prints_total_step_explanation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, _, _ = write_checkpoint_stub(root, completed_updates=4)
            _, arguments, explicit = self.parse(["--run-mode", "resume", "--resume-from", str(checkpoint), "--max-iters", "3", "--checkpoint-root", str(root / "checkpoints")])
            output = io.StringIO()
            with redirect_stderr(output), self.assertRaises(SystemExit):
                _prepare_context(arguments, explicit, 1)
            self.assertIn("-n is less than", output.getvalue())
            self.assertIn("total of all steps, not an additional amount", output.getvalue())

    def test_resume_n_equal_to_completed_uses_accurate_heading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, _, _ = write_checkpoint_stub(root, completed_updates=4)
            _, arguments, explicit = self.parse(["--run-mode", "resume", "--resume-from", str(checkpoint), "--max-iters", "4"])
            output = io.StringIO()
            with redirect_stderr(output), self.assertRaises(SystemExit):
                _prepare_context(arguments, explicit, 1)
            self.assertIn("-n is equal to", output.getvalue())

    def test_material_mismatch_is_rejected_but_equal_assertion_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = make_run_config(Path(directory))
            parser = build_parser()
            equal_args = parser.parse_args(["--batch-size", str(parent.batch_size)])
            equal_explicit = explicit_destinations(parser, ["--batch-size", str(parent.batch_size)])
            self.assertEqual(inherited_config(parent, equal_args, equal_explicit, run_mode="resume", max_iters=20, instrumentation_backend="none").batch_size, parent.batch_size)
            bad_args = parser.parse_args(["--batch-size", "2"])
            bad_explicit = explicit_destinations(parser, ["--batch-size", "2"])
            with self.assertRaisesRegex(ValueError, "material parameter mismatch"):
                inherited_config(parent, bad_args, bad_explicit, run_mode="resume", max_iters=20, instrumentation_backend="none")

    def test_operational_override_is_applied(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = make_run_config(Path(directory))
            parser = build_parser()
            argv = ["--eval-interval", "7", "--checkpoint-interval", "9", "--device", "cpu"]
            arguments = parser.parse_args(argv)
            config = inherited_config(parent, arguments, explicit_destinations(parser, argv), run_mode="resume", max_iters=20, instrumentation_backend="none")
            self.assertEqual(config.eval_interval, 7)
            self.assertEqual(config.checkpoint_interval, 9)
            self.assertEqual(config.device, "cpu")

    def test_dtype_and_nonfinite_controls_remain_material(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = make_run_config(Path(directory))
            parser = build_parser()
            for argv in (["--dtype", "bfloat16"], ["--nonfinite-update-policy", "raise"], ["--max-nonfinite-update-skips", "99"]):
                arguments = parser.parse_args(argv)
                with self.assertRaisesRegex(ValueError, "material parameter mismatch"):
                    inherited_config(parent, arguments, explicit_destinations(parser, argv), run_mode="resume", max_iters=20, instrumentation_backend="none")

    def test_resume_rejects_fork_only_options(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, _, _ = write_checkpoint_stub(root)
            _, arguments, explicit = self.parse(["--run-mode", "resume", "--resume-from", str(checkpoint), "--max-iters", "11", "--fork-lr-mode", "restart_cosine"])
            with self.assertRaisesRegex(ValueError, "rejects fork-only"):
                _prepare_context(arguments, explicit, 1)

    def test_fork_requires_restart_cosine_in_initial_implementation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, _, _ = write_checkpoint_stub(root)
            _, arguments, explicit = self.parse(["--run-mode", "fork", "--resume-from", str(checkpoint), "--max-iters", "12", "--fork-lr-mode", "continue"])
            with self.assertRaisesRegex(ValueError, "requires --fork-lr-mode restart_cosine"):
                _prepare_context(arguments, explicit, 1)

    def test_fork_requires_peak_minimum_and_rewarm_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, _, _ = write_checkpoint_stub(root)
            _, arguments, explicit = self.parse(["--run-mode", "fork", "--resume-from", str(checkpoint), "--max-iters", "12", "--fork-lr-mode", "restart_cosine"])
            with self.assertRaisesRegex(ValueError, "requires"):
                _prepare_context(arguments, explicit, 1)

    def test_resume_world_size_must_match_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, _, _ = write_checkpoint_stub(root)
            _, arguments, explicit = self.parse(["--run-mode", "resume", "--resume-from", str(checkpoint), "--max-iters", "12"])
            with self.assertRaisesRegex(ValueError, "world size mismatch"):
                _prepare_context(arguments, explicit, 2)

    def test_resume_rejects_new_start_label_and_manual_artifact_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, _, _ = write_checkpoint_stub(root)
            for option, message in ((["--run-start-label", "260715-1300"], "forbids --run-start-label"), (["--artifact-suffix", "manual"], "forbids --artifact-suffix")):
                argv = ["--run-mode", "resume", "--resume-from", str(checkpoint), "--max-iters", "11", *option]
                _, arguments, explicit = self.parse(argv)
                with self.assertRaisesRegex(ValueError, message):
                    _prepare_context(arguments, explicit, 1)

    def test_fork_rejects_manual_artifact_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, _, _ = write_checkpoint_stub(root)
            argv = ["--run-mode", "fork", "--resume-from", str(checkpoint), "--max-iters", "12", "--artifact-suffix", "manual"]
            _, arguments, explicit = self.parse(argv)
            with self.assertRaisesRegex(ValueError, "forbids --artifact-suffix"):
                _prepare_context(arguments, explicit, 1)

    def test_wandb_resume_reuses_saved_id_or_warns_without_changing_training_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_run_config(Path(directory))
            config = __import__("dataclasses").replace(config, instrumentation_backend="wandb", wandb_enabled=True)
            with patch.dict(os.environ, {}, clear=False):
                _configure_instrumentation_environment(config, {"tensorboard_dir": "curves/test", "wandb_run_id": "abc123"}, "resume")
                self.assertEqual(os.environ["WANDB_RUN_ID"], "abc123")
                self.assertEqual(os.environ["WANDB_RESUME"], "allow")
            output = io.StringIO()
            with patch.dict(os.environ, {"WANDB_RUN_ID": "stale", "WANDB_RESUME": "must"}, clear=False), redirect_stdout(output):
                _configure_instrumentation_environment(config, {"tensorboard_dir": "curves/test", "wandb_run_id": None}, "resume")
                self.assertNotIn("WANDB_RUN_ID", os.environ)
                self.assertNotIn("WANDB_RESUME", os.environ)
            self.assertIn("has no W&B run ID", output.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
