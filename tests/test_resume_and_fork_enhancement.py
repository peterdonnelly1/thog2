# vvv THOG
from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from run_thog2_lifecycle import (
    _target_updates_for_resume,
    _training_config_for_lifecycle,
    _wandb_continue_policy,
    build_parser,
    explicit_destinations,
)
from sheet.checkpoint_resolver import resolve_checkpoint
from sheet.lr_schedule import (
    COSINE_SCHEDULE,
    RESTART_COSINE_SCHEDULE,
    learning_rate_for_lifecycle,
    restart_cosine_learning_rate,
)
from sheet.run_config import OwtRunConfig
from sheet.run_lifecycle import fork_lifecycle, fresh_lifecycle, resume_lifecycle
from sheet.training_config import TrainingConfig


class ResumeAndForkEnhancementTests(unittest.TestCase):
    def test_timestamp_selector_is_exact_and_ambiguous_matches_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "260729-0100_first"
            first.mkdir()
            (first / "ckpt.pt").write_bytes(b"x")
            resolved = resolve_checkpoint("260729-0100", root)
            self.assertEqual(resolved.checkpoint_dir, first)

            second = root / "260729-0100_second"
            second.mkdir()
            (second / "ckpt.pt").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "ambiguous checkpoint timestamp"):
                resolve_checkpoint("260729-0100", root)

    def test_resume_selector_implies_resume_mode(self) -> None:
        parser = build_parser()
        argv = ["--resume", "260729-0100"]
        arguments = parser.parse_args(argv)
        explicit = explicit_destinations(parser, argv)
        from run_thog2_lifecycle import _resolve_mode_and_selector

        mode, selector = _resolve_mode_and_selector(arguments, explicit)
        self.assertEqual(mode, "resume")
        self.assertEqual(selector, "260729-0100")

    def test_resume_target_defaults_to_stored_target(self) -> None:
        parser = build_parser()
        argv = ["--resume", "260729-0100"]
        arguments = parser.parse_args(argv)
        explicit = explicit_destinations(parser, argv)
        target = _target_updates_for_resume(
            arguments,
            explicit,
            {"target_updates": 1000},
            400,
        )
        self.assertEqual(target, 1000)

    def test_resume_target_requires_explicit_extension_after_completion(self) -> None:
        parser = build_parser()
        argv = ["--resume", "260729-0100"]
        arguments = parser.parse_args(argv)
        explicit = explicit_destinations(parser, argv)
        with self.assertRaisesRegex(ValueError, "specify a larger -n"):
            _target_updates_for_resume(
                arguments,
                explicit,
                {"target_updates": 1000},
                1000,
            )

    def test_resume_extension_preserves_original_cosine_endpoint(self) -> None:
        checkpoint = TrainingConfig(
            max_updates=1000,
            learning_rate=1.0e-3,
            min_learning_rate=1.0e-4,
            warmup_updates=100,
            decay_updates=1000,
            device="cpu",
            dtype="float32",
        )
        config = OwtRunConfig(
            model_type="dense",
            run_mode="resume",
            max_iters=5000,
            warmup_iters=100,
            learning_rate=1.0e-3,
            min_lr=1.0e-4,
            residual_init_depth_source="true_layer_depth",
            device="cpu",
            dtype="float32",
        )
        resumed = _training_config_for_lifecycle(
            checkpoint,
            config,
            target_updates=5000,
            out_dir=Path("out"),
        )
        self.assertEqual(resumed.max_updates, 5000)
        self.assertEqual(resumed.decay_updates, 1000)
        self.assertEqual(resumed.warmup_updates, 100)

    def test_original_lr_phase_holds_minimum_after_original_decay(self) -> None:
        config = TrainingConfig(
            max_updates=5000,
            learning_rate=1.0e-3,
            min_learning_rate=1.0e-4,
            warmup_updates=100,
            decay_updates=1000,
            device="cpu",
            dtype="float32",
        )
        lifecycle = {
            "lr_phases": [
                {
                    "phase_type": COSINE_SCHEDULE,
                    "phase_start_update": 0,
                    "phase_end_update": 1000,
                    "phase_peak_lr": 1.0e-3,
                    "phase_min_lr": 1.0e-4,
                    "phase_warmup_iters": 100,
                }
            ],
            "active_lr_phase_index": 0,
        }
        self.assertAlmostEqual(learning_rate_for_lifecycle(config, lifecycle, 1001), 1.0e-4)
        self.assertAlmostEqual(learning_rate_for_lifecycle(config, lifecycle, 4500), 1.0e-4)

    def test_restart_cosine_rewarm_and_final_minimum(self) -> None:
        self.assertAlmostEqual(
            restart_cosine_learning_rate(
                completed_updates=100,
                phase_start_update=100,
                phase_start_lr=1.0e-4,
                phase_peak_lr=5.0e-4,
                phase_rewarm_iters=10,
                phase_end_update=200,
                phase_min_lr=5.0e-5,
            ),
            1.0e-4,
        )
        self.assertAlmostEqual(
            restart_cosine_learning_rate(
                completed_updates=110,
                phase_start_update=100,
                phase_start_lr=1.0e-4,
                phase_peak_lr=5.0e-4,
                phase_rewarm_iters=10,
                phase_end_update=200,
                phase_min_lr=5.0e-5,
            ),
            5.0e-4,
        )
        self.assertAlmostEqual(
            restart_cosine_learning_rate(
                completed_updates=199,
                phase_start_update=100,
                phase_start_lr=1.0e-4,
                phase_peak_lr=5.0e-4,
                phase_rewarm_iters=10,
                phase_end_update=200,
                phase_min_lr=5.0e-5,
            ),
            5.0e-5,
        )

    def test_wandb_continue_defaults_true_for_resume_false_for_fork(self) -> None:
        parser = build_parser()
        resume_arguments = parser.parse_args(["--resume", "260729-0100"])
        fork_arguments = parser.parse_args(["--fork", "260729-0100"])
        self.assertTrue(_wandb_continue_policy(resume_arguments, "resume", "wandb"))
        self.assertFalse(_wandb_continue_policy(fork_arguments, "fork", "wandb"))
        self.assertFalse(_wandb_continue_policy(resume_arguments, "resume", "tensorboard"))

    def test_wandb_continue_explicit_override_wins(self) -> None:
        parser = build_parser()
        resume_arguments = parser.parse_args(
            ["--resume", "260729-0100", "--no-wandb-continue-run"]
        )
        fork_arguments = parser.parse_args(
            ["--fork", "260729-0100", "--wandb-continue-run"]
        )
        self.assertFalse(_wandb_continue_policy(resume_arguments, "resume", "wandb"))
        self.assertTrue(_wandb_continue_policy(fork_arguments, "fork", "wandb"))

    def test_resume_preserves_logical_run_and_fork_creates_new_lineage(self) -> None:
        root = Path("/tmp/thog2-lifecycle-test")
        parent_config = OwtRunConfig(
            model_type="dense",
            run_start_label="260729-0100",
            max_iters=1000,
            warmup_iters=10,
            residual_init_depth_source="true_layer_depth",
            device="cpu",
            dtype="float32",
        )
        paths = parent_config.paths()
        paths["tensorboard_dir"] = root / "curves" / parent_config.artifact_name
        paths["manifest_path"] = paths["checkpoint_dir"] / "run_manifest.json"
        parent = fresh_lifecycle(
            config=parent_config,
            artifact_name=parent_config.artifact_name,
            paths=paths,
            world_size=1,
            instrumentation_backend="none",
            optimizer_name="adamw",
            optimizer_momentum=0.9,
            lr_phase={
                "phase_type": COSINE_SCHEDULE,
                "phase_start_update": 0,
                "phase_end_update": 1000,
                "phase_peak_lr": parent_config.learning_rate,
                "phase_min_lr": parent_config.min_lr,
                "phase_warmup_iters": parent_config.warmup_iters,
            },
        )
        resumed = resume_lifecycle(
            parent,
            config=parent_config,
            starting_completed_updates=400,
            target_updates=1000,
            instrumentation_backend="none",
            wandb_continue_run=False,
        )
        self.assertEqual(resumed["logical_run_id"], parent["logical_run_id"])
        self.assertNotEqual(resumed["session_id"], parent["session_id"])

        child_config = OwtRunConfig(
            model_type="dense",
            run_mode="resume",
            run_start_label="260729-0200",
            max_iters=2000,
            warmup_iters=10,
            residual_init_depth_source="true_layer_depth",
            device="cpu",
            dtype="float32",
        )
        child_paths = child_config.paths()
        child_paths["tensorboard_dir"] = root / "curves" / child_config.artifact_name
        child_paths["manifest_path"] = child_paths["checkpoint_dir"] / "run_manifest.json"
        child = fork_lifecycle(
            resumed,
            config=child_config,
            artifact_name=child_config.artifact_name,
            paths=child_paths,
            parent_checkpoint=paths["checkpoint_path"],
            parent_completed_updates=400,
            target_updates=2000,
            world_size=1,
            instrumentation_backend="none",
            wandb_continue_run=False,
            child_lr_phase={
                "phase_type": RESTART_COSINE_SCHEDULE,
                "phase_start_update": 400,
                "phase_start_lr": 1.0e-4,
                "phase_peak_lr": 5.0e-4,
                "phase_rewarm_iters": 10,
                "phase_end_update": 2000,
                "phase_min_lr": 5.0e-5,
            },
        )
        self.assertNotEqual(child["logical_run_id"], resumed["logical_run_id"])
        self.assertEqual(child["parent_logical_run_id"], resumed["logical_run_id"])
        self.assertEqual(child["root_run_id"], parent["logical_run_id"])
        self.assertEqual(child["fork_generation"], 1)
        self.assertEqual(child["active_lr_phase_index"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
