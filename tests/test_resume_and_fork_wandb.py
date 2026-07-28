# vvv THOG
from __future__ import annotations

import os
import unittest
from pathlib import Path

from run_thog2_lifecycle import _configure_instrumentation_environment
from sheet.run_config import OwtRunConfig


class ResumeAndForkWandbTests(unittest.TestCase):
    def setUp(self) -> None:
        self.saved = {
            name: os.environ.get(name)
            for name in ("THOG2_INSTRUMENTATION", "THOG2_CURVE_ROOT", "WANDB_MODE", "WANDB_RUN_ID", "WANDB_RESUME")
        }

    def tearDown(self) -> None:
        for name, value in self.saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    @staticmethod
    def config() -> OwtRunConfig:
        return OwtRunConfig(
            model_type="dense",
            run_mode="resume",
            run_start_label="260729-0100",
            max_iters=100,
            warmup_iters=10,
            residual_init_depth_source="true_layer_depth",
            device="cpu",
            dtype="float32",
            wandb_enabled=True,
            wandb_mode="online",
        )

    def test_resume_continuation_sets_strict_same_id_environment(self) -> None:
        context = {
            "backend": "wandb",
            "mode": "resume",
            "config": self.config(),
            "wandb_continue_run": True,
            "lifecycle": {
                "wandb_run_id": "abc123",
                "tensorboard_dir": str(Path("curves") / "run"),
            },
        }
        _configure_instrumentation_environment(context)
        self.assertEqual(os.environ["WANDB_RUN_ID"], "abc123")
        self.assertEqual(os.environ["WANDB_RESUME"], "must")
        self.assertEqual(os.environ["WANDB_MODE"], "online")

    def test_resume_continuation_without_id_fails_before_training(self) -> None:
        context = {
            "backend": "wandb",
            "mode": "resume",
            "config": self.config(),
            "wandb_continue_run": True,
            "lifecycle": {
                "wandb_run_id": None,
                "tensorboard_dir": str(Path("curves") / "run"),
            },
        }
        with self.assertRaisesRegex(ValueError, "--no-wandb-continue-run"):
            _configure_instrumentation_environment(context)

    def test_resume_opt_out_clears_same_id_environment(self) -> None:
        os.environ["WANDB_RUN_ID"] = "stale"
        os.environ["WANDB_RESUME"] = "must"
        context = {
            "backend": "wandb",
            "mode": "resume",
            "config": self.config(),
            "wandb_continue_run": False,
            "lifecycle": {
                "wandb_run_id": "abc123",
                "tensorboard_dir": str(Path("curves") / "run"),
            },
        }
        _configure_instrumentation_environment(context)
        self.assertNotIn("WANDB_RUN_ID", os.environ)
        self.assertNotIn("WANDB_RESUME", os.environ)

    def test_fork_default_new_run_clears_parent_same_id_environment(self) -> None:
        os.environ["WANDB_RUN_ID"] = "parent"
        os.environ["WANDB_RESUME"] = "must"
        context = {
            "backend": "both",
            "mode": "fork",
            "config": self.config(),
            "wandb_continue_run": False,
            "lifecycle": {
                "wandb_run_id": None,
                "tensorboard_dir": str(Path("curves") / "child"),
            },
        }
        _configure_instrumentation_environment(context)
        self.assertNotIn("WANDB_RUN_ID", os.environ)
        self.assertNotIn("WANDB_RESUME", os.environ)
        self.assertEqual(os.environ["THOG2_INSTRUMENTATION"], "both")


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
