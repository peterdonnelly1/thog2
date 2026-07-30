# vvv THOG
from __future__ import annotations

import unittest
from pathlib import Path

from sheet.lr_schedule import COSINE_SCHEDULE, RESTART_COSINE_SCHEDULE
from sheet.run_config import OwtRunConfig
from sheet.run_lifecycle import fork_lifecycle, fresh_lifecycle, resume_lifecycle, update_wandb_identity


class ResumeAndForkWandbLineageTests(unittest.TestCase):
    @staticmethod
    def _parent() -> tuple[OwtRunConfig, dict, dict]:
        config = OwtRunConfig(
            model_type="dense",
            run_start_label="260729-0500",
            max_iters=100,
            residual_init_depth_source="true_layer_depth",
            device="cpu",
            dtype="float32",
        )
        paths = config.paths()
        paths["tensorboard_dir"] = Path("curves") / config.artifact_name
        paths["manifest_path"] = paths["checkpoint_dir"] / "run_manifest.json"
        lifecycle = fresh_lifecycle(
            config=config,
            artifact_name=config.artifact_name,
            paths=paths,
            world_size=1,
            instrumentation_backend="wandb",
            optimizer_name="adamw",
            optimizer_momentum=0.9,
            lr_phase={
                "phase_type": COSINE_SCHEDULE,
                "phase_start_update": 0,
                "phase_end_update": 100,
                "phase_peak_lr": config.learning_rate,
                "phase_min_lr": config.min_lr,
                "phase_warmup_iters": config.warmup_iters,
            },
        )
        lifecycle = update_wandb_identity(lifecycle, "parent123")
        return config, paths, lifecycle

    def test_resume_opt_out_records_previous_wandb_id_when_new_id_arrives(self) -> None:
        config, _, parent = self._parent()
        resumed = resume_lifecycle(
            parent,
            config=config,
            starting_completed_updates=25,
            target_updates=100,
            instrumentation_backend="wandb",
            wandb_continue_run=False,
        )
        self.assertEqual(resumed["wandb_run_id"], "parent123")
        resumed = update_wandb_identity(resumed, "resume456")
        self.assertEqual(resumed["wandb_run_id"], "resume456")
        self.assertIn("parent123", resumed["wandb_run_history"])

    def test_fork_default_new_wandb_run_retains_parent_telemetry_ancestry(self) -> None:
        config, paths, parent = self._parent()
        child_config = OwtRunConfig(
            model_type="dense",
            run_mode="resume",
            run_start_label="260729-0510",
            max_iters=200,
            residual_init_depth_source="true_layer_depth",
            device="cpu",
            dtype="float32",
        )
        child_paths = child_config.paths()
        child_paths["tensorboard_dir"] = Path("curves") / child_config.artifact_name
        child_paths["manifest_path"] = child_paths["checkpoint_dir"] / "run_manifest.json"
        child = fork_lifecycle(
            parent,
            config=child_config,
            artifact_name=child_config.artifact_name,
            paths=child_paths,
            parent_checkpoint=paths["checkpoint_path"],
            parent_completed_updates=25,
            target_updates=200,
            world_size=1,
            instrumentation_backend="wandb",
            wandb_continue_run=False,
            child_lr_phase={
                "phase_type": RESTART_COSINE_SCHEDULE,
                "phase_start_update": 25,
                "phase_start_lr": 1.0e-4,
                "phase_peak_lr": 5.0e-4,
                "phase_rewarm_iters": 10,
                "phase_end_update": 200,
                "phase_min_lr": 5.0e-5,
            },
        )
        self.assertIsNone(child["wandb_run_id"])
        self.assertEqual(child["parent_wandb_run_id"], "parent123")
        self.assertIn("parent123", child["wandb_run_history"])
        self.assertEqual(child["lineage"][-1]["wandb_run_id"], "parent123")


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
