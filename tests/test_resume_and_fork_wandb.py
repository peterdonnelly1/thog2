# vvv THOG
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from run_thog2_lifecycle import _configure_instrumentation_environment, _inherited_instrumentation_configuration, _local_run_configuration, _wandb_continue_policy, build_parser
from sheet.run_config import OwtRunConfig


class ResumeAndForkWandbTests(unittest.TestCase):
    def setUp(self) -> None:
        self.saved = {
            name: os.environ.get(name)
            for name in ("THOG2_INSTRUMENTATION", "THOG2_CURVE_ROOT", "WANDB_MODE", "WANDB_RUN_ID", "WANDB_RESUME", "THOG2_INSTRUMENTATION_DEPTH_WEIGHT_CURVES_LOG_EVERY_N_STEPS", "THOG2_DEPTH_CURVE_LOCAL_ROOT", "THOG2_INSTRUMENTATION_LOCAL_ROOT")
        }

    def tearDown(self) -> None:
        for name, value in self.saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    @staticmethod
    def config(*, wandb_mode: str = "online") -> OwtRunConfig:
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
            wandb_mode=wandb_mode,
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

    def test_offline_resume_continuation_reuses_id_and_preserves_offline_mode(self) -> None:
        context = {
            "backend": "wandb",
            "mode": "resume",
            "config": self.config(wandb_mode="offline"),
            "wandb_continue_run": True,
            "lifecycle": {
                "wandb_run_id": "offline123",
                "tensorboard_dir": str(Path("curves") / "run"),
            },
        }
        _configure_instrumentation_environment(context)
        self.assertEqual(os.environ["WANDB_RUN_ID"], "offline123")
        self.assertEqual(os.environ["WANDB_RESUME"], "must")
        self.assertEqual(os.environ["WANDB_MODE"], "offline")

    def test_combined_backend_uses_resume_default_continuation(self) -> None:
        parser = build_parser()
        arguments = parser.parse_args(["--resume", "260729-0100"])
        self.assertTrue(_wandb_continue_policy(arguments, "resume", "both"))
        self.assertFalse(_wandb_continue_policy(arguments, "fork", "both"))

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

    # vvv THOG recorded curve instrumentation overrides parser/wrapper defaults on resume and fork
    def test_recorded_instrumentation_configuration_is_restored(self) -> None:
        os.environ["THOG2_INSTRUMENTATION_DEPTH_WEIGHT_CURVES_LOG_EVERY_N_STEPS"] = "999"
        os.environ["THOG2_DEPTH_CURVE_LOCAL_ROOT"] = "current-child-artifact/depth_curves"
        context = {
            "backend": "tensorboard",
            "mode": "resume",
            "config": self.config(),
            "wandb_continue_run": False,
            "lifecycle": {
                "wandb_run_id": None,
                "tensorboard_dir": str(Path("curves") / "run"),
                "instrumentation_configuration": {
                    "THOG2_INSTRUMENTATION_DEPTH_WEIGHT_CURVES_LOG_EVERY_N_STEPS": "17",
                    "THOG2_DEPTH_CURVE_LOCAL_ROOT": "parent-artifact/depth_curves",
                },
            },
        }

        _configure_instrumentation_environment(context)

        self.assertEqual(
            os.environ["THOG2_INSTRUMENTATION_DEPTH_WEIGHT_CURVES_LOG_EVERY_N_STEPS"],
            "17",
        )
        self.assertEqual(
            os.environ["THOG2_DEPTH_CURVE_LOCAL_ROOT"],
            "current-child-artifact/depth_curves",
        )                                                                                                                                                # <<< THOG fork output paths remain owned by the new artifact
    # ^^^ THOG

    # vvv THOG pre-enhancement checkpoints recover canonical weight-capture settings from the local W&B-linked run record
    def test_local_run_configuration_backfills_instrumentation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            os.environ["THOG2_INSTRUMENTATION_LOCAL_ROOT"] = directory
            database = Path(directory) / "artifact" / "wandb123" / "charts.sqlite3"
            database.parent.mkdir(parents=True)
            connection = sqlite3.connect(database)
            connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute(
                "INSERT INTO metadata(key, value) VALUES ('config_json', ?)",
                ('{"instrumentation__depth_weight_curves__log_every_n_steps":17}',),
            )
            connection.commit()
            connection.close()

            recovered = _inherited_instrumentation_configuration(
                {},
                fallback_configuration=_local_run_configuration("artifact"),
            )

        self.assertEqual(
            recovered["THOG2_INSTRUMENTATION_DEPTH_WEIGHT_CURVES_LOG_EVERY_N_STEPS"],
            "17",
        )
    # ^^^ THOG


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
