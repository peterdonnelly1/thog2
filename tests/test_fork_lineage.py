# vvv THOG
from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from sheet.run_lifecycle import fork_lifecycle, fork_suffix, resume_lifecycle
from sheet.run_manifest import load_run_manifest, manifest_path, write_run_manifest
from tests.enhanced_resume_test_support import make_lifecycle, make_run_config


class ForkLineageTests(unittest.TestCase):
    def test_resume_reuses_logical_run_artifact_and_root_but_creates_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_run_config(root)
            parent = make_lifecycle(config)
            resumed = resume_lifecycle(parent, config=replace(config, run_mode="resume", max_iters=20), starting_completed_updates=10, instrumentation_backend="none", execution_options=parent["execution_options"])
            self.assertEqual(resumed["logical_run_id"], parent["logical_run_id"])
            self.assertEqual(resumed["artifact_name"], parent["artifact_name"])
            self.assertEqual(resumed["root_start_label"], parent["root_start_label"])
            self.assertNotEqual(resumed["session_id"], parent["session_id"])
            self.assertEqual(resumed["sessions"][-1]["starting_completed_updates"], 10)

    def test_first_fork_uses_generation_one_and_root_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent_config = make_run_config(root)
            parent = make_lifecycle(parent_config)
            suffix = fork_suffix(1, parent["root_start_label"])
            child_config = replace(parent_config, run_mode="fork", run_start_label="260715-1201", artifact_suffix=suffix, learning_rate=5.0e-4)
            paths = child_config.paths(); paths["tensorboard_dir"] = root / "tb" / child_config.artifact_name
            child = fork_lifecycle(parent, config=child_config, paths=paths, parent_checkpoint=parent_config.paths()["checkpoint_path"], parent_completed_updates=10, world_size=1, instrumentation_backend="none", execution_options=parent["execution_options"], child_lr_phase={"phase_type": "restart_cosine"})
            self.assertEqual(child["fork_generation"], 1)
            self.assertEqual(child["root_start_label"], "260715-1200")
            self.assertTrue(child["artifact_name"].endswith("__FORK_1_FROM_260715-1200"))
            self.assertEqual(child["parent_logical_run_id"], parent["logical_run_id"])

    def test_fork_of_fork_increments_generation_and_retains_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_config = make_run_config(root)
            base = make_lifecycle(base_config)
            first_config = replace(base_config, run_mode="fork", run_start_label="260715-1201", artifact_suffix=fork_suffix(1, "260715-1200"))
            first_paths = first_config.paths(); first_paths["tensorboard_dir"] = root / "tb1"
            first = fork_lifecycle(base, config=first_config, paths=first_paths, parent_checkpoint=base_config.paths()["checkpoint_path"], parent_completed_updates=10, world_size=1, instrumentation_backend="none", execution_options=base["execution_options"], child_lr_phase={"phase_type": "restart_cosine"})
            second_config = replace(first_config, run_start_label="260715-1202", artifact_suffix=fork_suffix(2, "260715-1200"))
            second_paths = second_config.paths(); second_paths["tensorboard_dir"] = root / "tb2"
            second = fork_lifecycle(first, config=second_config, paths=second_paths, parent_checkpoint=first_config.paths()["checkpoint_path"], parent_completed_updates=20, world_size=1, instrumentation_backend="none", execution_options=first["execution_options"], child_lr_phase={"phase_type": "restart_cosine"})
            self.assertEqual(second["fork_generation"], 2)
            self.assertEqual(second["root_start_label"], "260715-1200")
            self.assertTrue(second["artifact_name"].endswith("__FORK_2_FROM_260715-1200"))
            self.assertNotIn("FORK_1", second["artifact_name"])
            self.assertEqual(len(second["lineage"]), 2)

    def test_manifest_round_trip_preserves_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lifecycle = make_lifecycle(make_run_config(root))
            path = manifest_path(root)
            write_run_manifest(path, lifecycle)
            self.assertEqual(load_run_manifest(path), lifecycle)
            self.assertFalse(any(root.glob(".*.tmp")))

    def test_fork_suffix_rejects_zero_generation(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            fork_suffix(0, "260715-1200")


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
