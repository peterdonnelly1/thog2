# vvv THOG
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from run_thog2_owt import main
from sheet.checkpoints import load_payload
from tests.enhanced_resume_test_support import write_tiny_dataset


class EnhancedResumeIntegrationTests(unittest.TestCase):
    def test_cpu_fresh_resume_and_fork_preserve_identity_and_create_truthful_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            checkpoints = root / "checkpoints"
            logs = root / "logs"
            results = root / "results"
            wandb = root / "wandb"
            write_tiny_dataset(data)
            common = [
                "--data-dir", str(data), "--checkpoint-root", str(checkpoints), "--log-root", str(logs),
                "--result-root", str(results), "--wandb-root", str(wandb), "--instrumentation", "none",
                "--no-wandb", "--device", "cpu", "--dtype", "float32", "--eval-interval", "0",
                "--eval-iters", "1", "--log-interval", "1", "--checkpoint-interval", "0",
            ]
            fresh = [
                "--model-type", "dense", "--run-mode", "fresh", "--run-start-label", "260715-1200",
                "--run-name", "TEST", "--experiment-prefix", "TEST", "--max-iters", "2",
                "--batch-size", "1", "--gradient-accumulation-steps", "1", "--block-size", "8",
                "--n-layer", "1", "--n-head", "1", "--n-embd", "8", "--warmup-iters", "0",
                "--learning-rate", "0.001", "--min-lr", "0.0001",
                "--residual-init-depth-source", "true_layer_depth", "--no-activation-checkpointing",
                *common,
            ]
            self.assertEqual(main(fresh), 0)
            original_dirs = list(checkpoints.iterdir())
            self.assertEqual(len(original_dirs), 1)
            original_dir = original_dirs[0]
            original_checkpoint = original_dir / "ckpt.pt"
            original_payload = load_payload(original_checkpoint)
            original_lifecycle = original_payload["lifecycle"]
            self.assertEqual(original_payload["completed_updates"], 2)

            resume = ["--run-mode", "resume", "--resume-from", "260715-1200", "--max-iters", "3", "--checkpoint-root", str(checkpoints), "--instrumentation", "none", "--eval-interval", "0"]
            self.assertEqual(main(resume), 0)
            resumed_payload = load_payload(original_checkpoint)
            resumed_lifecycle = resumed_payload["lifecycle"]
            self.assertEqual(resumed_payload["completed_updates"], 3)
            self.assertEqual(resumed_lifecycle["logical_run_id"], original_lifecycle["logical_run_id"])
            self.assertEqual(resumed_lifecycle["artifact_name"], original_lifecycle["artifact_name"])
            self.assertNotEqual(resumed_lifecycle["session_id"], original_lifecycle["session_id"])
            self.assertEqual(len(list(checkpoints.iterdir())), 1)

            fork = [
                "--run-mode", "fork", "--resume-from", "260715-1200", "--max-iters", "5",
                "--checkpoint-root", str(checkpoints), "--run-start-label", "260715-1201",
                "--fork-lr-mode", "restart_cosine", "--fork-learning-rate", "0.0005",
                "--fork-min-lr", "0.00005", "--fork-rewarm-iters", "1",
                "--instrumentation", "none", "--eval-interval", "0",
            ]
            self.assertEqual(main(fork), 0)
            dirs = sorted(checkpoints.iterdir())
            self.assertEqual(len(dirs), 2)
            child_dir = next(path for path in dirs if path != original_dir)
            self.assertTrue(child_dir.name.endswith("__FORK_1_FROM_260715-1200"))
            child_payload = load_payload(child_dir / "ckpt.pt")
            child_lifecycle = child_payload["lifecycle"]
            self.assertEqual(child_payload["completed_updates"], 5)
            self.assertEqual(child_lifecycle["fork_generation"], 1)
            self.assertEqual(child_lifecycle["root_start_label"], "260715-1200")
            self.assertEqual(child_lifecycle["parent_logical_run_id"], resumed_lifecycle["logical_run_id"])
            self.assertEqual(child_lifecycle["parent_completed_updates"], 3)
            self.assertEqual(child_lifecycle["active_lr_phase_index"], 1)
            self.assertEqual(child_payload["trainer_config"]["phase_end_update"], 5)
            self.assertEqual(child_payload["trainer_config"]["phase_rewarm_iters"], 1)
            self.assertTrue((child_dir / "run_manifest.json").is_file())
            aggregate_result = Path(child_lifecycle["result_path"])
            result = json.loads(aggregate_result.read_text())
            self.assertEqual(result["budget"]["session_completed_updates"], 2)
            self.assertEqual(result["budget"]["completed_updates"], 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
