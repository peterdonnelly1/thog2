# vvv THOG
from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from run_thog2_owt import main
from sheet.checkpoints import load_payload, save_payload


def _write_tiny_dataset(path: Path, vocab_size: int = 32) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (np.arange(512, dtype=np.uint16) % vocab_size).tofile(path / "train.bin")
    (np.arange(256, dtype=np.uint16) % vocab_size).tofile(path / "val.bin")
    with (path / "meta.pkl").open("wb") as handle:
        pickle.dump({"vocab_size": vocab_size}, handle)


class ResumeAndForkCpuIntegrationTests(unittest.TestCase):
    def test_real_cpu_fresh_resume_no_brainer_resume_and_fork(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            checkpoints = root / "checkpoints"
            logs = root / "logs"
            results = root / "results"
            wandb = root / "wandb"
            curves = root / "curves"
            _write_tiny_dataset(data)

            common = [
                "--data-dir", str(data),
                "--checkpoint-root", str(checkpoints),
                "--log-root", str(logs),
                "--result-root", str(results),
                "--wandb-root", str(wandb),
                "--instrumentation", "none",
                "--no-wandb",
                "--device", "cpu",
                "--dtype", "float32",
                "--eval-interval", "1",
                "--eval-iters", "1",
                "--log-interval", "1",
                "--checkpoint-interval", "1",
            ]

            old_curve_root = __import__("os").environ.get("THOG2_CURVE_ROOT")
            __import__("os").environ["THOG2_CURVE_ROOT"] = str(curves)
            try:
                fresh = [
                    "--model-type", "dense",
                    "--run-mode", "fresh",
                    "--run-start-label", "260729-0100",
                    "--run-name", "TEST",
                    "--experiment-prefix", "TEST",
                    "--max-iters", "2",
                    "--batch-size", "1",
                    "--gradient-accumulation-steps", "1",
                    "--block-size", "8",
                    "--n-layer", "1",
                    "--n-head", "1",
                    "--n-embd", "8",
                    "--warmup-iters", "0",
                    "--learning-rate", "0.001",
                    "--min-lr", "0.0001",
                    "--residual-init-depth-source", "true_layer_depth",
                    "--no-activation-checkpointing",
                    *common,
                ]
                self.assertEqual(main(fresh), 0)

                original_dirs = [path for path in checkpoints.iterdir() if path.is_dir()]
                self.assertEqual(len(original_dirs), 1)
                original_dir = original_dirs[0]
                original_checkpoint = original_dir / "ckpt.pt"
                original_payload = load_payload(original_checkpoint)
                original_lifecycle = original_payload["lifecycle"]
                self.assertEqual(original_payload["completed_updates"], 2)
                self.assertEqual(original_lifecycle["target_updates"], 2)
                self.assertEqual(original_payload["trainer_config"]["decay_updates"], 2)

                resume = [
                    "--resume", "260729-0100",
                    "-n", "3",
                    "--checkpoint-root", str(checkpoints),
                    "--instrumentation", "none",
                ]
                self.assertEqual(main(resume), 0)
                resumed_payload = load_payload(original_checkpoint)
                resumed_lifecycle = resumed_payload["lifecycle"]
                self.assertEqual(resumed_payload["completed_updates"], 3)
                self.assertEqual(resumed_lifecycle["logical_run_id"], original_lifecycle["logical_run_id"])
                self.assertEqual(resumed_lifecycle["artifact_name"], original_lifecycle["artifact_name"])
                self.assertNotEqual(resumed_lifecycle["session_id"], original_lifecycle["session_id"])
                self.assertEqual(resumed_payload["trainer_config"]["max_updates"], 3)
                self.assertEqual(resumed_payload["trainer_config"]["decay_updates"], 2)

                # Simulate an interrupted session whose intended lifetime target was four.
                interrupted_payload = load_payload(original_checkpoint)
                interrupted_payload["trainer_config"]["max_updates"] = 4
                interrupted_payload["lifecycle"]["target_updates"] = 4
                interrupted_payload["lifecycle"]["run_config"]["max_iters"] = 4
                save_payload(interrupted_payload, original_checkpoint)

                no_brainer_resume = [
                    "--resume", "260729-0100",
                    "--checkpoint-root", str(checkpoints),
                    "--instrumentation", "none",
                ]
                self.assertEqual(main(no_brainer_resume), 0)
                no_brainer_payload = load_payload(original_checkpoint)
                self.assertEqual(no_brainer_payload["completed_updates"], 4)
                self.assertEqual(no_brainer_payload["trainer_config"]["max_updates"], 4)
                self.assertEqual(no_brainer_payload["trainer_config"]["decay_updates"], 2)
                self.assertEqual(no_brainer_payload["lifecycle"]["target_updates"], 4)

                fork = [
                    "--fork", "260729-0100",
                    "-n", "6",
                    "--checkpoint-root", str(checkpoints),
                    "--run-start-label", "260729-0200",
                    "--fork-lr-mode", "restart_cosine",
                    "--fork-learning-rate", "0.0005",
                    "--fork-min-lr", "0.00005",
                    "--fork-rewarm-iters", "1",
                    "--instrumentation", "none",
                    # vvv THOG public wrapper operational controls remain mutable on fork exactly as on resume
                    "-e", "2",
                    "-u2",
                    "-k", "2",
                    # ^^^ THOG
                ]
                self.assertEqual(main(fork), 0)
                dirs = [path for path in checkpoints.iterdir() if path.is_dir()]
                self.assertEqual(len(dirs), 2)
                child_dir = next(path for path in dirs if path != original_dir)
                child_payload = load_payload(child_dir / "ckpt.pt")
                child_lifecycle = child_payload["lifecycle"]

                self.assertEqual(child_payload["completed_updates"], 6)
                self.assertEqual(child_lifecycle["fork_generation"], 1)
                self.assertEqual(child_lifecycle["root_start_label"], "260729-0100")
                self.assertEqual(child_lifecycle["parent_logical_run_id"], no_brainer_payload["lifecycle"]["logical_run_id"])
                self.assertEqual(child_lifecycle["parent_completed_updates"], 4)
                self.assertEqual(child_lifecycle["active_lr_phase_index"], 1)
                self.assertEqual(child_lifecycle["lr_phases"][1]["phase_type"], "restart_cosine")
                self.assertEqual(child_lifecycle["lr_phases"][1]["phase_end_update"], 6)
                self.assertEqual(child_payload["trainer_config"]["decay_updates"], 2)
                self.assertEqual(child_payload["trainer_config"]["max_updates"], 6)
                # vvv THOG prove -e/-u/-k survive the public CLI and apply to the forked child configuration
                self.assertEqual(child_payload["trainer_config"]["eval_interval"], 2)
                self.assertEqual(child_payload["trainer_config"]["eval_batches"], 2)
                self.assertEqual(child_payload["trainer_config"]["checkpoint_interval"], 2)
                # ^^^ THOG
                self.assertTrue((child_dir / "run_manifest.json").is_file())
            finally:
                if old_curve_root is None:
                    __import__("os").environ.pop("THOG2_CURVE_ROOT", None)
                else:
                    __import__("os").environ["THOG2_CURVE_ROOT"] = old_curve_root


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
