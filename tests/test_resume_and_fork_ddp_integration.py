# vvv THOG
from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sheet.checkpoints import load_payload


ROOT = Path(__file__).resolve().parents[1]


def _write_tiny_dataset(path: Path, vocab_size: int = 32) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (np.arange(1024, dtype=np.uint16) % vocab_size).tofile(path / "train.bin")
    (np.arange(256, dtype=np.uint16) % vocab_size).tofile(path / "val.bin")
    with (path / "meta.pkl").open("wb") as handle:
        pickle.dump({"vocab_size": vocab_size}, handle)


class ResumeAndForkDdpIntegrationTests(unittest.TestCase):
    def test_two_rank_cpu_resume_preserves_logical_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            checkpoints = root / "checkpoints"
            logs = root / "logs"
            results = root / "results"
            wandb = root / "wandb"
            curves = root / "curves"
            _write_tiny_dataset(data)

            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(ROOT)
            environment["THOG2_INSTRUMENTATION"] = "none"
            environment["THOG2_CURVE_ROOT"] = str(curves)
            environment["OMP_NUM_THREADS"] = "1"

            torchrun = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                "-m",
                "run_thog2_owt",
            ]
            fresh = [
                *torchrun,
                "--model-type", "dense",
                "--run-mode", "fresh",
                "--run-start-label", "260729-0420",
                "--run-name", "DDPRESUME",
                "--experiment-prefix", "DDPRESUME",
                "--max-iters", "1",
                "--data-dir", str(data),
                "--checkpoint-root", str(checkpoints),
                "--log-root", str(logs),
                "--result-root", str(results),
                "--wandb-root", str(wandb),
                "--instrumentation", "none",
                "--no-wandb",
                "--device", "cpu",
                "--dtype", "float32",
                "--batch-size", "2",
                "--gradient-accumulation-steps", "2",
                "--block-size", "8",
                "--n-layer", "1",
                "--n-head", "1",
                "--n-embd", "8",
                "--warmup-iters", "0",
                "--learning-rate", "0.001",
                "--min-lr", "0.0001",
                "--eval-interval", "1",
                "--eval-iters", "1",
                "--log-interval", "1",
                "--checkpoint-interval", "1",
                "--residual-init-depth-source", "true_layer_depth",
                "--no-activation-checkpointing",
            ]
            first = subprocess.run(
                fresh,
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
                timeout=120,
            )
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)

            checkpoint_dirs = [path for path in checkpoints.iterdir() if path.is_dir()]
            self.assertEqual(len(checkpoint_dirs), 1)
            checkpoint_path = checkpoint_dirs[0] / "ckpt.pt"
            before = load_payload(checkpoint_path)
            logical_run_id = before["lifecycle"]["logical_run_id"]
            self.assertEqual(before["completed_updates"], 1)
            self.assertEqual(before["lifecycle"]["world_size"], 2)

            resume = [
                *torchrun,
                "--resume", "260729-0420",
                "-n", "2",
                "--checkpoint-root", str(checkpoints),
                "--instrumentation", "none",
            ]
            second = subprocess.run(
                resume,
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
                timeout=120,
            )
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)

            after = load_payload(checkpoint_path)
            self.assertEqual(after["completed_updates"], 2)
            self.assertEqual(after["lifecycle"]["logical_run_id"], logical_run_id)
            self.assertEqual(after["lifecycle"]["world_size"], 2)
            self.assertEqual(len(after["lifecycle"]["sessions"]), 2)
            self.assertEqual(after["lifecycle"]["sessions"][1]["starting_completed_updates"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
