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

from run_thog2_owt import main
from sheet.checkpoints import load_payload


ROOT = Path(__file__).resolve().parents[1]


def _write_tiny_dataset(path: Path, vocab_size: int = 32) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (np.arange(512, dtype=np.uint16) % vocab_size).tofile(path / "train.bin")
    (np.arange(256, dtype=np.uint16) % vocab_size).tofile(path / "val.bin")
    with (path / "meta.pkl").open("wb") as handle:
        pickle.dump({"vocab_size": vocab_size}, handle)


class ResumeAndForkShellIntegrationTests(unittest.TestCase):
    def test_public_wrapper_no_brainer_resume_appends_session_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            checkpoints = root / "checkpoints"
            logs = root / "logs"
            results = root / "results"
            wandb = root / "wandb"
            curves = root / "curves"
            _write_tiny_dataset(data)

            old_curve_root = os.environ.get("THOG2_CURVE_ROOT")
            os.environ["THOG2_CURVE_ROOT"] = str(curves)
            try:
                fresh = [
                    "--model-type", "dense",
                    "--run-mode", "fresh",
                    "--run-start-label", "260729-0410",
                    "--run-name", "SHELLTEST",
                    "--experiment-prefix", "SHELLTEST",
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
                    "--batch-size", "1",
                    "--gradient-accumulation-steps", "1",
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
                self.assertEqual(main(fresh), 0)

                checkpoint_dirs = [path for path in checkpoints.iterdir() if path.is_dir()]
                self.assertEqual(len(checkpoint_dirs), 1)
                checkpoint_path = checkpoint_dirs[0] / "ckpt.pt"
                before = load_payload(checkpoint_path)
                logical_run_id = before["lifecycle"]["logical_run_id"]
                log_path = Path(before["lifecycle"]["log_path"])

                environment = dict(os.environ)
                environment["THOG2_PYTHON"] = sys.executable
                completed = subprocess.run(
                    [
                        "bash",
                        "train_OWT.sh",
                        "--resume",
                        "260729-0410",
                        "-n",
                        "2",
                        "--checkpoint-root",
                        str(checkpoints),
                        "-I",
                        "none",
                    ],
                    cwd=ROOT,
                    env=environment,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

                after = load_payload(checkpoint_path)
                self.assertEqual(after["completed_updates"], 2)
                self.assertEqual(after["lifecycle"]["logical_run_id"], logical_run_id)
                self.assertEqual(len(after["lifecycle"]["sessions"]), 2)
                self.assertTrue(log_path.is_file())
                log_text = log_path.read_text(encoding="utf-8")
                self.assertIn("THOG2 lifecycle session", log_text)
                self.assertIn("mode:               resume", log_text)
                self.assertIn("target updates:     2", log_text)
                self.assertIn(after["lifecycle"]["session_id"], log_text)
            finally:
                if old_curve_root is None:
                    os.environ.pop("THOG2_CURVE_ROOT", None)
                else:
                    os.environ["THOG2_CURVE_ROOT"] = old_curve_root


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
