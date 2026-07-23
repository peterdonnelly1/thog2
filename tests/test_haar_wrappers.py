# vvv THOG
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


class BalancedHaarWrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.environment = os.environ.copy()
        cls.environment["THOG2_PYTHON"] = sys.executable

    def dry_run(self, wrapper_name: str, basis_family: str = "haar") -> subprocess.CompletedProcess[str]:
        command = [
            "bash",
            str(self.root / wrapper_name),
            "-p", "depth",
            "-B", basis_family,
            "-x", "true",
            "-g", "HAAR_TEST",
            "-n", "2",
            "-w", "1",
            "-b", "1",
            "-A", "1",
            "-G", "1",
            "-u", "1",
            "-e", "1",
            "-l", "1",
            "-L", "4",
            "-H", "2",
            "-D", "16",
            "-C", "8",
            "-P", "3",
            "-Q", "8",
            "-J", "4",
            "-O", "4",
            "-X", "8",
            "-Y", "32",
            "-S", "1",
            "-I", "none",
            "-F", "none",
            "-T", "float32",
            "-K", "math",
        ]
        return subprocess.run(
            command,
            cwd=self.root,
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def test_01_scruffy_and_dreedle_dry_runs_accept_haar_and_use_haar_artifact_tag(self) -> None:
        for wrapper_name in ("current_scruffy_train_OWT.sh", "current_dreedle_train_OWT.sh"):
            with self.subTest(wrapper_name=wrapper_name):
                result = self.dry_run(wrapper_name)
                combined = result.stdout + result.stderr
                self.assertEqual(result.returncode, 0, combined)
                self.assertIn("HAAR_DEPTH", combined)
                self.assertIn("spectral / depth / haar", combined)
                self.assertIn("--basis-family haar", combined)

    def test_02_unregistered_basis_fails_before_training(self) -> None:
        result = self.dry_run("current_scruffy_train_OWT.sh", basis_family="not_a_basis")
        combined = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, combined)
        self.assertIn("unknown basis_family", combined)
        self.assertNotIn("OWT run finished", combined)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
