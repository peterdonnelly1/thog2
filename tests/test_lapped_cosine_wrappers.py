# vvv THOG
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


class LappedCosineWrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.environment = os.environ.copy()
        cls.environment["THOG2_PYTHON"] = sys.executable

    def dry_run(self, wrapper_name: str) -> subprocess.CompletedProcess[str]:
        command = [
            "bash",
            str(self.root / wrapper_name),
            "-p", "depth",
            "-B", "lapped_cosine",
            "-W", "8",
            "-i", "0.5",
            "-x", "true",
            "-g", "LAPPED_TEST",
            "-n", "2",
            "-w", "1",
            "-b", "1",
            "-A", "1",
            "-G", "1",
            "-u", "1",
            "-e", "1",
            "-l", "1",
            "-L", "8",
            "-H", "2",
            "-D", "16",
            "-C", "8",
            "-P", "4",
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

    def test_01_primary_wrappers_propagate_controls_and_identity(self) -> None:
        for wrapper_name in (
            "current_scruffy_train_OWT.sh",
            "current_dreedle_train_OWT.sh",
        ):
            with self.subTest(wrapper_name=wrapper_name):
                result = self.dry_run(wrapper_name)
                combined = result.stdout + result.stderr
                self.assertEqual(result.returncode, 0, combined)
                self.assertIn("LAPPED_COSINE_DEPTH", combined)
                self.assertIn("LCW_8", combined)
                self.assertIn("LCO_50", combined)
                self.assertIn("--basis-family lapped_cosine", combined)
                self.assertIn("--lapped-cosine-window-length 8", combined)
                self.assertIn("--lapped-cosine-overlap-fraction 0.5", combined)

    def test_02_invalid_overlap_fails_before_runner_launch(self) -> None:
        result = subprocess.run(
            [
                "bash",
                str(self.root / "current_scruffy_train_OWT.sh"),
                "-B", "lapped_cosine",
                "-i", "0.25",
                "-x", "true",
            ],
            cwd=self.root,
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        combined = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, combined)
        self.assertIn("supports only 0.5", combined)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
