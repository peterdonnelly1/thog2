from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = (
    ROOT / "current_scruffy_train_OWT.sh",
    ROOT / "current_dreedle_train_OWT.sh",
)


class JpegLikeV1WrapperTests(unittest.TestCase):
    def test_registry_comment_and_help_expose_new_contract(self) -> None:
        for path in WRAPPERS:
            source = path.read_text(encoding="utf-8")
            self.assertIn("# Registry:", source)
            self.assertIn("JPEG_LIKE_V1 segments MLP_UP along MLP_HIDDEN only", source)
            self.assertIn("--mlp-hidden-compressor", source)
            self.assertIn("--mlp-hidden-group-size", source)
            self.assertIn("jpeg_like_v1", source)

    def test_wrappers_remain_valid_bash(self) -> None:
        for path in WRAPPERS:
            subprocess.run(["bash", "-n", str(path)], cwd=ROOT, check=True)

    def test_scruffy_dry_run_propagates_geometry_compressor_group_and_y(self) -> None:
        environment = dict(os.environ)
        environment["THOG2_PYTHON"] = sys.executable
        result = subprocess.run(
            [
                "bash",
                "current_scruffy_train_OWT.sh",
                "-p", "jpeg_like_v1",
                "-g", "JPEG_SMOKE",
                "-n", "2",
                "-w", "0",
                "-b", "1",
                "-A", "1",
                "-u", "1",
                "-e", "1",
                "-l", "1",
                "-k", "0",
                "-L", "4",
                "-H", "2",
                "-D", "8",
                "-C", "8",
                "-P", "3",
                "-Q", "4",
                "-J", "2",
                "-O", "2",
                "-X", "4",
                "-Y", "2",
                "--mlp-hidden-group-size", "4",
                "--mlp-hidden-compressor", "dct",
                "-I", "none",
                "-F", "none",
                "-x", "true",
            ],
            cwd=ROOT,
            env=environment,
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertIn("JPEG_LIKE_V1:       compressor=dct group=4 Y=2", result.stdout)
        self.assertIn("CHEBY_JPEG_LIKE_V1_DCT", result.stdout)
        self.assertIn("MHG_4", result.stdout)
        self.assertIn("--mlp-hidden-compressor dct", result.stdout)
        self.assertIn("--mlp-hidden-group-size 4", result.stdout)

    def test_wrapper_rejects_y_larger_than_group(self) -> None:
        environment = dict(os.environ)
        environment["THOG2_PYTHON"] = sys.executable
        result = subprocess.run(
            [
                "bash", "current_scruffy_train_OWT.sh",
                "-p", "jpeg_like_v1",
                "-n", "2", "-w", "0",
                "-L", "4", "-H", "2", "-D", "8",
                "-P", "3", "-Q", "4", "-J", "2", "-O", "2", "-X", "4", "-Y", "5",
                "--mlp-hidden-group-size", "4",
                "-I", "none", "-F", "none", "-x", "true",
            ],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must not exceed MLP_HIDDEN_GROUP_SIZE", result.stderr)


if __name__ == "__main__":
    unittest.main()
