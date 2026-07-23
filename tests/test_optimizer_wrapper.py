from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class OptimizerWrapperTests(unittest.TestCase):
    def test_full_wrappers_retain_body_and_native_optimizer_help(self) -> None:
        for filename in ("current_scruffy_train_OWT.sh", "current_dreedle_train_OWT.sh"):
            path = ROOT / filename
            source = path.read_text(encoding="utf-8")
            self.assertTrue(source.startswith("#!/bin/bash\nset -euo pipefail\n"))
            self.assertGreater(len(source.splitlines()), 400)
            self.assertIn("run_grid_point()", source)
            self.assertLess(source.index("usage()"), source.index("accept long optimizer controls"))
            self.assertIn("-y NAME, --optimizer NAME", source)
            self.assertIn('export THOG2_OPTIMIZER="$OPTIMIZER"', source)
            self.assertNotIn(".current_", source)
            self.assertNotIn("optimizer_train_OWT_wrapper.sh", source)

    def test_shell_syntax(self) -> None:
        subprocess.run(
            ["bash", "-n", "current_scruffy_train_OWT.sh", "current_dreedle_train_OWT.sh"],
            cwd=ROOT,
            check=True,
        )

    def test_help_contains_old_and_new_controls(self) -> None:
        for filename in ("current_scruffy_train_OWT.sh", "current_dreedle_train_OWT.sh"):
            result = subprocess.run(
                ["bash", filename, "-h"],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertIn("-p PRESET=", result.stdout)
            self.assertIn("-B BASIS_FAMILY=", result.stdout)
            self.assertIn("-y OPTIMIZER=", result.stdout)
            self.assertIn("sgd_nesterov", result.stdout)
            self.assertIn("capital -C", (ROOT / filename).read_text(encoding="utf-8"))

    def run_dry_wrapper(self, *arguments: str) -> str:
        environment = dict(os.environ)
        environment["THOG2_PYTHON"] = sys.executable
        result = subprocess.run(
            [
                "bash",
                "current_scruffy_train_OWT.sh",
                *arguments,
                "-p", "dense",
                "-n", "2",
                "-w", "0",
                "-b", "1",
                "-A", "1",
                "-u", "1",
                "-e", "1",
                "-l", "1",
                "-k", "0",
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
        return result.stdout

    def test_short_optimizer_control_uses_optimizer_defaults(self) -> None:
        output = self.run_dry_wrapper("-y", "sgd")
        self.assertIn("optimizer:          sgd", output)
        self.assertIn("LR_1000", output)
        self.assertIn("OPT_SGD", output)

    def test_long_optimizer_controls_and_explicit_lr_override(self) -> None:
        output = self.run_dry_wrapper(
            "--optimizer=rmsprop",
            "--optimizer-momentum=0.95",
            "-c", "77",
            "-f", "07",
        )
        self.assertIn("optimizer:          rmsprop  momentum=0.95", output)
        self.assertIn("LR_77", output)
        self.assertIn("min_lr=7e-5", output)

    def test_existing_run_scruffy_scripts_remain_present(self) -> None:
        self.assertTrue((ROOT / "run_scruffy_dense_sheet_l144_smoke_owt.sh").is_file())
        self.assertTrue((ROOT / "run_scruffy_sheet_eden_best_long_owt.sh").is_file())


if __name__ == "__main__":
    unittest.main()
