from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class OptimizerWrapperTests(unittest.TestCase):
    def test_full_wrappers_retain_body_and_native_optimizer_help(self) -> None:
        for filename in ("current_scruffy_train_OWT.sh", "current_dreedle_train_OWT.sh"):
            path = ROOT / filename
            source = path.read_text(encoding="utf-8")
            self.assertGreater(len(source.splitlines()), 400)
            self.assertIn("run_grid_point()", source)
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


if __name__ == "__main__":
    unittest.main()
