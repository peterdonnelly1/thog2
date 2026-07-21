# vvv THOG
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class RunnerScriptTests(unittest.TestCase):
    def run_script(self, script: str, extra: list[str]) -> str:
        environment = dict(os.environ)
        environment["THOG2_PYTHON"] = sys.executable
        completed = subprocess.run(
            [
                "bash",
                str(REPOSITORY_ROOT / script),
                "-x",
                "true",
                "-I",
                "none",
                "-g",
                "TEST",
                "-n",
                "2",
                "-w",
                "0",
                "-b",
                "1",
                "-A",
                "2",
                "-G",
                "1",
                "-L",
                "2",
                "-H",
                "2",
                "-D",
                "8",
                "-C",
                "8",
                "-S",
                "1",
                *extra,
            ],
            cwd=REPOSITORY_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            self.fail(
                f"runner failed: {script}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        return completed.stdout

    def test_s6_33_dense_runner_resolves_canonical_identity_and_shared_defaults(self) -> None:
        output = self.run_script("current_scruffy_train_DENSE_OWT.sh", [])
        self.assertIn("TEST_DENSE_scruffy", output)
        self.assertIn("_r_depth_scaled_z_true_layer_depth_S_1", output)
        self.assertIn("--model-type dense", output)
        self.assertIn("--residual-init-policy depth_scaled", output)
        self.assertIn("--residual-init-depth-source true_layer_depth", output)
        self.assertIn("model/preset/basis: dense / dense / chebyshev", output)
        self.assertIn("instrumentation:    none", output)
        self.assertIn("DRY RUN:", output)

    def test_s6_34_depth_runner_resolves_all_orders_and_default_dof_implied_depth(self) -> None:
        output = self.run_script(
            "current_scruffy_train_SHEET_OWT.sh",
            [
                "-P",
                "2",
                "-Q",
                "4",
                "-J",
                "2",
                "-O",
                "2",
                "-X",
                "4",
                "-Y",
                "16",
                "-r",
                "depth_scaled",
                "-Z",
                "12",
            ],
        )
        self.assertIn("TEST_CHEBY_DEPTH_scruffy", output)
        self.assertIn(
            "_P_2_Q_4_J_2_O_2_X_4_Y_16_r_depth_scaled_z_dof_implied_depth_S_1",
            output,
        )
        self.assertIn("--model-type sheet", output)
        self.assertIn("--geometry-preset depth", output)
        self.assertIn("--residual-init-policy depth_scaled", output)
        self.assertIn("--residual-init-depth-source dof_implied_depth", output)
        self.assertIn("P2 Q4 J2 O2 X4 Y16", output)
        self.assertIn("instrumentation:    none", output)
        self.assertIn("DRY RUN:", output)

    def test_s6_35_shell_sources_pass_bash_syntax_check(self) -> None:
        for script in (
            "current_scruffy_train_DENSE_OWT.sh",
            "current_scruffy_train_SHEET_OWT.sh",
            "current_scruffy_train_OWT.sh",
            "current_dreedle_train_OWT.sh",
        ):
            completed = subprocess.run(
                ["bash", "-n", str(REPOSITORY_ROOT / script)],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
