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
                "-W",
                "false",
                "-M",
                "disabled",
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
        self.assertIn("DENSE2_scruffy__TEST__owt__", output)
        self.assertIn("--model-type dense", output)
        self.assertIn("activation checkpoint:  true", output)
        self.assertIn("checkpoint segment:     1", output)
        self.assertIn("DRY RUN:", output)

    def test_s6_34_sheet_runner_resolves_orders_and_same_shared_letters(self) -> None:
        output = self.run_script(
            "current_scruffy_train_SHEET_OWT.sh",
            ["-P", "2", "-Q", "4"],
        )
        self.assertIn("SHEET_scruffy__TEST__owt__", output)
        self.assertIn("_p_2_q_4__", output)
        self.assertIn("--model-type sheet", output)
        self.assertIn("sheet orders:           P2 / Q4", output)
        self.assertIn("DRY RUN:", output)

    def test_s6_35_shell_sources_pass_bash_syntax_check(self) -> None:
        for script in (
            "current_scruffy_train_DENSE_OWT.sh",
            "current_scruffy_train_SHEET_OWT.sh",
        ):
            completed = subprocess.run(
                ["bash", "-n", str(REPOSITORY_ROOT / script)],
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, msg=completed.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
