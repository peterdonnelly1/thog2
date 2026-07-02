# vvv THOG
from __future__ import annotations

import shlex
import subprocess
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "THOG2_Stage_3_Setup_Environment.sh"
)


class Stage3EnvironmentSetupTests(unittest.TestCase):
    def test_environment_setup_script_contract(self) -> None:
        syntax = subprocess.run(
            ["bash", "-n", str(SCRIPT_PATH)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            syntax.returncode,
            0,
            syntax.stderr,
        )

        content = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("conda deactivate", content)
        self.assertIn("--prompt thog2", content)
        self.assertIn("$HOME/.venvs/thog2", content)
        self.assertIn(
            "source docs/THOG2_Stage_3_Setup_Environment.sh",
            content,
        )

        executed = subprocess.run(
            ["bash", str(SCRIPT_PATH)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(executed.returncode, 2)
        self.assertIn("must be sourced", executed.stderr)

        help_command = (
            "source "
            + shlex.quote(str(SCRIPT_PATH))
            + " --help"
        )
        help_result = subprocess.run(
            ["bash", "-lc", help_command],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            help_result.returncode,
            0,
            help_result.stderr,
        )
        self.assertIn("--recreate", help_result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
