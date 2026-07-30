# vvv THOG
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class PublicTrainOwtWrapperTests(unittest.TestCase):
    def test_fresh_dry_run_preserves_master_cli_and_aligns_startup_values(self) -> None:
        environment = dict(os.environ)
        environment["THOG2_PYTHON"] = sys.executable
        completed = subprocess.run(
            [
                "bash",
                "train_OWT.sh",
                "-x",
                "true",
                "-I",
                "none",
                "-g",
                "WRAPPER_TEST",
                "-n",
                "2",
                "-w",
                "0",
                "-p",
                "dense",
                "-b",
                "1",
                "-A",
                "1",
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
            ],
            cwd=REPOSITORY_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("DRY RUN:", completed.stdout)

        lines = completed.stdout.splitlines()
        start = lines.index("scruffy OWT train") + 1
        end = next(index for index in range(start, len(lines)) if lines[index].startswith("DRY RUN:"))
        summary_lines = [line for line in lines[start:end] if line.startswith("  ")]
        self.assertTrue(summary_lines)
        self.assertTrue(
            any(line.startswith("  vectorise per-head materialisation:") for line in summary_lines)
        )
        for line in summary_lines:
            self.assertGreaterEqual(len(line), 38, line)
            self.assertEqual(line[37], " ", line)
            self.assertNotEqual(line[38:], "", line)

    def test_public_wrapper_sources_preserved_core_for_fresh_runs(self) -> None:
        wrapper = (REPOSITORY_ROOT / "train_OWT.sh").read_text(encoding="utf-8")
        self.assertIn('source ./train_OWT_core.sh "$@"', wrapper)
        self.assertIn('exec ./resume_and_fork_OWT.sh "$@"', wrapper)
        self.assertIn("printf '  %-35s %s\\n'", wrapper)


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
