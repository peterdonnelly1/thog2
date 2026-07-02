# vvv THOG
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def run_expected_failure(mode: str, expected_text: str) -> None:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        str(REPOSITORY_ROOT / "tests" / "stage5_ddp_failure_worker.py"),
        "--mode",
        mode,
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    combined = completed.stdout + "\n" + completed.stderr
    if completed.returncode == 0:
        raise AssertionError(
            f"distributed failure mode {mode!r} unexpectedly succeeded\n{combined}"
        )
    if expected_text not in combined:
        raise AssertionError(
            f"distributed failure mode {mode!r} lacked diagnostic {expected_text!r}\n"
            f"{combined}"
        )


class Stage5FailurePropagationTests(unittest.TestCase):
    def test_s5_structure_disagreement_fails_all_ranks_without_hang(self) -> None:
        run_expected_failure(
            "structure_mismatch",
            "distributed injected structure differs across ranks",
        )

    def test_s5_rank_local_nonfinite_state_fails_collectively(self) -> None:
        run_expected_failure(
            "nonfinite_rank",
            "non-finite training loss on at least one rank",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
