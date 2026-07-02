# vvv THOG
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests import run_sheet_stage5_gpu_gate


class Stage5GpuGateLaunchTests(unittest.TestCase):
    def test_s5_aggregate_gate_launches_target_suite_as_module(self) -> None:
        stage4_result = {
            "satisfied": True,
            "command_returncode": 0,
            "test_execution": {
                "tests_run": 7,
                "failure_count": 0,
                "error_count": 0,
                "skipped_count": 0,
                "successful": True,
            },
        }
        stage5_result = {
            "satisfied": True,
            "command_returncode": 0,
            "test_execution": {
                "tests_run": 9,
                "failure_count": 0,
                "error_count": 0,
                "skipped_count": 0,
                "successful": True,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            evidence_path = Path(directory) / "aggregate.json"
            argv = [
                "run_sheet_stage5_gpu_gate.py",
                "--evidence",
                str(evidence_path),
            ]
            with patch.object(sys, "argv", argv), patch.object(
                run_sheet_stage5_gpu_gate,
                "run_component",
                side_effect=(stage4_result, stage5_result),
            ) as run_component:
                run_sheet_stage5_gpu_gate.main()

            target_command = run_component.call_args_list[1].args[0]
            self.assertEqual(target_command[:3], [
                sys.executable,
                "-m",
                "tests.run_sheet_stage5_gpu_tests",
            ])
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            self.assertTrue(evidence["satisfied"])
            self.assertEqual(evidence["test_execution"]["tests_run"], 16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
