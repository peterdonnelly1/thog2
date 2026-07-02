# vvv THOG
from __future__ import annotations

import argparse
import json
import os
import sys
import unittest
from pathlib import Path
from typing import Dict, List, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tests.stage1_calibration import run_calibration


def _serialize_problems(
    problems: List[Tuple[unittest.case.TestCase, str]],
) -> List[Dict[str, str]]:
    return [
        {
            "test": test_case.id(),
            "traceback": traceback_text,
        }
        for test_case, traceback_text in problems
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    arguments = parser.parse_args()

    evidence_path = arguments.evidence.resolve()
    calibration = run_calibration(evidence_path)
    os.environ["THOG2_STAGE1_CALIBRATION"] = str(evidence_path)

    suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_sheet_stage1")
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["test_execution"] = {
        "command": f"python tests/run_sheet_stage1_tests.py --evidence {arguments.evidence}",
        "tests_run": result.testsRun,
        "failure_count": len(result.failures),
        "error_count": len(result.errors),
        "skipped_count": len(result.skipped),
        "failures": _serialize_problems(result.failures),
        "errors": _serialize_problems(result.errors),
        "skipped": [
            {"test": test_case.id(), "reason": reason}
            for test_case, reason in result.skipped
        ],
        "successful": result.wasSuccessful(),
    }
    evidence["accepted"] = bool(calibration["accepted"] and result.wasSuccessful())
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not evidence["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
# ^^^ THOG
