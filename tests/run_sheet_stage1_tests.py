# vvv THOG
from __future__ import annotations

import argparse
import json
import os
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tests.stage1_calibration import run_calibration


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
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "successful": result.wasSuccessful(),
    }
    evidence["accepted"] = bool(calibration["accepted"] and result.wasSuccessful())
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not evidence["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
# ^^^ THOG
