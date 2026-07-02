# vvv THOG
from __future__ import annotations

import argparse
import json
import platform
import sys
import unittest
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def serialize(items):
    return [
        {"test": case.id(), "traceback": text}
        for case, text in items
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.evidence.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        evidence = {
            "stage": 4,
            "suite": "cuda_acceptance",
            "satisfied": False,
            "reason": "CUDA is not available in this environment",
            "tests_run": 0,
        }
        arguments.evidence.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise SystemExit(1)

    suite = unittest.defaultTestLoader.loadTestsFromName(
        "tests.test_sheet_stage4_cuda"
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    evidence = {
        "stage": 4,
        "suite": "cuda_acceptance",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "bfloat16_supported": torch.cuda.is_bf16_supported(),
        "test_execution": {
            "tests_run": result.testsRun,
            "failure_count": len(result.failures),
            "error_count": len(result.errors),
            "skipped_count": len(result.skipped),
            "failures": serialize(result.failures),
            "errors": serialize(result.errors),
            "skipped": [
                {"test": case.id(), "reason": reason}
                for case, reason in result.skipped
            ],
            "successful": result.wasSuccessful(),
        },
        "satisfied": bool(
            result.wasSuccessful()
            and result.testsRun == 7
            and not result.skipped
        ),
    }
    arguments.evidence.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not evidence["satisfied"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
# ^^^ THOG
