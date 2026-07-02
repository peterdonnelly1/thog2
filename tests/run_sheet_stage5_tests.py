# vvv THOG
from __future__ import annotations

import argparse
import json
import os
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
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    suite = unittest.defaultTestLoader.discover(
        str(REPOSITORY_ROOT / "tests"),
        pattern="test_sheet_stage5_[!c]*.py",
        top_level_dir=str(REPOSITORY_ROOT),
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    evidence = {
        "stage": 5,
        "suite": "cpu_and_two_rank_ddp",
        "commit": os.environ.get("GITHUB_SHA"),
        "branch": (
            os.environ.get("GITHUB_HEAD_REF")
            or os.environ.get("GITHUB_REF_NAME")
        ),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
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
        "distributed_gate": {
            "backend": "gloo",
            "world_size": 2,
            "satisfied": result.wasSuccessful(),
        },
        "gpu_target_gate": {
            "required_evidence": "evidence/stage5_gpu_acceptance.json",
            "satisfied": False,
            "reason": (
                "The hosted workflow has no target CUDA device; run the Stage 5 "
                "GPU acceptance suite on the RTX 4090 Laptop GPU."
            ),
        },
    }
    arguments.evidence.parent.mkdir(parents=True, exist_ok=True)
    arguments.evidence.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
# ^^^ THOG
