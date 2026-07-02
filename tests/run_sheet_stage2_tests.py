# vvv THOG
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import unittest
from pathlib import Path
from typing import Dict, List, Tuple

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from sheet.model import SheetGPT, SheetGPTConfig


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

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    torch.manual_seed(2202)
    evidence_model = SheetGPT(
        SheetGPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=3,
            n_head=4,
            n_embd=16,
            dropout=0.0,
            bias=True,
            depth_order=3,
            base_row_order=8,
        )
    )

    suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_sheet_stage2")
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    evidence = {
        "stage": 2,
        "architecture": "thog2_sheet",
        "commit": os.environ.get("GITHUB_SHA"),
        "branch": os.environ.get("GITHUB_HEAD_REF") or os.environ.get("GITHUB_REF_NAME"),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "test_execution": {
            "command": (
                "python tests/run_sheet_stage2_tests.py "
                f"--evidence {arguments.evidence}"
            ),
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
        },
        "reference_model_config": evidence_model.config.to_dict(),
        "parameter_report": evidence_model.parameter_report(),
        "state_dict_key_count": len(evidence_model.state_dict()),
        "persistent_basis_keys": list(evidence_model.trajectory.persistent_basis_keys()),
        "compact_state_violations": list(evidence_model.compact_state_violations()),
        "epsilon_contract": {
            "arbitrary_sampled_sheet": (
                "Guaranteed to floating-point epsilon only for saturated "
                "P=L and Q=C bases."
            ),
            "sub_saturated_sheet": (
                "The orthogonal projection is best in sampled Frobenius norm, "
                "but no arbitrary-weight epsilon guarantee exists without "
                "smoothness assumptions or a measured residual."
            ),
            "smooth_or_in_span_sheet": (
                "Any sampled sheet already in the chosen tensor-product span "
                "is reconstructed to floating-point epsilon."
            ),
        },
        "accepted": bool(
            result.wasSuccessful()
            and not evidence_model.compact_state_violations()
            and not evidence_model.trajectory.persistent_basis_keys()
        ),
    }
    arguments.evidence.parent.mkdir(parents=True, exist_ok=True)
    arguments.evidence.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not evidence["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
# ^^^ THOG
