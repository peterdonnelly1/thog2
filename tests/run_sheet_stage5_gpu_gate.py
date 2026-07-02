# vvv THOG
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def run_component(command, evidence_path: Path) -> Dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if evidence_path.exists():
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    else:
        evidence = {
            "satisfied": False,
            "test_execution": {
                "tests_run": 0,
                "failure_count": 0,
                "error_count": 1,
                "skipped_count": 0,
                "successful": False,
            },
        }
    evidence["command_returncode"] = completed.returncode
    evidence["command_stdout"] = completed.stdout
    evidence["command_stderr"] = completed.stderr
    return evidence


def count(component: Dict[str, Any], name: str) -> int:
    return int(component.get("test_execution", {}).get(name, 0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--bounded-updates", type=int, default=4)
    parser.add_argument("--include-q1024", action="store_true")
    arguments = parser.parse_args()
    arguments.evidence.parent.mkdir(parents=True, exist_ok=True)

    stage4_path = arguments.evidence.parent / "stage5_stage4_cuda_regression.json"
    target_path = arguments.evidence.parent / "stage5_target_gpu_acceptance.json"

    stage4 = run_component(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "tests" / "run_sheet_stage4_gpu_tests.py"),
            "--evidence",
            str(stage4_path),
        ],
        stage4_path,
    )
    target_command = [
        sys.executable,
        str(REPOSITORY_ROOT / "tests" / "run_sheet_stage5_gpu_tests.py"),
        "--evidence",
        str(target_path),
        "--bounded-updates",
        str(arguments.bounded_updates),
    ]
    if arguments.include_q1024:
        target_command.append("--include-q1024")
    target = run_component(target_command, target_path)

    tests_run = count(stage4, "tests_run") + count(target, "tests_run")
    failure_count = count(stage4, "failure_count") + count(target, "failure_count")
    error_count = count(stage4, "error_count") + count(target, "error_count")
    skipped_count = count(stage4, "skipped_count") + count(target, "skipped_count")
    satisfied = (
        bool(stage4.get("satisfied"))
        and bool(target.get("satisfied"))
        and stage4.get("command_returncode") == 0
        and target.get("command_returncode") == 0
        and tests_run == 16
        and failure_count == 0
        and error_count == 0
        and skipped_count == 0
    )
    combined = {
        "stage": 5,
        "suite": "aggregate_gpu_gate",
        "bounded_updates": arguments.bounded_updates,
        "include_q1024": arguments.include_q1024,
        "components": {
            "stage4_cuda_regression": stage4,
            "stage5_target_gpu_acceptance": target,
        },
        "test_execution": {
            "tests_run": tests_run,
            "failure_count": failure_count,
            "error_count": error_count,
            "skipped_count": skipped_count,
            "successful": satisfied,
        },
        "satisfied": satisfied,
    }
    arguments.evidence.write_text(
        json.dumps(combined, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(combined["test_execution"], indent=2, sort_keys=True))
    if not satisfied:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
# ^^^ THOG
