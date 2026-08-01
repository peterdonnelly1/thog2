# vvv THOG
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Sequence, Tuple


Result = Dict[str, object]


def _test_files(checkout: Path) -> Tuple[Path, ...]:
    return tuple(
        path
        for path in sorted((checkout / "tests").glob("test_*.py"))
        if not path.name.startswith("test_hyperblock")
    )


def _node_ids(xml_path: Path, tag: str) -> Tuple[str, ...]:
    if not xml_path.exists():
        return ()
    root = ET.parse(xml_path).getroot()
    node_ids = []
    for case in root.iter("testcase"):
        if case.find(tag) is None:
            continue
        classname = case.attrib.get("classname", "")
        name = case.attrib.get("name", "")
        node_ids.append(f"{classname}::{name}")
    return tuple(sorted(node_ids))


def _counts(xml_path: Path) -> Dict[str, int]:
    if not xml_path.exists():
        return {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    root = ET.parse(xml_path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    if root.tag == "testsuites":
        suites = list(root)
    return {
        key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }


def _run_file(checkout: Path, relative_path: Path, timeout_seconds: int) -> Result:
    with tempfile.TemporaryDirectory() as directory:
        xml_path = Path(directory) / "pytest.xml"
        environment = dict(os.environ)
        environment.update(
            {
                "PYTHONPATH": str(checkout),
                "PYTHONHASHSEED": "0",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
        )
        command = (
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--tb=no",
            f"--junitxml={xml_path}",
            str(relative_path),
        )
        try:
            completed = subprocess.run(
                command,
                cwd=checkout,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
            )
            status = "pass" if completed.returncode == 0 else "fail"
            output = completed.stdout
            return_code = completed.returncode
        except subprocess.TimeoutExpired as error:
            status = "timeout"
            output = error.stdout or ""
            if isinstance(output, bytes):
                output = output.decode(errors="replace")
            return_code = 124
        failures = _node_ids(xml_path, "failure")
        errors = _node_ids(xml_path, "error")
        if status == "fail" and not failures and not errors:
            status = "error"
        return {
            "status": status,
            "return_code": return_code,
            "failures": failures,
            "errors": errors,
            "counts": _counts(xml_path),
            "output_tail": output[-4000:],
        }


def _equivalent_nonpass(head: Result, base: Result) -> bool:
    head_status = str(head["status"])
    base_status = str(base["status"])
    if head_status == "timeout":
        return base_status == "timeout"
    if head_status in {"fail", "error"}:
        if base_status not in {"fail", "error"}:
            return False
        head_nodes = set(head["failures"]) | set(head["errors"])
        base_nodes = set(base["failures"]) | set(base["errors"])
        return bool(head_nodes) and head_nodes <= base_nodes
    return False


def compare(
    head_checkout: Path,
    base_checkout: Path,
    *,
    timeout_seconds: int,
    max_files: Optional[int],
) -> Dict[str, object]:
    head_files = _test_files(head_checkout)
    if max_files is not None:
        head_files = head_files[:max_files]
    base_names = {path.name for path in _test_files(base_checkout)}
    records: List[Dict[str, object]] = []
    regressions: List[str] = []
    for index, head_file in enumerate(head_files, start=1):
        relative = Path("tests") / head_file.name
        head_result = _run_file(head_checkout, relative, timeout_seconds)
        record: Dict[str, object] = {
            "file": str(relative),
            "head": head_result,
            "base": None,
            "regression": False,
        }
        if head_result["status"] != "pass":
            if head_file.name not in base_names:
                record["regression"] = True
                regressions.append(str(relative))
            else:
                base_result = _run_file(base_checkout, relative, timeout_seconds)
                record["base"] = base_result
                if not _equivalent_nonpass(head_result, base_result):
                    record["regression"] = True
                    regressions.append(str(relative))
        records.append(record)
        print(
            f"[{index:03d}/{len(head_files):03d}] {head_result['status']:7} {relative}",
            flush=True,
        )
    return {
        "head_checkout": str(head_checkout),
        "base_checkout": str(base_checkout),
        "files_checked": len(head_files),
        "head_passed_files": sum(record["head"]["status"] == "pass" for record in records),
        "head_nonpass_files": sum(record["head"]["status"] != "pass" for record in records),
        "regressions": regressions,
        "records": records,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--max-files", type=int)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    result = compare(
        arguments.head.resolve(),
        arguments.base.resolve(),
        timeout_seconds=arguments.timeout_seconds,
        max_files=arguments.max_files,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        "REGRESSION SUMMARY: "
        f"files={result['files_checked']} "
        f"head_passed={result['head_passed_files']} "
        f"head_nonpass={result['head_nonpass_files']} "
        f"new_regressions={len(result['regressions'])}",
        flush=True,
    )
    return 1 if result["regressions"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
# ^^^ THOG
