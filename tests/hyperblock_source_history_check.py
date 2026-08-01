# vvv THOG
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Dict, List, Optional, Sequence, Tuple


SOURCE_SUFFIXES = (".py", ".sh")
EXCLUDED_PREFIXES = ("tests/", "docs/", ".github/")


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments),
        cwd=repository,
        text=True,
    )


def _modified_source_paths(repository: Path, base: str, head: str) -> Tuple[str, ...]:
    names = _git(
        repository,
        "diff",
        "--name-only",
        "--diff-filter=M",
        base,
        head,
    ).splitlines()
    return tuple(
        name
        for name in names
        if name.endswith(SOURCE_SUFFIXES)
        and not name.startswith(EXCLUDED_PREFIXES)
    )


def _removed_lines(repository: Path, base: str, head: str, path: str) -> Tuple[str, ...]:
    diff = _git(
        repository,
        "diff",
        "--unified=0",
        "--no-color",
        base,
        head,
        "--",
        path,
    )
    removed = []
    for line in diff.splitlines():
        if not line.startswith("-") or line.startswith("---"):
            continue
        candidate = line[1:].strip()
        if not candidate or candidate.startswith("#"):
            continue
        removed.append(candidate)
    return tuple(removed)


def _head_lines(repository: Path, head: str, path: str) -> Tuple[str, ...]:
    content = _git(repository, "show", f"{head}:{path}")
    return tuple(line.strip() for line in content.splitlines())


def audit(repository: Path, base: str, head: str) -> Dict[str, object]:
    violations: List[Dict[str, str]] = []
    checked_paths = _modified_source_paths(repository, base, head)
    for path in checked_paths:
        head_lines = set(_head_lines(repository, head, path))
        for removed in _removed_lines(repository, base, head, path):
            if removed in head_lines:
                continue
            if f"# {removed}" in head_lines or f"#{removed}" in head_lines:
                continue
            violations.append({"path": path, "removed_line": removed})
    return {
        "base": base,
        "head": head,
        "checked_paths": checked_paths,
        "violation_count": len(violations),
        "violations": violations,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    result = audit(arguments.repository.resolve(), arguments.base, arguments.head)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered)
    print(rendered, end="")
    return 1 if result["violation_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
# ^^^ THOG
