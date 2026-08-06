from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "# <<< THOG"
TARGET_COLUMN = 156


def align_file(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    aligned = []
    for line_number, line in enumerate(lines, start=1):
        ending = "\n" if line.endswith("\n") else ""
        body = line[:-1] if ending else line
        if MARKER not in body:
            aligned.append(line)
            continue
        prefix, marker_and_suffix = body.split(MARKER, 1)
        prefix = prefix.rstrip()
        required_spaces = TARGET_COLUMN - 1 - len(prefix)
        if required_spaces < 1:
            raise ValueError(
                f"{path}:{line_number}: code reaches column {len(prefix)}; "
                f"cannot place {MARKER!r} at column {TARGET_COLUMN}"
            )
        aligned.append(
            prefix + (" " * required_spaces) + MARKER + marker_and_suffix + ending
        )
    updated = "".join(aligned)
    if updated == original:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    arguments = parser.parse_args()
    for path in arguments.paths:
        align_file(path)


if __name__ == "__main__":
    main()
