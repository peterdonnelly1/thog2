#!/usr/bin/env python3
# vvv THOG
"""Collapse only the duplicate cadence lines produced by repeated legacy cleanup replay."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGETS = (
    ROOT / "tests/test_run_thog2_owt_startup_report.py",
    ROOT / "tests/test_plastic_depth_startup_lookahead_ui.py",
)
NEEDLE = "plastic__layer_count_probe__probe_every_n_steps"


def _dedupe_consecutive_cadence_lines(text: str) -> str:
    lines = text.splitlines(keepends=True)
    output = []
    for line in lines:
        if (
            output
            and NEEDLE in line
            and NEEDLE in output[-1]
            and line.strip() == output[-1].strip()
        ):
            continue
        output.append(line)
    return "".join(output)


def main() -> None:
    for path in TARGETS:
        text = path.read_text(encoding="utf-8")
        path.write_text(_dedupe_consecutive_cadence_lines(text), encoding="utf-8")


if __name__ == "__main__":
    main()
# ^^^ THOG
