#!/usr/bin/env python3
# vvv THOG
"""Run the v0.52 regression migration while preserving negative retired-option assertions."""

from __future__ import annotations

from pathlib import Path

import apply_plastic_v052_broad_regression_test_migration as migration


ROOT = Path(__file__).resolve().parents[1]


def _remove_retired_min_probe_pairs(text: str) -> str:
    lines = text.splitlines(keepends=True)
    output = []
    index = 0
    while index < len(lines):
        if '"--plastic-layer-count-probe-noise-min-observations",' in lines[index]:
            if index + 1 >= len(lines):
                raise RuntimeError("retired min-probes option has no value line")
            index += 2
            continue
        output.append(lines[index])
        index += 1
    return "".join(output)


def _remove_retired_min_probe_dry_run_expectation() -> None:
    path = ROOT / "tests/test_plastic_depth_interfaces.py"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text(
        "".join(
            line
            for line in lines
            if "--plastic-layer-count-probe-noise-min-observations 4" not in line
        ),
        encoding="utf-8",
    )


def main() -> None:
    migration._remove_retired_min_probe_pairs = _remove_retired_min_probe_pairs
    migration.main()
    _remove_retired_min_probe_dry_run_expectation()


if __name__ == "__main__":
    main()
# ^^^ THOG
