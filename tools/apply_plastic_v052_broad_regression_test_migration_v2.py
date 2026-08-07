#!/usr/bin/env python3
# vvv THOG
"""Run the v0.52 regression migration while preserving negative retired-option assertions."""

from __future__ import annotations

import apply_plastic_v052_broad_regression_test_migration as migration


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


def main() -> None:
    migration._remove_retired_min_probe_pairs = _remove_retired_min_probe_pairs
    migration.main()


if __name__ == "__main__":
    main()
# ^^^ THOG
