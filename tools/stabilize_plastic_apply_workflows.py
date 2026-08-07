#!/usr/bin/env python3
# vvv THOG
"""Make the older PLASTIC apply transform idempotent before the v0.52 migration."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    lifecycle = ROOT / "tests/test_plastic_depth_lifecycle.py"
    text = lifecycle.read_text(encoding="utf-8")
    text = re.sub(
        r"(?:    plastic__coarse_phase_roll_through: bool = False\n)+",
        "    plastic__coarse_phase_roll_through: bool = False\n",
        text,
    )
    lifecycle.write_text(text, encoding="utf-8")

    transform = ROOT / "tools/apply_plastic_core_canonicalisation_final.py"
    text = transform.read_text(encoding="utf-8")
    old = '''    content = content.replace(\n        '    plastic__initial_layer_count: int = 2\\n',\n        '    plastic__initial_layer_count: int = 2\\n    plastic__coarse_phase_roll_through: bool = False\\n',\n    )\n'''
    new = '''    if "    plastic__coarse_phase_roll_through: bool = False\\n" not in content:\n        content = content.replace(\n            '    plastic__initial_layer_count: int = 2\\n',\n            '    plastic__initial_layer_count: int = 2\\n    plastic__coarse_phase_roll_through: bool = False\\n',\n            1,\n        )\n'''
    if new not in text:
        if old not in text:
            raise RuntimeError("legacy PLASTIC core transform idempotence anchor not found")
        text = text.replace(old, new, 1)
    transform.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
# ^^^ THOG
# vvv THOG migration trigger after retiring superseded writeback workflows
# ^^^ THOG
