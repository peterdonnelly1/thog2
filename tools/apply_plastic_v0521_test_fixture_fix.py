#!/usr/bin/env python3
# vvv THOG
"""Apply PLASTIC v0.521 and make the pre-existing tiny artifact fixture explicit about its legal probe-token sample."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

subprocess.run(
    ["python", "tools/apply_plastic_v0521_probe_sampling_and_console.py"],
    cwd=ROOT,
    check=True,
)

path = ROOT / "tests/test_plastic_cli_console_refinements.py"
content = path.read_text(encoding="utf-8")
addition = "        plastic__layer_count_probe__number_of_sampled_valid_tokens=64,\n"
if addition not in content:
    anchor = "        plastic__max_permitted_layers=32,\n"
    if content.count(anchor) != 1:
        raise RuntimeError(
            "expected exactly one tiny artifact-fixture max-layer anchor; "
            f"found {content.count(anchor)}"
        )
    content = content.replace(anchor, anchor + addition, 1)
    path.write_text(content, encoding="utf-8")

print("Applied PLASTIC v0.521 and explicit 64-token tiny artifact fixture.")
# ^^^ THOG
