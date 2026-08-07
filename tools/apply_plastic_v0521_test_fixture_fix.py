#!/usr/bin/env python3
# vvv THOG
"""Apply PLASTIC v0.521 and make pre-existing tiny test fixtures explicit about legal probe-token samples."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

subprocess.run(
    ["python", "tools/apply_plastic_v0521_probe_sampling_and_console.py"],
    cwd=ROOT,
    check=True,
)

# vvv THOG the artifact-name fixture has 2 x 32 = 64 token positions and should not inherit the production 1024-token default
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
# ^^^ THOG

# vvv THOG shared PLASTIC unit-test training fixtures use the stage3 2 x 8 = 16 token microbatch unless a test overrides it
path = ROOT / "tests/test_plastic_depth.py"
content = path.read_text(encoding="utf-8")
addition = "        plastic__layer_count_probe__number_of_sampled_valid_tokens=16,\n"
if addition not in content:
    anchor = "        plastic__freeze_geometry_during_warmup=False,\n        depth_order=3,\n"
    if content.count(anchor) != 1:
        raise RuntimeError(
            "expected exactly one shared plastic_training_config fixture anchor; "
            f"found {content.count(anchor)}"
        )
    content = content.replace(
        anchor,
        "        plastic__freeze_geometry_during_warmup=False,\n" + addition + "        depth_order=3,\n",
        1,
    )
    path.write_text(content, encoding="utf-8")
# ^^^ THOG

print("Applied PLASTIC v0.521 and explicit legal probe-token sizes for tiny test fixtures.")
# ^^^ THOG
