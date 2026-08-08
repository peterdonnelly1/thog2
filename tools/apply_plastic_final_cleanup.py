#!/usr/bin/env python3
# vvv THOG
"""Align inline THOG markers after the PLASTIC refinement."""

from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    marker_files = (
        "sheet/__init__.py",
        "run_thog2_owt_core.py",
        "sheet/run_config.py",
        "sheet/training_config.py",
        "sheet/plastic_depth_coarse.py",
        "sheet/plastic_depth_coarse_runner.py",
        "sheet/plastic_depth_lifecycle.py",
        "sheet/plastic_depth_console_minor_patch.py",
        "sheet/plastic_depth_coarse_runtime_recovery_patch.py",
    )
    subprocess.run(
        ("python", "tools/align_thog_inline_markers.py", *marker_files),
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
# ^^^ THOG
