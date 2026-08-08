#!/usr/bin/env python3
# vvv THOG
"""Finish the one remaining startup-label expectation after the v0.52 cleanup transform."""

from pathlib import Path


path = Path(__file__).resolve().parents[1] / "tests/test_run_thog2_owt_startup_report.py"
text = path.read_text(encoding="utf-8")
text = text.replace(
    '    initial_row = next(line for line in rows if "initial layer indices:" in line)\n',
    '    initial_row = next(line for line in rows if "active sample_layer:" in line)\n',
    1,
)
path.write_text(text, encoding="utf-8")
# ^^^ THOG
