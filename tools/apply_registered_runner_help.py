#!/usr/bin/env python3
# vvv THOG
"""Append argparse-owned complete hyperparameter help to the registry surface."""

from pathlib import Path


path = Path(__file__).resolve().parents[1] / "train_OWT.sh"
content = path.read_text(encoding="utf-8")
marker = "registered runner hyperparameters"
if marker not in content:
    anchor = "      bash ./train_OWT_core.sh -h\n"
    if content.count(anchor) != 1:
        raise RuntimeError(
            "train_OWT.sh must contain exactly one canonical core-help invocation"
        )
    addition = (
        "      printf '\\nregistered runner hyperparameters\\n"
        "---------------------------------\\n'\n"
        "      \"$THOG2_REGISTRY_PYTHON\" -c 'from run_thog2_owt_core import "
        "build_parser; print(build_parser().format_help(), end=\"\")'\n"
    )
    content = content.replace(anchor, anchor + addition, 1)
    path.write_text(content, encoding="utf-8")
# ^^^ THOG
