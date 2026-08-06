#!/usr/bin/env python3
# vvv THOG
"""Keep the registry entrypoint as the sole complete parser-help renderer."""

from pathlib import Path


path = Path(__file__).resolve().parents[1] / "train_OWT.sh"
content = path.read_text(encoding="utf-8")

live_block = (
    "      printf '\\nregistered runner hyperparameters\\n"
    "---------------------------------\\n'\n"
    "      \"$THOG2_REGISTRY_PYTHON\" -c 'from run_thog2_owt_core import "
    "build_parser; print(build_parser().format_help(), end=\"\")'\n"
)
commented_block = (
    "      # vvv THOG the direct registry entrypoint already prints complete parser and descriptor help once\n"
    "      # printf '\\nregistered runner hyperparameters\\n"
    "---------------------------------\\n'\n"
    "      # \"$THOG2_REGISTRY_PYTHON\" -c 'from run_thog2_owt_core import "
    "build_parser; print(build_parser().format_help(), end=\"\")'\n"
    "      # ^^^ THOG\n"
)

if live_block in content:
    if content.count(live_block) != 1:
        raise RuntimeError("train_OWT.sh contains multiple live registered-runner help blocks")
    content = content.replace(live_block, commented_block, 1)
elif commented_block not in content:
    raise RuntimeError(
        "train_OWT.sh contains neither the expected live nor preserved registered-runner help block"
    )

path.write_text(content, encoding="utf-8")
# ^^^ THOG
