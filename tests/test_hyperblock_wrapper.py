# vvv THOG
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_hyperblock_wrapper_dry_run_propagates_fixed_field_controls() -> None:
    environment = dict(os.environ)
    environment["THOG2_PYTHON"] = sys.executable
    result = subprocess.run(
        [
            "bash",
            "train_OWT.sh",
            "--hyperblock",
            "--hyperblock-compressor",
            "chebyshev",
            "--hyperblock-common-family-order",
            "3",
            "--hyperblock-attention-family-order",
            "2",
            "--hyperblock-mlp-family-order",
            "1",
            "--hyperblock-depth-order",
            "2",
            "--hyperblock-d-model-order",
            "4",
            "--hyperblock-mlp-hidden-order",
            "4",
            "--hyperblock-attention-head-order",
            "2",
            "--hyperblock-attention-head-channel-order",
            "4",
            "--hyperblock-loop-count",
            "3",
            "--hyperblock-loop-decay",
            "0.8",
            "-g",
            "HB_SMOKE",
            "-n",
            "2",
            "-w",
            "0",
            "-b",
            "1",
            "-A",
            "1",
            "-u",
            "1",
            "-e",
            "1",
            "-l",
            "1",
            "-k",
            "0",
            "-L",
            "2",
            "-H",
            "2",
            "-D",
            "8",
            "-C",
            "8",
            "-S",
            "1",
            "-I",
            "none",
            "-F",
            "none",
            "-T",
            "float32",
            "-K",
            "math",
            "-x",
            "true",
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "hyperblock / hyperblock / chebyshev" in result.stdout
    assert "HFC3 HFA2 HFM1 HL2 HD4 HM4 HH2 HC4 loops=3 decay=0.8" in result.stdout
    assert "HB_chebyshev" in result.stdout
    assert "--hyperblock" in result.stdout
    assert "--hyperblock-common-family-order 3" in result.stdout
    assert "--hyperblock-attention-family-order 2" in result.stdout
    assert "--hyperblock-mlp-family-order 1" in result.stdout
    assert "--hyperblock-depth-order 2" in result.stdout
    assert "--hyperblock-loop-count 3" in result.stdout
    assert "--hyperblock-loop-decay 0.8" in result.stdout
    assert "depth curves:" in result.stdout and "none" in result.stdout
    assert "depth viewer:" not in result.stdout
    assert "sample elements:" not in result.stdout
    assert "--geometry-preset" not in result.stdout.split("DRY RUN:", 1)[1]


def test_hyperblock_wrapper_help_and_bash_syntax() -> None:
    subprocess.run(["bash", "-n", "train_OWT.sh"], cwd=ROOT, check=True)
    subprocess.run(["bash", "-n", "train_OWT_core.sh"], cwd=ROOT, check=True)
    result = subprocess.run(
        ["bash", "train_OWT.sh", "-h"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "HYPERBLOCK:" in result.stdout
    assert "--hyperblock-common-family-order" in result.stdout
    assert "--hyperblock-attention-head-channel-order" in result.stdout
    assert "--hyperblock-loop-count" in result.stdout
    assert "--hyperblock-loop-decay" in result.stdout
# ^^^ THOG
