# vvv THOG
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from sheet.run_config import OwtRunConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = ("current_scruffy_train_OWT.sh", "current_dreedle_train_OWT.sh")


def test_run_config_artifact_contains_two_digit_learning_rate_code() -> None:
    config = OwtRunConfig(model_type="sheet", learning_rate=7.0e-4, min_lr=7.0e-5)
    assert "_LR_70_" in f"_{config.parameter_artifact_fragment()}_"
    assert "_LR_70_" in f"_{config.artifact_name}_"


def test_run_config_default_learning_rate_uses_lr_60_artifact_code() -> None:
    config = OwtRunConfig(model_type="sheet")
    assert "_LR_60_" in f"_{config.parameter_artifact_fragment()}_"


def test_wrappers_expose_lr_and_min_lr_codes_and_nest_lr_and_batch_grid_axes() -> None:
    for wrapper_name in WRAPPERS:
        source = (REPOSITORY_ROOT / wrapper_name).read_text(encoding="utf-8")
        assert 'LEARNING_RATE_CODES="60"' in source
        assert 'MIN_LR_CODE="06"' in source
        assert '-c LR_CODES=${LEARNING_RATE_CODES}' in source
        assert '-f MIN_LR_CODE=${MIN_LR_CODE}' in source
        assert 'for batch_size_value in "${BATCH_SIZE_VALUES[@]}"; do' in source
        assert 'for learning_rate_code in "${LEARNING_RATE_CODE_VALUES[@]}"; do' in source
        assert '--learning-rate "$learning_rate_value" --min-lr "$min_lr_value"' in source
        assert 'learning_rate_value="${learning_rate_code}e-5"' in source
        assert 'min_lr_value="$((10#$MIN_LR_CODE))e-5"' in source


def test_scruffy_dry_run_expands_cartesian_product_of_batch_and_learning_rate_values(tmp_path: Path) -> None:
    fake_python = tmp_path / "fake_python"
    fake_python.write_text(
        """#!/usr/bin/env python3
import json
import sys

if len(sys.argv) >= 2 and sys.argv[1] == '-c':
    payload = json.load(sys.stdin)
    code = sys.argv[2]
    if '[\"artifact_name\"]' in code:
        print(payload['artifact_name'])
    elif '[\"paths\"][\"log_path\"]' in code:
        print(payload['paths']['log_path'])
    raise SystemExit(0)
args = sys.argv[1:]
def value(name):
    return args[args.index(name) + 1]
batch = value('--batch-size')
lr = float(value('--learning-rate'))
lr_code = round(lr / 1.0e-5)
artifact = f'FAKE__b_{batch}_LR_{lr_code:02d}_x'
if '--print-resolved-json' in args or '--dry-run' in args:
    print(json.dumps({'artifact_name': artifact, 'paths': {'log_path': f'logs/{artifact}/train.log'}}))
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = dict(os.environ)
    environment["THOG2_PYTHON"] = str(fake_python)
    completed = subprocess.run(
        [
            "bash", "current_scruffy_train_OWT.sh",
            "-x", "true", "-p", "depth", "-P", "2",
            "-b", "2,4", "-c", "60,70", "-f", "06",
            "-n", "2", "-w", "1", "-L", "4", "-H", "2", "-D", "16", "-C", "8",
            "-Q", "8", "-J", "4", "-O", "4", "-X", "8", "-Y", "20",
            "-A", "1", "-u", "1", "-e", "1", "-I", "none", "-F", "none",
        ],
        cwd=REPOSITORY_ROOT, env=environment, text=True, capture_output=True, check=False, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert completed.stdout.count("scruffy OWT train") == 4
    for batch_size in (2, 4):
        for lr_code in (60, 70):
            assert f"b_{batch_size}_LR_{lr_code:02d}_" in completed.stdout
            assert f"lr_code={lr_code}" in completed.stdout
# ^^^ THOG
