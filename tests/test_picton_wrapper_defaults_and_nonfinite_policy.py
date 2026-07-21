# vvv THOG
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from run_thog2_owt import build_parser
from sheet.run_config import OwtRunConfig
from sheet.training_config import TrainingConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRAINING_WRAPPERS = (
    REPOSITORY_ROOT / "current_scruffy_train_OWT.sh",
    REPOSITORY_ROOT / "current_dreedle_train_OWT.sh",
)


def _run_wrapper(
    wrapper: Path,
    python_bin: Path,
    *extra_arguments: str,
) -> subprocess.CompletedProcess[str]:
    command = [
        "bash", str(wrapper), "-g", "PICTON_TEST", "-p", "full_block",
        "-n", "2", "-b", "1", "-A", "1", "-u", "1", "-l", "1",
        "-w", "1", "-L", "2", "-H", "1", "-D", "4", "-C", "2",
        "-P", "1", "-Q", "2", "-J", "1", "-O", "1", "-X", "2",
        "-Y", "4", "-S", "1", "-T", "float32", "-K", "math",
        "-I", "none", "-F", "none", "-x", "true", *extra_arguments,
    ]
    return subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env={**os.environ, "THOG2_PYTHON": str(python_bin)},
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )


@pytest.fixture(scope="module")
def fake_python(tmp_path_factory: pytest.TempPathFactory) -> Path:
    executable = tmp_path_factory.mktemp("wrapper_python") / "fake_python"
    executable.write_text(
        """#!/usr/bin/env python3
import json
import sys

if len(sys.argv) >= 2 and sys.argv[1] == '-c':
    code = sys.argv[2]
    if 'basis_artifact_tag_for_family' in code:
        family = sys.argv[3]
        aliases = {
            'cheby': 'chebyshev',
            'chebyshev_first_kind_qr': 'chebyshev',
            'dct_ii': 'dct',
            'dct_ii_orthonormal': 'dct',
            'balanced_haar': 'haar',
            'haar_balanced': 'haar',
        }
        family = aliases.get(family, family)
        print({'chebyshev': 'CHEBY', 'dct': 'DCT', 'haar': 'HAAR'}[family])
        raise SystemExit(0)
    payload = json.load(sys.stdin)
    if '["artifact_name"]' in code:
        print(payload['artifact_name'])
    elif '["paths"]["log_path"]' in code:
        print(payload['paths']['log_path'])
    raise SystemExit(0)
args = sys.argv[1:]
def value(name):
    return args[args.index(name) + 1]
artifact = 'FAKE_' + value('--host-label')
if '--print-resolved-json' in args or '--dry-run' in args:
    print(json.dumps({'artifact_name': artifact, 'paths': {'log_path': f'logs/{artifact}/train.log'}}))
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


@pytest.fixture(scope="module")
def wrapper_outputs(fake_python: Path) -> dict[Path, str]:
    outputs: dict[Path, str] = {}
    for wrapper in TRAINING_WRAPPERS:
        completed = _run_wrapper(wrapper, fake_python, "-c", "1000", "-f", "100")
        assert completed.returncode == 0, completed.stdout + completed.stderr
        outputs[wrapper] = completed.stdout
    return outputs


def test_current_training_wrappers_default_to_fast_discard_eval_100_and_checkpoint_1000(
    wrapper_outputs: dict[Path, str],
) -> None:
    for output in wrapper_outputs.values():
        assert "fast discard:       true" in output
        assert "eval_every=100" in output
        assert "ckpt_every=1000" in output


def test_current_training_wrappers_print_start_time_before_artifact_and_split_shape_from_orders(
    wrapper_outputs: dict[Path, str],
) -> None:
    for output in wrapper_outputs.values():
        start_match = re.search(
            r"^  start time:\s+(\d{2}:\d{2}  \d{2}-\d{2}-\d{2})$",
            output,
            re.MULTILINE,
        )
        assert start_match is not None, output
        start_index = output.index("  start time:")
        artifact_index = output.index("  artifact:")
        shape_line = next(
            line for line in output.splitlines() if line.startswith("  shape:")
        )
        orders_line = next(
            line for line in output.splitlines() if line.startswith("  orders:")
        )
        assert start_index < artifact_index
        assert shape_line.rstrip().endswith("L2 H1 D4 C2")
        assert orders_line.rstrip().endswith("P1 Q2 J1 O1 X2 Y4")
        assert " P1" not in shape_line


def test_current_training_wrappers_remove_retired_direct_thog_mlp_application_name() -> None:
    for wrapper in TRAINING_WRAPPERS:
        source = wrapper.read_text(encoding="utf-8")
        assert "DIRECT_THOG_MLP_APPLICATION" not in source


def test_current_training_wrappers_accept_lr_code_1000_and_min_lr_code_100(
    wrapper_outputs: dict[Path, str],
) -> None:
    for output in wrapper_outputs.values():
        assert "1000e-5" in output
        assert "100e-5" in output


def test_current_training_wrappers_reject_lr_codes_above_the_new_limits(
    fake_python: Path,
) -> None:
    for wrapper in TRAINING_WRAPPERS:
        bad_learning_rate = _run_wrapper(wrapper, fake_python, "-c", "1001")
        bad_minimum = _run_wrapper(wrapper, fake_python, "-f", "101")
        assert bad_learning_rate.returncode == 2
        assert "expected 1..1000" in bad_learning_rate.stderr
        assert bad_minimum.returncode == 2
        assert "expected 1..100" in bad_minimum.stderr


def test_core_nonfinite_recovery_defaults_to_bounded_skip_across_public_config_layers() -> None:
    parser_arguments = build_parser().parse_args(["--model-type", "dense"])
    run_config = OwtRunConfig(model_type="dense")
    training_config = TrainingConfig()
    assert parser_arguments.nonfinite_update_policy == "skip"
    assert parser_arguments.max_nonfinite_update_skips == 10
    assert run_config.nonfinite_update_policy == "skip"
    assert run_config.max_nonfinite_update_skips == 10
    assert training_config.nonfinite_update_policy == "skip"
    assert training_config.max_nonfinite_update_skips == 10


def test_explicit_fail_fast_nonfinite_policy_remains_available() -> None:
    parser_arguments = build_parser().parse_args(
        ["--model-type", "dense", "--nonfinite-update-policy", "raise"]
    )
    assert parser_arguments.nonfinite_update_policy == "raise"
    assert OwtRunConfig(
        model_type="dense",
        nonfinite_update_policy="raise",
    ).nonfinite_update_policy == "raise"
    assert TrainingConfig(
        nonfinite_update_policy="raise"
    ).nonfinite_update_policy == "raise"
# ^^^ THOG
