# vvv THOG
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = (
    REPOSITORY_ROOT / "current_scruffy_train_OWT.sh",
    REPOSITORY_ROOT / "current_dreedle_train_OWT.sh",
)


@pytest.fixture()
def fake_python(tmp_path: Path) -> Path:
    executable = tmp_path / "fake_python"
    executable.write_text(
        """#!/usr/bin/env python3
import json
import sys

aliases = {
    'cheby': 'chebyshev',
    'chebyshev_first_kind_qr': 'chebyshev',
    'dct_ii': 'dct',
    'dct_ii_orthonormal': 'dct',
    'balanced_haar': 'haar',
    'haar_balanced': 'haar',
}
tags = {'chebyshev': 'CHEBY', 'dct': 'DCT', 'haar': 'HAAR'}

if len(sys.argv) >= 2 and sys.argv[1] == '-c':
    code = sys.argv[2]
    if 'basis_artifact_tag_for_family' in code:
        family = aliases.get(sys.argv[3], sys.argv[3])
        print(f'{family}\\t{tags[family]}')
        raise SystemExit(0)
    payload = json.load(sys.stdin)
    if '[\"artifact_name\"]' in code:
        print(payload['artifact_name'])
    elif '[\"paths\"][\"log_path\"]' in code:
        print(payload['paths']['log_path'])
    raise SystemExit(0)

args = sys.argv[1:]
def value(name):
    return args[args.index(name) + 1]
family = value('--basis-family') if '--basis-family' in args else 'dense'
order = value('--o-depth')
tag = tags.get(family, 'DENSE')
artifact = f'FAKE_{tag}_{family}_P_{order}'
if '--print-resolved-json' in args or '--dry-run' in args:
    print(json.dumps({'artifact_name': artifact, 'paths': {'log_path': f'logs/{artifact}/train.log'}}))
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def run_wrapper(wrapper: Path, fake_python: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    command = [
        "bash", str(wrapper), "-x", "true", "-g", "BASIS_GRID_TEST",
        "-p", "depth", "-B", "chebyshev,haar,dct", "-P", "1,2",
        "-n", "2", "-w", "1", "-b", "1", "-A", "1", "-u", "1",
        "-e", "1", "-l", "1", "-L", "2", "-H", "1", "-D", "4",
        "-C", "2", "-Q", "2", "-J", "1", "-O", "1", "-X", "2",
        "-Y", "4", "-S", "1", "-T", "float32", "-K", "math",
        "-I", "none", "-F", "none", *arguments,
    ]
    return subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env={**os.environ, "THOG2_PYTHON": str(fake_python)},
        text=True,
        capture_output=True,
        check=False,
        timeout=90,
    )


def test_basis_family_grid_runs_cartesian_product_with_depth_order(fake_python: Path) -> None:
    for wrapper in WRAPPERS:
        completed = run_wrapper(wrapper, fake_python)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        output = completed.stdout
        assert output.count("OWT train") == 6
        for family, tag in (("chebyshev", "CHEBY"), ("haar", "HAAR"), ("dct", "DCT")):
            assert output.count(f" / depth / {family}") == 2
            for order in (1, 2):
                assert f"FAKE_{tag}_{family}_P_{order}" in output


def test_basis_family_grid_canonicalizes_aliases(fake_python: Path) -> None:
    wrapper = REPOSITORY_ROOT / "current_scruffy_train_OWT.sh"
    completed = run_wrapper(wrapper, fake_python, "-B", "dct_ii,balanced_haar", "-P", "1")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.count("OWT train") == 2
    assert " / depth / dct" in completed.stdout
    assert " / depth / haar" in completed.stdout
    assert " / depth / dct_ii" not in completed.stdout
    assert " / depth / balanced_haar" not in completed.stdout


def test_multiple_basis_families_require_auto_version(fake_python: Path) -> None:
    wrapper = REPOSITORY_ROOT / "current_scruffy_train_OWT.sh"
    completed = run_wrapper(wrapper, fake_python, "-v", "dct_ii_orthonormal_v1")
    assert completed.returncode == 2
    assert "must be auto" in completed.stderr


def test_dense_preset_is_not_multiplied_by_basis_family_grid(fake_python: Path) -> None:
    wrapper = REPOSITORY_ROOT / "current_scruffy_train_OWT.sh"
    completed = run_wrapper(wrapper, fake_python, "-p", "dense", "-P", "1")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.count("OWT train") == 1


# ^^^ THOG
