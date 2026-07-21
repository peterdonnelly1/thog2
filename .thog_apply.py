from __future__ import annotations

from pathlib import Path


WRAPPERS = (
    Path("current_scruffy_train_OWT.sh"),
    Path("current_dreedle_train_OWT.sh"),
)


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match; found {count}")
    return text.replace(old, new, 1)


def update_wrapper(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    text = replace_once(
        text,
        "  -B BASIS_FAMILY=${BASIS_FAMILY}                   canonical: chebyshev | dct | haar\n",
        "  -B BASIS_FAMILY=${BASIS_FAMILY}                   canonical: chebyshev | dct | haar; single, comma, or quoted space list\n",
        label=f"{path}: basis help",
    )

    text = replace_once(
        text,
        "O_DEPTH_VALUES=()\nPRESET_VALUES=()\nBATCH_SIZE_VALUES=()",
        "O_DEPTH_VALUES=()\nPRESET_VALUES=()\nBASIS_FAMILY_VALUES=()                                                                                                                                    # <<< THOG basis-family grid axis\nBASIS_TAG_VALUES=()                                                                                                                                       # <<< THOG matching artifact tags for basis-family grid\nBATCH_SIZE_VALUES=()",
        label=f"{path}: grid arrays",
    )

    parser_calls = '''parse_o_depth_values "$O_DEPTH"
parse_geometry_preset_values "$GEOMETRY_PRESET"
'''
    parser_calls_with_basis = '''parse_basis_family_values() {
  local normalized="${1//,/ }" value
  for value in $normalized; do
    [[ "$value" =~ ^[a-z][a-z0-9_]*$ ]] || { echo "Invalid BASIS_FAMILY value: $value; expected a lowercase registry name or alias." >&2; exit 2; }
    BASIS_FAMILY_VALUES+=("$value")
  done
  (( ${#BASIS_FAMILY_VALUES[@]} > 0 )) || { echo "Invalid BASIS_FAMILY: empty value list." >&2; exit 2; }
}
parse_o_depth_values "$O_DEPTH"
parse_geometry_preset_values "$GEOMETRY_PRESET"
parse_basis_family_values "$BASIS_FAMILY"                                                                                                                  # <<< THOG parse basis-family grid
'''
    text = replace_once(
        text,
        parser_calls,
        parser_calls_with_basis,
        label=f"{path}: basis parser",
    )

    text = replace_once(
        text,
        '''# vvv THOG
[[ "$BASIS_FAMILY" =~ ^[a-z][a-z0-9_]*$ ]] || { echo "BASIS_FAMILY must be a lowercase registry name or alias." >&2; exit 2; }
# ^^^ THOG
''',
        '''# vvv THOG basis-family grid validation
if (( ${#BASIS_FAMILY_VALUES[@]} > 1 )) && [[ "$BASIS_VERSION" != auto ]]; then
  echo "BASIS_VERSION must be auto when BASIS_FAMILY contains multiple values." >&2
  exit 2
fi
# ^^^ THOG
''',
        label=f"{path}: basis validation",
    )

    text = replace_once(
        text,
        '''# vvv THOG
BASIS_TAG="$("$PYTHON_BIN" -c 'import sys; from sheet.bases import basis_artifact_tag_for_family; print(basis_artifact_tag_for_family(sys.argv[1]))' "$BASIS_FAMILY")"
# ^^^ THOG
''',
        '''# vvv THOG canonicalize and registry-validate every basis-family grid value before any run starts
# BASIS_TAG="$("$PYTHON_BIN" -c 'import sys; from sheet.bases import basis_artifact_tag_for_family; print(basis_artifact_tag_for_family(sys.argv[1]))' "$BASIS_FAMILY")"
BASIS_FAMILY_CANONICAL_VALUES=()
for requested_basis_family in "${BASIS_FAMILY_VALUES[@]}"; do
  basis_resolution="$("$PYTHON_BIN" -c 'import sys; from sheet.bases import basis_artifact_tag_for_family, normalize_registered_basis_family; family = normalize_registered_basis_family(sys.argv[1]); print(f"{family}\\t{basis_artifact_tag_for_family(family)}")' "$requested_basis_family")"
  IFS=$'\\t' read -r basis_family_value basis_tag <<< "$basis_resolution"
  BASIS_FAMILY_CANONICAL_VALUES+=("$basis_family_value")
  BASIS_TAG_VALUES+=("$basis_tag")
done
BASIS_FAMILY_VALUES=("${BASIS_FAMILY_CANONICAL_VALUES[@]}")
# ^^^ THOG
''',
        label=f"{path}: basis registry resolution",
    )

    text = replace_once(
        text,
        '''run_preset_o_depth_batch_lr() {
  local geometry_preset_value="$1"
  local o_depth_value="$2"
  local batch_size_value="$3"                                                                                                                             # <<< THOG batch grid coordinate
  local learning_rate_code="$4"                                                                                                                           # <<< THOG LR grid coordinate
''',
        '''run_grid_point() {
  local geometry_preset_value="$1"
  local o_depth_value="$2"
  local batch_size_value="$3"                                                                                                                             # <<< THOG batch grid coordinate
  local learning_rate_code="$4"                                                                                                                           # <<< THOG LR grid coordinate
  local basis_family_value="$5"                                                                                                                           # <<< THOG canonical basis-family grid coordinate
  local basis_tag="$6"                                                                                                                                    # <<< THOG matching basis artifact tag
''',
        label=f"{path}: grid function arguments",
    )

    text = replace_once(
        text,
        '    run_model_type="dense"; display_model_type="dense"; preset_tag="DENSE"; run_tag="DENSE"\n',
        '    run_model_type="dense"; display_model_type="dense"; preset_tag="DENSE"; run_tag="DENSE"; basis_family_value="n/a"\n',
        label=f"{path}: dense basis display",
    )

    text = replace_once(
        text,
        '''    run_tag="${BASIS_TAG}_${preset_tag}"
    compact_args=(--geometry-preset "$geometry_preset_value" --basis-family "$BASIS_FAMILY" --basis-version "$BASIS_VERSION")
''',
        '''    run_tag="${basis_tag}_${preset_tag}"
    compact_args=(--geometry-preset "$geometry_preset_value" --basis-family "$basis_family_value" --basis-version "$BASIS_VERSION")
''',
        label=f"{path}: compact basis identity",
    )

    text = replace_once(
        text,
        '  model/preset/basis: $display_model_type / $geometry_preset_value / $BASIS_FAMILY\n',
        '  model/preset/basis: $display_model_type / $geometry_preset_value / $basis_family_value\n',
        label=f"{path}: basis display",
    )

    loop_start = text.index(
        'if (( ${#PRESET_VALUES[@]} > 1 || ${#O_DEPTH_VALUES[@]} > 1 || ${#BATCH_SIZE_VALUES[@]} > 1 || ${#LEARNING_RATE_CODE_VALUES[@]} > 1 )); then\n'
    )
    loop_end = text.rindex("# ^^^ THOG")
    host_label = "scruffy" if "scruffy" in path.name else "dreedle"
    new_loops = f'''if (( ${{#PRESET_VALUES[@]}} > 1 || ${{#BASIS_FAMILY_VALUES[@]}} > 1 || ${{#O_DEPTH_VALUES[@]}} > 1 || ${{#BATCH_SIZE_VALUES[@]}} > 1 || ${{#LEARNING_RATE_CODE_VALUES[@]}} > 1 )); then
  echo "{host_label} OWT grid: p=${{PRESET_VALUES[*]}} B=${{BASIS_FAMILY_VALUES[*]}} P=${{O_DEPTH_VALUES[*]}} b=${{BATCH_SIZE_VALUES[*]}} LR=${{LEARNING_RATE_CODE_VALUES[*]}}"
fi
for geometry_preset_value in "${{PRESET_VALUES[@]}}"; do
  if [[ "$geometry_preset_value" == dense ]]; then
    for batch_size_value in "${{BATCH_SIZE_VALUES[@]}}"; do
      for learning_rate_code in "${{LEARNING_RATE_CODE_VALUES[@]}}"; do
        run_grid_point "$geometry_preset_value" "${{O_DEPTH_VALUES[0]}}" "$batch_size_value" "$learning_rate_code" "${{BASIS_FAMILY_VALUES[0]}}" "${{BASIS_TAG_VALUES[0]}}"
      done
    done
  else
    for basis_index in "${{!BASIS_FAMILY_VALUES[@]}}"; do
      basis_family_value="${{BASIS_FAMILY_VALUES[$basis_index]}}"
      basis_tag="${{BASIS_TAG_VALUES[$basis_index]}}"
      for batch_size_value in "${{BATCH_SIZE_VALUES[@]}}"; do
        for learning_rate_code in "${{LEARNING_RATE_CODE_VALUES[@]}}"; do
          for o_depth_value in "${{O_DEPTH_VALUES[@]}}"; do
            run_grid_point "$geometry_preset_value" "$o_depth_value" "$batch_size_value" "$learning_rate_code" "$basis_family_value" "$basis_tag"
          done
        done
      done
    done
  fi
done
'''
    text = text[:loop_start] + new_loops + text[loop_end:]
    path.write_text(text, encoding="utf-8")


FAKE_OLD = '''    if 'basis_artifact_tag_for_family' in code:
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
'''
FAKE_NEW = '''    if 'basis_artifact_tag_for_family' in code:
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
        tag = {'chebyshev': 'CHEBY', 'dct': 'DCT', 'haar': 'HAAR'}[family]
        if 'normalize_registered_basis_family' in code:
            print(f'{family}\\t{tag}')
        else:
            print(tag)
        raise SystemExit(0)
'''


TEST_CONTENT = r'''# vvv THOG
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
'''


def main() -> None:
    for wrapper in WRAPPERS:
        update_wrapper(wrapper)

    for path in (
        Path("tests/test_picton_wrapper_defaults_and_nonfinite_policy.py"),
        Path("tests/test_wrapper_learning_rate_and_batch_grids.py"),
    ):
        text = path.read_text(encoding="utf-8")
        text = replace_once(text, FAKE_OLD, FAKE_NEW, label=f"{path}: fake basis resolver")
        path.write_text(text, encoding="utf-8")

    Path("tests/test_wrapper_basis_family_grid.py").write_text(TEST_CONTENT, encoding="utf-8")


if __name__ == "__main__":
    main()
