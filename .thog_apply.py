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


HELP_OLD = """  -B BASIS_FAMILY=${BASIS_FAMILY}                   canonical: chebyshev | dct | haar
                                                    Chebyshev aliases: cheby | chebyshev_first_kind_qr
                                                    DCT aliases: dct_ii | dct_ii_orthonormal
                                                    Haar aliases: balanced_haar | haar_balanced
"""
HELP_NEW = """  -B BASIS_FAMILY=${BASIS_FAMILY}                   canonical: chebyshev | dct | haar; single, comma, or quoted space list
                                                    Chebyshev aliases: cheby | chebyshev_first_kind_qr
                                                    DCT aliases: dct_ii | dct_ii_orthonormal
                                                    Haar aliases: balanced_haar | haar_balanced
"""

ARRAYS_OLD = """O_DEPTH_VALUES=()
PRESET_VALUES=()
BATCH_SIZE_VALUES=()"""
ARRAYS_NEW = """O_DEPTH_VALUES=()
PRESET_VALUES=()
BASIS_FAMILY_VALUES=()                                                                                                                                    # <<< THOG basis-family grid axis
BATCH_SIZE_VALUES=()"""

PARSE_INSERT_OLD = """parse_geometry_preset_values() {
  local normalized="${1//,/ }"
  local value
  for value in $normalized; do
    case "$value" in
      dense) PRESET_VALUES+=("$value"); HAS_DENSE_PRESET=true ;;
      legacy_sheet_col|depth|head_aware_block|mlp_block|full_block) PRESET_VALUES+=("$value"); HAS_COMPACT_PRESET=true ;;
      *) echo "Bad PRESET: $value" >&2; exit 2 ;;
    esac
  done
  (( ${#PRESET_VALUES[@]} > 0 )) || { echo "Invalid PRESET: empty value list." >&2; exit 2; }
}
parse_o_depth_values "$O_DEPTH"
parse_geometry_preset_values "$GEOMETRY_PRESET"
"""
PARSE_INSERT_NEW = """parse_geometry_preset_values() {
  local normalized="${1//,/ }"
  local value
  for value in $normalized; do
    case "$value" in
      dense) PRESET_VALUES+=("$value"); HAS_DENSE_PRESET=true ;;
      legacy_sheet_col|depth|head_aware_block|mlp_block|full_block) PRESET_VALUES+=("$value"); HAS_COMPACT_PRESET=true ;;
      *) echo "Bad PRESET: $value" >&2; exit 2 ;;
    esac
  done
  (( ${#PRESET_VALUES[@]} > 0 )) || { echo "Invalid PRESET: empty value list." >&2; exit 2; }
}
parse_basis_family_values() {
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
"""

VALIDATION_OLD = """# vvv THOG
[[ "$BASIS_FAMILY" =~ ^[a-z][a-z0-9_]*$ ]] || { echo "BASIS_FAMILY must be a lowercase registry name or alias." >&2; exit 2; }
# ^^^ THOG
"""
VALIDATION_NEW = """# vvv THOG basis-family grid validation
if (( ${#BASIS_FAMILY_VALUES[@]} > 1 )) && [[ "$BASIS_VERSION" != auto ]]; then
  echo "BASIS_VERSION must be auto when BASIS_FAMILY contains multiple values." >&2
  exit 2
fi
# ^^^ THOG
"""

GLOBAL_TAG_OLD = """# vvv THOG
BASIS_TAG="$("$PYTHON_BIN" -c 'import sys; from sheet.bases import basis_artifact_tag_for_family; print(basis_artifact_tag_for_family(sys.argv[1]))' "$BASIS_FAMILY")"
# ^^^ THOG
"""
GLOBAL_TAG_NEW = """# vvv THOG basis family and artifact tag now resolve independently for each basis-family grid point
# BASIS_TAG="$("$PYTHON_BIN" -c 'import sys; from sheet.bases import basis_artifact_tag_for_family; print(basis_artifact_tag_for_family(sys.argv[1]))' "$BASIS_FAMILY")"
# ^^^ THOG
"""

FUNCTION_HEAD_OLD = """run_preset_o_depth_batch_lr() {
  local geometry_preset_value="$1"
  local o_depth_value="$2"
  local batch_size_value="$3""""
FUNCTION_HEAD_NEW = """run_grid_point() {
  local geometry_preset_value="$1"
  local o_depth_value="$2"
  local batch_size_value="$3""""

FUNCTION_ARGS_OLD = """  local learning_rate_code="$4""""
FUNCTION_ARGS_NEW = """  local learning_rate_code="$4"
  local requested_basis_family="$5"                                                                                                                         # <<< THOG basis-family grid coordinate"""

LOCALS_OLD = """  local run_model_type display_model_type preset_tag run_tag run_name_value LOG_TIMESTAMP resolved_json artifact_name log_path depth_curve_local_root
  local residual_init_depth_source_value n_layer_value n_head_value n_embd_value shape_summary orders_summary start_time_friendly log_url viewer_url serve_url run_status
"""
LOCALS_NEW = """  local run_model_type display_model_type preset_tag run_tag run_name_value LOG_TIMESTAMP resolved_json artifact_name log_path depth_curve_local_root
  local residual_init_depth_source_value n_layer_value n_head_value n_embd_value shape_summary orders_summary start_time_friendly log_url viewer_url serve_url run_status
  local basis_resolution basis_family_value basis_tag                                                                                                        # <<< THOG resolved basis-family grid identity
"""

DENSE_ELSE_OLD = """  if [[ "$geometry_preset_value" == dense ]]; then
    run_model_type="dense"; display_model_type="dense"; preset_tag="DENSE"; run_tag="DENSE"
"""
DENSE_ELSE_NEW = """  if [[ "$geometry_preset_value" == dense ]]; then
    run_model_type="dense"; display_model_type="dense"; preset_tag="DENSE"; run_tag="DENSE"; basis_family_value="n/a"
"""

COMPACT_OLD = """  else
    run_model_type="sheet"; display_model_type="spectral"; preset_tag="${geometry_preset_value^^}"
    [[ "$geometry_preset_value" == legacy_sheet_col ]] && preset_tag="SHEET_COL"
    run_tag="${BASIS_TAG}_${preset_tag}"
    compact_args=(--geometry-preset "$geometry_preset_value" --basis-family "$BASIS_FAMILY" --basis-version "$BASIS_VERSION")
"""
COMPACT_NEW = """  else
    run_model_type="sheet"; display_model_type="spectral"; preset_tag="${geometry_preset_value^^}"
    [[ "$geometry_preset_value" == legacy_sheet_col ]] && preset_tag="SHEET_COL"
    basis_resolution="$("$PYTHON_BIN" -c 'import sys; from sheet.bases import basis_artifact_tag_for_family, normalize_registered_basis_family; family = normalize_registered_basis_family(sys.argv[1]); print(f"{family}\\t{basis_artifact_tag_for_family(family)}")' "$requested_basis_family")"
    IFS=$'\\t' read -r basis_family_value basis_tag <<< "$basis_resolution"
    run_tag="${basis_tag}_${preset_tag}"
    compact_args=(--geometry-preset "$geometry_preset_value" --basis-family "$basis_family_value" --basis-version "$BASIS_VERSION")
"""

DISPLAY_OLD = """  model/preset/basis: $display_model_type / $geometry_preset_value / $BASIS_FAMILY
"""
DISPLAY_NEW = """  model/preset/basis: $display_model_type / $geometry_preset_value / $basis_family_value
"""

LOOPS_OLD = """if (( ${#PRESET_VALUES[@]} > 1 || ${#O_DEPTH_VALUES[@]} > 1 || ${#BATCH_SIZE_VALUES[@]} > 1 || ${#LEARNING_RATE_CODE_VALUES[@]} > 1 )); then
  echo "HOST_LABEL_PLACEHOLDER OWT grid: p=${PRESET_VALUES[*]} P=${O_DEPTH_VALUES[*]} b=${BATCH_SIZE_VALUES[*]} LR=${LEARNING_RATE_CODE_VALUES[*]}"
fi
for geometry_preset_value in "${PRESET_VALUES[@]}"; do
  for batch_size_value in "${BATCH_SIZE_VALUES[@]}"; do
    for learning_rate_code in "${LEARNING_RATE_CODE_VALUES[@]}"; do
      if [[ "$geometry_preset_value" == dense ]]; then
        run_preset_o_depth_batch_lr "$geometry_preset_value" "${O_DEPTH_VALUES[0]}" "$batch_size_value" "$learning_rate_code"
      else
        for o_depth_value in "${O_DEPTH_VALUES[@]}"; do run_preset_o_depth_batch_lr "$geometry_preset_value" "$o_depth_value" "$batch_size_value" "$learning_rate_code"; done
      fi
    done
  done
done
"""
LOOPS_NEW = """if (( ${#PRESET_VALUES[@]} > 1 || ${#BASIS_FAMILY_VALUES[@]} > 1 || ${#O_DEPTH_VALUES[@]} > 1 || ${#BATCH_SIZE_VALUES[@]} > 1 || ${#LEARNING_RATE_CODE_VALUES[@]} > 1 )); then
  echo "HOST_LABEL_PLACEHOLDER OWT grid: p=${PRESET_VALUES[*]} B=${BASIS_FAMILY_VALUES[*]} P=${O_DEPTH_VALUES[*]} b=${BATCH_SIZE_VALUES[*]} LR=${LEARNING_RATE_CODE_VALUES[*]}"
fi
for geometry_preset_value in "${PRESET_VALUES[@]}"; do
  if [[ "$geometry_preset_value" == dense ]]; then
    for batch_size_value in "${BATCH_SIZE_VALUES[@]}"; do
      for learning_rate_code in "${LEARNING_RATE_CODE_VALUES[@]}"; do
        run_grid_point "$geometry_preset_value" "${O_DEPTH_VALUES[0]}" "$batch_size_value" "$learning_rate_code" "${BASIS_FAMILY_VALUES[0]}"
      done
    done
  else
    for requested_basis_family in "${BASIS_FAMILY_VALUES[@]}"; do
      for batch_size_value in "${BATCH_SIZE_VALUES[@]}"; do
        for learning_rate_code in "${LEARNING_RATE_CODE_VALUES[@]}"; do
          for o_depth_value in "${O_DEPTH_VALUES[@]}"; do
            run_grid_point "$geometry_preset_value" "$o_depth_value" "$batch_size_value" "$learning_rate_code" "$requested_basis_family"
          done
        done
      done
    done
  fi
done
"""

FAKE_OLD = """    if 'basis_artifact_tag_for_family' in code:
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
"""
FAKE_NEW = """    if 'basis_artifact_tag_for_family' in code:
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
"""

TEST_CONTENT = '''# vvv THOG
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


def update_wrapper(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = replace_once(text, HELP_OLD, HELP_NEW, label=f"{path}: help")
    text = replace_once(text, ARRAYS_OLD, ARRAYS_NEW, label=f"{path}: arrays")
    text = replace_once(text, PARSE_INSERT_OLD, PARSE_INSERT_NEW, label=f"{path}: parsers")
    text = replace_once(text, VALIDATION_OLD, VALIDATION_NEW, label=f"{path}: validation")
    text = replace_once(text, GLOBAL_TAG_OLD, GLOBAL_TAG_NEW, label=f"{path}: global tag")
    text = replace_once(text, FUNCTION_HEAD_OLD, FUNCTION_HEAD_NEW, label=f"{path}: function head")
    text = replace_once(text, FUNCTION_ARGS_OLD, FUNCTION_ARGS_NEW, label=f"{path}: function args")
    text = replace_once(text, LOCALS_OLD, LOCALS_NEW, label=f"{path}: locals")
    text = replace_once(text, DENSE_ELSE_OLD, DENSE_ELSE_NEW, label=f"{path}: dense")
    text = replace_once(text, COMPACT_OLD, COMPACT_NEW, label=f"{path}: compact")
    text = replace_once(text, DISPLAY_OLD, DISPLAY_NEW, label=f"{path}: display")

    host_label = "scruffy" if "scruffy" in path.name else "dreedle"
    loops_old = LOOPS_OLD.replace("HOST_LABEL_PLACEHOLDER", host_label)
    loops_new = LOOPS_NEW.replace("HOST_LABEL_PLACEHOLDER", host_label)
    text = replace_once(text, loops_old, loops_new, label=f"{path}: loops")
    path.write_text(text, encoding="utf-8")


def main() -> None:
    for wrapper in WRAPPERS:
        update_wrapper(wrapper)

    fake_updates = 0
    for path in (
        Path("tests/test_picton_wrapper_defaults_and_nonfinite_policy.py"),
        Path("tests/test_wrapper_learning_rate_and_batch_grids.py"),
    ):
        text = path.read_text(encoding="utf-8")
        text = replace_once(text, FAKE_OLD, FAKE_NEW, label=f"{path}: fake basis resolver")
        path.write_text(text, encoding="utf-8")
        fake_updates += 1

    if fake_updates != 2:
        raise RuntimeError(f"expected two fake-Python updates; got {fake_updates}")

    Path("tests/test_wrapper_basis_family_grid.py").write_text(TEST_CONTENT, encoding="utf-8")


if __name__ == "__main__":
    main()
