from __future__ import annotations

from pathlib import Path


WRAPPERS = (
    Path("current_scruffy_train_OWT.sh"),
    Path("current_dreedle_train_OWT.sh"),
)


BASIS_PARSER = '''parse_basis_family_values() {
  local normalized="${1//,/ }" value
  for value in $normalized; do
    [[ "$value" =~ ^[a-z][a-z0-9_]*$ ]] || { echo "Invalid BASIS_FAMILY value: $value; expected a lowercase registry name or alias." >&2; exit 2; }
    BASIS_FAMILY_VALUES+=("$value")
  done
  (( ${#BASIS_FAMILY_VALUES[@]} > 0 )) || { echo "Invalid BASIS_FAMILY: empty value list." >&2; exit 2; }
}
'''.splitlines()


BASIS_RESOLUTION = '''# BASIS_TAG="$("$PYTHON_BIN" -c 'import sys; from sheet.bases import basis_artifact_tag_for_family; print(basis_artifact_tag_for_family(sys.argv[1]))' "$BASIS_FAMILY")"
BASIS_FAMILY_CANONICAL_VALUES=()
for requested_basis_family in "${BASIS_FAMILY_VALUES[@]}"; do
  basis_resolution="$("$PYTHON_BIN" -c 'import sys; from sheet.bases import basis_artifact_tag_for_family, normalize_registered_basis_family; family = normalize_registered_basis_family(sys.argv[1]); print(f"{family}\\t{basis_artifact_tag_for_family(family)}")' "$requested_basis_family")"
  IFS=$'\\t' read -r basis_family_value basis_tag <<< "$basis_resolution"
  BASIS_FAMILY_CANONICAL_VALUES+=("$basis_family_value")
  BASIS_TAG_VALUES+=("$basis_tag")
done
BASIS_FAMILY_VALUES=("${BASIS_FAMILY_CANONICAL_VALUES[@]}")'''.splitlines()


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


def update_wrapper(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    output: list[str] = []
    counters: dict[str, int] = {}

    def hit(name: str) -> None:
        counters[name] = counters.get(name, 0) + 1

    for line in lines:
        stripped = line.strip()

        if line.startswith("  -B BASIS_FAMILY=${BASIS_FAMILY}"):
            output.append("  -B BASIS_FAMILY=${BASIS_FAMILY}                   canonical: chebyshev | dct | haar; single, comma, or quoted space list")
            hit("help")
            continue

        if line == "PRESET_VALUES=()":
            output.append(line)
            output.append("BASIS_FAMILY_VALUES=()                                                                                                                                    # <<< THOG basis-family grid axis")
            output.append("BASIS_TAG_VALUES=()                                                                                                                                       # <<< THOG matching artifact tags for basis-family grid")
            hit("arrays")
            continue

        if line == 'parse_o_depth_values "$O_DEPTH"':
            output.extend(BASIS_PARSER)
            output.append(line)
            hit("parser")
            continue

        if line.startswith('parse_geometry_preset_values "$GEOMETRY_PRESET"'):
            output.append(line)
            output.append('parse_basis_family_values "$BASIS_FAMILY"                                                                                                                  # <<< THOG parse basis-family grid')
            hit("parser_call")
            continue

        if line == '[[ "$BASIS_FAMILY" =~ ^[a-z][a-z0-9_]*$ ]] || { echo "BASIS_FAMILY must be a lowercase registry name or alias." >&2; exit 2; }':
            output.extend(
                [
                    'if (( ${#BASIS_FAMILY_VALUES[@]} > 1 )) && [[ "$BASIS_VERSION" != auto ]]; then',
                    '  echo "BASIS_VERSION must be auto when BASIS_FAMILY contains multiple values." >&2',
                    '  exit 2',
                    'fi',
                ]
            )
            hit("validation")
            continue

        if line.startswith('BASIS_TAG="$("$PYTHON_BIN" -c '):
            output.extend(BASIS_RESOLUTION)
            hit("resolution")
            continue

        if line == "run_preset_o_depth_batch_lr() {":
            output.append("run_grid_point() {")
            hit("function")
            continue

        if stripped.startswith('local learning_rate_code="$4"'):
            output.append(line)
            output.append('  local basis_family_value="$5"                                                                                                                           # <<< THOG canonical basis-family grid coordinate')
            output.append('  local basis_tag="$6"                                                                                                                                    # <<< THOG matching basis artifact tag')
            hit("function_args")
            continue

        if 'run_model_type="dense"; display_model_type="dense"; preset_tag="DENSE"; run_tag="DENSE"' in line:
            output.append(line.replace('run_tag="DENSE"', 'run_tag="DENSE"; basis_family_value="n/a"'))
            hit("dense")
            continue

        if stripped == 'run_tag="${BASIS_TAG}_${preset_tag}"':
            output.append('    run_tag="${basis_tag}_${preset_tag}"')
            hit("run_tag")
            continue

        if 'compact_args=(--geometry-preset "$geometry_preset_value" --basis-family "$BASIS_FAMILY"' in line:
            output.append(line.replace('--basis-family "$BASIS_FAMILY"', '--basis-family "$basis_family_value"'))
            hit("compact_args")
            continue

        if "model/preset/basis:" in line and "$BASIS_FAMILY" in line:
            output.append(line.replace("$BASIS_FAMILY", "$basis_family_value"))
            hit("display")
            continue

        output.append(line)

    expected = {
        "help", "arrays", "parser", "parser_call", "validation", "resolution",
        "function", "function_args", "dense", "run_tag", "compact_args", "display",
    }
    if set(counters) != expected or any(count != 1 for count in counters.values()):
        raise RuntimeError(f"{path}: wrapper transformation counters were {counters}")

    loop_prefix = 'if (( ${#PRESET_VALUES[@]} > 1 || ${#O_DEPTH_VALUES[@]} > 1 || ${#BATCH_SIZE_VALUES[@]} > 1 || ${#LEARNING_RATE_CODE_VALUES[@]} > 1 )); then'
    try:
        loop_start = output.index(loop_prefix)
    except ValueError as exc:
        raise RuntimeError(f"{path}: could not find old grid loop") from exc
    loop_end = len(output) - 1
    if output[loop_end] != "# ^^^ THOG":
        raise RuntimeError(f"{path}: expected final THOG marker")

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
done'''.splitlines()

    output = output[:loop_start] + new_loops + output[loop_end:]
    final_text = "\n".join(output) + "\n"

    required_fragments = (
        'BASIS_FAMILY_VALUES=()',
        'BASIS_TAG_VALUES=()',
        'run_grid_point()',
        'for basis_index in "${!BASIS_FAMILY_VALUES[@]}"',
        '--basis-family "$basis_family_value"',
        'model/preset/basis: $display_model_type / $geometry_preset_value / $basis_family_value',
    )
    for fragment in required_fragments:
        if fragment not in final_text:
            raise RuntimeError(f"{path}: missing final fragment {fragment!r}")

    path.write_text(final_text, encoding="utf-8")


def update_fake_python_harness(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    old = "        print({'chebyshev': 'CHEBY', 'dct': 'DCT', 'haar': 'HAAR'}[family])\n"
    new = '''        tag = {'chebyshev': 'CHEBY', 'dct': 'DCT', 'haar': 'HAAR'}[family]
        if 'normalize_registered_basis_family' in code:
            print(f'{family}\\t{tag}')
        else:
            print(tag)
'''
    if text.count(old) != 1:
        raise RuntimeError(f"{path}: expected one fake tag-print line; found {text.count(old)}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    for wrapper in WRAPPERS:
        update_wrapper(wrapper)

    update_fake_python_harness(Path("tests/test_picton_wrapper_defaults_and_nonfinite_policy.py"))
    update_fake_python_harness(Path("tests/test_wrapper_learning_rate_and_batch_grids.py"))
    Path("tests/test_wrapper_basis_family_grid.py").write_text(TEST_CONTENT, encoding="utf-8")


if __name__ == "__main__":
    main()
