from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent
WRAPPERS = (
    ROOT / "current_scruffy_train_OWT.sh",
    ROOT / "current_dreedle_train_OWT.sh",
)

for path in WRAPPERS:
    text = path.read_text(encoding="utf-8")

    start_marker = (
        "# vvv THOG native optimizer selection; strip optimizer-only controls "
        "before canonical option parsing"
    )
    end_marker = "# ^^^ THOG\n\nusage() {"
    start = text.find(start_marker)
    end = text.find(end_marker, start)
    if start < 0 or end < 0:
        raise RuntimeError(f"optimizer pre-parser markers missing in {path.name}")

    text = (
        text[:start]
        + 'OPTIMIZER="${THOG2_OPTIMIZER:-adamw}"\n'
        + 'OPTIMIZER_MOMENTUM="${THOG2_OPTIMIZER_MOMENTUM:-0.9}"\n'
        + "OPTIMIZER_LR_EXPLICIT=false\n"
        + "OPTIMIZER_MIN_LR_EXPLICIT=false\n\n"
        + "usage() {"
        + text[end + len(end_marker) :]
    )

    getopts_marker = 'while getopts "'
    getopts_index = text.find(getopts_marker)
    if getopts_index < 0:
        raise RuntimeError(f"getopts marker missing in {path.name}")

    long_option_prepass = '''# vvv THOG accept long optimizer controls without disturbing the established getopts contract
OPTIMIZER_FILTERED_ARGS=()
OPTIMIZER_SAW_SEPARATOR=false
while (( $# > 0 )); do
  if [[ "$OPTIMIZER_SAW_SEPARATOR" == true ]]; then
    OPTIMIZER_FILTERED_ARGS+=("$1")
    shift
    continue
  fi
  case "$1" in
    --optimizer)
      (( $# >= 2 )) || { echo "--optimizer requires a name" >&2; exit 2; }
      OPTIMIZER="$2"
      shift 2
      ;;
    --optimizer=*)
      OPTIMIZER="${1#*=}"
      shift
      ;;
    --optimizer-momentum)
      (( $# >= 2 )) || { echo "--optimizer-momentum requires a numeric value" >&2; exit 2; }
      OPTIMIZER_MOMENTUM="$2"
      shift 2
      ;;
    --optimizer-momentum=*)
      OPTIMIZER_MOMENTUM="${1#*=}"
      shift
      ;;
    --)
      OPTIMIZER_FILTERED_ARGS+=("--")
      OPTIMIZER_SAW_SEPARATOR=true
      shift
      ;;
    *)
      OPTIMIZER_FILTERED_ARGS+=("$1")
      shift
      ;;
  esac
done
set -- "${OPTIMIZER_FILTERED_ARGS[@]}"
# ^^^ THOG

'''
    text = text[:getopts_index] + long_option_prepass + text[getopts_index:]

    old_getopts = 'while getopts ":q:g:n:b:c:f:'
    new_getopts = 'while getopts ":q:g:n:b:c:f:y:'
    if old_getopts not in text:
        raise RuntimeError(f"getopts option string missing in {path.name}")
    text = text.replace(old_getopts, new_getopts, 1)

    old_case = (
        'n) STEPS="$OPTARG" ;; b) BATCH_SIZE="$OPTARG" ;; '
        'c) LEARNING_RATE_CODES="$OPTARG" ;; '
        'f) MIN_LR_CODE="$OPTARG" ;; A)'
    )
    new_case = (
        'n) STEPS="$OPTARG" ;; b) BATCH_SIZE="$OPTARG" ;; '
        'c) LEARNING_RATE_CODES="$OPTARG"; OPTIMIZER_LR_EXPLICIT=true ;; '
        'f) MIN_LR_CODE="$OPTARG"; OPTIMIZER_MIN_LR_EXPLICIT=true ;; '
        'y) OPTIMIZER="$OPTARG" ;; A)'
    )
    if old_case not in text:
        raise RuntimeError(f"getopts case marker missing in {path.name}")
    text = text.replace(old_case, new_case, 1)

    extra_marker = (
        'EXTRA_ARGS=("$@")\n'
        '# vvv THOG make optimizer identity collision-safe in artifact naming'
    )
    normalization = '''EXTRA_ARGS=("$@")

# vvv THOG normalize optimizer and apply its LR defaults only when -c/-f were omitted
case "${OPTIMIZER,,}" in
  adam|adamw)
    OPTIMIZER="adamw"; OPTIMIZER_DEFAULT_LR_CODE="60"; OPTIMIZER_DEFAULT_MIN_LR_CODE="06" ;;
  sgd)
    OPTIMIZER="sgd"; OPTIMIZER_DEFAULT_LR_CODE="1000"; OPTIMIZER_DEFAULT_MIN_LR_CODE="100" ;;
  nesterov|sgd-nesterov|sgd_nesterov)
    OPTIMIZER="sgd_nesterov"; OPTIMIZER_DEFAULT_LR_CODE="1000"; OPTIMIZER_DEFAULT_MIN_LR_CODE="100" ;;
  adafactor)
    OPTIMIZER="adafactor"; OPTIMIZER_DEFAULT_LR_CODE="1000"; OPTIMIZER_DEFAULT_MIN_LR_CODE="100" ;;
  rmsprop)
    OPTIMIZER="rmsprop"; OPTIMIZER_DEFAULT_LR_CODE="100"; OPTIMIZER_DEFAULT_MIN_LR_CODE="10" ;;
  *)
    echo "Unsupported optimizer: $OPTIMIZER" >&2
    echo "Expected: adamw | sgd | sgd_nesterov | adafactor | rmsprop" >&2
    exit 2
    ;;
esac
[[ "$OPTIMIZER_LR_EXPLICIT" == true ]] || LEARNING_RATE_CODES="$OPTIMIZER_DEFAULT_LR_CODE"
[[ "$OPTIMIZER_MIN_LR_EXPLICIT" == true ]] || MIN_LR_CODE="$OPTIMIZER_DEFAULT_MIN_LR_CODE"
export THOG2_OPTIMIZER="$OPTIMIZER"
export THOG2_OPTIMIZER_MOMENTUM="$OPTIMIZER_MOMENTUM"
# ^^^ THOG

# vvv THOG make optimizer identity collision-safe in artifact naming'''
    if extra_marker not in text:
        raise RuntimeError(f"artifact-suffix marker missing in {path.name}")
    text = text.replace(extra_marker, normalization, 1)

    if not text.startswith("#!/bin/bash\nset -euo pipefail\n\n# vvv THOG"):
        raise RuntimeError(f"top comment block was displaced in {path.name}")
    if text.index("usage()") > text.index("accept long optimizer controls"):
        raise RuntimeError(f"usage block must precede optimizer long-option parsing in {path.name}")

    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)


test_path = ROOT / "tests" / "test_optimizer_wrapper.py"
test_source = test_path.read_text(encoding="utf-8")
test_source = test_source.replace(
    "import subprocess\nimport unittest\n",
    "import os\nimport subprocess\nimport sys\nimport unittest\n",
    1,
)
test_source = test_source.replace(
    '            self.assertIn("run_grid_point()", source)\n',
    '            self.assertIn("run_grid_point()", source)\n'
    '            self.assertLess(source.index("usage()"), source.index("accept long optimizer controls"))\n',
    1,
)
additional_tests = '''
    def run_dry_wrapper(self, *arguments: str) -> str:
        environment = dict(os.environ)
        environment["THOG2_PYTHON"] = sys.executable
        result = subprocess.run(
            [
                "bash",
                "current_scruffy_train_OWT.sh",
                *arguments,
                "-p", "dense",
                "-n", "2",
                "-w", "0",
                "-b", "1",
                "-A", "1",
                "-u", "1",
                "-e", "1",
                "-l", "1",
                "-k", "0",
                "-I", "none",
                "-F", "none",
                "-x", "true",
            ],
            cwd=ROOT,
            env=environment,
            check=True,
            text=True,
            capture_output=True,
        )
        return result.stdout

    def test_short_optimizer_control_uses_optimizer_defaults(self) -> None:
        output = self.run_dry_wrapper("-y", "sgd")
        self.assertIn("optimizer:          sgd", output)
        self.assertIn("LR_1000", output)
        self.assertIn("OPT_SGD", output)

    def test_long_optimizer_controls_and_explicit_lr_override(self) -> None:
        output = self.run_dry_wrapper(
            "--optimizer=rmsprop",
            "--optimizer-momentum=0.95",
            "-c", "77",
            "-f", "07",
        )
        self.assertIn("optimizer:          rmsprop  momentum=0.95", output)
        self.assertIn("LR_77", output)
        self.assertIn("min_lr=7e-5", output)

    def test_existing_run_scruffy_scripts_remain_present(self) -> None:
        self.assertTrue((ROOT / "run_scruffy_dense_sheet_l144_smoke_owt.sh").is_file())
        self.assertTrue((ROOT / "run_scruffy_sheet_eden_best_long_owt.sh").is_file())
'''
marker = '\n\nif __name__ == "__main__":\n'
if marker not in test_source:
    raise RuntimeError("test insertion marker missing")
test_source = test_source.replace(marker, additional_tests + marker, 1)
test_path.write_text(test_source, encoding="utf-8")

for expected in (
    ROOT / "run_scruffy_dense_sheet_l144_smoke_owt.sh",
    ROOT / "run_scruffy_sheet_eden_best_long_owt.sh",
):
    if not expected.is_file():
        raise RuntimeError(f"existing tracked run script missing: {expected.name}")

for forbidden in (
    ROOT / "optimizer_train_OWT_wrapper.sh",
    ROOT / ".current_scruffy_train_OWT_impl",
    ROOT / ".current_dreedle_train_OWT_impl",
    ROOT / ".current_scruffy_train_OWT_body",
    ROOT / ".current_dreedle_train_OWT_body",
):
    if forbidden.exists():
        raise RuntimeError(f"obsolete optimizer wrapper layer remains: {forbidden.name}")

print("Reduced optimizer integration to the existing full wrapper structure.")
