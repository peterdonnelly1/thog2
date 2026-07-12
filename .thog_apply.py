# vvv THOG
from __future__ import annotations

import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent

# Parts 01-04 contain the already-tested dreedle/model/test changes. Part 01
# begins with the tail of the scruffy diff, so discard everything before the
# first complete dreedle file header and apply the remainder exactly once.
patch_parts = [root / f'.thog_patch_part_{index:02d}' for index in range(1, 5)]
patch_data = b''.join(path.read_bytes() for path in patch_parts)
first_complete_header = b'--- thogorig/current_dreedle_train_OWT.sh'
header_index = patch_data.find(first_complete_header)
if header_index < 0:
    raise SystemExit('Complete dreedle patch header not found')
patch_path = Path('/tmp/thog_complete_without_scruffy.patch')
patch_path.write_bytes(patch_data[header_index:])
subprocess.run(['git', 'apply', '--check', '-p1', str(patch_path)], cwd=root, check=True)
subprocess.run(['git', 'apply', '-p1', str(patch_path)], cwd=root, check=True)

# Apply the matching scruffy-wrapper changes deterministically. The dreedle
# wrapper and regression tests are supplied by the prepared patch above.
path = root / 'current_scruffy_train_OWT.sh'
text = path.read_text(encoding='utf-8')
text = text.replace(
    'BATCH_SIZE=3\n',
    'BATCH_SIZE=3\n'
    'LEARNING_RATE_CODES="60"                                                                                                                               # <<< THOG LR grid codes; 70 means 7.0e-04\n'
    'MIN_LR_CODE="06"                                                                                                                                         # <<< THOG minimum LR code; 06 means 6.0e-05\n',
    1,
)
text = text.replace(
    'DIRECT_THOG_MLP_APPLICATION="${THOG2_DIRECT_THOG_MLP_APPLICATION:-false}"                                                                              # <<< THOG default-off exact direct application of existing THOG MLP factors\n',
    '# DIRECT_THOG_MLP_APPLICATION="${THOG2_DIRECT_THOG_MLP_APPLICATION:-false}"                                                                         # <<< THOG retired option name retained as history\n'
    'DIRECT_FACTORISED_MLP="${THOG2_DIRECT_FACTORISED_MLP:-true}"                                                                                              # <<< THOG renamed default-on exact factorised MLP application\n'
    'VECTORISE_PER_HEAD_MATERIALISATION="${THOG2_VECTORISE_PER_HEAD_MATERIALISATION:-true}"                                                                    # <<< THOG default-on selectable per-head batching\n',
    1,
)
text = text.replace(
    '  -b BATCH_SIZE=${BATCH_SIZE}\n',
    '  -b BATCH_SIZE=${BATCH_SIZE}                         single integer, comma list, or quoted space list\n'
    '  -c LR_CODES=${LEARNING_RATE_CODES}                    70 means 7.0e-04; comma or quoted space list\n'
    '  -f MIN_LR_CODE=${MIN_LR_CODE}                         06 means 6.0e-05\n',
    1,
)
text = text.replace('while getopts ":q:g:n:b:A:', 'while getopts ":q:g:n:b:c:f:A:', 1)
text = text.replace(
    'n) STEPS="$OPTARG" ;; b) BATCH_SIZE="$OPTARG" ;; A)',
    'n) STEPS="$OPTARG" ;; b) BATCH_SIZE="$OPTARG" ;; c) LEARNING_RATE_CODES="$OPTARG" ;; f) MIN_LR_CODE="$OPTARG" ;; A)',
    1,
)
text = text.replace(
    'O_DEPTH_VALUES=()\nPRESET_VALUES=()\n',
    'O_DEPTH_VALUES=()\nPRESET_VALUES=()\n'
    'BATCH_SIZE_VALUES=()                                                                                                                                        # <<< THOG batch grid axis\n'
    'LEARNING_RATE_CODE_VALUES=()                                                                                                                                # <<< THOG LR grid axis\n',
    1,
)
parsers = '''parse_positive_uint_values() {
  local normalized="${1//,/ }" value
  for value in $normalized; do validate_positive_uint "$value" "$2"; BATCH_SIZE_VALUES+=("$value"); done
  (( ${#BATCH_SIZE_VALUES[@]} > 0 )) || { echo "Invalid BATCH_SIZE list." >&2; exit 2; }
}
validate_lr_code() { [[ "$1" =~ ^[0-9]{1,2}$ ]] && (( 10#$1 > 0 )) || { echo "Invalid $2: $1; expected 01..99." >&2; exit 2; }; }
parse_lr_code_values() {
  local normalized="${1//,/ }" value
  for value in $normalized; do validate_lr_code "$value" "LEARNING_RATE_CODES"; LEARNING_RATE_CODE_VALUES+=("$((10#$value))"); done
  (( ${#LEARNING_RATE_CODE_VALUES[@]} > 0 )) || { echo "Invalid learning-rate code list." >&2; exit 2; }
}
'''
text = text.replace('parse_o_depth_values() {', parsers + 'parse_o_depth_values() {', 1)
text = text.replace(
    'parse_geometry_preset_values "$GEOMETRY_PRESET"\n',
    'parse_geometry_preset_values "$GEOMETRY_PRESET"\n'
    'parse_positive_uint_values "$BATCH_SIZE" "BATCH_SIZE"                                                                                                  # <<< THOG parse batch grid\n'
    'parse_lr_code_values "$LEARNING_RATE_CODES"                                                                                                              # <<< THOG parse LR grid\n'
    'validate_lr_code "$MIN_LR_CODE" "MIN_LR_CODE"                                                                                                          # <<< THOG validate min LR\n',
    1,
)
text = text.replace(
    'for setting in "$STEPS" "$BATCH_SIZE" "$GRADIENT_ACCUMULATION_STEPS"',
    'for setting in "$STEPS" "$GRADIENT_ACCUMULATION_STEPS"',
    1,
)
text = text.replace(
    'validate_true_false "$DIRECT_THOG_MLP_APPLICATION" "DIRECT_THOG_MLP_APPLICATION"',
    '# validate_true_false "$DIRECT_THOG_MLP_APPLICATION" "DIRECT_THOG_MLP_APPLICATION"                                                               # <<< THOG retired validation\n'
    'validate_true_false "$DIRECT_FACTORISED_MLP" "DIRECT_FACTORISED_MLP"                                                                                   # <<< THOG validate renamed exact MLP option\n'
    'validate_true_false "$VECTORISE_PER_HEAD_MATERIALISATION" "VECTORISE_PER_HEAD_MATERIALISATION"                                                         # <<< THOG validate per-head option',
    1,
)
text = text.replace(
    'export THOG2_DIRECT_THOG_MLP_APPLICATION="$DIRECT_THOG_MLP_APPLICATION"',
    '# export THOG2_DIRECT_THOG_MLP_APPLICATION="$DIRECT_THOG_MLP_APPLICATION"                                                                          # <<< THOG retired environment variable\n'
    'export THOG2_DIRECT_FACTORISED_MLP="$DIRECT_FACTORISED_MLP"                                                                                              # <<< THOG pass renamed option\n'
    'export THOG2_VECTORISE_PER_HEAD_MATERIALISATION="$VECTORISE_PER_HEAD_MATERIALISATION"                                                                    # <<< THOG pass per-head option',
    1,
)
text = text.replace(
    'run_preset_o_depth() {\n  local geometry_preset_value="$1"\n  local o_depth_value="$2"\n',
    'run_preset_o_depth_batch_lr() {\n'
    '  local geometry_preset_value="$1"\n'
    '  local o_depth_value="$2"\n'
    '  local batch_size_value="$3"                                                                                                                             # <<< THOG batch grid coordinate\n'
    '  local learning_rate_code="$4"                                                                                                                           # <<< THOG LR grid coordinate\n'
    '  local learning_rate_value="${learning_rate_code}e-5" min_lr_value="$((10#$MIN_LR_CODE))e-5"                                                         # <<< THOG decode LR codes\n',
    1,
)
text = text.replace('--max-iters "$STEPS" --batch-size "$BATCH_SIZE"', '--max-iters "$STEPS" --batch-size "$batch_size_value"', 1)
text = text.replace(
    '--checkpoint-interval "$CHECKPOINT_INTERVAL" --warmup-iters "$WARMUP_ITERS"',
    '--checkpoint-interval "$CHECKPOINT_INTERVAL" --warmup-iters "$WARMUP_ITERS" --learning-rate "$learning_rate_value" --min-lr "$min_lr_value"',
    1,
)
text = text.replace(
    '  direct THOG MLP apply:    $DIRECT_THOG_MLP_APPLICATION',
    '  direct factorised MLP:    $DIRECT_FACTORISED_MLP\n  vectorise per-head materialisation: $VECTORISE_PER_HEAD_MATERIALISATION',
    1,
)
text = text.replace(
    '  schedule:           steps=$STEPS',
    '  optimizer:          lr=$learning_rate_value (LR_$learning_rate_code) min_lr=$min_lr_value\n  schedule:           steps=$STEPS',
    1,
)
text = text.replace('  batch/accum/gpus:   $BATCH_SIZE /', '  batch/accum/gpus:   $batch_size_value /', 1)
old_tail_start = text.index('if (( ${#PRESET_VALUES[@]} > 1 || ${#O_DEPTH_VALUES[@]} > 1 )); then')
old_tail_end = text.index('# ^^^ THOG', old_tail_start)
new_tail = '''if (( ${#PRESET_VALUES[@]} > 1 || ${#O_DEPTH_VALUES[@]} > 1 || ${#BATCH_SIZE_VALUES[@]} > 1 || ${#LEARNING_RATE_CODE_VALUES[@]} > 1 )); then
  echo "scruffy OWT grid: p=${PRESET_VALUES[*]} P=${O_DEPTH_VALUES[*]} b=${BATCH_SIZE_VALUES[*]} LR=${LEARNING_RATE_CODE_VALUES[*]}"
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
'''
text = text[:old_tail_start] + new_tail + text[old_tail_end:]
path.write_text(text, encoding='utf-8')

# Let the artifact-code regression reach the new code-specific validation.
for test_path in root.glob('tests/test*.py'):
    test_text = test_path.read_text(encoding='utf-8')
    updated = test_text.replace('learning_rate=0.0', 'learning_rate=1.005e-5')
    if updated != test_text:
        test_path.write_text(updated, encoding='utf-8')
# ^^^ THOG
