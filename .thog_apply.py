from __future__ import annotations

from pathlib import Path


body_path = Path(__file__).with_name(".thog_apply_body.py")
source = body_path.read_text(encoding="utf-8")

old_summary_block = '''    summary_marker = "  optimizer:          lr=$learning_rate_value (LR_$learning_rate_code) min_lr=$min_lr_value"
    if summary_marker not in text:
        raise RuntimeError(f"missing optimizer summary marker in {source}")
    text = text.replace(
        summary_marker,
        "  optimizer:          $OPTIMIZER  momentum=$OPTIMIZER_MOMENTUM  lr=$learning_rate_value (LR_$learning_rate_code) min_lr=$min_lr_value",
        1,
    )
'''
new_summary_block = '''    summary_replacements = (
        (
            "  optimizer:          lr=$learning_rate_value (LR_$learning_rate_code) min_lr=$min_lr_value",
            "  optimizer:          $OPTIMIZER  momentum=$OPTIMIZER_MOMENTUM  lr=$learning_rate_value (LR_$learning_rate_code) min_lr=$min_lr_value",
        ),
        (
            "  optimiser:          lr_code=$learning_rate_code lr=$learning_rate_value min_lr_code=$MIN_LR_CODE min_lr=$min_lr_value",
            "  optimiser:          $OPTIMIZER  momentum=$OPTIMIZER_MOMENTUM  lr_code=$learning_rate_code lr=$learning_rate_value min_lr_code=$MIN_LR_CODE min_lr=$min_lr_value",
        ),
    )
    for summary_marker, summary_replacement in summary_replacements:
        if summary_marker in text:
            text = text.replace(summary_marker, summary_replacement, 1)
            break
    else:
        raise RuntimeError(f"missing optimizer summary marker in {source}")
'''
if old_summary_block not in source:
    raise RuntimeError("queued repair body no longer contains the expected summary block")
source = source.replace(old_summary_block, new_summary_block, 1)

cleanup_marker = '    ".github/workflows/repair_optimizer_wrappers.yml",\n):'
cleanup_replacement = '    ".github/workflows/repair_optimizer_wrappers.yml",\n    ".thog_apply_body.py",\n):'
if cleanup_marker not in source:
    raise RuntimeError("queued repair body no longer contains the expected cleanup block")
source = source.replace(cleanup_marker, cleanup_replacement, 1)

exec(compile(source, str(body_path), "exec"), {"__name__": "__main__", "__file__": str(body_path)})

basis_assignment = '''  basis_resolution="$($PYTHON_BIN -c 'import sys; from sheet.bases import basis_artifact_tag_for_family, normalize_registered_basis_family; family = normalize_registered_basis_family(sys.argv[1]); print(f"{family}\\t{basis_artifact_tag_for_family(family)}")' "$requested_basis_family")"'''
basis_guard = '''  if ! basis_resolution="$($PYTHON_BIN -c 'import sys; from sheet.bases import basis_artifact_tag_for_family, normalize_registered_basis_family; family = normalize_registered_basis_family(sys.argv[1]); print(f"{family}\\t{basis_artifact_tag_for_family(family)}")' "$requested_basis_family")"; then
    echo "Failed to resolve BASIS_FAMILY: $requested_basis_family" >&2
    exit 2
  fi'''

for wrapper_name in ("current_scruffy_train_OWT.sh", "current_dreedle_train_OWT.sh"):
    wrapper_path = Path(__file__).with_name(wrapper_name)
    wrapper_source = wrapper_path.read_text(encoding="utf-8")
    wrapper_source = wrapper_source.replace(
        "before canonical getopts parsing",
        "before canonical option parsing",
        1,
    )
    if basis_assignment not in wrapper_source:
        raise RuntimeError(f"missing basis-resolution assignment in {wrapper_name}")
    wrapper_source = wrapper_source.replace(basis_assignment, basis_guard, 1)
    wrapper_path.write_text(wrapper_source, encoding="utf-8")

obsolete_test = Path(__file__).with_name("tests") / "test_optimizer_entry_wrappers.py"
if obsolete_test.exists():
    obsolete_test.unlink()
