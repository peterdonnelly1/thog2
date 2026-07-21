from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parent
runpy.run_path(str(ROOT / ".thog_apply_basis_family_refactor.py"), run_name="__main__")

for wrapper_name in ("current_scruffy_train_OWT.sh", "current_dreedle_train_OWT.sh"):
    path = ROOT / wrapper_name
    text = path.read_text(encoding="utf-8")
    validation = '[[ "$BASIS_FAMILY" =~ ^[a-z][a-z0-9_]*$ ]] || { echo "BASIS_FAMILY must be a lowercase registry name or alias." >&2; exit 2; }\n'
    validation_block = '# vvv THOG\n' + validation + '# ^^^ THOG\n'
    if validation_block not in text:
        if text.count(validation) != 1:
            raise RuntimeError(f"expected one basis validation line in {wrapper_name}")
        text = text.replace(validation, validation_block, 1)

    tag = 'BASIS_TAG="$("$PYTHON_BIN" -c \'import sys; from sheet.bases import basis_artifact_tag_for_family; print(basis_artifact_tag_for_family(sys.argv[1]))\' "$BASIS_FAMILY")"                           # <<< THOG registry-derived basis validation and artifact tag\n'
    tag_block = '# vvv THOG\n' + tag.split('                           # <<< THOG', 1)[0] + '\n# ^^^ THOG\n'
    if tag_block not in text:
        if text.count(tag) != 1:
            raise RuntimeError(f"expected one registry-derived basis tag line in {wrapper_name}")
        text = text.replace(tag, tag_block, 1)

    path.write_text(text, encoding="utf-8")

print("Applied final THOG wrapper markings")
