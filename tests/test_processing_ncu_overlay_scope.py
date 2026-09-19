# vvv THOG
from pathlib import Path


def test_ncu_2024_runtime_overlay_uses_all_matrix_scope_contract():
    source = Path("sheet/processing_ncu_2024_instra_patch.py").read_text()
    assert "_processing_ncu_target_from_argv" not in source
    assert "_processing_ncu_scope_from_argv(arguments)" in source
    assert "premat_families=premat_families" in source
    assert "family=family" not in source
# ^^^ THOG
