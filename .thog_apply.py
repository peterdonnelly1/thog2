from __future__ import annotations

import base64
import gzip
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PAYLOAD = ROOT / ".thog_apply_script.py.gz.b64"
source = gzip.decompress(base64.b64decode(PAYLOAD.read_text(encoding="utf-8"))).decode("utf-8")
namespace = {
    "__file__": str(ROOT / ".thog_apply.py"),
    "__name__": "__main__",
}
exec(compile(source, namespace["__file__"], "exec"), namespace)


def replace_once(relative_path: str, old: str, new: str) -> None:
    path = ROOT / relative_path
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one occurrence in {relative_path}: {old!r}; found {count}"
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# The local compressor is independent of the global/DEPTH basis. Keep the
# trajectory fixtures on the legacy Chebyshev depth default while varying the
# registered local compressor.
path = ROOT / "tests" / "test_jpeg_like_v1_trajectory.py"
text = path.read_text(encoding="utf-8")
text = text.replace(
    "from sheet.bases import BASIS_FAMILY_DCT",
    "from sheet.bases import BASIS_FAMILY_CHEBYSHEV",
)
text = text.replace(
    "basis_family=BASIS_FAMILY_DCT",
    "basis_family=BASIS_FAMILY_CHEBYSHEV",
)
text = text.replace('basis_family="dct"', 'basis_family="chebyshev"')
path.write_text(text, encoding="utf-8")

replace_once(
    "tests/test_jpeg_like_v1_integration.py",
    '                geometry_preset="depth",\n                basis_family="dct",\n',
    '                geometry_preset="depth",\n                basis_family="chebyshev",\n',
)

# Preserve the established optimizer-wrapper regression marker while extending
# the same long-option pre-parser for JPEG_LIKE_V1 controls.
for wrapper in ("current_scruffy_train_OWT.sh", "current_dreedle_train_OWT.sh"):
    replace_once(
        wrapper,
        "# vvv THOG accept long optimizer and JPEG_LIKE_V1 controls without disturbing the established getopts contract",
        "# vvv THOG accept long optimizer controls and JPEG_LIKE_V1 controls without disturbing the established getopts contract",
    )

# Extend the existing exact preset registry contract.
replace_once(
    "tests/test_picton_preset_contract.py",
    "    GEOMETRY_PRESET_HEAD_AWARE_BLOCK,\n    GEOMETRY_PRESET_LEGACY_SHEET_COL,\n    GEOMETRY_PRESET_MLP_BLOCK,\n",
    "    GEOMETRY_PRESET_HEAD_AWARE_BLOCK,\n    GEOMETRY_PRESET_JPEG_LIKE_V1,\n    GEOMETRY_PRESET_LEGACY_SHEET_COL,\n    GEOMETRY_PRESET_MLP_BLOCK,\n",
)
replace_once(
    "tests/test_picton_preset_contract.py",
    "    HEAD_AWARE_BLOCK_MATERIALIZATION_VERSION,\n    LEGACY_SHEET_COL_MATERIALIZATION_VERSION,\n",
    "    HEAD_AWARE_BLOCK_MATERIALIZATION_VERSION,\n    JPEG_LIKE_V1_MATERIALIZATION_VERSION,\n    LEGACY_SHEET_COL_MATERIALIZATION_VERSION,\n",
)
replace_once(
    "tests/test_picton_preset_contract.py",
    "    MLP_GEOMETRY_DEPTH,\n    MLP_GEOMETRY_LEGACY_SHEET_COL,\n",
    "    MLP_GEOMETRY_DEPTH,\n    MLP_GEOMETRY_JPEG_LIKE_V1,\n    MLP_GEOMETRY_LEGACY_SHEET_COL,\n",
)
replace_once(
    "tests/test_picton_preset_contract.py",
    "        (\n            GEOMETRY_PRESET_MLP_BLOCK,\n",
    "        (\n            GEOMETRY_PRESET_JPEG_LIKE_V1,\n            ATTENTION_GEOMETRY_DEPTH,\n            MLP_GEOMETRY_JPEG_LIKE_V1,\n            JPEG_LIKE_V1_MATERIALIZATION_VERSION,\n        ),\n        (\n            GEOMETRY_PRESET_MLP_BLOCK,\n",
)
replace_once(
    "tests/test_picton_preset_contract.py",
    "        GEOMETRY_PRESET_DEPTH,\n        GEOMETRY_PRESET_MLP_BLOCK,\n",
    "        GEOMETRY_PRESET_DEPTH,\n        GEOMETRY_PRESET_JPEG_LIKE_V1,\n        GEOMETRY_PRESET_MLP_BLOCK,\n",
)
replace_once(
    "tests/test_stage8_mlp_channel_order_and_wrapper_loops.py",
    '        assert "dense | legacy_sheet_col | depth | head_aware_block | mlp_block | full_block" in text\n',
    '        assert "dense | legacy_sheet_col | depth | jpeg_like_v1 | head_aware_block | mlp_block | full_block" in text\n',
)

for temporary in (
    PAYLOAD,
    ROOT / ".thog_apply_jpeg_like_v1_trigger",
    ROOT / ".github" / "workflows" / "apply_jpeg_like_v1_now.yml",
):
    temporary.unlink(missing_ok=True)

for script in ROOT.glob("*.sh"):
    script.chmod(script.stat().st_mode | 0o111)
