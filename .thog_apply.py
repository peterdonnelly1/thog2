# vvv THOG
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD_VERSION = "lapped_cosine_balanced_orthonormal_v1"
NEW_VERSION = "lapped_cosine_dc_preserving_orthonormal_v1"


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, content: str) -> None:
    target = ROOT / path
    target.write_text(content, encoding="utf-8")
    print(f"wrote {path}")


def replace_once(path: str, old: str, new: str) -> None:
    text = read(path)
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected one replacement target, found {count}: {old[:120]!r}"
        )
    write(path, text.replace(old, new, 1))


for wrapper_path in (
    "current_scruffy_train_OWT.sh",
    "current_dreedle_train_OWT.sh",
):
    replace_once(wrapper_path, OLD_VERSION, NEW_VERSION)

replace_once(
    "docs/THOG2_Lapped_Cosine_Basis_Test_Plan_v0.1.md",
    f"- Version: `{OLD_VERSION}`",
    f"- Version: `{NEW_VERSION}`",
)
replace_once(
    "docs/THOG2_Lapped_Cosine_Basis_Test_Plan_v0.1.md",
    "- Transform: balanced block-local orthonormal DCT-IV atoms followed by orthogonal sine/cosine boundary rotations",
    "- Transform: normalized global DC, balanced block-mean coarse functions, block-local orthonormal DCT-II detail atoms, and DC-preserving orthogonal boundary prefilters",
)
replace_once(
    "docs/THOG2_Lapped_Cosine_Basis_Test_Plan_v0.1.md",
    "- Coefficient ordering: local mode first, balanced across all blocks before advancing to the next local mode",
    "- Coefficient ordering: global DC first, balanced block-mean coarse functions next, then each local detail mode balanced across blocks",
)

basis_test_path = "tests/test_lapped_cosine_basis_kernel.py"
replace_once(
    basis_test_path,
    "import unittest\nfrom pathlib import Path\n",
    "import math\nimport unittest\nfrom pathlib import Path\n",
)
replace_once(
    basis_test_path,
    "    def test_09_invalid_controls_versions_and_indices_fail_explicitly(self) -> None:\n",
    """    def test_09_first_column_is_exact_normalized_global_dc(self) -> None:
        for sample_count, window_length in ((8, 8), (72, 24), (144, 48)):
            with self.subTest(sample_count=sample_count, window_length=window_length):
                version = lapped_cosine_basis_version(window_length, 0.5)
                basis = build_registered_basis(
                    sample_count,
                    min(sample_count, 16),
                    basis_family=BASIS_FAMILY_LAPPED_COSINE,
                    version=version,
                )
                expected = torch.full(
                    (sample_count,),
                    1.0 / math.sqrt(float(sample_count)),
                    dtype=torch.float64,
                )
                self.assertTrue(torch.equal(basis[:, 0], expected))

    def test_10_invalid_controls_versions_and_indices_fail_explicitly(self) -> None:
""",
)

training_test_path = "tests/test_lapped_cosine_training_and_checkpoint.py"
replace_once(
    training_test_path,
    "    def test_02_controls_are_checkpoint_compatibility_fields(self) -> None:\n",
    """    def test_02_layernorm_initialization_materializes_ones_at_every_layer(self) -> None:
        config = self.training_config()
        model = build_training_model(config)
        expected = torch.ones(config.n_embd, dtype=torch.float32)
        for family_name in ("ln_1_weight", "ln_2_weight"):
            for layer_index in range(config.n_layer):
                with self.subTest(family_name=family_name, layer_index=layer_index):
                    actual = model.trajectory.materialize_vector(family_name, layer_index)
                    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0e-6)

    def test_03_controls_are_checkpoint_compatibility_fields(self) -> None:
""",
)
replace_once(
    training_test_path,
    "    def test_03_run_identity_and_training_config_preserve_controls(self) -> None:\n",
    "    def test_04_run_identity_and_training_config_preserve_controls(self) -> None:\n",
)
replace_once(
    training_test_path,
    "    def test_04_non_lapped_basis_rejects_nondefault_lapped_controls(self) -> None:\n",
    "    def test_05_non_lapped_basis_rejects_nondefault_lapped_controls(self) -> None:\n",
)

remaining = []
for path in ROOT.rglob("*"):
    if not path.is_file() or ".git" in path.parts:
        continue
    if path.name == ".thog_apply.py":
        continue
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        continue
    if OLD_VERSION in text:
        remaining.append(str(path.relative_to(ROOT)))
if remaining:
    raise RuntimeError(f"stale lapped cosine version remains in: {remaining}")

(ROOT / ".thog_commit_message").write_text(
    "Complete DC-preserving lapped cosine regressions\n",
    encoding="utf-8",
)
print("lapped cosine regression corrections staged")
# ^^^ THOG
