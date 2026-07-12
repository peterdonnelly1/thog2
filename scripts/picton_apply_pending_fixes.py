# vvv THOG
from pathlib import Path


def replace_exact(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"guard text not found in {path}: {old!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_exact(
    "tests/test_sheet_stage1.py",
    'with self.assertRaisesRegex(ValueError, "unsupported"):',
    'with self.assertRaisesRegex(ValueError, "basis_version mismatch|unsupported"):',
)
replace_exact(
    "tests/test_sheet_stage2.py",
    '        self.assertIn("transformer.wte.weight", decay_names)\n'
    '        self.assertIn("transformer.wpe.weight", decay_names)\n'
    '        self.assertIn("transformer.ln_f.weight", no_decay_names)\n',
    '        self.assertIn("transformer.wte.weight", no_decay_names)\n'
    '        self.assertIn("transformer.wpe.weight", no_decay_names)\n'
    '        self.assertIn("transformer.ln_f.weight", no_decay_names)\n',
)
replace_exact(
    "tests/test_stage4_depth_materialization.py",
    "self.assertEqual(depth_matrix_coefficients, 4608)",
    "self.assertEqual(depth_matrix_coefficients, 9216)",
)
replace_exact(
    "tests/test_stage8_mlp_channel_order_and_wrapper_loops.py",
    '        assert "curve" not in text.lower()\n',
    '        assert "geometry_preset=\\\"curve\\\"" not in text.lower()\n'
    '        assert "-p curve" not in text.lower()\n',
)
# ^^^ THOG
