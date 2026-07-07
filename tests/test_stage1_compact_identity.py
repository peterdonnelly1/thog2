# vvv THOG
from __future__ import annotations

import unittest

from sheet.compact_identity import (
    ATTENTION_GEOMETRY_CURVE,
    ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK,
    ATTENTION_GEOMETRY_LEGACY_SHEET_COL,
    BASIS_FAMILY_CHEBYSHEV,
    BASIS_FAMILY_DCT,
    GEOMETRY_PRESET_BLOCK,
    GEOMETRY_PRESET_CONVENTIONAL,
    GEOMETRY_PRESET_CURVE,
    GEOMETRY_PRESET_HEAD_AWARE_BLOCK,
    GEOMETRY_PRESET_LEGACY_SHEET_COL,
    GEOMETRY_PRESET_MLP_BLOCK,
    MLP_GEOMETRY_CURVE,
    MLP_GEOMETRY_LEGACY_SHEET_COL,
    MLP_GEOMETRY_MLP_BLOCK,
    resolve_compact_selectors,
)
from sheet.training_config import TrainingConfig
from tests.stage4_test_support import stage4_training_config


class Stage1CompactIdentityTests(unittest.TestCase):
    def test_preset_expansion_is_central_and_deterministic(self) -> None:
        cases = (
            ({}, GEOMETRY_PRESET_LEGACY_SHEET_COL, ATTENTION_GEOMETRY_LEGACY_SHEET_COL, MLP_GEOMETRY_LEGACY_SHEET_COL, BASIS_FAMILY_CHEBYSHEV),
            ({"geometry_preset": "curve"}, GEOMETRY_PRESET_CURVE, ATTENTION_GEOMETRY_CURVE, MLP_GEOMETRY_CURVE, BASIS_FAMILY_CHEBYSHEV),
            ({"geometry_preset": "mlp_block"}, GEOMETRY_PRESET_MLP_BLOCK, ATTENTION_GEOMETRY_CURVE, MLP_GEOMETRY_MLP_BLOCK, BASIS_FAMILY_CHEBYSHEV),
            ({"attention_geometry": "head_aware_block", "mlp_geometry": "curve"}, GEOMETRY_PRESET_HEAD_AWARE_BLOCK, ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK, MLP_GEOMETRY_CURVE, BASIS_FAMILY_CHEBYSHEV),
            ({"geometry_preset": "block"}, GEOMETRY_PRESET_BLOCK, ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK, MLP_GEOMETRY_MLP_BLOCK, BASIS_FAMILY_CHEBYSHEV),
        )
        for request, preset, attention, mlp, basis in cases:
            with self.subTest(request=request):
                resolved = resolve_compact_selectors(**request)
                self.assertEqual(resolved.geometry_preset, preset)
                self.assertEqual(resolved.attention_geometry, attention)
                self.assertEqual(resolved.mlp_geometry, mlp)
                self.assertEqual(resolved.basis_family, basis)

    def test_explicit_module_overrides_are_resolved_before_stage_support_validation(self) -> None:
        resolved = resolve_compact_selectors(attention_geometry="head_aware_block", mlp_geometry="mlp_block")
        self.assertEqual(resolved.geometry_preset, GEOMETRY_PRESET_BLOCK)
        resolved = resolve_compact_selectors(geometry_preset="mlp_block", attention_geometry="head_aware_block", basis_family="dct")
        self.assertEqual(resolved.basis_family, BASIS_FAMILY_DCT)

    def test_invalid_identity_values_and_stage5_unsupported_materializations_fail(self) -> None:
        stage4_training_config(geometry_preset="curve")
        stage4_training_config(attention_geometry="curve", mlp_geometry="curve")
        stage4_training_config(geometry_preset="mlp_block")
        stage4_training_config(attention_geometry="curve", mlp_geometry="mlp_block")
        stage4_training_config(attention_geometry="head_aware_block", mlp_geometry="curve")
        stage4_training_config(geometry_preset="block")
        stage4_training_config(attention_geometry="head_aware_block", mlp_geometry="mlp_block")
        for overrides in (
            {"geometry_preset": "legacy_sheet_col", "attention_geometry": "curve"},
            {"geometry_preset": "legacy_sheet_col", "mlp_geometry": "mlp_block"},
            {"geometry_preset": "legacy_sheet_col", "basis_family": "dct"},
            {"geometry_preset": "block", "basis_family": "dct"},
            {"geometry_preset": "block", "attention_geometry": "curve"},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, "supports only"):
                    stage4_training_config(**overrides)

    def test_dense_rejects_compact_fields_except_conventional(self) -> None:
        TrainingConfig(model_type="dense")
        TrainingConfig(model_type="dense", geometry_preset=GEOMETRY_PRESET_CONVENTIONAL)
        for overrides in (
            {"geometry_preset": "legacy_sheet_col"},
            {"geometry_preset": "curve"},
            {"attention_geometry": "curve"},
            {"mlp_geometry": "mlp_block"},
            {"basis_family": "chebyshev"},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    TrainingConfig(model_type="dense", **overrides)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
