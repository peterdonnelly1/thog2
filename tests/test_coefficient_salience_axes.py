# vvv THOG
from __future__ import annotations

import unittest

from sheet.coefficient_salience import discover_coefficient_banks, select_banks
from sheet.compact_identity import GEOMETRY_PRESET_FULL_BLOCK, GEOMETRY_PRESET_JPEG_LIKE_V1
from sheet.model import SheetGPT, SheetGPTConfig
from sheet.semantic_materializer import MLP_EXPANSION_WEIGHT


def _sheet_config(preset: str) -> SheetGPTConfig:
    return SheetGPTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=4,
        n_head=2,
        n_embd=8,
        depth_order=2,
        base_row_order=4,
        mlp_channel_order=4,
        o_attn_d_model=4,
        o_attn_qkv_per_channel=2,
        o_attn_out_per_channel=2,
        o_mlp_d_model=4,
        o_mlp_hidden=4,
        mlp_hidden_group_size=8,
        geometry_preset=preset,
        basis_family="chebyshev",
    )


class NonDepthScopeTest(unittest.TestCase):
    def test_jpeg_like_exposes_local_and_depth_orders(self):
        model = SheetGPT(_sheet_config(GEOMETRY_PRESET_JPEG_LIKE_V1))
        banks = discover_coefficient_banks(model)
        scopes = {bank.scope_name: bank for bank in banks}
        self.assertIn(f"{MLP_EXPANSION_WEIGHT}:local", scopes)
        self.assertIn(f"{MLP_EXPANSION_WEIGHT}:depth", scopes)
        self.assertEqual(scopes[f"{MLP_EXPANSION_WEIGHT}:local"].order_axis, 1)
        self.assertEqual(scopes[f"{MLP_EXPANSION_WEIGHT}:depth"].order_axis, 3)
        self.assertEqual(scopes[f"{MLP_EXPANSION_WEIGHT}:local"].order_count, 4)
        self.assertEqual(scopes[f"{MLP_EXPANSION_WEIGHT}:depth"].order_count, 2)

    def test_ambiguous_family_requires_axis_label(self):
        model = SheetGPT(_sheet_config(GEOMETRY_PRESET_FULL_BLOCK))
        banks = discover_coefficient_banks(model)
        with self.assertRaisesRegex(ValueError, "multiple order axes"):
            select_banks(banks, MLP_EXPANSION_WEIGHT)
        selected = select_banks(banks, f"{MLP_EXPANSION_WEIGHT}:output")
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].axis_label, "output")


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
