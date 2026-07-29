# vvv THOG
from __future__ import annotations

import unittest

import torch

from sheet.coefficient_salience import (
    concentration_statistics,
    discover_depth_banks,
    select_banks,
    spearman_rho,
    zero_order_temporarily,
)
from sheet.compact_identity import (
    GEOMETRY_PRESET_DEPTH,
    GEOMETRY_PRESET_FULL_BLOCK,
    GEOMETRY_PRESET_HEAD_AWARE_BLOCK,
    GEOMETRY_PRESET_JPEG_LIKE_V1,
    GEOMETRY_PRESET_LEGACY_SHEET_COL,
    GEOMETRY_PRESET_MLP_BLOCK,
)
from sheet.model import SheetGPT, SheetGPTConfig
from sheet.semantic_materializer import (
    ATTENTION_QUERY_WEIGHT,
    LEGACY_ATTENTION_INPUT_WEIGHT,
    MLP_EXPANSION_WEIGHT,
)


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


class CoefficientDepthAxisTest(unittest.TestCase):
    def test_depth_axis_is_explicit_for_every_current_compact_geometry(self):
        expected = {
            GEOMETRY_PRESET_LEGACY_SHEET_COL: (LEGACY_ATTENTION_INPUT_WEIGHT, 1),
            GEOMETRY_PRESET_DEPTH: (ATTENTION_QUERY_WEIGHT, 2),
            GEOMETRY_PRESET_JPEG_LIKE_V1: (MLP_EXPANSION_WEIGHT, 3),
            GEOMETRY_PRESET_MLP_BLOCK: (MLP_EXPANSION_WEIGHT, 0),
            GEOMETRY_PRESET_HEAD_AWARE_BLOCK: (ATTENTION_QUERY_WEIGHT, 1),
            GEOMETRY_PRESET_FULL_BLOCK: (MLP_EXPANSION_WEIGHT, 0),
        }
        for preset, (family, expected_axis) in expected.items():
            with self.subTest(preset=preset):
                model = SheetGPT(_sheet_config(preset))
                banks = discover_depth_banks(model)
                by_name = {bank.name: bank for bank in banks}
                self.assertIn(family, by_name)
                self.assertEqual(by_name[family].order_axis, expected_axis)
                for bank in banks:
                    self.assertEqual(bank.order_count, 2)
                    self.assertEqual(bank.parameter.shape[bank.order_axis], 2)

    def test_public_depth_excludes_uncompressed_vectors(self):
        model = SheetGPT(_sheet_config(GEOMETRY_PRESET_DEPTH))
        names = {bank.name for bank in discover_depth_banks(model)}
        self.assertNotIn("ln_1_weight", names)
        self.assertNotIn("ln_2_weight", names)

    def test_family_scope_selects_exact_bank(self):
        model = SheetGPT(_sheet_config(GEOMETRY_PRESET_FULL_BLOCK))
        banks = discover_depth_banks(model)
        selected = select_banks(banks, MLP_EXPANSION_WEIGHT)
        self.assertEqual(tuple(bank.name for bank in selected), (MLP_EXPANSION_WEIGHT,))


class CoefficientAblationTest(unittest.TestCase):
    def test_zero_ablation_restores_exact_coefficients(self):
        model = SheetGPT(_sheet_config(GEOMETRY_PRESET_FULL_BLOCK))
        bank = select_banks(discover_depth_banks(model), MLP_EXPANSION_WEIGHT)[0]
        with torch.no_grad():
            bank.parameter.copy_(torch.randn_like(bank.parameter))
        before = bank.parameter.detach().clone()
        with zero_order_temporarily((bank,), 1):
            self.assertEqual(int(torch.count_nonzero(bank.order_slice(1)).item()), 0)
            self.assertGreater(int(torch.count_nonzero(bank.order_slice(0)).item()), 0)
        self.assertTrue(torch.equal(bank.parameter, before))


class SalienceStatisticsTest(unittest.TestCase):
    def test_concentration_detects_single_dominant_order(self):
        result = concentration_statistics((4.0, 0.0, -1.0, 0.0))
        self.assertAlmostEqual(result["effective_salience_dimension"], 1.0)
        self.assertAlmostEqual(result["effective_salience_dimension_ratio"], 0.25)
        self.assertAlmostEqual(result["top_quartile_positive_salience_fraction"], 1.0)

    def test_spearman_handles_matching_and_reversed_rankings(self):
        self.assertAlmostEqual(spearman_rho((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)), 1.0)
        self.assertAlmostEqual(spearman_rho((1.0, 2.0, 3.0), (6.0, 5.0, 4.0)), -1.0)


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
