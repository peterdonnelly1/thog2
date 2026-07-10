# vvv THOG
from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from sheet.compact_identity import GEOMETRY_PRESET_CURVE
from sheet.depth_curve_diagnostics import curve_depth_summaries
from sheet.model import SheetGPT, SheetGPTConfig
from sheet.semantic_materializer import (
    ATTENTION_KEY_WEIGHT,
    ATTENTION_OUTPUT_WEIGHT,
    ATTENTION_QUERY_WEIGHT,
    ATTENTION_VALUE_WEIGHT,
    MLP_CONTRACTION_WEIGHT,
    MLP_EXPANSION_WEIGHT,
)


class DepthCurveDiagnosticsTests(unittest.TestCase):
    def curve_model(self) -> SheetGPT:
        torch.manual_seed(8921)
        with patch.dict("os.environ", {"THOG2_MLP_CHANNEL_ORDER": "16"}):
            return SheetGPT(
                SheetGPTConfig(
                    block_size=8,
                    vocab_size=16,
                    n_layer=3,
                    n_head=2,
                    n_embd=4,
                    dropout=0.0,
                    bias=True,
                    depth_order=3,
                    base_row_order=4,
                    geometry_preset=GEOMETRY_PRESET_CURVE,
                )
            )

    def test_01_curve_depth_summaries_return_six_big_weight_families_spanning_all_logical_layers(self) -> None:
        model = self.curve_model()
        summaries = curve_depth_summaries(model, sample_elements=7)
        self.assertEqual(
            tuple(summary.family_name for summary in summaries),
            (
                ATTENTION_QUERY_WEIGHT,
                ATTENTION_KEY_WEIGHT,
                ATTENTION_VALUE_WEIGHT,
                ATTENTION_OUTPUT_WEIGHT,
                MLP_EXPANSION_WEIGHT,
                MLP_CONTRACTION_WEIGHT,
            ),
        )
        self.assertEqual(tuple(summary.label for summary in summaries), ("W_q", "W_k", "W_v", "W_o", "W_mlp_in", "W_mlp_out"))
        for summary in summaries:
            with self.subTest(family=summary.family_name):
                self.assertEqual(summary.layers, (0, 1, 2))
                self.assertEqual(len(summary.means), 3)
                self.assertEqual(len(summary.stds), 3)
                self.assertEqual(summary.sampled_elements, min(7, summary.total_elements))
                self.assertTrue(all(torch.isfinite(torch.tensor(summary.means))))
                self.assertTrue(all(torch.isfinite(torch.tensor(summary.stds))))

    def test_02_curve_depth_summaries_match_materialized_matrix_mean_and_std_when_sampling_all_elements(self) -> None:
        model = self.curve_model()
        summaries = curve_depth_summaries(model, sample_elements=10_000)
        for summary in summaries:
            with self.subTest(family=summary.family_name):
                for layer_index in summary.layers:
                    materialized = model.trajectory.materialize(summary.family_name, layer_index).float()
                    self.assertAlmostEqual(summary.means[layer_index], float(materialized.mean().item()), places=6)
                    self.assertAlmostEqual(summary.stds[layer_index], float(materialized.std(unbiased=False).item()), places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
