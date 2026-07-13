# vvv THOG
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from sheet.compact_identity import GEOMETRY_PRESET_DEPTH
from sheet.depth_curve_diagnostics import curve_depth_summaries, write_depth_curve_local_viewer
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
    def depth_model(self) -> SheetGPT:
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
                    geometry_preset=GEOMETRY_PRESET_DEPTH,
                )
            )

    def test_01_depth_summaries_return_six_big_weight_families_spanning_all_logical_layers(self) -> None:
        model = self.depth_model()
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

    def test_02_depth_summaries_match_materialized_matrix_mean_and_std_when_sampling_all_elements(self) -> None:
        model = self.depth_model()
        summaries = curve_depth_summaries(model, sample_elements=10_000)
        for summary in summaries:
            with self.subTest(family=summary.family_name):
                for layer_index in summary.layers:
                    materialized = model.trajectory.materialize(summary.family_name, layer_index).float()
                    self.assertAlmostEqual(summary.means[layer_index], float(materialized.mean().item()), places=6)
                    self.assertAlmostEqual(summary.stds[layer_index], float(materialized.std(unbiased=False).item()), places=6)

    # vvv THOG
    def test_03_depth_summaries_include_histograms_for_each_layer_for_interactive_local_plotly_viewer(self) -> None:
        model = self.depth_model()
        summaries = curve_depth_summaries(model, sample_elements=7, histogram_bins=5)
        for summary in summaries:
            with self.subTest(family=summary.family_name):
                self.assertEqual(len(summary.histogram_edges), 6)
                self.assertEqual(len(summary.histogram_counts_by_layer), 3)
                for counts in summary.histogram_counts_by_layer:
                    self.assertEqual(len(counts), 5)
                    self.assertEqual(sum(counts), summary.sampled_elements)

    @unittest.skipUnless(importlib.util.find_spec("plotly") is not None, "plotly is not installed")
    def test_04_plotly_local_viewer_writes_structured_dashboard_and_per_family_html_files_for_easy_navigation(self) -> None:
        model = self.depth_model()
        summaries = curve_depth_summaries(model, sample_elements=7, histogram_bins=5)
        with tempfile.TemporaryDirectory() as directory:
            index_path = write_depth_curve_local_viewer(summaries, output_root=Path(directory), step=10, artifact_name="TEST_ARTIFACT")
            index_html = index_path.read_text(encoding="utf-8")
            self.assertTrue(index_path.exists())
            self.assertIn("TEST_ARTIFACT", index_html)
            self.assertIn("Weight-family dashboard", index_html)
            self.assertIn("Open full W_q chart", index_html)
            self.assertIn("step_000010_W_q.html", index_html)
            self.assertIn("<iframe src=\"step_000010_W_q.html\"", index_html)
            self.assertTrue((index_path.parent / "step_000010_W_q.html").exists())
    # ^^^ THOG


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
