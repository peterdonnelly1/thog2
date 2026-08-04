# vvv THOG
from __future__ import annotations

import unittest

from sheet.model import SheetGPTConfig
from sheet.stage6_diagnostics import generated_weight_report
from sheet.training_model import TrainingSheetGPT


class PlasticDepthDiagnosticsTests(unittest.TestCase):
    def test_generated_weight_report_uses_active_layers_for_learned_count(self) -> None:
        model = TrainingSheetGPT(
            SheetGPTConfig(
                block_size=8,
                vocab_size=32,
                n_layer=8,
                n_head=2,
                n_embd=16,
                dropout=0.0,
                bias=True,
                depth_order=4,
                geometry_preset="depth",
                basis_family="chebyshev",
                plastic__enabled=True,
                plastic__layers_to_sample=None,
                plastic__do_learn_layer_count=True,
                plastic__initial_layer_count=4,
                plastic__max_permitted_layers=8,
                plastic__freeze_geometry_during_warmup=False,
            )
        )
        model.set_plastic_depth_active_layer_count(5)

        report = generated_weight_report(model)

        self.assertEqual(tuple(report), ("0", "2", "4"))
        self.assertNotIn("7", report)


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
