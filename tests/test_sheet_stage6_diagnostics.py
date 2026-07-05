# vvv THOG
from __future__ import annotations

import math
import unittest

import torch

from sheet.model import SheetGPTConfig
from sheet.stage6_diagnostics import coefficient_utilization_report
from sheet.stage6_diagnostics import generated_weight_report
from sheet.stage6_diagnostics import gradient_report
from sheet.training_model import TrainingSheetGPT


class Stage6DiagnosticsTests(unittest.TestCase):
    def make_model(self) -> TrainingSheetGPT:
        torch.manual_seed(6101)
        config = SheetGPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=2,
            n_head=2,
            n_embd=8,
            dropout=0.0,
            bias=True,
            depth_order=2,
            base_row_order=4,
        )
        return TrainingSheetGPT(config)

    def test_s6_07_order_energy_is_finite(self) -> None:
        model = self.make_model()
        report = coefficient_utilization_report(model)
        expected = {item.name for item in model.trajectory.metadata}
        self.assertEqual(set(report), expected)
        for row in report.values():
            self.assertTrue(math.isfinite(row["coefficient_rms"]))
            for key in (
                "depth_order_energy_fraction",
                "row_order_energy_fraction",
            ):
                total = sum(row[key])
                normalized = abs(total) < 1.0e-12
                normalized = normalized or abs(total - 1.0) < 1.0e-12
                self.assertTrue(normalized)

    def test_s6_08_gradient_and_weight_statistics_are_finite(self) -> None:
        model = self.make_model()
        inputs = torch.arange(16, dtype=torch.long).view(2, 8) % 32
        expected = torch.roll(inputs, shifts=-1, dims=1)
        _, loss = model(inputs, expected)
        self.assertIsNotNone(loss)
        loss.backward()
        gradients = gradient_report(model)
        self.assertTrue(all(
            math.isfinite(row["gradient_l2_norm"])
            for row in gradients.values()
        ))
        generated = generated_weight_report(model)
        self.assertEqual(set(generated), {"0", "1"})
        self.assertTrue(all(
            math.isfinite(value)
            for layer in generated.values()
            for family in layer.values()
            for value in family.values()
        ))


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
