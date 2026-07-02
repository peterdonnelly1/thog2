# vvv THOG
from __future__ import annotations

import math
import unittest

from sheet.stage4_summary import summarize_sheet
from tests.stage4_test_support import stage4_batch, stage4_model


class Stage4SummaryTests(unittest.TestCase):
    def test_s4_11_summaries_are_finite_and_detached(self) -> None:
        model = stage4_model(checkpoint_segment_size=2)
        model.train()
        inputs, targets = stage4_batch()
        _, loss = model(inputs, targets)
        assert loss is not None
        loss.backward()
        gradients_before = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        summary = summarize_sheet(model, (0, model.config.n_layer - 1))
        self.assertEqual(
            set(summary),
            {"orders", "derivatives", "generated"},
        )
        for family in summary["orders"].values():
            values = [
                family["coefficient_rms"],
                *family["depth_order_rms"],
                *family["row_order_rms"],
            ]
            self.assertTrue(all(math.isfinite(value) for value in values))
        self.assertTrue(
            all(math.isfinite(value) for value in summary["derivatives"].values())
        )
        for samples in summary["generated"].values():
            for sample in samples:
                self.assertTrue(
                    all(
                        math.isfinite(sample[key])
                        for key in (
                            "mean",
                            "rms",
                            "standard_deviation",
                            "maximum_absolute",
                            "endpoint_to_interior_rms_ratio",
                        )
                    )
                )
        for name, expected in gradients_before.items():
            self.assertTrue(expected.equal(dict(model.named_parameters())[name].grad))


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
