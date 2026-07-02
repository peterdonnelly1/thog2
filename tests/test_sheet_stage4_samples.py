# vvv THOG
from __future__ import annotations

import unittest

import torch

from sheet.sampling import generate_samples
from tests.stage4_test_support import stage4_model


class Stage4SamplesTests(unittest.TestCase):
    def test_s4_10_multiple_continuations(self) -> None:
        model = stage4_model()
        model.eval()
        output = generate_samples(
            model,
            torch.tensor([[1, 2, 3]], dtype=torch.long),
            device=torch.device("cpu"),
            dtype="float32",
            num_samples=3,
            max_new_tokens=2,
            temperature=0.9,
            top_k=8,
            seed=100,
        )
        self.assertEqual(tuple(output.shape), (3, 5))

    def test_s4_10_zero_continuations_fail(self) -> None:
        model = stage4_model()
        model.eval()
        with self.assertRaises(ValueError):
            generate_samples(
                model,
                torch.tensor([[1, 2]], dtype=torch.long),
                device=torch.device("cpu"),
                dtype="float32",
                num_samples=0,
                max_new_tokens=1,
            )


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
