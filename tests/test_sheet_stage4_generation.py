# vvv THOG
from __future__ import annotations

import unittest

import torch

from sheet.generation import generate_tokens
from tests.stage4_test_support import stage4_model


class Stage4GenerationTests(unittest.TestCase):
    def test_s4_09_deterministic_generation(self) -> None:
        model = stage4_model()
        model.eval()
        prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
        first = generate_tokens(
            model,
            prompt,
            device=torch.device("cpu"),
            dtype="float32",
            max_new_tokens=4,
            temperature=0.9,
            top_k=8,
            seed=444,
        )
        second = generate_tokens(
            model,
            prompt,
            device=torch.device("cpu"),
            dtype="float32",
            max_new_tokens=4,
            temperature=0.9,
            top_k=8,
            seed=444,
        )
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(tuple(first.shape), (1, 7))

    def test_s4_10_invalid_controls_fail(self) -> None:
        model = stage4_model()
        model.eval()
        prompt = torch.tensor([[1, 2]], dtype=torch.long)
        with self.assertRaises(ValueError):
            generate_tokens(
                model,
                prompt,
                device=torch.device("cpu"),
                dtype="float32",
                max_new_tokens=-1,
            )
        with self.assertRaises(ValueError):
            generate_tokens(
                model,
                prompt,
                device=torch.device("cpu"),
                dtype="float32",
                max_new_tokens=1,
                temperature=0.0,
            )
        with self.assertRaises(ValueError):
            generate_tokens(
                model,
                prompt,
                device=torch.device("cpu"),
                dtype="float32",
                max_new_tokens=1,
                top_k=0,
            )


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
