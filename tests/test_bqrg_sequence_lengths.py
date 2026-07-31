# vvv THOG
from __future__ import annotations

import unittest

import torch

from sheet.recurrence_generators import BQRG_FAMILY, BQRG_PERSISTENT_WIDTH, get_recurrence_generator_definition, materialize_bqrg_sequence


class BqrgSequenceLengthTests(unittest.TestCase):
    def test_representative_lengths_remain_finite(self) -> None:
        definition = get_recurrence_generator_definition(BQRG_FAMILY)
        parameters = torch.empty(8, BQRG_PERSISTENT_WIDTH, dtype=torch.float32)
        torch.manual_seed(17)
        definition.initialize_parameters(parameters, "depth_matrix_normal", 0.02, 32)
        for length in (32, 64, 768):
            generated = materialize_bqrg_sequence(parameters, length)
            self.assertEqual(generated.shape, (8, length))
            self.assertTrue(torch.isfinite(generated).all())


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
