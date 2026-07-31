# vvv THOG
from __future__ import annotations

import unittest

import torch

import sheet.recurrence_generators.bqrg as bqrg_module


class BqrgChunkedBackwardTests(unittest.TestCase):
    def test_chunked_backward_matches_reference_across_boundaries(self) -> None:
        torch.manual_seed(31)
        original_chunk_size = bqrg_module.BQRG_BACKWARD_CHUNK_TRAJECTORIES
        try:
            bqrg_module.BQRG_BACKWARD_CHUNK_TRAJECTORIES = 2
            chunked_parameters = (
                torch.randn(5, bqrg_module.BQRG_PERSISTENT_WIDTH, dtype=torch.float64) * 0.05
            ).requires_grad_()
            reference_parameters = chunked_parameters.detach().clone().requires_grad_()

            chunked = bqrg_module.materialize_bqrg_at(chunked_parameters, 15)
            reference = bqrg_module.materialize_bqrg_sequence(reference_parameters, 16)[..., 15]
            upstream = torch.tensor([0.5, -0.25, 1.0, -1.5, 0.75], dtype=torch.float64)
            chunked.backward(upstream)
            reference.backward(upstream)

            self.assertTrue(torch.allclose(chunked, reference, atol=1.0e-12, rtol=1.0e-12))
            self.assertIsNotNone(chunked_parameters.grad)
            self.assertIsNotNone(reference_parameters.grad)
            self.assertTrue(
                torch.allclose(
                    chunked_parameters.grad,
                    reference_parameters.grad,
                    atol=1.0e-10,
                    rtol=1.0e-10,
                )
            )
        finally:
            bqrg_module.BQRG_BACKWARD_CHUNK_TRAJECTORIES = original_chunk_size


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
