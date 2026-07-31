# vvv THOG
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import torch

from sheet.bases import BASIS_FAMILY_CHEBYSHEV
from sheet.depth_materialisation_runtime import install_depth_materialisation_runtime
from sheet.depth_trajectory import DepthTrajectory
from sheet.geometry import SheetGeometryConfig
from sheet.recurrence_generators import BQRG_FAMILY, BQRG_VERSION
from sheet.semantic_materializer import ATTENTION_QUERY_WEIGHT


def _tiny_geometry() -> SheetGeometryConfig:
    return SheetGeometryConfig(
        n_layer=16,
        n_embd=8,
        n_head=2,
        depth_order=16,
        base_row_order=1,
        mlp_channel_order=1,
        o_attn_d_model=1,
        o_attn_qkv_per_channel=1,
        o_attn_out_per_channel=1,
        o_mlp_d_model=1,
        o_mlp_hidden=1,
        bias=True,
    )


class BqrgDepthMaterialisationRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        install_depth_materialisation_runtime()

    def test_bqrg_bypasses_fixed_basis_matmul_even_when_requested(self) -> None:
        with patch.dict(os.environ, {"THOG2_DEPTH_MATERIALISATION_MATMUL": "true"}, clear=False):
            trajectory = DepthTrajectory(
                _tiny_geometry(),
                runtime_dtype=torch.float32,
                basis_family=BQRG_FAMILY,
                basis_version=BQRG_VERSION,
                depth_compress_layer_norm_and_bias=False,
            )
        self.assertFalse(trajectory.depth_materialisation_matmul)
        generated = trajectory.materialize(ATTENTION_QUERY_WEIGHT, 15)
        self.assertEqual(generated.shape, (8, 8))
        self.assertTrue(torch.isfinite(generated).all())

    def test_fixed_basis_still_uses_matmul_when_requested(self) -> None:
        with patch.dict(os.environ, {"THOG2_DEPTH_MATERIALISATION_MATMUL": "true"}, clear=False):
            trajectory = DepthTrajectory(
                _tiny_geometry(),
                runtime_dtype=torch.float32,
                basis_family=BASIS_FAMILY_CHEBYSHEV,
                depth_compress_layer_norm_and_bias=False,
            )
        self.assertTrue(trajectory.depth_materialisation_matmul)
        generated = trajectory.materialize(ATTENTION_QUERY_WEIGHT, 15)
        self.assertEqual(generated.shape, (8, 8))
        self.assertTrue(torch.isfinite(generated).all())


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
