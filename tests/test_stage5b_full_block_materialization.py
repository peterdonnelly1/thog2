# vvv THOG
from __future__ import annotations

import unittest

import torch

from sheet.basis_kernel import DCT_BASIS_VERSION
from sheet.block_trajectory import FULL_BLOCK_MATRIX_FAMILIES, BlockTrajectory
from sheet.compact_identity import (
    ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK,
    BASIS_FAMILY_CHEBYSHEV,
    BASIS_FAMILY_DCT,
    FULL_BLOCK_MATERIALIZATION_VERSION,
    GEOMETRY_PRESET_FULL_BLOCK,
    MLP_GEOMETRY_MLP_BLOCK,
)
from sheet.model import SheetGPT, SheetGPTConfig
from sheet.semantic_materializer import (
    ATTENTION_KEY_WEIGHT,
    ATTENTION_OUTPUT_WEIGHT,
    ATTENTION_QUERY_WEIGHT,
    ATTENTION_VALUE_WEIGHT,
    MLP_CONTRACTION_WEIGHT,
    MLP_EXPANSION_WEIGHT,
)
from tests.stage4_test_support import stage4_training_config


class Stage5bFullBlockMaterializationTests(unittest.TestCase):
    def full_block_config(self) -> SheetGPTConfig:
        return SheetGPTConfig(block_size=8, vocab_size=32, n_layer=4, n_head=2, n_embd=16, dropout=0.0, bias=True, depth_order=3, base_row_order=8, geometry_preset=GEOMETRY_PRESET_FULL_BLOCK)

    def test_01_full_block_config_identity_resolves_to_head_aware_attention_plus_mlp_block_and_accepts_dct(self) -> None:
        config = stage4_training_config(geometry_preset=GEOMETRY_PRESET_FULL_BLOCK)
        identity = config.compact_identity_metadata()
        self.assertEqual(identity["geometry_preset"], GEOMETRY_PRESET_FULL_BLOCK)
        self.assertEqual(identity["attention_geometry"], ATTENTION_GEOMETRY_HEAD_AWARE_BLOCK)
        self.assertEqual(identity["mlp_geometry"], MLP_GEOMETRY_MLP_BLOCK)
        self.assertEqual(identity["basis_family"], BASIS_FAMILY_CHEBYSHEV)
        self.assertEqual(identity["materialization_version"], FULL_BLOCK_MATERIALIZATION_VERSION)
        dct_config = stage4_training_config(geometry_preset=GEOMETRY_PRESET_FULL_BLOCK, basis_family=BASIS_FAMILY_DCT)
        self.assertEqual(dct_config.compact_identity_metadata()["basis_version"], DCT_BASIS_VERSION)

    def test_02_full_block_trajectory_uses_head_aware_attention_coefficient_shapes_and_mlp_block_shapes(self) -> None:
        trajectory = BlockTrajectory(self.full_block_config().sheet_geometry(), runtime_dtype=torch.float32)
        expected_shapes = {
            ATTENTION_QUERY_WEIGHT: (2, 3, 4, 8),
            ATTENTION_KEY_WEIGHT: (2, 3, 4, 8),
            ATTENTION_VALUE_WEIGHT: (2, 3, 4, 8),
            ATTENTION_OUTPUT_WEIGHT: (2, 3, 8, 4),
            MLP_EXPANSION_WEIGHT: (3, 32, 8),
            MLP_CONTRACTION_WEIGHT: (3, 8, 32),
        }
        self.assertEqual(FULL_BLOCK_MATRIX_FAMILIES, tuple(expected_shapes))
        for name, expected_shape in expected_shapes.items():
            with self.subTest(name=name):
                self.assertEqual(tuple(trajectory.coefficients[name].shape), expected_shape)

    def test_03_full_block_model_forward_backward_reaches_all_head_aware_attention_and_mlp_block_coefficients(self) -> None:
        torch.manual_seed(4101)
        model = SheetGPT(self.full_block_config())
        inputs = torch.randint(0, model.config.vocab_size, (2, 4))
        targets = torch.randint(0, model.config.vocab_size, (2, 4))
        logits, loss = model(inputs, targets)
        self.assertEqual(tuple(logits.shape), (2, 4, model.config.vocab_size))
        self.assertIsNotNone(loss)
        loss.backward()
        for name in FULL_BLOCK_MATRIX_FAMILIES:
            with self.subTest(name=name):
                gradient = model.trajectory.coefficients[name].grad
                self.assertIsNotNone(gradient)
                self.assertGreater(float(gradient.abs().sum().item()), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
