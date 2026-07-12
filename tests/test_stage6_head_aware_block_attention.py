# vvv THOG
from __future__ import annotations

import unittest

import torch
from sheet.block_trajectory import BlockTrajectory
from sheet.compact_identity import GEOMETRY_PRESET_FULL_BLOCK
from sheet.model import SheetGPT, SheetGPTConfig
from sheet.semantic_materializer import (
    ATTENTION_KEY_WEIGHT,
    ATTENTION_OUTPUT_WEIGHT,
    ATTENTION_QUERY_WEIGHT,
    ATTENTION_VALUE_WEIGHT,
    LEGACY_ATTENTION_INPUT_WEIGHT,
    MLP_CONTRACTION_WEIGHT,
    MLP_EXPANSION_WEIGHT,
)


class Stage6HeadAwareBlockAttentionTests(unittest.TestCase):
    def full_block_config(self) -> SheetGPTConfig:
        return SheetGPTConfig(block_size=8, vocab_size=32, n_layer=4, n_head=2, n_embd=16, dropout=0.0, bias=True, depth_order=3, base_row_order=8, geometry_preset=GEOMETRY_PRESET_FULL_BLOCK)

    def test_01_head_aware_block_metadata_keeps_qkv_roles_and_attention_heads_explicit_without_smoothing_across_head_index(self) -> None:
        trajectory = BlockTrajectory(self.full_block_config().sheet_geometry(), runtime_dtype=torch.float32)
        report = {row["name"]: row for row in trajectory.family_report()}
        expected_shapes = {
            ATTENTION_QUERY_WEIGHT: (2, 3, 4, 8),
            ATTENTION_KEY_WEIGHT: (2, 3, 4, 8),
            ATTENTION_VALUE_WEIGHT: (2, 3, 4, 8),
            ATTENTION_OUTPUT_WEIGHT: (2, 3, 8, 4),
            MLP_EXPANSION_WEIGHT: (3, 32, 8),
            MLP_CONTRACTION_WEIGHT: (3, 8, 32),
        }
        for name, expected_shape in expected_shapes.items():
            with self.subTest(name=name):
                self.assertEqual(tuple(trajectory.coefficients[name].shape), expected_shape)
                self.assertEqual(report[name]["coefficient_shape"], expected_shape)
        for name in (ATTENTION_QUERY_WEIGHT, ATTENTION_KEY_WEIGHT, ATTENTION_VALUE_WEIGHT):
            with self.subTest(name=name):
                self.assertEqual(report[name]["attention_head_axis"], "output")
                self.assertEqual(report[name]["head_count"], 2)
                self.assertEqual(report[name]["head_dim"], 8)
                self.assertFalse(report[name]["basis_crosses_attention_head_boundary"])
        self.assertEqual(report[ATTENTION_OUTPUT_WEIGHT]["attention_head_axis"], "input")
        self.assertFalse(report[ATTENTION_OUTPUT_WEIGHT]["basis_crosses_attention_head_boundary"])
        self.assertEqual(report[MLP_EXPANSION_WEIGHT]["attention_head_axis"], "none")

    def test_02_qkv_head_aware_block_materialization_isolated_to_the_selected_output_head_and_reconstructs_legacy_packed_qkv(self) -> None:
        trajectory = BlockTrajectory(self.full_block_config().sheet_geometry(), runtime_dtype=torch.float32)
        with torch.no_grad():
            for name in (ATTENTION_QUERY_WEIGHT, ATTENTION_KEY_WEIGHT, ATTENTION_VALUE_WEIGHT):
                trajectory.coefficients[name].zero_()
            trajectory.coefficients[ATTENTION_QUERY_WEIGHT][0, :, 1, 2] = torch.tensor([1.0, -0.5, 2.0])
            trajectory.coefficients[ATTENTION_KEY_WEIGHT][1, :, 2, 3] = torch.tensor([-1.0, 0.25, 0.75])
        layer_index = 1
        query = trajectory.materialize(ATTENTION_QUERY_WEIGHT, layer_index)
        key = trajectory.materialize(ATTENTION_KEY_WEIGHT, layer_index)
        value = trajectory.materialize(ATTENTION_VALUE_WEIGHT, layer_index)
        self.assertGreater(float(query[:8].abs().sum().item()), 0.0)
        self.assertEqual(float(query[8:].abs().sum().item()), 0.0)
        self.assertEqual(float(key[:8].abs().sum().item()), 0.0)
        self.assertGreater(float(key[8:].abs().sum().item()), 0.0)
        self.assertEqual(float(value.abs().sum().item()), 0.0)
        packed = trajectory.materialize(LEGACY_ATTENTION_INPUT_WEIGHT, layer_index)
        reconstructed = torch.cat((query, key, value), dim=0)
        torch.testing.assert_close(packed, reconstructed, rtol=0.0, atol=0.0)
        torch.testing.assert_close(trajectory.direct_value(ATTENTION_QUERY_WEIGHT, layer_index, 3, 5), query[3, 5], rtol=0.0, atol=0.0)
        torch.testing.assert_close(trajectory.direct_value(LEGACY_ATTENTION_INPUT_WEIGHT, layer_index, 16 + 4, 6), key[4, 6], rtol=0.0, atol=0.0)

    def test_03_attention_output_head_aware_block_materialization_isolated_to_the_selected_input_head(self) -> None:
        trajectory = BlockTrajectory(self.full_block_config().sheet_geometry(), runtime_dtype=torch.float32)
        with torch.no_grad():
            trajectory.coefficients[ATTENTION_OUTPUT_WEIGHT].zero_()
            trajectory.coefficients[ATTENTION_OUTPUT_WEIGHT][1, :, 4, 2] = torch.tensor([0.5, -1.5, 2.5])
        layer_index = 2
        output = trajectory.materialize(ATTENTION_OUTPUT_WEIGHT, layer_index)
        self.assertEqual(float(output[:, :8].abs().sum().item()), 0.0)
        self.assertGreater(float(output[:, 8:].abs().sum().item()), 0.0)
        torch.testing.assert_close(trajectory.direct_value(ATTENTION_OUTPUT_WEIGHT, layer_index, 7, 12), output[7, 12], rtol=0.0, atol=0.0)

    def test_04_full_block_model_forward_backward_reaches_head_aware_attention_coefficients_and_mlp_block_coefficients(self) -> None:
        torch.manual_seed(4101)
        model = SheetGPT(self.full_block_config())
        self.assertIsInstance(model.trajectory, BlockTrajectory)
        inputs = torch.randint(0, model.config.vocab_size, (2, 4))
        targets = torch.randint(0, model.config.vocab_size, (2, 4))
        logits, loss = model(inputs, targets)
        self.assertEqual(tuple(logits.shape), (2, 4, model.config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name in (ATTENTION_QUERY_WEIGHT, ATTENTION_KEY_WEIGHT, ATTENTION_VALUE_WEIGHT, ATTENTION_OUTPUT_WEIGHT, MLP_EXPANSION_WEIGHT, MLP_CONTRACTION_WEIGHT):
            with self.subTest(name=name):
                gradient = model.trajectory.coefficients[name].grad
                self.assertIsNotNone(gradient)
                self.assertGreater(float(gradient.abs().sum().item()), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
