# vvv THOG
from __future__ import annotations

import unittest
from unittest import mock

import torch
from sheet.block_trajectory import BlockTrajectory
from sheet.compact_identity import GEOMETRY_PRESET_FULL_BLOCK
from sheet.model import SheetGPT, SheetGPTConfig
from sheet.semantic_materializer import (
    ATTENTION_KEY_WEIGHT,
    ATTENTION_OUTPUT_WEIGHT,
    ATTENTION_QUERY_WEIGHT,
    ATTENTION_VALUE_WEIGHT,
    LEGACY_ATTENTION_INPUT_BIAS,
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

    # vvv THOG regression test for the packed-QKV rematerialization elimination
    def test_05_attention_forward_materializes_packed_qkv_weight_and_bias_once_each(self) -> None:
        torch.manual_seed(4102)
        model = SheetGPT(self.full_block_config())
        inputs = torch.randn(2, 4, model.config.n_embd)
        original_materialize = model.trajectory.materialize
        original_materialize_vector = model.trajectory.materialize_vector
        with mock.patch.object(model.trajectory, "materialize", wraps=original_materialize) as materialize_spy:
            with mock.patch.object(model.trajectory, "materialize_vector", wraps=original_materialize_vector) as vector_spy:
                output = model._attention(inputs, layer_index=1)
        self.assertEqual(tuple(output.shape), tuple(inputs.shape))
        packed_weight_calls = [
            call
            for call in materialize_spy.call_args_list
            if call.args == (LEGACY_ATTENTION_INPUT_WEIGHT, 1)
        ]
        packed_bias_calls = [
            call
            for call in vector_spy.call_args_list
            if call.args == (LEGACY_ATTENTION_INPUT_BIAS, 1)
        ]
        self.assertEqual(len(packed_weight_calls), 1)
        self.assertEqual(len(packed_bias_calls), 1)
    # ^^^ THOG

    # vvv THOG regression tests for removing the semantic adapter from the model attention hot path
    def test_06_attention_hot_path_bypasses_semantic_reconstruction_adapter_and_materializes_packed_qkv_once(self) -> None:
        torch.manual_seed(4105)
        model = SheetGPT(self.full_block_config())
        inputs = torch.randn(2, 4, model.config.n_embd)
        original_materialize = model.trajectory.materialize
        original_materialize_vector = model.trajectory.materialize_vector
        with mock.patch.object(model.semantic_materializer, "reconstructed_attention_input_weight", side_effect=AssertionError("semantic weight adapter entered")):
            with mock.patch.object(model.semantic_materializer, "reconstructed_attention_input_bias", side_effect=AssertionError("semantic bias adapter entered")):
                with mock.patch.object(model.trajectory, "materialize", wraps=original_materialize) as materialize_spy:
                    with mock.patch.object(model.trajectory, "materialize_vector", wraps=original_materialize_vector) as vector_spy:
                        output = model._attention(inputs, layer_index=1)
        self.assertEqual(tuple(output.shape), tuple(inputs.shape))
        self.assertEqual(sum(call.args == (LEGACY_ATTENTION_INPUT_WEIGHT, 1) for call in materialize_spy.call_args_list), 1)
        self.assertEqual(sum(call.args == (LEGACY_ATTENTION_INPUT_BIAS, 1) for call in vector_spy.call_args_list), 1)

    def test_07_direct_trajectory_attention_hot_path_matches_legacy_adapter_outputs_and_gradients(self) -> None:
        torch.manual_seed(4106)
        reference = SheetGPT(self.full_block_config())
        candidate = SheetGPT(self.full_block_config())
        candidate.load_state_dict(reference.state_dict())
        inputs_reference = torch.randn(2, 4, reference.config.n_embd, requires_grad=True)
        inputs_candidate = inputs_reference.detach().clone().requires_grad_(True)

        def legacy_attention_input_weight(layer_index: int) -> torch.Tensor:
            return reference.semantic_materializer.reconstructed_attention_input_weight(layer_index)

        def legacy_attention_input_bias(layer_index: int) -> torch.Tensor:
            return reference.semantic_materializer.reconstructed_attention_input_bias(layer_index)

        reference_weight = legacy_attention_input_weight(1)
        reference_bias = legacy_attention_input_bias(1)
        candidate_weight = candidate.trajectory.materialize(LEGACY_ATTENTION_INPUT_WEIGHT, 1)
        candidate_bias = candidate.trajectory.materialize_vector(LEGACY_ATTENTION_INPUT_BIAS, 1)
        torch.testing.assert_close(candidate_weight, reference_weight, rtol=0.0, atol=0.0)
        torch.testing.assert_close(candidate_bias, reference_bias, rtol=0.0, atol=0.0)
        reference_output = torch.nn.functional.linear(inputs_reference, reference_weight, reference_bias)
        candidate_output = torch.nn.functional.linear(inputs_candidate, candidate_weight, candidate_bias)
        torch.testing.assert_close(candidate_output, reference_output, rtol=0.0, atol=0.0)
        reference_output.square().sum().backward()
        candidate_output.square().sum().backward()
        torch.testing.assert_close(inputs_candidate.grad, inputs_reference.grad, rtol=0.0, atol=0.0)
        for name in (ATTENTION_QUERY_WEIGHT, ATTENTION_KEY_WEIGHT, ATTENTION_VALUE_WEIGHT):
            torch.testing.assert_close(
                candidate.trajectory.coefficients[name].grad,
                reference.trajectory.coefficients[name].grad,
                rtol=0.0,
                atol=0.0,
            )
    # ^^^ THOG

    # vvv THOG exact forward and backward equivalence tests for vectorized per-head materialization
    def _explicit_per_head_materialization(self, trajectory: BlockTrajectory, name: str, layer_index: int) -> torch.Tensor:
        item = trajectory.family_metadata(name)
        coefficient = trajectory.coefficients[name]
        depth_row = trajectory.depth_basis[layer_index].to(coefficient)
        output_basis = trajectory.output_basis(name).to(coefficient)
        input_basis = trajectory.input_basis(name).to(coefficient)
        pieces = []
        for head_index in range(item.head_count):
            mixed = torch.einsum("p,pab->ab", depth_row, coefficient[head_index])
            pieces.append(output_basis @ mixed @ input_basis.transpose(0, 1))
        concatenate_dimension = 0 if item.attention_head_axis == "output" else 1
        return torch.cat(pieces, dim=concatenate_dimension)

    def test_08_vectorized_head_materialization_matches_explicit_per_head_reference_for_every_attention_family_and_layer(self) -> None:
        torch.manual_seed(4103)
        trajectory = BlockTrajectory(self.full_block_config().sheet_geometry(), runtime_dtype=torch.float64)
        with torch.no_grad():
            for name in (ATTENTION_QUERY_WEIGHT, ATTENTION_KEY_WEIGHT, ATTENTION_VALUE_WEIGHT, ATTENTION_OUTPUT_WEIGHT):
                trajectory.coefficients[name].normal_(mean=0.0, std=0.2)
        for layer_index in range(trajectory.config.n_layer):
            for name in (ATTENTION_QUERY_WEIGHT, ATTENTION_KEY_WEIGHT, ATTENTION_VALUE_WEIGHT, ATTENTION_OUTPUT_WEIGHT):
                with self.subTest(layer_index=layer_index, name=name):
                    vectorized = trajectory.materialize(name, layer_index)
                    reference = self._explicit_per_head_materialization(trajectory, name, layer_index)
                    torch.testing.assert_close(vectorized, reference, rtol=1.0e-12, atol=1.0e-12)

    def test_09_vectorized_head_materialization_preserves_coefficient_gradients_for_output_and_input_head_axes(self) -> None:
        torch.manual_seed(4104)
        trajectory = BlockTrajectory(self.full_block_config().sheet_geometry(), runtime_dtype=torch.float64)
        for name in (ATTENTION_QUERY_WEIGHT, ATTENTION_OUTPUT_WEIGHT):
            with self.subTest(name=name):
                coefficient = trajectory.coefficients[name]
                vectorized = trajectory.materialize(name, layer_index=2)
                vectorized_objective = vectorized.square().sum() + 0.25 * vectorized.sum()
                vectorized_gradient = torch.autograd.grad(vectorized_objective, coefficient, retain_graph=True)[0]
                reference = self._explicit_per_head_materialization(trajectory, name, layer_index=2)
                reference_objective = reference.square().sum() + 0.25 * reference.sum()
                reference_gradient = torch.autograd.grad(reference_objective, coefficient)[0]
                torch.testing.assert_close(vectorized_gradient, reference_gradient, rtol=1.0e-11, atol=1.0e-11)
    # ^^^ THOG


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
