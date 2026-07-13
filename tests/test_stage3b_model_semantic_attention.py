# vvv THOG
from __future__ import annotations

import copy
import json
import math
import unittest
from unittest import mock
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from sheet.model import SheetGPT, SheetGPTConfig
from sheet.semantic_materializer import (
    LEGACY_ATTENTION_INPUT_BIAS,
    LEGACY_ATTENTION_INPUT_WEIGHT,
    LegacySheetColMaterializer,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "stage0_legacy_sheet_col_fixture.json"


def legacy_attention_reference(model: SheetGPT, inputs: Tensor, layer_index: int) -> Tensor:
    batch_size, sequence_length, embedding_width = inputs.shape
    attention_weight = model.trajectory.materialize(LEGACY_ATTENTION_INPUT_WEIGHT, layer_index)
    attention_bias: Optional[Tensor] = None
    if model.config.bias:
        attention_bias = model.trajectory.materialize_vector(LEGACY_ATTENTION_INPUT_BIAS, layer_index)
    query, key, value = F.linear(inputs, attention_weight, attention_bias).split(model.config.n_embd, dim=2)
    head_width = embedding_width // model.config.n_head
    key = key.view(batch_size, sequence_length, model.config.n_head, head_width).transpose(1, 2)
    query = query.view(batch_size, sequence_length, model.config.n_head, head_width).transpose(1, 2)
    value = value.view(batch_size, sequence_length, model.config.n_head, head_width).transpose(1, 2)
    if hasattr(F, "scaled_dot_product_attention"):
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=model.config.dropout if model.training else 0.0, is_causal=True)
    else:
        scores = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(head_width))
        causal_mask = torch.tril(torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=inputs.device))
        scores = scores.masked_fill(~causal_mask.view(1, 1, sequence_length, sequence_length), float("-inf"))
        probabilities = F.softmax(scores, dim=-1)
        probabilities = F.dropout(probabilities, p=model.config.dropout, training=model.training)
        attended = probabilities @ value
    attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_width)
    output_weight = model.trajectory.materialize("attention_output_weight", layer_index)
    output_bias = model._optional_bias("attention_output_bias", layer_index)
    projected = F.linear(attended, output_weight, output_bias)
    return F.dropout(projected, p=model.config.dropout, training=model.training)


class SpySemanticMaterializer:
    def __init__(self, weight: Tensor, bias: Optional[Tensor]) -> None:
        self.weight = weight
        self.bias = bias
        self.weight_calls = 0
        self.bias_calls = 0

    def reconstructed_attention_input_weight(self, layer_index: int) -> Tensor:
        self.weight_calls += 1
        return self.weight

    def reconstructed_attention_input_bias(self, layer_index: int) -> Tensor:
        self.bias_calls += 1
        if self.bias is None:
            raise AssertionError("bias was requested for a bias-free model")
        return self.bias


class Stage3bModelSemanticAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with FIXTURE_PATH.open("r", encoding="utf-8") as handle:
            cls.fixture: Dict[str, Any] = json.load(handle)

    def config(self, **overrides: Any) -> SheetGPTConfig:
        values = dict(self.fixture["config"])
        values.update(overrides)
        return SheetGPTConfig(**values)

    def model(self, **overrides: Any) -> SheetGPT:
        torch.manual_seed(self.fixture["seed"])
        model = SheetGPT(self.config(**overrides))
        model.eval()
        return model

    def test_sheetgpt_owns_semantic_materializer_wrapping_actual_trajectory(self) -> None:
        model = self.model()
        self.assertTrue(hasattr(model, "semantic_materializer"))
        self.assertIsInstance(model.semantic_materializer, LegacySheetColMaterializer)
        self.assertIs(model.semantic_materializer.trajectory, model.trajectory)
        torch.testing.assert_close(
            model.semantic_materializer.reconstructed_attention_input_weight(2),
            model.trajectory.materialize(LEGACY_ATTENTION_INPUT_WEIGHT, 2),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            model.semantic_materializer.reconstructed_attention_input_bias(2),
            model.trajectory.materialize_vector(LEGACY_ATTENTION_INPUT_BIAS, 2),
            rtol=0.0,
            atol=0.0,
        )

    # vvv THOG update the former semantic-boundary regressions for the direct trajectory hot path
    # def test_attention_path_uses_semantic_qkv_boundary_not_direct_legacy_qkv_calls(self) -> None:
    #     ...
    def test_attention_path_uses_direct_packed_qkv_trajectory_calls_and_bypasses_semantic_adapter(self) -> None:
        model = self.model()
        inputs = torch.randn(2, 4, model.config.n_embd)
        weight = model.trajectory.materialize(LEGACY_ATTENTION_INPUT_WEIGHT, 1)
        bias = model.trajectory.materialize_vector(LEGACY_ATTENTION_INPUT_BIAS, 1)
        spy = SpySemanticMaterializer(weight, bias)
        model.semantic_materializer = spy
        original_materialize = model.trajectory.materialize
        original_materialize_vector = model.trajectory.materialize_vector
        with mock.patch.object(model.trajectory, "materialize", wraps=original_materialize) as materialize_spy:
            with mock.patch.object(model.trajectory, "materialize_vector", wraps=original_materialize_vector) as vector_spy:
                model._attention(inputs, 1)
        self.assertEqual(sum(call.args == (LEGACY_ATTENTION_INPUT_WEIGHT, 1) for call in materialize_spy.call_args_list), 1)
        self.assertEqual(sum(call.args == (LEGACY_ATTENTION_INPUT_BIAS, 1) for call in vector_spy.call_args_list), 1)
        self.assertEqual(spy.weight_calls, 0)
        self.assertEqual(spy.bias_calls, 0)

    # def test_bias_false_attention_path_does_not_request_semantic_bias(self) -> None:
    #     ...
    def test_bias_false_attention_path_materializes_direct_weight_once_and_no_packed_bias(self) -> None:
        model = self.model(bias=False)
        inputs = torch.randn(2, 4, model.config.n_embd)
        weight = model.trajectory.materialize(LEGACY_ATTENTION_INPUT_WEIGHT, 1)
        spy = SpySemanticMaterializer(weight, None)
        model.semantic_materializer = spy
        original_materialize = model.trajectory.materialize
        original_materialize_vector = model.trajectory.materialize_vector
        with mock.patch.object(model.trajectory, "materialize", wraps=original_materialize) as materialize_spy:
            with mock.patch.object(model.trajectory, "materialize_vector", wraps=original_materialize_vector) as vector_spy:
                model._attention(inputs, 1)
        self.assertEqual(sum(call.args == (LEGACY_ATTENTION_INPUT_WEIGHT, 1) for call in materialize_spy.call_args_list), 1)
        self.assertEqual(sum(call.args == (LEGACY_ATTENTION_INPUT_BIAS, 1) for call in vector_spy.call_args_list), 0)
        self.assertEqual(spy.weight_calls, 0)
        self.assertEqual(spy.bias_calls, 0)
    # ^^^ THOG

    def test_attention_output_forward_loss_and_backward_match_legacy_reference(self) -> None:
        model = self.model()
        reference = copy.deepcopy(model)
        inputs = torch.randn(2, 4, model.config.n_embd)
        semantic_attention = model._attention(inputs, 2)
        legacy_attention = legacy_attention_reference(reference, inputs, 2)
        torch.testing.assert_close(semantic_attention, legacy_attention, rtol=0.0, atol=0.0)

        idx = torch.randint(0, model.config.vocab_size, (2, 4))
        targets = torch.randint(0, model.config.vocab_size, (2, 4))
        logits, loss = model(idx, targets)
        reference_logits, reference_loss = reference(idx, targets)
        torch.testing.assert_close(logits, reference_logits, rtol=0.0, atol=0.0)
        torch.testing.assert_close(loss, reference_loss, rtol=0.0, atol=0.0)
        loss.backward()
        gradient = model.trajectory.coefficients[LEGACY_ATTENTION_INPUT_WEIGHT].grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum().item()), 0.0)

    def test_optimizer_report_and_compact_state_are_unchanged(self) -> None:
        model = self.model()
        report = model.parameter_report()
        self.assertEqual(report["persistent_parameters"], self.fixture["parameter_report"]["persistent_parameters"])
        self.assertEqual(report["sheet_coefficients"], self.fixture["parameter_report"]["sheet_coefficients"])
        self.assertEqual(model.compact_state_violations(), ())
        groups = model.optimizer_parameter_groups(0.1)
        grouped_names = tuple(tuple(group["parameter_names"]) for group in groups)
        self.assertTrue(any("trajectory.coefficients.attention_input_weight" in group for group in grouped_names))
        self.assertFalse(any("semantic_materializer" in name for group in grouped_names for name in group))


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
