# vvv THOG
from __future__ import annotations

import os
from unittest import mock

import torch

from sheet.compact_identity import GEOMETRY_PRESET_FULL_BLOCK
from sheet.model import SheetGPT, SheetGPTConfig
from sheet.semantic_materializer import LEGACY_ATTENTION_INPUT_BIAS, LEGACY_ATTENTION_INPUT_WEIGHT


def _config() -> SheetGPTConfig:
    return SheetGPTConfig(block_size=8, vocab_size=32, n_layer=2, n_head=2, n_embd=16, dropout=0.0, bias=True, depth_order=2, base_row_order=8, geometry_preset=GEOMETRY_PRESET_FULL_BLOCK)


def test_bypass_semantic_qkv_adapter_defaults_true() -> None:
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("THOG2_BYPASS_SEMANTIC_QKV_ADAPTER", None)
        assert _config().bypass_semantic_qkv_adapter is True


def test_bypass_semantic_qkv_adapter_false_selects_legacy_adapter_path() -> None:
    with mock.patch.dict(os.environ, {"THOG2_BYPASS_SEMANTIC_QKV_ADAPTER": "false"}):
        model = SheetGPT(_config())
    inputs = torch.randn(2, 4, model.config.n_embd)
    with mock.patch.object(model.semantic_materializer, "reconstructed_attention_input_weight", wraps=model.semantic_materializer.reconstructed_attention_input_weight) as weight_spy:
        with mock.patch.object(model.semantic_materializer, "reconstructed_attention_input_bias", wraps=model.semantic_materializer.reconstructed_attention_input_bias) as bias_spy:
            output = model._attention(inputs, layer_index=1)
    assert tuple(output.shape) == tuple(inputs.shape)
    weight_spy.assert_called_once_with(1)
    bias_spy.assert_called_once_with(1)


def test_bypass_semantic_qkv_adapter_true_bypasses_adapter_and_materializes_packed_qkv_once() -> None:
    with mock.patch.dict(os.environ, {"THOG2_BYPASS_SEMANTIC_QKV_ADAPTER": "true"}):
        model = SheetGPT(_config())
    inputs = torch.randn(2, 4, model.config.n_embd)
    original_materialize = model.trajectory.materialize
    original_materialize_vector = model.trajectory.materialize_vector
    with mock.patch.object(model.semantic_materializer, "reconstructed_attention_input_weight") as adapter_weight_spy:
        with mock.patch.object(model.semantic_materializer, "reconstructed_attention_input_bias") as adapter_bias_spy:
            with mock.patch.object(model.trajectory, "materialize", wraps=original_materialize) as materialize_spy:
                with mock.patch.object(model.trajectory, "materialize_vector", wraps=original_materialize_vector) as vector_spy:
                    output = model._attention(inputs, layer_index=1)
    assert tuple(output.shape) == tuple(inputs.shape)
    adapter_weight_spy.assert_not_called()
    adapter_bias_spy.assert_not_called()
    assert sum(call.args == (LEGACY_ATTENTION_INPUT_WEIGHT, 1) for call in materialize_spy.call_args_list) == 1
    assert sum(call.args == (LEGACY_ATTENTION_INPUT_BIAS, 1) for call in vector_spy.call_args_list) == 1


def test_bypass_and_adapter_paths_are_forward_and_gradient_equivalent() -> None:
    torch.manual_seed(4401)
    direct_config = _config()
    adapter_config = SheetGPTConfig(**{**direct_config.to_dict(), "bypass_semantic_qkv_adapter": False})
    direct_model = SheetGPT(direct_config)
    adapter_model = SheetGPT(adapter_config)
    adapter_model.load_state_dict(direct_model.state_dict())
    inputs_direct = torch.randn(2, 4, direct_model.config.n_embd, requires_grad=True)
    inputs_adapter = inputs_direct.detach().clone().requires_grad_(True)
    direct_output = direct_model._attention(inputs_direct, layer_index=1)
    adapter_output = adapter_model._attention(inputs_adapter, layer_index=1)
    torch.testing.assert_close(direct_output, adapter_output, rtol=0.0, atol=0.0)
    direct_output.square().sum().backward()
    adapter_output.square().sum().backward()
    torch.testing.assert_close(inputs_direct.grad, inputs_adapter.grad, rtol=0.0, atol=0.0)
    for name in direct_model.trajectory.coefficients:
        direct_gradient = direct_model.trajectory.coefficients[name].grad
        adapter_gradient = adapter_model.trajectory.coefficients[name].grad
        if direct_gradient is None and adapter_gradient is None:
            continue
        assert direct_gradient is not None
        assert adapter_gradient is not None
        torch.testing.assert_close(direct_gradient, adapter_gradient, rtol=0.0, atol=0.0)
# ^^^ THOG
