# vvv THOG
from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

import run_thog2_owt  # noqa: F401  # <<< THOG install the public execution policy while protecting the pure-DEPTH GPT2 contract
from sheet.model import SheetGPT, SheetGPTConfig


class PureDepthGpt2ContractTests(unittest.TestCase):
    @staticmethod
    def _model() -> SheetGPT:
        torch.manual_seed(2468)
        model = SheetGPT(
            SheetGPTConfig(
                block_size=8,
                vocab_size=32,
                n_layer=2,
                n_head=2,
                n_embd=8,
                dropout=0.0,
                bias=True,
                depth_order=2,
                base_row_order=1,
                mlp_channel_order=1,
                o_attn_d_model=1,
                o_attn_qkv_per_channel=1,
                o_attn_out_per_channel=1,
                o_mlp_d_model=1,
                o_mlp_hidden=1,
                geometry_preset="depth",
                basis_family="chebyshev",
                depth_compress_layer_norm_and_bias=False,
            )
        )
        model.eval()
        return model

    def test_attention_is_standard_packed_qkv_multihead_attention(self) -> None:
        model = self._model()
        inputs = torch.randn(2, 5, model.config.n_embd)
        layer_index = 0
        actual = model._attention(inputs, layer_index)

        attention_weight = model.trajectory.materialize("attention_input_weight", layer_index)
        attention_bias = model.trajectory.materialize_vector("attention_input_bias", layer_index)
        query, key, value = F.linear(inputs, attention_weight, attention_bias).split(model.config.n_embd, dim=2)
        batch_size, sequence_length, embedding_width = inputs.shape
        head_width = embedding_width // model.config.n_head
        query = query.view(batch_size, sequence_length, model.config.n_head, head_width).transpose(1, 2)
        key = key.view(batch_size, sequence_length, model.config.n_head, head_width).transpose(1, 2)
        value = value.view(batch_size, sequence_length, model.config.n_head, head_width).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_width)
        output_weight = model.trajectory.materialize("attention_output_weight", layer_index)
        output_bias = model.trajectory.materialize_vector("attention_output_bias", layer_index)
        expected = F.linear(attended, output_weight, output_bias)

        torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-6)

    def test_mlp_is_standard_d_to_4d_gelu_to_d(self) -> None:
        model = self._model()
        inputs = torch.randn(2, 5, model.config.n_embd)
        layer_index = 1
        actual = model._mlp(inputs, layer_index)

        expansion_weight = model.trajectory.materialize("mlp_expansion_weight", layer_index)
        expansion_bias = model.trajectory.materialize_vector("mlp_expansion_bias", layer_index)
        hidden = F.gelu(F.linear(inputs, expansion_weight, expansion_bias))
        contraction_weight = model.trajectory.materialize("mlp_contraction_weight", layer_index)
        contraction_bias = model.trajectory.materialize_vector("mlp_contraction_bias", layer_index)
        expected = F.linear(hidden, contraction_weight, contraction_bias)

        torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
