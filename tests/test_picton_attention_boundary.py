# vvv THOG
from __future__ import annotations

import torch
from torch.nn import functional as F

from sheet.compact_identity import GEOMETRY_PRESET_FULL_BLOCK
from sheet.model import SheetGPT, SheetGPTConfig
from sheet.semantic_materializer import (
    ATTENTION_KEY_WEIGHT,
    ATTENTION_OUTPUT_WEIGHT,
    ATTENTION_QUERY_WEIGHT,
    ATTENTION_VALUE_WEIGHT,
    LEGACY_ATTENTION_INPUT_WEIGHT,
)


def picton_attention_model() -> SheetGPT:
    torch.manual_seed(6127)
    return SheetGPT(
        SheetGPTConfig(
            block_size=6,
            vocab_size=32,
            n_layer=3,
            n_head=3,
            n_embd=12,
            dropout=0.0,
            bias=False,
            depth_order=2,
            base_row_order=6,
            o_attn_d_model=7,
            o_attn_qkv_per_channel=3,
            o_attn_out_per_channel=2,
            o_mlp_d_model=5,
            o_mlp_hidden=11,
            geometry_preset=GEOMETRY_PRESET_FULL_BLOCK,
        )
    )


def test_picton_materialization_presents_exact_dense_qkv_and_output_weight_shapes() -> None:
    model = picton_attention_model()
    layer_index = 1
    packed_qkv = model.semantic_materializer.reconstructed_attention_input_weight(layer_index)
    output_weight = model.trajectory.materialize(ATTENTION_OUTPUT_WEIGHT, layer_index)

    assert tuple(packed_qkv.shape) == (3 * model.config.n_embd, model.config.n_embd)
    assert tuple(output_weight.shape) == (model.config.n_embd, model.config.n_embd)
    assert not isinstance(packed_qkv, torch.nn.Parameter)
    assert not isinstance(output_weight, torch.nn.Parameter)


def test_picton_packed_qkv_is_exact_concatenation_of_the_three_materialized_roles() -> None:
    model = picton_attention_model()
    layer_index = 2
    query_weight = model.trajectory.materialize(ATTENTION_QUERY_WEIGHT, layer_index)
    key_weight = model.trajectory.materialize(ATTENTION_KEY_WEIGHT, layer_index)
    value_weight = model.trajectory.materialize(ATTENTION_VALUE_WEIGHT, layer_index)
    packed_qkv = model.trajectory.materialize(LEGACY_ATTENTION_INPUT_WEIGHT, layer_index)

    torch.testing.assert_close(
        packed_qkv,
        torch.cat((query_weight, key_weight, value_weight), dim=0),
        rtol=0.0,
        atol=0.0,
    )


def test_picton_packed_linear_projection_matches_three_conventional_role_projections_to_float_rounding() -> None:
    model = picton_attention_model()
    layer_index = 0
    inputs = torch.randn(2, 5, model.config.n_embd)
    packed_weight = model.trajectory.materialize(LEGACY_ATTENTION_INPUT_WEIGHT, layer_index)
    packed_query, packed_key, packed_value = F.linear(inputs, packed_weight).split(model.config.n_embd, dim=-1)

    for packed, family in (
        (packed_query, ATTENTION_QUERY_WEIGHT),
        (packed_key, ATTENTION_KEY_WEIGHT),
        (packed_value, ATTENTION_VALUE_WEIGHT),
    ):
        separate = F.linear(inputs, model.trajectory.materialize(family, layer_index))
        torch.testing.assert_close(packed, separate, rtol=1.0e-5, atol=1.0e-7)


def test_picton_attention_calls_unmodified_scaled_dot_product_attention_with_dense_head_tensors(monkeypatch) -> None:
    model = picton_attention_model().eval()
    original_sdpa = F.scaled_dot_product_attention
    captured: dict[str, object] = {}

    def recording_sdpa(query, key, value, **kwargs):
        captured["query_shape"] = tuple(query.shape)
        captured["key_shape"] = tuple(key.shape)
        captured["value_shape"] = tuple(value.shape)
        captured["kwargs"] = dict(kwargs)
        return original_sdpa(query, key, value, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", recording_sdpa)
    inputs = torch.randn(2, 5, model.config.n_embd)
    output = model._attention(inputs, layer_index=1)

    assert tuple(output.shape) == (2, 5, model.config.n_embd)
    assert captured["query_shape"] == (2, model.config.n_head, 5, model.config.n_embd // model.config.n_head)
    assert captured["key_shape"] == captured["query_shape"]
    assert captured["value_shape"] == captured["query_shape"]
    assert captured["kwargs"] == {
        "attn_mask": None,
        "dropout_p": 0.0,
        "is_causal": True,
    }


def test_picton_attention_loss_backpropagates_through_materialized_weights_to_all_compact_attention_families() -> None:
    model = picton_attention_model().train()
    idx = torch.randint(0, model.config.vocab_size, (2, 6))
    targets = torch.randint(0, model.config.vocab_size, (2, 6))
    _, loss = model(idx, targets)
    assert loss is not None
    assert torch.isfinite(loss)
    loss.backward()

    for name in (
        ATTENTION_QUERY_WEIGHT,
        ATTENTION_KEY_WEIGHT,
        ATTENTION_VALUE_WEIGHT,
        ATTENTION_OUTPUT_WEIGHT,
    ):
        gradient = model.trajectory.coefficients[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert float(gradient.abs().sum()) > 0.0, name


def test_picton_optimizer_contains_compact_coefficients_and_no_materialized_dense_attention_parameter() -> None:
    model = picton_attention_model()
    named_parameters = dict(model.named_parameters())
    assert "attention_weight" not in named_parameters
    assert "attention_output_weight" not in named_parameters
    assert all(not name.endswith("materialized_weight") for name in named_parameters)

    grouped_names = {
        name
        for group in model.optimizer_parameter_groups(weight_decay=0.1)
        for name in group["parameter_names"]
    }
    for family in (
        ATTENTION_QUERY_WEIGHT,
        ATTENTION_KEY_WEIGHT,
        ATTENTION_VALUE_WEIGHT,
        ATTENTION_OUTPUT_WEIGHT,
    ):
        assert f"trajectory.coefficients.{family}" in grouped_names
# ^^^ THOG
