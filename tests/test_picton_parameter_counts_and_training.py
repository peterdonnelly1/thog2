# vvv THOG
from __future__ import annotations

import torch

from sheet.block_trajectory import BlockTrajectory
from sheet.compact_identity import GEOMETRY_PRESET_FULL_BLOCK
from sheet.geometry import SheetGeometryConfig
from sheet.model import SheetGPT, SheetGPTConfig
from sheet.semantic_materializer import (
    ATTENTION_OUTPUT_WEIGHT,
    ATTENTION_QUERY_WEIGHT,
    MLP_CONTRACTION_WEIGHT,
    MLP_EXPANSION_WEIGHT,
)


def picton_geometry(**overrides: int) -> SheetGeometryConfig:
    values = dict(
        n_layer=5,
        n_embd=12,
        n_head=3,
        depth_order=2,
        base_row_order=6,
        mlp_channel_order=11,
        o_attn_d_model=7,
        o_attn_qkv_per_channel=3,
        o_attn_out_per_channel=2,
        o_mlp_d_model=5,
        o_mlp_hidden=11,
        bias=False,
    )
    values.update(overrides)
    return SheetGeometryConfig(**values)


def test_picton_full_block_matrix_coefficient_count_matches_closed_form() -> None:
    config = picton_geometry()
    trajectory = BlockTrajectory(config, compact_attention=True, compact_mlp=True)

    qkv = 3 * config.n_head * config.depth_order * config.resolved_o_attn_qkv_per_channel * config.resolved_o_attn_d_model
    output = config.n_head * config.depth_order * config.resolved_o_attn_d_model * config.resolved_o_attn_out_per_channel
    mlp_expansion = config.depth_order * config.resolved_o_mlp_hidden * config.resolved_o_mlp_d_model
    mlp_contraction = config.depth_order * config.resolved_o_mlp_d_model * config.resolved_o_mlp_hidden
    expected = qkv + output + mlp_expansion + mlp_contraction

    assert qkv == 378
    assert output == 84
    assert mlp_expansion == 110
    assert mlp_contraction == 110
    assert expected == 682
    assert trajectory.matrix_sheet_parameter_count() == expected


def test_picton_each_order_has_the_expected_marginal_matrix_parameter_cost() -> None:
    base = picton_geometry()
    base_count = BlockTrajectory(base, compact_attention=True, compact_mlp=True).matrix_sheet_parameter_count()

    qkv_channel = BlockTrajectory(
        picton_geometry(o_attn_qkv_per_channel=4), compact_attention=True, compact_mlp=True
    ).matrix_sheet_parameter_count()
    assert qkv_channel - base_count == 3 * base.n_head * base.depth_order * base.resolved_o_attn_d_model

    output_channel = BlockTrajectory(
        picton_geometry(o_attn_out_per_channel=3), compact_attention=True, compact_mlp=True
    ).matrix_sheet_parameter_count()
    assert output_channel - base_count == base.n_head * base.depth_order * base.resolved_o_attn_d_model

    attention_model = BlockTrajectory(
        picton_geometry(o_attn_d_model=8), compact_attention=True, compact_mlp=True
    ).matrix_sheet_parameter_count()
    assert attention_model - base_count == (
        3 * base.n_head * base.depth_order * base.resolved_o_attn_qkv_per_channel
        + base.n_head * base.depth_order * base.resolved_o_attn_out_per_channel
    )

    mlp_model = BlockTrajectory(
        picton_geometry(o_mlp_d_model=6), compact_attention=True, compact_mlp=True
    ).matrix_sheet_parameter_count()
    assert mlp_model - base_count == 2 * base.depth_order * base.resolved_o_mlp_hidden

    mlp_hidden = BlockTrajectory(
        picton_geometry(o_mlp_hidden=12), compact_attention=True, compact_mlp=True
    ).matrix_sheet_parameter_count()
    assert mlp_hidden - base_count == 2 * base.depth_order * base.resolved_o_mlp_d_model


def test_picton_optimizer_step_updates_compact_coefficients_and_their_materialized_dense_weight() -> None:
    torch.manual_seed(9181)
    model = SheetGPT(
        SheetGPTConfig(
            block_size=6,
            vocab_size=32,
            n_layer=5,
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
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=1.0e-3,
        betas=(0.9, 0.95),
        device_type="cpu",
    )
    coefficient_before = model.trajectory.coefficients[ATTENTION_QUERY_WEIGHT].detach().clone()
    materialized_before = model.trajectory.materialize(ATTENTION_QUERY_WEIGHT, 1).detach().clone()

    idx = torch.randint(0, model.config.vocab_size, (2, 6))
    targets = torch.randint(0, model.config.vocab_size, (2, 6))
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(idx, targets)
    assert loss is not None
    loss.backward()
    optimizer.step()

    coefficient_after = model.trajectory.coefficients[ATTENTION_QUERY_WEIGHT].detach()
    materialized_after = model.trajectory.materialize(ATTENTION_QUERY_WEIGHT, 1).detach()
    assert not torch.equal(coefficient_before, coefficient_after)
    assert not torch.equal(materialized_before, materialized_after)


def test_picton_full_block_all_six_matrix_families_receive_finite_gradients() -> None:
    torch.manual_seed(9182)
    trajectory = BlockTrajectory(picton_geometry(), compact_attention=True, compact_mlp=True)
    loss = (
        trajectory.materialize(ATTENTION_QUERY_WEIGHT, 1).square().mean()
        + trajectory.materialize(ATTENTION_OUTPUT_WEIGHT, 1).square().mean()
        + trajectory.materialize(MLP_EXPANSION_WEIGHT, 1).square().mean()
        + trajectory.materialize(MLP_CONTRACTION_WEIGHT, 1).square().mean()
    )
    loss.backward()
    for name in (
        ATTENTION_QUERY_WEIGHT,
        ATTENTION_OUTPUT_WEIGHT,
        MLP_EXPANSION_WEIGHT,
        MLP_CONTRACTION_WEIGHT,
    ):
        gradient = trajectory.coefficients[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert float(gradient.abs().sum()) > 0.0, name
# ^^^ THOG
