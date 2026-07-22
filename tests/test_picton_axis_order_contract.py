# vvv THOG
from __future__ import annotations

from dataclasses import replace

from sheet.block_trajectory import BlockTrajectory
from sheet.geometry import SheetGeometryConfig
from sheet.mlp_block_trajectory import MlpBlockTrajectory
from sheet.semantic_materializer import (
    ATTENTION_KEY_WEIGHT,
    ATTENTION_OUTPUT_WEIGHT,
    ATTENTION_QUERY_WEIGHT,
    ATTENTION_VALUE_WEIGHT,
    MLP_CONTRACTION_WEIGHT,
    MLP_EXPANSION_WEIGHT,
)


PICTON_ATTENTION_FAMILIES = (
    ATTENTION_QUERY_WEIGHT,
    ATTENTION_KEY_WEIGHT,
    ATTENTION_VALUE_WEIGHT,
    ATTENTION_OUTPUT_WEIGHT,
)
PICTON_MLP_FAMILIES = (MLP_EXPANSION_WEIGHT, MLP_CONTRACTION_WEIGHT)


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


def picton_matrix_shapes(trajectory: BlockTrajectory) -> dict[str, tuple[int, ...]]:
    return {
        name: tuple(trajectory.coefficients[name].shape)
        for name in PICTON_ATTENTION_FAMILIES + PICTON_MLP_FAMILIES
        if name in trajectory.coefficients
    }


def test_picton_head_aware_block_uses_three_independent_attention_orders() -> None:
    trajectory = BlockTrajectory(picton_geometry(), compact_attention=True, compact_mlp=False)
    assert tuple(trajectory.coefficients[ATTENTION_QUERY_WEIGHT].shape) == (3, 2, 3, 7)
    assert tuple(trajectory.coefficients[ATTENTION_KEY_WEIGHT].shape) == (3, 2, 3, 7)
    assert tuple(trajectory.coefficients[ATTENTION_VALUE_WEIGHT].shape) == (3, 2, 3, 7)
    assert tuple(trajectory.coefficients[ATTENTION_OUTPUT_WEIGHT].shape) == (3, 2, 7, 2)

    query = trajectory.family_metadata(ATTENTION_QUERY_WEIGHT)
    output = trajectory.family_metadata(ATTENTION_OUTPUT_WEIGHT)
    assert query.head_count == 3
    assert query.head_dim == 4
    assert query.output_order == 3
    assert query.input_order == 7
    assert output.output_order == 7
    assert output.input_order == 2
    assert not query.basis_crosses_attention_head_boundary
    assert not output.basis_crosses_attention_head_boundary


def test_picton_mlp_block_uses_independent_model_and_hidden_orders() -> None:
    trajectory = MlpBlockTrajectory(picton_geometry())
    assert tuple(trajectory.coefficients[MLP_EXPANSION_WEIGHT].shape) == (2, 11, 5)
    assert tuple(trajectory.coefficients[MLP_CONTRACTION_WEIGHT].shape) == (2, 5, 11)

    expansion = trajectory.family_metadata(MLP_EXPANSION_WEIGHT)
    contraction = trajectory.family_metadata(MLP_CONTRACTION_WEIGHT)
    assert expansion.output_order == 11
    assert expansion.input_order == 5
    assert contraction.output_order == 5
    assert contraction.input_order == 11


def test_picton_full_block_contains_exact_union_of_attention_and_mlp_matrix_blocks() -> None:
    trajectory = BlockTrajectory(picton_geometry(), compact_attention=True, compact_mlp=True)
    assert picton_matrix_shapes(trajectory) == {
        ATTENTION_QUERY_WEIGHT: (3, 2, 3, 7),
        ATTENTION_KEY_WEIGHT: (3, 2, 3, 7),
        ATTENTION_VALUE_WEIGHT: (3, 2, 3, 7),
        ATTENTION_OUTPUT_WEIGHT: (3, 2, 7, 2),
        MLP_EXPANSION_WEIGHT: (2, 11, 5),
        MLP_CONTRACTION_WEIGHT: (2, 5, 11),
    }


def test_picton_each_attention_knob_changes_only_its_declared_matrix_dimensions() -> None:
    base = picton_geometry()
    base_shapes = picton_matrix_shapes(BlockTrajectory(base, compact_attention=True, compact_mlp=True))

    qkv_channel_shapes = picton_matrix_shapes(
        BlockTrajectory(replace(base, o_attn_qkv_per_channel=4), compact_attention=True, compact_mlp=True)
    )
    for name in (ATTENTION_QUERY_WEIGHT, ATTENTION_KEY_WEIGHT, ATTENTION_VALUE_WEIGHT):
        assert qkv_channel_shapes[name] == (3, 2, 4, 7)
    assert qkv_channel_shapes[ATTENTION_OUTPUT_WEIGHT] == base_shapes[ATTENTION_OUTPUT_WEIGHT]
    assert qkv_channel_shapes[MLP_EXPANSION_WEIGHT] == base_shapes[MLP_EXPANSION_WEIGHT]
    assert qkv_channel_shapes[MLP_CONTRACTION_WEIGHT] == base_shapes[MLP_CONTRACTION_WEIGHT]

    output_channel_shapes = picton_matrix_shapes(
        BlockTrajectory(replace(base, o_attn_out_per_channel=4), compact_attention=True, compact_mlp=True)
    )
    assert output_channel_shapes[ATTENTION_OUTPUT_WEIGHT] == (3, 2, 7, 4)
    for name in (ATTENTION_QUERY_WEIGHT, ATTENTION_KEY_WEIGHT, ATTENTION_VALUE_WEIGHT):
        assert output_channel_shapes[name] == base_shapes[name]
    assert output_channel_shapes[MLP_EXPANSION_WEIGHT] == base_shapes[MLP_EXPANSION_WEIGHT]
    assert output_channel_shapes[MLP_CONTRACTION_WEIGHT] == base_shapes[MLP_CONTRACTION_WEIGHT]

    model_shapes = picton_matrix_shapes(
        BlockTrajectory(replace(base, o_attn_d_model=9), compact_attention=True, compact_mlp=True)
    )
    for name in (ATTENTION_QUERY_WEIGHT, ATTENTION_KEY_WEIGHT, ATTENTION_VALUE_WEIGHT):
        assert model_shapes[name] == (3, 2, 3, 9)
    assert model_shapes[ATTENTION_OUTPUT_WEIGHT] == (3, 2, 9, 2)
    assert model_shapes[MLP_EXPANSION_WEIGHT] == base_shapes[MLP_EXPANSION_WEIGHT]
    assert model_shapes[MLP_CONTRACTION_WEIGHT] == base_shapes[MLP_CONTRACTION_WEIGHT]


def test_picton_each_mlp_knob_changes_only_its_declared_matrix_dimensions() -> None:
    base = picton_geometry()
    base_shapes = picton_matrix_shapes(BlockTrajectory(base, compact_attention=True, compact_mlp=True))

    model_shapes = picton_matrix_shapes(
        BlockTrajectory(replace(base, o_mlp_d_model=8), compact_attention=True, compact_mlp=True)
    )
    assert model_shapes[MLP_EXPANSION_WEIGHT] == (2, 11, 8)
    assert model_shapes[MLP_CONTRACTION_WEIGHT] == (2, 8, 11)
    for name in PICTON_ATTENTION_FAMILIES:
        assert model_shapes[name] == base_shapes[name]

    hidden_shapes = picton_matrix_shapes(
        BlockTrajectory(replace(base, o_mlp_hidden=13), compact_attention=True, compact_mlp=True)
    )
    assert hidden_shapes[MLP_EXPANSION_WEIGHT] == (2, 13, 5)
    assert hidden_shapes[MLP_CONTRACTION_WEIGHT] == (2, 5, 13)
    for name in PICTON_ATTENTION_FAMILIES:
        assert hidden_shapes[name] == base_shapes[name]


def test_picton_order_limits_are_checked_against_the_actual_semantic_axis_lengths() -> None:
    for overrides in (
        {"o_attn_d_model": 13},
        {"o_attn_qkv_per_channel": 5},
        {"o_attn_out_per_channel": 5},
        {"o_mlp_d_model": 13},
        {"o_mlp_hidden": 49},
    ):
        try:
            picton_geometry(**overrides)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected semantic order limit failure for {overrides}")
# ^^^ THOG
