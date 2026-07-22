# vvv THOG
from __future__ import annotations

import torch

from sheet.block_trajectory import BlockTrajectory
from sheet.compact_identity import (
    GEOMETRY_PRESET_DEPTH,
    GEOMETRY_PRESET_FULL_BLOCK,
    GEOMETRY_PRESET_HEAD_AWARE_BLOCK,
    GEOMETRY_PRESET_LEGACY_SHEET_COL,
    GEOMETRY_PRESET_MLP_BLOCK,
)
from sheet.depth_trajectory import DepthTrajectory
from sheet.mlp_block_trajectory import MlpBlockTrajectory
from sheet.model import SheetGPT, SheetGPTConfig
from sheet.semantic_materializer import (
    ATTENTION_OUTPUT_WEIGHT,
    ATTENTION_QUERY_WEIGHT,
    LEGACY_ATTENTION_INPUT_WEIGHT,
    MLP_CONTRACTION_WEIGHT,
    MLP_EXPANSION_WEIGHT,
)
from sheet.trajectory import SheetTrajectory


def picton_model_config(preset: str) -> SheetGPTConfig:
    return SheetGPTConfig(
        block_size=6,
        vocab_size=32,
        n_layer=4,
        n_head=3,
        n_embd=12,
        dropout=0.0,
        bias=False,
        depth_order=2,
        base_row_order=6,
        mlp_channel_order=11,
        o_attn_d_model=7,
        o_attn_qkv_per_channel=3,
        o_attn_out_per_channel=2,
        o_mlp_d_model=5,
        o_mlp_hidden=11,
        geometry_preset=preset,
    )


def test_picton_legacy_sheet_col_still_uses_original_sheet_trajectory_and_original_matrix_shapes() -> None:
    model = SheetGPT(picton_model_config(GEOMETRY_PRESET_LEGACY_SHEET_COL))
    assert type(model.trajectory) is SheetTrajectory
    assert tuple(model.trajectory.coefficients[LEGACY_ATTENTION_INPUT_WEIGHT].shape) == (36, 2, 6)
    assert tuple(model.trajectory.coefficients[ATTENTION_OUTPUT_WEIGHT].shape) == (12, 2, 6)
    assert tuple(model.trajectory.coefficients[MLP_EXPANSION_WEIGHT].shape) == (48, 2, 6)
    assert tuple(model.trajectory.coefficients[MLP_CONTRACTION_WEIGHT].shape) == (12, 2, 11)


def test_picton_depth_uses_depth_trajectory_and_leaves_both_matrix_axes_full() -> None:
    model = SheetGPT(picton_model_config(GEOMETRY_PRESET_DEPTH))
    assert type(model.trajectory) is DepthTrajectory
    assert LEGACY_ATTENTION_INPUT_WEIGHT not in model.trajectory.coefficients
    assert tuple(model.trajectory.coefficients[ATTENTION_QUERY_WEIGHT].shape) == (12, 12, 2)
    assert tuple(model.trajectory.coefficients[ATTENTION_OUTPUT_WEIGHT].shape) == (12, 12, 2)
    assert tuple(model.trajectory.coefficients[MLP_EXPANSION_WEIGHT].shape) == (48, 12, 2)
    assert tuple(model.trajectory.coefficients[MLP_CONTRACTION_WEIGHT].shape) == (12, 48, 2)


def test_picton_each_final_compact_preset_selects_the_expected_trajectory_implementation() -> None:
    assert type(SheetGPT(picton_model_config(GEOMETRY_PRESET_LEGACY_SHEET_COL)).trajectory) is SheetTrajectory
    assert type(SheetGPT(picton_model_config(GEOMETRY_PRESET_DEPTH)).trajectory) is DepthTrajectory
    assert type(SheetGPT(picton_model_config(GEOMETRY_PRESET_MLP_BLOCK)).trajectory) is MlpBlockTrajectory

    head_aware = SheetGPT(picton_model_config(GEOMETRY_PRESET_HEAD_AWARE_BLOCK)).trajectory
    assert type(head_aware) is BlockTrajectory
    assert head_aware.compact_attention
    assert not head_aware.compact_mlp

    full_block = SheetGPT(picton_model_config(GEOMETRY_PRESET_FULL_BLOCK)).trajectory
    assert type(full_block) is BlockTrajectory
    assert full_block.compact_attention
    assert full_block.compact_mlp


def test_picton_every_compact_preset_materializes_the_exact_conventional_dense_weight_shapes() -> None:
    for preset in (
        GEOMETRY_PRESET_LEGACY_SHEET_COL,
        GEOMETRY_PRESET_DEPTH,
        GEOMETRY_PRESET_MLP_BLOCK,
        GEOMETRY_PRESET_HEAD_AWARE_BLOCK,
        GEOMETRY_PRESET_FULL_BLOCK,
    ):
        model = SheetGPT(picton_model_config(preset))
        with torch.no_grad():
            assert tuple(model.trajectory.materialize(LEGACY_ATTENTION_INPUT_WEIGHT, 1).shape) == (36, 12)
            assert tuple(model.trajectory.materialize(ATTENTION_OUTPUT_WEIGHT, 1).shape) == (12, 12)
            assert tuple(model.trajectory.materialize(MLP_EXPANSION_WEIGHT, 1).shape) == (48, 12)
            assert tuple(model.trajectory.materialize(MLP_CONTRACTION_WEIGHT, 1).shape) == (12, 48)


def test_picton_legacy_and_depth_forward_backward_regression_paths_remain_finite() -> None:
    for preset in (GEOMETRY_PRESET_LEGACY_SHEET_COL, GEOMETRY_PRESET_DEPTH):
        torch.manual_seed(7319)
        model = SheetGPT(picton_model_config(preset))
        idx = torch.randint(0, model.config.vocab_size, (2, 6))
        targets = torch.randint(0, model.config.vocab_size, (2, 6))
        logits, loss = model(idx, targets)
        assert tuple(logits.shape) == (2, 6, model.config.vocab_size)
        assert loss is not None
        assert torch.isfinite(loss)
        loss.backward()
        assert any(parameter.grad is not None for parameter in model.trajectory.coefficients.values())
        assert model.compact_state_violations() == ()
# ^^^ THOG
