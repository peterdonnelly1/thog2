# vvv THOG
from __future__ import annotations

from pathlib import Path

from sheet.block_trajectory import BlockTrajectory
from sheet.geometry import SheetGeometryConfig
from sheet.run_config import OwtRunConfig
from sheet.semantic_materializer import MLP_CONTRACTION_WEIGHT, MLP_EXPANSION_WEIGHT
from sheet.training_model_factory import build_training_model


def test_stage8_training_factory_passes_mlp_channel_order_into_actual_block_geometry(monkeypatch) -> None:
    monkeypatch.delenv("THOG2_MLP_CHANNEL_ORDER", raising=False)
    config = OwtRunConfig(
        model_type="sheet",
        geometry_preset="block",
        max_iters=20,
        warmup_iters=1,
        n_layer=4,
        n_head=2,
        n_embd=16,
        depth_order=3,
        base_row_order=8,
        mlp_channel_order=6,
        device="cpu",
        dtype="float32",
    )
    training = config.to_training_config(vocab_size=64, world_size=1, out_dir=Path("out"))
    model = build_training_model(training)
    assert isinstance(model.trajectory, BlockTrajectory)
    metadata = {item.name: item for item in model.trajectory.block_metadata}
    assert metadata[MLP_EXPANSION_WEIGHT].output_order == 6
    assert metadata[MLP_CONTRACTION_WEIGHT].input_order == 6


def test_stage8_mixed_trajectories_treat_nested_curve_fallback_coefficients_as_compact_state(monkeypatch) -> None:
    monkeypatch.delenv("THOG2_MLP_CHANNEL_ORDER", raising=False)
    configs = (
        OwtRunConfig(
            model_type="sheet",
            geometry_preset="mlp_block",
            max_iters=20,
            warmup_iters=1,
            n_layer=4,
            n_head=2,
            n_embd=16,
            depth_order=3,
            base_row_order=8,
            mlp_channel_order=6,
            device="cpu",
            dtype="float32",
        ),
        OwtRunConfig(
            model_type="sheet",
            geometry_preset=None,
            attention_geometry="head_aware_block",
            mlp_geometry="curve",
            max_iters=20,
            warmup_iters=1,
            n_layer=4,
            n_head=2,
            n_embd=16,
            depth_order=3,
            base_row_order=8,
            mlp_channel_order=6,
            device="cpu",
            dtype="float32",
        ),
    )
    for config in configs:
        training = config.to_training_config(vocab_size=64, world_size=1, out_dir=Path("out"))
        model = build_training_model(training)
        assert model.compact_state_violations() == ()


def test_stage8_sheet_geometry_keeps_legacy_mlp_hidden_order_when_no_explicit_or_env_order_is_set(monkeypatch) -> None:
    monkeypatch.delenv("THOG2_MLP_CHANNEL_ORDER", raising=False)
    config = SheetGeometryConfig(n_layer=4, n_embd=16, n_head=2, depth_order=3, base_row_order=8)
    assert config.resolved_mlp_channel_order == 32


def test_stage8_scruffy_wrapper_uses_spectral_user_model_name_and_maps_it_to_runner_sheet() -> None:
    text = Path("current_scruffy_train_OWT.sh").read_text(encoding="utf-8")
    assert "-O MODEL_TYPE=${MODEL_TYPE}                    dense | spectral" in text
    assert "MODEL_TYPE=\"spectral\"" in text
    assert "RUNNER_MODEL_TYPE=\"$MODEL_TYPE\"" in text
    assert "[[ \"$MODEL_TYPE\" == spectral ]] && RUNNER_MODEL_TYPE=\"sheet\"" in text
    assert "--model-type \"$RUNNER_MODEL_TYPE\"" in text


def test_stage8_scruffy_wrapper_loops_geometry_presets_nested_inside_depth_orders() -> None:
    text = Path("current_scruffy_train_OWT.sh").read_text(encoding="utf-8")
    assert "PRESET_VALUES=()" in text
    assert "parse_geometry_preset_values \"$GEOMETRY_PRESET\"" in text
    assert "single value, comma list, or quoted space list" in text
    assert "for geometry_preset_value in \"${PRESET_VALUES[@]}\"; do" in text
    assert "for depth_order_value in \"${DEPTH_ORDER_VALUES[@]}\"; do" in text
    assert "run_preset_depth_order \"$geometry_preset_value\" \"$depth_order_value\"" in text
# ^^^ THOG
