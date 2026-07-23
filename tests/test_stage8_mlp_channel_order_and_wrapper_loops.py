# vvv THOG
from __future__ import annotations

from pathlib import Path

from sheet.block_trajectory import BlockTrajectory
from sheet.geometry import SheetGeometryConfig
from sheet.run_config import OwtRunConfig
from sheet.semantic_materializer import MLP_CONTRACTION_WEIGHT, MLP_EXPANSION_WEIGHT
from sheet.training_model_factory import build_training_model


def picton_run_config(**overrides: object) -> OwtRunConfig:
    values: dict[str, object] = {
        "model_type": "sheet",
        "geometry_preset": "full_block",
        "max_iters": 20,
        "warmup_iters": 1,
        "n_layer": 4,
        "n_head": 2,
        "n_embd": 16,
        "o_depth": 3,
        "o_attn_d_model": 8,
        "o_attn_qkv_per_channel": 4,
        "o_attn_out_per_channel": 3,
        "o_mlp_d_model": 7,
        "o_mlp_hidden": 6,
        "device": "cpu",
        "dtype": "float32",
    }
    values.update(overrides)
    return OwtRunConfig(**values)


def test_stage8_training_factory_passes_semantic_mlp_orders_into_actual_full_block_geometry() -> None:
    config = picton_run_config()
    training = config.to_training_config(vocab_size=64, world_size=1, out_dir=Path("out"))
    model = build_training_model(training)
    assert isinstance(model.trajectory, BlockTrajectory)
    metadata = {item.name: item for item in model.trajectory.block_metadata}
    assert metadata[MLP_EXPANSION_WEIGHT].output_order == 6
    assert metadata[MLP_EXPANSION_WEIGHT].input_order == 7
    assert metadata[MLP_CONTRACTION_WEIGHT].output_order == 7
    assert metadata[MLP_CONTRACTION_WEIGHT].input_order == 6


def test_stage8_mixed_trajectories_treat_nested_depth_fallback_coefficients_as_compact_state() -> None:
    configs = (
        picton_run_config(geometry_preset="mlp_block"),
        picton_run_config(
            geometry_preset=None,
            attention_geometry="head_aware_block",
            mlp_geometry="depth",
        ),
    )
    for config in configs:
        training = config.to_training_config(vocab_size=64, world_size=1, out_dir=Path("out"))
        model = build_training_model(training)
        assert model.compact_state_violations() == ()


def test_stage8_sheet_geometry_keeps_legacy_mlp_hidden_order_when_no_semantic_or_environment_order_is_set(monkeypatch) -> None:
    monkeypatch.delenv("THOG2_MLP_CHANNEL_ORDER", raising=False)
    config = SheetGeometryConfig(n_layer=4, n_embd=16, n_head=2, depth_order=3, base_row_order=8)
    assert config.resolved_o_mlp_hidden == 32


def test_stage8_training_wrappers_use_final_preset_as_single_architecture_selector() -> None:
    for wrapper in ("current_scruffy_train_OWT.sh", "current_dreedle_train_OWT.sh"):
        text = Path(wrapper).read_text(encoding="utf-8")
        assert "-p PRESET=${GEOMETRY_PRESET}" in text
        assert "dense | legacy_sheet_col | depth | jpeg_like_v1 | head_aware_block | mlp_block | full_block" in text
        assert "geometry_preset=\"curve\"" not in text.lower()
        assert "-p curve" not in text.lower()
        assert "Deprecated compatibility:" not in text
        assert "MODEL_TYPE_ARGUMENT" not in text
        assert "run_model_type=\"dense\"" in text
        assert "run_model_type=\"sheet\"" in text


def test_stage8_training_wrappers_share_the_six_order_letters() -> None:
    required_lines = (
        "-P O_DEPTH=${O_DEPTH}",
        "-Q O_ATTN_D_MODEL=${O_ATTN_D_MODEL}",
        "-J O_ATTN_QKV_PER_CHANNEL=${O_ATTN_QKV_PER_CHANNEL}",
        "-O O_ATTN_OUT_PER_CHANNEL=${O_ATTN_OUT_PER_CHANNEL}",
        "-X O_MLP_D_MODEL=${O_MLP_D_MODEL}",
        "-Y O_MLP_HIDDEN=${O_MLP_HIDDEN}",
    )
    for wrapper in ("current_scruffy_train_OWT.sh", "current_dreedle_train_OWT.sh"):
        text = Path(wrapper).read_text(encoding="utf-8")
        for required_line in required_lines:
            assert required_line in text


def test_stage8_training_wrappers_loop_dense_once_and_compact_presets_across_depth_orders() -> None:
    for wrapper in ("current_scruffy_train_OWT.sh", "current_dreedle_train_OWT.sh"):
        text = Path(wrapper).read_text(encoding="utf-8")
        assert "PRESET_VALUES=()" in text
        assert "O_DEPTH_VALUES=()" in text
        assert "parse_geometry_preset_values \"$GEOMETRY_PRESET\"" in text
        assert "parse_o_depth_values \"$O_DEPTH\"" in text
        assert "single value, comma list, or quoted space list" in text
        assert "case \"$value\" in" in text
        assert "dense) PRESET_VALUES+=(\"$value\"); HAS_DENSE_PRESET=true" in text
        assert "depth) PRESET_VALUES+=(\"$value\"); HAS_COMPACT_PRESET=true" in text
        assert "legacy_sheet_col|head_aware_block|mlp_block|full_block) PRESET_VALUES+=(\"$value\"); HAS_COMPACT_PRESET=true; HAS_NON_DEPTH_COMPACT_PRESET=true" in text
        assert "--depth-compress-layer-norm-and-bias" in text
        assert "if [[ \"$geometry_preset_value\" == dense ]]; then" in text
        # assert "run_preset_o_depth \"$geometry_preset_value\" \"${O_DEPTH_VALUES[0]}\"" in text
        assert "run_grid_point \"$geometry_preset_value\" \"${O_DEPTH_VALUES[0]}\" \"$batch_size_value\" \"$learning_rate_code\" \"${BASIS_FAMILY_VALUES[0]}\" \"${BASIS_TAG_VALUES[0]}\"" in text
        assert "for o_depth_value in \"${O_DEPTH_VALUES[@]}\"; do" in text
        assert "for batch_size_value in \"${BATCH_SIZE_VALUES[@]}\"; do" in text
        assert "for learning_rate_code in \"${LEARNING_RATE_CODE_VALUES[@]}\"; do" in text
# ^^^ THOG
