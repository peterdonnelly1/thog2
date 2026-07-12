# vvv THOG
from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from run_thog2_owt import run_start_label_from_arguments
from sheet.basis_kernel import DCT_BASIS_VERSION
from sheet.block_trajectory import BlockTrajectory
from sheet.geometry import SheetGeometryConfig
from sheet.run_config import OwtRunConfig
from sheet.semantic_materializer import MLP_CONTRACTION_WEIGHT, MLP_EXPANSION_WEIGHT


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
        "o_mlp_hidden": 32,
        "device": "cpu",
        "dtype": "float32",
    }
    values.update(overrides)
    return OwtRunConfig(**values)


def test_stage8_run_config_records_dct_full_block_identity_in_manifest_and_artifact_name() -> None:
    config = picton_run_config(basis_family="dct", run_start_label="260708-0904")
    identity = config.canonical_dict(world_size=1)["compact_identity"]
    assert identity["geometry_preset"] == "full_block"
    assert identity["attention_geometry"] == "head_aware_block"
    assert identity["mlp_geometry"] == "mlp_block"
    assert identity["basis_family"] == "dct"
    assert identity["basis_version"] == DCT_BASIS_VERSION
    assert config.artifact_name.startswith("260708-0904_" + "NEL" + "SON_DCT_FULL_BLOCK_")
    for fragment in ("P_3", "Q_8", "J_4", "O_3", "X_7", "Y_32"):
        assert fragment in config.artifact_name


def test_stage8_training_config_conversion_preserves_dct_geometry_and_all_semantic_orders() -> None:
    config = picton_run_config(geometry_preset="mlp_block", basis_family="dct")
    training = config.to_training_config(vocab_size=64, world_size=1, out_dir=Path("out"))
    assert training.geometry_preset == "mlp_block"
    assert training.basis_family == "dct"
    assert training.basis_version == DCT_BASIS_VERSION
    assert training.depth_order == 3
    assert training.o_attn_d_model == 8
    assert training.o_attn_qkv_per_channel == 4
    assert training.o_attn_out_per_channel == 3
    assert training.o_mlp_d_model == 7
    assert training.o_mlp_hidden == 32


def test_stage8_semantic_order_defaults_are_recorded_in_manifest_identity_and_training_config() -> None:
    config = OwtRunConfig(model_type="sheet", geometry_preset="full_block", max_iters=20, warmup_iters=1, n_layer=72, n_head=12, n_embd=768)
    manifest = config.canonical_dict(world_size=1)
    identity = manifest["compact_identity"]
    training = config.to_training_config(vocab_size=128, world_size=1, out_dir=Path("out"))
    assert manifest["o_depth"] == 16
    assert manifest["o_attn_d_model"] == 64
    assert manifest["o_attn_qkv_per_channel"] == 6
    assert manifest["o_attn_out_per_channel"] == 6
    assert manifest["o_mlp_d_model"] == 64
    assert manifest["o_mlp_hidden"] == 256
    assert identity["o_mlp_hidden"] == 256
    assert manifest["compact_artifact_fragment"] == "CHEBY_FULL_BLOCK"
    assert "CHEBY_FULL_BLOCK" in config.artifact_name
    assert training.depth_order == 16
    assert training.o_mlp_hidden == 256
    assert training.mlp_channel_order == 256


def test_stage8_semantic_order_overrides_are_recorded_in_manifest_artifact_identity_and_training_config() -> None:
    config = picton_run_config(experiment_prefix="PICTON")
    manifest = config.canonical_dict(world_size=1)
    training = config.to_training_config(vocab_size=128, world_size=1, out_dir=Path("out"))
    assert manifest["o_mlp_hidden"] == 32
    assert manifest["compact_artifact_fragment"] == "CHEBY_FULL_BLOCK"
    for fragment in ("P_3", "Q_8", "J_4", "O_3", "X_7", "Y_32"):
        assert fragment in config.artifact_name
    assert training.o_attn_d_model == 8
    assert training.o_attn_qkv_per_channel == 4
    assert training.o_attn_out_per_channel == 3
    assert training.o_mlp_d_model == 7
    assert training.o_mlp_hidden == 32


def test_stage8_log_timestamp_becomes_compact_run_start_label() -> None:
    arguments = Namespace(run_start_label=None, log_timestamp="20260708_090412")
    assert run_start_label_from_arguments(arguments) == "260708-0904"


def test_stage8_full_block_geometry_uses_independent_mlp_model_and_hidden_orders() -> None:
    config = SheetGeometryConfig(
        n_layer=2,
        n_embd=4,
        n_head=2,
        depth_order=2,
        base_row_order=2,
        mlp_channel_order=6,
        o_mlp_d_model=3,
        o_mlp_hidden=6,
    )
    trajectory = BlockTrajectory(config)
    metadata = {item.name: item for item in trajectory.block_metadata}
    assert metadata[MLP_EXPANSION_WEIGHT].output_order == 6
    assert metadata[MLP_EXPANSION_WEIGHT].input_order == 3
    assert metadata[MLP_CONTRACTION_WEIGHT].output_order == 3
    assert metadata[MLP_CONTRACTION_WEIGHT].input_order == 6


def test_stage8_geometry_retains_environment_fallback_only_when_semantic_hidden_order_is_absent(monkeypatch) -> None:
    monkeypatch.setenv("THOG2_MLP_CHANNEL_ORDER", "6")
    legacy_fallback = SheetGeometryConfig(n_layer=2, n_embd=4, n_head=2, depth_order=2, base_row_order=2)
    explicit_semantic = SheetGeometryConfig(n_layer=2, n_embd=4, n_head=2, depth_order=2, base_row_order=2, o_mlp_hidden=5)
    legacy_trajectory = BlockTrajectory(legacy_fallback)
    explicit_trajectory = BlockTrajectory(explicit_semantic)
    legacy_metadata = {item.name: item for item in legacy_trajectory.block_metadata}
    explicit_metadata = {item.name: item for item in explicit_trajectory.block_metadata}
    assert legacy_metadata[MLP_EXPANSION_WEIGHT].output_order == 6
    assert explicit_metadata[MLP_EXPANSION_WEIGHT].output_order == 5


def test_stage8_geometry_marks_semantic_order_changes_as_thog() -> None:
    text = Path("sheet/geometry.py").read_text(encoding="utf-8")
    assert "# <<< THOG semantic attention model-axis order" in text
    assert "# <<< THOG semantic QKV per-head channel order" in text
    assert "# <<< THOG semantic output per-head channel order" in text
    assert "# <<< THOG semantic MLP model-axis order" in text
    assert "# <<< THOG semantic MLP hidden-axis order" in text
# ^^^ THOG
