# vvv THOG
from __future__ import annotations

from pathlib import Path

from run_thog2_owt_stage8 import Stage8OwtRunConfig
from sheet.basis_kernel import DCT_BASIS_VERSION
from sheet.block_trajectory import BlockTrajectory
from sheet.geometry import SheetGeometryConfig
from sheet.semantic_materializer import MLP_CONTRACTION_WEIGHT, MLP_EXPANSION_WEIGHT


def test_stage8_run_config_records_dct_block_identity_in_manifest_and_artifact_name() -> None:
    config = Stage8OwtRunConfig(model_type="sheet", geometry_preset="block", basis_family="dct", max_iters=20, warmup_iters=1, n_layer=4, n_head=2, n_embd=16, depth_order=3, base_row_order=8, mlp_channel_order=32)
    identity = config.canonical_dict(world_size=1)["compact_identity"]
    assert identity["geometry_preset"] == "block"
    assert identity["attention_geometry"] == "head_aware_block"
    assert identity["mlp_geometry"] == "mlp_block"
    assert identity["basis_family"] == "dct"
    assert identity["basis_version"] == DCT_BASIS_VERSION
    assert "DCT_BLOCK" in config.artifact_name


def test_stage8_training_config_conversion_preserves_dct_geometry_fields() -> None:
    config = Stage8OwtRunConfig(model_type="sheet", geometry_preset="mlp_block", basis_family="dct", max_iters=20, warmup_iters=1, n_layer=4, n_head=2, n_embd=16, depth_order=3, base_row_order=8, mlp_channel_order=32)
    training = config.to_training_config(vocab_size=64, world_size=1, out_dir=Path("out"))
    assert training.geometry_preset == "mlp_block"
    assert training.basis_family == "dct"
    assert training.basis_version == DCT_BASIS_VERSION


def test_stage8_mlp_channel_order_defaults_to_256_and_is_recorded_everywhere(monkeypatch) -> None:
    monkeypatch.delenv("THOG2_MLP_CHANNEL_ORDER", raising=False)
    config = Stage8OwtRunConfig(model_type="sheet", geometry_preset="block", max_iters=20, warmup_iters=1, n_layer=8, n_head=2, n_embd=128, depth_order=4, base_row_order=8)
    manifest = config.canonical_dict(world_size=1)
    training = config.to_training_config(vocab_size=128, world_size=1, out_dir=Path("out"))
    assert manifest["mlp_channel_order"] == 256
    assert "R_256" in manifest["compact_artifact_fragment"]
    assert "R_256" in config.artifact_name
    assert training.mlp_channel_order == 256


def test_stage8_mlp_channel_order_override_is_recorded_in_manifest_artifact_and_training_config() -> None:
    config = Stage8OwtRunConfig(model_type="sheet", geometry_preset="block", max_iters=20, warmup_iters=1, n_layer=8, n_head=2, n_embd=64, depth_order=4, base_row_order=8, mlp_channel_order=32)
    manifest = config.canonical_dict(world_size=1)
    training = config.to_training_config(vocab_size=128, world_size=1, out_dir=Path("out"))
    assert manifest["mlp_channel_order"] == 32
    assert "R_32" in manifest["compact_artifact_fragment"]
    assert "R_32" in config.artifact_name
    assert training.mlp_channel_order == 32


def test_stage8_block_geometry_uses_separate_mlp_channel_order_for_hidden_axis(monkeypatch) -> None:
    monkeypatch.delenv("THOG2_MLP_CHANNEL_ORDER", raising=False)
    config = SheetGeometryConfig(n_layer=2, n_embd=4, n_head=2, depth_order=2, base_row_order=2, mlp_channel_order=6)
    trajectory = BlockTrajectory(config)
    metadata = {item.name: item for item in trajectory.block_metadata}
    assert metadata[MLP_EXPANSION_WEIGHT].output_order == 6
    assert metadata[MLP_EXPANSION_WEIGHT].input_order == 2
    assert metadata[MLP_CONTRACTION_WEIGHT].output_order == 2
    assert metadata[MLP_CONTRACTION_WEIGHT].input_order == 6


def test_stage8_geometry_uses_environment_default_for_mlp_channel_order(monkeypatch) -> None:
    monkeypatch.setenv("THOG2_MLP_CHANNEL_ORDER", "6")
    config = SheetGeometryConfig(n_layer=2, n_embd=4, n_head=2, depth_order=2, base_row_order=2)
    trajectory = BlockTrajectory(config)
    metadata = {item.name: item for item in trajectory.block_metadata}
    assert metadata[MLP_EXPANSION_WEIGHT].output_order == 6
    assert metadata[MLP_CONTRACTION_WEIGHT].input_order == 6


def test_stage8_geometry_marks_mlp_channel_order_changes_as_thog() -> None:
    text = Path("sheet/geometry.py").read_text(encoding="utf-8")
    assert "# <<< THOG default separate MLP hidden-axis basis order" in text
    assert "# <<< THOG host wrapper override for MLP hidden-axis basis order" in text
    assert "# <<< THOG independent order for MLP hidden-axis bases" in text
# ^^^ THOG
