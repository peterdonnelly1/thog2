# vvv THOG
from __future__ import annotations

from pathlib import Path

from run_thog2_owt_stage8 import Stage8OwtRunConfig
from sheet.basis_kernel import DCT_BASIS_VERSION


def test_stage8_run_config_records_dct_block_identity_in_manifest_and_artifact_name() -> None:
    config = Stage8OwtRunConfig(model_type="sheet", geometry_preset="block", basis_family="dct", max_iters=20, warmup_iters=1, n_layer=4, n_head=2, n_embd=16, depth_order=3, base_row_order=8)
    identity = config.canonical_dict(world_size=1)["compact_identity"]
    assert identity["geometry_preset"] == "block"
    assert identity["attention_geometry"] == "head_aware_block"
    assert identity["mlp_geometry"] == "mlp_block"
    assert identity["basis_family"] == "dct"
    assert identity["basis_version"] == DCT_BASIS_VERSION
    assert "DCT_BLOCK" in config.artifact_name


def test_stage8_training_config_conversion_preserves_dct_geometry_fields() -> None:
    config = Stage8OwtRunConfig(model_type="sheet", geometry_preset="mlp_block", basis_family="dct", max_iters=20, warmup_iters=1, n_layer=4, n_head=2, n_embd=16, depth_order=3, base_row_order=8)
    training = config.to_training_config(vocab_size=64, world_size=1, out_dir=Path("out"))
    assert training.geometry_preset == "mlp_block"
    assert training.basis_family == "dct"
    assert training.basis_version == DCT_BASIS_VERSION
# ^^^ THOG
