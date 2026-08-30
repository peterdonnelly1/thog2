from __future__ import annotations

import copy
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch

import sheet.dense_snapshot as dense_snapshot
from run_thog2_owt_core import (
    build_parser,
    config_from_arguments,
    geometry_plan_from_arguments,
    validate_dense_snapshot_cli,
)
from sheet.basis import BASIS_VERSION
from sheet.compact_identity import BASIS_FAMILY_CHEBYSHEV, GEOMETRY_PRESET_DEPTH
from sheet.dense_snapshot import (
    DENSE_SNAPSHOT_ROLE_B,
    DENSE_SNAPSHOT_ROLE_C,
    dense_snapshot_filename,
    initialise_model_from_dense_snapshot,
    load_dense_initialisation_snapshot,
    save_dense_initialisation_snapshot,
    validate_physical_compatibility,
)
from sheet.training_config import TrainingConfig
from sheet.training_model_factory import build_training_model
from sheet.trainer import SharedTrainer


def _dense_config(**changes) -> TrainingConfig:
    values = {
        "model_type": "dense",
        "block_size": 8,
        "vocab_size": 16,
        "n_layer": 3,
        "n_head": 1,
        "n_embd": 4,
        "depth_order": 1,
        "base_row_order": 1,
        "batch_size": 1,
        "max_updates": 2,
        "decay_updates": 2,
        "eval_batches": 1,
        "device": "cpu",
        "dtype": "float32",
    }
    values.update(changes)
    return TrainingConfig(**values)


def _compact_config(snapshot_path: Path, order: int, **changes) -> TrainingConfig:
    values = {
        "model_type": "thog2_sheet",
        "block_size": 8,
        "vocab_size": 16,
        "n_layer": 3,
        "n_head": 1,
        "n_embd": 4,
        "depth_order": order,
        "base_row_order": 1,
        "o_attn_d_model": 1,
        "o_attn_qkv_per_channel": 1,
        "o_attn_out_per_channel": 1,
        "o_mlp_d_model": 1,
        "o_mlp_hidden": 1,
        "batch_size": 1,
        "max_updates": 2,
        "decay_updates": 2,
        "eval_batches": 1,
        "device": "cpu",
        "dtype": "float32",
        "geometry_preset": GEOMETRY_PRESET_DEPTH,
        "basis_family": BASIS_FAMILY_CHEBYSHEV,
        "basis_version": BASIS_VERSION,
        "initialise_from_dense_snapshot": str(snapshot_path),
    }
    values.update(changes)
    return TrainingConfig(**values)


def _create_snapshot(tmp_path: Path):
    config = _dense_config()
    model = build_training_model(config)
    path, payload = save_dense_initialisation_snapshot(
        model,
        config,
        root=tmp_path,
        created_at=datetime(2026, 8, 29, 15, 30, 12),
        host_name="test-host",
    )
    return config, model, path, payload


def _state(model):
    return {name: value.detach().clone() for name, value in model.named_parameters()}


def _assert_state_equal(model, expected) -> None:
    actual = dict(model.named_parameters())
    assert set(actual) == set(expected)
    for name in expected:
        assert torch.equal(actual[name].detach(), expected[name]), name


def test_filename_uses_minute_resolution_and_physical_identity() -> None:
    config = _dense_config()
    filename = dense_snapshot_filename(
        config,
        "abcdef0123456789",
        created_at=datetime(2026, 8, 29, 15, 30, 59),
        host_name="scruffy.example",
    )
    assert filename == (
        "260829-1530_scruffy-example_L3_H1_D4_C8_Tf32_VS16_BI1_"
        "MM4_TE1_IF1_CHabcdef01.dense_snapshot.pt"
    )
    assert not re.search(r"260829-1530\d", filename)


def test_payload_contains_unique_model_parameters_aliases_and_no_seed(tmp_path: Path) -> None:
    _, _, _, payload = _create_snapshot(tmp_path)
    assert set(payload) == {
        "snapshot_schema_version",
        "tensor_manifest_version",
        "compatibility_payload",
        "compatibility_hash",
        "tensor_manifest",
        "tensors",
        "tensor_payload_hash",
        "paired_rng_boundary",
        "paired_rng_boundary_identifier",
    }
    manifest = {row["canonical_name"]: row for row in payload["tensor_manifest"]}
    tied = [row for row in manifest.values() if "lm_head.weight" in row["aliases"]]
    assert len(tied) == 1
    assert tied[0]["aliases"] == ["lm_head.weight", "transformer.wte.weight"]
    assert "model_seed" not in repr(payload)


def test_snapshot_save_does_not_advance_any_rng(tmp_path: Path) -> None:
    config = _dense_config()
    model = build_training_model(config)
    torch.manual_seed(31)
    np.random.seed(32)
    random.seed(33)
    torch_state = torch.get_rng_state().clone()
    numpy_state = copy.deepcopy(np.random.get_state())
    python_state = random.getstate()
    save_dense_initialisation_snapshot(model, config, root=tmp_path, host_name="rng")
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]
    assert random.getstate() == python_state


def test_consumers_restore_the_saved_paired_rng_boundary(tmp_path: Path) -> None:
    torch.manual_seed(41)
    np.random.seed(42)
    random.seed(43)
    _, _, path, payload = _create_snapshot(tmp_path)
    expected_torch = torch.rand(3)
    expected_numpy = np.random.rand(3)
    expected_python = [random.random() for _ in range(3)]
    torch.manual_seed(101)
    np.random.seed(102)
    random.seed(103)
    config = _dense_config(
        initialise_from_dense_snapshot=str(path),
        dense_snapshot_chebyshev_order=2,
        dense_snapshot_chebyshev_version=BASIS_VERSION,
    )
    model = build_training_model(config)
    metadata = initialise_model_from_dense_snapshot(model, config, path)
    assert metadata["paired_rng_boundary_identifier"] == payload["paired_rng_boundary_identifier"]
    assert torch.equal(torch.rand(3), expected_torch)
    assert np.array_equal(np.random.rand(3), expected_numpy)
    assert [random.random() for _ in range(3)] == expected_python


def test_atomic_writer_refuses_overwrite_and_preserves_original(tmp_path: Path) -> None:
    config, model, path, payload = _create_snapshot(tmp_path)
    with pytest.raises(FileExistsError, match="refuses to overwrite"):
        save_dense_initialisation_snapshot(
            model,
            config,
            root=tmp_path,
            created_at=datetime(2026, 8, 29, 15, 30, 59),
            host_name="test-host",
        )
    assert load_dense_initialisation_snapshot(path)["tensor_payload_hash"] == payload["tensor_payload_hash"]
    assert not list(path.parent.glob("*.tmp-*"))


def test_integrity_corruption_is_rejected(tmp_path: Path) -> None:
    _, _, _, payload = _create_snapshot(tmp_path)
    damaged = copy.deepcopy(payload)
    first = next(iter(damaged["tensors"]))
    damaged["tensors"][first].view(-1)[0] += 1
    path = tmp_path / "damaged.pt"
    torch.save(damaged, path)
    with pytest.raises(ValueError, match="tensor payload hash mismatch"):
        load_dense_initialisation_snapshot(path)


def test_physical_mismatch_fails_without_target_mutation(tmp_path: Path) -> None:
    _, _, path, _ = _create_snapshot(tmp_path)
    config = TrainingConfig(
        **{
            **_dense_config().persistent_dict(),
            "n_embd": 6,
            "n_head": 1,
            "initialise_from_dense_snapshot": str(path),
            "dense_snapshot_chebyshev_order": 2,
            "dense_snapshot_chebyshev_version": BASIS_VERSION,
        }
    )
    model = build_training_model(config)
    before = _state(model)
    with pytest.raises(ValueError, match="class=physical_source_structure"):
        initialise_model_from_dense_snapshot(model, config, path)
    _assert_state_equal(model, before)


def test_supported_dtype_conversion_does_not_make_source_incompatible(tmp_path: Path) -> None:
    _, _, _, payload = _create_snapshot(tmp_path)
    validate_physical_compatibility(payload, _dense_config(dtype="bfloat16"))


def test_p_equals_l_reconstructs_the_saved_dense_model(tmp_path: Path) -> None:
    _, source, path, _ = _create_snapshot(tmp_path)
    config = _dense_config(
        initialise_from_dense_snapshot=str(path),
        dense_snapshot_chebyshev_order=3,
        dense_snapshot_chebyshev_version=BASIS_VERSION,
    )
    target = build_training_model(config)
    metadata = initialise_model_from_dense_snapshot(target, config, path)
    source_parameters = dict(source.named_parameters())
    for name, parameter in target.named_parameters():
        assert torch.allclose(parameter, source_parameters[name], atol=2.0e-5, rtol=0.0), name
    assert max(row["absolute_reconstruction_error"] for row in metadata["numerical_diagnostics"]) < 2.0e-5


@pytest.mark.parametrize("order", [1, 2])
def test_b_and_c_share_mapping_and_step_zero_physical_state(tmp_path: Path, order: int) -> None:
    _, _, path, _ = _create_snapshot(tmp_path)
    dense_config = _dense_config(
        initialise_from_dense_snapshot=str(path),
        dense_snapshot_chebyshev_order=order,
        dense_snapshot_chebyshev_version=BASIS_VERSION,
    )
    compact_config = _compact_config(path, order)
    dense_model = build_training_model(dense_config)
    compact_model = build_training_model(compact_config)
    dense_metadata = initialise_model_from_dense_snapshot(dense_model, dense_config, path)
    compact_metadata = initialise_model_from_dense_snapshot(compact_model, compact_config, path)
    assert dense_metadata["lifecycle_role"] == DENSE_SNAPSHOT_ROLE_B
    assert compact_metadata["lifecycle_role"] == DENSE_SNAPSHOT_ROLE_C
    assert dense_metadata["mapping_fingerprint"] == compact_metadata["mapping_fingerprint"]
    assert dense_metadata["step_zero_manifest_identifier"] == compact_metadata["step_zero_manifest_identifier"]


def test_direct_copy_families_and_tied_aliases_are_exact(tmp_path: Path) -> None:
    _, source, path, _ = _create_snapshot(tmp_path)
    config = _dense_config(
        initialise_from_dense_snapshot=str(path),
        dense_snapshot_chebyshev_order=1,
        dense_snapshot_chebyshev_version=BASIS_VERSION,
    )
    target = build_training_model(config)
    initialise_model_from_dense_snapshot(target, config, path)
    source_parameters = dict(source.named_parameters())
    target_parameters = dict(target.named_parameters())
    direct_names = [
        name
        for name in source_parameters
        if not dense_snapshot._is_mapped_dense_parameter(name)
    ]
    for name in direct_names:
        assert torch.equal(target_parameters[name], source_parameters[name]), name
    assert target.transformer.wte.weight is target.lm_head.weight


def test_rank_deficiency_fails_without_target_mutation(tmp_path: Path, monkeypatch) -> None:
    _, _, path, _ = _create_snapshot(tmp_path)
    config = _dense_config(
        initialise_from_dense_snapshot=str(path),
        dense_snapshot_chebyshev_order=2,
        dense_snapshot_chebyshev_version=BASIS_VERSION,
    )
    model = build_training_model(config)
    before = _state(model)
    monkeypatch.setattr(dense_snapshot, "_mapping_basis", lambda model, config, order: torch.zeros(3, 2))
    with pytest.raises(ValueError, match="rank-deficient"):
        initialise_model_from_dense_snapshot(model, config, path)
    _assert_state_equal(model, before)


def test_nonfinite_mapping_fails_without_target_mutation(tmp_path: Path) -> None:
    _, _, _, payload = _create_snapshot(tmp_path)
    damaged = copy.deepcopy(payload)
    alias_lookup = dense_snapshot._manifest_alias_lookup(damaged)
    name = alias_lookup["transformer.h.0.attn.c_proj.weight"]
    damaged["tensors"][name].view(-1)[0] = float("nan")
    damaged["tensor_payload_hash"] = dense_snapshot._tensor_payload_hash(
        damaged["tensors"], damaged["tensor_manifest"]
    )
    path = tmp_path / "valid-but-nonfinite.pt"
    torch.save(damaged, path)
    config = _dense_config(
        initialise_from_dense_snapshot=str(path),
        dense_snapshot_chebyshev_order=2,
        dense_snapshot_chebyshev_version=BASIS_VERSION,
    )
    model = build_training_model(config)
    before = _state(model)
    with pytest.raises(FloatingPointError, match="non-finite"):
        initialise_model_from_dense_snapshot(model, config, path)
    _assert_state_equal(model, before)


def test_one_shot_actions_are_not_persisted() -> None:
    config = _dense_config(
        initialise_from_dense_snapshot="snapshot.pt",
        dense_snapshot_chebyshev_order=2,
        dense_snapshot_chebyshev_version=BASIS_VERSION,
        dense_snapshot_host_label="host",
    )
    persisted = config.persistent_dict()
    for name in (
        "save_dense_initialisation_snapshot",
        "initialise_from_dense_snapshot",
        "dense_snapshot_chebyshev_order",
        "dense_snapshot_chebyshev_version",
        "dense_snapshot_host_label",
    ):
        assert name not in persisted
    resumed = TrainingConfig(**persisted)
    assert resumed.initialise_from_dense_snapshot is None


def test_cli_resolves_b_dense_as_pure_chebyshev_without_compact_identity() -> None:
    parser = build_parser()
    arguments = parser.parse_args(
        [
            "--model-type",
            "dense",
            "--initialise-from-dense-snapshot",
            "snapshot.pt",
            "--select-depth",
            "--option",
            "DEPTH.compressor=chebyshev",
            "--option",
            "DEPTH.order=2",
        ]
    )
    plan = geometry_plan_from_arguments(arguments)
    config = config_from_arguments(arguments, geometry_plan=plan)
    assert config.model_type == "dense"
    assert config.dense_snapshot_chebyshev_order == 2
    assert config.dense_snapshot_chebyshev_version == BASIS_VERSION
    assert config.resolved_geometry_plan is None


@pytest.mark.parametrize("mode", ["resume", "fork"])
def test_cli_rejects_snapshot_actions_with_resume_or_fork(mode: str) -> None:
    parser = build_parser()
    arguments = parser.parse_args(
        ["--model-type", "dense", "--initialise-from-dense-snapshot", "snapshot.pt"]
    )
    with pytest.raises(ValueError, match="may not be combined with resume or fork"):
        validate_dense_snapshot_cli(arguments, resolved_mode=mode)


def test_cli_rejects_explicit_residual_initialisation() -> None:
    parser = build_parser()
    arguments = parser.parse_args(
        ["--model-type", "dense", "--initialise-from-dense-snapshot", "snapshot.pt"]
    )
    with pytest.raises(ValueError, match="residual-initialisation"):
        validate_dense_snapshot_cli(arguments, explicit={"residual_init_policy"})


def test_cli_rejects_non_chebyshev_and_non_depth_v1_geometry() -> None:
    parser = build_parser()
    dct_arguments = parser.parse_args(
        [
            "--model-type",
            "dense",
            "--initialise-from-dense-snapshot",
            "snapshot.pt",
            "--select-depth",
            "--option",
            "DEPTH.compressor=dct",
        ]
    )
    with pytest.raises(ValueError, match="only the Chebyshev"):
        config_from_arguments(dct_arguments)
    block_arguments = parser.parse_args(
        [
            "--model-type",
            "dense",
            "--initialise-from-dense-snapshot",
            "snapshot.pt",
            "--select-depth",
            "--select-element",
            "MLP_UP.MLP_HIDDEN",
        ]
    )
    with pytest.raises(ValueError, match="only pure DEPTH"):
        config_from_arguments(block_arguments)


def test_cli_rejects_save_and_initialise_together() -> None:
    parser = build_parser()
    arguments = parser.parse_args(
        [
            "--model-type",
            "dense",
            "--save-dense-initialisation-snapshot",
            "--initialise-from-dense-snapshot",
            "snapshot.pt",
        ]
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_dense_snapshot_cli(arguments)


def test_shared_trainer_runs_a_b_c_before_optimizer_and_persists_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(dense_snapshot, "repository_root", lambda: tmp_path)
    tokens = torch.arange(64, dtype=torch.long) % 16
    a_config = _dense_config(
        save_dense_initialisation_snapshot=True,
        dense_snapshot_host_label="integration",
    )
    a = SharedTrainer(a_config, tokens, tokens)
    try:
        a_metadata = a.dense_snapshot_metadata
        assert a_metadata["lifecycle_role"] == "A Normal DENSE"
        path = Path(a_metadata["snapshot_path"])
        assert path.is_file()
        checkpoint = a.checkpoint_payload()
        assert checkpoint["dense_snapshot_baselining"] == a_metadata
        assert "save_dense_initialisation_snapshot" not in checkpoint["trainer_config"]

        b_config = _dense_config(
            initialise_from_dense_snapshot=str(path),
            dense_snapshot_chebyshev_order=2,
            dense_snapshot_chebyshev_version=BASIS_VERSION,
        )
        b = SharedTrainer(b_config, tokens, tokens)
        try:
            c_config = _compact_config(path, 2)
            c = SharedTrainer(c_config, tokens, tokens)
            try:
                assert b.dense_snapshot_metadata["lifecycle_role"] == DENSE_SNAPSHOT_ROLE_B
                assert c.dense_snapshot_metadata["lifecycle_role"] == DENSE_SNAPSHOT_ROLE_C
                assert (
                    b.dense_snapshot_metadata["step_zero_manifest_identifier"]
                    == c.dense_snapshot_metadata["step_zero_manifest_identifier"]
                )
                assert "dense_snapshot_baselining" in b.parameter_report
                assert "dense_snapshot_baselining" in c.parameter_report
            finally:
                c.close()
            checkpoint_path = b.save_checkpoint(tmp_path / "b-checkpoint.pt")
            path.unlink()
            resumed = SharedTrainer.from_checkpoint(
                checkpoint_path,
                tokens,
                tokens,
                expected_config=b_config,
            )
            try:
                assert resumed.config.initialise_from_dense_snapshot is None
                assert resumed.dense_snapshot_metadata == b.dense_snapshot_metadata
                assert "dense_snapshot_baselining" in resumed.parameter_report
            finally:
                resumed.close()
        finally:
            b.close()
    finally:
        a.close()
