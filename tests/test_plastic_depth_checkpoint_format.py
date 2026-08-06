# vvv THOG
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from sheet.checkpoints import (
    PLASTIC_DEPTH_CHECKPOINT_FORMAT_VERSION,
    PLASTIC_DEPTH_LEGACY_PHANTOM_VERSION,
    load_payload,
    save_payload,
    validate_compatibility,
    validate_plastic_depth_checkpoint_format,
)
from sheet.compact_state import model_from_compact_state
from sheet.plastic_depth import PLASTIC_DEPTH_VERSION
from sheet.trainer import SharedTrainer
from run_thog2_owt_core import validate_resume_controls
from tests.stage3_test_support import stage3_config, token_splits
from tests.test_plastic_depth import plastic_training_config


def _learned_plastic_config():
    return plastic_training_config(
        plastic__layers_to_sample=None,
        plastic__do_learn_layer_count=True,
        plastic__initial_layer_count=2,
        plastic__max_permitted_layers=4,
        plastic__layer_count_update_brake=11,
        plastic__layer_count_probe_noise_window=19,
        plastic__layer_count_min_probes=3,
        plastic__layer_count_probe_noise_lambda=1.5,
        plastic__layer_count_cost_weight=0.2,
        plastic__cuda_allocator_reserve_gib=0.75,
        plastic__geometry_learning_rate_multiplier=0.25,
        plastic__freeze_geometry_during_warmup=False,
    )


def _retired_short_identity(canonical_identity: Dict[str, Any]) -> Dict[str, Any]:
    plastic = canonical_identity["plastic_depth"]
    return {
        "version": PLASTIC_DEPTH_VERSION,
        "maximum_layers": canonical_identity["n_layer"],
        "initial_active_layers": plastic["plastic__initial_active_layers"],
        "learn_layer_count": plastic["plastic__do_learn_layer_count"],
        "sampling_initialisation": plastic["plastic__layer_sampling_initialisation"],
        "count_objective": plastic["plastic__layer_count_objective"],
        "count_update_brake": plastic["plastic__layer_count_update_brake"],
        "probe_noise_window": plastic["plastic__layer_count_probe_noise_window"],
        "probe_noise_min_observations": plastic["plastic__layer_count_min_probes"],
        "probe_noise_lambda": plastic["plastic__layer_count_probe_noise_lambda"],
        "count_cost_weight": plastic["plastic__layer_count_cost_weight"],
        "memory_budget_gib": plastic["plastic__layer_memory_budget_gib"],
        "cuda_allocator_reserve_gib": plastic["plastic__cuda_allocator_reserve_gib"],
        "geometry_lr_multiplier": plastic["plastic__geometry_learning_rate_multiplier"],
        "freeze_geometry_during_warmup": plastic["plastic__freeze_geometry_during_warmup"],
    }


def _checkpoint_payload(config) -> Dict[str, Any]:
    train_tokens, validation_tokens = token_splits()
    trainer = SharedTrainer(config, train_tokens, validation_tokens)
    try:
        return trainer.checkpoint_payload()
    finally:
        trainer.close()


def test_new_enabled_checkpoint_has_explicit_active_prefix_format() -> None:
    payload = _checkpoint_payload(_learned_plastic_config())
    assert payload["plastic_depth_checkpoint_format_version"] == PLASTIC_DEPTH_CHECKPOINT_FORMAT_VERSION
    assert payload["compact_identity"]["plastic_depth"]["version"] == PLASTIC_DEPTH_VERSION


def test_disabled_checkpoint_surface_is_unchanged() -> None:
    payload = _checkpoint_payload(stage3_config("thog2_sheet"))
    assert "plastic_depth_checkpoint_format_version" not in payload
    assert "plastic_depth" not in payload["compact_identity"]
    validate_plastic_depth_checkpoint_format(payload)


def test_existing_canonical_and_retired_short_v03_checkpoints_are_accepted() -> None:
    config = _learned_plastic_config()
    payload = _checkpoint_payload(config)
    payload.pop("plastic_depth_checkpoint_format_version")
    validate_plastic_depth_checkpoint_format(payload)
    validate_compatibility(payload, config)

    payload["compact_identity"]["plastic_depth"] = _retired_short_identity(
        payload["compact_identity"]
    )
    validate_plastic_depth_checkpoint_format(payload)
    validate_compatibility(payload, config)


def test_explicit_v01_checkpoint_is_rejected_with_precise_reason() -> None:
    config = _learned_plastic_config()
    payload = _checkpoint_payload(config)
    payload.pop("plastic_depth_checkpoint_format_version")
    payload["compact_identity"]["plastic_depth"]["version"] = PLASTIC_DEPTH_LEGACY_PHANTOM_VERSION
    with pytest.raises(ValueError, match="phantom-lattice chart cannot be converted safely"):
        validate_plastic_depth_checkpoint_format(payload)
    with pytest.raises(ValueError, match="explicit legacy version"):
        validate_compatibility(payload, config)


def test_versionless_enabled_plastic_checkpoint_is_rejected() -> None:
    payload = _checkpoint_payload(_learned_plastic_config())
    payload.pop("plastic_depth_checkpoint_format_version")
    payload["compact_identity"]["plastic_depth"].pop("version")
    with pytest.raises(ValueError, match="no trustworthy version discriminator"):
        validate_plastic_depth_checkpoint_format(payload)


def test_unknown_or_inconsistent_format_discriminator_is_rejected() -> None:
    payload = _checkpoint_payload(_learned_plastic_config())
    payload["plastic_depth_checkpoint_format_version"] = "plastic_depth_unknown_format"
    with pytest.raises(ValueError, match="unsupported format discriminator"):
        validate_plastic_depth_checkpoint_format(payload)

    payload["plastic_depth_checkpoint_format_version"] = PLASTIC_DEPTH_CHECKPOINT_FORMAT_VERSION
    payload["compact_identity"]["plastic_depth"]["version"] = PLASTIC_DEPTH_LEGACY_PHANTOM_VERSION
    with pytest.raises(ValueError, match="format discriminator and compact identity version disagree"):
        validate_plastic_depth_checkpoint_format(payload)


def test_enabled_checkpoint_rejects_explicitly_disabled_canonical_identity() -> None:
    config = _learned_plastic_config()
    payload = _checkpoint_payload(config)
    payload["compact_identity"]["plastic_depth"]["plastic__enabled"] = False
    with pytest.raises(ValueError, match="enabled trainer state disagrees with compact identity"):
        validate_plastic_depth_checkpoint_format(payload)
    with pytest.raises(ValueError, match="enabled trainer state disagrees with compact identity"):
        validate_compatibility(payload, config)


def test_resume_control_preflight_rejects_before_training_config() -> None:
    config = _learned_plastic_config()
    payload = _checkpoint_payload(config)
    payload.pop("plastic_depth_checkpoint_format_version")
    payload["compact_identity"]["plastic_depth"]["version"] = PLASTIC_DEPTH_LEGACY_PHANTOM_VERSION
    payload["trainer_config"]["obsolete_v01_only_field"] = "must never reach TrainingConfig"

    with tempfile.TemporaryDirectory() as directory:
        path = save_payload(payload, Path(directory) / "unsafe-v01-controls.pt")
        with pytest.raises(ValueError, match="phantom-lattice chart cannot be converted safely"):
            validate_resume_controls(path, config)



def test_disabled_trainer_with_plastic_metadata_is_rejected_before_model_construction() -> None:
    payload = _checkpoint_payload(_learned_plastic_config())
    payload["trainer_config"]["plastic__enabled"] = False
    payload["model"] = {"invalid": "must never reach load_state_dict"}
    with pytest.raises(ValueError, match="disabled trainer state contains PLASTIC checkpoint metadata"):
        validate_plastic_depth_checkpoint_format(payload)
    with pytest.raises(ValueError, match="disabled trainer state contains PLASTIC checkpoint metadata"):
        model_from_compact_state(payload)


def test_disabled_trainer_with_format_only_is_rejected() -> None:
    payload = _checkpoint_payload(_learned_plastic_config())
    payload["trainer_config"]["plastic__enabled"] = False
    payload["compact_identity"].pop("plastic_depth")
    with pytest.raises(ValueError, match="disabled trainer state contains PLASTIC checkpoint metadata"):
        validate_plastic_depth_checkpoint_format(payload)

def test_retired_short_v03_checkpoint_resumes_end_to_end() -> None:
    train_tokens, validation_tokens = token_splits()
    config = _learned_plastic_config()
    with tempfile.TemporaryDirectory() as directory:
        source = SharedTrainer(config, train_tokens, validation_tokens)
        try:
            source.run(target_updates=1)
            source_path = source.save_checkpoint(Path(directory) / "source.pt")
        finally:
            source.close()

        payload = load_payload(source_path)
        payload.pop("plastic_depth_checkpoint_format_version")
        payload["compact_identity"]["plastic_depth"] = _retired_short_identity(
            payload["compact_identity"]
        )
        retired_path = save_payload(payload, Path(directory) / "retired-short-v03.pt")

        resumed = SharedTrainer.from_checkpoint(
            retired_path,
            train_tokens,
            validation_tokens,
        )
        try:
            assert resumed.state.completed_updates == 1
            resumed.train_one_update()
            assert resumed.state.completed_updates == 2
        finally:
            resumed.close()


def test_unsafe_plastic_inference_rejection_precedes_model_construction() -> None:
    payload = _checkpoint_payload(_learned_plastic_config())
    payload.pop("plastic_depth_checkpoint_format_version")
    payload["compact_identity"]["plastic_depth"].pop("version")
    payload["trainer_config"]["obsolete_v01_only_field"] = "must never reach TrainingConfig"
    payload["model"] = {"invalid": "must never reach load_state_dict"}
    with pytest.raises(ValueError, match="no trustworthy version discriminator"):
        model_from_compact_state(payload)


def test_unsafe_plastic_rejection_precedes_config_and_state_application() -> None:
    train_tokens, validation_tokens = token_splits()
    payload = _checkpoint_payload(_learned_plastic_config())
    payload.pop("plastic_depth_checkpoint_format_version")
    payload["compact_identity"]["plastic_depth"]["version"] = PLASTIC_DEPTH_LEGACY_PHANTOM_VERSION
    payload["trainer_config"]["obsolete_v01_only_field"] = "must never reach TrainingConfig"
    payload["model"] = {"invalid": "must never reach load_state_dict"}
    payload["optimizer"] = {"invalid": "must never reach optimizer.load_state_dict"}

    with tempfile.TemporaryDirectory() as directory:
        path = save_payload(payload, Path(directory) / "unsafe-v01.pt")
        with pytest.raises(ValueError, match="phantom-lattice chart cannot be converted safely"):
            SharedTrainer.from_checkpoint(path, train_tokens, validation_tokens)
# ^^^ THOG
