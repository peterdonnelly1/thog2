# vvv THOG
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from torch import Tensor

from .checkpoints import (
    load_payload,
    optimizer_group_names,
    restore_rng_state,
    strip_compiled_prefix,
    validate_compatibility,
)
from .compact_identity import (
    BASIS_FAMILY_CHEBYSHEV,
    GEOMETRY_PRESET_LEGACY_SHEET_COL,
)
from .trainer_state import TrainerState
from .training_config import (
    CHECKPOINT_SCHEMA_VERSION,
    EXECUTION_OVERRIDE_FIELDS,
    TrainingConfig,
)


# vvv THOG schema-1 SHEET checkpoints predate explicit geometry selectors but map exactly to legacy_sheet_col
_LEGACY_SHEET_SCHEMA_VERSION = 1
_LEGACY_SHEET_COMPATIBILITY_FIELDS = (
    "model_type",
    "block_size",
    "vocab_size",
    "n_layer",
    "n_head",
    "n_embd",
    "dropout",
    "bias",
    "depth_order",
    "base_row_order",
    "residual_init_policy",
    "residual_init_depth_source",
    "residual_init_depth_value",
    "basis_version",
    "row_order_scaling_rule",
)
_LEGACY_SHEET_CONTROL_FIELDS = (
    "batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "min_learning_rate",
    "warmup_updates",
    "decay_updates",
    "decay_learning_rate",
    "weight_decay",
    "beta1",
    "beta2",
    "grad_clip",
    "model_seed",
    "data_seed",
)


def _trainer_state_from_payload(payload: Mapping[str, Any]) -> TrainerState:
    values: Dict[str, Any] = asdict(TrainerState())
    values.update(dict(payload["trainer_state"]))
    return TrainerState(**values)


def _validate_override_fields(overrides: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    override_values = dict(overrides or {})
    forbidden = sorted(set(override_values) - EXECUTION_OVERRIDE_FIELDS)
    if forbidden:
        raise ValueError(
            "resume overrides are restricted to execution fields; "
            f"got {forbidden}"
        )
    return override_values


def _align_legacy_optimizer_groups_to_checkpoint(
    trainer: Any,
    payload: Mapping[str, Any],
) -> None:
    """Restore the schema-1 optimizer's exact named parameter order before loading state."""
    checkpoint_groups = tuple(
        tuple(group)
        for group in payload["optimizer_group_parameter_names"]
    )
    if len(trainer.optimizer.param_groups) != len(checkpoint_groups):
        raise ValueError(
            "optimizer group count is incompatible with schema-1 SHEET checkpoint"
        )

    current_parameters = dict(trainer.raw_model.named_parameters())
    checkpoint_names = [
        name
        for group in checkpoint_groups
        for name in group
    ]
    if len(checkpoint_names) != len(set(checkpoint_names)):
        raise ValueError(
            "schema-1 SHEET checkpoint repeats an optimizer parameter name"
        )

    missing_names = sorted(set(checkpoint_names) - set(current_parameters))
    if missing_names:
        raise ValueError(
            "schema-1 SHEET checkpoint optimizer names are missing from the "
            f"preserved model: {missing_names}"
        )

    checkpoint_parameter_ids = {
        id(current_parameters[name])
        for name in checkpoint_names
    }
    trainable_parameter_ids = {
        id(parameter)
        for parameter in trainer.raw_model.parameters()
        if parameter.requires_grad
    }
    if checkpoint_parameter_ids != trainable_parameter_ids:
        raise ValueError(
            "schema-1 SHEET checkpoint optimizer groups do not cover the "
            "preserved trainable parameters exactly"
        )

    for optimizer_group, checkpoint_names_for_group in zip(
        trainer.optimizer.param_groups,
        checkpoint_groups,
    ):
        optimizer_group["params"] = [
            current_parameters[name]
            for name in checkpoint_names_for_group
        ]
        optimizer_group["parameter_names"] = checkpoint_names_for_group

    if optimizer_group_names(trainer.optimizer) != checkpoint_groups:
        raise RuntimeError(
            "failed to reconstruct schema-1 SHEET optimizer parameter order"
        )


def _validate_legacy_sheet_checkpoint(
    payload: Mapping[str, Any],
    expected_config: TrainingConfig,
) -> None:
    if expected_config.model_type != "thog2_sheet":
        raise ValueError("schema-1 SHEET checkpoint requires a SHEET expected_config")
    identity = expected_config.compact_identity_metadata()
    if identity["geometry_preset"] != GEOMETRY_PRESET_LEGACY_SHEET_COL:
        raise ValueError(
            "schema-1 SHEET checkpoint can resume only as legacy_sheet_col"
        )
    if identity["basis_family"] != BASIS_FAMILY_CHEBYSHEV:
        raise ValueError(
            "schema-1 SHEET checkpoint can resume only with the Chebyshev basis"
        )

    checkpoint_signature = dict(payload.get("compatibility_signature", {}))
    expected_signature = expected_config.compatibility_signature()
    mismatches = []
    for name in _LEGACY_SHEET_COMPATIBILITY_FIELDS:
        actual = checkpoint_signature.get(name)
        expected = expected_signature[name]
        if actual != expected:
            mismatches.append(
                f"{name}: checkpoint={actual!r}, expected={expected!r}"
            )

    checkpoint_config = dict(payload.get("trainer_config", {}))
    for name in _LEGACY_SHEET_CONTROL_FIELDS:
        actual = checkpoint_config.get(name)
        expected = getattr(expected_config, name)
        if actual != expected:
            mismatches.append(
                f"{name}: checkpoint={actual!r}, expected={expected!r}"
            )

    if mismatches:
        raise ValueError(
            "incompatible schema-1 legacy_sheet_col checkpoint: "
            + "; ".join(mismatches)
        )
# ^^^ THOG


class TrainerCheckpointResumeMixin:
    @classmethod
    def from_checkpoint(
        cls,
        path: Union[str, Path],
        train_tokens: Tensor,
        validation_tokens: Tensor,
        *,
        overrides: Optional[Mapping[str, Any]] = None,
        expected_config: Optional[TrainingConfig] = None,
    ):
        payload = load_payload(path)
        if "schema_version" not in payload:
            if expected_config is None:
                raise ValueError(
                    "legacy dense checkpoint load requires expected_config"
                )
            return cls.from_legacy_dense_checkpoint(
                path,
                train_tokens,
                validation_tokens,
                expected_config=expected_config,
            )

        # vvv THOG resume original schema-1 SHEET checkpoints through the preserved legacy_sheet_col geometry
        if (
            payload.get("schema_version") == _LEGACY_SHEET_SCHEMA_VERSION
            and payload.get("model_type") == "thog2_sheet"
        ):
            if expected_config is None:
                raise ValueError(
                    "schema-1 SHEET checkpoint load requires expected_config"
                )
            return cls.from_legacy_sheet_checkpoint(
                path,
                train_tokens,
                validation_tokens,
                payload=payload,
                overrides=overrides,
                expected_config=expected_config,
            )
        # ^^^ THOG

        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"checkpoint schema_version mismatch: expected "
                f"{CHECKPOINT_SCHEMA_VERSION}, got {payload.get('schema_version')!r}"
            )

        checkpoint_config = TrainingConfig(**payload["trainer_config"])
        if expected_config is not None:
            validate_compatibility(payload, expected_config)
        # override_values = dict(overrides or {})
        # forbidden = sorted(
        #     set(override_values) - EXECUTION_OVERRIDE_FIELDS
        # )
        # if forbidden:
        #     raise ValueError(
        #         "resume overrides are restricted to execution fields; "
        #         f"got {forbidden}"
        #     )
        # vvv THOG share exact override validation with schema-1 SHEET resume
        override_values = _validate_override_fields(overrides)
        # ^^^ THOG
        values = asdict(checkpoint_config)
        values.update(override_values)
        resumed_config = TrainingConfig(**values)
        validate_compatibility(payload, resumed_config)

        trainer = cls(resumed_config, train_tokens, validation_tokens)
        trainer.raw_model.load_state_dict(
            strip_compiled_prefix(payload["model"])
        )
        expected_groups = tuple(
            tuple(group)
            for group in payload["optimizer_group_parameter_names"]
        )
        if optimizer_group_names(trainer.optimizer) != expected_groups:
            raise ValueError(
                "optimizer group parameter ordering is incompatible "
                "with checkpoint"
            )
        trainer.optimizer.load_state_dict(payload["optimizer"])
        # trainer.state = TrainerState(**payload["trainer_state"])
        # vvv THOG restore new recovery counters safely from older schema-2 checkpoints
        trainer.state = _trainer_state_from_payload(payload)
        # ^^^ THOG
        if trainer.state.completed_updates != int(
            payload["completed_updates"]
        ):
            raise ValueError(
                "checkpoint completed update counters disagree"
            )
        trainer.batch_source.load_state_dict(payload["batch_source"])
        restore_rng_state(payload["rng_state"])
        trainer.distributed.barrier()
        trainer._record("checkpoint_resumed", path=str(path))
        return trainer

    # vvv THOG exact compatibility bridge for the original KARITANE_LONG checkpoint family
    @classmethod
    def from_legacy_sheet_checkpoint(
        cls,
        path: Union[str, Path],
        train_tokens: Tensor,
        validation_tokens: Tensor,
        *,
        payload: Mapping[str, Any],
        overrides: Optional[Mapping[str, Any]],
        expected_config: TrainingConfig,
    ):
        _validate_legacy_sheet_checkpoint(payload, expected_config)
        override_values = _validate_override_fields(overrides)
        values = asdict(expected_config)
        values.update(override_values)
        resumed_config = TrainingConfig(**values)

        trainer = cls(resumed_config, train_tokens, validation_tokens)
        trainer.raw_model.load_state_dict(
            strip_compiled_prefix(payload["model"])
        )
        expected_groups = tuple(
            tuple(group)
            for group in payload["optimizer_group_parameter_names"]
        )
        if optimizer_group_names(trainer.optimizer) != expected_groups:
            _align_legacy_optimizer_groups_to_checkpoint(trainer, payload)
        if optimizer_group_names(trainer.optimizer) != expected_groups:
            raise ValueError(
                "optimizer group parameter ordering is incompatible "
                "with schema-1 SHEET checkpoint"
            )
        trainer.optimizer.load_state_dict(payload["optimizer"])
        trainer.state = _trainer_state_from_payload(payload)
        if trainer.state.completed_updates != int(payload["completed_updates"]):
            raise ValueError("checkpoint completed update counters disagree")
        trainer.batch_source.load_state_dict(payload["batch_source"])
        restore_rng_state(payload["rng_state"])
        trainer.distributed.barrier()
        trainer._record(
            "legacy_sheet_checkpoint_resumed",
            path=str(path),
            source_schema_version=_LEGACY_SHEET_SCHEMA_VERSION,
        )
        return trainer
    # ^^^ THOG

    @classmethod
    def from_legacy_dense_checkpoint(
        cls,
        path: Union[str, Path],
        train_tokens: Tensor,
        validation_tokens: Tensor,
        *,
        expected_config: TrainingConfig,
    ):
        if expected_config.model_type != "dense":
            raise ValueError(
                "legacy checkpoints are supported only for dense models"
            )
        payload = load_payload(path)
        required = (
            "n_layer",
            "n_head",
            "n_embd",
            "block_size",
            "bias",
            "vocab_size",
        )
        mismatches = []
        for name in required:
            actual = payload.get("model_args", {}).get(name)
            expected = getattr(expected_config, name)
            if actual != expected:
                mismatches.append(
                    f"{name}: checkpoint={actual!r}, "
                    f"expected={expected!r}"
                )
        if mismatches:
            raise ValueError(
                "incompatible legacy dense checkpoint: "
                + "; ".join(mismatches)
            )
        trainer = cls(expected_config, train_tokens, validation_tokens)
        trainer.raw_model.load_state_dict(
            strip_compiled_prefix(payload["model"])
        )
        if "optimizer" in payload:
            trainer.optimizer.load_state_dict(payload["optimizer"])
        trainer.state.completed_updates = int(
            payload.get("completed_updates", payload.get("iter_num", 0))
        )
        trainer.state.best_validation_loss = float(
            payload.get("best_val_loss", float("inf"))
        )
        trainer.distributed.barrier()
        trainer._record("legacy_checkpoint_loaded", path=str(path))
        return trainer
# ^^^ THOG
