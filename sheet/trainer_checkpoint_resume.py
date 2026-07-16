# vvv THOG
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Union

from torch import Tensor

from .checkpoints import (
    load_payload,
    optimizer_group_names,
    restore_rng_state,
    strip_compiled_prefix,
    validate_compatibility,
)
from .run_lifecycle import lifecycle_from_checkpoint
from .trainer_state import TrainerState
from .training_config import EXECUTION_OVERRIDE_FIELDS, TrainingConfig


class TrainerCheckpointResumeMixin:
    @staticmethod
    def _validate_target_config(
        checkpoint_config: TrainingConfig,
        target_config: TrainingConfig,
        allowed_override_fields: Iterable[str],
    ) -> None:
        allowed = set(allowed_override_fields)
        checkpoint_values = asdict(checkpoint_config)
        target_values = asdict(target_config)
        mismatches = [
            f"{name}: checkpoint={checkpoint_values[name]!r}, requested={target_values[name]!r}"
            for name in checkpoint_values
            if name not in allowed and checkpoint_values[name] != target_values[name]
        ]
        if mismatches:
            raise ValueError("checkpoint configuration mismatch: " + "; ".join(mismatches))

    @classmethod
    def from_checkpoint(
        cls,
        path: Union[str, Path],
        train_tokens: Tensor,
        validation_tokens: Tensor,
        *,
        overrides: Optional[Mapping[str, Any]] = None,
        expected_config: Optional[TrainingConfig] = None,
        target_config: Optional[TrainingConfig] = None,
        allowed_override_fields: Optional[Iterable[str]] = None,
    ):
        payload = load_payload(path)
        if "schema_version" not in payload or "trainer_config" not in payload:
            raise ValueError("checkpoint does not use the enhanced THOG2 checkpoint schema")
        checkpoint_config = TrainingConfig(**payload["trainer_config"])

        if target_config is not None and (overrides or expected_config is not None):
            raise ValueError("target_config cannot be combined with overrides or expected_config")
        if target_config is not None:
            resumed_config = target_config
            cls._validate_target_config(
                checkpoint_config,
                resumed_config,
                allowed_override_fields or EXECUTION_OVERRIDE_FIELDS,
            )
        else:
            if expected_config is not None:
                validate_compatibility(payload, expected_config)
            override_values = dict(overrides or {})
            forbidden = sorted(set(override_values) - EXECUTION_OVERRIDE_FIELDS)
            if forbidden:
                raise ValueError(
                    "resume overrides are restricted to execution fields; "
                    f"got {forbidden}"
                )
            values = asdict(checkpoint_config)
            values.update(override_values)
            resumed_config = TrainingConfig(**values)
        validate_compatibility(payload, resumed_config)

        trainer = cls(resumed_config, train_tokens, validation_tokens)
        checkpoint_world_size = int(payload.get("distributed_training", {}).get("world_size", 1))
        if checkpoint_world_size != int(trainer.distributed.world_size):
            trainer.close()
            raise ValueError(
                "resume world size mismatch: "
                f"checkpoint={checkpoint_world_size}, current={trainer.distributed.world_size}"
            )
        trainer.raw_model.load_state_dict(strip_compiled_prefix(payload["model"]))
        expected_groups = tuple(tuple(group) for group in payload["optimizer_group_parameter_names"])
        if optimizer_group_names(trainer.optimizer) != expected_groups:
            trainer.close()
            raise ValueError("optimizer group parameter ordering is incompatible with checkpoint")
        trainer.optimizer.load_state_dict(payload["optimizer"])
        if "scaler" not in payload:
            trainer.close()
            raise ValueError("enhanced checkpoint is missing GradScaler state")
        trainer.scaler.load_state_dict(payload["scaler"])
        trainer.state = TrainerState(**payload["trainer_state"])
        if trainer.state.completed_updates != int(payload["completed_updates"]):
            trainer.close()
            raise ValueError("checkpoint completed update counters disagree")
        trainer.batch_source.load_state_dict(payload["batch_source"])
        restore_rng_state(payload["rng_state"])
        if "lifecycle" in payload:
            trainer.lifecycle_metadata = lifecycle_from_checkpoint(payload)                                                                           # <<< THOG restore logical run metadata when present; OWT resolver requires it
        trainer.distributed.barrier()
        trainer._record("checkpoint_resumed", path=str(path))
        return trainer
# ^^^ THOG
