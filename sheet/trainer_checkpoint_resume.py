# vvv THOG
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from torch import Tensor

from .checkpoints import (
    load_payload,
    optimizer_group_names,
    restore_rng_state,
    strip_compiled_prefix,
    validate_compatibility,
)
from .trainer_state import TrainerState
from .training_config import EXECUTION_OVERRIDE_FIELDS, TrainingConfig


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

        checkpoint_config = TrainingConfig(**payload["trainer_config"])
        if expected_config is not None:
            validate_compatibility(payload, expected_config)
        override_values = dict(overrides or {})
        forbidden = sorted(
            set(override_values) - EXECUTION_OVERRIDE_FIELDS
        )
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
        trainer.model.load_state_dict(
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
        trainer.state = TrainerState(**payload["trainer_state"])
        if trainer.state.completed_updates != int(
            payload["completed_updates"]
        ):
            raise ValueError(
                "checkpoint completed update counters disagree"
            )
        trainer.batch_source.load_state_dict(payload["batch_source"])
        restore_rng_state(payload["rng_state"])
        trainer._record("checkpoint_resumed", path=str(path))
        return trainer

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
        trainer.model.load_state_dict(
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
        trainer._record("legacy_checkpoint_loaded", path=str(path))
        return trainer
# ^^^ THOG
