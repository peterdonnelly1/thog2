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
        target_config: Optional[TrainingConfig] = None,
        allowed_override_fields: Optional[set[str]] = None,
    ):
        payload = load_payload(path)
        if "schema_version" not in payload:
            if expected_config is None and target_config is None:
                raise ValueError("legacy dense checkpoint load requires expected_config")
            return cls.from_legacy_dense_checkpoint(
                path,
                train_tokens,
                validation_tokens,
                expected_config=expected_config or target_config,
            )

        checkpoint_config = TrainingConfig(**payload["trainer_config"])
        requested_config = target_config or expected_config
        if requested_config is not None:
            validate_compatibility(payload, requested_config)
        # vvv THOG
        # Resume permits operational overrides and material equality assertions. A supplied
        # material field that exactly matches the checkpoint is an assertion, not a mutation.
        checkpoint_values = asdict(checkpoint_config)
        requested_override_values = dict(overrides or {})
        allowed_fields = set(allowed_override_fields or EXECUTION_OVERRIDE_FIELDS)
        override_values = {}
        forbidden = []
        for name, value in requested_override_values.items():
            if name in allowed_fields:
                override_values[name] = value
            elif checkpoint_values.get(name) == value:
                continue
            else:
                forbidden.append(name)
        if requested_config is not None:
            requested_values = asdict(requested_config)
            for name, value in requested_values.items():
                if checkpoint_values.get(name) != value:
                    if name in allowed_fields:
                        override_values[name] = value
                    else:
                        forbidden.append(name)
        forbidden = sorted(set(forbidden))
        if forbidden:
            raise ValueError(f"resume overrides are not allowed for fields: {forbidden}")
        values = asdict(checkpoint_config)
        values.update(override_values)
        # ^^^ THOG
        resumed_config = TrainingConfig(**values)
        validate_compatibility(payload, resumed_config)

        trainer = cls(resumed_config, train_tokens, validation_tokens)
        trainer.raw_model.load_state_dict(strip_compiled_prefix(payload["model"]))
        expected_groups = tuple(tuple(group) for group in payload["optimizer_group_parameter_names"])
        if optimizer_group_names(trainer.optimizer) != expected_groups:
            raise ValueError("optimizer group parameter ordering is incompatible with checkpoint")
        trainer.optimizer.load_state_dict(payload["optimizer"])
        trainer.state = TrainerState(**payload["trainer_state"])
        if trainer.state.completed_updates != int(payload["completed_updates"]):
            raise ValueError("checkpoint completed update counters disagree")
        trainer.batch_source.load_state_dict(payload["batch_source"])
        restore_rng_state(payload["rng_state"])
        trainer.distributed.barrier()
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
