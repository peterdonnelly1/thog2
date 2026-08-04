# vvv THOG
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Union

from .checkpoints import (
    capture_rng_state,
    compact_model_state,
    optimizer_group_names,
    save_payload,
)
# vvv THOG explicit PLASTIC checkpoint-format identity
from .checkpoints import PLASTIC_DEPTH_CHECKPOINT_FORMAT_VERSION
# ^^^ THOG
from .training_config import CHECKPOINT_SCHEMA_VERSION


class TrainerCheckpointSaveMixin:
    def checkpoint_payload(self) -> Dict[str, Any]:
        compact_identity = self.config.compact_identity_metadata()
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_type": self.config.model_type,
            "model_args": self.config.model_arguments(),
            "compatibility_signature": self.config.compatibility_signature(),
            "compact_identity": compact_identity,
            # vvv THOG enabled PLASTIC checkpoints state their active-prefix/gauge format independently of mutable identity aliases
            **(
                {"plastic_depth_checkpoint_format_version": PLASTIC_DEPTH_CHECKPOINT_FORMAT_VERSION}
                if self.config.plastic__enabled
                else {}
            ),
            # ^^^ THOG
            "basis_version": self.config.basis_version,
            "row_order_scaling_rule": self.config.row_order_scaling_rule,
            "model": compact_model_state(self.raw_model, self.config.model_type),
            "optimizer": self.optimizer.state_dict(),
            "optimizer_group_parameter_names": optimizer_group_names(self.optimizer),
            "trainer_state": asdict(self.state),
            "completed_updates": self.state.completed_updates,
            # vvv THOG preserve the pre-PLASTIC checkpoint configuration serialization for source history
            # "trainer_config": asdict(self.config),
            "trainer_config": self.config.persistent_dict(),
            # ^^^ THOG
            "batch_source": self.batch_source.state_dict(),
            "rng_state": capture_rng_state(),
            "parameter_report": {**self.parameter_report, "compact_identity": compact_identity},
            "distributed_training": self.distributed.report(),
            "lifecycle": getattr(self, "lifecycle_metadata", None),                                                                                       # <<< THOG persist logical-run identity, lineage, W&B identity, target and LR phases
        }

    def save_checkpoint(self, path: Union[str, Path]) -> Path:
        target = Path(path)
        if self.distributed.is_primary:
            target = save_payload(self.checkpoint_payload(), target)
        self.distributed.barrier()
        self._record(
            "checkpoint_saved",
            path=str(target),
            writer_rank=0,
        )
        return target

    def startup_report_json(self) -> str:
        return json.dumps(
            {
                "model_type": self.config.model_type,
                "model_args": self.config.model_arguments(),
                "completed_updates": self.state.completed_updates,
                "parameter_report": self.parameter_report,
                "compact_identity": self.config.compact_identity_metadata(),
                "distributed": self.distributed.report(),
            },
            indent=2,
            sort_keys=True,
        )
# ^^^ THOG
