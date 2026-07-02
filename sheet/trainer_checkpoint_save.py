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
from .training_config import CHECKPOINT_SCHEMA_VERSION


class TrainerCheckpointSaveMixin:
    def checkpoint_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_type": self.config.model_type,
            "model_args": self.config.model_arguments(),
            "compatibility_signature": self.config.compatibility_signature(),
            "basis_version": self.config.basis_version,
            "row_order_scaling_rule": self.config.row_order_scaling_rule,
            "model": compact_model_state(self.raw_model, self.config.model_type),
            "optimizer": self.optimizer.state_dict(),
            "optimizer_group_parameter_names": optimizer_group_names(self.optimizer),
            "trainer_state": asdict(self.state),
            "completed_updates": self.state.completed_updates,
            "trainer_config": asdict(self.config),
            "batch_source": self.batch_source.state_dict(),
            "rng_state": capture_rng_state(),
            "parameter_report": self.parameter_report,
            "distributed_training": self.distributed.report(),
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
                "distributed": self.distributed.report(),
            },
            indent=2,
            sort_keys=True,
        )
# ^^^ THOG
