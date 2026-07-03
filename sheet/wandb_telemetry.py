# vvv THOG
from __future__ import annotations

import importlib
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .stage6_source import (
    evaluation_metric_payload,
    init_resilient_telemetry,
    training_metric_payload,
)


class WandbTelemetry:
    """One resilient rank-zero W&B owner for THOG2 training runs."""

    def __init__(
        self,
        *,
        enabled: bool,
        project: str,
        entity: Optional[str],
        mode: str,
        root: Path,
        name: str,
        group: str,
        job_type: str,
        config: Mapping[str, Any],
    ) -> None:
        self.enabled = bool(enabled)
        self.project = project
        self.entity = entity
        self.mode = mode
        self.root = Path(root)
        self.name = name
        self.group = group
        self.job_type = job_type
        self.config = dict(config)
        self.module: Optional[Any] = None
        self.run: Optional[Any] = None

    def start(self) -> None:
        if not self.enabled or self.run is not None:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        os.environ["WANDB_DIR"] = str(self.root)
        os.environ["WANDB_MODE"] = self.mode
        module = importlib.import_module("wandb")
        run = init_resilient_telemetry(
            module,
            project=self.project,
            entity=self.entity,
            name=self.name,
            group=self.group,
            job_type=self.job_type,
            config=self.config,
        )
        define_metric = getattr(run, "define_metric", module.define_metric)
        define_metric("optimizer_update")
        for metric in (
            "iter",
            "tokens_seen",
            "clean_training_seconds",
            "training_loss",
            "validation_loss",
            "training_evaluation_loss",
            "learning_rate",
            "gradient_norm",
            "train/*",
            "eval/*",
            "resource/*",
            "sheet/*",
        ):
            define_metric(metric, step_metric="optimizer_update")
        self.module = module
        self.run = run

    def log_event(self, event: str, payload: Mapping[str, Any]) -> None:
        if self.run is None:
            return
        metric: Optional[Dict[str, Any]] = None
        if event == "optimizer_progress":
            metric = training_metric_payload(payload)
            metric.update({
                "train/loss": metric["training_loss"],
                "train/learning_rate": metric["learning_rate"],
                "train/gradient_norm": metric["gradient_norm"],
            })
        elif event == "evaluation_completed":
            metric = evaluation_metric_payload(payload)
            validation_loss = metric["validation_loss"]
            training_loss = metric["training_evaluation_loss"]
            metric.update({
                "validation_perplexity": math.exp(validation_loss),
                "eval/val_loss": validation_loss,
                "eval/train_loss": training_loss,
                "eval/val_perplexity": math.exp(validation_loss),
                "eval/train_perplexity": math.exp(training_loss),
            })
        if metric is not None:
            self.run.log(metric)

    def add_initial_summary(self, parameter_report: Mapping[str, Any]) -> None:
        if self.run is None:
            return
        self.run.summary.update({
            "artifact_name": self.name,
            "artifact_prefix": self.config.get("artifact_prefix"),
            "model_type": self.config.get("model_type"),
            "comparison_group": self.group,
            "persistent_parameters": parameter_report["persistent_parameters"],
            "dense_equivalent_parameters": parameter_report[
                "dense_equivalent_total_parameters"
            ],
        })

    def add_final_result(self, result: Mapping[str, Any]) -> None:
        if self.run is None:
            return
        update = int(result["budget"]["completed_updates"])
        final_payload: Dict[str, Any] = {
            "optimizer_update": update,
            "iter": update,
            "tokens_seen": int(result["budget"]["consumed_tokens"]),
            "clean_training_seconds": float(result["timing"]["training_seconds"]),
            "resource/persistent_parameters": int(
                result["parameter_report"]["persistent_parameters"]
            ),
            "resource/dense_equivalent_parameters": int(
                result["parameter_report"]["dense_equivalent_total_parameters"]
            ),
            "resource/checkpoint_bytes": int(result["checkpoint"]["bytes"]),
            "resource/tokens_per_training_second": float(
                result["timing"]["tokens_per_training_second"]
            ),
        }
        samples = result.get("memory", {}).get("samples", [])
        if samples:
            final_payload["resource/peak_allocated_bytes"] = max(
                int(row["peak_allocated_bytes"]) for row in samples
            )
            final_payload["resource/peak_reserved_bytes"] = max(
                int(row["peak_reserved_bytes"]) for row in samples
            )
        diagnostics = result.get("sheet_diagnostics")
        if diagnostics is not None:
            for family, row in diagnostics["coefficient_utilization"].items():
                final_payload[f"sheet/{family}/coefficient_rms"] = float(
                    row["coefficient_rms"]
                )
                final_payload[
                    f"sheet/{family}/high_depth_order_energy_fraction"
                ] = float(row["high_depth_order_energy_fraction"])
                final_payload[
                    f"sheet/{family}/high_row_order_energy_fraction"
                ] = float(row["high_row_order_energy_fraction"])
            final_payload["sheet/compact_state_violation_count"] = len(
                diagnostics["compact_state_violations"]
            )
        self.run.log(final_payload)
        evaluations = result.get("evaluations", [])
        final_validation_loss = evaluations[-1]["val"] if evaluations else None
        best_validation_loss = (
            min(row["val"] for row in evaluations) if evaluations else None
        )
        self.run.summary.update({
            "completed_updates": update,
            "consumed_tokens": result["budget"]["consumed_tokens"],
            "final_validation_loss": final_validation_loss,
            "best_validation_loss": best_validation_loss,
            "training_seconds": result["timing"]["training_seconds"],
            "tokens_per_training_second": result["timing"][
                "tokens_per_training_second"
            ],
            "checkpoint_bytes": result["checkpoint"]["bytes"],
        })

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None


def attach_telemetry(trainer: Any, telemetry: WandbTelemetry) -> None:
    """Attach event logging without moving telemetry into clean train/eval timers."""

    original_progress = trainer._print_progress

    def progress(run_id: str, event: str, **payload: Any) -> None:
        original_progress(run_id, event, **payload)
        if trainer.distributed.is_primary:
            telemetry.log_event(event, payload)

    trainer._print_progress = progress


__all__ = ["WandbTelemetry", "attach_telemetry"]
# ^^^ THOG
