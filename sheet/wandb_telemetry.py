# vvv THOG
from __future__ import annotations

import importlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

from .stage6_source import (
    evaluation_metric_payload,
    init_resilient_telemetry,
    training_metric_payload,
)


INSTRUMENTATION_BACKENDS = ("tensorboard", "wandb", "none")
_BYTES_PER_GIB = float(1024 ** 3)


def _safe_exp(value: float) -> float:
    if value > 80.0:
        return math.inf
    return math.exp(value)


def _safe_scalar(value: Any) -> Optional[float | int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return None


def _scalar_metrics(metrics: Mapping[str, Any]) -> Dict[str, float | int]:
    scalars: Dict[str, float | int] = {}
    for name, value in metrics.items():
        scalar = _safe_scalar(value)
        if scalar is not None:
            scalars[name] = scalar
    return scalars


def _selected_backend() -> str:
    selected = os.environ.get("THOG2_INSTRUMENTATION", "tensorboard").strip().lower()
    if selected in INSTRUMENTATION_BACKENDS:
        return selected
    raise ValueError(
        "THOG2_INSTRUMENTATION must be tensorboard, wandb, or none; "
        f"got {selected!r}"
    )


def _tensorboard_root(name: str) -> Path:
    root = Path(os.environ.get("THOG2_CURVE_ROOT", "curves"))
    return root / name


def _training_metrics(payload: Mapping[str, Any]) -> Dict[str, Any]:
    metric = training_metric_payload(payload)
    update = int(metric["optimizer_update"])
    training_loss = float(metric["training_loss"])
    return {
        "optimizer/update": update,
        "tokens/seen": int(metric["tokens_seen"]),
        "time/train_seconds": float(metric["clean_training_seconds"]),
        "train/step_loss": training_loss,
        "train/loss": training_loss,
        "optim/lr": float(metric["learning_rate"]),
        "optim/grad_norm": float(metric["gradient_norm"]),
    }


def _evaluation_metrics(payload: Mapping[str, Any]) -> Dict[str, Any]:
    metric = evaluation_metric_payload(payload)
    test_loss = float(metric["validation_loss"])
    eval_loss = float(metric["training_evaluation_loss"])
    return {
        "optimizer/update": int(metric["optimizer_update"]),
        "tokens/seen": int(metric["tokens_seen"]),
        "test/loss": test_loss,
        "eval/val_loss": test_loss,
        "eval/loss": eval_loss,
        "test/perplexity": _safe_exp(test_loss),
        "eval/perplexity": _safe_exp(eval_loss),
    }


def _event_metrics(event: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    if event == "optimizer_progress":
        return _training_metrics(payload)
    if event == "evaluation_completed":
        return _evaluation_metrics(payload)
    return {}


def _final_metrics(result: Mapping[str, Any]) -> Dict[str, Any]:
    update = int(result["budget"]["completed_updates"])
    metrics: Dict[str, Any] = {
        "optimizer/update": update,
        "tokens/seen": int(result["budget"]["consumed_tokens"]),
        "time/train_seconds": float(result["timing"]["training_seconds"]),
        "perf/tokens_per_training_second": float(
            result["timing"]["tokens_per_training_second"]
        ),
        "model/persistent_parameters": int(
            result["parameter_report"]["persistent_parameters"]
        ),
        "model/dense_equivalent_parameters": int(
            result["parameter_report"]["dense_equivalent_total_parameters"]
        ),
        "resource/checkpoint_bytes": int(result["checkpoint"]["bytes"]),
    }

    samples = result.get("memory", {}).get("samples", [])
    if samples:
        metrics["gpu/peak_allocated_gb"] = (
            max(int(row["peak_allocated_bytes"]) for row in samples) / _BYTES_PER_GIB
        )
        metrics["gpu/peak_reserved_gb"] = (
            max(int(row["peak_reserved_bytes"]) for row in samples) / _BYTES_PER_GIB
        )

    evaluations = result.get("evaluations", [])
    if evaluations:
        final_test_loss = float(evaluations[-1]["val"])
        best_test_loss = min(float(row["val"]) for row in evaluations)
        metrics["test/final_loss"] = final_test_loss
        metrics["test/best_loss"] = best_test_loss

    diagnostics = result.get("sheet_diagnostics")
    if diagnostics is not None:
        for family, row in diagnostics["coefficient_utilization"].items():
            metrics[f"sheet/{family}/coefficient_rms"] = float(row["coefficient_rms"])
            metrics[f"sheet/{family}/high_depth_order_energy_fraction"] = float(
                row["high_depth_order_energy_fraction"]
            )
            metrics[f"sheet/{family}/high_row_order_energy_fraction"] = float(
                row["high_row_order_energy_fraction"]
            )
        metrics["sheet/compact_state_violation_count"] = len(
            diagnostics["compact_state_violations"]
        )
    return metrics


class _CudaSampler:
    def __init__(self, device: str, interval: int = 10) -> None:
        self.device = device
        self.interval = interval
        self.last_step: Optional[int] = None

    def _device_index(self) -> Optional[int]:
        if not torch.cuda.is_available():
            return None
        try:
            device = torch.device(self.device)
        except (TypeError, RuntimeError):
            return torch.cuda.current_device()
        if device.type != "cuda":
            return None
        if device.index is not None:
            return int(device.index)
        return torch.cuda.current_device()

    def sample(self, step: int, *, force: bool = False) -> Dict[str, Any]:
        if not force and step > 1 and step % self.interval != 0:
            return {}
        if self.last_step == step and not force:
            return {}
        self.last_step = step
        device_index = self._device_index()
        if device_index is None:
            return {}
        try:
            return {
                "gpu/memory_allocated_gb": (
                    torch.cuda.memory_allocated(device_index) / _BYTES_PER_GIB
                ),
                "gpu/memory_reserved_gb": (
                    torch.cuda.memory_reserved(device_index) / _BYTES_PER_GIB
                ),
                "gpu/max_memory_allocated_gb": (
                    torch.cuda.max_memory_allocated(device_index) / _BYTES_PER_GIB
                ),
            }
        except RuntimeError:
            return {}


class WandbTelemetry:
    """Rank-zero telemetry owner with TensorBoard default and W&B fallback option."""

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
        self.backend = _selected_backend()
        self.module: Optional[Any] = None
        self.run: Optional[Any] = None
        self.writer: Optional[Any] = None
        self.sampler = _CudaSampler(str(self.config.get("device", "cuda")))

    def start(self) -> None:
        if not self.enabled or self.backend == "none":
            return
        if self.backend == "wandb":
            self._start_wandb()
            return
        self._start_tensorboard()

    def _start_wandb(self) -> None:
        if self.run is not None:
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
            config={**self.config, "instrumentation": self.backend},
        )
        define_metric = run.define_metric if hasattr(run, "define_metric") else module.define_metric
        define_metric("optimizer/update")
        for metric in (
            "tokens/*",
            "time/*",
            "train/*",
            "eval/*",
            "test/*",
            "optim/*",
            "perf/*",
            "model/*",
            "resource/*",
            "gpu/*",
            "sheet/*",
        ):
            define_metric(metric, step_metric="optimizer/update")
        self.module = module
        self.run = run

    def _start_tensorboard(self) -> None:
        if self.writer is not None:
            return
        root = _tensorboard_root(self.name)
        root.mkdir(parents=True, exist_ok=True)
        module = importlib.import_module("torch.utils.tensorboard")
        summary_writer = getattr(module, "SummaryWriter", None)
        if summary_writer is None:
            self.backend = "wandb"
            self._start_wandb()
            return
        self.writer = summary_writer(log_dir=str(root))
        self.writer.add_text("run/artifact_name", self.name, 0)
        self.writer.add_text("run/group", self.group, 0)
        self.writer.add_text("run/job_type", self.job_type, 0)
        self.writer.add_text(
            "run/config_json",
            json.dumps({**self.config, "instrumentation": self.backend}, indent=2, sort_keys=True),
            0,
        )

    def log_event(self, event: str, payload: Mapping[str, Any]) -> None:
        if not self.enabled or self.backend == "none":
            return
        metrics = _event_metrics(event, payload)
        if not metrics:
            return
        step = int(metrics["optimizer/update"])
        metrics.update(self.sampler.sample(step))
        self._log_scalars(metrics, step)

    def _log_scalars(self, metrics: Mapping[str, Any], step: int) -> None:
        scalars = _scalar_metrics(metrics)
        if self.run is not None:
            try:
                self.run.log(scalars, step=step)
            except TypeError:
                self.run.log(scalars)
        if self.writer is not None:
            for name, value in scalars.items():
                self.writer.add_scalar(name, value, step)

    def add_initial_summary(self, parameter_report: Mapping[str, Any]) -> None:
        if self.backend == "none":
            return
        if self.run is not None:
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
        if self.writer is not None:
            self.writer.add_text(
                "run/parameter_report_json",
                json.dumps(dict(parameter_report), indent=2, sort_keys=True),
                0,
            )
            self._log_scalars({
                "model/persistent_parameters": int(parameter_report["persistent_parameters"]),
                "model/dense_equivalent_parameters": int(
                    parameter_report["dense_equivalent_total_parameters"]
                ),
            }, 0)

    def add_final_result(self, result: Mapping[str, Any]) -> None:
        metrics = _final_metrics(result)
        step = int(metrics["optimizer/update"])
        metrics.update(self.sampler.sample(step, force=True))
        self._log_scalars(metrics, step)
        if self.run is not None:
            evaluations = result.get("evaluations", [])
            final_test_loss = evaluations[-1]["val"] if evaluations else None
            best_test_loss = (
                min(row["val"] for row in evaluations) if evaluations else None
            )
            self.run.summary.update({
                "completed_updates": step,
                "consumed_tokens": result["budget"]["consumed_tokens"],
                "final_test_loss": final_test_loss,
                "best_test_loss": best_test_loss,
                "training_seconds": result["timing"]["training_seconds"],
                "tokens_per_training_second": result["timing"][
                    "tokens_per_training_second"
                ],
                "checkpoint_bytes": result["checkpoint"]["bytes"],
            })
        if self.writer is not None:
            self.writer.flush()

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None


def attach_telemetry(trainer: Any, telemetry: WandbTelemetry) -> None:
    """Attach event logging without moving telemetry into clean train/eval timers."""

    original_progress = trainer._print_progress

    def progress(run_id: str, event: str, **payload: Any) -> None:
        original_progress(run_id, event, **payload)
        if trainer.distributed.is_primary:
            telemetry_payload = dict(payload)
            multiplier = int(getattr(trainer, "telemetry_token_multiplier", 1))
            if "consumed_tokens" in telemetry_payload:
                telemetry_payload["consumed_tokens"] = (
                    int(telemetry_payload["consumed_tokens"]) * multiplier
                )
            telemetry.log_event(event, telemetry_payload)

    trainer._print_progress = progress


__all__ = ["INSTRUMENTATION_BACKENDS", "WandbTelemetry", "attach_telemetry"]
# ^^^ THOG
