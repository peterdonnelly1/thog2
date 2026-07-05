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


def _safe_exp(value: float) -> float:
    if value > 80.0:
        return math.inf
    return math.exp(value)


def training_metrics(payload: Mapping[str, Any]) -> Dict[str, Any]:
    raw = training_metric_payload(payload)
    update = int(raw["optimizer_update"])
    return {
        "optimizer/update": update,
        "tokens/seen": int(raw["tokens_seen"]),
        "time/train_seconds": float(raw["clean_training_seconds"]),
        "train/step_loss": float(raw["training_loss"]),
        "optim/lr": float(raw["learning_rate"]),
        "optim/grad_norm": float(raw["gradient_norm"]),
    }


def evaluation_metrics(payload: Mapping[str, Any]) -> Dict[str, Any]:
    raw = evaluation_metric_payload(payload)
    validation_loss = float(raw["validation_loss"])
    training_loss = float(raw["training_evaluation_loss"])
    return {
        "optimizer/update": int(raw["optimizer_update"]),
        "tokens/seen": int(raw["tokens_seen"]),
        "val/loss": validation_loss,
        "train/loss": training_loss,
        "val/perplexity": _safe_exp(validation_loss),
        "train/perplexity": _safe_exp(training_loss),
    }


def final_result_metrics(result: Mapping[str, Any]) -> Dict[str, Any]:
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
        final_validation_loss = float(evaluations[-1]["val"])
        best_validation_loss = min(float(row["val"]) for row in evaluations)
        metrics["val/final_loss"] = final_validation_loss
        metrics["val/best_loss"] = best_validation_loss

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


class SystemMetricSampler:
    """Low-rate local system sampler for TensorBoard/W&B curve backends."""

    def __init__(
        self,
        *,
        enabled: bool,
        interval: int,
        device: str,
        disk_root: Path,
    ) -> None:
        self.enabled = bool(enabled) and int(interval) > 0
        self.interval = int(interval)
        self.device = device
        self.disk_root = Path(disk_root)
        self.last_step: Optional[int] = None
        self.psutil_module: Optional[Any] = None
        self.psutil_checked = False
        self.nvml_module: Optional[Any] = None
        self.nvml_handle: Optional[Any] = None
        self.nvml_checked = False

    def _cuda_index(self) -> Optional[int]:
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

    def _psutil(self) -> Optional[Any]:
        if not self.psutil_checked:
            self.psutil_checked = True
            try:
                self.psutil_module = importlib.import_module("psutil")
            except ImportError:
                self.psutil_module = None
        return self.psutil_module

    def _nvml(self) -> tuple[Optional[Any], Optional[Any]]:
        if not self.nvml_checked:
            self.nvml_checked = True
            try:
                module = importlib.import_module("pynvml")
                module.nvmlInit()
                device_index = self._cuda_index()
                if device_index is not None:
                    self.nvml_handle = module.nvmlDeviceGetHandleByIndex(device_index)
                    self.nvml_module = module
            except Exception:
                self.nvml_module = None
                self.nvml_handle = None
        return self.nvml_module, self.nvml_handle

    def sample(self, step: int, *, force: bool = False) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        if not force and step > 1 and step % self.interval != 0:
            return {}
        if self.last_step == step and not force:
            return {}
        self.last_step = step

        metrics: Dict[str, Any] = {}
        self._sample_torch_cuda(metrics)
        self._sample_psutil(metrics)
        self._sample_nvml(metrics)
        return metrics

    def _sample_torch_cuda(self, metrics: Dict[str, Any]) -> None:
        device_index = self._cuda_index()
        if device_index is None:
            return
        try:
            metrics["gpu/memory_allocated_gb"] = (
                torch.cuda.memory_allocated(device_index) / _BYTES_PER_GIB
            )
            metrics["gpu/memory_reserved_gb"] = (
                torch.cuda.memory_reserved(device_index) / _BYTES_PER_GIB
            )
            metrics["gpu/max_memory_allocated_gb"] = (
                torch.cuda.max_memory_allocated(device_index) / _BYTES_PER_GIB
            )
        except RuntimeError:
            return

    def _sample_psutil(self, metrics: Dict[str, Any]) -> None:
        module = self._psutil()
        if module is None:
            return
        try:
            memory = module.virtual_memory()
            disk = module.disk_usage(str(self.disk_root))
            process = module.Process(os.getpid())
            process_memory = process.memory_info()
            metrics.update({
                "system/cpu_percent": float(module.cpu_percent(interval=None)),
                "system/ram_used_gb": float(memory.used) / _BYTES_PER_GIB,
                "system/ram_percent": float(memory.percent),
                "system/disk_used_gb": float(disk.used) / _BYTES_PER_GIB,
                "system/disk_percent": float(disk.percent),
                "process/rss_gb": float(process_memory.rss) / _BYTES_PER_GIB,
                "process/vms_gb": float(process_memory.vms) / _BYTES_PER_GIB,
            })
        except Exception:
            return

    def _sample_nvml(self, metrics: Dict[str, Any]) -> None:
        module, handle = self._nvml()
        if module is None or handle is None:
            return
        try:
            utilization = module.nvmlDeviceGetUtilizationRates(handle)
            memory = module.nvmlDeviceGetMemoryInfo(handle)
            metrics.update({
                "gpu/util_percent": float(utilization.gpu),
                "gpu/nvml_memory_used_gb": float(memory.used) / _BYTES_PER_GIB,
                "gpu/nvml_memory_percent": (
                    100.0 * float(memory.used) / float(memory.total)
                    if memory.total
                    else 0.0
                ),
                "gpu/temp_c": float(
                    module.nvmlDeviceGetTemperature(
                        handle,
                        module.NVML_TEMPERATURE_GPU,
                    )
                ),
                "gpu/power_w": float(module.nvmlDeviceGetPowerUsage(handle)) / 1000.0,
            })
            try:
                metrics["gpu/fan_percent"] = float(module.nvmlDeviceGetFanSpeed(handle))
            except Exception:
                pass
        except Exception:
            return


class BaseTelemetry:
    def start(self) -> None:
        raise NotImplementedError

    def log_event(self, event: str, payload: Mapping[str, Any]) -> None:
        raise NotImplementedError

    def add_initial_summary(self, parameter_report: Mapping[str, Any]) -> None:
        raise NotImplementedError

    def add_final_result(self, result: Mapping[str, Any]) -> None:
        raise NotImplementedError

    def finish(self) -> None:
        raise NotImplementedError


class NullTelemetry(BaseTelemetry):
    def start(self) -> None:
        return

    def log_event(self, event: str, payload: Mapping[str, Any]) -> None:
        return

    def add_initial_summary(self, parameter_report: Mapping[str, Any]) -> None:
        return

    def add_final_result(self, result: Mapping[str, Any]) -> None:
        return

    def finish(self) -> None:
        return


class WandbTelemetry(BaseTelemetry):
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
        system_sampler: Optional[SystemMetricSampler] = None,
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
        self.system_sampler = system_sampler
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
        define_metric = run.define_metric if hasattr(run, "define_metric") else module.define_metric
        define_metric("optimizer/update")
        for metric in (
            "tokens/*",
            "time/*",
            "train/*",
            "val/*",
            "optim/*",
            "perf/*",
            "model/*",
            "resource/*",
            "system/*",
            "process/*",
            "gpu/*",
            "sheet/*",
        ):
            define_metric(metric, step_metric="optimizer/update")
        self.module = module
        self.run = run

    def log_event(self, event: str, payload: Mapping[str, Any]) -> None:
        if self.run is None:
            return
        metrics = metrics_for_event(event, payload)
        if not metrics:
            return
        step = int(metrics["optimizer/update"])
        if self.system_sampler is not None:
            metrics.update(self.system_sampler.sample(step))
        self.run.log(_scalar_metrics(metrics))

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
        metrics = final_result_metrics(result)
        step = int(metrics["optimizer/update"])
        if self.system_sampler is not None:
            metrics.update(self.system_sampler.sample(step, force=True))
        self.run.log(_scalar_metrics(metrics))
        evaluations = result.get("evaluations", [])
        final_validation_loss = evaluations[-1]["val"] if evaluations else None
        best_validation_loss = (
            min(row["val"] for row in evaluations) if evaluations else None
        )
        self.run.summary.update({
            "completed_updates": step,
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


class TensorboardTelemetry(BaseTelemetry):
    """One rank-zero TensorBoard SummaryWriter owner for local THOG2 curves."""

    def __init__(
        self,
        *,
        enabled: bool,
        root: Path,
        name: str,
        group: str,
        job_type: str,
        config: Mapping[str, Any],
        system_sampler: Optional[SystemMetricSampler] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.root = Path(root)
        self.name = name
        self.group = group
        self.job_type = job_type
        self.config = dict(config)
        self.system_sampler = system_sampler
        self.writer: Optional[Any] = None

    def start(self) -> None:
        if not self.enabled or self.writer is not None:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        module = importlib.import_module("torch.utils.tensorboard")
        self.writer = module.SummaryWriter(log_dir=str(self.root))
        self.writer.add_text("run/artifact_name", self.name, 0)
        self.writer.add_text("run/group", self.group, 0)
        self.writer.add_text("run/job_type", self.job_type, 0)
        self.writer.add_text(
            "run/config_json",
            json.dumps(self.config, indent=2, sort_keys=True),
            0,
        )

    def log_event(self, event: str, payload: Mapping[str, Any]) -> None:
        if self.writer is None:
            return
        metrics = metrics_for_event(event, payload)
        if not metrics:
            return
        step = int(metrics["optimizer/update"])
        if self.system_sampler is not None:
            metrics.update(self.system_sampler.sample(step))
        self._write_scalars(metrics, step)

    def add_initial_summary(self, parameter_report: Mapping[str, Any]) -> None:
        if self.writer is None:
            return
        self.writer.add_text(
            "run/parameter_report_json",
            json.dumps(dict(parameter_report), indent=2, sort_keys=True),
            0,
        )
        self._write_scalars({
            "model/persistent_parameters": int(parameter_report["persistent_parameters"]),
            "model/dense_equivalent_parameters": int(
                parameter_report["dense_equivalent_total_parameters"]
            ),
        }, 0)

    def add_final_result(self, result: Mapping[str, Any]) -> None:
        if self.writer is None:
            return
        metrics = final_result_metrics(result)
        step = int(metrics["optimizer/update"])
        if self.system_sampler is not None:
            metrics.update(self.system_sampler.sample(step, force=True))
        self._write_scalars(metrics, step)
        self.writer.flush()

    def finish(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

    def _write_scalars(self, metrics: Mapping[str, Any], step: int) -> None:
        if self.writer is None:
            return
        for name, value in _scalar_metrics(metrics).items():
            self.writer.add_scalar(name, value, step)


def metrics_for_event(event: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    if event == "optimizer_progress":
        return training_metrics(payload)
    if event == "evaluation_completed":
        return evaluation_metrics(payload)
    return {}


def build_telemetry(
    *,
    instrumentation: str,
    enabled: bool,
    curve_dir: Path,
    wandb_project: str,
    wandb_entity: Optional[str],
    wandb_mode: str,
    name: str,
    group: str,
    job_type: str,
    config: Mapping[str, Any],
    system_log_interval: int,
    device: str,
    disk_root: Path,
) -> BaseTelemetry:
    if instrumentation not in INSTRUMENTATION_BACKENDS:
        raise ValueError(f"instrumentation must be one of {INSTRUMENTATION_BACKENDS}")
    if instrumentation == "none":
        return NullTelemetry()
    sampler = SystemMetricSampler(
        enabled=enabled,
        interval=system_log_interval,
        device=device,
        disk_root=disk_root,
    )
    if instrumentation == "wandb":
        return WandbTelemetry(
            enabled=enabled,
            project=wandb_project,
            entity=wandb_entity,
            mode=wandb_mode,
            root=curve_dir,
            name=name,
            group=group,
            job_type=job_type,
            config=config,
            system_sampler=sampler,
        )
    return TensorboardTelemetry(
        enabled=enabled,
        root=curve_dir,
        name=name,
        group=group,
        job_type=job_type,
        config=config,
        system_sampler=sampler,
    )


def attach_telemetry(trainer: Any, telemetry: BaseTelemetry) -> None:
    """Attach event logging without moving telemetry into clean train/eval timers."""

    original_progress = trainer._print_progress

    def progress(run_id: str, event: str, **payload: Any) -> None:
        original_progress(run_id, event, **payload)
        if trainer.distributed.is_primary:
            telemetry_payload = dict(payload)
            multiplier = 1
            if hasattr(trainer, "telemetry_token_multiplier"):
                multiplier = int(trainer.telemetry_token_multiplier)
            if "consumed_tokens" in telemetry_payload:
                telemetry_payload["consumed_tokens"] = (
                    int(telemetry_payload["consumed_tokens"]) * multiplier
                )
            telemetry.log_event(event, telemetry_payload)

    trainer._print_progress = progress


__all__ = [
    "BaseTelemetry",
    "INSTRUMENTATION_BACKENDS",
    "NullTelemetry",
    "SystemMetricSampler",
    "TensorboardTelemetry",
    "WandbTelemetry",
    "attach_telemetry",
    "build_telemetry",
    "evaluation_metrics",
    "final_result_metrics",
    "metrics_for_event",
    "training_metrics",
]
# ^^^ THOG
