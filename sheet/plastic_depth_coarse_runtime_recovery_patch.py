# vvv THOG
"""Hard COARSE/FINE CUDA teardown, visible failed-trial causes, and phase-safe telemetry."""

from __future__ import annotations

import gc
import os
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from . import plastic_depth_coarse as _coarse
from . import plastic_depth_fresh_state as _fresh_state
from . import plastic_depth_lifecycle as _lifecycle
from . import wandb_telemetry as _telemetry


_BYTES_PER_GIB = float(1024**3)
_CUDA_LEAK_TOLERANCE_BYTES = 64 * 1024 * 1024
_CUDA_BASELINES: Dict[str, Tuple[int, int]] = {}
_RETAINED_CONTROLLER_ATTRIBUTE = "_update_retained_materializations_controller"
_ORIGINAL_BUILD_FRESH_TRAINING_STATE = _fresh_state.build_fresh_training_state
_ORIGINAL_RENDER_PLASTIC_COARSE_REPORT = _coarse.render_plastic_coarse_report
_ORIGINAL_LOG_PLASTIC_COARSE_FINE = _telemetry.WandbTelemetry.log_plastic_coarse_fine
_ORIGINAL_LOG_SCALARS = _telemetry.WandbTelemetry._log_scalars


def _cuda_device(device: Any) -> Optional[torch.device]:
    if not torch.cuda.is_available():
        return None
    try:
        resolved = torch.device(device)
    except (TypeError, RuntimeError):
        return None
    if resolved.type != "cuda":
        return None
    if resolved.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return resolved


def _cuda_snapshot(device: Any) -> Optional[Tuple[int, int]]:
    resolved = _cuda_device(device)
    if resolved is None:
        return None
    try:
        torch.cuda.synchronize(resolved)
    except RuntimeError:
        pass
    return (
        int(torch.cuda.memory_allocated(resolved)),
        int(torch.cuda.memory_reserved(resolved)),
    )


def _cuda_key(device: Any) -> Optional[str]:
    resolved = _cuda_device(device)
    return None if resolved is None else str(resolved)


def _build_fresh_training_state_with_cuda_baseline(*args: Any, **kwargs: Any):
    device = getattr(kwargs.get("resolved_config"), "device", "cpu")
    baseline = _cuda_snapshot(device)
    state = _ORIGINAL_BUILD_FRESH_TRAINING_STATE(*args, **kwargs)
    if baseline is not None:
        key = _cuda_key(device)
        if key is not None:
            _CUDA_BASELINES[key] = baseline
        state.thog_cuda_baseline_allocated_bytes = int(baseline[0])
        state.thog_cuda_baseline_reserved_bytes = int(baseline[1])
    return state


def _call_cleanup_method(target: Any, name: str, errors: list[str]) -> None:
    method = getattr(target, name, None)
    if not callable(method):
        return
    try:
        method()
    except BaseException as error:
        errors.append(f"{name}: {type(error).__name__}: {error}")


def _detach_retained_materialisation_controller(raw_model: Any, errors: list[str]) -> None:
    trajectory = getattr(raw_model, "trajectory", None)
    controller = getattr(raw_model, "_update_retained_materializations", None)
    if controller is None:
        return
    try:
        end = getattr(controller, "end", None)
        if callable(end):
            end()
        retained = getattr(controller, "_retained", None)
        if hasattr(retained, "clear"):
            retained.clear()
        if trajectory is not None:
            if getattr(trajectory, _RETAINED_CONTROLLER_ATTRIBUTE, None) is controller:
                delattr(trajectory, _RETAINED_CONTROLLER_ATTRIBUTE)
            for method_name in ("materialize", "materialize_layer_matrices"):
                trajectory.__dict__.pop(method_name, None)
        if hasattr(controller, "_trajectory"):
            controller._trajectory = None
        raw_model._update_retained_materializations = None
    except BaseException as error:
        errors.append(
            "detach retained materialisations: "
            f"{type(error).__name__}: {error}"
        )


def _hard_destroy_fresh_training_state(state: Any) -> None:
    trainer = getattr(state, "trainer", None)
    if trainer is None:
        return
    device = getattr(trainer, "device", "cpu")
    before = _cuda_snapshot(device)
    key = _cuda_key(device)
    baseline = (
        int(getattr(state, "thog_cuda_baseline_allocated_bytes")),
        int(getattr(state, "thog_cuda_baseline_reserved_bytes")),
    ) if hasattr(state, "thog_cuda_baseline_allocated_bytes") else _CUDA_BASELINES.get(key or "")
    cleanup_errors: list[str] = []

    raw_model = getattr(trainer, "raw_model", None)
    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is not None:
        try:
            optimizer.zero_grad(set_to_none=True)
            optimizer.state.clear()
            optimizer.param_groups.clear()
        except BaseException as error:
            cleanup_errors.append(
                f"optimizer release: {type(error).__name__}: {error}"
            )
    if raw_model is not None:
        _call_cleanup_method(raw_model, "end_optimizer_update", cleanup_errors)
        _call_cleanup_method(raw_model, "clear_plastic_depth_update_layer_count", cleanup_errors)
        trajectory = getattr(raw_model, "trajectory", None)
        if trajectory is not None:
            _call_cleanup_method(trajectory, "clear_plastic_depth_basis_cache", cleanup_errors)
        try:
            raw_model.zero_grad(set_to_none=True)
        except BaseException as error:
            cleanup_errors.append(f"model.zero_grad: {type(error).__name__}: {error}")
        _detach_retained_materialisation_controller(raw_model, cleanup_errors)
    _call_cleanup_method(trainer, "_clear_plastic_depth_inline_update", cleanup_errors)

    try:
        trainer.close()
    except BaseException as error:
        cleanup_errors.append(f"trainer.close: {type(error).__name__}: {error}")

    state.trainer = None
    for attribute in (
        "_print_progress",
        "model",
        "raw_model",
        "optimizer",
        "scaler",
        "batch_source",
        "memory_telemetry",
        "events",
        "parameter_report",
        "distributed",
    ):
        if hasattr(trainer, attribute):
            try:
                setattr(trainer, attribute, None)
            except BaseException as error:
                cleanup_errors.append(
                    f"clear {attribute}: {type(error).__name__}: {error}"
                )
    del optimizer
    del raw_model
    del trainer

    gc.collect()
    gc.collect()
    resolved_device = _cuda_device(device)
    if resolved_device is not None:
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize(resolved_device)
        except RuntimeError as error:
            cleanup_errors.append(f"CUDA cache release: {type(error).__name__}: {error}")
    gc.collect()
    after = _cuda_snapshot(device)

    if before is not None and after is not None and int(os.environ.get("RANK", "0")) == 0:
        baseline_allocated = 0 if baseline is None else baseline[0]
        print(
            "PLASTIC CUDA cleanup "
            f"[{getattr(state, 'phase', '?')} layers={getattr(state, 'active_layer_count', '?')}]: "
            f"allocated {before[0] / _BYTES_PER_GIB:.2f} -> {after[0] / _BYTES_PER_GIB:.2f} GiB; "
            f"reserved {before[1] / _BYTES_PER_GIB:.2f} -> {after[1] / _BYTES_PER_GIB:.2f} GiB; "
            f"baseline allocated={baseline_allocated / _BYTES_PER_GIB:.2f} GiB",
            flush=True,
        )

    if baseline is not None and after is not None:
        leaked_allocated = after[0] - baseline[0]
        if leaked_allocated > _CUDA_LEAK_TOLERANCE_BYTES:
            cleanup_errors.append(
                "CUDA allocated memory did not return to the pre-state baseline: "
                f"baseline={baseline[0] / _BYTES_PER_GIB:.3f} GiB, "
                f"after={after[0] / _BYTES_PER_GIB:.3f} GiB, "
                f"delta={leaked_allocated / _BYTES_PER_GIB:.3f} GiB"
            )
    if cleanup_errors:
        raise RuntimeError("PLASTIC fresh-state teardown failed: " + "; ".join(cleanup_errors))


def _failure_message(error_message: Any) -> str:
    text = " ".join(str(error_message or "no error message recorded").split())
    return text if len(text) <= 500 else text[:497] + "..."


def _render_plastic_coarse_report_with_failures(*args: Any, **kwargs: Any) -> str:
    report = _ORIGINAL_RENDER_PLASTIC_COARSE_REPORT(*args, **kwargs)
    scored_trials = args[0] if args else kwargs.get("scored_trials", ())
    failures = [row.result for row in scored_trials if row.result.status == "failed"]
    if not failures:
        return report
    lines = [report, "", "PLASTIC COARSE FAILURES"]
    requested_steps = int(kwargs.get("training_steps", 0))
    for result in failures:
        lines.append(
            f"  trial {result.trial_index} layers={result.layers} "
            f"completed={result.training_steps}/{requested_steps} "
            f"{result.error_class or 'Exception'}: {_failure_message(result.error_message)}"
        )
    return "\n".join(lines)


def _log_scalars_without_rewinding_wandb(self: Any, metrics: Mapping[str, Any], step: int) -> None:
    if not getattr(self, "_thog_plastic_coarse_logged", False):
        _ORIGINAL_LOG_SCALARS(self, metrics, step)
        return
    scalars = _telemetry._scalar_metrics(metrics)
    if self.run is not None:
        self.run.log(scalars)
    if self.writer is not None:
        for name, value in scalars.items():
            self.writer.add_scalar(name, value, step)


def _log_plastic_coarse_fine_phase_safe(self: Any, provenance: Mapping[str, Any]) -> None:
    self._thog_plastic_coarse_logged = True
    _ORIGINAL_LOG_PLASTIC_COARSE_FINE(self, provenance)
    for trial in provenance.get("trials", ()):
        trial_index = int(trial["trial_index"])
        status = str(trial.get("status", "unknown"))
        if self.run is not None:
            summary = getattr(self.run, "summary", None)
            if hasattr(summary, "update"):
                summary.update(
                    {
                        f"coarse/trial_{trial_index}/status": status,
                        f"coarse/trial_{trial_index}/completed_steps": int(
                            trial.get("training_steps", 0)
                        ),
                        f"coarse/trial_{trial_index}/error_class": trial.get("error_class"),
                        f"coarse/trial_{trial_index}/error_message": trial.get("error_message"),
                    }
                )
        if self.writer is not None:
            add_text = getattr(self.writer, "add_text", None)
            if callable(add_text):
                add_text(
                    f"coarse/trial_{trial_index}/status",
                    status,
                    0,
                )
                if trial.get("error_class") or trial.get("error_message"):
                    add_text(
                        f"coarse/trial_{trial_index}/failure",
                        f"{trial.get('error_class')}: {_failure_message(trial.get('error_message'))}",
                        0,
                    )


_fresh_state.build_fresh_training_state = _build_fresh_training_state_with_cuda_baseline
_fresh_state.destroy_fresh_training_state = _hard_destroy_fresh_training_state
_lifecycle.build_fresh_training_state = _build_fresh_training_state_with_cuda_baseline
_lifecycle.destroy_fresh_training_state = _hard_destroy_fresh_training_state
_lifecycle.render_plastic_coarse_report = _render_plastic_coarse_report_with_failures
_coarse.render_plastic_coarse_report = _render_plastic_coarse_report_with_failures

_lifecycle_defaults = dict(
    _lifecycle.run_plastic_coarse_fine_lifecycle.__kwdefaults__ or {}
)
_lifecycle_defaults["fresh_state_builder"] = _build_fresh_training_state_with_cuda_baseline
_lifecycle_defaults["state_destroyer"] = _hard_destroy_fresh_training_state
_lifecycle.run_plastic_coarse_fine_lifecycle.__kwdefaults__ = _lifecycle_defaults

_telemetry.WandbTelemetry._log_scalars = _log_scalars_without_rewinding_wandb
_telemetry.WandbTelemetry.log_plastic_coarse_fine = _log_plastic_coarse_fine_phase_safe
# ^^^ THOG
