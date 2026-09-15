# vvv THOG
"""Low-perturbation whole-update timing for PREMAT performance attribution.

This deliberately stays separate from Nsight Processing capture.  Set
THOG2_PROCESSING_UPDATE_TIMING_UPDATE to one positive optimizer-update number.
The selected update is timed with host perf_counter_ns boundaries plus CUDA
Events on the current MAIN stream.  One device-wide completion drain occurs
after the logical optimizer update returns, at the same boundary already owned
by Stage6Trainer._timed().
"""

from __future__ import annotations

import json
import os
import time
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from . import wandb_telemetry as _wandb
from .local_chart_store import ensure_local_chart_store


_TIMING_UPDATE_ENV = "THOG2_PROCESSING_UPDATE_TIMING_UPDATE"
_TIMING_SCHEMA_VERSION = 2
_TIMING_FILENAME = "update_timing.json"


def _requested_update() -> Optional[int]:
    raw = os.environ.get(_TIMING_UPDATE_ENV, "").strip()
    if not raw:
        return None
    try:
        update = int(raw)
    except ValueError as error:
        raise ValueError(f"{_TIMING_UPDATE_ENV} must be a positive integer; got {raw!r}") from error
    if update < 1:
        raise ValueError(f"{_TIMING_UPDATE_ENV} must be a positive integer; got {update}")
    return update


_ACCOUNTING_PHASES = (
    "setup",
    "forward",
    "backward",
    "post_backward",
    "optimizer",
    "update_cleanup",
    "gpu_completion_drain",
    "unexplained_residual",
)


def _premat_is_enabled(value: Any) -> bool:
    return str(value).strip().lower() == "enabled"


def _gap_phase(previous: Optional[Dict[str, Any]], following: Optional[Dict[str, Any]]) -> str:
    if following is not None and str(following["phase"]) == "forward":
        return "setup"
    if (
        previous is not None
        and following is not None
        and str(previous["phase"]) == "forward"
        and str(following["phase"]) == "backward"
        and previous.get("micro_step") == following.get("micro_step")
    ):
        return "forward"
    if following is not None and str(following["phase"]) == "optimizer":
        return "post_backward"
    if previous is not None and str(previous["phase"]) == "optimizer":
        return "update_cleanup"
    if previous is not None and str(previous["phase"]) == "backward" and following is None:
        return "post_backward"
    return "setup"


def _apply_official_elapsed(payload: Dict[str, Any], official_update_ms: float) -> Dict[str, Any]:
    official_ms = float(official_update_ms)
    captured_ms = float(payload["captured_update_host_ms"])
    unexplained_ms = official_ms - captured_ms
    if unexplained_ms < 0.0 and abs(unexplained_ms) < 0.001:
        unexplained_ms = 0.0
    if unexplained_ms < 0.0:
        raise RuntimeError(
            "captured complete-update span exceeds the official synchronized update time: "
            f"captured={captured_ms:.6f} ms official={official_ms:.6f} ms"
        )

    timeline = list(payload["timeline"])
    timeline.append(
        {
            "phase": "unexplained_residual",
            "micro_step": None,
            "host_start_ms": captured_ms,
            "host_end_ms": official_ms,
            "host_duration_ms": unexplained_ms,
            "cuda_main_ms": None,
        }
    )
    totals = dict(payload["phase_totals_host_ms"])
    totals["unexplained_residual"] = unexplained_ms
    partition_sum_ms = sum(float(totals[phase]) for phase in _ACCOUNTING_PHASES)

    payload["timeline"] = timeline
    payload["phase_totals_host_ms"] = totals
    payload["official_update_ms"] = official_ms
    payload["host_update_ms"] = official_ms
    payload["unexplained_residual_ms"] = unexplained_ms
    payload["host_partition_sum_ms"] = partition_sum_ms
    payload["host_partition_residual_ms"] = official_ms - partition_sum_ms
    return payload


class _UpdateTimingRecorder:
    def __init__(self, trainer: Any, optimizer_update: int, *, run_label: str) -> None:
        self.trainer = trainer
        self.optimizer_update = int(optimizer_update)
        self.run_label = str(run_label).strip()
        self.device = torch.device(trainer.device)
        if self.device.type != "cuda":
            raise RuntimeError("processing update timing requires CUDA")
        self.host_update_start_ns = time.perf_counter_ns()
        self.cuda_update_start = self._record_cuda_event()
        self.phases: list[Dict[str, Any]] = []
        self.forward_count = 0
        self.backward_count = 0
        self.optimizer_count = 0
        self.cuda_update_end: Optional[torch.cuda.Event] = None
        self.drain_started_ns: Optional[int] = None
        self.drain_finished_ns: Optional[int] = None

    def _record_cuda_event(self) -> torch.cuda.Event:
        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream(self.device))
        return event

    def begin_phase(self, phase: str, *, micro_step: Optional[int] = None) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "phase": str(phase),
            "micro_step": None if micro_step is None else int(micro_step),
            "host_start_ns": time.perf_counter_ns(),
            "cuda_start": self._record_cuda_event(),
        }
        self.phases.append(record)
        return record

    def end_phase(self, record: Dict[str, Any]) -> None:
        record["cuda_end"] = self._record_cuda_event()
        record["host_end_ns"] = time.perf_counter_ns()

    @staticmethod
    def _elapsed_cuda_ms(record: Dict[str, Any]) -> Optional[float]:
        start = record.get("cuda_start")
        end = record.get("cuda_end")
        if start is None or end is None:
            return None
        return float(start.elapsed_time(end))

    def complete_gpu_work(self) -> None:
        self.cuda_update_end = self._record_cuda_event()
        self.drain_started_ns = time.perf_counter_ns()
        torch.cuda.synchronize(self.device)
        self.drain_finished_ns = time.perf_counter_ns()

    def finish(self) -> Dict[str, Any]:
        if self.cuda_update_end is None or self.drain_started_ns is None or self.drain_finished_ns is None:
            raise RuntimeError("complete-update GPU work was not drained before timing finalization")
        gpu_completion_drain_ms = (self.drain_finished_ns - self.drain_started_ns) / 1_000_000.0

        completed = []
        for record in self.phases:
            if "host_end_ns" not in record:
                continue
            host_start_ms = (int(record["host_start_ns"]) - self.host_update_start_ns) / 1_000_000.0
            host_end_ms = (int(record["host_end_ns"]) - self.host_update_start_ns) / 1_000_000.0
            completed.append(
                {
                    "phase": str(record["phase"]),
                    "micro_step": record.get("micro_step"),
                    "host_start_ms": host_start_ms,
                    "host_end_ms": host_end_ms,
                    "host_duration_ms": host_end_ms - host_start_ms,
                    "cuda_main_ms": self._elapsed_cuda_ms(record),
                }
            )
        completed.sort(key=lambda row: (float(row["host_start_ms"]), float(row["host_end_ms"])))

        timeline = []
        cursor_ms = 0.0
        overlap_detected = False
        previous: Optional[Dict[str, Any]] = None
        for row in completed:
            start_ms = float(row["host_start_ms"])
            end_ms = float(row["host_end_ms"])
            if start_ms > cursor_ms:
                timeline.append(
                    {
                        "phase": _gap_phase(previous, row),
                        "micro_step": row.get("micro_step"),
                        "host_start_ms": cursor_ms,
                        "host_end_ms": start_ms,
                        "host_duration_ms": start_ms - cursor_ms,
                        "cuda_main_ms": None,
                    }
                )
            elif start_ms < cursor_ms:
                overlap_detected = True
            timeline.append(row)
            cursor_ms = max(cursor_ms, end_ms)
            previous = row

        drain_start_ms = (self.drain_started_ns - self.host_update_start_ns) / 1_000_000.0
        if cursor_ms < drain_start_ms:
            timeline.append(
                {
                    "phase": _gap_phase(previous, None),
                    "micro_step": None,
                    "host_start_ms": cursor_ms,
                    "host_end_ms": drain_start_ms,
                    "host_duration_ms": drain_start_ms - cursor_ms,
                    "cuda_main_ms": None,
                }
            )
        drain_end_ms = (self.drain_finished_ns - self.host_update_start_ns) / 1_000_000.0
        timeline.append(
            {
                "phase": "gpu_completion_drain",
                "micro_step": None,
                "host_start_ms": drain_start_ms,
                "host_end_ms": drain_end_ms,
                "host_duration_ms": gpu_completion_drain_ms,
                "cuda_main_ms": None,
            }
        )

        totals: Dict[str, float] = {phase: 0.0 for phase in _ACCOUNTING_PHASES}
        for row in timeline:
            phase = str(row["phase"])
            totals[phase] = totals.get(phase, 0.0) + float(row["host_duration_ms"])

        microsteps = []
        accumulation_steps = int(self.trainer.config.gradient_accumulation_steps)
        for micro_step in range(1, accumulation_steps + 1):
            forward = next(
                (row for row in completed if row["phase"] == "forward" and row.get("micro_step") == micro_step),
                None,
            )
            backward = next(
                (row for row in completed if row["phase"] == "backward" and row.get("micro_step") == micro_step),
                None,
            )
            if forward is None and backward is None:
                continue
            starts = [float(row["host_start_ms"]) for row in (forward, backward) if row is not None]
            ends = [float(row["host_end_ms"]) for row in (forward, backward) if row is not None]
            microsteps.append(
                {
                    "micro_step": micro_step,
                    "host_span_ms": max(ends) - min(starts),
                    "forward_host_ms": None if forward is None else float(forward["host_duration_ms"]),
                    "backward_host_ms": None if backward is None else float(backward["host_duration_ms"]),
                    "forward_cuda_main_ms": None if forward is None else forward.get("cuda_main_ms"),
                    "backward_cuda_main_ms": None if backward is None else backward.get("cuda_main_ms"),
                }
            )

        config = self.trainer.config
        premat_enabled = _premat_is_enabled(config.premat)
        return {
            "schema_version": _TIMING_SCHEMA_VERSION,
            "optimizer_update": self.optimizer_update,
            "official_update_ms": None,
            "host_update_ms": None,
            "captured_update_host_ms": drain_end_ms,
            "cuda_main_update_ms": float(self.cuda_update_start.elapsed_time(self.cuda_update_end)),
            "gpu_completion_drain_ms": gpu_completion_drain_ms,
            "harvest_wait_ms": gpu_completion_drain_ms,
            "host_partition_sum_ms": None,
            "host_partition_residual_ms": None,
            "host_phase_overlap_detected": overlap_detected,
            "phase_totals_host_ms": totals,
            "timeline": timeline,
            "microsteps": microsteps,
            "capture": {
                "gradient_accumulation_steps": accumulation_steps,
                "batch_size": int(config.batch_size),
                "block_size": int(config.block_size),
                "run_label": self.run_label,
                "premat": premat_enabled,
                "premat_mode": str(config.premat),
                "premat_target_layer": config.premat_target_layer,
                "premat_target_matrix": config.premat_target_matrix,
                "nsight_processing_logging": str(config.premat_processing_logging),
                "cuda_owner": "MAIN",
                "match_signature": {
                    "optimizer_update": self.optimizer_update,
                    "model_type": str(config.model_type),
                    "geometry_preset": config.geometry_preset,
                    "n_layer": int(config.n_layer),
                    "n_head": int(config.n_head),
                    "n_embd": int(config.n_embd),
                    "dropout": float(config.dropout),
                    "bias": bool(config.bias),
                    "depth_order": int(config.depth_order),
                    "block_size": int(config.block_size),
                    "batch_size": int(config.batch_size),
                    "gradient_accumulation_steps": accumulation_steps,
                    "checkpoint_segment_size": int(config.checkpoint_segment_size),
                    "layer_dropout_stratum_size": config.layer_dropout_stratum_size,
                    "layer_dropout_active_per_stratum": config.layer_dropout_active_per_stratum,
                    "layer_dropout_resample_steps": int(config.layer_dropout_resample_steps),
                    "dtype": str(config.dtype),
                    "optimizer_class": type(self.trainer.optimizer).__name__,
                    "learning_rate": float(config.learning_rate),
                    "min_learning_rate": float(config.min_learning_rate),
                    "warmup_updates": int(config.warmup_updates),
                    "decay_updates": int(config.decay_updates),
                    "decay_learning_rate": bool(config.decay_learning_rate),
                    "weight_decay": float(config.weight_decay),
                    "beta1": float(config.beta1),
                    "beta2": float(config.beta2),
                    "grad_clip": float(config.grad_clip),
                    "model_seed": int(config.model_seed),
                    "data_seed": int(config.data_seed),
                    "premat_attention_mode": str(config.premat_attention_mode),
                    "world_size": int(self.trainer.distributed.world_size),
                },
            },
        }


def _write_payload(store: Any, payload: Dict[str, Any]) -> Path:
    processing_directory = Path(store.path).parent / "processing"
    processing_directory.mkdir(parents=True, exist_ok=True)
    destination = processing_directory / _TIMING_FILENAME
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(destination)
    return destination


def _capture_train_one_update(
    trainer: Any,
    original_train_one_update: Any,
    optimizer_update: int,
    *,
    run_label: str,
    capture_state: Dict[str, Any],
) -> Dict[str, Any]:
    recorder = _UpdateTimingRecorder(trainer, optimizer_update, run_label=run_label)
    forward_open: list[Dict[str, Any]] = []
    backward_depth = 0
    backward_open: Optional[Dict[str, Any]] = None
    optimizer_open: list[Dict[str, Any]] = []

    def forward_pre_hook(_module: Any, _args: Any) -> None:
        recorder.forward_count += 1
        forward_open.append(recorder.begin_phase("forward", micro_step=recorder.forward_count))

    def forward_post_hook(_module: Any, _args: Any, _output: Any) -> None:
        if forward_open:
            recorder.end_phase(forward_open.pop())

    original_backward = torch.autograd.backward

    def timed_backward(*args: Any, **kwargs: Any) -> Any:
        nonlocal backward_depth, backward_open
        outermost = backward_depth == 0
        if outermost:
            recorder.backward_count += 1
            backward_open = recorder.begin_phase("backward", micro_step=recorder.backward_count)
        backward_depth += 1
        try:
            return original_backward(*args, **kwargs)
        finally:
            backward_depth -= 1
            if outermost and backward_open is not None:
                recorder.end_phase(backward_open)
                backward_open = None

    def optimizer_pre_hook(_optimizer: Any, _args: Any, _kwargs: Any) -> None:
        recorder.optimizer_count += 1
        optimizer_open.append(recorder.begin_phase("optimizer"))

    def optimizer_post_hook(_optimizer: Any, _args: Any, _kwargs: Any) -> None:
        if optimizer_open:
            recorder.end_phase(optimizer_open.pop())

    handles = [
        trainer.model.register_forward_pre_hook(forward_pre_hook),
        trainer.model.register_forward_hook(forward_post_hook),
    ]
    if (
        hasattr(trainer.optimizer, "register_step_pre_hook")
        and hasattr(trainer.optimizer, "register_step_post_hook")
    ):
        handles.append(trainer.optimizer.register_step_pre_hook(optimizer_pre_hook))
        handles.append(trainer.optimizer.register_step_post_hook(optimizer_post_hook))

    torch.autograd.backward = timed_backward
    result: Optional[Dict[str, Any]] = None
    caught: Optional[BaseException] = None
    try:
        result = original_train_one_update()
    except BaseException as error:
        caught = error
    finally:
        torch.autograd.backward = original_backward
        for handle in handles:
            handle.remove()

    try:
        recorder.complete_gpu_work()
        capture_state["recorder"] = recorder
        capture_state["completed_updates_after"] = int(trainer.state.completed_updates)
        capture_state["successful_update"] = bool(
            result is not None
            and not bool(float(result.get("skipped_update", 0.0)))
            and int(trainer.state.completed_updates) >= int(optimizer_update)
        )
        capture_state["exception"] = None if caught is None else type(caught).__name__
    except Exception as timing_error:
        print(
            f"THOG2 WARNING: Processing update timing export failed; continuing training: {timing_error}",
            flush=True,
        )

    if caught is not None:
        raise caught
    if result is None:
        raise RuntimeError("processing update timing wrapper completed without a trainer result")
    return result


def _publish_payload(
    trainer: Any,
    store: Any,
    payload: Dict[str, Any],
    *,
    official_elapsed_seconds: float,
) -> None:
    payload = _apply_official_elapsed(payload, float(official_elapsed_seconds) * 1000.0)
    destination = _write_payload(store, payload)
    world_size = max(1, int(trainer.distributed.world_size))
    tokens_per_update = (
        int(trainer.config.batch_size)
        * int(trainer.config.block_size)
        * int(trainer.config.gradient_accumulation_steps)
        * world_size
    )
    official_update_seconds = float(payload["official_update_ms"]) / 1000.0
    if official_update_seconds > 0.0:
        store.append_processing_throughput(
            int(payload["optimizer_update"]),
            float(tokens_per_update) / official_update_seconds,
        )
    store.heartbeat(int(trainer.state.completed_updates), run_state="recording", force=True)
    print(
        "THOG2 Processing complete-update timing: "
        f"update={payload['optimizer_update']} official={payload['official_update_ms']:.3f} ms "
        f"MAIN-stream-elapsed={payload['cuda_main_update_ms']:.3f} ms "
        f"GPU-completion-drain={payload['gpu_completion_drain_ms']:.3f} ms "
        f"unexplained={payload['unexplained_residual_ms']:.3f} ms "
        f"file={destination}",
        flush=True,
    )


_ORIGINAL_ATTACH_TELEMETRY = _wandb.attach_telemetry


def _attach_telemetry_with_processing_update_timing(trainer: Any, telemetry: Any) -> None:
    requested_update = _requested_update()
    if requested_update is None:
        _ORIGINAL_ATTACH_TELEMETRY(trainer, telemetry)
        return
    if requested_update > int(trainer.config.max_updates):
        raise ValueError(
            f"{_TIMING_UPDATE_ENV}={requested_update} exceeds max_updates={trainer.config.max_updates}"
        )

    original_train_one_update = trainer.train_one_update
    store_holder: Dict[str, Any] = {}
    capture_state: Dict[str, Any] = {}

    @wraps(original_train_one_update)
    def timed_train_one_update() -> Dict[str, Any]:
        prospective_update = int(trainer.state.completed_updates) + 1
        if prospective_update != requested_update:
            return original_train_one_update()
        if "store" not in store_holder:
            raise RuntimeError("processing update timing store was not attached before training")
        return _capture_train_one_update(
            trainer,
            original_train_one_update,
            requested_update,
            run_label=str(telemetry.group),
            capture_state=capture_state,
        )

    trainer.train_one_update = timed_train_one_update
    _ORIGINAL_ATTACH_TELEMETRY(trainer, telemetry)
    store_holder["store"] = ensure_local_chart_store(telemetry)
    original_timed = trainer._timed

    @wraps(original_timed)
    def timed_complete_update(function: Any):
        result, elapsed = original_timed(function)
        recorder = capture_state.pop("recorder", None)
        if function == timed_train_one_update and recorder is not None:
            try:
                payload = recorder.finish()
                payload["completed_updates_after"] = capture_state.pop("completed_updates_after")
                payload["successful_update"] = capture_state.pop("successful_update")
                payload["exception"] = capture_state.pop("exception")
                _publish_payload(
                    trainer,
                    store_holder["store"],
                    payload,
                    official_elapsed_seconds=float(elapsed),
                )
            except Exception as timing_error:
                print(
                    f"THOG2 WARNING: Processing complete-update timing export failed; "
                    f"continuing training: {timing_error}",
                    flush=True,
                )
        return result, elapsed

    trainer._timed = timed_complete_update
    if str(trainer.config.premat_processing_logging) == "enabled":
        print(
            "THOG2 WARNING: whole-update timing is running in the Nsight Processing process; "
            "for an uncontaminated non-Nsight capture, disable --premat_processing_logging.",
            flush=True,
        )


_wandb.attach_telemetry = _attach_telemetry_with_processing_update_timing


__all__ = [
    "_apply_official_elapsed",
    "_attach_telemetry_with_processing_update_timing",
    "_gap_phase",
    "_premat_is_enabled",
]
# ^^^ THOG
