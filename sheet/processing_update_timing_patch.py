# vvv THOG
"""Low-perturbation whole-update timing for PREMAT performance attribution.

This deliberately stays separate from Nsight Processing capture.  Set
THOG2_PROCESSING_UPDATE_TIMING_UPDATE to one positive optimizer-update number.
The selected update is timed with host perf_counter_ns boundaries plus CUDA
Events on the current MAIN stream; the only event synchronization occurs after
the logical optimizer update has returned.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from . import wandb_telemetry as _wandb
from .local_chart_store import ensure_local_chart_store


_TIMING_UPDATE_ENV = "THOG2_PROCESSING_UPDATE_TIMING_UPDATE"
_TIMING_SCHEMA_VERSION = 1
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


class _UpdateTimingRecorder:
    def __init__(self, trainer: Any, optimizer_update: int) -> None:
        self.trainer = trainer
        self.optimizer_update = int(optimizer_update)
        self.device = torch.device(trainer.device)
        if self.device.type != "cuda":
            raise RuntimeError("processing update timing requires CUDA")
        self.host_update_start_ns = time.perf_counter_ns()
        self.cuda_update_start = self._record_cuda_event()
        self.phases: list[Dict[str, Any]] = []
        self.forward_count = 0
        self.backward_count = 0
        self.optimizer_count = 0

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

    def finish(self) -> Dict[str, Any]:
        host_update_end_ns = time.perf_counter_ns()
        cuda_update_end = self._record_cuda_event()
        harvest_started_ns = time.perf_counter_ns()
        cuda_update_end.synchronize()
        harvest_wait_ms = (time.perf_counter_ns() - harvest_started_ns) / 1_000_000.0

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

        host_update_ms = (host_update_end_ns - self.host_update_start_ns) / 1_000_000.0
        timeline = []
        cursor_ms = 0.0
        overlap_detected = False
        for row in completed:
            start_ms = float(row["host_start_ms"])
            end_ms = float(row["host_end_ms"])
            if start_ms > cursor_ms:
                timeline.append(
                    {
                        "phase": "other",
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
        if cursor_ms < host_update_ms:
            timeline.append(
                {
                    "phase": "other",
                    "micro_step": None,
                    "host_start_ms": cursor_ms,
                    "host_end_ms": host_update_ms,
                    "host_duration_ms": host_update_ms - cursor_ms,
                    "cuda_main_ms": None,
                }
            )

        totals: Dict[str, float] = {"forward": 0.0, "backward": 0.0, "optimizer": 0.0, "other": 0.0}
        for row in timeline:
            phase = str(row["phase"])
            totals[phase] = totals.get(phase, 0.0) + float(row["host_duration_ms"])
        partition_sum_ms = sum(totals.values())

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
        return {
            "schema_version": _TIMING_SCHEMA_VERSION,
            "optimizer_update": self.optimizer_update,
            "host_update_ms": host_update_ms,
            "cuda_main_update_ms": float(self.cuda_update_start.elapsed_time(cuda_update_end)),
            "harvest_wait_ms": harvest_wait_ms,
            "host_partition_sum_ms": partition_sum_ms,
            "host_partition_residual_ms": host_update_ms - partition_sum_ms,
            "host_phase_overlap_detected": overlap_detected,
            "phase_totals_host_ms": totals,
            "timeline": timeline,
            "microsteps": microsteps,
            "capture": {
                "gradient_accumulation_steps": accumulation_steps,
                "batch_size": int(config.batch_size),
                "block_size": int(config.block_size),
                "premat": bool(getattr(config, "premat", False)),
                "premat_target_layer": getattr(config, "premat_target_layer", None),
                "premat_target_matrix": getattr(config, "premat_target_matrix", None),
                "nsight_processing_logging": str(getattr(config, "premat_processing_logging", "disabled")),
                "cuda_owner": "MAIN",
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


def _capture_train_one_update(trainer: Any, original_train_one_update: Any, store: Any, optimizer_update: int) -> Dict[str, Any]:
    recorder = _UpdateTimingRecorder(trainer, optimizer_update)
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
    register_optimizer_pre = getattr(trainer.optimizer, "register_step_pre_hook", None)
    register_optimizer_post = getattr(trainer.optimizer, "register_step_post_hook", None)
    if callable(register_optimizer_pre) and callable(register_optimizer_post):
        handles.append(register_optimizer_pre(optimizer_pre_hook))
        handles.append(register_optimizer_post(optimizer_post_hook))

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
        payload = recorder.finish()
        payload["completed_updates_after"] = int(trainer.state.completed_updates)
        payload["successful_update"] = bool(
            result is not None
            and not bool(float(result.get("skipped_update", 0.0)))
            and int(trainer.state.completed_updates) >= int(optimizer_update)
        )
        payload["exception"] = None if caught is None else type(caught).__name__
        destination = _write_payload(store, payload)
        world_size = max(1, int(getattr(trainer.distributed, "world_size", 1)))
        tokens_per_update = (
            int(trainer.config.batch_size)
            * int(trainer.config.block_size)
            * int(trainer.config.gradient_accumulation_steps)
            * world_size
        )
        host_update_seconds = float(payload["host_update_ms"]) / 1000.0
        if host_update_seconds > 0.0:
            store.append_processing_throughput(
                int(optimizer_update),
                float(tokens_per_update) / host_update_seconds,
            )
        store.heartbeat(int(trainer.state.completed_updates), run_state="recording", force=True)
        print(
            "THOG2 Processing update timing: "
            f"update={optimizer_update} host={payload['host_update_ms']:.3f} ms "
            f"MAIN-CUDA={payload['cuda_main_update_ms']:.3f} ms "
            f"harvest-wait={payload['harvest_wait_ms']:.3f} ms "
            f"file={destination}",
            flush=True,
        )
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

    def timed_train_one_update() -> Dict[str, Any]:
        prospective_update = int(trainer.state.completed_updates) + 1
        if prospective_update != requested_update:
            return original_train_one_update()
        store = store_holder.get("store")
        if store is None:
            raise RuntimeError("processing update timing store was not attached before training")
        return _capture_train_one_update(
            trainer,
            original_train_one_update,
            store,
            requested_update,
        )

    trainer.train_one_update = timed_train_one_update
    _ORIGINAL_ATTACH_TELEMETRY(trainer, telemetry)
    store_holder["store"] = ensure_local_chart_store(telemetry)
    if str(getattr(trainer.config, "premat_processing_logging", "disabled")) == "enabled":
        print(
            "THOG2 WARNING: whole-update timing is running in the Nsight Processing process; "
            "for an uncontaminated non-Nsight capture, disable --premat_processing_logging.",
            flush=True,
        )


_wandb.attach_telemetry = _attach_telemetry_with_processing_update_timing


__all__ = ["_attach_telemetry_with_processing_update_timing"]
# ^^^ THOG
