# vvv THOG
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor

from .stage4_trainer import Stage4Trainer
from .stage6_diagnostics import gradient_report, stage6_sheet_diagnostics
from .training_model import TrainingSheetGPT


def trace_digest(trace) -> str:
    payload = json.dumps(trace, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# vvv THOG resumed runs retain lifetime token totals but measure throughput from this process session only
def _session_consumed_tokens(
    starting_completed_updates: int,
    completed_updates: int,
    tokens_per_update: int,
) -> int:
    session_completed_updates = completed_updates - starting_completed_updates
    if session_completed_updates < 0:
        raise ValueError("completed updates moved backwards during the training session")
    return session_completed_updates * tokens_per_update
# ^^^ THOG


# vvv THOG soft wall-clock budget helpers for equal-time geometry grids
def _max_wall_seconds(max_wall_minutes: int) -> Optional[float]:
    if max_wall_minutes <= 0:
        return None
    return float(max_wall_minutes) * 60.0


def _wall_limit_reached(
    wall_started: float,
    max_wall_seconds: Optional[float],
) -> bool:
    if max_wall_seconds is None:
        return False
    return (time.perf_counter() - wall_started) >= max_wall_seconds
# ^^^ THOG


# vvv THOG optionally skip the update-zero validation tax for known-good long runs
def _initial_eval_enabled() -> bool:
    value = os.environ.get("THOG2_INITIAL_EVAL", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}
# ^^^ THOG


# vvv THOG
_PROGRESS_FIELD_LABELS = {
    "consumed_tokens": "cum tokens",
    "training_loss": "training loss",
    "validation_loss": "validation loss",
    "learning_rate": "learning rate",
    "gradient_norm": "gradient norm",
}
_PROGRESS_LOSS_LABEL_WIDTH = len("validation loss")
_PROGRESS_VALIDATION_STYLE_START = "\033[1;33m"
_PROGRESS_VALIDATION_STYLE_END = "\033[0m"


def _progress_field(label: str, value: Any) -> str:
    if label in {"training loss", "validation loss"}:
        return f"{label:<{_PROGRESS_LOSS_LABEL_WIDTH}}={value}"
    return f"{label}={value}"


def _progress_elapsed_seconds(value: Any) -> str:
    return f"{value}s"


def format_progress_line(run_id: str, event: str, payload: Dict[str, Any]) -> str:
    if event == "optimizer_progress":
        ordered_fields = (
            "completed_updates",
            "cumulative_training_seconds",
            "tok/s",
            "consumed_tokens",
            "training_loss",
            "learning_rate",
            "gradient_norm",
        )
        prefix = "T"
    elif event == "evaluation_completed":
        ordered_fields = (
            "completed_updates",
            "cumulative_training_seconds",
            "tok/s",
            "consumed_tokens",
            "training_loss",
            "validation_loss",
        )
        prefix = "V"
    else:
        fields = [event]
        fields.extend(
            _progress_field(key.replace("_", " "), value)
            for key, value in payload.items()
        )
        fields.append(_progress_field("run_id", run_id))
        return "  ".join(fields)

    fields = [prefix]
    for key in ordered_fields:
        if key not in payload:
            continue
        if key == "completed_updates":
            fields.append(str(payload[key]))
            continue
        if key == "cumulative_training_seconds":
            fields.append(_progress_elapsed_seconds(payload[key]))
            continue
        label = _PROGRESS_FIELD_LABELS.get(key, key)
        fields.append(_progress_field(label, payload[key]))
    fields.append(_progress_field("run_id", run_id))
    line = "  ".join(fields)
    if event == "evaluation_completed":
        return f"{_PROGRESS_VALIDATION_STYLE_START}{line}{_PROGRESS_VALIDATION_STYLE_END}"
    return line
# ^^^ THOG


class Stage6Trainer(Stage4Trainer):
    """Stage 4 trainer with controlled-pilot timing and detached diagnostics."""

    def __init__(
        self,
        config,
        train_tokens: Tensor,
        validation_tokens: Tensor,
    ) -> None:
        super().__init__(config, train_tokens, validation_tokens)
        self.gradient_diagnostics: List[Dict[str, Any]] = []

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _timed(self, function):
        self._synchronize()
        started = time.perf_counter()
        result = function()
        self._synchronize()
        return result, time.perf_counter() - started

    def _print_progress(self, run_id: str, event: str, **payload: Any) -> None:
        if not self.distributed.is_primary:
            return
        print(format_progress_line(run_id, event, payload), flush=True)                                                                                  # <<< THOG emit brace-free T/V progress with run_id last
        if event == "run_started":
            print(flush=True)                                                                                                                            # <<< THOG separate startup summary from progress rows

    def _before_optimizer_step(self) -> None:
        next_update = self.state.completed_updates + 1
        capture = (
            next_update == 1
            or next_update == self.config.max_updates
            or next_update % self.config.log_interval == 0
        )
        if not capture or not isinstance(self.raw_model, TrainingSheetGPT):
            return
        self.gradient_diagnostics.append(
            {
                "completed_update": next_update,
                "families": gradient_report(self.raw_model),
            }
        )

    def optimizer_batch_trace(self):
        return tuple(
            tuple(int(value) for value in event.payload["starts"])
            for event in self.events
            if event.name == "microbatch"
        )

    def _record_evaluation(
        self,
        *,
        run_id: str,
        evaluation_rows: List[Dict[str, Any]],
        training_seconds: float,
        evaluation_seconds: float,
        wall_started: float,
        tokens_per_update: int,
        session_consumed_tokens: int,
    ) -> float:
        losses, eval_elapsed = self._timed(self.evaluate)
        evaluation_seconds += eval_elapsed
        completed_updates = self.state.completed_updates
        evaluation_rows.append(
            {
                "completed_updates": completed_updates,
                "consumed_tokens": completed_updates * tokens_per_update,
                "session_consumed_tokens": session_consumed_tokens,
                "training_seconds": training_seconds,
                "wall_seconds": time.perf_counter() - wall_started,
                "evaluation_seconds": eval_elapsed,
                **losses,
            }
        )
        self._print_progress(
            run_id,
            "evaluation_completed",
            completed_updates=completed_updates,
            consumed_tokens=completed_updates * tokens_per_update,
            session_consumed_tokens=session_consumed_tokens,
            cumulative_training_seconds=training_seconds,                                                                                               # <<< THOG expose cumulative training time and tok/s on validation rows
            validation_loss=losses["val"],
            training_loss=losses["train"],
        )
        return evaluation_seconds

    def run_pilot(
        self,
        *,
        run_id: str,
        protocol_sha256: str,
        dataset: Dict[str, Any],
        result_path: Union[str, Path],
    ) -> Dict[str, Any]:
        target = Path(result_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tokens_per_update = self.config.batch_size * self.config.gradient_accumulation_steps * self.config.block_size
        starting_completed_updates = self.state.completed_updates                                                                                        # <<< THOG anchor resumed-session timing before the first update in this process
        training_seconds = 0.0
        evaluation_seconds = 0.0
        checkpoint_seconds = 0.0
        update_rows: List[Dict[str, Any]] = []
        evaluation_rows: List[Dict[str, Any]] = []
        wall_started = time.perf_counter()
        max_wall_seconds = _max_wall_seconds(int(self.config.max_wall_minutes))                                                                           # <<< THOG stop equal-time geometry screens before starting an update after the budget expires
        stop_reason = "max_updates"
        self._print_progress(
            run_id,
            "run_started",
            max_updates=self.config.max_updates,
            max_wall_minutes=self.config.max_wall_minutes,
            tokens_per_update=tokens_per_update,
        )

        if self.config.eval_interval > 0 and self.state.completed_updates == 0 and _initial_eval_enabled():
            evaluation_seconds = self._record_evaluation(
                run_id=run_id,
                evaluation_rows=evaluation_rows,
                training_seconds=0.0,
                evaluation_seconds=evaluation_seconds,
                wall_started=wall_started,
                tokens_per_update=tokens_per_update,
                session_consumed_tokens=0,
            )

        while self.state.completed_updates < self.config.max_updates:
            if _wall_limit_reached(wall_started, max_wall_seconds):
                stop_reason = "max_wall_minutes"
                self._print_progress(
                    run_id,
                    "wall_time_limit_reached",
                    completed_updates=self.state.completed_updates,
                    cumulative_wall_seconds=time.perf_counter() - wall_started,
                    max_wall_minutes=self.config.max_wall_minutes,
                )
                break
            metrics, elapsed = self._timed(self.train_one_update)
            training_seconds += elapsed
            completed_updates = self.state.completed_updates
            current_session_consumed_tokens = _session_consumed_tokens(starting_completed_updates, completed_updates, tokens_per_update)                 # <<< THOG separate lifetime token accounting from tokens processed since this resume/start
            update_rows.append(
                {
                    **metrics,
                    "update_seconds": elapsed,
                    "cumulative_training_seconds": training_seconds,
                    "cumulative_wall_seconds": time.perf_counter() - wall_started,
                    "consumed_tokens": completed_updates * tokens_per_update,
                    "session_consumed_tokens": current_session_consumed_tokens,
                }
            )
            report_update = completed_updates == 1 or completed_updates == self.config.max_updates or completed_updates % self.config.log_interval == 0
            if report_update:
                self._print_progress(
                    run_id,
                    "optimizer_progress",
                    completed_updates=completed_updates,
                    consumed_tokens=completed_updates * tokens_per_update,
                    session_consumed_tokens=current_session_consumed_tokens,
                    training_loss=metrics["training_loss"],
                    learning_rate=metrics["learning_rate"],
                    gradient_norm=metrics["gradient_norm"],
                    cumulative_training_seconds=training_seconds,
                )
            if self.config.eval_interval > 0 and completed_updates % self.config.eval_interval == 0:
                evaluation_seconds = self._record_evaluation(
                    run_id=run_id,
                    evaluation_rows=evaluation_rows,
                    training_seconds=training_seconds,
                    evaluation_seconds=evaluation_seconds,
                    wall_started=wall_started,
                    tokens_per_update=tokens_per_update,
                    session_consumed_tokens=current_session_consumed_tokens,
                )
            if self.config.checkpoint_interval > 0 and completed_updates % self.config.checkpoint_interval == 0:
                _, save_elapsed = self._timed(lambda: self.save_checkpoint(Path(self.config.out_dir) / "ckpt.pt"))
                checkpoint_seconds += save_elapsed

        final_session_consumed_tokens = _session_consumed_tokens(starting_completed_updates, self.state.completed_updates, tokens_per_update)             # <<< THOG final session token count is independent of lifetime completed updates
        if self.config.eval_interval > 0 and (not evaluation_rows or evaluation_rows[-1]["completed_updates"] != self.state.completed_updates):
            evaluation_seconds = self._record_evaluation(
                run_id=run_id,
                evaluation_rows=evaluation_rows,
                training_seconds=training_seconds,
                evaluation_seconds=evaluation_seconds,
                wall_started=wall_started,
                tokens_per_update=tokens_per_update,
                session_consumed_tokens=final_session_consumed_tokens,
            )

        checkpoint_path = Path(self.config.out_dir) / "ckpt.pt"
        self._print_progress(run_id, "final_checkpoint_started", completed_updates=self.state.completed_updates)
        _, final_checkpoint_seconds = self._timed(lambda: self.save_checkpoint(checkpoint_path))
        checkpoint_seconds += final_checkpoint_seconds
        wall_seconds = time.perf_counter() - wall_started

        diagnostics: Optional[Dict[str, Any]] = None
        if isinstance(self.raw_model, TrainingSheetGPT):
            diagnostics = stage6_sheet_diagnostics(self.raw_model)

        optimizer_trace = self.optimizer_batch_trace()
        train_stream_trace = self.batch_source.training_trace()
        validation_trace = self.batch_source.validation_trace()
        optimizer_trace_sha256 = trace_digest(optimizer_trace)
        result: Dict[str, Any] = {
            "stage": 6,
            "suite": "controlled_pilot_run",
            "status": "completed",
            "run_id": run_id,
            "protocol_sha256": protocol_sha256,
            "dataset": dataset,
            "training_config": asdict(self.config),
            "parameter_report": self.parameter_report,
            "distributed": self.distributed.report(),
            "hardware": {
                "device": str(self.device),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_device_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
                "cuda_total_memory_bytes": int(torch.cuda.get_device_properties(self.device).total_memory) if self.device.type == "cuda" else None,
            },
            "budget": {
                "completed_updates": self.state.completed_updates,
                "tokens_per_update": tokens_per_update,
                "consumed_tokens": self.state.completed_updates * tokens_per_update,
                "stop_reason": stop_reason,
                "max_wall_minutes": self.config.max_wall_minutes,
                "session_completed_updates": self.state.completed_updates - starting_completed_updates,
                "session_consumed_tokens": final_session_consumed_tokens,
            },
            "timing": {
                "training_seconds": training_seconds,
                "evaluation_seconds": evaluation_seconds,
                "checkpoint_seconds": checkpoint_seconds,
                "wall_seconds": wall_seconds,
                "max_wall_seconds": max_wall_seconds,
                "tokens_per_training_second": final_session_consumed_tokens / training_seconds if training_seconds > 0.0 else 0.0,
            },
            "updates": update_rows,
            "evaluations": evaluation_rows,
            "trace": {
                "training_sha256": optimizer_trace_sha256,
                "training_starts": optimizer_trace,
                "optimizer_training_sha256": optimizer_trace_sha256,
                "optimizer_training_starts": optimizer_trace,
                "train_stream_sha256": trace_digest(train_stream_trace),
                "train_stream_starts": train_stream_trace,
                "validation_sha256": trace_digest(validation_trace),
                "validation_starts": validation_trace,
                "all_sha256": self.batch_source.trace_digest("all"),
            },
            "memory": self.memory_telemetry.report(),
            "gradient_diagnostics": self.gradient_diagnostics,
            "sheet_diagnostics": diagnostics,
            "checkpoint": {"path": str(checkpoint_path), "bytes": checkpoint_path.stat().st_size},
        }
        finite_values = [training_seconds, evaluation_seconds, checkpoint_seconds, wall_seconds]
        if not all(math.isfinite(value) and value >= 0.0 for value in finite_values):
            raise FloatingPointError("non-finite Stage 6 timing evidence")
        if self.distributed.is_primary:
            target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.distributed.barrier()
        final_validation_loss = evaluation_rows[-1]["val"] if evaluation_rows else None
        self._print_progress(
            run_id,
            "run_completed",
            completed_updates=self.state.completed_updates,
            consumed_tokens=self.state.completed_updates * tokens_per_update,
            session_consumed_tokens=final_session_consumed_tokens,
            final_validation_loss=final_validation_loss,
            training_seconds=training_seconds,
            checkpoint_bytes=checkpoint_path.stat().st_size,
        )
        return result


__all__ = ["Stage6Trainer", "format_progress_line", "trace_digest"]
# ^^^ THOG
