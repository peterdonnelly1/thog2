# vvv THOG
from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Callable, Optional, Sequence, Tuple

import torch

from .plastic_depth_coarse import PlasticCoarseTrialResult
from .plastic_depth_fresh_state import PlasticFreshTrainingState


Clock = Callable[[], float]
ProgressSink = Callable[[str], None]


@dataclass(frozen=True)
class PlasticCoarseTrialProgress:
    trial_index: int
    layers: int
    completed_steps: int
    n_steps: int
    training_losses: Tuple[float, ...]
    training_elapsed_seconds: float


TrialCheckpointCallback = Callable[[PlasticCoarseTrialProgress, object], None]


def plastic_tokens_per_update(config) -> int:
    return int(config.batch_size) * int(config.block_size) * int(
        config.gradient_accumulation_steps
    )


def render_plastic_coarse_trial_header(
    *,
    trial_index: int,
    trial_count: int,
    layers: int,
    n_steps: int,
    evaluation_steps_count: int,
    objective: str,
    geometry_initialisation: str,
) -> str:
    return "\n".join(
        (
            f"TRIAL {trial_index}/{trial_count}",
            f"  layers:      {layers}",
            f"  steps:       {n_steps}",
            f"  evaluation:  validation mean over final {evaluation_steps_count} batches",
            f"  goal:        {objective}",
            f"  geometry:    fixed {geometry_initialisation}",
        )
    )


def _validation_losses(trainer, evaluation_steps_count: int) -> Tuple[float, ...]:
    if evaluation_steps_count < 1:
        raise ValueError("evaluation_steps_count must be positive")
    was_training = trainer.model.training
    trainer.model.eval()
    losses = []
    try:
        with torch.no_grad():
            for _ in range(evaluation_steps_count):
                batch = trainer.batch_source.get_batch("val", device=trainer.device)
                with trainer.autocast_context():
                    _, loss = trainer.model(batch.inputs, batch.targets)
                local_finite = loss is not None and bool(torch.isfinite(loss).item())
                trainer.distributed.require_all_true(
                    local_finite,
                    "non-finite PLASTIC COARSE validation loss on at least one rank",
                )
                if loss is None:
                    raise RuntimeError("model did not return a PLASTIC COARSE validation loss")
                losses.append(trainer.distributed.mean_float(loss.detach()))
    finally:
        trainer.model.train(was_training)
    return tuple(float(value) for value in losses)


def _peak_memory_gib(trainer) -> Tuple[Optional[float], Optional[float]]:
    if torch.device(trainer.device).type != "cuda":
        return None, None
    allocated = torch.tensor(
        torch.cuda.max_memory_allocated(trainer.device) / (1024.0**3),
        dtype=torch.float64,
        device=trainer.device,
    )
    reserved = torch.tensor(
        torch.cuda.max_memory_reserved(trainer.device) / (1024.0**3),
        dtype=torch.float64,
        device=trainer.device,
    )
    return (
        trainer.distributed.max_float(allocated),
        trainer.distributed.max_float(reserved),
    )


def _synchronize(trainer) -> None:
    if torch.device(trainer.device).type == "cuda":
        torch.cuda.synchronize(trainer.device)


def _worst_rank_elapsed(trainer, local_elapsed: float) -> float:
    value = torch.tensor(
        float(local_elapsed),
        dtype=torch.float64,
        device=trainer.device,
    )
    return trainer.distributed.max_float(value)


def _validate_resume_prefix(
    *,
    completed_updates: int,
    n_steps: int,
    prior_training_losses: Sequence[float],
    prior_training_elapsed_seconds: float,
) -> Tuple[float, ...]:
    losses = tuple(float(value) for value in prior_training_losses)
    if completed_updates < 0 or completed_updates > n_steps:
        raise ValueError(
            "COARSE resume completed_updates lies outside the trial budget; "
            f"completed={completed_updates}, n_steps={n_steps}"
        )
    if len(losses) != completed_updates:
        raise ValueError(
            "COARSE resume loss history does not match completed updates; "
            f"losses={len(losses)}, completed={completed_updates}"
        )
    if not all(math.isfinite(value) for value in losses):
        raise ValueError("COARSE resume loss history contains a non-finite value")
    if (
        not math.isfinite(float(prior_training_elapsed_seconds))
        or float(prior_training_elapsed_seconds) < 0.0
    ):
        raise ValueError(
            "COARSE resume prior training elapsed time must be finite and non-negative"
        )
    return losses


def run_fixed_plastic_coarse_trial(
    state: PlasticFreshTrainingState,
    *,
    trial_index: int,
    n_steps: int,
    evaluation_steps_count: int,
    clock: Clock = time.perf_counter,
    progress_clock: Clock = time.perf_counter,
    progress_sink: Optional[ProgressSink] = None,
    prior_training_losses: Sequence[float] = (),
    prior_training_elapsed_seconds: float = 0.0,
    checkpoint_interval: int = 0,
    checkpoint_callback: Optional[TrialCheckpointCallback] = None,
) -> PlasticCoarseTrialResult:
    if state.phase != "coarse":
        raise ValueError("fixed COARSE trial requires a coarse fresh state")
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be non-negative")
    if checkpoint_interval > 0 and checkpoint_callback is None:
        raise ValueError(
            "positive COARSE checkpoint_interval requires checkpoint_callback"
        )
    trainer = state.trainer
    if getattr(trainer.config, "plastic__runtime_phase", "fine") != "coarse":
        raise RuntimeError("COARSE trainer was not constructed in coarse runtime phase")
    completed_at_start = int(trainer.state.completed_updates)
    training_losses = list(
        _validate_resume_prefix(
            completed_updates=completed_at_start,
            n_steps=n_steps,
            prior_training_losses=prior_training_losses,
            prior_training_elapsed_seconds=prior_training_elapsed_seconds,
        )
    )
    if int(trainer.config.max_updates) < n_steps:
        raise ValueError(
            "COARSE trainer max_updates is below the requested trial length: "
            f"max_updates={trainer.config.max_updates}, n_steps={n_steps}"
        )

    log_interval_coarse = int(getattr(trainer.config, "plastic__log_interval_coarse", 10))
    if progress_sink is not None:
        status = "local step zero" if completed_at_start == 0 else "resumed"
        progress_sink(
            f"C {trial_index:02d} {completed_at_start:6d}/{n_steps:<6d} "
            f"{status:<22} {float(prior_training_elapsed_seconds):8.1f}s"
        )
    if torch.device(trainer.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(trainer.device)
    _synchronize(trainer)
    started = clock()
    progress_started = progress_clock()
    try:
        for local_step in range(completed_at_start + 1, n_steps + 1):
            metrics = trainer.train_one_update()
            if float(metrics.get("skipped_update", 0.0)) != 0.0:
                raise FloatingPointError(
                    "PLASTIC COARSE trial encountered a skipped non-finite update"
                )
            training_loss = float(metrics["training_loss"])
            training_losses.append(training_loss)
            completed = int(trainer.state.completed_updates)
            if completed != local_step:
                raise RuntimeError(
                    "PLASTIC COARSE local step sequence diverged: "
                    f"expected={local_step}, completed={completed}"
                )
            progress_due = (
                local_step == 1
                or local_step == n_steps
                or local_step % log_interval_coarse == 0
            )
            if progress_sink is not None and progress_due:
                elapsed_seconds = (
                    float(prior_training_elapsed_seconds)
                    + max(0.0, float(progress_clock() - progress_started))
                )
                progress_sink(
                    f"C {trial_index:02d} {local_step:6d}/{n_steps:<6d} "
                    f"loss={training_loss:.6f} {elapsed_seconds:8.1f}s"
                )
            should_checkpoint = (
                checkpoint_interval > 0
                and local_step < n_steps
                and local_step % checkpoint_interval == 0
            )
            if should_checkpoint:
                _synchronize(trainer)
                segment_elapsed = _worst_rank_elapsed(
                    trainer,
                    float(clock() - started),
                )
                progress = PlasticCoarseTrialProgress(
                    trial_index=int(trial_index),
                    layers=int(state.active_layer_count),
                    completed_steps=local_step,
                    n_steps=n_steps,
                    training_losses=tuple(training_losses),
                    training_elapsed_seconds=(
                        float(prior_training_elapsed_seconds) + segment_elapsed
                    ),
                )
                assert checkpoint_callback is not None
                checkpoint_callback(progress, trainer)
        _synchronize(trainer)
        local_elapsed = float(clock() - started)
        if not math.isfinite(local_elapsed) or local_elapsed < 0.0:
            raise RuntimeError(
                f"invalid PLASTIC COARSE training elapsed time: {local_elapsed!r}"
            )
        segment_elapsed = _worst_rank_elapsed(trainer, local_elapsed)
        training_elapsed = (
            float(prior_training_elapsed_seconds) + segment_elapsed
        )
        if training_elapsed <= 0.0:
            raise RuntimeError(
                f"invalid PLASTIC COARSE accumulated training time: {training_elapsed!r}"
            )
        peak_allocated, peak_reserved = _peak_memory_gib(trainer)
        validation_losses = _validation_losses(trainer, evaluation_steps_count)
        return PlasticCoarseTrialResult(
            trial_index=trial_index,
            layers=state.active_layer_count,
            status="success",
            validation_losses=validation_losses,
            training_losses=tuple(training_losses),
            training_elapsed_seconds=training_elapsed,
            training_steps=n_steps,
            tokens_per_update=plastic_tokens_per_update(trainer.config),
            peak_allocated_gib=peak_allocated,
            peak_reserved_gib=peak_reserved,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        elapsed = float(clock() - started)
        accumulated_elapsed = (
            float(prior_training_elapsed_seconds)
            + (elapsed if math.isfinite(elapsed) and elapsed >= 0.0 else 0.0)
        )
        return PlasticCoarseTrialResult(
            trial_index=trial_index,
            layers=state.active_layer_count,
            status="failed",
            training_losses=tuple(training_losses),
            training_elapsed_seconds=(
                accumulated_elapsed if accumulated_elapsed > 0.0 else None
            ),
            training_steps=int(trainer.state.completed_updates),
            tokens_per_update=plastic_tokens_per_update(trainer.config),
            error_class=type(error).__name__,
            error_message=str(error),
        )


def coarse_trial_training_config(
    base_config,
    *,
    active_layer_count: int,
    n_steps: int,
):
    return replace(
        base_config,
        plastic__coarse_phase="disabled",
        plastic__runtime_phase="coarse",
        plastic__initial_layer_count=int(active_layer_count),
        max_updates=int(n_steps),
    )


__all__ = [
    "PlasticCoarseTrialProgress",
    "coarse_trial_training_config",
    "plastic_tokens_per_update",
    "render_plastic_coarse_trial_header",
    "run_fixed_plastic_coarse_trial",
]
# ^^^ THOG
