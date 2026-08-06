from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import Callable, Optional, Sequence, Tuple

import torch

from .plastic_depth_coarse import PlasticCoarseTrialResult
from .plastic_depth_fresh_state import PlasticFreshTrainingState


Clock = Callable[[], float]
ProgressSink = Callable[[str], None]


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
            f"PLASTIC COARSE - trial {trial_index}/{trial_count}",
            f"  layers:      {layers}",
            f"  training:    {n_steps} steps, starting at step 0",
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


def run_fixed_plastic_coarse_trial(
    state: PlasticFreshTrainingState,
    *,
    trial_index: int,
    n_steps: int,
    evaluation_steps_count: int,
    clock: Clock = time.perf_counter,
    progress_sink: Optional[ProgressSink] = None,
) -> PlasticCoarseTrialResult:
    if state.phase != "coarse":
        raise ValueError("fixed COARSE trial requires a coarse fresh state")
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    trainer = state.trainer
    if getattr(trainer.config, "plastic__runtime_phase", "fine") != "coarse":
        raise RuntimeError("COARSE trainer was not constructed in coarse runtime phase")
    if int(trainer.state.completed_updates) != 0:
        raise RuntimeError("COARSE trial did not begin at local step zero")
    if int(trainer.config.max_updates) < n_steps:
        raise ValueError(
            "COARSE trainer max_updates is below the requested trial length: "
            f"max_updates={trainer.config.max_updates}, n_steps={n_steps}"
        )

    if progress_sink is not None:
        progress_sink(
            f"C {trial_index:02d} step {0:6d}/{n_steps:<6d} "
            f"layers={state.active_layer_count:<4d} local step zero"
        )
    if torch.device(trainer.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(trainer.device)
    _synchronize(trainer)
    started = clock()
    try:
        for local_step in range(1, n_steps + 1):
            metrics = trainer.train_one_update()
            if float(metrics.get("skipped_update", 0.0)) != 0.0:
                raise FloatingPointError(
                    "PLASTIC COARSE trial encountered a skipped non-finite update"
                )
            completed = int(trainer.state.completed_updates)
            if completed != local_step:
                raise RuntimeError(
                    "PLASTIC COARSE local step sequence diverged: "
                    f"expected={local_step}, completed={completed}"
                )
            if progress_sink is not None:
                progress_sink(
                    f"C {trial_index:02d} step {local_step:6d}/{n_steps:<6d} "
                    f"layers={state.active_layer_count:<4d} "
                    f"loss={float(metrics['training_loss']):.6f}"
                )
        _synchronize(trainer)
        training_elapsed = float(clock() - started)
        if not math.isfinite(training_elapsed) or training_elapsed <= 0.0:
            raise RuntimeError(
                f"invalid PLASTIC COARSE training elapsed time: {training_elapsed!r}"
            )
        peak_allocated, peak_reserved = _peak_memory_gib(trainer)
        validation_losses = _validation_losses(trainer, evaluation_steps_count)
        return PlasticCoarseTrialResult(
            trial_index=trial_index,
            layers=state.active_layer_count,
            status="success",
            validation_losses=validation_losses,
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
        return PlasticCoarseTrialResult(
            trial_index=trial_index,
            layers=state.active_layer_count,
            status="failed",
            training_elapsed_seconds=(
                elapsed if math.isfinite(elapsed) and elapsed > 0.0 else None
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
