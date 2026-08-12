# vvv THOG
"""Final non-authoritative DEPTH candidate executor for fixed-count THOG runs."""

from __future__ import annotations

import math
from typing import Any, Tuple

import torch
from torch.nn import functional as F

from . import depth_weight_curves_and_observational_probes_patch as _depth
from .checkpointing import execute_logical_layers
from .semantic_materializer import ATTENTION_QUERY_WEIGHT


@torch.no_grad()
def _observational_candidate_loss(
    trainer: Any,
    trajectory: Any,
    batch: Any,
    probe_targets: torch.Tensor,
    candidate: int,
) -> Tuple[float, int]:
    raw_model = trainer.raw_model
    positions = torch.arange(
        batch.inputs.shape[1],
        dtype=torch.long,
        device=batch.inputs.device,
    )
    hidden = raw_model.transformer.drop(
        raw_model.transformer.wte(batch.inputs)
        + raw_model.transformer.wpe(positions)
    )
    hidden, execution_report = execute_logical_layers(
        hidden,
        n_layer=int(candidate),
        segment_size=int(raw_model.checkpoint_segment_size),
        logical_block=raw_model._logical_block,
        training=False,
    )
    hidden = raw_model.transformer.ln_f(hidden)
    logits = raw_model.lm_head(hidden)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        probe_targets.reshape(-1),
        ignore_index=-1,
    )
    if not bool(torch.isfinite(loss).item()):
        return float("inf"), int(execution_report.logical_layers)
    return trainer.distributed.mean_float(loss.detach()), int(execution_report.logical_layers)


@torch.no_grad()
def _run_observational_probe_final(trainer: Any, *, update: int) -> None:
    trajectory = _depth._depth_trajectory_from_model(trainer.raw_model)
    if trajectory is None:
        return
    current = _depth._current_observational_layer_count(trainer, trajectory)
    radius = int(trainer.config.plastic__layer_count_probe_radius)
    candidates = tuple(range(max(1, current - radius), current + radius + 1))
    batch = _depth._observational_probe_batch(trainer, update)
    probe_targets, sampled_token_count = _depth._masked_observational_targets(
        trainer,
        batch.targets,
    )
    was_training = trainer.model.training
    trainer.model.eval()
    measurements = []
    try:
        for candidate in candidates:
            coordinates = torch.linspace(
                1.0,
                100.0,
                int(candidate),
                dtype=torch.float64,
                device=trajectory.coefficients[ATTENTION_QUERY_WEIGHT].device,
            )
            trajectory._thog_observational_depth_coordinates = coordinates
            executed_logical_layers = 0
            try:
                with trainer.autocast_context():
                    value, executed_logical_layers = _observational_candidate_loss(
                        trainer,
                        trajectory,
                        batch,
                        probe_targets,
                        candidate,
                    )
            except torch.OutOfMemoryError:
                value = float("inf")
                if trainer.device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            measurements.append(
                {
                    "active_layers": int(candidate),
                    "validation_loss": float(value),
                    "training_time": None,
                    "peak_allocated_gib": None,
                    "peak_reserved_gib": None,
                    "feasible": math.isfinite(float(value)),
                    "score": float(value),
                    "observational_only": True,
                    "executed_logical_layers": int(executed_logical_layers),
                }
            )
    finally:
        if hasattr(trajectory, "_thog_observational_depth_coordinates"):
            delattr(trajectory, "_thog_observational_depth_coordinates")
        trainer.model.train(was_training)
    current_measurement = next(
        (item for item in measurements if int(item["active_layers"]) == current),
        None,
    )
    if current_measurement is None or not math.isfinite(
        float(current_measurement["validation_loss"])
    ):
        raise RuntimeError(
            "observational DEPTH probe did not produce a finite current-count loss"
        )
    trainer._record(
        "plastic_depth_count_decision",
        previous_active_layers=current,
        selected_active_layers=current,
        candidates=tuple(measurements),
        objective="observational_only",
        sampled_token_count=sampled_token_count,
        observational_only=True,
        probe_update=int(update),
        probe_radius=radius,
        public_coordinates=tuple(
            float(value)
            for value in torch.linspace(
                1.0,
                100.0,
                current,
                dtype=torch.float64,
            ).tolist()
        ),
        transition={},
    )


_depth._run_observational_probe = _run_observational_probe_final
# ^^^ THOG


__all__ = ["_observational_candidate_loss", "_run_observational_probe_final"]
# ^^^ THOG
