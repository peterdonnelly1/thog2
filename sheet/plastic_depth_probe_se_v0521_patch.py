# vvv THOG
"""PLASTIC v0.521 diagnostic paired-token standard errors; controller decisions remain unchanged."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from . import trainer_step as _trainer_step
from . import training_model as _training_model


_ORIGINAL_CANDIDATE_HEAD_LOSS = _training_model.TrainingSheetGPT._plastic_depth_candidate_head_loss
_ORIGINAL_INLINE_PROBE_REQUEST = _trainer_step.TrainerStepMixin._plastic_depth_inline_probe_request


def _candidate_head_loss_with_token_diagnostics(
    self: Any,
    hidden: Tensor,
    targets: Tensor,
    sampled_token_indices: Optional[Tensor],
) -> Tensor:
    normalized = self.transformer.ln_f(hidden.detach())
    flattened_hidden = normalized.reshape(-1, normalized.shape[-1])
    flattened_targets = targets.reshape(-1)
    if sampled_token_indices is not None:
        indices = sampled_token_indices.to(device=flattened_hidden.device)
        if indices.numel() == 0:
            raise ValueError("PLASTIC DEPTH sampled probe requires at least one token")
        if int(indices.min().item()) < 0 or int(indices.max().item()) >= flattened_targets.numel():
            raise ValueError("PLASTIC DEPTH sampled token index is out of range")
        flattened_hidden = flattened_hidden.index_select(0, indices)
        flattened_targets = flattened_targets.index_select(0, indices)
    logits = self.lm_head(flattened_hidden)
    token_losses = F.cross_entropy(
        logits,
        flattened_targets,
        ignore_index=-1,
        reduction="none",
    )
    valid = flattened_targets != -1
    valid_token_losses = token_losses[valid]
    if valid_token_losses.numel() == 0:
        raise RuntimeError("PLASTIC DEPTH candidate probe has no valid target tokens")
    diagnostic_buffer = getattr(
        self,
        "_plastic_depth_v0521_candidate_token_losses",
        None,
    )
    if isinstance(diagnostic_buffer, list):
        diagnostic_buffer.append(valid_token_losses.detach())
    return valid_token_losses.mean()


def _local_paired_delta_stats(
    *,
    counts: Sequence[int],
    current_count: int,
    token_losses: Sequence[Tensor],
) -> Dict[int, Tuple[int, float, float]]:
    resolved_counts = tuple(int(value) for value in counts)
    if len(resolved_counts) != len(token_losses):
        raise RuntimeError(
            "PLASTIC DEPTH paired-token diagnostics lost candidate alignment: "
            f"counts={len(resolved_counts)}, token_loss_vectors={len(token_losses)}"
        )
    if current_count not in resolved_counts:
        raise RuntimeError(
            "PLASTIC DEPTH paired-token diagnostics are missing current L; "
            f"current={current_count}, counts={resolved_counts}"
        )
    reference = token_losses[resolved_counts.index(current_count)].detach().to(dtype=torch.float64)
    stats: Dict[int, Tuple[int, float, float]] = {}
    for count, values in zip(resolved_counts, token_losses):
        candidate = values.detach().to(dtype=torch.float64)
        if candidate.shape != reference.shape:
            raise RuntimeError(
                "PLASTIC DEPTH paired-token diagnostic vectors differ in shape; "
                f"current_shape={tuple(reference.shape)}, candidate={count}, candidate_shape={tuple(candidate.shape)}"
            )
        delta = candidate - reference
        n = int(delta.numel())
        stats[int(count)] = (
            n,
            float(delta.sum().item()),
            float(delta.square().sum().item()),
        )
    return stats


def _combine_paired_delta_standard_errors(
    *,
    counts: Sequence[int],
    current_count: int,
    gathered_stats: Sequence[Mapping[int, Tuple[int, float, float]]],
) -> Dict[int, Optional[float]]:
    result: Dict[int, Optional[float]] = {}
    for count_value in counts:
        count = int(count_value)
        n_total = 0
        sum_total = 0.0
        sum_sq_total = 0.0
        for rank_stats in gathered_stats:
            if count not in rank_stats:
                raise RuntimeError(
                    "PLASTIC DEPTH paired-token diagnostic candidate differs across ranks; "
                    f"candidate={count}"
                )
            n, value_sum, value_sum_sq = rank_stats[count]
            n_total += int(n)
            sum_total += float(value_sum)
            sum_sq_total += float(value_sum_sq)
        if count == int(current_count):
            result[count] = 0.0
            continue
        if n_total < 2:
            result[count] = None
            continue
        centered_sum_sq = sum_sq_total - (sum_total * sum_total / n_total)
        sample_variance = max(0.0, centered_sum_sq / (n_total - 1))
        result[count] = math.sqrt(sample_variance / n_total)
    return result


def _inline_probe_request_with_paired_token_se(
    self: Any,
    targets: Tensor,
    context: Dict[str, Any],
):
    diagnostic_buffer: list[Tensor] = []
    self.raw_model._plastic_depth_v0521_candidate_token_losses = diagnostic_buffer
    request = _ORIGINAL_INLINE_PROBE_REQUEST(self, targets, context)
    original_selector = request.selector

    def select(candidates: Tuple[Tuple[int, Tensor], ...]) -> int:
        try:
            selected_count = int(original_selector(candidates))
            counts = tuple(int(count) for count, _ in candidates)
            local_stats = _local_paired_delta_stats(
                counts=counts,
                current_count=int(context["current_count"]),
                token_losses=tuple(diagnostic_buffer),
            )
            gathered_stats = self.distributed.all_gather_object(local_stats)
            standard_errors = _combine_paired_delta_standard_errors(
                counts=counts,
                current_count=int(context["current_count"]),
                gathered_stats=gathered_stats,
            )
            context["paired_delta_standard_errors"] = dict(standard_errors)
            score_report = context.get("score_report")
            if score_report is not None:
                context["score_report"] = tuple(
                    {
                        **dict(item),
                        "paired_delta_standard_error": standard_errors.get(
                            int(item["active_layers"])
                        ),
                    }
                    for item in score_report
                )
            return selected_count
        finally:
            self.raw_model._plastic_depth_v0521_candidate_token_losses = None

    return replace(request, selector=select)


_training_model.TrainingSheetGPT._plastic_depth_candidate_head_loss = (
    _candidate_head_loss_with_token_diagnostics
)
_trainer_step.TrainerStepMixin._plastic_depth_inline_probe_request = (
    _inline_probe_request_with_paired_token_se
)


__all__ = [
    "_combine_paired_delta_standard_errors",
    "_local_paired_delta_stats",
]
# ^^^ THOG
