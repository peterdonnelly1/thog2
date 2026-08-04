# vvv THOG
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor


PlasticDepthCandidateSelector = Callable[[Tuple[Tuple[int, Tensor], ...]], int]
PlasticDepthUpwardPreparation = Callable[[], None]
PlasticDepthUpwardFeasibilitySynchronizer = Callable[[bool], bool]


@dataclass(frozen=True)
class PlasticDepthInlineProbeRequest:
    candidate_counts: Tuple[int, ...]
    sampled_token_indices: Optional[Tensor]
    selector: PlasticDepthCandidateSelector
    # vvv THOG optional CUDA-only split permits one recoverable N+1 layer while leaving the established path exact
    recoverable_upward_count: Optional[int] = None
    prepare_recoverable_upward: Optional[PlasticDepthUpwardPreparation] = None
    synchronize_recoverable_upward: Optional[PlasticDepthUpwardFeasibilitySynchronizer] = None
    # ^^^ THOG

    def validate(self, *, maximum_count: int) -> None:
        if not self.candidate_counts:
            raise ValueError("PLASTIC DEPTH inline probe requires at least one candidate")
        previous = 0
        for count in self.candidate_counts:
            if isinstance(count, bool) or not isinstance(count, int):
                raise ValueError(f"PLASTIC DEPTH candidate count must be an integer; got {count!r}")
            if count <= previous or count > maximum_count:
                raise ValueError(
                    "PLASTIC DEPTH candidate counts must be strictly increasing and bounded; "
                    f"counts={self.candidate_counts}, maximum_count={maximum_count}"
                )
            previous = count
        # vvv THOG a recoverable upward candidate is always the final adjacent prefix and requires both lifecycle callbacks
        callbacks = (self.prepare_recoverable_upward, self.synchronize_recoverable_upward)
        if self.recoverable_upward_count is None:
            if any(callback is not None for callback in callbacks):
                raise ValueError("PLASTIC DEPTH upward callbacks require recoverable_upward_count")
        else:
            if self.recoverable_upward_count != self.candidate_counts[-1]:
                raise ValueError("recoverable_upward_count must be the final candidate count")
            if len(self.candidate_counts) < 2 or self.recoverable_upward_count != self.candidate_counts[-2] + 1:
                raise ValueError("recoverable_upward_count must extend the lower prefix by exactly one layer")
            if any(callback is None for callback in callbacks):
                raise ValueError("recoverable_upward_count requires preparation and synchronization callbacks")
        # ^^^ THOG
        if self.sampled_token_indices is not None:
            if self.sampled_token_indices.ndim != 1:
                raise ValueError("sampled_token_indices must be one-dimensional")
            if self.sampled_token_indices.dtype != torch.long:
                raise ValueError(
                    "sampled_token_indices must use torch.int64; "
                    f"got {self.sampled_token_indices.dtype}"
                )


@dataclass(frozen=True)
class PlasticDepthInlineProbeReport:
    candidate_counts: Tuple[int, ...]
    local_candidate_losses: Tuple[float, ...]
    selected_count: int
    sampled_token_count: int


__all__ = [
    "PlasticDepthCandidateSelector",
    "PlasticDepthUpwardFeasibilitySynchronizer",
    "PlasticDepthUpwardPreparation",
    "PlasticDepthInlineProbeReport",
    "PlasticDepthInlineProbeRequest",
]
# ^^^ THOG
