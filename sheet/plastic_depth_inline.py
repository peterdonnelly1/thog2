# vvv THOG
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor


PlasticDepthCandidateSelector = Callable[[Tuple[Tuple[int, Tensor], ...]], int]
PlasticDepthUpwardPreparation = Callable[[], None]
PlasticDepthUpwardFeasibilitySynchronizer = Callable[[bool], bool]
PlasticDepthCandidateFeasibilitySynchronizer = Callable[[int, bool], bool]


@dataclass(frozen=True)
class PlasticDepthInlineProbeRequest:
    candidate_counts: Tuple[int, ...]
    sampled_token_indices: Optional[Tensor]
    selector: PlasticDepthCandidateSelector
    # vvv THOG retain the adjacent N+1 API for checkpoint compatibility while adding a recoverable contiguous upward suffix for full-radius probing
    recoverable_upward_count: Optional[int] = None
    prepare_recoverable_upward: Optional[PlasticDepthUpwardPreparation] = None
    synchronize_recoverable_upward: Optional[PlasticDepthUpwardFeasibilitySynchronizer] = None
    recoverable_upward_counts: Tuple[int, ...] = ()
    prepare_recoverable_upward_counts: Optional[PlasticDepthUpwardPreparation] = None
    synchronize_recoverable_upward_candidate: Optional[
        PlasticDepthCandidateFeasibilitySynchronizer
    ] = None
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
        # vvv THOG legacy one-candidate recovery and full-radius suffix recovery are mutually exclusive, contiguous and callback-complete
        legacy_callbacks = (
            self.prepare_recoverable_upward,
            self.synchronize_recoverable_upward,
        )
        suffix_callbacks = (
            self.prepare_recoverable_upward_counts,
            self.synchronize_recoverable_upward_candidate,
        )
        if self.recoverable_upward_count is not None and self.recoverable_upward_counts:
            raise ValueError(
                "legacy and full-radius PLASTIC upward recovery cannot both be enabled"
            )
        if self.recoverable_upward_count is None:
            if any(callback is not None for callback in legacy_callbacks):
                raise ValueError("PLASTIC DEPTH upward callbacks require recoverable_upward_count")
        else:
            if self.recoverable_upward_count != self.candidate_counts[-1]:
                raise ValueError("recoverable_upward_count must be the final candidate count")
            if len(self.candidate_counts) < 2 or self.recoverable_upward_count != self.candidate_counts[-2] + 1:
                raise ValueError("recoverable_upward_count must extend the lower prefix by exactly one layer")
            if any(callback is None for callback in legacy_callbacks):
                raise ValueError("recoverable_upward_count requires preparation and synchronization callbacks")
        if not self.recoverable_upward_counts:
            if any(callback is not None for callback in suffix_callbacks):
                raise ValueError(
                    "full-radius PLASTIC upward callbacks require recoverable_upward_counts"
                )
        else:
            suffix = tuple(int(value) for value in self.recoverable_upward_counts)
            if suffix != tuple(sorted(set(suffix))):
                raise ValueError(
                    "recoverable_upward_counts must be strictly increasing and unique"
                )
            if self.candidate_counts[-len(suffix):] != suffix:
                raise ValueError(
                    "recoverable_upward_counts must be the final candidate suffix"
                )
            lower = self.candidate_counts[:-len(suffix)]
            if not lower or suffix[0] != lower[-1] + 1:
                raise ValueError(
                    "recoverable_upward_counts must extend a retained lower prefix by one layer"
                )
            if any(right != left + 1 for left, right in zip(suffix, suffix[1:])):
                raise ValueError(
                    "recoverable_upward_counts must form a contiguous layer-count suffix"
                )
            if any(callback is None for callback in suffix_callbacks):
                raise ValueError(
                    "recoverable_upward_counts require preparation and per-candidate synchronization callbacks"
                )
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
    "PlasticDepthCandidateFeasibilitySynchronizer",
    "PlasticDepthCandidateSelector",
    "PlasticDepthUpwardFeasibilitySynchronizer",
    "PlasticDepthUpwardPreparation",
    "PlasticDepthInlineProbeReport",
    "PlasticDepthInlineProbeRequest",
]
# ^^^ THOG
