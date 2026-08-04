# vvv THOG
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor


PlasticDepthCandidateSelector = Callable[[Tuple[Tuple[int, Tensor], ...]], int]


@dataclass(frozen=True)
class PlasticDepthInlineProbeRequest:
    candidate_counts: Tuple[int, ...]
    sampled_token_indices: Optional[Tensor]
    selector: PlasticDepthCandidateSelector

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
    "PlasticDepthInlineProbeReport",
    "PlasticDepthInlineProbeRequest",
]
# ^^^ THOG
