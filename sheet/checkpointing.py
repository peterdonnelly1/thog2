# vvv THOG
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint


LogicalBlock = Callable[[Tensor, int], Tensor]


@dataclass(frozen=True)
class CheckpointExecutionReport:
    checkpointing_used: bool
    checkpoint_segments: int
    logical_layers: int
    segment_size: int


def validate_checkpoint_segment_size(segment_size: int) -> int:
    if isinstance(segment_size, bool) or not isinstance(segment_size, int):
        raise ValueError(
            "checkpoint_segment_size must be a non-negative integer; "
            f"got {segment_size!r}"
        )
    if segment_size < 0:
        raise ValueError(
            "checkpoint_segment_size must be a non-negative integer; "
            f"got {segment_size}"
        )
    return segment_size


def execute_logical_layers(
    hidden: Tensor,
    *,
    n_layer: int,
    segment_size: int,
    logical_block: LogicalBlock,
    training: bool,
) -> Tuple[Tensor, CheckpointExecutionReport]:
    validate_checkpoint_segment_size(segment_size)
    if isinstance(n_layer, bool) or not isinstance(n_layer, int) or n_layer <= 0:
        raise ValueError(f"n_layer must be a positive integer; got {n_layer!r}")

    use_checkpointing = (
        training
        and torch.is_grad_enabled()
        and segment_size > 0
    )
    if not use_checkpointing:
        for layer_index in range(n_layer):
            hidden = logical_block(hidden, layer_index)
        return hidden, CheckpointExecutionReport(
            checkpointing_used=False,
            checkpoint_segments=0,
            logical_layers=n_layer,
            segment_size=segment_size,
        )

    checkpoint_segments = 0
    for start in range(0, n_layer, segment_size):
        end = min(start + segment_size, n_layer)

        def run_segment(
            segment_input: Tensor,
            *,
            segment_start: int = start,
            segment_end: int = end,
        ) -> Tensor:
            segment_output = segment_input
            for layer_index in range(segment_start, segment_end):
                segment_output = logical_block(segment_output, layer_index)
            return segment_output

        hidden = checkpoint(
            run_segment,
            hidden,
            use_reentrant=False,
            preserve_rng_state=True,
        )
        checkpoint_segments += 1

    return hidden, CheckpointExecutionReport(
        checkpointing_used=True,
        checkpoint_segments=checkpoint_segments,
        logical_layers=n_layer,
        segment_size=segment_size,
    )
# ^^^ THOG
