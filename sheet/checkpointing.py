# vvv THOG
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint


LogicalBlock = Callable[[Tensor, int], Tensor]
# vvv THOG regional compilation hooks one cached compiled callable to each existing checkpoint segment
RegionalSegmentRunnerFactory = Callable[[Tuple[int, ...]], Callable[[Tensor], Tensor]]
# ^^^ THOG


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


# vvv THOG layer dropout preserves the existing all-layer path and adds sparse nominal-index execution

def _validate_layer_indices(layer_indices: Sequence[int], n_layer: int) -> Tuple[int, ...]:
    resolved = tuple(layer_indices)
    if not resolved:
        raise ValueError("layer_indices must not be empty")
    previous = -1
    for layer_index in resolved:
        if isinstance(layer_index, bool) or not isinstance(layer_index, int):
            raise ValueError(f"layer index must be an integer; got {layer_index!r}")
        if layer_index < 0 or layer_index >= n_layer:
            raise ValueError(f"layer index out of range: {layer_index}; n_layer={n_layer}")
        if layer_index <= previous:
            raise ValueError("layer_indices must be strictly increasing")
        previous = layer_index
    return resolved
# ^^^ THOG


def execute_logical_layers(
    hidden: Tensor,
    *,
    n_layer: int,
    segment_size: int,
    logical_block: LogicalBlock,
    training: bool,
    layer_indices: Optional[Sequence[int]] = None,                                                                                                         # <<< THOG optional sparse nominal execution sequence
    regional_segment_runner_factory: Optional[RegionalSegmentRunnerFactory] = None,
) -> Tuple[Tensor, CheckpointExecutionReport]:
    validate_checkpoint_segment_size(segment_size)
    if isinstance(n_layer, bool) or not isinstance(n_layer, int) or n_layer <= 0:
        raise ValueError(f"n_layer must be a positive integer; got {n_layer!r}")

    use_checkpointing = (
        training
        and torch.is_grad_enabled()
        and segment_size > 0
    )

    # vvv THOG preserve the pre-layer-dropout fast path byte-for-byte in its inner loops unless regional compilation explicitly supplies segment runners
    if layer_indices is None:
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

            if regional_segment_runner_factory is None:
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
            else:
                run_segment = regional_segment_runner_factory(tuple(range(start, end)))

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

    # vvv THOG sparse path chunks active execution positions while preserving nominal layer indices; regional mode deliberately rejects changing random subsets
    if regional_segment_runner_factory is not None:
        raise RuntimeError("regional torch compilation does not support layer dropout yet")
    active_layer_indices = _validate_layer_indices(layer_indices, n_layer)
    active_count = len(active_layer_indices)
    if not use_checkpointing:
        for layer_index in active_layer_indices:
            hidden = logical_block(hidden, layer_index)
        return hidden, CheckpointExecutionReport(
            checkpointing_used=False,
            checkpoint_segments=0,
            logical_layers=active_count,
            segment_size=segment_size,
        )

    checkpoint_segments = 0
    for start in range(0, active_count, segment_size):
        end = min(start + segment_size, active_count)
        segment_indices = active_layer_indices[start:end]

        def run_sparse_segment(
            segment_input: Tensor,
            *,
            nominal_indices: Tuple[int, ...] = segment_indices,
        ) -> Tensor:
            segment_output = segment_input
            for layer_index in nominal_indices:
                segment_output = logical_block(segment_output, layer_index)
            return segment_output

        hidden = checkpoint(
            run_sparse_segment,
            hidden,
            use_reentrant=False,
            preserve_rng_state=True,
        )
        checkpoint_segments += 1

    return hidden, CheckpointExecutionReport(
        checkpointing_used=True,
        checkpoint_segments=checkpoint_segments,
        logical_layers=active_count,
        segment_size=segment_size,
    )
    # ^^^ THOG


# vvv THOG PLASTIC DEPTH exposes selected prefix checkpoints without replaying the shared transformer chain
def execute_logical_layer_checkpoints(
    hidden: Tensor,
    *,
    n_layer: int,
    segment_size: int,
    logical_block: LogicalBlock,
    training: bool,
    layer_indices: Sequence[int],
    checkpoint_counts: Sequence[int],
) -> Tuple[Tuple[Tuple[int, Tensor], ...], CheckpointExecutionReport]:
    validate_checkpoint_segment_size(segment_size)
    active_layer_indices = _validate_layer_indices(layer_indices, n_layer)
    counts = tuple(int(value) for value in checkpoint_counts)
    if not counts:
        raise ValueError("checkpoint_counts must not be empty")
    previous = 0
    for count in counts:
        if count <= previous or count > len(active_layer_indices):
            raise ValueError(
                "checkpoint_counts must be strictly increasing and bounded by the active prefix; "
                f"counts={counts}, active_count={len(active_layer_indices)}"
            )
        previous = count
    if counts[-1] != len(active_layer_indices):
        raise ValueError(
            "the final checkpoint count must equal the executed prefix length; "
            f"final={counts[-1]}, active_count={len(active_layer_indices)}"
        )

    use_checkpointing = training and torch.is_grad_enabled() and segment_size > 0
    outputs = []
    checkpoint_segments = 0
    position = 0
    count_set = set(counts)
    while position < len(active_layer_indices):
        if use_checkpointing:
            regular_end = min(position + segment_size, len(active_layer_indices))
            candidate_boundaries = [count for count in counts if position < count <= regular_end]
            end = min(candidate_boundaries) if candidate_boundaries else regular_end
        else:
            end = position + 1
        segment_indices = active_layer_indices[position:end]

        def run_segment(
            segment_input: Tensor,
            *,
            nominal_indices: Tuple[int, ...] = segment_indices,
        ) -> Tensor:
            segment_output = segment_input
            for layer_index in nominal_indices:
                segment_output = logical_block(segment_output, layer_index)
            return segment_output

        if use_checkpointing:
            hidden = checkpoint(
                run_segment,
                hidden,
                use_reentrant=False,
                preserve_rng_state=True,
            )
            checkpoint_segments += 1
        else:
            hidden = run_segment(hidden)
        position = end
        if position in count_set:
            outputs.append((position, hidden))

    return tuple(outputs), CheckpointExecutionReport(
        checkpointing_used=use_checkpointing,
        checkpoint_segments=checkpoint_segments,
        logical_layers=len(active_layer_indices),
        segment_size=segment_size,
    )
# ^^^ THOG

# ^^^ THOG
