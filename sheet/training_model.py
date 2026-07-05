# vvv THOG
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from model import GPT
from .checkpointing import (
    CheckpointExecutionReport,
    execute_logical_layers,
    validate_checkpoint_segment_size,
)
from .model import SheetGPT, SheetGPTConfig


class TrainingDenseGPT(GPT):
    """nanoGPT dense control with the same segmented activation checkpointing API."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.checkpoint_segment_size = 0
        self.last_execution_report = CheckpointExecutionReport(
            checkpointing_used=False,
            checkpoint_segments=0,
            logical_layers=config.n_layer,
            segment_size=0,
        )

    def set_checkpoint_segment_size(self, segment_size: int) -> None:
        self.checkpoint_segment_size = validate_checkpoint_segment_size(segment_size)

    def _logical_block(self, hidden: Tensor, layer_index: int) -> Tensor:
        return self.transformer.h[layer_index](hidden)

    def forward(
        self,
        idx: Tensor,
        targets: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if idx.ndim != 2:
            raise ValueError(f"idx must have shape [batch, time]; got {tuple(idx.shape)}")
        _, sequence_length = idx.shape
        if sequence_length > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {sequence_length}; "
                f"block size is {self.config.block_size}"
            )
        positions = torch.arange(
            sequence_length,
            dtype=torch.long,
            device=idx.device,
        )
        token_embeddings = self.transformer.wte(idx)
        position_embeddings = self.transformer.wpe(positions)
        hidden = self.transformer.drop(token_embeddings + position_embeddings)
        hidden, self.last_execution_report = execute_logical_layers(
            hidden,
            n_layer=self.config.n_layer,
            segment_size=self.checkpoint_segment_size,
            logical_block=self._logical_block,
            training=self.training,
        )
        hidden = self.transformer.ln_f(hidden)

        if targets is not None:
            logits = self.lm_head(hidden)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            logits = self.lm_head(hidden[:, [-1], :])
            loss = None
        return logits, loss


class TrainingSheetGPT(SheetGPT):
    """SheetGPT with training-only segmented checkpoint recomputation."""

    def __init__(self, config: SheetGPTConfig) -> None:
        super().__init__(config)
        self.checkpoint_segment_size = 0
        self.last_execution_report = CheckpointExecutionReport(
            checkpointing_used=False,
            checkpoint_segments=0,
            logical_layers=config.n_layer,
            segment_size=0,
        )

    def set_checkpoint_segment_size(self, segment_size: int) -> None:
        self.checkpoint_segment_size = validate_checkpoint_segment_size(segment_size)

    def _sheet_layer_norm(
        self,
        inputs: Tensor,
        weight_name: str,
        bias_name: str,
        layer_index: int,
    ) -> Tensor:
        with torch.autocast(
            device_type=inputs.device.type,
            enabled=False,
        ):
            weight = self.trajectory.materialize_vector(
                weight_name,
                layer_index,
            ).float()
            bias = self._optional_bias(bias_name, layer_index)
            if bias is not None:
                bias = bias.float()
            return F.layer_norm(
                inputs,
                (self.config.n_embd,),
                weight,
                bias,
                1.0e-5,
            )

    def forward(
        self,
        idx: Tensor,
        targets: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if idx.ndim != 2:
            raise ValueError(f"idx must have shape [batch, time]; got {tuple(idx.shape)}")
        _, sequence_length = idx.shape
        if sequence_length > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {sequence_length}; "
                f"block size is {self.config.block_size}"
            )
        positions = torch.arange(
            sequence_length,
            dtype=torch.long,
            device=idx.device,
        )
        token_embeddings = self.transformer.wte(idx)
        position_embeddings = self.transformer.wpe(positions)
        hidden = self.transformer.drop(token_embeddings + position_embeddings)
        hidden, self.last_execution_report = execute_logical_layers(
            hidden,
            n_layer=self.config.n_layer,
            segment_size=self.checkpoint_segment_size,
            logical_block=self._logical_block,
            training=self.training,
        )
        hidden = self.transformer.ln_f(hidden)

        if targets is not None:
            logits = self.lm_head(hidden)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            logits = self.lm_head(hidden[:, [-1], :])
            loss = None
        return logits, loss


__all__ = ["TrainingDenseGPT", "TrainingSheetGPT"]
# ^^^ THOG
