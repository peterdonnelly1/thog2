# vvv THOG
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

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
from .update_retained_materializations import attach_update_retained_materializations


# vvv THOG regional compilation uses the existing activation-checkpoint segmentation as the compilation region boundary
def _validate_torch_compile_mode(mode: str) -> str:
    if mode not in {"false", "true", "regional"}:
        raise ValueError(f"torch compile mode must be false, true, or regional; got {mode!r}")
    return mode


def _compiled_segment_runner(
    logical_block: Callable[[Tensor, int], Tensor],
    layer_indices: Tuple[int, ...],
) -> Callable[[Tensor], Tensor]:
    compile_function = getattr(torch, "compile", None)
    if compile_function is None:
        raise RuntimeError("regional torch compilation requires torch.compile support")

    def run_segment(
        segment_input: Tensor,
        *,
        nominal_indices: Tuple[int, ...] = layer_indices,
    ) -> Tensor:
        segment_output = segment_input
        for layer_index in nominal_indices:
            segment_output = logical_block(segment_output, layer_index)
        return segment_output

    return compile_function(run_segment)
# ^^^ THOG


class TrainingDenseGPT(GPT):
    """nanoGPT dense control with the same segmented activation checkpointing API."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.checkpoint_segment_size = 0
        # vvv THOG training-only sparse nominal layer selection; None preserves the original path
        self._active_layer_indices: Optional[Tuple[int, ...]] = None
        # ^^^ THOG
        # vvv THOG regional torch.compile state is execution-only and excluded from parameters/checkpoints
        self._torch_compile_mode = "false"
        self._regional_segment_runners: Dict[Tuple[int, ...], Callable[[Tensor], Tensor]] = {}
        # ^^^ THOG
        self.last_execution_report = CheckpointExecutionReport(
            checkpointing_used=False,
            checkpoint_segments=0,
            logical_layers=config.n_layer,
            segment_size=0,
        )

    def set_checkpoint_segment_size(self, segment_size: int) -> None:
        self.checkpoint_segment_size = validate_checkpoint_segment_size(segment_size)

    # vvv THOG layer-dropout selection is external trainer state, not model parameters
    def set_active_layer_indices(self, layer_indices: Optional[Sequence[int]]) -> None:
        self._active_layer_indices = None if layer_indices is None else tuple(int(value) for value in layer_indices)
    # ^^^ THOG

    # vvv THOG regional torch.compile keeps the outer model, vocabulary head, loss, and checkpoint machinery eager
    def set_torch_compile_mode(self, mode: str) -> None:
        resolved_mode = _validate_torch_compile_mode(mode)
        if resolved_mode == "regional" and self.checkpoint_segment_size <= 0:
            raise ValueError("regional torch compilation requires checkpoint_segment_size > 0")
        self._torch_compile_mode = resolved_mode
        self._regional_segment_runners.clear()

    def _regional_segment_runner(self, layer_indices: Tuple[int, ...]) -> Callable[[Tensor], Tensor]:
        runner = self._regional_segment_runners.get(layer_indices)
        if runner is None:
            runner = _compiled_segment_runner(self._logical_block, layer_indices)
            self._regional_segment_runners[layer_indices] = runner
        return runner
    # ^^^ THOG

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
        positions = torch.arange(sequence_length, dtype=torch.long, device=idx.device)
        token_embeddings = self.transformer.wte(idx)
        position_embeddings = self.transformer.wpe(positions)
        hidden = self.transformer.drop(token_embeddings + position_embeddings)
        # vvv THOG evaluation/generation and the all-active case take the unchanged executor call
        layer_indices = self._active_layer_indices if self.training and torch.is_grad_enabled() else None
        regional_segment_runner_factory = self._regional_segment_runner if self._torch_compile_mode == "regional" else None
        if layer_indices is None:
            hidden, self.last_execution_report = execute_logical_layers(
                hidden,
                n_layer=self.config.n_layer,
                segment_size=self.checkpoint_segment_size,
                logical_block=self._logical_block,
                training=self.training,
                regional_segment_runner_factory=regional_segment_runner_factory,
            )
        else:
            hidden, self.last_execution_report = execute_logical_layers(
                hidden,
                n_layer=self.config.n_layer,
                segment_size=self.checkpoint_segment_size,
                logical_block=self._logical_block,
                training=self.training,
                layer_indices=layer_indices,
                regional_segment_runner_factory=regional_segment_runner_factory,
            )
        # ^^^ THOG
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
        # vvv THOG fast_discard=false retains one operational materialisation per active layer and family for an optimiser update
        self._update_retained_materializations = attach_update_retained_materializations(
            self.trajectory,
            enabled=not config.fast_discard,
        )
        # ^^^ THOG
        self.checkpoint_segment_size = 0
        # vvv THOG training-only sparse nominal layer selection; None preserves the original path
        self._active_layer_indices: Optional[Tuple[int, ...]] = None
        # ^^^ THOG
        # vvv THOG regional torch.compile state is execution-only and excluded from parameters/checkpoints
        self._torch_compile_mode = "false"
        self._regional_segment_runners: Dict[Tuple[int, ...], Callable[[Tensor], Tensor]] = {}
        # ^^^ THOG
        self.last_execution_report = CheckpointExecutionReport(
            checkpointing_used=False,
            checkpoint_segments=0,
            logical_layers=config.n_layer,
            segment_size=0,
        )

    def set_checkpoint_segment_size(self, segment_size: int) -> None:
        self.checkpoint_segment_size = validate_checkpoint_segment_size(segment_size)

    # vvv THOG explicit optimiser-update lifetime for non-fast-discard materialisations
    def _optimizer_update_layer_indices(self) -> Tuple[int, ...]:
        # if self._active_layer_indices is None:
        if self._active_layer_indices is not None:
            return self._active_layer_indices
        # vvv THOG PLASTIC DEPTH active ranks apply to the complete optimiser update, including retained materialisations
        if self.plastic_depth_enabled:
            return self.plastic_depth_active_layer_indices()
        # ^^^ THOG
        return tuple(range(self.config.n_layer))

    def _materialize_layer_norm_parameters_for_update(
        self,
        layer_indices: Tuple[int, ...],
    ) -> None:
        parameter = next(self.parameters())
        with torch.autocast(device_type=parameter.device.type, enabled=False):
            for layer_index in layer_indices:
                self.trajectory.materialize_vector("ln_1_weight", layer_index)
                self.trajectory.materialize_vector("ln_2_weight", layer_index)
                if self.config.bias:
                    self.trajectory.materialize_vector("ln_1_bias", layer_index)
                    self.trajectory.materialize_vector("ln_2_bias", layer_index)

    def _materialize_block_parameters_for_update(
        self,
        layer_indices: Tuple[int, ...],
    ) -> None:
        # vvv THOG preserve the legacy direct-factorised predicate and extend only the retained warmup skip decision
        # direct_application = (
        #     self.config.direct_factorised_mlp
        #     and self._supports_direct_factorised_mlp()
        # )
        direct_application = (
            (
                self.config.direct_factorised_mlp
                and self._supports_direct_factorised_mlp()
            )
            or self.config.direct_factorised_hyperblock_mlp
        )
        # ^^^ THOG
        for layer_index in layer_indices:
            if self.config.bypass_semantic_qkv_adapter:
                self.trajectory.materialize("attention_input_weight", layer_index)
                if self.config.bias:
                    self.trajectory.materialize_vector(
                        "attention_input_bias",
                        layer_index,
                    )
            else:
                self.semantic_materializer.reconstructed_attention_input_weight(
                    layer_index
                )
                if self.config.bias:
                    self.semantic_materializer.reconstructed_attention_input_bias(
                        layer_index
                    )
            self.trajectory.materialize("attention_output_weight", layer_index)
            if self.config.bias:
                self.trajectory.materialize_vector(
                    "attention_output_bias",
                    layer_index,
                )
                self.trajectory.materialize_vector(
                    "mlp_expansion_bias",
                    layer_index,
                )
                self.trajectory.materialize_vector(
                    "mlp_contraction_bias",
                    layer_index,
                )
            if not direct_application:
                self.trajectory.materialize("mlp_expansion_weight", layer_index)
                self.trajectory.materialize("mlp_contraction_weight", layer_index)

    def begin_optimizer_update(self) -> bool:
        controller = self._update_retained_materializations
        if not controller.begin():
            return False
        try:
            if not torch.is_grad_enabled():
                raise RuntimeError(
                    "update-retained materialisations require gradient tracking"
                )
            layer_indices = self._optimizer_update_layer_indices()
            self._materialize_layer_norm_parameters_for_update(layer_indices)
            self._materialize_block_parameters_for_update(layer_indices)
        except BaseException:
            controller.end()
            raise
        return True

    def finalize_optimizer_update(self) -> Tuple[Tensor, ...]:
        return self._update_retained_materializations.finalize()

    def end_optimizer_update(self) -> bool:
        return self._update_retained_materializations.end()

    def requires_find_unused_parameters(self) -> bool:
        return self._update_retained_materializations.enabled

    def update_retained_materialization_report(self) -> Dict[str, object]:
        controller = self._update_retained_materializations
        return {
            "enabled": controller.enabled,
            "active": controller.active,
            "retained_count": controller.retained_count,
            "request_count": controller.request_count,
            "materialization_count": controller.materialization_count,
        }
    # ^^^ THOG

    # vvv THOG layer-dropout selection is external trainer state, not compact model state
    def set_active_layer_indices(self, layer_indices: Optional[Sequence[int]]) -> None:
        self._active_layer_indices = None if layer_indices is None else tuple(int(value) for value in layer_indices)
    # ^^^ THOG

    # vvv THOG regional torch.compile keeps the outer model, vocabulary head, loss, and checkpoint machinery eager
    def set_torch_compile_mode(self, mode: str) -> None:
        resolved_mode = _validate_torch_compile_mode(mode)
        if resolved_mode == "regional" and self.checkpoint_segment_size <= 0:
            raise ValueError("regional torch compilation requires checkpoint_segment_size > 0")
        self._torch_compile_mode = resolved_mode
        self._regional_segment_runners.clear()

    def _regional_segment_runner(self, layer_indices: Tuple[int, ...]) -> Callable[[Tensor], Tensor]:
        runner = self._regional_segment_runners.get(layer_indices)
        if runner is None:
            runner = _compiled_segment_runner(self._logical_block, layer_indices)
            self._regional_segment_runners[layer_indices] = runner
        return runner
    # ^^^ THOG

    def _sheet_layer_norm(
        self,
        inputs: Tensor,
        weight_name: str,
        bias_name: str,
        layer_index: int,
    ) -> Tensor:
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            weight = self.trajectory.materialize_vector(weight_name, layer_index).float()
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
        positions = torch.arange(sequence_length, dtype=torch.long, device=idx.device)
        token_embeddings = self.transformer.wte(idx)
        position_embeddings = self.transformer.wpe(positions)
        hidden = self.transformer.drop(token_embeddings + position_embeddings)
        # vvv THOG PLASTIC DEPTH sampling is canonical in training, validation and generation; layer dropout remains training-only
        if self._active_layer_indices is not None and self.training and torch.is_grad_enabled():
            layer_indices: Optional[Tuple[int, ...]] = self._active_layer_indices
        elif self.plastic_depth_enabled:
            plastic_layer_indices = self.plastic_depth_active_layer_indices()
            full_layer_indices = tuple(range(self.config.n_layer))
            layer_indices = None if plastic_layer_indices == full_layer_indices else plastic_layer_indices
        else:
            layer_indices = None
        if self._torch_compile_mode == "regional" and layer_indices is not None:
            raise RuntimeError(
                "regional torch.compile does not support a PLASTIC DEPTH active count below the persistent maximum"
            )
        regional_segment_runner_factory = self._regional_segment_runner if self._torch_compile_mode == "regional" else None
        if layer_indices is None:
            hidden, self.last_execution_report = execute_logical_layers(
                hidden,
                n_layer=self.config.n_layer,
                segment_size=self.checkpoint_segment_size,
                logical_block=self._logical_block,
                training=self.training,
                regional_segment_runner_factory=regional_segment_runner_factory,
            )
        else:
            hidden, self.last_execution_report = execute_logical_layers(
                hidden,
                n_layer=self.config.n_layer,
                segment_size=self.checkpoint_segment_size,
                logical_block=self._logical_block,
                training=self.training,
                layer_indices=layer_indices,
                regional_segment_runner_factory=regional_segment_runner_factory,
            )
        # ^^^ THOG
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

    # vvv THOG
    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tensor:
        """Use the unchanged nanoGPT autoregressive sampling contract."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("Inf")
            probabilities = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    # ^^^ THOG


__all__ = ["TrainingDenseGPT", "TrainingSheetGPT"]
# ^^^ THOG
