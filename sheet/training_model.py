# vvv THOG
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from model import GPT
from .checkpointing import (
    CheckpointExecutionReport,
    execute_logical_layer_checkpoints,
    execute_logical_layers,
    validate_checkpoint_segment_size,
)
from .model import SheetGPT, SheetGPTConfig
from .plastic_depth_cuda import is_cuda_out_of_memory
from .plastic_depth_inline import (
    PlasticDepthInlineProbeReport,
    PlasticDepthInlineProbeRequest,
)
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

    # vvv THOG CUDA learned-count probing executes N-1/N first, then treats only the adjacent N+1 allocation as recoverable
    def _plastic_depth_recoverable_probe_candidates(
        self,
        hidden: Tensor,
        targets: Tensor,
        request: PlasticDepthInlineProbeRequest,
    ) -> Tuple[Dict[int, Tensor], Tuple[Tuple[int, Tensor], ...], CheckpointExecutionReport]:
        upward_count = request.recoverable_upward_count
        prepare_upward = request.prepare_recoverable_upward
        synchronize_upward = request.synchronize_recoverable_upward
        if upward_count is None or prepare_upward is None or synchronize_upward is None:
            raise RuntimeError("PLASTIC DEPTH recoverable probe request is incomplete")
        lower_counts = tuple(count for count in request.candidate_counts if count != upward_count)
        if not lower_counts or upward_count != lower_counts[-1] + 1:
            raise RuntimeError("PLASTIC DEPTH recoverable N+1 candidate is not adjacent to the lower prefix")
        lower_checkpoints, lower_report = execute_logical_layer_checkpoints(
            hidden,
            n_layer=self.config.n_layer,
            segment_size=self.checkpoint_segment_size,
            logical_block=self._logical_block,
            training=self.training,
            layer_indices=tuple(range(lower_counts[-1])),
            checkpoint_counts=lower_counts,
        )
        checkpoint_by_count = dict(lower_checkpoints)
        candidate_losses = []
        with torch.no_grad():
            for count in lower_counts:
                candidate_loss = self._plastic_depth_candidate_head_loss(
                    checkpoint_by_count[count],
                    targets,
                    request.sampled_token_indices,
                )
                candidate_losses.append((count, candidate_loss.detach()))

        prepare_upward()
        upward_hidden: Optional[Tensor] = None
        upward_loss: Optional[Tensor] = None
        upward_report: Optional[CheckpointExecutionReport] = None
        local_feasible = True
        try:
            upward_hidden, upward_report = execute_logical_layers(
                checkpoint_by_count[lower_counts[-1]],
                n_layer=self.config.n_layer,
                segment_size=self.checkpoint_segment_size,
                logical_block=self._logical_block,
                training=self.training,
                layer_indices=(upward_count - 1,),
            )
            with torch.no_grad():
                upward_loss = self._plastic_depth_candidate_head_loss(
                    upward_hidden,
                    targets,
                    request.sampled_token_indices,
                ).detach()
        except BaseException as error:
            if not is_cuda_out_of_memory(error):
                raise
            local_feasible = False
            upward_hidden = None
            upward_loss = None
            upward_report = None
            if hidden.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        globally_feasible = bool(synchronize_upward(local_feasible))
        if globally_feasible:
            if upward_hidden is None or upward_loss is None or upward_report is None:
                raise RuntimeError("distributed PLASTIC DEPTH feasibility accepted a missing local N+1 candidate")
            checkpoint_by_count[upward_count] = upward_hidden
            candidate_losses.append((upward_count, upward_loss))
            execution_report = CheckpointExecutionReport(
                checkpointing_used=lower_report.checkpointing_used or upward_report.checkpointing_used,
                checkpoint_segments=lower_report.checkpoint_segments + upward_report.checkpoint_segments,
                logical_layers=lower_report.logical_layers + upward_report.logical_layers,
                segment_size=lower_report.segment_size,
            )
        else:
            upward_hidden = None
            upward_loss = None
            execution_report = lower_report
        return checkpoint_by_count, tuple(candidate_losses), execution_report
    # ^^^ THOG

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
        # vvv THOG PLASTIC DEPTH inline probing may pre-materialise one larger prefix for a complete optimiser update
        self._plastic_depth_update_layer_count: Optional[int] = None
        self.last_plastic_depth_inline_probe_report: Optional[PlasticDepthInlineProbeReport] = None
        # ^^^ THOG

    def set_checkpoint_segment_size(self, segment_size: int) -> None:
        self.checkpoint_segment_size = validate_checkpoint_segment_size(segment_size)

    # vvv THOG explicit optimiser-update lifetime for non-fast-discard materialisations
    def _optimizer_update_layer_indices(self) -> Tuple[int, ...]:
        # if self._active_layer_indices is None:
        if self._active_layer_indices is not None:
            return self._active_layer_indices
        # vvv THOG PLASTIC DEPTH active ranks apply to the complete optimiser update, including retained materialisations
        if self.plastic_depth_enabled:
            if self._plastic_depth_update_layer_count is not None:
                return tuple(range(self._plastic_depth_update_layer_count))
            return self.plastic_depth_active_layer_indices()
        # ^^^ THOG
        return tuple(range(self.config.n_layer))

    # vvv THOG transient execution capacity is update-scoped and never checkpointed
    def set_plastic_depth_update_layer_count(self, active_layers: Optional[int]) -> None:
        if active_layers is None:
            self._plastic_depth_update_layer_count = None
            return
        if not self.plastic_depth_enabled:
            raise RuntimeError("PLASTIC DEPTH is not enabled")
        lattice = self.trajectory.plastic_sampling
        if lattice is None:
            raise RuntimeError("enabled PLASTIC DEPTH has no sampling lattice")
        resolved_count = int(active_layers)
        if resolved_count < 1 or resolved_count > lattice.maximum_layers:
            raise ValueError(
                "PLASTIC DEPTH update layer count must lie within allocated capacity; "
                f"got active_layers={resolved_count}, maximum_layers={lattice.maximum_layers}"
            )
        self._plastic_depth_update_layer_count = resolved_count

    def clear_plastic_depth_update_layer_count(self) -> None:
        self._plastic_depth_update_layer_count = None
    # ^^^ THOG

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
            # vvv THOG retained PLASTIC DEPTH materialisations share one update-scoped differentiable basis
            if self.plastic_depth_enabled:
                self.trajectory.prepare_plastic_depth_basis_cache()
            # ^^^ THOG
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
        # return self._update_retained_materializations.finalize()
        try:
            return self._update_retained_materializations.finalize()
        finally:
            if self.plastic_depth_enabled:
                self.trajectory.clear_plastic_depth_basis_cache()

    def end_optimizer_update(self) -> bool:
        # return self._update_retained_materializations.end()
        try:
            return self._update_retained_materializations.end()
        finally:
            if self.plastic_depth_enabled:
                self.trajectory.clear_plastic_depth_basis_cache()

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

    # vvv THOG PLASTIC DEPTH evaluates detached candidate heads at shared prefix checkpoints, then one selected grad-bearing head
    def _plastic_depth_candidate_head_loss(
        self,
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
        return F.cross_entropy(logits, flattened_targets, ignore_index=-1)

    # vvv THOG CUDA learned-count probing executes N-1/N first, then treats only the adjacent N+1 allocation as recoverable
    def _plastic_depth_recoverable_probe_candidates(
        self,
        hidden: Tensor,
        targets: Tensor,
        request: PlasticDepthInlineProbeRequest,
    ) -> Tuple[Dict[int, Tensor], Tuple[Tuple[int, Tensor], ...], CheckpointExecutionReport]:
        upward_count = request.recoverable_upward_count
        prepare_upward = request.prepare_recoverable_upward
        synchronize_upward = request.synchronize_recoverable_upward
        if upward_count is None or prepare_upward is None or synchronize_upward is None:
            raise RuntimeError("PLASTIC DEPTH recoverable probe request is incomplete")
        lower_counts = tuple(count for count in request.candidate_counts if count != upward_count)
        if not lower_counts or upward_count != lower_counts[-1] + 1:
            raise RuntimeError("PLASTIC DEPTH recoverable N+1 candidate is not adjacent to the lower prefix")
        lower_checkpoints, lower_report = execute_logical_layer_checkpoints(
            hidden,
            n_layer=self.config.n_layer,
            segment_size=self.checkpoint_segment_size,
            logical_block=self._logical_block,
            training=self.training,
            layer_indices=tuple(range(lower_counts[-1])),
            checkpoint_counts=lower_counts,
        )
        checkpoint_by_count = dict(lower_checkpoints)
        candidate_losses = []
        with torch.no_grad():
            for count in lower_counts:
                candidate_loss = self._plastic_depth_candidate_head_loss(
                    checkpoint_by_count[count],
                    targets,
                    request.sampled_token_indices,
                )
                candidate_losses.append((count, candidate_loss.detach()))

        prepare_upward()
        upward_hidden: Optional[Tensor] = None
        upward_loss: Optional[Tensor] = None
        upward_report: Optional[CheckpointExecutionReport] = None
        local_feasible = True
        try:
            upward_hidden, upward_report = execute_logical_layers(
                checkpoint_by_count[lower_counts[-1]],
                n_layer=self.config.n_layer,
                segment_size=self.checkpoint_segment_size,
                logical_block=self._logical_block,
                training=self.training,
                layer_indices=(upward_count - 1,),
            )
            with torch.no_grad():
                upward_loss = self._plastic_depth_candidate_head_loss(
                    upward_hidden,
                    targets,
                    request.sampled_token_indices,
                ).detach()
        except BaseException as error:
            if not is_cuda_out_of_memory(error):
                raise
            local_feasible = False
            upward_hidden = None
            upward_loss = None
            upward_report = None
            if hidden.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        globally_feasible = bool(synchronize_upward(local_feasible))
        if globally_feasible:
            if upward_hidden is None or upward_loss is None or upward_report is None:
                raise RuntimeError("distributed PLASTIC DEPTH feasibility accepted a missing local N+1 candidate")
            checkpoint_by_count[upward_count] = upward_hidden
            candidate_losses.append((upward_count, upward_loss))
            execution_report = CheckpointExecutionReport(
                checkpointing_used=lower_report.checkpointing_used or upward_report.checkpointing_used,
                checkpoint_segments=lower_report.checkpoint_segments + upward_report.checkpoint_segments,
                logical_layers=lower_report.logical_layers + upward_report.logical_layers,
                segment_size=lower_report.segment_size,
            )
        else:
            upward_hidden = None
            upward_loss = None
            execution_report = lower_report
        return checkpoint_by_count, tuple(candidate_losses), execution_report
    # ^^^ THOG

    # vvv THOG full-radius CUDA probing retains every completed lower candidate when one upward suffix candidate OOMs
    def _plastic_depth_recoverable_probe_candidate_suffix(
        self,
        hidden: Tensor,
        targets: Tensor,
        request: PlasticDepthInlineProbeRequest,
    ) -> Tuple[Dict[int, Tensor], Tuple[Tuple[int, Tensor], ...], CheckpointExecutionReport]:
        upward_counts = tuple(int(value) for value in request.recoverable_upward_counts)
        prepare_upward = request.prepare_recoverable_upward_counts
        synchronize_candidate = request.synchronize_recoverable_upward_candidate
        if not upward_counts or prepare_upward is None or synchronize_candidate is None:
            raise RuntimeError("PLASTIC DEPTH full-radius recoverable probe request is incomplete")
        lower_counts = tuple(
            count for count in request.candidate_counts if count not in upward_counts
        )
        if not lower_counts or upward_counts[0] != lower_counts[-1] + 1:
            raise RuntimeError(
                "PLASTIC DEPTH recoverable upward suffix is not adjacent to the retained lower prefix"
            )
        lower_checkpoints, lower_report = execute_logical_layer_checkpoints(
            hidden,
            n_layer=self.config.n_layer,
            segment_size=self.checkpoint_segment_size,
            logical_block=self._logical_block,
            training=self.training,
            layer_indices=tuple(range(lower_counts[-1])),
            checkpoint_counts=lower_counts,
        )
        checkpoint_by_count = dict(lower_checkpoints)
        candidate_losses = []
        with torch.no_grad():
            for count in lower_counts:
                candidate_loss = self._plastic_depth_candidate_head_loss(
                    checkpoint_by_count[count],
                    targets,
                    request.sampled_token_indices,
                )
                candidate_losses.append((count, candidate_loss.detach()))

        prepare_upward()
        execution_report = lower_report
        for upward_count in upward_counts:
            prior_count = upward_count - 1
            prior_hidden = checkpoint_by_count.get(prior_count)
            if prior_hidden is None:
                raise RuntimeError(
                    "PLASTIC DEPTH upward suffix lost its preceding checkpoint; "
                    f"candidate={upward_count}, available={tuple(checkpoint_by_count)}"
                )
            upward_hidden: Optional[Tensor] = None
            upward_loss: Optional[Tensor] = None
            upward_report: Optional[CheckpointExecutionReport] = None
            local_feasible = True
            try:
                upward_hidden, upward_report = execute_logical_layers(
                    prior_hidden,
                    n_layer=self.config.n_layer,
                    segment_size=self.checkpoint_segment_size,
                    logical_block=self._logical_block,
                    training=self.training,
                    layer_indices=(upward_count - 1,),
                )
                with torch.no_grad():
                    upward_loss = self._plastic_depth_candidate_head_loss(
                        upward_hidden,
                        targets,
                        request.sampled_token_indices,
                    ).detach()
            except BaseException as error:
                if not is_cuda_out_of_memory(error):
                    raise
                local_feasible = False
                upward_hidden = None
                upward_loss = None
                upward_report = None
                if hidden.device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            globally_feasible = bool(
                synchronize_candidate(upward_count, local_feasible)
            )
            if not globally_feasible:
                upward_hidden = None
                upward_loss = None
                upward_report = None
                if hidden.device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break
            if upward_hidden is None or upward_loss is None or upward_report is None:
                raise RuntimeError(
                    "distributed PLASTIC DEPTH feasibility accepted a missing upward candidate"
                )
            checkpoint_by_count[upward_count] = upward_hidden
            candidate_losses.append((upward_count, upward_loss))
            execution_report = CheckpointExecutionReport(
                checkpointing_used=(
                    execution_report.checkpointing_used
                    or upward_report.checkpointing_used
                ),
                checkpoint_segments=(
                    execution_report.checkpoint_segments
                    + upward_report.checkpoint_segments
                ),
                logical_layers=(
                    execution_report.logical_layers
                    + upward_report.logical_layers
                ),
                segment_size=execution_report.segment_size,
            )
        return checkpoint_by_count, tuple(candidate_losses), execution_report
    # ^^^ THOG

    def forward(
        self,
        idx: Tensor,
        targets: Optional[Tensor] = None,
        *,
        plastic_depth_probe_request: Optional[PlasticDepthInlineProbeRequest] = None,
        plastic_depth_active_layers_override: Optional[int] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if idx.ndim != 2:
            raise ValueError(f"idx must have shape [batch, time]; got {tuple(idx.shape)}")
        _, sequence_length = idx.shape
        if sequence_length > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {sequence_length}; "
                f"block size is {self.config.block_size}"
            )
        if plastic_depth_probe_request is not None and plastic_depth_active_layers_override is not None:
            raise ValueError("PLASTIC DEPTH probe and active-layer override are mutually exclusive")
        if (plastic_depth_probe_request is not None or plastic_depth_active_layers_override is not None) and not self.plastic_depth_enabled:
            raise RuntimeError("PLASTIC DEPTH execution controls require PLASTIC DEPTH")
        if plastic_depth_probe_request is not None and targets is None:
            raise ValueError("PLASTIC DEPTH inline probing requires targets")
        if self._active_layer_indices is not None and plastic_depth_probe_request is not None:
            raise RuntimeError("PLASTIC DEPTH inline probing cannot be combined with layer dropout")

        # vvv THOG fast-discard forwards own one basis each; retained-materialisation updates prepare one basis before all microbatches
        if self.plastic_depth_enabled and not self._update_retained_materializations.active:
            self.trajectory.prepare_plastic_depth_basis_cache()
        # ^^^ THOG
        positions = torch.arange(sequence_length, dtype=torch.long, device=idx.device)
        token_embeddings = self.transformer.wte(idx)
        position_embeddings = self.transformer.wpe(positions)
        hidden = self.transformer.drop(token_embeddings + position_embeddings)

        if plastic_depth_probe_request is not None:
            if not self.training or not torch.is_grad_enabled():
                raise RuntimeError("PLASTIC DEPTH inline probing requires a grad-enabled training forward")
            if self._torch_compile_mode == "regional":
                raise RuntimeError("regional torch.compile does not support PLASTIC DEPTH inline probing")
            plastic_depth_probe_request.validate(maximum_count=self.config.n_layer)
            # vvv THOG preserve the established path unless the trainer explicitly supplies recoverable CUDA upward candidates
            if plastic_depth_probe_request.recoverable_upward_counts:
                checkpoint_by_count, candidate_losses, self.last_execution_report = (
                    self._plastic_depth_recoverable_probe_candidate_suffix(
                        hidden,
                        targets,
                        plastic_depth_probe_request,
                    )
                )
                checkpoints = tuple(checkpoint_by_count.items())
            elif plastic_depth_probe_request.recoverable_upward_count is None:
                maximum_count = plastic_depth_probe_request.candidate_counts[-1]
                checkpoints, self.last_execution_report = execute_logical_layer_checkpoints(
                    hidden,
                    n_layer=self.config.n_layer,
                    segment_size=self.checkpoint_segment_size,
                    logical_block=self._logical_block,
                    training=self.training,
                    layer_indices=tuple(range(maximum_count)),
                    checkpoint_counts=plastic_depth_probe_request.candidate_counts,
                )
                checkpoint_by_count = dict(checkpoints)
                candidate_losses = []
                with torch.no_grad():
                    for count in plastic_depth_probe_request.candidate_counts:
                        candidate_loss = self._plastic_depth_candidate_head_loss(
                            checkpoint_by_count[count],
                            targets,
                            plastic_depth_probe_request.sampled_token_indices,
                        )
                        candidate_losses.append((count, candidate_loss.detach()))
            else:
                checkpoint_by_count, candidate_losses, self.last_execution_report = (
                    self._plastic_depth_recoverable_probe_candidates(
                        hidden,
                        targets,
                        plastic_depth_probe_request,
                    )
                )
                checkpoints = tuple(checkpoint_by_count.items())
            # ^^^ THOG
            selected_count = int(
                plastic_depth_probe_request.selector(tuple(candidate_losses))
            )
            if selected_count not in checkpoint_by_count:
                raise RuntimeError(
                    "PLASTIC DEPTH inline selector returned a non-candidate count; "
                    f"selected={selected_count}, candidates={plastic_depth_probe_request.candidate_counts}"
                )
            hidden = checkpoint_by_count[selected_count]
            local_losses = tuple(float(loss.item()) for _, loss in candidate_losses)
            sampled_count = (
                int(targets.numel())
                if plastic_depth_probe_request.sampled_token_indices is None
                else int(plastic_depth_probe_request.sampled_token_indices.numel())
            )
            self.last_plastic_depth_inline_probe_report = PlasticDepthInlineProbeReport(
                # candidate_counts=plastic_depth_probe_request.candidate_counts,
                candidate_counts=tuple(count for count, _ in candidate_losses),
                local_candidate_losses=local_losses,
                selected_count=selected_count,
                sampled_token_count=sampled_count,
            )
            del checkpoint_by_count, checkpoints, candidate_losses
        else:
            self.last_plastic_depth_inline_probe_report = None
            # vvv THOG PLASTIC DEPTH sampling is canonical in training, validation and generation; layer dropout remains training-only
            if plastic_depth_active_layers_override is not None:
                lattice = self.trajectory.plastic_sampling
                if lattice is None:
                    raise RuntimeError("enabled PLASTIC DEPTH has no sampling lattice")
                resolved_override = int(plastic_depth_active_layers_override)
                if resolved_override < 1 or resolved_override > lattice.maximum_layers:
                    raise ValueError(
                        "PLASTIC DEPTH active-layer override must lie within allocated capacity; "
                        f"got active_layers={resolved_override}, maximum_layers={lattice.maximum_layers}"
                    )
                layer_indices: Optional[Tuple[int, ...]] = tuple(range(resolved_override))
            elif self._active_layer_indices is not None and self.training and torch.is_grad_enabled():
                layer_indices = self._active_layer_indices
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
        # vvv THOG no-gradient evaluation and generation do not need to retain the forward-scoped PLASTIC DEPTH basis
        if self.plastic_depth_enabled and not torch.is_grad_enabled():
            self.trajectory.clear_plastic_depth_basis_cache()
        # ^^^ THOG
        return logits, loss
    # ^^^ THOG

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
