# vvv THOG
"""Candidate-local CUDA OOM recovery for the contiguous PLASTIC FINE upward suffix."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch

from . import trainer_step as _trainer_step
from .plastic_depth_cuda import PlasticDepthCudaAllocatorReserve
from .plastic_depth_inline import PlasticDepthInlineProbeRequest


_ORIGINAL_BEGIN_INLINE_UPDATE = _trainer_step.TrainerStepMixin._begin_plastic_depth_inline_update
_ORIGINAL_INLINE_PROBE_REQUEST = _trainer_step.TrainerStepMixin._plastic_depth_inline_probe_request


def _release_cuda_allocator_reserve(
    context: Dict[str, Any],
    *,
    empty_cache: bool,
) -> None:
    reserve = context.get("cuda_allocator_reserve")
    release = getattr(reserve, "release", None)
    if callable(release):
        release(empty_cache=empty_cache)
    context["cuda_allocator_reserve"] = None


def _begin_plastic_depth_inline_update_with_full_radius_oom(
    self: Any,
) -> Optional[Dict[str, Any]]:
    context = _ORIGINAL_BEGIN_INLINE_UPDATE(self)
    if context is None or self.device.type != "cuda":
        return context
    current_count = int(context["current_count"])
    upward_counts = tuple(
        int(count)
        for count in context["candidate_counts"]
        if int(count) > current_count
    )
    if not upward_counts:
        context["recoverable_upward_counts"] = ()
        return context
    # vvv THOG preserve the established adjacent N+1 path exactly when full-radius probing produces only one upward candidate
    if len(upward_counts) == 1:
        context["recoverable_upward_counts"] = ()
        return context
    # ^^^ THOG

    reserve = context.get("cuda_allocator_reserve")
    if reserve is None:
        reserve = PlasticDepthCudaAllocatorReserve(
            device=self.device,
            reserve_gib=float(self.config.plastic__cuda_allocator_reserve_gib),
        )
        local_preflight_feasible = reserve.acquire()
        globally_preflight_feasible = self.distributed.all_true(
            local_preflight_feasible
        )
        if not globally_preflight_feasible:
            reserve.release(empty_cache=True)
            retained_counts = tuple(
                int(count)
                for count in context["candidate_counts"]
                if int(count) <= current_count
            )
            if not retained_counts:
                raise RuntimeError(
                    "PLASTIC FINE CUDA preflight removed the current layer count"
                )
            context["candidate_counts"] = retained_counts
            context["decision_candidate_counts"] = tuple(
                int(count)
                for count in context["decision_candidate_counts"]
                if int(count) <= current_count
            )
            setter = getattr(
                self.raw_model,
                "set_plastic_depth_update_layer_count",
                None,
            )
            if not callable(setter):
                raise RuntimeError(
                    "PLASTIC DEPTH training model lacks update-prefix control"
                )
            setter(max(retained_counts))
            context["recoverable_upward_counts"] = ()
            context["recoverable_upward_count"] = None
            context["cuda_allocator_reserve"] = None
            context["upward_preflight_feasible"] = False
            context["upward_candidate_feasible"] = False
            return context
        context["cuda_allocator_reserve"] = reserve
        context["upward_preflight_feasible"] = True

    context["recoverable_upward_counts"] = upward_counts
    context["recoverable_upward_count"] = None
    context["upward_candidate_feasible_by_count"] = {
        count: None for count in upward_counts
    }
    return context


def _plastic_depth_inline_probe_request_with_full_radius_oom(
    self: Any,
    targets: torch.Tensor,
    context: Dict[str, Any],
) -> PlasticDepthInlineProbeRequest:
    request = _ORIGINAL_INLINE_PROBE_REQUEST(self, targets, context)
    upward_counts = tuple(
        int(value) for value in context.get("recoverable_upward_counts", ())
    )
    if not upward_counts:
        return request

    # vvv THOG keep the configured allocator reserve live while upward candidates execute so reserve means hard headroom, not temporary preflight memory
    def prepare_upward_counts() -> None:
        reserve = context.get("cuda_allocator_reserve")
        if reserve is None or not bool(getattr(reserve, "active", False)):
            raise RuntimeError(
                "PLASTIC DEPTH upward-radius probe lost its CUDA allocator reserve before candidate execution"
            )
    # ^^^ THOG

    def synchronize_candidate(count: int, local_feasible: bool) -> bool:
        candidate_count = int(count)
        globally_feasible = self.distributed.all_true(bool(local_feasible))
        context["upward_candidate_feasible_by_count"][candidate_count] = (
            globally_feasible
        )
        if not globally_feasible:
            context["candidate_counts"] = tuple(
                int(value)
                for value in context["candidate_counts"]
                if int(value) < candidate_count
            )
            context["decision_candidate_counts"] = tuple(
                int(value)
                for value in context["decision_candidate_counts"]
                if int(value) < candidate_count
            )
            for higher_count in upward_counts:
                if higher_count > candidate_count:
                    context["upward_candidate_feasible_by_count"][higher_count] = False
            # vvv THOG a failed candidate ends upward exploration; release the barrier only after all ranks reject the candidate and before cleanup
            _release_cuda_allocator_reserve(context, empty_cache=True)
            # ^^^ THOG
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return globally_feasible

    # vvv THOG the reserve protects upward evidence only; if the selector stays or shrinks, release it before the real gradient-bearing update at the already-safe count
    def select_with_reserve_lifetime(candidates):
        selected_count = int(request.selector(candidates))
        current_count = int(context["current_count"])
        if selected_count <= current_count:
            _release_cuda_allocator_reserve(context, empty_cache=True)
        return selected_count
    # ^^^ THOG

    return PlasticDepthInlineProbeRequest(
        candidate_counts=request.candidate_counts,
        sampled_token_indices=request.sampled_token_indices,
        selector=select_with_reserve_lifetime,
        recoverable_upward_counts=upward_counts,
        prepare_recoverable_upward_counts=prepare_upward_counts,
        synchronize_recoverable_upward_candidate=synchronize_candidate,
    )


_trainer_step.TrainerStepMixin._begin_plastic_depth_inline_update = (
    _begin_plastic_depth_inline_update_with_full_radius_oom
)
_trainer_step.TrainerStepMixin._plastic_depth_inline_probe_request = (
    _plastic_depth_inline_probe_request_with_full_radius_oom
)
# ^^^ THOG