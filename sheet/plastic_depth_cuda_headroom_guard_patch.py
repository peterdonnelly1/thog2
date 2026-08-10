# vvv THOG
"""Enforce PLASTIC CUDA growth headroom with a real grad-bearing training preflight."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

import torch

from . import plastic_depth_console_minor_patch as _console_minor
from . import plastic_depth_same_batch_all_probes_patch as _same_batch
from . import stage6_trainer as _stage6
from . import trainer_step as _trainer_step
from .plastic_depth_cuda import PlasticDepthCudaAllocatorReserve, is_cuda_out_of_memory


_EVENT_NAME = "plastic_depth_cuda_growth_headroom"
_CONSOLE_MEMORY_HOLD_KEY = "plastic_cuda_growth_memory_hold"
_CONSOLE_MEMORY_HOLD_UPDATE_ATTRIBUTE = "_plastic_cuda_growth_memory_hold_update"
_ORIGINAL_BEGIN_INLINE_UPDATE = _trainer_step.TrainerStepMixin._begin_plastic_depth_inline_update
_ORIGINAL_COMMIT_INLINE_UPDATE = _trainer_step.TrainerStepMixin._commit_plastic_depth_inline_update
_ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD = _stage6.Stage6Trainer._prepare_console_progress_payload
_ORIGINAL_FORMAT_PROGRESS_LINE = _stage6.format_progress_line


def _release_context_reserve(
    context: Dict[str, Any],
    *,
    empty_cache: bool,
) -> None:
    reserve = context.get("cuda_allocator_reserve")
    release = getattr(reserve, "release", None)
    if callable(release):
        release(empty_cache=empty_cache)
    context["cuda_allocator_reserve"] = None


def _release_returned_reserve(
    context: Dict[str, Any],
    reserve: Any,
    *,
    empty_cache: bool,
) -> None:
    release = getattr(reserve, "release", None)
    if callable(release):
        release(empty_cache=empty_cache)
    context["cuda_allocator_reserve"] = None


def _force_growth_hold(
    trainer: Any,
    context: Dict[str, Any],
    *,
    reason: str,
) -> None:
    current_count = int(context["current_count"])
    decision = context.get("decision")
    if decision is None:
        raise RuntimeError("PLASTIC CUDA headroom rejection lacks a completed count decision")
    # vvv THOG retain the pre-veto winner so the established v0.55/v0.56 audit bridge can replay a CUDA framework hold independently of the statistical decision
    raw_selected_count = int(decision.selected_count)
    # ^^^ THOG
    evidence = tuple(replace(item, significant=False) for item in decision.evidence)
    held = replace(decision, selected_count=current_count, evidence=evidence)
    context["decision"] = held
    context["selected_count"] = current_count
    context["paired_evidence"] = held.report()
    context["score_evidence"] = held.report()
    context["plastic_same_batch_framework_hold_reason"] = str(reason)
    for report_key in (
        "plastic_v055_sen_kendall_report",
        "plastic_directional_report",
    ):
        report = context.get(report_key)
        if isinstance(report, dict):
            report["selected_count"] = current_count
            report["cuda_growth_headroom_verified"] = False
            report["cuda_growth_headroom_reason"] = str(reason)
            # vvv THOG use the same replay contract as incomplete same-batch holds: raw Sen/Kendall evidence remains auditable while the framework-selected count stays current
            report["framework_hold_reason"] = str(reason)
            report["framework_raw_selected_count"] = raw_selected_count
            # ^^^ THOG
    setter = getattr(trainer.raw_model, "set_plastic_depth_update_layer_count", None)
    if not callable(setter):
        raise RuntimeError("PLASTIC DEPTH training model lacks update-prefix control")
    setter(current_count)
    context["cuda_growth_headroom_verified"] = False
    context["cuda_growth_headroom_reason"] = str(reason)
    context["upward_candidate_feasible"] = False


def _headroom_reserve(
    trainer: Any,
    context: Dict[str, Any],
) -> Tuple[Optional[PlasticDepthCudaAllocatorReserve], bool]:
    reserve_gib = float(trainer.config.plastic__cuda_allocator_reserve_gib)
    if reserve_gib <= 0.0:
        return None, True
    reserve = context.get("cuda_allocator_reserve")
    if not isinstance(reserve, PlasticDepthCudaAllocatorReserve):
        reserve = PlasticDepthCudaAllocatorReserve(
            device=trainer.device,
            reserve_gib=reserve_gib,
        )
        context["cuda_allocator_reserve"] = reserve
    local_feasible = bool(reserve.active or reserve.acquire())
    globally_feasible = bool(trainer.distributed.all_true(local_feasible))
    if not globally_feasible:
        reserve.release(empty_cache=True)
        context["cuda_allocator_reserve"] = None
        return None, False
    return reserve, True


def _same_batch_training_memory_preflight(
    trainer: Any,
    context: Dict[str, Any],
    *,
    selected_count: int,
) -> bool:
    state = _same_batch._window_state(trainer)
    active = state.get("active")
    if active is None:
        raise RuntimeError("PLASTIC CUDA headroom preflight has no active same-batch window")
    batch = _same_batch._cached_probe_batch(trainer, active)
    rng_state = _same_batch._capture_probe_rng(trainer)
    prior_probe_report = trainer.raw_model.last_plastic_depth_inline_probe_report
    retained_materializations_active = False
    local_feasible = True
    accumulation_steps = max(1, int(trainer.config.gradient_accumulation_steps))
    # vvv THOG one backward from zero gradients understates an accumulated update: the first backward creates persistent gradient storage, so a second pass tests the steady peak with those gradients already resident
    preflight_micro_steps = min(2, accumulation_steps)
    context["cuda_growth_headroom_preflight_microsteps"] = int(preflight_micro_steps)
    context["cuda_growth_headroom_failed_preflight_microstep"] = None
    preflight_micro_step = -1
    # ^^^ THOG
    try:
        trainer.optimizer.zero_grad(set_to_none=True)
        setter = getattr(trainer.raw_model, "set_plastic_depth_update_layer_count", None)
        if not callable(setter):
            raise RuntimeError("PLASTIC DEPTH training model lacks update-prefix control")
        setter(int(selected_count))
        retained_materializations_active = bool(
            trainer.raw_model.begin_optimizer_update()
        )
        # vvv THOG reproduce the memory-relevant accumulation state under the configured reserve without paying for the complete accumulation window; after pass one gradient storage is resident and subsequent same-shaped passes do not add another gradient set
        for preflight_micro_step in range(preflight_micro_steps):
            with trainer.distributed.no_sync_context(trainer.model, synchronize=False):
                with trainer.autocast_context():
                    _, loss = trainer.model(
                        batch.inputs,
                        batch.targets,
                        plastic_depth_active_layers_override=int(selected_count),
                    )
                    if loss is None:
                        raise RuntimeError("PLASTIC CUDA headroom preflight returned no training loss")
                    scaled_loss = trainer.scaler.scale(loss / accumulation_steps)
                scaled_loss.backward()
        # ^^^ THOG
        if retained_materializations_active:
            trainer.raw_model.finalize_optimizer_update()
            retained_materializations_active = False
        torch.cuda.synchronize(trainer.device)
    except BaseException as error:
        if not is_cuda_out_of_memory(error):
            raise
        local_feasible = False
        # vvv THOG retain the exact failing preflight ordinal for audit/diagnostics; -1 means the OOM preceded the first synthetic microstep
        context["cuda_growth_headroom_failed_preflight_microstep"] = int(preflight_micro_step)
        # ^^^ THOG
    finally:
        if retained_materializations_active:
            trainer.raw_model.end_optimizer_update()
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.raw_model.last_plastic_depth_inline_probe_report = prior_probe_report
        _same_batch._restore_probe_rng(trainer, rng_state)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return bool(trainer.distributed.all_true(local_feasible))


def _begin_plastic_depth_inline_update_with_cuda_headroom(
    self: Any,
) -> Optional[Dict[str, Any]]:
    context = _ORIGINAL_BEGIN_INLINE_UPDATE(self)
    if context is None or self.device.type != "cuda":
        return context
    if not bool(context.get("plastic_same_batch_precomputed", False)):
        return context
    selected = context.get("selected_count")
    if selected is None:
        return context
    current_count = int(context["current_count"])
    selected_count = int(selected)
    # vvv THOG full-radius evidence may acquire the growth reserve before same-batch window logic forces STAY; never charge that reserve to an already-safe non-growth update
    if selected_count <= current_count:
        _release_context_reserve(context, empty_cache=True)
        return context
    # ^^^ THOG

    # vvv THOG a grow decision is provisional until a grad-bearing accumulated-state preflight at the proposed count succeeds while the configured reserve remains allocated
    reserve, reserve_feasible = _headroom_reserve(self, context)
    if not reserve_feasible:
        _force_growth_hold(
            self,
            context,
            reason="cuda_allocator_reserve_unavailable",
        )
        return context
    if reserve is None:
        context["cuda_growth_headroom_verified"] = True
        context["cuda_growth_headroom_reason"] = "reserve_disabled"
        return context

    training_feasible = _same_batch_training_memory_preflight(
        self,
        context,
        selected_count=selected_count,
    )
    if not training_feasible:
        _release_returned_reserve(context, reserve, empty_cache=True)
        _force_growth_hold(
            self,
            context,
            reason="grad_bearing_training_exceeds_cuda_reserve_barrier",
        )
        return context

    context["cuda_growth_headroom_verified"] = True
    context["cuda_growth_headroom_reason"] = "grad_bearing_training_with_reserve_succeeded"
    for report_key in (
        "plastic_v055_sen_kendall_report",
        "plastic_directional_report",
    ):
        report = context.get(report_key)
        if isinstance(report, dict):
            report["cuda_growth_headroom_verified"] = True
            report["cuda_growth_headroom_reason"] = context["cuda_growth_headroom_reason"]
    # ^^^ THOG
    return context


def _commit_plastic_depth_inline_update_with_cuda_headroom(
    self: Any,
    context: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    transition = _ORIGINAL_COMMIT_INLINE_UPDATE(self, context)
    if context is None or "cuda_growth_headroom_verified" not in context:
        return transition
    update_number = int(self.state.completed_updates) + 1
    verified = bool(context["cuda_growth_headroom_verified"])
    if not verified and int(context["selected_count"]) == int(context["current_count"]):
        setattr(self, _CONSOLE_MEMORY_HOLD_UPDATE_ATTRIBUTE, update_number)
    self._record(
        _EVENT_NAME,
        update_number=update_number,
        current_count=int(context["current_count"]),
        selected_count=int(context["selected_count"]),
        reserve_gib=float(self.config.plastic__cuda_allocator_reserve_gib),
        verified=verified,
        reason=str(context.get("cuda_growth_headroom_reason", "")),
        # vvv THOG expose how far the accumulation-aware preflight ran and where an OOM was trapped without changing the established event name
        preflight_microsteps=int(context.get("cuda_growth_headroom_preflight_microsteps", 0)),
        failed_preflight_microstep=context.get("cuda_growth_headroom_failed_preflight_microstep"),
        # ^^^ THOG
    )
    return transition


_trainer_step.TrainerStepMixin._begin_plastic_depth_inline_update = (
    _begin_plastic_depth_inline_update_with_cuda_headroom
)
_trainer_step.TrainerStepMixin._commit_plastic_depth_inline_update = (
    _commit_plastic_depth_inline_update_with_cuda_headroom
)
# ^^^ THOG


# vvv THOG expose a CUDA-vetoed statistical GROW on its exact operator row without changing the committed STAY decision or audit evidence
def _prepare_console_progress_payload_with_cuda_growth_hold(
    self: Any,
    event: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    values = _ORIGINAL_PREPARE_CONSOLE_PROGRESS_PAYLOAD(self, event, payload)
    values.pop(_CONSOLE_MEMORY_HOLD_KEY, None)
    if event != "optimizer_progress":
        return values
    try:
        completed_updates = int(
            str(values.get("completed_updates", payload.get("completed_updates")))
            .strip()
            .replace(",", "")
        )
    except (TypeError, ValueError):
        return values
    if int(getattr(self, _CONSOLE_MEMORY_HOLD_UPDATE_ATTRIBUTE, -1)) == completed_updates:
        values[_CONSOLE_MEMORY_HOLD_KEY] = True
    return values


def _format_progress_line_with_cuda_growth_hold(
    run_id: str,
    event: str,
    payload: Dict[str, Any],
) -> str:
    local = dict(payload)
    memory_hold = bool(local.pop(_CONSOLE_MEMORY_HOLD_KEY, False))
    line = _ORIGINAL_FORMAT_PROGRESS_LINE(run_id, event, local)
    if event == "optimizer_progress" and memory_hold:
        line = (
            f"{line.rstrip()}  {_console_minor._PALE_CYAN}"
            f"<<< stopped by memory limit{_console_minor._RESET}"
        )
    return line


_stage6.Stage6Trainer._prepare_console_progress_payload = (
    _prepare_console_progress_payload_with_cuda_growth_hold
)
_stage6.format_progress_line = _format_progress_line_with_cuda_growth_hold
# ^^^ THOG

# vvv THOG install PLASTIC v0.56 objective-neutral decision ownership only after CUDA/same-batch/v0.55 runtime guards are fully established
from . import plastic_depth_v056_objective_decision_patch as _plastic_depth_v056_objective_decision_patch
from . import plastic_depth_v056_transition_highlight_patch as _plastic_depth_v056_transition_highlight_patch
from . import plastic_depth_v056_help_patch as _plastic_depth_v056_help_patch
# ^^^ THOG
