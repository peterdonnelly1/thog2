# vvv THOG
"""Install the v1.3 sampling-only chaos bump after every PLASTIC overlay."""

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from . import plastic_depth as _plastic_depth
from . import plastic_depth_console_minor_patch as _console_minor
from . import plastic_depth_same_batch_all_probes_patch as _same_batch
from . import stage6_trainer as _stage6
from . import trainer_checkpoint_resume as _checkpoint_resume
from . import trainer_checkpoint_save as _checkpoint_save
from . import trainer_step as _trainer_step
from .chaos_bump_sampling import (
    CHAOS_BUMP_SAMPLING_VERSION,
    ResolvedChaosBumpSamplingConfig,
    chaos_bump_sampling_duration_steps,
    chaos_bump_sampling_interlude_steps,
    rattle_sampling_coordinates,
    resolve_chaos_bump_sampling_config,
)


_CHECKPOINT_STATE_KEY = "chaos_bump_sampling_state"
_STATE_ATTRIBUTE = "_chaos_bump_sampling_state"
_OVERRIDE_ATTRIBUTE = "_chaos_bump_sampling_public_coordinates_override"
_STATE_VERSION = 1

_ORIGINAL_PUBLIC_COORDINATES = _plastic_depth.PlasticDepthSamplingLattice.public_coordinates
_ORIGINAL_TRAIN_ONE_UPDATE = _trainer_step.TrainerStepMixin.train_one_update
_ORIGINAL_BEGIN_INLINE_UPDATE = _trainer_step.TrainerStepMixin._begin_plastic_depth_inline_update
_ORIGINAL_CHECKPOINT_PAYLOAD = _checkpoint_save.TrainerCheckpointSaveMixin.checkpoint_payload
_ORIGINAL_FROM_CHECKPOINT = _checkpoint_resume.TrainerCheckpointResumeMixin.from_checkpoint.__func__
_ORIGINAL_PREPARE_CONSOLE_PAYLOAD = _stage6.Stage6Trainer._prepare_console_progress_payload
_ORIGINAL_FORMAT_PROGRESS_LINE = _stage6.format_progress_line


def _resolved_config(trainer: Any) -> ResolvedChaosBumpSamplingConfig:
    config = trainer.config
    return resolve_chaos_bump_sampling_config(
        enabled=config.chaos_bump__sampling__enabled,
        plastic_enabled=config.plastic__enabled,
        initial_lockout_steps=config.chaos_bump__sampling__initial_lockout__steps,
        maximum_bumps=config.chaos_bump__sampling__maximum_bumps,
        interlude_min_steps=config.chaos_bump__sampling__interlude__min_steps,
        interlude_max_steps=config.chaos_bump__sampling__interlude__max_steps,
        duration_min_steps=config.chaos_bump__sampling__duration__min_steps,
        duration_max_steps=config.chaos_bump__sampling__duration__max_steps,
        duration_max_fraction_of_elapsed_steps=config.chaos_bump__sampling__duration__max_fraction_of_elapsed_steps,
        max_movement_fraction_of_local_gap=config.chaos_bump__sampling__max_movement_fraction_of_local_gap,
    )


def _new_state(trainer: Any) -> Dict[str, Any]:
    config = _resolved_config(trainer)
    first_start_update = max(
        config.initial_lockout_steps,
        (
            int(trainer.config.warmup_updates)
            if bool(trainer.config.plastic__freeze_geometry_during_warmup)
            else 0
        ),
    ) + 1
    return {
        "version": _STATE_VERSION,
        "feature_version": CHAOS_BUMP_SAMPLING_VERSION,
        "bumps_started": 0,
        "active": False,
        "active_bump_number": None,
        "active_layer_count": None,
        "start_update": None,
        "end_update": None,
        "duration_steps": None,
        "next_start_update": first_start_update,
        "base_coordinates": None,
        "rattled_coordinates": None,
        "base_raw_intervals": None,
        "signed_movements": None,
        "movement_fractions": None,
        "movement_directions": None,
        "visit_order": None,
        "last_end_update": None,
        "probe_lockout_until_update": 0,
        "last_transition": None,
        "last_transition_update": None,
        "next_interlude_steps": None,
    }


def _validate_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    resolved = copy.deepcopy(dict(state))
    if int(resolved.get("version", -1)) != _STATE_VERSION:
        raise ValueError(
            "unsupported sampling chaos bump state version: "
            f"{resolved.get('version')!r}"
        )
    if resolved.get("feature_version") != CHAOS_BUMP_SAMPLING_VERSION:
        raise ValueError(
            "sampling chaos bump feature version mismatch: "
            f"{resolved.get('feature_version')!r}"
        )
    required = set(_new_state_keys())
    missing = sorted(required - set(resolved))
    if missing:
        raise ValueError(f"sampling chaos bump state is missing fields: {missing}")
    if int(resolved["bumps_started"]) < 0:
        raise ValueError("sampling chaos bump count must be non-negative")
    if bool(resolved["active"]):
        active_required = (
            "active_bump_number",
            "active_layer_count",
            "start_update",
            "end_update",
            "duration_steps",
            "base_coordinates",
            "rattled_coordinates",
            "base_raw_intervals",
        )
        missing_active = [name for name in active_required if resolved.get(name) is None]
        if missing_active:
            raise ValueError(
                "active sampling chaos bump state is incomplete: "
                f"{missing_active}"
            )
        if int(resolved["end_update"]) < int(resolved["start_update"]):
            raise ValueError("sampling chaos bump end update precedes its start")
    return resolved


def _new_state_keys() -> Tuple[str, ...]:
    return (
        "version",
        "feature_version",
        "bumps_started",
        "active",
        "active_bump_number",
        "active_layer_count",
        "start_update",
        "end_update",
        "duration_steps",
        "next_start_update",
        "base_coordinates",
        "rattled_coordinates",
        "base_raw_intervals",
        "signed_movements",
        "movement_fractions",
        "movement_directions",
        "visit_order",
        "last_end_update",
        "probe_lockout_until_update",
        "last_transition",
        "last_transition_update",
        "next_interlude_steps",
    )


def _state(trainer: Any) -> Dict[str, Any]:
    state = getattr(trainer, _STATE_ATTRIBUTE, None)
    if state is None:
        state = _new_state(trainer)
        setattr(trainer, _STATE_ATTRIBUTE, state)
    return state


def _lattice(trainer: Any) -> Any:
    lattice = trainer._plastic_depth_lattice()
    if lattice is None:
        raise RuntimeError("sampling chaos bump requires the PLASTIC sampling lattice")
    return lattice


def _base_active_coordinates(lattice: Any, active_count: int) -> torch.Tensor:
    return _ORIGINAL_PUBLIC_COORDINATES(
        lattice,
        active_count,
        include_probe=True,
    )[:active_count]


def _install_override(trainer: Any, state: Mapping[str, Any]) -> None:
    if not bool(state["active"]):
        return
    lattice = _lattice(trainer)
    active_count = int(state["active_layer_count"])
    if int(lattice.current_active_layers) != active_count:
        raise RuntimeError(
            "sampling chaos bump layer count changed while restoring its override: "
            f"expected={active_count}, actual={lattice.current_active_layers}"
        )
    override = lattice.raw_intervals.new_tensor(
        tuple(float(value) for value in state["rattled_coordinates"])
    ).detach()
    if override.numel() != active_count:
        raise RuntimeError("sampling chaos bump override length disagrees with active count")
    setattr(lattice, _OVERRIDE_ATTRIBUTE, override)


def _clear_override(trainer: Any) -> None:
    lattice = _lattice(trainer)
    if hasattr(lattice, _OVERRIDE_ATTRIBUTE):
        delattr(lattice, _OVERRIDE_ATTRIBUTE)


def _public_coordinates_with_sampling_bump(
    self: Any,
    active_layers: Optional[int] = None,
    *,
    include_probe: Optional[bool] = None,
) -> torch.Tensor:
    base = _ORIGINAL_PUBLIC_COORDINATES(
        self,
        active_layers,
        include_probe=include_probe,
    )
    override = getattr(self, _OVERRIDE_ATTRIBUTE, None)
    if override is None:
        return base
    resolved_count = self.current_active_layers if active_layers is None else int(active_layers)
    if resolved_count != int(self.current_active_layers):
        return base.detach()
    active_count = int(self.current_active_layers)
    if base.numel() < active_count:
        raise RuntimeError("sampling chaos bump received a truncated coordinate lattice")
    active_override = override.to(device=base.device, dtype=base.dtype).detach()
    if base.numel() == active_count:
        return active_override
    return torch.cat((active_override, base[active_count:].detach()))


def _invalidate_count_evidence(trainer: Any, *, reason: str) -> None:
    trainer.state.plastic_depth_probe_histories = {}
    invalidate = getattr(_same_batch, "_invalidate_active_window", None)
    if callable(invalidate):
        invalidate(trainer, reason=reason)


def _assert_distributed_state(trainer: Any, state: Mapping[str, Any]) -> None:
    trainer.distributed.assert_identical_object(
        {
            "state": dict(state),
            "coordinates": state.get("rattled_coordinates"),
        },
        "sampling chaos bump state",
    )


def _assert_active_invariants(trainer: Any, state: Mapping[str, Any]) -> None:
    if not bool(state["active"]):
        return
    lattice = _lattice(trainer)
    expected_count = int(state["active_layer_count"])
    if int(lattice.current_active_layers) != expected_count:
        raise RuntimeError(
            "sampling chaos bump froze layer count but it changed: "
            f"expected={expected_count}, actual={lattice.current_active_layers}"
        )
    expected_raw = torch.tensor(
        tuple(float(value) for value in state["base_raw_intervals"]),
        dtype=lattice.raw_intervals.dtype,
        device="cpu",
    )
    actual_raw = lattice.raw_intervals.detach().to(device="cpu")
    if not torch.equal(actual_raw, expected_raw):
        raise RuntimeError("sampling chaos bump modified raw sampling intervals")


def _start_bump(trainer: Any, state: Dict[str, Any]) -> None:
    config = _resolved_config(trainer)
    lattice = _lattice(trainer)
    target_update = int(trainer.state.completed_updates) + 1
    bump_number = int(state["bumps_started"]) + 1
    active_count = int(lattice.current_active_layers)
    base = _base_active_coordinates(lattice, active_count).detach()
    rattle = rattle_sampling_coordinates(
        base,
        maximum_fraction_of_local_gap=config.max_movement_fraction_of_local_gap,
        model_seed=int(trainer.config.model_seed),
        bump_number=bump_number,
    )
    rattled = base.new_tensor(rattle.coordinates).detach()
    duration = chaos_bump_sampling_duration_steps(
        config,
        start_update=target_update,
        model_seed=int(trainer.config.model_seed),
        bump_number=bump_number,
    )
    base_values = tuple(float(value) for value in base.cpu().tolist())
    rattled_values = tuple(float(value) for value in rattled.cpu().tolist())
    actual_movements = tuple(
        rattled_value - base_value
        for base_value, rattled_value in zip(base_values, rattled_values)
    )
    state.update(
        {
            "bumps_started": bump_number,
            "active": True,
            "active_bump_number": bump_number,
            "active_layer_count": active_count,
            "start_update": target_update,
            "end_update": target_update + duration - 1,
            "duration_steps": duration,
            "next_start_update": None,
            "base_coordinates": base_values,
            "rattled_coordinates": rattled_values,
            "base_raw_intervals": tuple(
                float(value)
                for value in lattice.raw_intervals.detach().cpu().tolist()
            ),
            "signed_movements": actual_movements,
            "movement_fractions": tuple(float(value) for value in rattle.movement_fractions),
            "movement_directions": tuple(int(value) for value in rattle.movement_directions),
            "visit_order": tuple(int(value) for value in rattle.visit_order),
            "last_transition": "started",
            "last_transition_update": target_update,
            "next_interlude_steps": None,
        }
    )
    _invalidate_count_evidence(trainer, reason="sampling_chaos_bump_started")
    _install_override(trainer, state)
    _assert_active_invariants(trainer, state)
    _assert_distributed_state(trainer, state)
    trainer._record(
        "chaos_bump_sampling_started",
        bump_number=bump_number,
        start_update=target_update,
        end_update=int(state["end_update"]),
        duration_steps=duration,
        active_layer_count=active_count,
        base_coordinates=base_values,
        rattled_coordinates=rattled_values,
        signed_movements=actual_movements,
        visit_order=state["visit_order"],
    )


def _end_bump(trainer: Any, state: Dict[str, Any]) -> None:
    _assert_active_invariants(trainer, state)
    lattice = _lattice(trainer)
    base_coordinates = tuple(float(value) for value in state["base_coordinates"])
    _clear_override(trainer)
    restored = tuple(
        float(value)
        for value in _base_active_coordinates(
            lattice,
            int(state["active_layer_count"]),
        ).detach().cpu().tolist()
    )
    if restored != base_coordinates:
        raise RuntimeError(
            "sampling chaos bump failed to restore the exact pre-bump indices"
        )
    completed_update = int(trainer.state.completed_updates)
    bump_number = int(state["active_bump_number"])
    config = _resolved_config(trainer)
    interlude = None
    next_start = None
    if int(state["bumps_started"]) < config.maximum_bumps:
        interlude = chaos_bump_sampling_interlude_steps(
            config,
            model_seed=int(trainer.config.model_seed),
            completed_bump_number=bump_number,
        )
        next_start = completed_update + interlude + 1
    state.update(
        {
            "active": False,
            "last_end_update": completed_update,
            "probe_lockout_until_update": (
                completed_update
                + int(trainer.config.plastic__layer_count_update_brake)
            ),
            "last_transition": "ended",
            "last_transition_update": completed_update,
            "next_interlude_steps": interlude,
            "next_start_update": next_start,
        }
    )
    _invalidate_count_evidence(trainer, reason="sampling_chaos_bump_ended")
    _assert_distributed_state(trainer, state)
    trainer._record(
        "chaos_bump_sampling_ended",
        bump_number=bump_number,
        completed_update=completed_update,
        restored_coordinates=restored,
        next_interlude_steps=interlude,
        next_start_update=next_start,
        probe_lockout_until_update=int(state["probe_lockout_until_update"]),
    )


def _prepare_bump_for_attempt(trainer: Any) -> Optional[str]:
    config = _resolved_config(trainer)
    if not config.enabled:
        return None
    if getattr(trainer.config, "plastic__runtime_phase", "fine") != "fine":
        return None
    state = _state(trainer)
    if bool(state["active"]):
        _install_override(trainer, state)
        _assert_active_invariants(trainer, state)
        return None
    target_update = int(trainer.state.completed_updates) + 1
    next_start = state.get("next_start_update")
    if (
        next_start is not None
        and target_update >= int(next_start)
        and int(state["bumps_started"]) < config.maximum_bumps
    ):
        _start_bump(trainer, state)
        return "started"
    return None


def _movement_summary(state: Mapping[str, Any]) -> Tuple[float, float]:
    movements = tuple(abs(float(value)) for value in (state.get("signed_movements") or ()))
    if not movements:
        return 0.0, 0.0
    return sum(movements) / len(movements), max(movements)


def _train_one_update_with_sampling_bump(self: Any) -> Dict[str, Any]:
    if not bool(getattr(self.config, "chaos_bump__sampling__enabled", False)):
        return _ORIGINAL_TRAIN_ONE_UPDATE(self)
    transition = _prepare_bump_for_attempt(self)
    state = _state(self)
    _assert_active_invariants(self, state)
    metrics = _ORIGINAL_TRAIN_ONE_UPDATE(self)
    _assert_active_invariants(self, state)
    if not bool(metrics.get("skipped_update", 0.0)) and bool(state["active"]):
        if int(self.state.completed_updates) >= int(state["end_update"]):
            _end_bump(self, state)
            transition = "ended"
    mean_movement, max_movement = _movement_summary(state)
    bump_number = state.get("active_bump_number")
    bump_step = None
    if bool(state["active"]):
        bump_step = int(self.state.completed_updates) - int(state["start_update"]) + 1
    metrics.update(
        {
            "chaos_bump__sampling__active": float(bool(state["active"])),
            "chaos_bump__sampling__bump_number": (
                None if bump_number is None else int(bump_number)
            ),
            "chaos_bump__sampling__bump_step": bump_step,
            "chaos_bump__sampling__duration_steps": state.get("duration_steps"),
            "chaos_bump__sampling__mean_absolute_movement": mean_movement,
            "chaos_bump__sampling__maximum_absolute_movement": max_movement,
            "chaos_bump__sampling__transition": transition,
        }
    )
    return metrics


def _begin_inline_update_with_sampling_count_freeze(self: Any) -> Optional[Dict[str, Any]]:
    if not bool(getattr(self.config, "chaos_bump__sampling__enabled", False)):
        return _ORIGINAL_BEGIN_INLINE_UPDATE(self)
    state = _state(self)
    target_update = int(self.state.completed_updates) + 1
    if bool(state["active"]) or target_update <= int(state["probe_lockout_until_update"]):
        return None
    return _ORIGINAL_BEGIN_INLINE_UPDATE(self)


def _checkpoint_payload_with_sampling_bump(self: Any) -> Dict[str, Any]:
    payload = _ORIGINAL_CHECKPOINT_PAYLOAD(self)
    if bool(getattr(self.config, "chaos_bump__sampling__enabled", False)):
        state = _state(self)
        _assert_active_invariants(self, state)
        payload[_CHECKPOINT_STATE_KEY] = copy.deepcopy(state)
    return payload


def _from_checkpoint_with_sampling_bump(
    cls: Any,
    path: Any,
    train_tokens: torch.Tensor,
    validation_tokens: torch.Tensor,
    *,
    overrides: Optional[Mapping[str, Any]] = None,
    expected_config: Optional[Any] = None,
) -> Any:
    payload = _checkpoint_resume.load_payload(path)
    trainer = _ORIGINAL_FROM_CHECKPOINT(
        cls,
        path,
        train_tokens,
        validation_tokens,
        overrides=overrides,
        expected_config=expected_config,
    )
    enabled = bool(getattr(trainer.config, "chaos_bump__sampling__enabled", False))
    raw_state = payload.get(_CHECKPOINT_STATE_KEY)
    if raw_state is not None and not enabled:
        raise ValueError("disabled sampling chaos bump checkpoint unexpectedly carries runtime state")
    if enabled:
        if raw_state is None:
            raise ValueError("enabled sampling chaos bump checkpoint lacks runtime state")
        state = _validate_state(raw_state)
        setattr(trainer, _STATE_ATTRIBUTE, state)
        if bool(state["active"]):
            _install_override(trainer, state)
            _assert_active_invariants(trainer, state)
        _assert_distributed_state(trainer, state)
    return trainer


def _prepare_console_payload_with_sampling_bump(
    self: Any,
    event: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    values = _ORIGINAL_PREPARE_CONSOLE_PAYLOAD(self, event, payload)
    config = getattr(self, "config", None)
    if not bool(getattr(config, "chaos_bump__sampling__enabled", False)):
        return values
    state = _state(self)
    show_transition = (
        state.get("last_transition_update") is not None
        and int(state["last_transition_update"]) == int(self.state.completed_updates)
    )
    if event in {"optimizer_progress", "evaluation_completed"} and (
        bool(state["active"]) or show_transition
    ):
        step = None
        if bool(state["active"]):
            step = int(self.state.completed_updates) - int(state["start_update"]) + 1
        values["chaos_bump_sampling"] = {
            "active": bool(state["active"]),
            "transition": state.get("last_transition") if show_transition else None,
            "bump_number": int(state["active_bump_number"]),
            "step": step,
            "duration": int(state["duration_steps"]),
        }
    return values


def _format_progress_line_with_sampling_bump(
    run_id: str,
    event: str,
    payload: Dict[str, Any],
) -> str:
    marker = payload.get("chaos_bump_sampling")
    line = _ORIGINAL_FORMAT_PROGRESS_LINE(run_id, event, payload)
    if not isinstance(marker, Mapping):
        return line
    bump_number = int(marker["bump_number"])
    if bool(marker["active"]):
        status = f"B{bump_number} {int(marker['step'])}/{int(marker['duration'])}"
    else:
        status = f"B{bump_number} ended"
    return (
        f"{line.rstrip()}  {_console_minor._PALE_CYAN}"
        f"<<< chaos bump sampling {status}{_console_minor._RESET}"
    )


_plastic_depth.PlasticDepthSamplingLattice.public_coordinates = _public_coordinates_with_sampling_bump
_trainer_step.TrainerStepMixin.train_one_update = _train_one_update_with_sampling_bump
_trainer_step.TrainerStepMixin._begin_plastic_depth_inline_update = _begin_inline_update_with_sampling_count_freeze
_checkpoint_save.TrainerCheckpointSaveMixin.checkpoint_payload = _checkpoint_payload_with_sampling_bump
_checkpoint_resume.TrainerCheckpointResumeMixin.from_checkpoint = classmethod(_from_checkpoint_with_sampling_bump)
_stage6.Stage6Trainer._prepare_console_progress_payload = _prepare_console_payload_with_sampling_bump
_stage6.format_progress_line = _format_progress_line_with_sampling_bump


__all__ = []
# ^^^ THOG
