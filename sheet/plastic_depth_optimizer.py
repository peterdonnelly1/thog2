# vvv THOG
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from .depth_trajectory import PlasticDepthModelTransition
from .plastic_depth_gauge import apply_depth_coefficient_transform_chunked


PLASTIC_DEPTH_ADAMW_MAX_CONDITION_NUMBER = 1.0e8


@dataclass(frozen=True)
class PlasticDepthAdamWStateSource:
    key: str
    tensor_id: int
    tensor_version: int
    shape: Tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True)
class PlasticDepthAdamWStateReplacement:
    name: str
    parameter: nn.Parameter
    depth_axis: int
    sources: Tuple[PlasticDepthAdamWStateSource, ...]
    exp_avg: Optional[Tensor]
    exp_avg_sq: Optional[Tensor]
    max_exp_avg_sq: Optional[Tensor]
    had_initialized_state: bool


@dataclass(frozen=True)
class PlasticDepthAdamWTransition:
    model_transition: PlasticDepthModelTransition
    optimizer_id: int
    covector_transform: Tensor
    squared_covector_transform: Tensor
    replacements: Tuple[PlasticDepthAdamWStateReplacement, ...]
    migration_mode: str
    fallback_reason: Optional[str]
    condition_number: float
    migrated_parameter_count: int
    reset_parameter_count: int


def _state_sources(state: Dict[str, object]) -> Tuple[PlasticDepthAdamWStateSource, ...]:
    sources = []
    for key in ("step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        value = state.get(key)
        if isinstance(value, Tensor):
            sources.append(
                PlasticDepthAdamWStateSource(
                    key=key,
                    tensor_id=id(value),
                    tensor_version=int(value._version),
                    shape=tuple(int(dimension) for dimension in value.shape),
                    dtype=value.dtype,
                    device=value.device,
                )
            )
    return tuple(sources)


def _validate_adamw_state_layout(
    parameter: nn.Parameter,
    state: Dict[str, object],
) -> bool:
    exp_avg = state.get("exp_avg")
    exp_avg_sq = state.get("exp_avg_sq")
    if exp_avg is None and exp_avg_sq is None:
        return False
    if not isinstance(exp_avg, Tensor) or not isinstance(exp_avg_sq, Tensor):
        raise RuntimeError("AdamW coefficient state must own both exp_avg and exp_avg_sq tensors")
    for key, value in (("exp_avg", exp_avg), ("exp_avg_sq", exp_avg_sq)):
        if value.shape != parameter.shape:
            raise RuntimeError(
                f"AdamW {key} shape does not match coefficient parameter; "
                f"state={tuple(value.shape)}, parameter={tuple(parameter.shape)}"
            )
        if not value.is_floating_point():
            raise RuntimeError(f"AdamW {key} must use a floating dtype; got {value.dtype}")
    maximum = state.get("max_exp_avg_sq")
    if maximum is not None:
        if not isinstance(maximum, Tensor) or maximum.shape != parameter.shape:
            raise RuntimeError("AdamW max_exp_avg_sq must match the coefficient parameter shape")
        if not maximum.is_floating_point():
            raise RuntimeError(
                f"AdamW max_exp_avg_sq must use a floating dtype; got {maximum.dtype}"
            )
    return True


def _transform_state_tensor(
    value: Tensor,
    transform: Tensor,
    *,
    depth_axis: int,
    maximum_series_per_chunk: int,
) -> Tensor:
    return apply_depth_coefficient_transform_chunked(
        value.detach(),
        transform,
        depth_axis=depth_axis,
        output_dtype=value.dtype,
        maximum_series_per_chunk=maximum_series_per_chunk,
    )


def _zero_state_replacement(
    *,
    name: str,
    parameter: nn.Parameter,
    depth_axis: int,
    state: Dict[str, object],
    had_initialized_state: bool,
) -> PlasticDepthAdamWStateReplacement:
    def zero_for(key: str) -> Optional[Tensor]:
        value = state.get(key)
        return torch.zeros_like(value) if isinstance(value, Tensor) else None

    return PlasticDepthAdamWStateReplacement(
        name=name,
        parameter=parameter,
        depth_axis=depth_axis,
        sources=_state_sources(state),
        exp_avg=zero_for("exp_avg"),
        exp_avg_sq=zero_for("exp_avg_sq"),
        max_exp_avg_sq=zero_for("max_exp_avg_sq"),
        had_initialized_state=had_initialized_state,
    )


def prepare_plastic_depth_adamw_transition(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_transition: PlasticDepthModelTransition,
    *,
    maximum_condition_number: float = PLASTIC_DEPTH_ADAMW_MAX_CONDITION_NUMBER,
    maximum_series_per_chunk: int = 65536,
) -> PlasticDepthAdamWTransition:
    """Prepare stock-AdamW coefficient-state migration without mutating model or optimizer."""

    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError(
            "PLASTIC DEPTH gauge migration requires the stock torch.optim.AdamW optimizer"
        )
    if not math.isfinite(maximum_condition_number) or maximum_condition_number <= 0.0:
        raise ValueError(
            "maximum_condition_number must be finite and positive; "
            f"got {maximum_condition_number!r}"
        )
    trajectory = getattr(model, "trajectory", None)
    coefficients = getattr(trajectory, "coefficients", None)
    if coefficients is None:
        raise RuntimeError("model has no PLASTIC DEPTH coefficient registry")

    fallback_reason: Optional[str] = None
    covector_transform = torch.empty_like(model_transition.transform)
    squared_covector_transform = torch.empty_like(model_transition.transform)
    condition_number = float("inf")
    try:
        transform = model_transition.transform.to(dtype=torch.float64)
        condition_number = float(torch.linalg.cond(transform).item())
        if not math.isfinite(condition_number) or condition_number > maximum_condition_number:
            raise RuntimeError(
                "PLASTIC DEPTH AdamW covector transform is ill-conditioned; "
                f"condition_number={condition_number:.6e}, "
                f"maximum={maximum_condition_number:.6e}"
            )
        identity = torch.eye(
            transform.shape[0],
            dtype=torch.float64,
            device=transform.device,
        )
        covector_transform = torch.linalg.solve(transform.transpose(0, 1), identity)
        squared_covector_transform = covector_transform.square()
        if not bool(torch.isfinite(covector_transform).all().item()):
            raise RuntimeError("PLASTIC DEPTH AdamW covector transform is non-finite")
    except (RuntimeError, ValueError) as error:
        fallback_reason = str(error)

    replacements = []
    migrated_parameter_count = 0
    reset_parameter_count = 0
    for coefficient_replacement in model_transition.replacements:
        parameter = coefficients[coefficient_replacement.name]
        state = optimizer.state.get(parameter, {})
        if not isinstance(state, dict):
            raise RuntimeError("stock AdamW parameter state must be a dictionary")
        try:
            had_initialized_state = _validate_adamw_state_layout(parameter, state)
            if not had_initialized_state:
                replacements.append(
                    PlasticDepthAdamWStateReplacement(
                        name=coefficient_replacement.name,
                        parameter=parameter,
                        depth_axis=coefficient_replacement.depth_axis,
                        sources=_state_sources(state),
                        exp_avg=None,
                        exp_avg_sq=None,
                        max_exp_avg_sq=None,
                        had_initialized_state=False,
                    )
                )
                continue
            if fallback_reason is not None:
                raise RuntimeError(fallback_reason)
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            assert isinstance(exp_avg, Tensor)
            assert isinstance(exp_avg_sq, Tensor)
            transformed_exp_avg = _transform_state_tensor(
                exp_avg,
                covector_transform,
                depth_axis=coefficient_replacement.depth_axis,
                maximum_series_per_chunk=maximum_series_per_chunk,
            )
            transformed_exp_avg_sq = _transform_state_tensor(
                exp_avg_sq,
                squared_covector_transform,
                depth_axis=coefficient_replacement.depth_axis,
                maximum_series_per_chunk=maximum_series_per_chunk,
            )
            maximum = state.get("max_exp_avg_sq")
            transformed_maximum = (
                _transform_state_tensor(
                    maximum,
                    squared_covector_transform,
                    depth_axis=coefficient_replacement.depth_axis,
                    maximum_series_per_chunk=maximum_series_per_chunk,
                )
                if isinstance(maximum, Tensor)
                else None
            )
            candidates = [transformed_exp_avg, transformed_exp_avg_sq]
            if transformed_maximum is not None:
                candidates.append(transformed_maximum)
            if not all(bool(torch.isfinite(value).all().item()) for value in candidates):
                raise RuntimeError("PLASTIC DEPTH AdamW migrated moments are non-finite")
            if not bool((transformed_exp_avg_sq >= 0.0).all().item()):
                raise RuntimeError("PLASTIC DEPTH AdamW migrated second moment is negative")
            if transformed_maximum is not None and not bool(
                (transformed_maximum >= 0.0).all().item()
            ):
                raise RuntimeError("PLASTIC DEPTH AdamW migrated AMSGrad moment is negative")
            replacements.append(
                PlasticDepthAdamWStateReplacement(
                    name=coefficient_replacement.name,
                    parameter=parameter,
                    depth_axis=coefficient_replacement.depth_axis,
                    sources=_state_sources(state),
                    exp_avg=transformed_exp_avg,
                    exp_avg_sq=transformed_exp_avg_sq,
                    max_exp_avg_sq=transformed_maximum,
                    had_initialized_state=True,
                )
            )
            migrated_parameter_count += 1
        except (RuntimeError, ValueError) as error:
            if fallback_reason is None:
                fallback_reason = str(error)
            replacements = []
            migrated_parameter_count = 0
            reset_parameter_count = 0
            for reset_replacement in model_transition.replacements:
                reset_parameter = coefficients[reset_replacement.name]
                reset_state = optimizer.state.get(reset_parameter, {})
                if not isinstance(reset_state, dict):
                    raise RuntimeError("stock AdamW parameter state must be a dictionary")
                reset_initialized = _validate_adamw_state_layout(reset_parameter, reset_state)
                replacements.append(
                    _zero_state_replacement(
                        name=reset_replacement.name,
                        parameter=reset_parameter,
                        depth_axis=reset_replacement.depth_axis,
                        state=reset_state,
                        had_initialized_state=reset_initialized,
                    )
                )
                if reset_initialized:
                    reset_parameter_count += 1
            break

    migration_mode = "reset" if fallback_reason is not None else "transform"
    return PlasticDepthAdamWTransition(
        model_transition=model_transition,
        optimizer_id=id(optimizer),
        covector_transform=covector_transform,
        squared_covector_transform=squared_covector_transform,
        replacements=tuple(replacements),
        migration_mode=migration_mode,
        fallback_reason=fallback_reason,
        condition_number=condition_number,
        migrated_parameter_count=migrated_parameter_count,
        reset_parameter_count=reset_parameter_count,
    )


def _validate_prepared_optimizer_state(
    optimizer: torch.optim.Optimizer,
    transition: PlasticDepthAdamWTransition,
) -> None:
    if id(optimizer) != transition.optimizer_id:
        raise RuntimeError("PLASTIC DEPTH AdamW transition belongs to a different optimizer")
    for replacement in transition.replacements:
        state = optimizer.state.get(replacement.parameter, {})
        if not isinstance(state, dict):
            raise RuntimeError("stock AdamW parameter state must be a dictionary")
        current_sources = {source.key: source for source in _state_sources(state)}
        if set(current_sources) != {source.key for source in replacement.sources}:
            raise RuntimeError(
                "PLASTIC DEPTH AdamW state layout changed after transition preparation; "
                f"family={replacement.name}"
            )
        for source in replacement.sources:
            current = current_sources[source.key]
            if (
                current.tensor_id != source.tensor_id
                or current.tensor_version != source.tensor_version
                or current.shape != source.shape
                or current.dtype != source.dtype
                or current.device != source.device
            ):
                raise RuntimeError(
                    "PLASTIC DEPTH AdamW state changed after transition preparation; "
                    f"family={replacement.name}, state={source.key}"
                )


def commit_plastic_depth_adamw_transition(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    transition: PlasticDepthAdamWTransition,
) -> Dict[str, object]:
    """Commit the verified model re-gauge and prepared stock-AdamW state together."""

    _validate_prepared_optimizer_state(optimizer, transition)
    model_report = model.commit_plastic_depth_count_transition(
        transition.model_transition
    )
    with torch.no_grad():
        for replacement in transition.replacements:
            if not replacement.had_initialized_state:
                continue
            state = optimizer.state[replacement.parameter]
            if replacement.exp_avg is None or replacement.exp_avg_sq is None:
                raise RuntimeError("prepared AdamW state replacement is incomplete")
            state["exp_avg"].copy_(replacement.exp_avg)
            state["exp_avg_sq"].copy_(replacement.exp_avg_sq)
            if replacement.max_exp_avg_sq is not None:
                state["max_exp_avg_sq"].copy_(replacement.max_exp_avg_sq)
    return {
        **model_report,
        "adamw_state_migration_mode": transition.migration_mode,
        "adamw_state_fallback_reason": transition.fallback_reason,
        "adamw_state_condition_number": transition.condition_number,
        "adamw_state_migrated_parameter_count": transition.migrated_parameter_count,
        "adamw_state_reset_parameter_count": transition.reset_parameter_count,
    }


__all__ = [
    "PLASTIC_DEPTH_ADAMW_MAX_CONDITION_NUMBER",
    "PlasticDepthAdamWStateReplacement",
    "PlasticDepthAdamWTransition",
    "commit_plastic_depth_adamw_transition",
    "prepare_plastic_depth_adamw_transition",
]
# ^^^ THOG
