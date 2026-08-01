# vvv THOG
from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Callable, Dict, Tuple

import torch
from torch import Tensor, nn


MaterializationKey = Tuple[str, int]
MaterializeFunction = Callable[[nn.Module, str, int], Tensor]
_CONTROLLER_ATTRIBUTE = "_update_retained_materializations_controller"


@dataclass
class _RetainedMaterialization:
    generated: Tensor
    operational: Tensor


def _retained_materialize(
    trajectory: nn.Module,
    name: str,
    layer_index: int,
) -> Tensor:
    controller = getattr(trajectory, _CONTROLLER_ATTRIBUTE, None)
    if not isinstance(controller, UpdateRetainedMaterializations):
        raise RuntimeError("trajectory has no update-retained materialisation controller")
    return controller.materialize(name, layer_index)


class UpdateRetainedMaterializations:
    """Retain generated operational materialisations for one optimiser update."""

    def __init__(self, trajectory: nn.Module, *, enabled: bool) -> None:
        if not isinstance(enabled, bool):
            raise TypeError(f"enabled must be bool; got {enabled!r}")
        original_materialize = getattr(type(trajectory), "materialize", None)
        if not callable(original_materialize):
            raise TypeError("trajectory type must provide a callable materialize method")
        self._trajectory = trajectory
        self._enabled = enabled
        self._original_materialize: MaterializeFunction = original_materialize
        self._retained: Dict[MaterializationKey, _RetainedMaterialization] = {}
        self._active = False
        self._request_count = 0
        self._materialization_count = 0
        trajectory.materialize = MethodType(_retained_materialize, trajectory)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def active(self) -> bool:
        return self._active

    @property
    def retained_count(self) -> int:
        return len(self._retained)

    @property
    def request_count(self) -> int:
        return self._request_count

    @property
    def materialization_count(self) -> int:
        return self._materialization_count

    def _is_generated_family(self, name: str) -> bool:
        family_metadata = getattr(self._trajectory, "family_metadata", None)
        is_generated = getattr(self._trajectory, "_is_generated", None)
        if not callable(family_metadata) or not callable(is_generated):
            return True
        try:
            metadata = family_metadata(name)
        except KeyError:
            return True
        return bool(is_generated(metadata))

    def begin(self) -> bool:
        if not self._enabled:
            return False
        if self._active:
            raise RuntimeError("update-retained materialisations are already active")
        if self._retained:
            raise RuntimeError("stale update-retained materialisations were not released")
        self._request_count = 0
        self._materialization_count = 0
        self._active = True
        return True

    def materialize(self, name: str, layer_index: int) -> Tensor:
        if not self._active or not self._is_generated_family(name):
            return self._original_materialize(self._trajectory, name, layer_index)
        self._request_count += 1
        key = (name, layer_index)
        retained = self._retained.get(key)
        if retained is not None:
            return retained.operational
        generated = self._original_materialize(self._trajectory, name, layer_index)
        operational = generated.detach()
        if generated.requires_grad:
            operational.requires_grad_(True)
        self._retained[key] = _RetainedMaterialization(
            generated=generated,
            operational=operational,
        )
        self._materialization_count += 1
        return operational

    def finalize(self) -> Tuple[nn.Parameter, ...]:
        if not self._active:
            raise RuntimeError("update-retained materialisations are not active")
        parameters = tuple(
            parameter
            for parameter in self._trajectory.parameters()
            if parameter.requires_grad
        )
        generated_outputs = []
        operational_gradients = []
        for retained in self._retained.values():
            gradient = retained.operational.grad
            if gradient is None or not retained.generated.requires_grad:
                continue
            generated_outputs.append(retained.generated)
            operational_gradients.append(gradient.detach())
        projected_parameters = []
        try:
            if generated_outputs:
                projected_gradients = torch.autograd.grad(
                    tuple(generated_outputs),
                    parameters,
                    grad_outputs=tuple(operational_gradients),
                    allow_unused=True,
                    retain_graph=False,
                    create_graph=False,
                )
                for parameter, gradient in zip(parameters, projected_gradients):
                    if gradient is None:
                        continue
                    if parameter.grad is None:
                        parameter.grad = gradient
                    else:
                        parameter.grad.add_(gradient)
                    projected_parameters.append(parameter)
            return tuple(projected_parameters)
        finally:
            self._retained.clear()
            self._active = False

    def end(self) -> bool:
        was_active = self._active
        self._retained.clear()
        self._active = False
        return was_active


def attach_update_retained_materializations(
    trajectory: nn.Module,
    *,
    enabled: bool,
) -> UpdateRetainedMaterializations:
    existing = getattr(trajectory, _CONTROLLER_ATTRIBUTE, None)
    if existing is not None:
        if not isinstance(existing, UpdateRetainedMaterializations):
            raise RuntimeError(
                f"trajectory attribute {_CONTROLLER_ATTRIBUTE} is already occupied"
            )
        if existing.enabled != enabled:
            raise RuntimeError(
                "update-retained materialisation controller already attached "
                "with a different enabled setting"
            )
        return existing
    controller = UpdateRetainedMaterializations(trajectory, enabled=enabled)
    setattr(trajectory, _CONTROLLER_ATTRIBUTE, controller)
    return controller
# ^^^ THOG
