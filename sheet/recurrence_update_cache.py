# vvv THOG
from __future__ import annotations

from types import MethodType
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn


class RecurrenceUpdateCacheController:
    """Update-local host cache for recurrence-generated DEPTH parameters."""

    def __init__(self, trajectory: nn.Module) -> None:
        recurrence_generator = getattr(trajectory, "recurrence_generator", None)
        if recurrence_generator is None:
            raise ValueError("recurrence update cache requires a recurrence-generated trajectory")
        self.trajectory = trajectory
        self.recurrence_generator = recurrence_generator
        self._original_materialize_depth_parameter = trajectory._materialize_depth_parameter
        self._cached_layers: Optional[Dict[str, Tuple[Tensor, ...]]] = None
        self._parameter_versions: Dict[str, int] = {}

    @property
    def active(self) -> bool:
        return self._cached_layers is not None

    @property
    def parameters(self) -> Tuple[nn.Parameter, ...]:
        return tuple(
            self.trajectory.coefficients[name]
            for name in self._generated_family_names()
        )

    def _generated_family_names(self) -> Tuple[str, ...]:
        return tuple(
            item.name
            for item in self.trajectory.metadata
            if self.trajectory._representation(item) == "depth_coefficients"
        )

    def begin(self) -> None:
        if self.active:
            raise RuntimeError("recurrence update cache is already active")
        cached_layers: Dict[str, Tuple[Tensor, ...]] = {}
        parameter_versions: Dict[str, int] = {}
        with torch.no_grad():
            for name in self._generated_family_names():
                parameter = self.trajectory.coefficients[name]
                sequence = self.recurrence_generator.materialize_sequence(
                    parameter.detach(),
                    self.trajectory.config.n_layer,
                )
                expected_shape = (*parameter.shape[:-1], self.trajectory.config.n_layer)
                if tuple(sequence.shape) != expected_shape:
                    raise RuntimeError(
                        f"recurrence update cache sequence {name} has shape {tuple(sequence.shape)}; "
                        f"expected {expected_shape}"
                    )
                layers = []
                for layer_index in range(self.trajectory.config.n_layer):
                    host_layer = sequence[..., layer_index].to(
                        device="cpu",
                        dtype=parameter.dtype,
                    ).contiguous()
                    host_layer.requires_grad_(True)
                    layers.append(host_layer)
                cached_layers[name] = tuple(layers)
                parameter_versions[name] = int(parameter._version)
                del sequence
        self._cached_layers = cached_layers
        self._parameter_versions = parameter_versions

    def materialize(self, name: str, layer_index: int) -> Tensor:
        if self._cached_layers is None:
            return self._original_materialize_depth_parameter(name, layer_index)
        try:
            host_layer = self._cached_layers[name][layer_index]
        except KeyError as error:
            raise KeyError(f"uncached recurrence family: {name}") from error
        parameter = self.trajectory.coefficients[name]
        return host_layer.to(
            device=parameter.device,
            dtype=parameter.dtype,
            non_blocking=False,
        )

    def finalize(self, *, unscale_factor: float = 1.0) -> Tuple[nn.Parameter, ...]:
        if self._cached_layers is None:
            return ()
        if not isinstance(unscale_factor, (int, float)) or isinstance(unscale_factor, bool) or unscale_factor <= 0.0:
            raise ValueError(f"unscale_factor must be positive; got {unscale_factor!r}")

        finalized_parameters = []
        try:
            for name in self._generated_family_names():
                parameter = self.trajectory.coefficients[name]
                expected_version = self._parameter_versions[name]
                if int(parameter._version) != expected_version:
                    raise RuntimeError(
                        f"recurrence parameter {name} changed while its update cache was active"
                    )
                layer_gradients = tuple(
                    None if layer.grad is None else layer.grad.detach()
                    for layer in self._cached_layers[name]
                )
                parameter_gradient = self.recurrence_generator.parameter_gradient_from_layer_gradients(
                    parameter.detach(),
                    layer_gradients,
                )
                if float(unscale_factor) != 1.0:
                    parameter_gradient.mul_(float(unscale_factor))
                if parameter.grad is None:
                    parameter.grad = parameter_gradient
                else:
                    parameter.grad.add_(parameter_gradient)
                finalized_parameters.append(parameter)
        finally:
            self.discard()
        return tuple(finalized_parameters)

    def discard(self) -> None:
        self._cached_layers = None
        self._parameter_versions = {}


def attach_recurrence_update_cache(trajectory: nn.Module) -> Optional[RecurrenceUpdateCacheController]:
    recurrence_generator = getattr(trajectory, "recurrence_generator", None)
    if recurrence_generator is None:
        return None
    existing = getattr(trajectory, "_recurrence_update_cache_controller", None)
    if existing is not None:
        if not isinstance(existing, RecurrenceUpdateCacheController):
            raise TypeError("trajectory recurrence update cache controller has an unexpected type")
        return existing

    controller = RecurrenceUpdateCacheController(trajectory)
    trajectory._recurrence_update_cache_controller = controller

    def materialize_depth_parameter_with_update_cache(
        self: nn.Module,
        name: str,
        layer_index: int,
    ) -> Tensor:
        return self._recurrence_update_cache_controller.materialize(name, layer_index)

    trajectory._materialize_depth_parameter = MethodType(
        materialize_depth_parameter_with_update_cache,
        trajectory,
    )
    return controller


__all__ = ["RecurrenceUpdateCacheController", "attach_recurrence_update_cache"]
# ^^^ THOG
