# vvv THOG
from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
# vvv THOG preserve the pre-bundle typing import for source history
# from typing import Callable, Dict, Tuple
from typing import Callable, Dict, Mapping, Optional, Tuple
# ^^^ THOG

import torch
from torch import Tensor, nn


MaterializationKey = Tuple[str, int]
MaterializeFunction = Callable[[nn.Module, str, int], Tensor]
# MaterializeLayerMatricesFunction = Callable[[nn.Module, int], Mapping[str, Tensor]]                                                               # <<< THOG preserved full-bundle-only callable
MaterializeLayerMatricesFunction = Callable[..., Mapping[str, Tensor]]                                                                                   # <<< THOG optional include_mlp keyword for partial HYPERBLOCK bundles
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


# vvv THOG let architecture-wide trajectories retain one batched matrix bundle per layer

# def _retained_materialize_layer_matrices(                                                                                                       # <<< THOG preserved pre-partial-bundle signature
#     trajectory: nn.Module,
#     layer_index: int,
# ) -> Mapping[str, Tensor]:
def _retained_materialize_layer_matrices(
    trajectory: nn.Module,
    layer_index: int,
    *,
    include_mlp: bool = True,
) -> Mapping[str, Tensor]:
    controller = getattr(trajectory, _CONTROLLER_ATTRIBUTE, None)
    if not isinstance(controller, UpdateRetainedMaterializations):
        raise RuntimeError("trajectory has no update-retained materialisation controller")
#     return controller.materialize_layer_matrices(layer_index)                                                                                    # <<< THOG preserved full-bundle-only delegation
    return controller.materialize_layer_matrices(
        layer_index,
        include_mlp=include_mlp,
    )


# ^^^ THOG


def _optimizer_compatible_gradient(
    parameter: nn.Parameter,
    gradient: Tensor,
) -> Tensor:
    if tuple(gradient.shape) != tuple(parameter.shape):
        raise RuntimeError(
            "projected gradient shape does not match its compact parameter: "
            f"gradient={tuple(gradient.shape)} parameter={tuple(parameter.shape)}"
        )
    if gradient.layout != parameter.layout:
        raise RuntimeError(
            "projected gradient layout does not match its compact parameter: "
            f"gradient={gradient.layout} parameter={parameter.layout}"
        )
    converted = gradient.detach().to(
        device=parameter.device,
        dtype=parameter.dtype,
    )
    if parameter.layout != torch.strided:
        return converted
    if converted.stride() == parameter.stride():
        return converted
    matched = torch.empty_strided(
        tuple(parameter.shape),
        tuple(parameter.stride()),
        dtype=parameter.dtype,
        device=parameter.device,
    )
    matched.copy_(converted)
    return matched


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
        # vvv THOG optional batched layer API remains transparent to trajectories that do not implement it
        original_materialize_layer_matrices = getattr(
            type(trajectory),
            "materialize_layer_matrices",
            None,
        )
        self._original_materialize_layer_matrices: Optional[
            MaterializeLayerMatricesFunction
        ] = (
            original_materialize_layer_matrices
            if callable(original_materialize_layer_matrices)
            else None
        )
        # ^^^ THOG
        self._retained: Dict[MaterializationKey, _RetainedMaterialization] = {}
        self._active = False
        self._request_count = 0
        self._materialization_count = 0
        trajectory.materialize = MethodType(_retained_materialize, trajectory)
        # vvv THOG route layer bundles through the same detach/project/release lifecycle as individual matrices
        if self._original_materialize_layer_matrices is not None:
            trajectory.materialize_layer_matrices = MethodType(
                _retained_materialize_layer_matrices,
                trajectory,
            )
        # ^^^ THOG

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

    # vvv THOG centralize generated-to-operational retention for individual and batched APIs
    def _retain_generated(
        self,
        name: str,
        layer_index: int,
        generated: Tensor,
    ) -> Tensor:
        if not isinstance(generated, Tensor):
            raise TypeError(
                "generated materialisation must be a Tensor; "
                f"got {type(generated).__name__} for {name!r}"
            )
        key = (name, layer_index)
        retained = self._retained.get(key)
        if retained is not None:
            return retained.operational
        operational = generated.detach()
        if generated.requires_grad:
            operational.requires_grad_(True)
        self._retained[key] = _RetainedMaterialization(
            generated=generated,
            operational=operational,
        )
        self._materialization_count += 1
        return operational
    # ^^^ THOG

    def materialize(self, name: str, layer_index: int) -> Tensor:
        if not self._active or not self._is_generated_family(name):
            return self._original_materialize(self._trajectory, name, layer_index)
        self._request_count += 1
        key = (name, layer_index)
        retained = self._retained.get(key)
        if retained is not None:
            return retained.operational
        generated = self._original_materialize(self._trajectory, name, layer_index)
        # vvv THOG preserve the pre-bundle retention statements for source history
        # operational = generated.detach()
        # if generated.requires_grad:
        #     operational.requires_grad_(True)
        # self._retained[key] = _RetainedMaterialization(
        #     generated=generated,
        #     operational=operational,
        # )
        # self._materialization_count += 1
        # return operational
        # ^^^ THOG
        return self._retain_generated(name, layer_index, generated)

    # vvv THOG retain and reuse all matrix families produced by one shared layer contraction
    # def materialize_layer_matrices(self, layer_index: int) -> Mapping[str, Tensor]:                                                               # <<< THOG preserved pre-partial-bundle signature
    def materialize_layer_matrices(
        self,
        layer_index: int,
        *,
        include_mlp: bool = True,
    ) -> Mapping[str, Tensor]:
        original = self._original_materialize_layer_matrices
        if original is None:
            raise AttributeError(
                f"{type(self._trajectory).__name__} has no materialize_layer_matrices API"
            )
        if not self._active:
            return original(self._trajectory, layer_index)

        # vvv THOG a direct HYPERBLOCK MLP update retains only attention matrices; full bundles keep the established identity
        expected_attribute = (
            "materialized_matrix_family_names"
            if include_mlp
            else "attention_materialized_matrix_family_names"
        )
        expected_names = tuple(
            getattr(self._trajectory, expected_attribute, ())
        )
        # ^^^ THOG
        if expected_names:
            retained_bundle = {
                name: self._retained[(name, layer_index)].operational
                for name in expected_names
                if (name, layer_index) in self._retained
            }
            if len(retained_bundle) == len(expected_names):
                self._request_count += len(expected_names)
                return retained_bundle

#         generated_bundle = original(self._trajectory, layer_index)                                                                               # <<< THOG preserved full-bundle-only call
        generated_bundle = (
            original(self._trajectory, layer_index)
            if include_mlp
            else original(self._trajectory, layer_index, include_mlp=False)
        )
        if not isinstance(generated_bundle, Mapping):
            raise TypeError(
                "materialize_layer_matrices must return a mapping; "
                f"got {type(generated_bundle).__name__}"
            )
        generated_names = tuple(generated_bundle)
        if expected_names and generated_names != expected_names:
            raise RuntimeError(
                "batched materialisation family order mismatch; "
                f"expected {expected_names}, got {generated_names}"
            )
        self._request_count += len(generated_names)
        return {
            name: self._retain_generated(name, layer_index, generated)
            for name, generated in generated_bundle.items()
        }
    # ^^^ THOG

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
                    compatible_gradient = _optimizer_compatible_gradient(
                        parameter,
                        gradient,
                    )
                    if parameter.grad is None:
                        parameter.grad = compatible_gradient
                    else:
                        compatible_existing = _optimizer_compatible_gradient(
                            parameter,
                            parameter.grad,
                        )
                        compatible_existing.add_(compatible_gradient)
                        parameter.grad = compatible_existing
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
