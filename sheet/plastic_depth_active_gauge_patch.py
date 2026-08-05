# vvv THOG
from __future__ import annotations

import math
from typing import Any

import torch

from .basis import chebyshev_first_kind_basis
from .depth_trajectory import (
    DepthTrajectory,
    PlasticDepthCoefficientReplacement,
    PlasticDepthModelTransition,
)
from .plastic_depth import public_to_internal_depth
from .plastic_depth_gauge import (
    apply_depth_coefficient_transform_chunked,
    stabilized_chebyshev_affine_change_of_chart,
)


# vvv THOG learned-count transitions verify only realised active samples, not the dormant probe or synthetic interior chart points
def _active_prefix_verification_coordinates(
    trajectory: DepthTrajectory,
    geometry: Any,
) -> torch.Tensor:
    if trajectory.plastic_sampling is None:
        raise RuntimeError("PLASTIC DEPTH is not enabled")
    public_coordinates = trajectory.plastic_sampling._public_coordinates_from_raw(
        geometry.new_active_layers,
        include_probe=(
            trajectory.plastic_sampling.learn_layer_count
            and geometry.new_active_layers < trajectory.plastic_sampling.maximum_layers
        ),
        raw_intervals=geometry.proposed_raw_intervals,
    )
    active_public = public_coordinates[: geometry.new_active_layers]
    active_internal = public_to_internal_depth(active_public).to(dtype=torch.float64)
    return torch.unique(active_internal, sorted=True)


# vvv THOG unstable count re-gauges must not kill training; preserve coefficients and commit only the sampling geometry
def _geometry_only_plastic_depth_transition(
    trajectory: DepthTrajectory,
    geometry: Any,
    *,
    verification_coordinate_count: int,
) -> PlasticDepthModelTransition:
    identity = torch.eye(
        trajectory.config.depth_order,
        dtype=torch.float64,
        device=trajectory.plastic_depth_inverse_r.device,
    )
    return PlasticDepthModelTransition(
        geometry=geometry,
        transform=identity,
        replacements=(),
        verification_coordinate_count=verification_coordinate_count,
        maximum_absolute_error=float("nan"),
        maximum_relative_error=float("nan"),
        absolute_tolerance=float("nan"),
        relative_tolerance=float("nan"),
        condition_number=1.0,
    )
# ^^^ THOG


def _prepare_plastic_depth_count_transition_active_prefix(
    self: DepthTrajectory,
    new_active_layers: int,
    *,
    maximum_series_per_chunk: int = 65536,
) -> PlasticDepthModelTransition:
    if not self.plastic_enabled or self.plastic_sampling is None:
        raise RuntimeError("PLASTIC DEPTH is not enabled")
    geometry = self.plastic_sampling.prepare_count_transition(new_active_layers)
    if geometry.new_active_layers == geometry.previous_active_layers:
        raise ValueError("model-wide PLASTIC DEPTH re-gauge requires a count change")

    transform = stabilized_chebyshev_affine_change_of_chart(
        self.plastic_depth_inverse_r.to(dtype=torch.float64),
        old_from_new_scale=geometry.old_from_new_scale,
        old_from_new_shift=geometry.old_from_new_shift,
    )
    condition_number = float(torch.linalg.cond(transform).item())
    if not math.isfinite(condition_number):
        raise RuntimeError("PLASTIC DEPTH gauge transform is singular or non-finite")

    verification_new = _active_prefix_verification_coordinates(self, geometry)
    verification_old = (
        geometry.old_from_new_scale * verification_new
        + geometry.old_from_new_shift
    )
    inverse_r = self.plastic_depth_inverse_r.to(
        device=verification_new.device,
        dtype=torch.float64,
    )
    old_basis = (
        chebyshev_first_kind_basis(verification_old, self.config.depth_order)
        @ inverse_r
    )
    new_basis = (
        chebyshev_first_kind_basis(verification_new, self.config.depth_order)
        @ inverse_r
    )

    replacements = []
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    absolute_tolerance = 0.0
    relative_tolerance = 0.0
    for item in self.metadata:
        depth_axis = self._plastic_generated_depth_axis(item)
        if depth_axis is None:
            continue
        parameter = self.coefficients[item.name]
        family_atol, family_rtol = self._plastic_gauge_tolerances(parameter.dtype)
        transformed = apply_depth_coefficient_transform_chunked(
            parameter.detach(),
            transform,
            depth_axis=depth_axis,
            output_dtype=parameter.dtype,
            maximum_series_per_chunk=maximum_series_per_chunk,
        )
        try:
            family_absolute, family_relative = self._verify_plastic_depth_replacement(
                source=parameter,
                candidate=transformed,
                depth_axis=depth_axis,
                old_basis=old_basis.to(parameter.device),
                new_basis=new_basis.to(parameter.device),
                absolute_tolerance=family_atol,
                relative_tolerance=family_rtol,
                maximum_series_per_chunk=maximum_series_per_chunk,
            )
        except RuntimeError as error:
            if "PLASTIC DEPTH gauge verification failed" not in str(error):
                raise
            return _geometry_only_plastic_depth_transition(
                self,
                geometry,
                verification_coordinate_count=int(verification_new.numel()),
            )
        replacements.append(
            PlasticDepthCoefficientReplacement(
                name=item.name,
                depth_axis=depth_axis,
                source_version=int(parameter._version),
                transformed=transformed,
            )
        )
        maximum_absolute_error = max(maximum_absolute_error, family_absolute)
        maximum_relative_error = max(maximum_relative_error, family_relative)
        absolute_tolerance = max(absolute_tolerance, family_atol)
        relative_tolerance = max(relative_tolerance, family_rtol)

    return PlasticDepthModelTransition(
        geometry=geometry,
        transform=transform,
        replacements=tuple(replacements),
        verification_coordinate_count=int(verification_new.numel()),
        maximum_absolute_error=maximum_absolute_error,
        maximum_relative_error=maximum_relative_error,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        condition_number=condition_number,
    )


DepthTrajectory.prepare_plastic_depth_count_transition = _prepare_plastic_depth_count_transition_active_prefix
# ^^^ THOG
