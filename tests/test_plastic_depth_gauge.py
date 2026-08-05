# vvv THOG
from __future__ import annotations

import math

import pytest
import torch

from sheet.basis import (
    chebyshev_first_kind_basis,
    deterministic_reduced_qr,
    differentiable_chebyshev_first_kind_basis,
    normalized_coordinates,
)
from sheet.plastic_depth_gauge import (
    apply_depth_coefficient_transform,
    chebyshev_gauss_nodes,
    gauge_error,
    stabilized_chebyshev_affine_change_of_chart,
)


def _inverse_r(reference_sample_count: int, order: int) -> torch.Tensor:
    coordinates = normalized_coordinates(
        reference_sample_count,
        dtype=torch.float64,
        device="cpu",
    )
    raw = chebyshev_first_kind_basis(coordinates, order)
    _, r = deterministic_reduced_qr(raw)
    return torch.linalg.inv(r)


def _evaluate(
    coordinates: torch.Tensor,
    coefficients: torch.Tensor,
    inverse_r: torch.Tensor,
) -> torch.Tensor:
    order = int(coefficients.shape[-1])
    basis = differentiable_chebyshev_first_kind_basis(coordinates, order) @ inverse_r
    return torch.matmul(basis, coefficients.transpose(0, 1)).transpose(0, 1)


@pytest.mark.parametrize("order", [1, 2, 4, 8, 16])
def test_affine_change_of_chart_identity_is_identity(order: int) -> None:
    inverse_r = _inverse_r(max(order, 16), order)
    transform = stabilized_chebyshev_affine_change_of_chart(
        inverse_r,
        old_from_new_scale=1.0,
        old_from_new_shift=0.0,
    )

    assert torch.allclose(
        transform,
        torch.eye(order, dtype=torch.float64),
        atol=2.0e-12,
        rtol=2.0e-12,
    )


@pytest.mark.parametrize("order", [2, 4, 8, 16])
@pytest.mark.parametrize(
    ("scale", "shift"),
    [
        (0.5, -0.5),
        (0.75, -0.25),
        (1.1, 0.1),
        (1.5, 0.5),
        (0.9, 0.03),
    ],
)
def test_affine_change_of_chart_preserves_stabilized_field(
    order: int,
    scale: float,
    shift: float,
) -> None:
    inverse_r = _inverse_r(max(order, 19), order)
    transform = stabilized_chebyshev_affine_change_of_chart(
        inverse_r,
        old_from_new_scale=scale,
        old_from_new_shift=shift,
    )
    generator = torch.Generator(device="cpu").manual_seed(9100 + order)
    old_coefficients = torch.randn(7, order, generator=generator, dtype=torch.float64)
    new_coefficients = apply_depth_coefficient_transform(old_coefficients, transform)
    new_coordinates = torch.linspace(-1.0, 1.0, 257, dtype=torch.float64)
    old_coordinates = scale * new_coordinates + shift

    old_values = _evaluate(old_coordinates, old_coefficients, inverse_r)
    new_values = _evaluate(new_coordinates, new_coefficients, inverse_r)
    error = gauge_error(old_values, new_values)

    scale_aware_absolute_tolerance = max(
        2.0e-9,
        float(old_values.abs().max().item()) * 5.0e-15,
    )
    assert error.max_absolute_error < scale_aware_absolute_tolerance
    if scale <= 1.1:
        assert torch.allclose(old_values, new_values, atol=2.0e-9, rtol=2.0e-12)


def test_apply_depth_coefficient_transform_supports_nonfinal_axis() -> None:
    inverse_r = _inverse_r(12, 6)
    transform = stabilized_chebyshev_affine_change_of_chart(
        inverse_r,
        old_from_new_scale=1.2,
        old_from_new_shift=0.2,
    )
    coefficients = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3)

    transformed = apply_depth_coefficient_transform(
        coefficients,
        transform,
        depth_axis=1,
    )

    assert transformed.shape == coefficients.shape
    assert transformed.dtype == torch.float64


def test_chebyshev_gauss_nodes_are_strictly_inside_interval() -> None:
    nodes = chebyshev_gauss_nodes(16)

    assert nodes.dtype == torch.float64
    assert bool((nodes < 1.0).all().item())
    assert bool((nodes > -1.0).all().item())
    assert len(torch.unique(nodes)) == 16


def test_stored_float32_transform_is_close_to_pretransform_values() -> None:
    order = 16
    inverse_r = _inverse_r(144, order)
    scale = 12.0 / 11.0
    shift = scale - 1.0
    transform = stabilized_chebyshev_affine_change_of_chart(
        inverse_r,
        old_from_new_scale=scale,
        old_from_new_shift=shift,
    )
    generator = torch.Generator(device="cpu").manual_seed(147)
    old_coefficients = torch.randn(32, order, generator=generator, dtype=torch.float32)
    transformed64 = apply_depth_coefficient_transform(old_coefficients, transform)
    stored = transformed64.to(dtype=torch.float32)
    new_coordinates = torch.linspace(-1.0, 1.0, 129, dtype=torch.float64)
    old_coordinates = scale * new_coordinates + shift

    old_values = _evaluate(
        old_coordinates,
        old_coefficients.to(dtype=torch.float64),
        inverse_r,
    )
    stored_values = _evaluate(
        new_coordinates,
        stored.to(dtype=torch.float64),
        inverse_r,
    )

    assert torch.allclose(old_values, stored_values, atol=2.0e-5, rtol=2.0e-5)


def test_repeated_add_then_subtract_reexpression_is_stable() -> None:
    order = 16
    inverse_r = _inverse_r(144, order)
    add_scale = 12.0 / 11.0
    add_shift = add_scale - 1.0
    subtract_scale = 1.0 / add_scale
    subtract_shift = subtract_scale - 1.0
    add_transform = stabilized_chebyshev_affine_change_of_chart(
        inverse_r,
        old_from_new_scale=add_scale,
        old_from_new_shift=add_shift,
    )
    subtract_transform = stabilized_chebyshev_affine_change_of_chart(
        inverse_r,
        old_from_new_scale=subtract_scale,
        old_from_new_shift=subtract_shift,
    )
    generator = torch.Generator(device="cpu").manual_seed(1144)
    original = torch.randn(11, order, generator=generator, dtype=torch.float64)
    current = original

    for _ in range(100):
        current = apply_depth_coefficient_transform(current, add_transform)
        current = apply_depth_coefficient_transform(current, subtract_transform)

    assert torch.allclose(current, original, atol=5.0e-9, rtol=5.0e-9)


@pytest.mark.parametrize(
    ("scale", "shift"),
    [(0.0, 0.0), (-1.0, 0.0), (math.inf, 0.0), (1.0, math.nan)],
)
def test_affine_change_of_chart_rejects_invalid_mapping(
    scale: float,
    shift: float,
) -> None:
    inverse_r = _inverse_r(8, 4)

    with pytest.raises(ValueError):
        stabilized_chebyshev_affine_change_of_chart(
            inverse_r,
            old_from_new_scale=scale,
            old_from_new_shift=shift,
        )
# ^^^ THOG
