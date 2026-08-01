# vvv THOG
from __future__ import annotations

from typing import Dict

import pytest
import torch

from sheet.hyperblock import (
    HyperblockBasisTables,
    HyperblockOrders,
    ResolvedHyperblockPlan,
    materialize_attention_family_layer,
    materialize_mlp_family_layer,
    materialize_regions_reference,
    materialize_regions_staged,
)


def _coefficients(plan: ResolvedHyperblockPlan, *, requires_grad: bool = False):
    generator = torch.Generator().manual_seed(1234)
    tensors = []
    for name in ("common", "attention", "mlp"):
        tensor = torch.randn(
            plan.coefficient_shapes[name],
            dtype=torch.float64,
            generator=generator,
        )
        tensor.requires_grad_(requires_grad)
        tensors.append(tensor)
    return tuple(tensors)


def _bases(plan: ResolvedHyperblockPlan) -> Dict[str, torch.Tensor]:
    return HyperblockBasisTables(plan, runtime_dtype=torch.float64).as_mapping()


def _plan() -> ResolvedHyperblockPlan:
    return ResolvedHyperblockPlan(
        n_layer=3,
        n_embd=6,
        n_head=2,
        mlp_hidden_multiplier=2,
        orders=HyperblockOrders(
            depth=3,
            d_model=4,
            mlp_hidden=5,
            attention_head=2,
            attention_head_channel=3,
        ),
    )


def _scalar_reference_regions(
    common_coefficients: torch.Tensor,
    attention_coefficients: torch.Tensor,
    mlp_coefficients: torch.Tensor,
    bases: Dict[str, torch.Tensor],
):
    family_common = bases["family_common"]
    family_attention = bases["family_attention"]
    family_mlp = bases["family_mlp"]
    depth = bases["depth"]
    d_model = bases["d_model"]
    head = bases["attention_head"]
    channel = bases["attention_head_channel"]
    hidden = bases["mlp_hidden"]

    common_values = []
    for family_index in range(family_common.shape[0]):
        for layer_index in range(depth.shape[0]):
            for d_model_index in range(d_model.shape[0]):
                value = common_coefficients.new_zeros(())
                for family_mode in range(family_common.shape[1]):
                    for depth_mode in range(depth.shape[1]):
                        for d_model_mode in range(d_model.shape[1]):
                            value = value + (
                                common_coefficients[family_mode, depth_mode, d_model_mode]
                                * family_common[family_index, family_mode]
                                * depth[layer_index, depth_mode]
                                * d_model[d_model_index, d_model_mode]
                            )
                common_values.append(value)
    common = torch.stack(common_values).reshape(
        family_common.shape[0],
        depth.shape[0],
        d_model.shape[0],
    )

    attention_values = []
    channel_order = channel.shape[1]
    for family_index in range(family_attention.shape[0]):
        for layer_index in range(depth.shape[0]):
            for d_model_index in range(d_model.shape[0]):
                for head_index in range(head.shape[0]):
                    for channel_index in range(channel.shape[0]):
                        value = attention_coefficients.new_zeros(())
                        for family_mode in range(family_attention.shape[1]):
                            for depth_mode in range(depth.shape[1]):
                                for d_model_mode in range(d_model.shape[1]):
                                    for head_mode in range(head.shape[1]):
                                        for channel_mode in range(channel.shape[1]):
                                            flat_mode = head_mode * channel_order + channel_mode
                                            if flat_mode == 0:
                                                continue
                                            value = value + (
                                                attention_coefficients[
                                                    family_mode,
                                                    depth_mode,
                                                    d_model_mode,
                                                    flat_mode - 1,
                                                ]
                                                * family_attention[family_index, family_mode]
                                                * depth[layer_index, depth_mode]
                                                * d_model[d_model_index, d_model_mode]
                                                * head[head_index, head_mode]
                                                * channel[channel_index, channel_mode]
                                            )
                        attention_values.append(value)
    attention_extension = torch.stack(attention_values).reshape(
        family_attention.shape[0],
        depth.shape[0],
        d_model.shape[0],
        head.shape[0],
        channel.shape[0],
    )

    mlp_values = []
    for family_index in range(family_mlp.shape[0]):
        for layer_index in range(depth.shape[0]):
            for d_model_index in range(d_model.shape[0]):
                for hidden_index in range(hidden.shape[0]):
                    value = mlp_coefficients.new_zeros(())
                    for family_mode in range(family_mlp.shape[1]):
                        for depth_mode in range(depth.shape[1]):
                            for d_model_mode in range(d_model.shape[1]):
                                for hidden_mode in range(1, hidden.shape[1]):
                                    value = value + (
                                        mlp_coefficients[
                                            family_mode,
                                            depth_mode,
                                            d_model_mode,
                                            hidden_mode - 1,
                                        ]
                                        * family_mlp[family_index, family_mode]
                                        * depth[layer_index, depth_mode]
                                        * d_model[d_model_index, d_model_mode]
                                        * hidden[hidden_index, hidden_mode]
                                    )
                    mlp_values.append(value)
    mlp_extension = torch.stack(mlp_values).reshape(
        family_mlp.shape[0],
        depth.shape[0],
        d_model.shape[0],
        hidden.shape[0],
    )
    return common, attention_extension, mlp_extension


def test_einsum_reference_matches_literal_scalar_definition() -> None:
    plan = ResolvedHyperblockPlan(
        n_layer=2,
        n_embd=3,
        n_head=1,
        mlp_hidden_multiplier=2,
        orders=HyperblockOrders(
            depth=2,
            d_model=2,
            mlp_hidden=2,
            attention_head=1,
            attention_head_channel=2,
            common_family=2,
            attention_family=2,
            mlp_family=1,
        ),
    )
    coefficients = _coefficients(plan)
    bases = _bases(plan)
    scalar = _scalar_reference_regions(*coefficients, bases)
    reference = materialize_regions_reference(*coefficients, bases)
    for scalar_region, reference_region in zip(
        scalar,
        (reference.common, reference.attention_extension, reference.mlp_extension),
    ):
        torch.testing.assert_close(scalar_region, reference_region, rtol=1.0e-12, atol=1.0e-12)


def _finite_difference_objective(
    common: torch.Tensor,
    attention: torch.Tensor,
    mlp: torch.Tensor,
    bases: Dict[str, torch.Tensor],
) -> torch.Tensor:
    attention_value = materialize_attention_family_layer(
        common,
        attention,
        bases,
        family_index=1,
        layer_index=1,
    )
    mlp_value = materialize_mlp_family_layer(
        common,
        mlp,
        bases,
        common_family_index=5,
        mlp_family_index=1,
        layer_index=0,
    )
    return attention_value.square().mean() + 0.7 * mlp_value.square().mean()


@pytest.mark.parametrize(
    ("region_index", "coefficient_index"),
    (
        (0, (1, 1, 2)),
        (1, (2, 1, 1, 3)),
        (2, (1, 1, 2, 2)),
    ),
)
def test_all_coefficient_regions_match_finite_difference_gradients(
    region_index: int,
    coefficient_index,
) -> None:
    plan = _plan()
    coefficients = list(_coefficients(plan, requires_grad=True))
    bases = _bases(plan)
    objective = _finite_difference_objective(*coefficients, bases)
    analytic = torch.autograd.grad(objective, coefficients[region_index])[0][coefficient_index]

    epsilon = 1.0e-6
    numerical_values = []
    for direction in (1.0, -1.0):
        perturbed = [value.detach().clone() for value in coefficients]
        perturbed[region_index][coefficient_index] += direction * epsilon
        numerical_values.append(_finite_difference_objective(*perturbed, bases))
    numerical = (numerical_values[0] - numerical_values[1]) / (2.0 * epsilon)
    torch.testing.assert_close(analytic, numerical, rtol=2.0e-5, atol=2.0e-7)


def test_zero_unique_orders_reduce_cleanly_to_the_common_field() -> None:
    plan = ResolvedHyperblockPlan(
        n_layer=2,
        n_embd=4,
        n_head=2,
        mlp_hidden_multiplier=2,
        orders=HyperblockOrders(
            depth=2,
            d_model=3,
            mlp_hidden=1,
            attention_head=1,
            attention_head_channel=1,
            common_family=3,
            attention_family=2,
            mlp_family=1,
        ),
    )
    assert plan.coefficient_shapes["attention"][-1] == 0
    assert plan.coefficient_shapes["mlp"][-1] == 0
    common, attention, mlp = _coefficients(plan)
    regions = materialize_regions_staged(common, attention, mlp, _bases(plan))
    assert torch.count_nonzero(regions.attention_extension) == 0
    assert torch.count_nonzero(regions.mlp_extension) == 0
# ^^^ THOG
