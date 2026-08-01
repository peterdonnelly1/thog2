# vvv THOG
from __future__ import annotations

import math

import pytest
import torch

from sheet.hyperblock import CoupledFieldTrajectory, HyperblockOrders, ResolvedHyperblockPlan
from sheet.update_retained_materializations import attach_update_retained_materializations


def _plan() -> ResolvedHyperblockPlan:
    return ResolvedHyperblockPlan(
        n_layer=3,
        n_embd=8,
        n_head=2,
        mlp_hidden_multiplier=2,
        orders=HyperblockOrders(
            depth=3,
            d_model=5,
            mlp_hidden=6,
            attention_head=2,
            attention_head_channel=4,
        ),
    )


def _trajectory(*, bias: bool = True) -> CoupledFieldTrajectory:
    torch.manual_seed(123)
    return CoupledFieldTrajectory(
        _plan(),
        bias=bias,
        runtime_dtype=torch.float64,
    )


def test_materialized_family_shapes_match_nanogpt() -> None:
    trajectory = _trajectory()
    for layer_index in range(trajectory.plan.n_layer):
        assert trajectory.materialize("attention_input_weight", layer_index).shape == (24, 8)
        assert trajectory.materialize("attention_output_weight", layer_index).shape == (8, 8)
        assert trajectory.materialize("mlp_expansion_weight", layer_index).shape == (16, 8)
        assert trajectory.materialize("mlp_contraction_weight", layer_index).shape == (8, 16)
        assert trajectory.materialize_vector("ln_1_weight", layer_index).shape == (8,)
        assert trajectory.materialize_vector("attention_input_bias", layer_index).shape == (24,)
        assert trajectory.materialize_vector("mlp_expansion_bias", layer_index).shape == (16,)


def test_no_bias_trajectory_omits_only_bias_vectors() -> None:
    trajectory = _trajectory(bias=False)
    assert set(trajectory.vector_parameters) == {"ln_1_weight", "ln_2_weight"}
    with pytest.raises(KeyError, match="unknown HYPERBLOCK vector family"):
        trajectory.materialize_vector("attention_input_bias", 0)


def test_layer_norm_and_bias_initialization_matches_nanogpt_contract() -> None:
    trajectory = _trajectory()
    assert torch.equal(
        trajectory.vector_parameters["ln_1_weight"],
        torch.ones_like(trajectory.vector_parameters["ln_1_weight"]),
    )
    assert torch.equal(
        trajectory.vector_parameters["ln_2_weight"],
        torch.ones_like(trajectory.vector_parameters["ln_2_weight"]),
    )
    for name, parameter in trajectory.vector_parameters.items():
        if name not in {"ln_1_weight", "ln_2_weight"}:
            assert torch.equal(parameter, torch.zeros_like(parameter))


def test_common_coefficients_receive_attention_and_mlp_gradients() -> None:
    trajectory = _trajectory()
    attention_loss = trajectory.materialize("attention_output_weight", 1).square().sum()
    mlp_loss = trajectory.materialize("mlp_contraction_weight", 2).square().sum()

    attention_gradient = torch.autograd.grad(
        attention_loss,
        trajectory.coefficients["common"],
        retain_graph=True,
    )[0]
    mlp_gradient = torch.autograd.grad(
        mlp_loss,
        trajectory.coefficients["common"],
        retain_graph=True,
    )[0]
    joint_gradient = torch.autograd.grad(
        attention_loss + mlp_loss,
        trajectory.coefficients["common"],
    )[0]

    assert attention_gradient.norm() > 0.0
    assert mlp_gradient.norm() > 0.0
    assert torch.allclose(
        joint_gradient,
        attention_gradient + mlp_gradient,
        atol=1.0e-11,
        rtol=1.0e-11,
    )


def test_residual_scaling_changes_only_output_and_down_families() -> None:
    trajectory = _trajectory()
    before = {
        name: trajectory.materialize(name, 1).detach().clone()
        for name in (
            "attention_input_weight",
            "attention_output_weight",
            "mlp_expansion_weight",
            "mlp_contraction_weight",
        )
    }
    previous_std = 0.02 / math.sqrt(2.0 * trajectory.plan.n_layer)
    new_std = previous_std * 0.5
    trajectory.apply_residual_init_scaling(new_std)
    after = {
        name: trajectory.materialize(name, 1).detach().clone()
        for name in before
    }

    assert torch.allclose(after["attention_input_weight"], before["attention_input_weight"])
    assert torch.allclose(after["mlp_expansion_weight"], before["mlp_expansion_weight"])
    assert torch.allclose(
        after["attention_output_weight"],
        before["attention_output_weight"] * 0.5,
        atol=1.0e-11,
        rtol=1.0e-11,
    )
    assert torch.allclose(
        after["mlp_contraction_weight"],
        before["mlp_contraction_weight"] * 0.5,
        atol=1.0e-11,
        rtol=1.0e-11,
    )


def test_parameter_accounting_is_exact() -> None:
    trajectory = _trajectory()
    assert trajectory.sheet_parameter_count() == trajectory.plan.coefficient_counts["total"]
    assert trajectory.matrix_sheet_parameter_count() == trajectory.sheet_parameter_count()
    assert trajectory.matrix_dense_equivalent_count() == 12 * 3 * 8 * 8
    assert trajectory.conventional_vector_parameter_count() == sum(
        parameter.numel() for parameter in trajectory.vector_parameters.values()
    )
    assert trajectory.dense_equivalent_count() == (
        trajectory.matrix_dense_equivalent_count()
        + trajectory.conventional_vector_parameter_count()
    )


def test_update_retained_materialization_projects_shared_gradients_once() -> None:
    trajectory = _trajectory()
    controller = attach_update_retained_materializations(trajectory, enabled=True)
    assert controller.begin() is True
    attention = trajectory.materialize("attention_output_weight", 0)
    mlp = trajectory.materialize("mlp_contraction_weight", 0)
    repeated_attention = trajectory.materialize("attention_output_weight", 0)
    assert repeated_attention is attention

    loss = attention.square().sum() + mlp.square().sum()
    loss.backward()
    projected = controller.finalize()

    assert any(parameter is trajectory.coefficients["common"] for parameter in projected)
    assert any(parameter is trajectory.coefficients["attention"] for parameter in projected)
    assert any(parameter is trajectory.coefficients["mlp"] for parameter in projected)
    assert trajectory.coefficients["common"].grad is not None
    assert trajectory.coefficients["attention"].grad is not None
    assert trajectory.coefficients["mlp"].grad is not None
    assert controller.materialization_count == 2
    assert controller.active is False


def test_state_dict_regenerates_basis_tables() -> None:
    trajectory = _trajectory()
    state = trajectory.state_dict()
    assert not any(name.startswith("bases.") for name in state)
    restored = _trajectory()
    restored.load_state_dict(state)
    for name in (
        "attention_input_weight",
        "attention_output_weight",
        "mlp_expansion_weight",
        "mlp_contraction_weight",
    ):
        assert torch.allclose(
            restored.materialize(name, 2),
            trajectory.materialize(name, 2),
            atol=1.0e-12,
            rtol=1.0e-12,
        )
# ^^^ THOG

# vvv THOG

def test_initial_materialized_weight_stds_match_nanogpt_policy() -> None:
    plan = ResolvedHyperblockPlan(
        n_layer=8,
        n_embd=32,
        n_head=4,
        mlp_hidden_multiplier=4,
        orders=HyperblockOrders(
            depth=8,
            d_model=16,
            mlp_hidden=16,
            attention_head=4,
            attention_head_channel=8,
        ),
    )
    torch.manual_seed(7123)
    trajectory = CoupledFieldTrajectory(
        plan,
        bias=False,
        runtime_dtype=torch.float64,
    )
    materialized = {
        name: torch.stack(
            [trajectory.materialize(name, layer) for layer in range(plan.n_layer)]
        )
        for name in (
            "attention_input_weight",
            "attention_output_weight",
            "mlp_expansion_weight",
            "mlp_contraction_weight",
        )
    }
    standard_target = 0.02
    residual_target = 0.02 / math.sqrt(2.0 * plan.n_layer)
    assert float(materialized["attention_input_weight"].std().item()) == pytest.approx(
        standard_target,
        rel=0.10,
    )
    assert float(materialized["mlp_expansion_weight"].std().item()) == pytest.approx(
        standard_target,
        rel=0.10,
    )
    assert float(materialized["attention_output_weight"].std().item()) == pytest.approx(
        residual_target,
        rel=0.10,
    )
    assert float(materialized["mlp_contraction_weight"].std().item()) == pytest.approx(
        residual_target,
        rel=0.10,
    )
# ^^^ THOG

# vvv THOG

def test_reduced_family_orders_initialize_and_materialize_all_six_families() -> None:
    plan = ResolvedHyperblockPlan(
        n_layer=3,
        n_embd=8,
        n_head=2,
        mlp_hidden_multiplier=2,
        orders=HyperblockOrders(
            depth=3,
            d_model=5,
            mlp_hidden=6,
            attention_head=2,
            attention_head_channel=4,
            common_family=3,
            attention_family=2,
            mlp_family=1,
        ),
    )
    torch.manual_seed(177)
    trajectory = CoupledFieldTrajectory(
        plan,
        bias=True,
        runtime_dtype=torch.float64,
    )
    assert trajectory.coefficients["common"].shape[0] == 3
    assert trajectory.coefficients["attention"].shape[0] == 2
    assert trajectory.coefficients["mlp"].shape[0] == 1
    for name, expected_shape in {
        "attention_input_weight": (24, 8),
        "attention_output_weight": (8, 8),
        "mlp_expansion_weight": (16, 8),
        "mlp_contraction_weight": (8, 16),
    }.items():
        generated = trajectory.materialize(name, 1)
        assert generated.shape == expected_shape
        assert bool(torch.isfinite(generated).all())
# ^^^ THOG
