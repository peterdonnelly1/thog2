# vvv THOG
from __future__ import annotations

import copy
import math

import pytest
import torch

from sheet.model import SheetGPT
from sheet.plastic_depth_gauge import apply_depth_coefficient_transform_chunked
from sheet.plastic_depth_optimizer import (
    commit_plastic_depth_adamw_transition,
    prepare_plastic_depth_adamw_transition,
)
from tests.test_plastic_depth import plastic_sheet_config


def _model_and_optimizer():
    torch.manual_seed(771)
    model = SheetGPT(
        plastic_sheet_config(
            n_layer=5,
            depth_order=4,
            plastic__layers_to_sample=None,
            plastic__do_learn_layer_count=True,
            plastic__initial_layer_count=3,
            plastic__max_permitted_layers=5,
            plastic__layer_sampling_initialisation="random",
        )
    )
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=1.0e-3,
        betas=(0.9, 0.95),
        device_type="cpu",
    )
    for parameter in model.parameters():
        parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer


def _clone_state(state):
    return {
        key: value.detach().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        for key, value in state.items()
    }


def test_adamw_migration_uses_covector_and_squared_covector_rules() -> None:
    model, optimizer = _model_and_optimizer()
    model_transition = model.prepare_plastic_depth_count_transition(4)
    first = model_transition.replacements[0]
    parameter = model.trajectory.coefficients[first.name]
    state = optimizer.state[parameter]
    state["exp_avg"].copy_(
        torch.linspace(-0.25, 0.5, state["exp_avg"].numel()).reshape_as(state["exp_avg"])
    )
    state["exp_avg_sq"].copy_(
        torch.linspace(0.01, 0.4, state["exp_avg_sq"].numel()).reshape_as(state["exp_avg_sq"])
    )
    step_before = state["step"].detach().clone()
    prepared = prepare_plastic_depth_adamw_transition(model, optimizer, model_transition)
    expected_first = apply_depth_coefficient_transform_chunked(
        state["exp_avg"],
        prepared.covector_transform,
        depth_axis=first.depth_axis,
        output_dtype=state["exp_avg"].dtype,
    )
    expected_second = apply_depth_coefficient_transform_chunked(
        state["exp_avg_sq"],
        prepared.squared_covector_transform,
        depth_axis=first.depth_axis,
        output_dtype=state["exp_avg_sq"].dtype,
    )

    report = commit_plastic_depth_adamw_transition(model, optimizer, prepared)

    assert report["adamw_state_migration_mode"] == "transform"
    torch.testing.assert_close(state["exp_avg"], expected_first, rtol=0.0, atol=0.0)
    torch.testing.assert_close(state["exp_avg_sq"], expected_second, rtol=0.0, atol=0.0)
    torch.testing.assert_close(state["step"], step_before, rtol=0.0, atol=0.0)
    assert bool((state["exp_avg_sq"] >= 0.0).all().item())
    assert model.trajectory.plastic_sampling.current_active_layers == 4


def test_adamw_migration_leaves_unaffected_parameter_state_bit_identical() -> None:
    model, optimizer = _model_and_optimizer()
    unaffected = model.transformer.wte.weight
    before = _clone_state(optimizer.state[unaffected])
    model_transition = model.prepare_plastic_depth_count_transition(4)
    prepared = prepare_plastic_depth_adamw_transition(model, optimizer, model_transition)

    commit_plastic_depth_adamw_transition(model, optimizer, prepared)

    after = optimizer.state[unaffected]
    assert set(after) == set(before)
    for key, expected in before.items():
        value = after[key]
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(value, expected, rtol=0.0, atol=0.0)
        else:
            assert value == expected


def test_ill_conditioned_migration_resets_moments_and_retains_step() -> None:
    model, optimizer = _model_and_optimizer()
    model_transition = model.prepare_plastic_depth_count_transition(4)
    first = model_transition.replacements[0]
    parameter = model.trajectory.coefficients[first.name]
    state = optimizer.state[parameter]
    step_before = state["step"].detach().clone()

    prepared = prepare_plastic_depth_adamw_transition(
        model,
        optimizer,
        model_transition,
        maximum_condition_number=0.5,
    )
    report = commit_plastic_depth_adamw_transition(model, optimizer, prepared)

    assert report["adamw_state_migration_mode"] == "reset"
    assert report["adamw_state_reset_parameter_count"] > 0
    assert report["adamw_state_fallback_reason"]
    torch.testing.assert_close(state["exp_avg"], torch.zeros_like(state["exp_avg"]), rtol=0.0, atol=0.0)
    torch.testing.assert_close(state["exp_avg_sq"], torch.zeros_like(state["exp_avg_sq"]), rtol=0.0, atol=0.0)
    torch.testing.assert_close(state["step"], step_before, rtol=0.0, atol=0.0)
    assert model.trajectory.plastic_sampling.current_active_layers == 4


def test_nonfinite_adamw_state_selects_reset_fallback() -> None:
    model, optimizer = _model_and_optimizer()
    model_transition = model.prepare_plastic_depth_count_transition(4)
    first = model_transition.replacements[0]
    parameter = model.trajectory.coefficients[first.name]
    optimizer.state[parameter]["exp_avg"].view(-1)[0] = float("nan")

    prepared = prepare_plastic_depth_adamw_transition(model, optimizer, model_transition)
    report = commit_plastic_depth_adamw_transition(model, optimizer, prepared)

    assert report["adamw_state_migration_mode"] == "reset"
    assert math.isfinite(float(optimizer.state[parameter]["exp_avg"].sum().item()))


def test_stale_adamw_state_aborts_before_model_count_change() -> None:
    model, optimizer = _model_and_optimizer()
    model_transition = model.prepare_plastic_depth_count_transition(4)
    prepared = prepare_plastic_depth_adamw_transition(model, optimizer, model_transition)
    first = prepared.replacements[0]
    optimizer.state[first.parameter]["exp_avg"].add_(1.0e-4)

    with pytest.raises(RuntimeError, match="state changed"):
        commit_plastic_depth_adamw_transition(model, optimizer, prepared)

    assert model.trajectory.plastic_sampling.current_active_layers == 3


def test_amsgrad_second_moment_uses_squared_covector_rule() -> None:
    model, optimizer = _model_and_optimizer()
    model_transition = model.prepare_plastic_depth_count_transition(4)
    first = model_transition.replacements[0]
    parameter = model.trajectory.coefficients[first.name]
    state = optimizer.state[parameter]
    state["max_exp_avg_sq"] = torch.linspace(
        0.02,
        0.8,
        state["exp_avg_sq"].numel(),
    ).reshape_as(state["exp_avg_sq"])
    prepared = prepare_plastic_depth_adamw_transition(model, optimizer, model_transition)
    expected = apply_depth_coefficient_transform_chunked(
        state["max_exp_avg_sq"],
        prepared.squared_covector_transform,
        depth_axis=first.depth_axis,
        output_dtype=state["max_exp_avg_sq"].dtype,
    )

    commit_plastic_depth_adamw_transition(model, optimizer, prepared)

    torch.testing.assert_close(state["max_exp_avg_sq"], expected, rtol=0.0, atol=0.0)
    assert bool((state["max_exp_avg_sq"] >= 0.0).all().item())


def test_uninitialized_adamw_state_remains_uninitialized() -> None:
    torch.manual_seed(772)
    model = SheetGPT(
        plastic_sheet_config(
            n_layer=5,
            depth_order=4,
            plastic__layers_to_sample=None,
            plastic__do_learn_layer_count=True,
            plastic__initial_layer_count=3,
            plastic__max_permitted_layers=5,
        )
    )
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=1.0e-3,
        betas=(0.9, 0.95),
        device_type="cpu",
    )
    model_transition = model.prepare_plastic_depth_count_transition(4)
    prepared = prepare_plastic_depth_adamw_transition(model, optimizer, model_transition)

    report = commit_plastic_depth_adamw_transition(model, optimizer, prepared)

    assert report["adamw_state_migrated_parameter_count"] == 0
    assert report["adamw_state_reset_parameter_count"] == 0
    for replacement in prepared.replacements:
        assert optimizer.state.get(replacement.parameter, {}) == {}


def test_stale_adamw_step_aborts_before_model_count_change() -> None:
    model, optimizer = _model_and_optimizer()
    model_transition = model.prepare_plastic_depth_count_transition(4)
    prepared = prepare_plastic_depth_adamw_transition(model, optimizer, model_transition)
    first = prepared.replacements[0]
    optimizer.state[first.parameter]["step"].add_(1)

    with pytest.raises(RuntimeError, match="state changed"):
        commit_plastic_depth_adamw_transition(model, optimizer, prepared)

    assert model.trajectory.plastic_sampling.current_active_layers == 3


def test_non_adamw_optimizer_is_rejected() -> None:
    model, _ = _model_and_optimizer()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-3)
    model_transition = model.prepare_plastic_depth_count_transition(4)

    with pytest.raises(TypeError, match="stock torch.optim.AdamW"):
        prepare_plastic_depth_adamw_transition(model, optimizer, model_transition)


def test_stock_adamw_can_step_after_transformed_and_reset_migrations() -> None:
    for maximum_condition_number in (1.0e8, 0.5):
        model, optimizer = _model_and_optimizer()
        model_transition = model.prepare_plastic_depth_count_transition(4)
        prepared = prepare_plastic_depth_adamw_transition(
            model,
            optimizer,
            model_transition,
            maximum_condition_number=maximum_condition_number,
        )
        commit_plastic_depth_adamw_transition(model, optimizer, prepared)
        for parameter in model.parameters():
            parameter.grad = torch.randn_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        for parameter in model.trajectory.coefficients.values():
            assert bool(torch.isfinite(parameter).all().item())
# ^^^ THOG
