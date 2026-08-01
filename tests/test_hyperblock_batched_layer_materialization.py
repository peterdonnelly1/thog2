# vvv THOG
from __future__ import annotations

from types import MethodType

import torch

from sheet.hyperblock import (
    HYPERBLOCK_TOPOLOGY_COUPLED_FIELD_MACHINE,
    CoupledFieldTrajectory,
    HyperblockBasisTables,
    HyperblockOrders,
    ResolvedHyperblockPlan,
    materialize_attention_family_layer,
    materialize_layer_staged,
    materialize_mlp_family_layer,
)
from sheet.model import SheetGPT, SheetGPTConfig
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


def _coefficients(plan: ResolvedHyperblockPlan):
    generator = torch.Generator().manual_seed(260801)
    return tuple(
        torch.randn(
            plan.coefficient_shapes[name],
            dtype=torch.float64,
            generator=generator,
            requires_grad=True,
        )
        for name in ("common", "attention", "mlp")
    )


def test_batched_layer_materialization_matches_every_per_family_reference() -> None:
    plan = _plan()
    common, attention, mlp = _coefficients(plan)
    bases = HyperblockBasisTables(
        plan,
        runtime_dtype=torch.float64,
    ).as_mapping()

    for layer_index in range(plan.n_layer):
        layer = materialize_layer_staged(
            common,
            attention,
            mlp,
            bases,
            layer_index=layer_index,
        )
        for family_index in range(4):
            expected = materialize_attention_family_layer(
                common,
                attention,
                bases,
                family_index=family_index,
                layer_index=layer_index,
            )
            torch.testing.assert_close(
                layer.attention[family_index],
                expected,
                rtol=1.0e-11,
                atol=1.0e-11,
            )
        for mlp_family_index, common_family_index in enumerate((4, 5)):
            expected = materialize_mlp_family_layer(
                common,
                mlp,
                bases,
                common_family_index=common_family_index,
                mlp_family_index=mlp_family_index,
                layer_index=layer_index,
            )
            torch.testing.assert_close(
                layer.mlp[mlp_family_index],
                expected,
                rtol=1.0e-11,
                atol=1.0e-11,
            )

    objective = layer.attention.square().mean() + layer.mlp.square().mean()
    gradients = torch.autograd.grad(objective, (common, attention, mlp))
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)


def test_trajectory_layer_bundle_matches_the_existing_individual_matrix_api() -> None:
    torch.manual_seed(260802)
    trajectory = CoupledFieldTrajectory(
        _plan(),
        bias=True,
        runtime_dtype=torch.float64,
    )
    for layer_index in range(trajectory.plan.n_layer):
        bundle = trajectory.materialize_layer_matrices(layer_index)
        assert tuple(bundle) == trajectory.materialized_matrix_family_names
        for name, matrix in bundle.items():
            torch.testing.assert_close(
                matrix,
                trajectory.materialize(name, layer_index),
                rtol=1.0e-11,
                atol=1.0e-11,
            )


def test_retained_controller_reuses_and_projects_one_layer_bundle() -> None:
    torch.manual_seed(260803)
    trajectory = CoupledFieldTrajectory(
        _plan(),
        bias=True,
        runtime_dtype=torch.float64,
    )
    controller = attach_update_retained_materializations(
        trajectory,
        enabled=True,
    )
    assert controller.begin() is True
    first = trajectory.materialize_layer_matrices(1)
    second = trajectory.materialize_layer_matrices(1)
    assert tuple(first) == trajectory.materialized_matrix_family_names
    assert all(second[name] is first[name] for name in first)
    assert controller.request_count == 8
    assert controller.materialization_count == 4
    assert controller.retained_count == 4

    loss = sum(matrix.square().mean() for matrix in first.values())
    loss.backward()
    projected = controller.finalize()
    assert controller.active is False
    assert controller.retained_count == 0
    assert any(parameter is trajectory.coefficients["common"] for parameter in projected)
    assert any(parameter is trajectory.coefficients["attention"] for parameter in projected)
    assert any(parameter is trajectory.coefficients["mlp"] for parameter in projected)
    assert all(
        parameter.grad is not None
        for parameter in trajectory.coefficients.values()
    )


def test_sheet_gpt_requests_exactly_one_hyperblock_bundle_per_layer() -> None:
    torch.manual_seed(260804)
    model = SheetGPT(
        SheetGPTConfig(
            block_size=4,
            vocab_size=32,
            n_layer=2,
            n_head=2,
            n_embd=8,
            dropout=0.0,
            bias=True,
            hyperblock_topology=HYPERBLOCK_TOPOLOGY_COUPLED_FIELD_MACHINE,
            hyperblock_depth_order=2,
            hyperblock_d_model_order=4,
            hyperblock_mlp_hidden_order=4,
            hyperblock_attention_head_order=2,
            hyperblock_attention_head_channel_order=4,
            fast_discard=True,
        )
    )
    original = model.trajectory.materialize_layer_matrices
    calls = []

    def counted_materialize_layer_matrices(self, layer_index: int):
        calls.append(layer_index)
        return original(layer_index)

    model.trajectory.materialize_layer_matrices = MethodType(
        counted_materialize_layer_matrices,
        model.trajectory,
    )
    tokens = torch.randint(0, 32, (1, 4))
    logits, loss = model(tokens, tokens)
    assert logits.shape == (1, 4, 32)
    assert loss is not None
    loss.backward()
    assert calls == [0, 1]
    assert all(
        parameter.grad is not None
        for parameter in model.trajectory.coefficients.values()
    )
# ^^^ THOG
