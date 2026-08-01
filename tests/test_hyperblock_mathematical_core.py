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
    route_attention_matrix,
    route_mlp_matrix,
)


def _plan(*, compressor_family: str = "chebyshev", compressor_version: str = "auto") -> ResolvedHyperblockPlan:
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
        compressor_family=compressor_family,
        compressor_version=compressor_version,
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
    return HyperblockBasisTables(
        plan,
        runtime_dtype=torch.float64,
    ).as_mapping()


def test_plan_counts_match_allocated_shapes() -> None:
    plan = _plan()
    expected = {
        "common": 6 * 3 * 4,
        "attention": 4 * 3 * 4 * (2 * 3 - 1),
        "mlp": 2 * 3 * 4 * (5 - 1),
    }
    assert plan.coefficient_counts["common"] == expected["common"]
    assert plan.coefficient_counts["attention"] == expected["attention"]
    assert plan.coefficient_counts["mlp"] == expected["mlp"]
    assert plan.coefficient_counts["total"] == sum(expected.values())
    assert plan.dense_equivalent_matrix_count == (4 + 2 * 2) * 3 * 6 * 6


def test_dense_equivalent_count_respects_mlp_hidden_multiplier() -> None:
    plan = ResolvedHyperblockPlan(
        n_layer=5,
        n_embd=12,
        n_head=3,
        mlp_hidden_multiplier=3,
        orders=HyperblockOrders(
            depth=5,
            d_model=6,
            mlp_hidden=7,
            attention_head=3,
            attention_head_channel=4,
        ),
    )
    expected = 4 * 5 * 12 * 12 + 2 * 5 * 12 * (3 * 12)
    assert plan.dense_equivalent_matrix_count == expected


def test_canonical_l32_d1024_count_matches_specification() -> None:
    plan = ResolvedHyperblockPlan(
        n_layer=32,
        n_embd=1024,
        n_head=16,
        mlp_hidden_multiplier=4,
        orders=HyperblockOrders(
            depth=16,
            d_model=16,
            mlp_hidden=16,
            attention_head=16,
            attention_head_channel=16,
        ),
    )
    assert plan.coefficient_counts == {
        "common": 1_536,
        "attention": 261_120,
        "mlp": 7_680,
        "total": 270_336,
    }
    assert plan.dense_equivalent_matrix_count == 402_653_184
    assert plan.compression_ratio == pytest.approx(1489.4547267119567)


def test_basis_tables_are_orthonormal() -> None:
    tables = HyperblockBasisTables(_plan(), runtime_dtype=torch.float64)
    for basis in tables.as_mapping().values():
        gram = basis.transpose(0, 1) @ basis
        assert torch.allclose(
            gram,
            torch.eye(gram.shape[0], dtype=gram.dtype),
            atol=1.0e-12,
            rtol=1.0e-12,
        )


def test_staged_materialization_matches_reference() -> None:
    plan = _plan()
    coefficients = _coefficients(plan)
    bases = _bases(plan)
    reference = materialize_regions_reference(*coefficients, bases)
    staged = materialize_regions_staged(*coefficients, bases)
    assert torch.allclose(staged.common, reference.common, atol=1.0e-11, rtol=1.0e-11)
    assert torch.allclose(
        staged.attention_extension,
        reference.attention_extension,
        atol=1.0e-11,
        rtol=1.0e-11,
    )
    assert torch.allclose(
        staged.mlp_extension,
        reference.mlp_extension,
        atol=1.0e-11,
        rtol=1.0e-11,
    )


def test_per_family_layer_materialization_matches_full_regions() -> None:
    plan = _plan()
    common, attention, mlp = _coefficients(plan)
    bases = _bases(plan)
    regions = materialize_regions_reference(common, attention, mlp, bases)

    for family_index in range(4):
        for layer_index in range(plan.n_layer):
            actual = materialize_attention_family_layer(
                common,
                attention,
                bases,
                family_index=family_index,
                layer_index=layer_index,
            )
            expected = (
                regions.common[family_index, layer_index, :, None, None]
                + regions.attention_extension[family_index, layer_index]
            )
            assert torch.allclose(actual, expected, atol=1.0e-11, rtol=1.0e-11)

    for mlp_family_index, common_family_index in enumerate((4, 5)):
        for layer_index in range(plan.n_layer):
            actual = materialize_mlp_family_layer(
                common,
                mlp,
                bases,
                common_family_index=common_family_index,
                mlp_family_index=mlp_family_index,
                layer_index=layer_index,
            )
            expected = (
                regions.common[common_family_index, layer_index, :, None]
                + regions.mlp_extension[mlp_family_index, layer_index]
            )
            assert torch.allclose(actual, expected, atol=1.0e-11, rtol=1.0e-11)


def test_branch_extensions_have_no_constant_unique_axis_component() -> None:
    plan = _plan()
    common, attention, mlp = _coefficients(plan)
    bases = _bases(plan)
    regions = materialize_regions_reference(common, attention, mlp, bases)

    attention_constant = torch.einsum(
        "fldhk,h,k->fld",
        regions.attention_extension,
        bases["attention_head"][:, 0],
        bases["attention_head_channel"][:, 0],
    )
    mlp_constant = torch.einsum(
        "fldm,m->fld",
        regions.mlp_extension,
        bases["mlp_hidden"][:, 0],
    )
    assert torch.allclose(attention_constant, torch.zeros_like(attention_constant), atol=1.0e-11)
    assert torch.allclose(mlp_constant, torch.zeros_like(mlp_constant), atol=1.0e-11)


def test_common_gradient_is_sum_of_attention_and_mlp_contributions() -> None:
    plan = _plan()
    common, attention, mlp = _coefficients(plan, requires_grad=True)
    bases = _bases(plan)

    attention_value = materialize_attention_family_layer(
        common,
        attention,
        bases,
        family_index=1,
        layer_index=2,
    ).square().sum()
    attention_common_gradient = torch.autograd.grad(
        attention_value,
        common,
        retain_graph=True,
    )[0]

    mlp_value = materialize_mlp_family_layer(
        common,
        mlp,
        bases,
        common_family_index=5,
        mlp_family_index=1,
        layer_index=1,
    ).square().sum()
    mlp_common_gradient = torch.autograd.grad(
        mlp_value,
        common,
        retain_graph=True,
    )[0]

    joint_common_gradient = torch.autograd.grad(
        attention_value + mlp_value,
        common,
    )[0]
    assert torch.allclose(
        joint_common_gradient,
        attention_common_gradient + mlp_common_gradient,
        atol=1.0e-11,
        rtol=1.0e-11,
    )


def test_weight_router_orientations_are_exact() -> None:
    attention = torch.arange(6 * 2 * 3, dtype=torch.float64).reshape(6, 2, 3)
    qkv = route_attention_matrix(attention, output_projection=False)
    output = route_attention_matrix(attention, output_projection=True)
    assert qkv.shape == (6, 6)
    assert output.shape == (6, 6)
    assert torch.equal(qkv, attention.permute(1, 2, 0).reshape(6, 6))
    assert torch.equal(output, attention.reshape(6, 6))

    mlp = torch.arange(6 * 12, dtype=torch.float64).reshape(6, 12)
    assert torch.equal(route_mlp_matrix(mlp, expansion=True), mlp.transpose(0, 1))
    assert torch.equal(route_mlp_matrix(mlp, expansion=False), mlp)


def test_registered_dct_provider_is_plug_compatible() -> None:
    plan = _plan(
        compressor_family="dct",
        compressor_version="auto",
    )
    coefficients = _coefficients(plan)
    bases = _bases(plan)
    regions = materialize_regions_staged(*coefficients, bases)
    assert regions.common.shape == (6, 3, 6)
    assert regions.attention_extension.shape == (4, 3, 6, 2, 3)
    assert regions.mlp_extension.shape == (2, 3, 6, 12)


def test_staged_materializer_is_torch_compile_compatible() -> None:
    compile_function = getattr(torch, "compile", None)
    if compile_function is None:
        pytest.skip("torch.compile is unavailable")
    plan = _plan()
    common, attention, mlp = _coefficients(plan)
    bases = _bases(plan)

    def materialize_tuple(c0, ca, cm, bf0, bfa, bfm, bl, bd, bm, bh, bc):
        result = materialize_regions_staged(
            c0,
            ca,
            cm,
            {
                "family_common": bf0,
                "family_attention": bfa,
                "family_mlp": bfm,
                "depth": bl,
                "d_model": bd,
                "mlp_hidden": bm,
                "attention_head": bh,
                "attention_head_channel": bc,
            },
        )
        return result.common, result.attention_extension, result.mlp_extension

    compiled = compile_function(materialize_tuple, backend="eager", fullgraph=True)
    arguments = (
        common,
        attention,
        mlp,
        bases["family_common"],
        bases["family_attention"],
        bases["family_mlp"],
        bases["depth"],
        bases["d_model"],
        bases["mlp_hidden"],
        bases["attention_head"],
        bases["attention_head_channel"],
    )
    expected = materialize_tuple(*arguments)
    actual = compiled(*arguments)
    for actual_tensor, expected_tensor in zip(actual, expected):
        assert torch.allclose(actual_tensor, expected_tensor, atol=1.0e-11, rtol=1.0e-11)
# ^^^ THOG

# vvv THOG

def test_anisotropic_family_orders_reduce_the_coupled_field_budget() -> None:
    plan = ResolvedHyperblockPlan(
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
            common_family=3,
            attention_family=2,
            mlp_family=1,
        ),
    )
    assert plan.coefficient_shapes == {
        "common": (3, 3, 4),
        "attention": (2, 3, 4, 5),
        "mlp": (1, 3, 4, 4),
    }
    assert plan.coefficient_counts["total"] == 204
    assert plan.retained_axis_orders["WEIGHT_FAMILY_COMMON"] == 3
    assert plan.retained_axis_orders["WEIGHT_FAMILY_ATTENTION"] == 2
    assert plan.retained_axis_orders["WEIGHT_FAMILY_MLP"] == 1


def test_family_orders_cannot_exceed_their_physical_family_sets() -> None:
    with pytest.raises(ValueError, match="WEIGHT_FAMILY_COMMON"):
        ResolvedHyperblockPlan(
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
                common_family=7,
            ),
        )
# ^^^ THOG
