# vvv THOG
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class MaterializedHyperblockRegions:
    common: Tensor
    attention_extension: Tensor
    mlp_extension: Tensor


def _restore_attention_unique_modes(
    coefficients: Tensor,
    attention_head_order: int,
    attention_head_channel_order: int,
) -> Tensor:
    expected_unique = attention_head_order * attention_head_channel_order - 1
    if coefficients.shape[-1] != expected_unique:
        raise ValueError(
            "attention coefficient unique-mode count mismatch; "
            f"expected {expected_unique}, got {coefficients.shape[-1]}"
        )
    zero = torch.zeros(
        (*coefficients.shape[:-1], 1),
        dtype=coefficients.dtype,
        device=coefficients.device,
    )
    return torch.cat((zero, coefficients), dim=-1).reshape(
        *coefficients.shape[:-1],
        attention_head_order,
        attention_head_channel_order,
    )


def _restore_mlp_unique_modes(coefficients: Tensor, mlp_hidden_order: int) -> Tensor:
    expected_unique = mlp_hidden_order - 1
    if coefficients.shape[-1] != expected_unique:
        raise ValueError(
            "MLP coefficient unique-mode count mismatch; "
            f"expected {expected_unique}, got {coefficients.shape[-1]}"
        )
    zero = torch.zeros(
        (*coefficients.shape[:-1], 1),
        dtype=coefficients.dtype,
        device=coefficients.device,
    )
    return torch.cat((zero, coefficients), dim=-1)


def materialize_regions_reference(
    common_coefficients: Tensor,
    attention_coefficients: Tensor,
    mlp_coefficients: Tensor,
    bases: Mapping[str, Tensor],
) -> MaterializedHyperblockRegions:
    family_common = bases["family_common"].to(common_coefficients)
    family_attention = bases["family_attention"].to(attention_coefficients)
    family_mlp = bases["family_mlp"].to(mlp_coefficients)
    depth = bases["depth"].to(common_coefficients)
    d_model = bases["d_model"].to(common_coefficients)
    attention_head = bases["attention_head"].to(attention_coefficients)
    attention_head_channel = bases["attention_head_channel"].to(attention_coefficients)
    mlp_hidden = bases["mlp_hidden"].to(mlp_coefficients)

    attention_full = _restore_attention_unique_modes(
        attention_coefficients,
        attention_head.shape[1],
        attention_head_channel.shape[1],
    )
    mlp_full = _restore_mlp_unique_modes(
        mlp_coefficients,
        mlp_hidden.shape[1],
    )

    common = torch.einsum(
        "fa,lb,dc,abc->fld",
        family_common,
        depth,
        d_model,
        common_coefficients,
    )
    attention_extension = torch.einsum(
        "fa,lb,dc,hr,ks,abcrs->fldhk",
        family_attention,
        depth.to(attention_coefficients),
        d_model.to(attention_coefficients),
        attention_head,
        attention_head_channel,
        attention_full,
    )
    mlp_extension = torch.einsum(
        "fa,lb,dc,mt,abct->fldm",
        family_mlp,
        depth.to(mlp_coefficients),
        d_model.to(mlp_coefficients),
        mlp_hidden,
        mlp_full,
    )
    return MaterializedHyperblockRegions(common, attention_extension, mlp_extension)


def _mode_product(values: Tensor, basis: Tensor, axis: int) -> Tensor:
    if axis < 0 or axis >= values.ndim:
        raise IndexError(f"axis {axis} is invalid for shape {tuple(values.shape)}")
    if values.shape[axis] != basis.shape[1]:
        raise ValueError(
            "mode-product dimension mismatch; "
            f"values axis={values.shape[axis]}, basis order={basis.shape[1]}"
        )
    contracted = torch.tensordot(
        basis.to(values),
        values,
        dims=([1], [axis]),
    )
    return contracted.movedim(0, axis)


def materialize_regions_staged(
    common_coefficients: Tensor,
    attention_coefficients: Tensor,
    mlp_coefficients: Tensor,
    bases: Mapping[str, Tensor],
) -> MaterializedHyperblockRegions:
    common = common_coefficients
    common = _mode_product(common, bases["family_common"], 0)
    common = _mode_product(common, bases["depth"], 1)
    common = _mode_product(common, bases["d_model"], 2)

    attention_full = _restore_attention_unique_modes(
        attention_coefficients,
        bases["attention_head"].shape[1],
        bases["attention_head_channel"].shape[1],
    )
    attention_extension = attention_full
    attention_extension = _mode_product(attention_extension, bases["family_attention"], 0)
    attention_extension = _mode_product(attention_extension, bases["depth"], 1)
    attention_extension = _mode_product(attention_extension, bases["attention_head"], 3)
    attention_extension = _mode_product(attention_extension, bases["attention_head_channel"], 4)
    attention_extension = _mode_product(attention_extension, bases["d_model"], 2)

    mlp_full = _restore_mlp_unique_modes(
        mlp_coefficients,
        bases["mlp_hidden"].shape[1],
    )
    mlp_extension = mlp_full
    mlp_extension = _mode_product(mlp_extension, bases["family_mlp"], 0)
    mlp_extension = _mode_product(mlp_extension, bases["depth"], 1)
    mlp_extension = _mode_product(mlp_extension, bases["d_model"], 2)
    mlp_extension = _mode_product(mlp_extension, bases["mlp_hidden"], 3)

    return MaterializedHyperblockRegions(common, attention_extension, mlp_extension)


def materialize_attention_family_layer(
    common_coefficients: Tensor,
    attention_coefficients: Tensor,
    bases: Mapping[str, Tensor],
    *,
    family_index: int,
    layer_index: int,
) -> Tensor:
    family_common = bases["family_common"][family_index].to(common_coefficients)
    family_attention = bases["family_attention"][family_index].to(attention_coefficients)
    depth_common = bases["depth"][layer_index].to(common_coefficients)
    depth_attention = bases["depth"][layer_index].to(attention_coefficients)
    d_model_common = bases["d_model"].to(common_coefficients)
    d_model_attention = bases["d_model"].to(attention_coefficients)
    head = bases["attention_head"].to(attention_coefficients)
    channel = bases["attention_head_channel"].to(attention_coefficients)

    common_modes = torch.einsum(
        "a,b,abc->c",
        family_common,
        depth_common,
        common_coefficients,
    )
    common = d_model_common @ common_modes

    attention_full = _restore_attention_unique_modes(
        attention_coefficients,
        head.shape[1],
        channel.shape[1],
    )
    branch_modes = torch.einsum(
        "a,b,abcrs->crs",
        family_attention,
        depth_attention,
        attention_full,
    )
    branch = torch.einsum(
        "dc,hr,ks,crs->dhk",
        d_model_attention,
        head,
        channel,
        branch_modes,
    )
    return common[:, None, None] + branch


def materialize_mlp_family_layer(
    common_coefficients: Tensor,
    mlp_coefficients: Tensor,
    bases: Mapping[str, Tensor],
    *,
    common_family_index: int,
    mlp_family_index: int,
    layer_index: int,
) -> Tensor:
    family_common = bases["family_common"][common_family_index].to(common_coefficients)
    family_mlp = bases["family_mlp"][mlp_family_index].to(mlp_coefficients)
    depth_common = bases["depth"][layer_index].to(common_coefficients)
    depth_mlp = bases["depth"][layer_index].to(mlp_coefficients)
    d_model_common = bases["d_model"].to(common_coefficients)
    d_model_mlp = bases["d_model"].to(mlp_coefficients)
    mlp_hidden = bases["mlp_hidden"].to(mlp_coefficients)

    common_modes = torch.einsum(
        "a,b,abc->c",
        family_common,
        depth_common,
        common_coefficients,
    )
    common = d_model_common @ common_modes

    mlp_full = _restore_mlp_unique_modes(
        mlp_coefficients,
        mlp_hidden.shape[1],
    )
    branch_modes = torch.einsum(
        "a,b,abct->ct",
        family_mlp,
        depth_mlp,
        mlp_full,
    )
    branch = torch.einsum(
        "dc,mt,ct->dm",
        d_model_mlp,
        mlp_hidden,
        branch_modes,
    )
    return common[:, None] + branch


def route_attention_matrix(canonical: Tensor, *, output_projection: bool) -> Tensor:
    if canonical.ndim != 3:
        raise ValueError(
            "canonical attention field must have [D_MODEL, HEAD, HEAD_CHANNEL]; "
            f"got {tuple(canonical.shape)}"
        )
    d_model, n_head, head_dim = canonical.shape
    if output_projection:
        return canonical.reshape(d_model, n_head * head_dim)
    return canonical.permute(1, 2, 0).reshape(n_head * head_dim, d_model)


def route_mlp_matrix(canonical: Tensor, *, expansion: bool) -> Tensor:
    if canonical.ndim != 2:
        raise ValueError(
            "canonical MLP field must have [D_MODEL, MLP_HIDDEN]; "
            f"got {tuple(canonical.shape)}"
        )
    return canonical.transpose(0, 1) if expansion else canonical


def materializer_diagnostics(regions: MaterializedHyperblockRegions) -> Dict[str, object]:
    return {
        "common_shape": tuple(regions.common.shape),
        "attention_extension_shape": tuple(regions.attention_extension.shape),
        "mlp_extension_shape": tuple(regions.mlp_extension.shape),
        "common_finite": bool(torch.isfinite(regions.common).all()),
        "attention_finite": bool(torch.isfinite(regions.attention_extension).all()),
        "mlp_finite": bool(torch.isfinite(regions.mlp_extension).all()),
    }
# ^^^ THOG
