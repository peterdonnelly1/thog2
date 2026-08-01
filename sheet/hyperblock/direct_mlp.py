# vvv THOG
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import torch
from torch import Tensor


@dataclass(frozen=True)
class FactorisedHyperblockMlpLayer:
    common_modes: Tensor
    branch_modes: Tensor
    d_model_basis: Tensor
    mlp_hidden_basis: Tensor


def _restore_mlp_unique_modes(
    coefficients: Tensor,
    mlp_hidden_order: int,
) -> Tensor:
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


def factorise_hyperblock_mlp_layer(
    common_coefficients: Tensor,
    mlp_coefficients: Tensor,
    bases: Mapping[str, Tensor],
    *,
    layer_index: int,
) -> FactorisedHyperblockMlpLayer:
    depth_basis = bases["depth"]
    if isinstance(layer_index, bool) or not isinstance(layer_index, int):
        raise ValueError(f"layer_index must be an integer; got {layer_index!r}")
    if layer_index < 0 or layer_index >= depth_basis.shape[0]:
        raise IndexError(
            f"layer_index out of range: {layer_index}; n_layer={depth_basis.shape[0]}"
        )

    family_common = bases["family_common"][4:6].to(common_coefficients)
    family_mlp = bases["family_mlp"].to(mlp_coefficients)
    depth_common = depth_basis[layer_index].to(common_coefficients)
    depth_mlp = depth_basis[layer_index].to(mlp_coefficients)
    d_model_basis = bases["d_model"].to(common_coefficients)
    mlp_hidden_basis = bases["mlp_hidden"].to(mlp_coefficients)

    common_modes = torch.einsum(
        "fa,b,abc->fc",
        family_common,
        depth_common,
        common_coefficients,
    )
    mlp_full = _restore_mlp_unique_modes(
        mlp_coefficients,
        mlp_hidden_basis.shape[1],
    )
    branch_modes = torch.einsum(
        "fa,b,abct->fct",
        family_mlp,
        depth_mlp,
        mlp_full,
    )
    return FactorisedHyperblockMlpLayer(
        common_modes=common_modes,
        branch_modes=branch_modes,
        d_model_basis=d_model_basis,
        mlp_hidden_basis=mlp_hidden_basis,
    )


def apply_factorised_hyperblock_mlp(
    inputs: Tensor,
    factors: FactorisedHyperblockMlpLayer,
    *,
    family_index: int,
    expansion: bool,
    bias: Optional[Tensor],
) -> Tensor:
    if family_index not in (0, 1):
        raise ValueError(f"family_index must be 0 or 1; got {family_index!r}")
    if not isinstance(expansion, bool):
        raise TypeError(f"expansion must be bool; got {expansion!r}")

    common_modes = factors.common_modes[family_index]
    branch_modes = factors.branch_modes[family_index]
    common_vector = torch.matmul(factors.d_model_basis, common_modes)

    if expansion:
        projected = torch.matmul(inputs, factors.d_model_basis)
        projected = torch.matmul(projected, branch_modes)
        output = torch.matmul(
            projected,
            factors.mlp_hidden_basis.transpose(0, 1),
        )
        output = output + torch.matmul(inputs, common_vector).unsqueeze(-1)
    else:
        projected = torch.matmul(inputs, factors.mlp_hidden_basis)
        projected = torch.matmul(projected, branch_modes.transpose(0, 1))
        output = torch.matmul(
            projected,
            factors.d_model_basis.transpose(0, 1),
        )
        output = output + inputs.sum(dim=-1, keepdim=True) * common_vector

    if bias is not None:
        output = output + bias
    return output
# ^^^ THOG
