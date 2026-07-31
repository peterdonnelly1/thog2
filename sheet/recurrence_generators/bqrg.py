# vvv THOG
from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .protocol import RecurrenceGeneratorDefinition


BQRG_FAMILY = "bqrg"
BQRG_VERSION = "bqrg_v1"
BQRG_ARTIFACT_TAG = "BQRG"
BQRG_PERSISTENT_WIDTH = 16
BQRG_SUPPORTED_TARGETS = ("DEPTH",)


def _validate_parameters(parameters: Tensor) -> None:
    if parameters.ndim < 1:
        raise ValueError("BQRG parameters must have at least one dimension")
    if parameters.shape[-1] != BQRG_PERSISTENT_WIDTH:
        raise ValueError(
            f"BQRG requires persistent width {BQRG_PERSISTENT_WIDTH}; got {parameters.shape[-1]}"
        )
    if not parameters.is_floating_point():
        raise ValueError(f"BQRG parameters must be floating point; got {parameters.dtype}")


def _state_update(parameters: Tensor, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    a0, a1, a2, a3, a4, a5 = parameters[..., 2:8].unbind(dim=-1)
    b0, b1, b2, b3, b4, b5 = parameters[..., 8:14].unbind(dim=-1)
    x_next = torch.tanh(a0 + a1 * x + a2 * y + a3 * x.square() + a4 * x * y + a5 * y.square())
    y_next = torch.tanh(b0 + b1 * x + b2 * y + b3 * x.square() + b4 * x * y + b5 * y.square())
    return x_next, y_next


def materialize_bqrg_at(parameters: Tensor, index: int) -> Tensor:
    _validate_parameters(parameters)
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError(f"BQRG index must be a non-negative integer; got {index!r}")
    x = torch.tanh(parameters[..., 0])
    y = torch.tanh(parameters[..., 1])
    for _ in range(index):
        x, y = _state_update(parameters, x, y)
    offset = parameters[..., 14]
    scale = F.softplus(parameters[..., 15])
    generated = offset + scale * x
    if not torch.isfinite(generated).all():
        raise FloatingPointError("BQRG materialisation produced a non-finite value")
    return generated


def materialize_bqrg_sequence(parameters: Tensor, length: int) -> Tensor:
    _validate_parameters(parameters)
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ValueError(f"BQRG length must be a positive integer; got {length!r}")
    x = torch.tanh(parameters[..., 0])
    y = torch.tanh(parameters[..., 1])
    offset = parameters[..., 14]
    scale = F.softplus(parameters[..., 15])
    values = [offset + scale * x]
    for _ in range(1, length):
        x, y = _state_update(parameters, x, y)
        values.append(offset + scale * x)
    generated = torch.stack(values, dim=-1)
    if not torch.isfinite(generated).all():
        raise FloatingPointError("BQRG materialisation produced a non-finite value")
    return generated


def _raw_scale_for_target(target_weight_std: float) -> float:
    target_scale = max(float(target_weight_std) * 0.25, 1.0e-6)
    return math.log(math.expm1(target_scale))


def initialize_bqrg_parameters(parameters: Tensor, initialization: str, target_weight_std: float, n_layer: int) -> None:
    _validate_parameters(parameters)
    if isinstance(n_layer, bool) or not isinstance(n_layer, int) or n_layer < 1:
        raise ValueError(f"n_layer must be a positive integer; got {n_layer!r}")
    with torch.no_grad():
        parameters.zero_()
        if initialization == "depth_matrix_normal":
            torch.nn.init.normal_(parameters[..., 0], mean=0.0, std=0.05)
            torch.nn.init.normal_(parameters[..., 1], mean=0.0, std=0.05)
            parameters[..., 3].fill_(0.5)
            parameters[..., 10].fill_(0.5)
            torch.nn.init.normal_(parameters[..., 2], mean=0.0, std=0.01)
            torch.nn.init.normal_(parameters[..., 8], mean=0.0, std=0.01)
            torch.nn.init.normal_(parameters[..., 14], mean=0.0, std=float(target_weight_std))
            parameters[..., 15].fill_(_raw_scale_for_target(target_weight_std))
            return
        if initialization == "layernorm_one":
            parameters[..., 14].fill_(1.0)
            parameters[..., 15].fill_(_raw_scale_for_target(0.01))
            return
        if initialization == "zero":
            parameters[..., 15].fill_(_raw_scale_for_target(1.0e-5))
            return
        raise RuntimeError(f"unsupported BQRG initialization policy {initialization!r}")


BQRG_DEFINITION = RecurrenceGeneratorDefinition(
    family=BQRG_FAMILY,
    aliases=("bounded_quadratic_recurrent_generator",),
    version=BQRG_VERSION,
    artifact_tag=BQRG_ARTIFACT_TAG,
    persistent_widths=(BQRG_PERSISTENT_WIDTH,),
    supported_targets=BQRG_SUPPORTED_TARGETS,
    option_names=(),
    description="Bounded two-state quadratic recurrence with tanh state updates and learned offset/scale.",
    materialize_at=materialize_bqrg_at,
    initialize_parameters=initialize_bqrg_parameters,
)
# ^^^ THOG
