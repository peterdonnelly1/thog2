# vvv THOG
from __future__ import annotations

import math
from typing import Tuple

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


def _state_update(parameters: Tensor, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
    a0, a1, a2, a3, a4, a5 = parameters[..., 2:8].unbind(dim=-1)
    b0, b1, b2, b3, b4, b5 = parameters[..., 8:14].unbind(dim=-1)
    x_next = torch.tanh(a0 + a1 * x + a2 * y + a3 * x.square() + a4 * x * y + a5 * y.square())
    y_next = torch.tanh(b0 + b1 * x + b2 * y + b3 * x.square() + b4 * x * y + b5 * y.square())
    return x_next, y_next


# vvv THOG BQRG recurrent history is derived execution state; use an analytic rematerialising adjoint so backward never retains the full slab-sized recurrence graph.
def _advance_state(parameters: Tensor, x: Tensor, y: Tensor, steps: int) -> Tuple[Tensor, Tensor]:
    for _ in range(steps):
        x, y = _state_update(parameters, x, y)
    return x, y


def _materialize_bqrg_at_uncheckpointed(parameters: Tensor, index: int) -> Tensor:
    x = torch.tanh(parameters[..., 0])
    y = torch.tanh(parameters[..., 1])
    x, y = _advance_state(parameters, x, y, index)
    del y
    offset = parameters[..., 14]
    scale = F.softplus(parameters[..., 15])
    return offset + scale * x


def _reverse_state_update(
    parameters: Tensor,
    x: Tensor,
    y: Tensor,
    grad_x_next: Tensor,
    grad_y_next: Tensor,
    grad_parameters: Tensor,
) -> Tuple[Tensor, Tensor]:
    a0, a1, a2, a3, a4, a5 = parameters[..., 2:8].unbind(dim=-1)
    b0, b1, b2, b3, b4, b5 = parameters[..., 8:14].unbind(dim=-1)
    x_next, y_next = _state_update(parameters, x, y)
    grad_u = grad_x_next * (1.0 - x_next.square())
    grad_v = grad_y_next * (1.0 - y_next.square())
    del x_next, y_next

    x_squared = x.square()
    y_squared = y.square()
    xy = x * y
    grad_parameters[..., 2].add_(grad_u)
    grad_parameters[..., 3].add_(grad_u * x)
    grad_parameters[..., 4].add_(grad_u * y)
    grad_parameters[..., 5].add_(grad_u * x_squared)
    grad_parameters[..., 6].add_(grad_u * xy)
    grad_parameters[..., 7].add_(grad_u * y_squared)
    grad_parameters[..., 8].add_(grad_v)
    grad_parameters[..., 9].add_(grad_v * x)
    grad_parameters[..., 10].add_(grad_v * y)
    grad_parameters[..., 11].add_(grad_v * x_squared)
    grad_parameters[..., 12].add_(grad_v * xy)
    grad_parameters[..., 13].add_(grad_v * y_squared)
    del x_squared, y_squared, xy

    derivative_u_x = a1 + 2.0 * a3 * x + a4 * y
    derivative_u_y = a2 + a4 * x + 2.0 * a5 * y
    derivative_v_x = b1 + 2.0 * b3 * x + b4 * y
    derivative_v_y = b2 + b4 * x + 2.0 * b5 * y
    grad_x = grad_u * derivative_u_x + grad_v * derivative_v_x
    grad_y = grad_u * derivative_u_y + grad_v * derivative_v_y
    return grad_x, grad_y


def _reverse_segment(
    parameters: Tensor,
    x_start: Tensor,
    y_start: Tensor,
    steps: int,
    grad_x_end: Tensor,
    grad_y_end: Tensor,
    grad_parameters: Tensor,
) -> Tuple[Tensor, Tensor]:
    if steps == 0:
        return grad_x_end, grad_y_end
    if steps == 1:
        return _reverse_state_update(
            parameters,
            x_start,
            y_start,
            grad_x_end,
            grad_y_end,
            grad_parameters,
        )

    left_steps = steps // 2
    right_steps = steps - left_steps
    x_mid, y_mid = _advance_state(parameters, x_start, y_start, left_steps)
    grad_x_mid, grad_y_mid = _reverse_segment(
        parameters,
        x_mid,
        y_mid,
        right_steps,
        grad_x_end,
        grad_y_end,
        grad_parameters,
    )
    del x_mid, y_mid
    return _reverse_segment(
        parameters,
        x_start,
        y_start,
        left_steps,
        grad_x_mid,
        grad_y_mid,
        grad_parameters,
    )


class _BQRGMaterializeAt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, parameters: Tensor, index: int) -> Tensor:
        ctx.index = int(index)
        ctx.save_for_backward(parameters)
        return _materialize_bqrg_at_uncheckpointed(parameters, ctx.index)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (parameters,) = ctx.saved_tensors
        index = int(ctx.index)
        with torch.no_grad():
            grad_parameters = torch.zeros_like(parameters)
            x_start = torch.tanh(parameters[..., 0])
            y_start = torch.tanh(parameters[..., 1])
            x_end, y_end = _advance_state(parameters, x_start, y_start, index)
            del y_end

            scale = F.softplus(parameters[..., 15])
            grad_parameters[..., 14].copy_(grad_output)
            grad_parameters[..., 15].copy_(grad_output * torch.sigmoid(parameters[..., 15]) * x_end)
            grad_x_end = grad_output * scale
            grad_y_end = torch.zeros_like(grad_x_end)
            del scale, x_end

            grad_x_start, grad_y_start = _reverse_segment(
                parameters,
                x_start,
                y_start,
                index,
                grad_x_end,
                grad_y_end,
                grad_parameters,
            )
            grad_parameters[..., 0].add_(grad_x_start * (1.0 - x_start.square()))
            grad_parameters[..., 1].add_(grad_y_start * (1.0 - y_start.square()))
        return grad_parameters, None


def materialize_bqrg_at(parameters: Tensor, index: int) -> Tensor:
    _validate_parameters(parameters)
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError(f"BQRG index must be a non-negative integer; got {index!r}")
    if not torch.is_grad_enabled() or not parameters.requires_grad:
        return _materialize_bqrg_at_uncheckpointed(parameters, index)
    return _BQRGMaterializeAt.apply(parameters, index)
# ^^^ THOG


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
    return torch.stack(values, dim=-1)


def _raw_scale_for_target(target_weight_std: float) -> float:
    target_scale = max(float(target_weight_std) * 0.25, 1.0e-6)
    return math.log(math.expm1(target_scale))


def _inverse_softplus(value: Tensor) -> Tensor:
    return value + torch.log(-torch.expm1(-value))


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


def rescale_bqrg_output(parameters: Tensor, factor: float) -> None:
    _validate_parameters(parameters)
    if not isinstance(factor, (int, float)) or isinstance(factor, bool) or factor <= 0.0:
        raise ValueError(f"BQRG output scale factor must be positive; got {factor!r}")
    with torch.no_grad():
        parameters[..., 14].mul_(float(factor))
        scaled_output = F.softplus(parameters[..., 15]) * float(factor)
        parameters[..., 15].copy_(_inverse_softplus(scaled_output))


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
    rescale_output=rescale_bqrg_output,
)
# ^^^ THOG
