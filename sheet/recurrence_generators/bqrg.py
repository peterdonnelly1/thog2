# vvv THOG
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from .protocol import RecurrenceGeneratorDefinition


BQRG_FAMILY = "bqrg"
BQRG_VERSION = "bqrg_v1"
BQRG_ARTIFACT_TAG = "BQRG"
BQRG_PERSISTENT_WIDTH = 16
BQRG_SUPPORTED_TARGETS = ("DEPTH",)
BQRG_BACKWARD_CHUNK_TRAJECTORIES = 262144                                                                                                              # <<< THOG bound each custom-autograd gradient object and update-level analytic-BPTT transient state


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
    x_squared = x.square()
    y_squared = y.square()
    xy = x * y
    x_next = torch.tanh(a0 + a1 * x + a2 * y + a3 * x_squared + a4 * xy + a5 * y_squared)
    y_next = torch.tanh(b0 + b1 * x + b2 * y + b3 * x_squared + b4 * xy + b5 * y_squared)
    return x_next, y_next


# vvv THOG BQRG recurrent history is derived execution state; use custom analytic adjoints for both random-access fallback and one-adjoint-per-update training.
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


def _accumulate_transition_gradient(
    parameters: Tensor,
    x: Tensor,
    y: Tensor,
    x_next: Tensor,
    y_next: Tensor,
    grad_x_next: Tensor,
    grad_y_next: Tensor,
    grad_parameters: Tensor,
) -> Tuple[Tensor, Tensor]:
    a1, a2, a3, a4, a5 = parameters[..., 3:8].unbind(dim=-1)
    b1, b2, b3, b4, b5 = parameters[..., 9:14].unbind(dim=-1)
    grad_u = grad_x_next * (1.0 - x_next.square())
    grad_v = grad_y_next * (1.0 - y_next.square())

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

    derivative_u_x = a1 + 2.0 * a3 * x + a4 * y
    derivative_u_y = a2 + a4 * x + 2.0 * a5 * y
    derivative_v_x = b1 + 2.0 * b3 * x + b4 * y
    derivative_v_y = b2 + b4 * x + 2.0 * b5 * y
    grad_x = grad_u * derivative_u_x + grad_v * derivative_v_x
    grad_y = grad_u * derivative_u_y + grad_v * derivative_v_y
    return grad_x, grad_y


def _accumulate_chunk_gradient(
    parameters: Tensor,
    grad_output: Tensor,
    index: int,
    grad_parameters: Tensor,
) -> None:
    x = torch.tanh(parameters[..., 0])
    y = torch.tanh(parameters[..., 1])
    x_states: List[Tensor] = [x]
    y_states: List[Tensor] = [y]
    for _ in range(index):
        x, y = _state_update(parameters, x, y)
        x_states.append(x)
        y_states.append(y)

    scale = F.softplus(parameters[..., 15])
    grad_parameters[..., 14].copy_(grad_output)
    grad_parameters[..., 15].copy_(grad_output * torch.sigmoid(parameters[..., 15]) * x_states[-1])
    grad_x = grad_output * scale
    grad_y = torch.zeros_like(grad_x)

    for step in range(index - 1, -1, -1):
        grad_x, grad_y = _accumulate_transition_gradient(
            parameters,
            x_states[step],
            y_states[step],
            x_states[step + 1],
            y_states[step + 1],
            grad_x,
            grad_y,
            grad_parameters,
        )

    grad_parameters[..., 0].add_(grad_x * (1.0 - x_states[0].square()))
    grad_parameters[..., 1].add_(grad_y * (1.0 - y_states[0].square()))


class _BQRGMaterializeChunk(torch.autograd.Function):
    @staticmethod
    def forward(ctx, parameters: Tensor, index: int) -> Tensor:
        ctx.index = int(index)
        ctx.save_for_backward(parameters)
        return _materialize_bqrg_at_uncheckpointed(parameters, ctx.index)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (parameters,) = ctx.saved_tensors
        with torch.no_grad():
            grad_parameters = torch.zeros_like(parameters)
            _accumulate_chunk_gradient(
                parameters,
                grad_output,
                int(ctx.index),
                grad_parameters,
            )
        return grad_parameters, None


def materialize_bqrg_at(parameters: Tensor, index: int) -> Tensor:
    _validate_parameters(parameters)
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError(f"BQRG index must be a non-negative integer; got {index!r}")
    if not torch.is_grad_enabled() or not parameters.requires_grad:
        return _materialize_bqrg_at_uncheckpointed(parameters, index)

    output_shape = parameters.shape[:-1]
    flat_parameters = parameters.reshape(-1, BQRG_PERSISTENT_WIDTH)
    parameter_chunks = torch.split(
        flat_parameters,
        BQRG_BACKWARD_CHUNK_TRAJECTORIES,
        dim=0,
    )                                                                                                                                                   # <<< THOG SplitBackward rejoins bounded chunk gradients once instead of materialising one full input gradient per slice
    generated_chunks = tuple(
        _BQRGMaterializeChunk.apply(parameter_chunk, index)
        for parameter_chunk in parameter_chunks
    )
    generated = generated_chunks[0] if len(generated_chunks) == 1 else torch.cat(generated_chunks, dim=0)
    return generated.reshape(output_shape)


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


def _validate_layer_gradients(parameters: Tensor, layer_gradients: Sequence[Optional[Tensor]]) -> Tuple[Optional[Tensor], ...]:
    gradients = tuple(layer_gradients)
    if not gradients:
        raise ValueError("BQRG layer gradients must contain at least one layer")
    expected_shape = parameters.shape[:-1]
    for layer_index, gradient in enumerate(gradients):
        if gradient is None:
            continue
        if tuple(gradient.shape) != expected_shape:
            raise ValueError(
                f"BQRG layer gradient {layer_index} has shape {tuple(gradient.shape)}; expected {tuple(expected_shape)}"
            )
        if not gradient.is_floating_point():
            raise ValueError(f"BQRG layer gradient {layer_index} must be floating point; got {gradient.dtype}")
    return gradients


def parameter_gradient_from_bqrg_layer_gradients(
    parameters: Tensor,
    layer_gradients: Sequence[Optional[Tensor]],
) -> Tensor:
    _validate_parameters(parameters)
    gradients = _validate_layer_gradients(parameters, layer_gradients)
    flat_parameters = parameters.reshape(-1, BQRG_PERSISTENT_WIDTH)
    flat_result = torch.empty_like(flat_parameters)
    trajectory_count = flat_parameters.shape[0]

    with torch.no_grad():
        for start in range(0, trajectory_count, BQRG_BACKWARD_CHUNK_TRAJECTORIES):
            end = min(start + BQRG_BACKWARD_CHUNK_TRAJECTORIES, trajectory_count)
            parameter_chunk = flat_parameters[start:end]
            gradient_chunk = flat_result[start:end]
            gradient_chunk.zero_()

            x = torch.tanh(parameter_chunk[..., 0])
            y = torch.tanh(parameter_chunk[..., 1])
            x_states: List[Tensor] = [x]
            y_states: List[Tensor] = [y]
            for _ in range(1, len(gradients)):
                x, y = _state_update(parameter_chunk, x, y)
                x_states.append(x)
                y_states.append(y)

            scale = F.softplus(parameter_chunk[..., 15])
            scale_derivative = torch.sigmoid(parameter_chunk[..., 15])
            grad_x = torch.zeros_like(x)
            grad_y = torch.zeros_like(y)

            for layer_index in range(len(gradients) - 1, -1, -1):
                source_gradient = gradients[layer_index]
                if source_gradient is not None:
                    direct = source_gradient.reshape(-1)[start:end].to(
                        device=parameter_chunk.device,
                        dtype=parameter_chunk.dtype,
                    )
                    gradient_chunk[..., 14].add_(direct)
                    gradient_chunk[..., 15].add_(direct * scale_derivative * x_states[layer_index])
                    grad_x.add_(direct * scale)
                if layer_index == 0:
                    break
                grad_x, grad_y = _accumulate_transition_gradient(
                    parameter_chunk,
                    x_states[layer_index - 1],
                    y_states[layer_index - 1],
                    x_states[layer_index],
                    y_states[layer_index],
                    grad_x,
                    grad_y,
                    gradient_chunk,
                )

            gradient_chunk[..., 0].add_(grad_x * (1.0 - x_states[0].square()))
            gradient_chunk[..., 1].add_(grad_y * (1.0 - y_states[0].square()))

    return flat_result.reshape_as(parameters)
# ^^^ THOG


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
    materialize_sequence=materialize_bqrg_sequence,
    parameter_gradient_from_layer_gradients=parameter_gradient_from_bqrg_layer_gradients,
    initialize_parameters=initialize_bqrg_parameters,
    rescale_output=rescale_bqrg_output,
)
# ^^^ THOG
