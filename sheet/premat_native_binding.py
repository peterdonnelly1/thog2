# vvv THOG cached values use native autograd without repeating their physical contraction
from __future__ import annotations

import math

import torch
from torch.utils._python_dispatch import TorchDispatchMode


class _CachedDepthContraction(TorchDispatchMode):
    """Replace one physical contraction below autograd; keep its native node."""

    def __init__(self, cached, coefficient, row, policy):
        super().__init__()
        self.cached = cached
        self.coefficient = coefficient
        self.row = row
        self.policy = policy
        self.calls = 0
        self.coefficient_storage = coefficient.untyped_storage().data_ptr()

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        order = self.coefficient.shape[-1]
        elements = self.cached.numel()
        if function == torch.ops.aten.mv.default and self.policy.backend == "matmul":
            if tuple(args[0].shape) != (elements, order) or tuple(args[1].shape) != (order,):
                raise RuntimeError("cached DEPTH mv operands do not match the native contract")
            self.calls += 1
            return self.cached.reshape(elements)
        if function == torch.ops.aten.bmm.default and self.policy.backend == "einsum":
            if tuple(args[0].shape) != (1, 1, order) or tuple(args[1].shape) != (1, order, elements):
                raise RuntimeError("cached DEPTH bmm operands do not match the native contract")
            self.calls += 1
            return self.cached.reshape(1, 1, elements)
        if function == torch.ops.aten.mul.Tensor and self.policy.backend == "einsum" and order == 1:
            shape = torch.broadcast_shapes(args[0].shape, args[1].shape)
            if math.prod(shape) != elements:
                raise RuntimeError("cached order-one DEPTH operands do not match the native contract")
            self.calls += 1
            return self.cached.reshape(shape)
        if (
            function == torch.ops.aten._to_copy.default
            and self.policy.backend == "einsum"
            and not self.row.requires_grad
            and args[0].numel() == self.coefficient.numel()
            and args[0].untyped_storage().data_ptr() == self.coefficient_storage
            and kwargs.get("dtype") == getattr(torch, self.policy.operand_dtype)
        ):
            # Native autocast creates a coefficient cast before bmm. When only
            # coefficients differentiate, backward saves only the depth row.
            # The intercepted contraction never reads this operand, so retain
            # its native cast edge using a scalar-backed shape, without copying
            # the complete coefficient tensor. Autocast caching is private here.
            return torch.empty((), device=args[0].device, dtype=kwargs["dtype"]).expand(args[0].shape)
        return function(*args, **kwargs)


class _CachedDepthConcatenation(TorchDispatchMode):
    def __init__(self, cached, outputs):
        super().__init__()
        self.cached = cached
        self.shapes = tuple(tuple(value.shape) for value in outputs)
        self.calls = 0

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        if function == torch.ops.aten.cat.default:
            dimension = args[1] if len(args) > 1 else (kwargs or {}).get("dim", 0)
            if dimension != 0 or tuple(tuple(value.shape) for value in args[0]) != self.shapes:
                raise RuntimeError("cached DEPTH concatenation does not match the native contract")
            self.calls += 1
            return self.cached.view_as(self.cached)
        return function(*args, **(kwargs or {}))


def bind_cached_depth(cached, binding, *pairs):
    """Build ordinary mv/bmm/cat backward nodes around detached cached storage."""
    if not pairs or len(pairs) % 2:
        raise RuntimeError("prematerialised DEPTH binding requires coefficient/depth-row pairs")
    numerical_policy = binding.policy
    if numerical_policy.backend not in ("matmul", "einsum"):
        raise ValueError("unqualified cached DEPTH backend")
    if cached.dtype != getattr(torch, numerical_policy.output_dtype):
        raise RuntimeError("cached DEPTH dtype does not match its numerical policy")
    if not torch.is_grad_enabled() or not any(value.requires_grad for value in pairs):
        return cached
    outputs = []
    offset = 0
    for index in range(0, len(pairs), 2):
        coefficient, row = pairs[index:index + 2]
        if coefficient.ndim != 3 or row.ndim != 1 or coefficient.shape[-1] != row.numel():
            raise RuntimeError("invalid native DEPTH coefficient/depth-row shapes")
        rows, columns, order = coefficient.shape
        if cached.ndim != 2 or cached.shape[1] != columns or offset + rows > cached.shape[0]:
            raise RuntimeError("cached DEPTH shape does not match its coefficient bundle")
        part = cached.narrow(0, offset, rows)
        mode = _CachedDepthContraction(part, coefficient, row, numerical_policy)
        working_dtype = getattr(torch, numerical_policy.operand_dtype)
        mixed = numerical_policy.backend == "einsum" and working_dtype != coefficient.dtype
        options = {"enabled": mixed, "cache_enabled": False}
        if mixed:
            options["dtype"] = working_dtype
        with torch.autocast(coefficient.device.type, **options), mode:
            if numerical_policy.backend == "matmul":
                value = coefficient.reshape(-1, order).matmul(row).reshape(rows, columns)
            else:
                value = torch.einsum("p,rcp->rc", row, coefficient)
        if mode.calls != 1:
            raise RuntimeError("native DEPTH contraction was not intercepted exactly once")
        outputs.append(value)
        offset += rows
    if offset != cached.shape[0]:
        raise RuntimeError("cached DEPTH bundle contains unmatched rows")
    if len(outputs) == 1:
        result = outputs[0]
    else:
        concatenation = _CachedDepthConcatenation(cached, outputs)
        with concatenation:
            result = torch.cat(outputs, dim=0)
        if concatenation.calls != 1:
            raise RuntimeError("native DEPTH concatenation was not intercepted exactly once")
    if binding.storage_lease is not None:
        # Native nodes own this metadata for their graph lifetime, including
        # projection input gradients and checkpoint replay. No extra backward
        # node or copied dense value is needed to retain uploaded storage.
        result.grad_fn.metadata["thog2_premat_storage_lease"] = binding.storage_lease
    return result
# ^^^ THOG
