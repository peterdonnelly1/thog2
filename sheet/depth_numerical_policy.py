# vvv THOG effective DEPTH arithmetic and checkpoint-compatible CPU-value gradient binding
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class DepthNumericalPolicy:
    backend: str
    coefficient_dtype: str
    operand_dtype: str
    accumulation_dtype: str
    output_dtype: str

    @property
    def policy_id(self):
        return ":".join(vars(self).values())


def dtype_name(dtype):
    return str(dtype).split(".")[-1]


def effective_depth_policy(trajectory):
    coefficient = next(iter(trajectory.coefficients.values()))
    backend = "matmul" if getattr(trajectory, "depth_materialisation_matmul", False) else "einsum"
    mixed = torch.is_autocast_enabled("cuda") and backend == "einsum"
    working = torch.get_autocast_dtype("cuda") if mixed else coefficient.dtype
    if coefficient.dtype not in (torch.float32, torch.float16, torch.bfloat16) or working not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("unqualified DEPTH numerical policy")
    return DepthNumericalPolicy(backend, dtype_name(coefficient.dtype), dtype_name(working), "float32", dtype_name(working))


def materialize_cpu(coefficient, depth_row, policy):
    working = getattr(torch, policy.operand_dtype)
    # Conversion precedes arithmetic: a post-hoc BF16 cast of FP32 operands is not CUDA autocast.
    coefficient = coefficient.to(dtype=working).float()
    depth_row = depth_row.to(dtype=working).float()
    with torch.no_grad():
        if policy.backend == "matmul":
            output = coefficient.reshape(-1, coefficient.shape[-1]).matmul(depth_row)
            output = output.reshape(coefficient.shape[:2])
        else:
            output = torch.einsum("p,rcp->rc", depth_row, coefficient)
        return output.to(dtype=getattr(torch, policy.output_dtype)).contiguous()


@dataclass
class PrematBindingContext:
    policy: DepthNumericalPolicy
    storage_lease: object = None


class CpuDepthBinding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cached, binding, *pairs):
        ctx.binding = binding
        dtype = getattr(torch, binding.policy.operand_dtype)
        saved, specs = [], []
        for i in range(0, len(pairs), 2):
            coefficient, row = pairs[i:i + 2]
            learn_c, learn_b = coefficient.requires_grad, row.requires_grad
            # Native mm/bmm saves its left operand first when both dependencies differentiate.
            if learn_b:
                c = coefficient.to(dtype=dtype).reshape(-1, coefficient.shape[-1])
                saved.append(c if binding.policy.backend == "matmul" else c.t().reshape(1, coefficient.shape[-1], -1))
            if learn_c:
                b = row.to(dtype=dtype)
                saved.append(b if binding.policy.backend == "matmul" else b.reshape(1, 1, -1))
            specs.append((coefficient.shape, coefficient.dtype, row.dtype, learn_c, learn_b))
        ctx.save_for_backward(*saved)
        ctx.specs = specs
        return cached.view_as(cached)

    @staticmethod
    def backward(ctx, gradient):
        values = iter(ctx.saved_tensors)
        result, offset = [None, None], 0
        for shape, c_dtype, b_dtype, learn_c, learn_b in ctx.specs:
            coefficient = next(values) if learn_b else None
            if coefficient is not None:
                coefficient = coefficient.reshape(-1, shape[-1]) if ctx.binding.policy.backend == "matmul" else coefficient.reshape(shape[-1], -1).t()
            row = next(values).reshape(-1) if learn_c else None
            g = gradient[offset:offset + shape[0]].reshape(-1)
            with torch.autocast(device_type=gradient.device.type, enabled=False):
                if ctx.binding.policy.backend == "matmul":
                    gc = g[:, None].matmul(row[None, :]).reshape(shape).to(c_dtype) if learn_c else None
                    gb = coefficient.t().matmul(g).to(b_dtype) if learn_b else None
                else:
                    gc = torch.bmm(g.reshape(1, -1, 1), row.reshape(1, 1, -1)).reshape(shape).to(c_dtype) if learn_c else None
                    gb = torch.bmm(coefficient.t().reshape(1, shape[-1], -1), g.reshape(1, -1, 1)).reshape(-1).to(b_dtype) if learn_b else None
            result.extend((gc, gb))
            offset += shape[0]
        return tuple(result)
# ^^^ THOG
