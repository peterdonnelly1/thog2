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


def effective_depth_policy(trajectory, *, qualify_cpu=True):
    coefficient = next(iter(trajectory.coefficients.values()))
    backend = "matmul" if getattr(trajectory, "depth_materialisation_matmul", False) else "einsum"
    device_type = coefficient.device.type
    # CUDA casts einsum operands before decomposition, including order-one mul.
    # CPU only casts the inner bmm; order-one pointwise arithmetic stays FP32.
    mixed = torch.is_autocast_enabled(device_type) and backend == "einsum" and (device_type == "cuda" or coefficient.shape[-1] > 1)
    working = torch.get_autocast_dtype(device_type) if mixed else coefficient.dtype
    if qualify_cpu:
        if coefficient.dtype not in (torch.float32, torch.float16, torch.bfloat16) or working not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("unqualified DEPTH numerical policy")
        if coefficient.dtype != torch.float32:
            raise ValueError("cpu_and_gpu currently qualifies FP32 coefficient storage with FP32/FP16/BF16 training autocast")
        if backend == "einsum" and working == torch.float32 and torch.backends.cuda.matmul.allow_tf32:
            raise ValueError("unqualified TF32 DEPTH einsum policy; disable CUDA matmul TF32")
    accumulation = "float64" if working == torch.float64 else "float32"
    return DepthNumericalPolicy(backend, dtype_name(coefficient.dtype), dtype_name(working), accumulation, dtype_name(working))


def materialize_cpu(coefficient, depth_row, policy):
    working = getattr(torch, policy.operand_dtype)
    # Conversion precedes arithmetic: a post-hoc BF16 cast of FP32 operands is not CUDA autocast.
    coefficient = coefficient.to(dtype=working).float()
    depth_row = depth_row.to(dtype=working).float()
    with torch.no_grad(), torch.autocast(coefficient.device.type, enabled=False):
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


class CpuDepthBinding:
    """Compatibility entry point for the shared PREMAT autograd node."""

    @staticmethod
    def apply(cached, binding, *pairs):
        from .depth_trajectory import _PrematerializedDepthBundle
        return _PrematerializedDepthBundle.apply(cached, binding, *pairs)

# ^^^ THOG
