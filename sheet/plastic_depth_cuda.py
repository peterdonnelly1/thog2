# vvv THOG PLASTIC DEPTH CUDA reserve and recoverable upward-probe allocation helpers
from __future__ import annotations

import math
from typing import Callable, Optional

import torch
from torch import Tensor


GIB_BYTES = 1024 ** 3


def validate_cuda_allocator_reserve_gib(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            "plastic__cuda_allocator_reserve_gib must be a finite non-negative number; "
            f"got {value!r}"
        )
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(
            "plastic__cuda_allocator_reserve_gib must be a finite non-negative number; "
            f"got {value!r}"
        )
    return resolved


def is_cuda_out_of_memory(error: BaseException) -> bool:
    out_of_memory_type = getattr(torch, "OutOfMemoryError", None)
    if out_of_memory_type is not None and isinstance(error, out_of_memory_type):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


class PlasticDepthCudaAllocatorReserve:
    def __init__(
        self,
        *,
        device: torch.device,
        reserve_gib: float,
        allocator: Callable[..., Tensor] = torch.empty,
    ) -> None:
        self.device = torch.device(device)
        self.reserve_gib = validate_cuda_allocator_reserve_gib(reserve_gib)
        self.reserve_bytes = int(round(self.reserve_gib * GIB_BYTES))
        self._allocator = allocator
        self._buffer: Optional[Tensor] = None

    @property
    def active(self) -> bool:
        return self._buffer is not None

    def acquire(self) -> bool:
        if self.device.type != "cuda" or self.reserve_bytes == 0:
            return True
        if self._buffer is not None:
            return True
        try:
            self._buffer = self._allocator(
                self.reserve_bytes,
                dtype=torch.uint8,
                device=self.device,
            )
        except BaseException as error:
            if not is_cuda_out_of_memory(error):
                raise
            self._buffer = None
            return False
        return True

    def release(self, *, empty_cache: bool = False) -> None:
        self._buffer = None
        if empty_cache and self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = [
    "GIB_BYTES",
    "PlasticDepthCudaAllocatorReserve",
    "is_cuda_out_of_memory",
    "validate_cuda_allocator_reserve_gib",
]
# ^^^ THOG
