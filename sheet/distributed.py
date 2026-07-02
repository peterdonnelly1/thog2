# vvv THOG
from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedIdentity:
    active: bool
    backend: str
    rank: int
    local_rank: int
    world_size: int
    device: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DistributedContext:
    """Small explicit wrapper around torch.distributed process state."""

    def __init__(
        self,
        *,
        identity: DistributedIdentity,
        device: torch.device,
        owns_process_group: bool,
    ) -> None:
        self.identity = identity
        self.device = device
        self.owns_process_group = owns_process_group

    @classmethod
    def from_environment(
        cls,
        requested_device: str,
        *,
        timeout_seconds: int = 180,
    ) -> "DistributedContext":
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if world_size <= 0:
            raise ValueError(f"WORLD_SIZE must be positive; got {world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(
                f"RANK must be in [0, WORLD_SIZE); got rank={rank}, world_size={world_size}"
            )
        if local_rank < 0:
            raise ValueError(f"LOCAL_RANK must be non-negative; got {local_rank}")

        requested = torch.device(requested_device)
        active = world_size > 1
        if not active:
            if requested.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA training requested but no CUDA device is available")
            device = requested
            if device.type == "cuda" and device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            return cls(
                identity=DistributedIdentity(
                    active=False,
                    backend="none",
                    rank=0,
                    local_rank=0,
                    world_size=1,
                    device=str(device),
                ),
                device=device,
                owns_process_group=False,
            )

        if requested.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("distributed CUDA training requested but CUDA is unavailable")
            if requested.index is not None and requested.index != local_rank:
                raise ValueError(
                    "an explicitly indexed CUDA device must match LOCAL_RANK under DDP; "
                    f"got device={requested}, local_rank={local_rank}"
                )
            if local_rank >= torch.cuda.device_count():
                raise RuntimeError(
                    "LOCAL_RANK exceeds the visible CUDA device count; "
                    f"local_rank={local_rank}, device_count={torch.cuda.device_count()}"
                )
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            device = requested
            backend = "gloo"

        owns_process_group = False
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                timeout=timedelta(seconds=timeout_seconds),
            )
            owns_process_group = True
        actual_world_size = dist.get_world_size()
        actual_rank = dist.get_rank()
        actual_backend = str(dist.get_backend())
        if actual_world_size != world_size or actual_rank != rank:
            raise RuntimeError(
                "torch.distributed state disagrees with launcher environment; "
                f"environment=({rank}, {world_size}), distributed=({actual_rank}, {actual_world_size})"
            )
        if actual_backend != backend:
            raise RuntimeError(
                f"distributed backend mismatch: expected {backend}, got {actual_backend}"
            )
        return cls(
            identity=DistributedIdentity(
                active=True,
                backend=backend,
                rank=rank,
                local_rank=local_rank,
                world_size=world_size,
                device=str(device),
            ),
            device=device,
            owns_process_group=owns_process_group,
        )

    @property
    def active(self) -> bool:
        return self.identity.active

    @property
    def rank(self) -> int:
        return self.identity.rank

    @property
    def local_rank(self) -> int:
        return self.identity.local_rank

    @property
    def world_size(self) -> int:
        return self.identity.world_size

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def wrap_model(self, model: nn.Module) -> nn.Module:
        if not self.active:
            return model
        if self.device.type == "cuda":
            return DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        return DistributedDataParallel(
            model,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    def no_sync_context(self, model: nn.Module, *, synchronize: bool):
        if self.active and not synchronize:
            if not isinstance(model, DistributedDataParallel):
                raise TypeError("active distributed training requires a DDP-wrapped model")
            return model.no_sync()
        return nullcontext()

    def barrier(self) -> None:
        if self.active:
            dist.barrier()

    def mean_tensor(self, value: Tensor) -> Tensor:
        result = value.detach().clone()
        if self.active:
            dist.all_reduce(result, op=dist.ReduceOp.SUM)
            result /= self.world_size
        return result

    def mean_float(self, value: Tensor) -> float:
        reduced = self.mean_tensor(value.to(dtype=torch.float64))
        return float(reduced.item())

    def all_true(self, condition: bool) -> bool:
        flag = torch.tensor(
            1 if condition else 0,
            dtype=torch.int32,
            device=self.device,
        )
        if self.active:
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def require_all_true(self, condition: bool, message: str) -> None:
        if not self.all_true(condition):
            raise FloatingPointError(message)

    def all_gather_object(self, value: Any) -> List[Any]:
        if not self.active:
            return [value]
        gathered: List[Any] = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, value)
        return gathered

    def assert_identical_object(self, value: Any, description: str) -> None:
        gathered = self.all_gather_object(value)
        reference = gathered[0]
        mismatches = [index for index, item in enumerate(gathered) if item != reference]
        if mismatches:
            raise RuntimeError(
                f"distributed {description} differs across ranks; mismatching ranks={mismatches}"
            )

    def report(self) -> Dict[str, Any]:
        return self.identity.to_dict()

    def close(self) -> None:
        if self.active and self.owns_process_group and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
            self.owns_process_group = False


__all__ = ["DistributedContext", "DistributedIdentity"]
# ^^^ THOG
