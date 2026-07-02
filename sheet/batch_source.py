# vvv THOG
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple, Union

import torch
from torch import Tensor


@dataclass(frozen=True)
class Batch:
    inputs: Tensor
    targets: Tensor
    starts: Tuple[int, ...]
    split: str


class DeterministicBatchSource:
    """Independent RNG streams with deterministic global-batch rank sharding."""

    def __init__(
        self,
        train_tokens: Tensor,
        validation_tokens: Tensor,
        *,
        block_size: int,
        batch_size: int,
        data_seed: int,
        rank: int = 0,
        world_size: int = 1,
        trace_limit: int = 10000,
    ) -> None:
        if train_tokens.ndim != 1 or validation_tokens.ndim != 1:
            raise ValueError("token splits must be one-dimensional")
        if train_tokens.numel() <= block_size or validation_tokens.numel() <= block_size:
            raise ValueError("each token split must contain more than block_size tokens")
        if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
            raise ValueError(f"world_size must be a positive integer; got {world_size!r}")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0 or rank >= world_size:
            raise ValueError(
                f"rank must be in [0, world_size); got rank={rank!r}, world_size={world_size}"
            )
        if batch_size % world_size != 0:
            raise ValueError(
                "global batch_size must be divisible by world_size; "
                f"got batch_size={batch_size}, world_size={world_size}"
            )
        self.train_tokens = train_tokens.detach().to(dtype=torch.long, device="cpu")
        self.validation_tokens = validation_tokens.detach().to(dtype=torch.long, device="cpu")
        self.block_size = block_size
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.local_batch_size = batch_size // world_size
        self.trace_limit = trace_limit
        self.train_generator = torch.Generator(device="cpu").manual_seed(data_seed)
        self.validation_generator = torch.Generator(device="cpu").manual_seed(data_seed + 1)
        self.trace: List[Dict[str, Any]] = []

    def get_batch(self, split: str, *, device: Union[str, torch.device]) -> Batch:
        if split == "train":
            storage, generator = self.train_tokens, self.train_generator
        elif split == "val":
            storage, generator = self.validation_tokens, self.validation_generator
        else:
            raise ValueError(f"invalid split: {split!r}")
        starts_tensor = torch.randint(
            storage.numel() - self.block_size,
            (self.batch_size,),
            generator=generator,
        )
        global_starts = tuple(int(value) for value in starts_tensor.tolist())
        local_start = self.rank * self.local_batch_size
        local_end = local_start + self.local_batch_size
        starts = global_starts[local_start:local_end]
        inputs = torch.stack([storage[start:start + self.block_size].clone() for start in starts])
        targets = torch.stack([storage[start + 1:start + self.block_size + 1].clone() for start in starts])
        if len(self.trace) < self.trace_limit:
            self.trace.append({"split": split, "starts": global_starts})
        target_device = torch.device(device)
        if target_device.type == "cuda":
            inputs = inputs.pin_memory().to(target_device, non_blocking=True)
            targets = targets.pin_memory().to(target_device, non_blocking=True)
        else:
            inputs, targets = inputs.to(target_device), targets.to(target_device)
        return Batch(inputs, targets, starts, split)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "train_generator_state": self.train_generator.get_state(),
            "validation_generator_state": self.validation_generator.get_state(),
            "trace": list(self.trace),
            "block_size": self.block_size,
            "batch_size": self.batch_size,
            "world_size": self.world_size,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["block_size"]) != self.block_size:
            raise ValueError("batch-source block_size is incompatible with checkpoint")
        if int(state["batch_size"]) != self.batch_size:
            raise ValueError("batch-source batch_size is incompatible with checkpoint")
        checkpoint_world_size = int(state.get("world_size", 1))
        if checkpoint_world_size != self.world_size:
            raise ValueError(
                "batch-source world_size is incompatible with checkpoint; "
                f"checkpoint={checkpoint_world_size}, current={self.world_size}"
            )
        self.train_generator.set_state(state["train_generator_state"])
        self.validation_generator.set_state(state["validation_generator_state"])
        self.trace = list(state.get("trace", []))

    def full_trace(self) -> Tuple[Tuple[str, Tuple[int, ...]], ...]:
        return tuple(
            (str(item["split"]), tuple(int(value) for value in item["starts"]))
            for item in self.trace
        )

    def training_trace(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(
            starts
            for split, starts in self.full_trace()
            if split == "train"
        )

    def validation_trace(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(
            starts
            for split, starts in self.full_trace()
            if split == "val"
        )

    def trace_digest(self, split: str) -> str:
        if split == "train":
            trace = self.training_trace()
        elif split == "val":
            trace = self.validation_trace()
        elif split == "all":
            trace = self.full_trace()
        else:
            raise ValueError(f"invalid trace split: {split!r}")
        payload = json.dumps(trace, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
# ^^^ THOG
