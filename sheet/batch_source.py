# vvv THOG
from __future__ import annotations

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
    """Independent train/validation RNG streams with an auditable trace."""

    def __init__(self, train_tokens: Tensor, validation_tokens: Tensor, *, block_size: int,
                 batch_size: int, data_seed: int, trace_limit: int = 10000) -> None:
        if train_tokens.ndim != 1 or validation_tokens.ndim != 1:
            raise ValueError("token splits must be one-dimensional")
        if train_tokens.numel() <= block_size or validation_tokens.numel() <= block_size:
            raise ValueError("each token split must contain more than block_size tokens")
        self.train_tokens = train_tokens.detach().to(dtype=torch.long, device="cpu")
        self.validation_tokens = validation_tokens.detach().to(dtype=torch.long, device="cpu")
        self.block_size = block_size
        self.batch_size = batch_size
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
        starts = tuple(int(value) for value in starts_tensor.tolist())
        inputs = torch.stack([storage[start:start + self.block_size].clone() for start in starts])
        targets = torch.stack([storage[start + 1:start + self.block_size + 1].clone() for start in starts])
        if len(self.trace) < self.trace_limit:
            self.trace.append({"split": split, "starts": starts})
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
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["block_size"]) != self.block_size:
            raise ValueError("batch-source block_size is incompatible with checkpoint")
        if int(state["batch_size"]) != self.batch_size:
            raise ValueError("batch-source batch_size is incompatible with checkpoint")
        self.train_generator.set_state(state["train_generator_state"])
        self.validation_generator.set_state(state["validation_generator_state"])
        self.trace = list(state.get("trace", []))

    def training_trace(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(tuple(item["starts"]) for item in self.trace if item["split"] == "train")
# ^^^ THOG
