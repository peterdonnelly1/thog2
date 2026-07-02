# vvv THOG
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .generation import generate_tokens


def generate_samples(
    model: nn.Module,
    prompt: Tensor,
    *,
    device: torch.device,
    dtype: str,
    num_samples: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    seed: int = 1337,
) -> Tensor:
    if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")
    outputs = [
        generate_tokens(
            model,
            prompt,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=seed + sample_index,
        )
        for sample_index in range(num_samples)
    ]
    return torch.cat(outputs, dim=0)
# ^^^ THOG
