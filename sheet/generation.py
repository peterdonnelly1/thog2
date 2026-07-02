# vvv THOG
from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import torch
from torch import Tensor, nn


def generate_tokens(
    model: nn.Module,
    prompt: Tensor,
    *,
    device: torch.device,
    dtype: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    seed: int = 1337,
) -> Tensor:
    if prompt.ndim != 2 or prompt.shape[0] <= 0 or prompt.shape[1] <= 0:
        raise ValueError("prompt must have shape [batch, time]")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a non-negative integer")
    if not isinstance(temperature, (int, float)) or temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if top_k is not None and (
        isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0
    ):
        raise ValueError("top_k must be None or a positive integer")
    if dtype == "float32":
        cast_context = nullcontext()
    else:
        cast_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        cast_context = torch.autocast(device_type=device.type, dtype=cast_dtype)
    fork_devices = []
    if device.type == "cuda":
        fork_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        with torch.no_grad(), cast_context:
            generated = model.generate(
                prompt.to(device),
                max_new_tokens,
                temperature=float(temperature),
                top_k=top_k,
            )
    return generated.detach().cpu()
# ^^^ THOG
