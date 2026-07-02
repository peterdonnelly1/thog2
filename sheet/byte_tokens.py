# vvv THOG
from __future__ import annotations

import torch
from torch import Tensor


def encode_text_bytes(text: str, vocab_size: int) -> Tensor:
    values = list(text.encode("utf-8"))
    if not values:
        raise ValueError("prompt text must be non-empty")
    if vocab_size <= max(values):
        raise ValueError("vocabulary is too small for UTF-8 byte token conversion")
    return torch.tensor([values], dtype=torch.long)


def decode_text_bytes(tokens: Tensor) -> str:
    values = [int(value) for value in tokens.detach().cpu().reshape(-1)]
    normalized = bytes(value if 0 <= value <= 255 else 63 for value in values)
    return normalized.decode("utf-8", errors="replace")
# ^^^ THOG
