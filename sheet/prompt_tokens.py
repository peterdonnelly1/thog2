# vvv THOG
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import torch
from torch import Tensor

from .tokenizer import load_text_tokenizer


@dataclass(frozen=True)
class PreparedPrompt:
    tokens: Tensor
    decode: Callable[[List[int]], str]
    tokenizer_source: str


def prepare_prompt_tokens(
    text: str,
    *,
    vocab_size: int,
    meta_path: Optional[Path] = None,
) -> PreparedPrompt:
    tokenizer = load_text_tokenizer(meta_path)
    token_ids = tokenizer.encode(text)
    if not token_ids:
        raise ValueError("encoded prompt must contain at least one token")
    if max(token_ids) >= vocab_size:
        raise ValueError("encoded prompt contains a token outside checkpoint vocabulary")
    return PreparedPrompt(
        tokens=torch.tensor([token_ids], dtype=torch.long),
        decode=tokenizer.decode,
        tokenizer_source=tokenizer.source,
    )
# ^^^ THOG
