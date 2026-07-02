# vvv THOG
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
from torch import nn

from .prompt_tokens import prepare_prompt_tokens
from .sampling import generate_samples
from .training_config import TrainingConfig


def generate_text_outputs(
    model: nn.Module,
    config: TrainingConfig,
    prompt_text: str,
    *,
    meta_path: Optional[Path],
    device: str,
    dtype: str,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
) -> List[str]:
    prepared = prepare_prompt_tokens(
        prompt_text,
        vocab_size=config.vocab_size,
        meta_path=meta_path,
    )
    outputs = generate_samples(
        model,
        prepared.tokens,
        device=torch.device(device),
        dtype=dtype,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
    )
    return [
        prepared.decode([int(token_id) for token_id in output])
        for output in outputs
    ]
# ^^^ THOG
