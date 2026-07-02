# vvv THOG
from __future__ import annotations

from pathlib import Path

import torch

from .prompt_source import load_prompt_text
from .text_model import load_text_model
from .text_output import generate_text_outputs


def main() -> None:
    settings = {
        "checkpoint": "out-thog2/ckpt.pt",
        "prompt": "\n",
        "prompt_file": "",
        "meta_path": "",
        "num_samples": 1,
        "max_new_tokens": 64,
        "temperature": 0.8,
        "top_k": 200,
        "seed": 1337,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dtype": (
            "bfloat16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "float32"
        ),
    }
    exec(open("configurator.py").read(), settings)
    prompt_file = settings["prompt_file"]
    prompt_text = load_prompt_text(
        inline_text=settings["prompt"] if not prompt_file else None,
        file_path=Path(prompt_file) if prompt_file else None,
    )
    model, config = load_text_model(
        Path(settings["checkpoint"]),
        device=settings["device"],
        dtype=settings["dtype"],
    )
    meta_path = settings["meta_path"]
    outputs = generate_text_outputs(
        model,
        config,
        prompt_text,
        meta_path=Path(meta_path) if meta_path else None,
        device=settings["device"],
        dtype=settings["dtype"],
        num_samples=settings["num_samples"],
        max_new_tokens=settings["max_new_tokens"],
        temperature=settings["temperature"],
        top_k=settings["top_k"],
        seed=settings["seed"],
    )
    for output in outputs:
        print(output)
        print("---------------")
# ^^^ THOG
