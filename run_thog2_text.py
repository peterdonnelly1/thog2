# vvv THOG
from __future__ import annotations

from pathlib import Path

import torch

from sheet.prompt_source import load_prompt_text
from sheet.text_model import load_text_model
from sheet.text_output import generate_text_outputs


checkpoint = "out-thog2/ckpt.pt"
prompt = "\n"
prompt_file = None
meta_path = None
num_samples = 1
max_new_tokens = 64
temperature = 0.8
top_k = 200
seed = 1337
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32"
exec(open("configurator.py").read())


def main() -> None:
    prompt_text = load_prompt_text(
        inline_text=prompt if prompt_file is None else None,
        file_path=Path(prompt_file) if prompt_file is not None else None,
    )
    model, config = load_text_model(
        Path(checkpoint),
        device=device,
        dtype=dtype,
    )
    outputs = generate_text_outputs(
        model,
        config,
        prompt_text,
        meta_path=Path(meta_path) if meta_path is not None else None,
        device=device,
        dtype=dtype,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
    )
    for output in outputs:
        print(output)
        print("---------------")


if __name__ == "__main__":
    main()
# ^^^ THOG
