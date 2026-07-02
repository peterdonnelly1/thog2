# vvv THOG
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sheet.byte_tokens import decode_text_bytes, encode_text_bytes
from sheet.checkpoints import load_payload
from sheet.compact_state import model_from_compact_state
from sheet.prompt_source import load_prompt_text
from sheet.sampling import generate_samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate continuations directly from a compact THOG2 checkpoint"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--prompt-file", type=Path)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
    )
    arguments = parser.parse_args()

    prompt_text = load_prompt_text(
        inline_text=arguments.prompt,
        file_path=arguments.prompt_file,
    )
    payload = load_payload(arguments.checkpoint, map_location="cpu")
    model, config = model_from_compact_state(
        payload,
        device=arguments.device,
        dtype=arguments.dtype,
    )
    prompt = encode_text_bytes(prompt_text, config.vocab_size)
    outputs = generate_samples(
        model,
        prompt,
        device=torch.device(arguments.device),
        dtype=arguments.dtype,
        num_samples=arguments.num_samples,
        max_new_tokens=arguments.max_new_tokens,
        temperature=arguments.temperature,
        top_k=arguments.top_k,
        seed=arguments.seed,
    )
    for output in outputs:
        print(decode_text_bytes(output))
        print("---------------")


if __name__ == "__main__":
    main()
# ^^^ THOG
