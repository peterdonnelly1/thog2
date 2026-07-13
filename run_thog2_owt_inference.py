# vvv THOG
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Callable, Tuple

import torch

from sheet.checkpoints import load_payload
from sheet.compact_state import model_from_compact_state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run THOG2/nanoGPT checkpoint inference for OpenWebText-style GPT-2 tokens")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/openwebtext")
    parser.add_argument("--prompt", default="The meaning of life is")
    parser.add_argument("--prompt-file")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--attention-backend", choices=("auto", "flash2", "sdpa", "math"), default="auto")
    return parser


def configure_attention_backend(attention_backend: str) -> None:
    if attention_backend == "auto" or not torch.cuda.is_available():
        return
    if attention_backend == "flash2":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        return
    if attention_backend == "sdpa":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        return
    if attention_backend == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        return
    raise ValueError(f"unsupported attention backend: {attention_backend}")


def load_codec(data_dir: Path) -> Tuple[Callable[[str], list[int]], Callable[[list[int]], str]]:
    meta_path = data_dir / "meta.pkl"
    if meta_path.is_file():
        with meta_path.open("rb") as handle:
            meta = pickle.load(handle)
        if "stoi" in meta and "itos" in meta:
            stoi = meta["stoi"]
            itos = meta["itos"]
            return (lambda text: [stoi[ch] for ch in text], lambda tokens: "".join(itos[index] for index in tokens))
    try:
        import tiktoken
    except ImportError as error:
        raise RuntimeError("OpenWebText inference requires tiktoken unless meta.pkl contains stoi/itos") from error
    encoding = tiktoken.get_encoding("gpt2")
    return (lambda text: encoding.encode(text, allowed_special={"<|endoftext|>"}), lambda tokens: encoding.decode(tokens))


def main() -> int:
    args = build_parser().parse_args()
    if args.num_samples < 1:
        raise ValueError("num_samples must be positive")
    if args.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if args.temperature <= 0.0:
        raise ValueError("temperature must be positive")
    configure_attention_backend(args.attention_backend)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    prompt = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else args.prompt
    encode, decode = load_codec(Path(args.data_dir))
    payload = load_payload(args.checkpoint)
    model, config = model_from_compact_state(payload, device=args.device, dtype=args.dtype)
    model.eval()
    device = next(model.parameters()).device
    idx = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    context = idx if idx.size(1) <= config.block_size else idx[:, -config.block_size :]
    print(f"checkpoint: {args.checkpoint}")
    print(f"model_type: {config.model_type}")
    print(f"basis: {config.basis_family} / {config.basis_version}")
    print(f"geometry: {config.geometry_preset} / {config.attention_geometry} / {config.mlp_geometry}")
    for sample_index in range(args.num_samples):
        with torch.no_grad():
            output = model.generate(context, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
        print(f"\n=== sample {sample_index + 1} ===")
        print(decode(output[0].tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# ^^^ THOG
