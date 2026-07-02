# vvv THOG
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional


@dataclass(frozen=True)
class TextTokenizer:
    encode: Callable[[str], List[int]]
    decode: Callable[[List[int]], str]
    source: str


def load_text_tokenizer(meta_path: Optional[Path] = None) -> TextTokenizer:
    if meta_path is not None:
        with meta_path.open("rb") as handle:
            metadata = pickle.load(handle)
        if "stoi" not in metadata or "itos" not in metadata:
            raise ValueError("meta.pkl must contain stoi and itos mappings")
        stoi = metadata["stoi"]
        itos = metadata["itos"]
        return TextTokenizer(
            encode=lambda text: [stoi[character] for character in text],
            decode=lambda token_ids: "".join(itos[token_id] for token_id in token_ids),
            source=str(meta_path),
        )
    try:
        import tiktoken
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "GPT-2 prompt encoding requires tiktoken; install it in the thog2 environment"
        ) from error
    encoding = tiktoken.get_encoding("gpt2")
    return TextTokenizer(
        encode=lambda text: encoding.encode(
            text,
            allowed_special={"<|endoftext|>"},
        ),
        decode=lambda token_ids: encoding.decode(token_ids),
        source="gpt2",
    )
# ^^^ THOG
