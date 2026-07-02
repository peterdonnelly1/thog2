# vvv THOG
from __future__ import annotations

from pathlib import Path
from typing import Optional


def load_prompt_text(
    *,
    inline_text: Optional[str] = None,
    file_path: Optional[Path] = None,
) -> str:
    if inline_text is not None and file_path is not None:
        raise ValueError("provide either inline_text or file_path, not both")
    if file_path is not None:
        text = file_path.read_text(encoding="utf-8")
    elif inline_text is not None:
        text = inline_text
    else:
        raise ValueError("an inline prompt or prompt file is required")
    if not text:
        raise ValueError("prompt text must be non-empty")
    return text
# ^^^ THOG
