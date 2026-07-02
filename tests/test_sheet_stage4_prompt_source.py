# vvv THOG
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sheet.prompt_source import load_prompt_text


class Stage4PromptSourceTests(unittest.TestCase):
    def test_s4_10_inline_and_file_sources(self) -> None:
        self.assertEqual(load_prompt_text(inline_text="hello"), "hello")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompt.txt"
            path.write_text("world", encoding="utf-8")
            self.assertEqual(load_prompt_text(file_path=path), "world")
        with self.assertRaises(ValueError):
            load_prompt_text(inline_text="x", file_path=Path("x"))


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
