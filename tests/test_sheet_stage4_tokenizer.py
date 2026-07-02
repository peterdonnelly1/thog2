# vvv THOG
from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

from sheet.prompt_tokens import prepare_prompt_tokens
from sheet.tokenizer import load_text_tokenizer


class Stage4TokenizerTests(unittest.TestCase):
    def test_s4_10_meta_tokenizer_round_trip(self) -> None:
        metadata = {
            "stoi": {"a": 0, "b": 1, " ": 2},
            "itos": {0: "a", 1: "b", 2: " "},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meta.pkl"
            with path.open("wb") as handle:
                pickle.dump(metadata, handle)
            prepared = prepare_prompt_tokens(
                "ab a",
                vocab_size=3,
                meta_path=path,
            )
        self.assertEqual(prepared.tokens.tolist(), [[0, 1, 2, 0]])
        self.assertEqual(prepared.decode([0, 1, 2, 0]), "ab a")

    def test_s4_10_invalid_meta_fails_directly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meta.pkl"
            with path.open("wb") as handle:
                pickle.dump({"vocab_size": 3}, handle)
            with self.assertRaisesRegex(ValueError, "stoi and itos"):
                load_text_tokenizer(path)


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
