# vvv THOG
from __future__ import annotations

import unittest

from sheet.byte_tokens import decode_text_bytes, encode_text_bytes


class Stage4ByteTokenTests(unittest.TestCase):
    def test_s4_10_byte_token_round_trip(self) -> None:
        text = "THOG2 prompt"
        tokens = encode_text_bytes(text, 256)
        self.assertEqual(decode_text_bytes(tokens), text)
        with self.assertRaises(ValueError):
            encode_text_bytes(text, 64)


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
