# vvv THOG
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from run_thog2_stage6_one import load_tokens
from sheet.batch_source import DeterministicBatchSource


class Stage6MemmapTests(unittest.TestCase):
    def write_tokens(self, path: Path) -> np.ndarray:
        tokens = np.arange(2048, dtype=np.uint16) % 127
        tokens.tofile(path)
        return tokens

    def test_s6_14_token_loader_keeps_read_only_memmap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.bin"
            tokens = self.write_tokens(path)
            storage = load_tokens(path)
            self.assertIsInstance(storage, np.memmap)
            self.assertEqual(storage.mode, "r")
            self.assertEqual(storage.dtype, np.uint16)
            self.assertEqual(int(storage.size), int(tokens.size))

    def test_s6_15_memmap_and_tensor_batches_are_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.bin"
            tokens = self.write_tokens(path)
            mapped = load_tokens(path)
            mapped_source = DeterministicBatchSource(
                mapped,
                mapped,
                block_size=16,
                batch_size=4,
                data_seed=6102,
            )
            tensor = torch.from_numpy(tokens.astype(np.int64))
            tensor_source = DeterministicBatchSource(
                tensor,
                tensor,
                block_size=16,
                batch_size=4,
                data_seed=6102,
            )
            mapped_batch = mapped_source.get_batch("train", device="cpu")
            tensor_batch = tensor_source.get_batch("train", device="cpu")
            self.assertEqual(mapped_batch.starts, tensor_batch.starts)
            self.assertTrue(torch.equal(mapped_batch.inputs, tensor_batch.inputs))
            self.assertTrue(torch.equal(mapped_batch.targets, tensor_batch.targets))


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
