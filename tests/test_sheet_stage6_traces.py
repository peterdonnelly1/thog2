# vvv THOG
from __future__ import annotations

import unittest

import torch

from sheet.batch_source import DeterministicBatchSource


class Stage6TraceTests(unittest.TestCase):
    def make_source(self) -> DeterministicBatchSource:
        train = torch.arange(4096, dtype=torch.long) % 128
        validation = torch.roll(train, shifts=17)
        return DeterministicBatchSource(
            train,
            validation,
            block_size=16,
            batch_size=4,
            data_seed=6102,
        )

    def test_s6_05_identical_sources_have_identical_traces(self) -> None:
        left = self.make_source()
        right = self.make_source()
        for _ in range(5):
            left.get_batch("train", device="cpu")
            right.get_batch("train", device="cpu")
        for _ in range(3):
            left.get_batch("val", device="cpu")
            right.get_batch("val", device="cpu")
        self.assertEqual(left.training_trace(), right.training_trace())
        self.assertEqual(left.validation_trace(), right.validation_trace())
        self.assertEqual(left.trace_digest("train"), right.trace_digest("train"))
        self.assertEqual(left.trace_digest("val"), right.trace_digest("val"))
        self.assertEqual(left.trace_digest("all"), right.trace_digest("all"))

    def test_s6_06_split_digests_change_with_trace(self) -> None:
        source = self.make_source()
        source.get_batch("train", device="cpu")
        source.get_batch("val", device="cpu")
        self.assertNotEqual(source.trace_digest("train"), source.trace_digest("val"))
        prior_digest = source.trace_digest("train")
        source.get_batch("train", device="cpu")
        self.assertNotEqual(prior_digest, source.trace_digest("train"))

    def test_s6_unknown_trace_split_is_rejected(self) -> None:
        source = self.make_source()
        with self.assertRaisesRegex(ValueError, "invalid trace split"):
            source.trace_digest("unknown")


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
