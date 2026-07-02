# vvv THOG
from __future__ import annotations

import unittest

import torch

from sheet.batch_source import DeterministicBatchSource
from tests.stage5_test_support import token_splits


class Stage5BatchShardingTests(unittest.TestCase):
    def test_s5_global_batch_is_partitioned_without_overlap(self) -> None:
        train_tokens, validation_tokens = token_splits()
        common = dict(
            block_size=8,
            batch_size=4,
            data_seed=202,
            world_size=2,
        )
        rank_zero = DeterministicBatchSource(
            train_tokens,
            validation_tokens,
            rank=0,
            **common,
        )
        rank_one = DeterministicBatchSource(
            train_tokens,
            validation_tokens,
            rank=1,
            **common,
        )
        single = DeterministicBatchSource(
            train_tokens,
            validation_tokens,
            block_size=8,
            batch_size=4,
            data_seed=202,
        )

        batch_zero = rank_zero.get_batch("train", device="cpu")
        batch_one = rank_one.get_batch("train", device="cpu")
        batch_single = single.get_batch("train", device="cpu")

        self.assertEqual(batch_zero.starts + batch_one.starts, batch_single.starts)
        self.assertTrue(torch.equal(
            torch.cat((batch_zero.inputs, batch_one.inputs), dim=0),
            batch_single.inputs,
        ))
        self.assertTrue(torch.equal(
            torch.cat((batch_zero.targets, batch_one.targets), dim=0),
            batch_single.targets,
        ))
        self.assertEqual(rank_zero.training_trace(), rank_one.training_trace())
        self.assertEqual(rank_zero.training_trace(), single.training_trace())

    def test_s5_global_batch_requires_equal_rank_shards(self) -> None:
        train_tokens, validation_tokens = token_splits()
        with self.assertRaisesRegex(ValueError, "divisible by world_size"):
            DeterministicBatchSource(
                train_tokens,
                validation_tokens,
                block_size=8,
                batch_size=3,
                data_seed=202,
                rank=0,
                world_size=2,
            )

    def test_s5_batch_state_rejects_world_size_change(self) -> None:
        train_tokens, validation_tokens = token_splits()
        distributed = DeterministicBatchSource(
            train_tokens,
            validation_tokens,
            block_size=8,
            batch_size=4,
            data_seed=202,
            rank=0,
            world_size=2,
        )
        state = distributed.state_dict()
        single = DeterministicBatchSource(
            train_tokens,
            validation_tokens,
            block_size=8,
            batch_size=4,
            data_seed=202,
        )
        with self.assertRaisesRegex(ValueError, "world_size"):
            single.load_state_dict(state)


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
