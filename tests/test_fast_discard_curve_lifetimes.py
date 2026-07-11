# vvv THOG
from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from sheet.compact_identity import GEOMETRY_PRESET_CURVE
from sheet.model import SheetGPT, SheetGPTConfig
from sheet.training_model import TrainingSheetGPT


class FastDiscardCurveLifetimeTests(unittest.TestCase):
    def curve_config(self, *, fast_discard: bool, bias: bool = True) -> SheetGPTConfig:
        with patch.dict("os.environ", {"THOG2_MLP_CHANNEL_ORDER": "16"}):
            return SheetGPTConfig(
                block_size=8,
                vocab_size=32,
                n_layer=4,
                n_head=2,
                n_embd=8,
                dropout=0.0,
                bias=bias,
                depth_order=3,
                base_row_order=4,
                geometry_preset=GEOMETRY_PRESET_CURVE,
                fast_discard=fast_discard,
            )

    def assert_fast_discard_matches_reference(
        self,
        *,
        model_cls: type[SheetGPT],
        checkpoint_segment_size: int,
        bias: bool,
    ) -> None:
        torch.manual_seed(7319)
        reference = model_cls(self.curve_config(fast_discard=False, bias=bias))
        candidate = model_cls(self.curve_config(fast_discard=True, bias=bias))
        candidate.load_state_dict(reference.state_dict())
        if hasattr(reference, "set_checkpoint_segment_size"):
            reference.set_checkpoint_segment_size(checkpoint_segment_size)
            candidate.set_checkpoint_segment_size(checkpoint_segment_size)
        reference.train()
        candidate.train()

        idx = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10, 11],
            ],
            dtype=torch.long,
        )
        targets = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12],
            ],
            dtype=torch.long,
        )

        reference_logits, reference_loss = reference(idx, targets)
        candidate_logits, candidate_loss = candidate(idx, targets)
        self.assertIsNotNone(reference_loss)
        self.assertIsNotNone(candidate_loss)
        torch.testing.assert_close(candidate_logits, reference_logits, rtol=0.0, atol=0.0)
        torch.testing.assert_close(candidate_loss, reference_loss, rtol=0.0, atol=0.0)
        if hasattr(reference, "last_execution_report"):
            self.assertEqual(candidate.last_execution_report, reference.last_execution_report)

        reference_loss.backward()
        candidate_loss.backward()

        reference_gradients = dict(reference.named_parameters())
        for name, candidate_parameter in candidate.named_parameters():
            with self.subTest(parameter=name, bias=bias, checkpoint_segment_size=checkpoint_segment_size):
                reference_gradient = reference_gradients[name].grad
                candidate_gradient = candidate_parameter.grad
                if reference_gradient is None:
                    self.assertIsNone(candidate_gradient)
                    continue
                self.assertIsNotNone(candidate_gradient)
                torch.testing.assert_close(candidate_gradient, reference_gradient, rtol=0.0, atol=0.0)

    def test_01_curve_fast_discard_can_be_enabled_by_environment_without_touching_call_sites(self) -> None:
        with patch.dict("os.environ", {"THOG2_FAST_DISCARD": "true", "THOG2_MLP_CHANNEL_ORDER": "16"}):
            self.assertTrue(
                SheetGPTConfig(
                    block_size=8,
                    vocab_size=32,
                    n_layer=4,
                    n_head=2,
                    n_embd=8,
                    dropout=0.0,
                    bias=True,
                    depth_order=3,
                    base_row_order=4,
                    geometry_preset=GEOMETRY_PRESET_CURVE,
                ).fast_discard
            )
        with patch.dict("os.environ", {"THOG2_FAST_DISCARD": "false", "THOG2_MLP_CHANNEL_ORDER": "16"}):
            self.assertFalse(
                SheetGPTConfig(
                    block_size=8,
                    vocab_size=32,
                    n_layer=4,
                    n_head=2,
                    n_embd=8,
                    dropout=0.0,
                    bias=True,
                    depth_order=3,
                    base_row_order=4,
                    geometry_preset=GEOMETRY_PRESET_CURVE,
                ).fast_discard
            )

    def test_02_curve_fast_discard_preserves_forward_loss_and_all_gradients_without_activation_checkpointing(self) -> None:
        for bias in (True, False):
            with self.subTest(bias=bias):
                self.assert_fast_discard_matches_reference(
                    model_cls=SheetGPT,
                    checkpoint_segment_size=0,
                    bias=bias,
                )

    def test_03_curve_fast_discard_preserves_forward_loss_and_all_gradients_with_activation_checkpointing(self) -> None:
        for segment_size in (1, 3):
            for bias in (True, False):
                with self.subTest(segment_size=segment_size, bias=bias):
                    self.assert_fast_discard_matches_reference(
                        model_cls=TrainingSheetGPT,
                        checkpoint_segment_size=segment_size,
                        bias=bias,
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
