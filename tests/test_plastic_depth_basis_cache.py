# vvv THOG
from __future__ import annotations

import copy
import os
import unittest
from unittest.mock import patch

import torch

from sheet.basis import differentiable_chebyshev_first_kind_basis
from sheet.trainer import SharedTrainer
from sheet.training_model import TrainingSheetGPT
from tests.stage3_test_support import token_splits
from tests.test_plastic_depth import plastic_sheet_config, plastic_training_config


class PlasticDepthBasisCacheTests(unittest.TestCase):
    def test_forward_scoped_cache_matches_uncached_checkpointed_execution(self) -> None:
        torch.manual_seed(2468)
        cached = TrainingSheetGPT(
            plastic_sheet_config(
                n_layer=6,
                depth_order=4,
                plastic__layers_to_sample=None,
                plastic__do_learn_layer_count=True,
                plastic__initial_layer_count=4,
                plastic__max_permitted_layers=6,
            )
        )
        uncached = copy.deepcopy(cached)
        cached.set_checkpoint_segment_size(2)
        uncached.set_checkpoint_segment_size(2)
        uncached.trajectory.prepare_plastic_depth_basis_cache = (
            lambda: uncached.trajectory.clear_plastic_depth_basis_cache()
        )
        with torch.no_grad():
            for model in (cached, uncached):
                model.trajectory.coefficients["attention_query_weight"][:, :, 1].fill_(0.125)
                model.trajectory.coefficients["attention_output_weight"][:, :, 2].fill_(-0.075)
        indices = torch.arange(8, dtype=torch.long).view(1, 8) % 32

        cached_logits, cached_loss = cached(indices, indices)
        uncached_logits, uncached_loss = uncached(indices, indices)
        self.assertIsNotNone(cached_loss)
        self.assertIsNotNone(uncached_loss)
        cached_loss.backward()
        uncached_loss.backward()

        torch.testing.assert_close(cached_logits, uncached_logits, rtol=2.0e-5, atol=2.0e-6)
        torch.testing.assert_close(cached_loss, uncached_loss, rtol=2.0e-5, atol=2.0e-6)
        cached_parameters = dict(cached.named_parameters())
        uncached_parameters = dict(uncached.named_parameters())
        self.assertEqual(tuple(cached_parameters), tuple(uncached_parameters))
        for name, parameter in cached_parameters.items():
            reference = uncached_parameters[name]
            if parameter.grad is None or reference.grad is None:
                self.assertIsNone(parameter.grad, name)
                self.assertIsNone(reference.grad, name)
                continue
            torch.testing.assert_close(
                parameter.grad,
                reference.grad,
                rtol=3.0e-5,
                atol=3.0e-6,
                msg=name,
            )
        self.assertIsNone(cached.trajectory._plastic_depth_basis_cache)
        self.assertFalse(cached.trajectory._plastic_depth_runtime_basis_cache)

    def test_checkpointed_forward_constructs_basis_once(self) -> None:
        model = TrainingSheetGPT(
            plastic_sheet_config(
                n_layer=8,
                depth_order=4,
                plastic__layers_to_sample=None,
                plastic__do_learn_layer_count=True,
                plastic__initial_layer_count=4,
                plastic__max_permitted_layers=8,
            )
        )
        model.set_checkpoint_segment_size(2)
        indices = torch.arange(8, dtype=torch.long).view(1, 8) % 32
        with patch(
            "sheet.depth_trajectory.differentiable_chebyshev_first_kind_basis",
            wraps=differentiable_chebyshev_first_kind_basis,
        ) as basis_builder:
            _, loss = model(indices, indices)
            self.assertIsNotNone(loss)
            loss.backward()
            self.assertEqual(basis_builder.call_count, 1)
            self.assertIsNone(model.trajectory._plastic_depth_basis_cache)
            model.zero_grad(set_to_none=True)
            _, second_loss = model(indices, indices)
            self.assertIsNotNone(second_loss)
            second_loss.backward()
            self.assertEqual(basis_builder.call_count, 2)

    def test_retained_materialisations_construct_one_basis_per_update(self) -> None:
        train_tokens, validation_tokens = token_splits()
        with patch.dict(os.environ, {"THOG2_FAST_DISCARD": "false"}):
            trainer = SharedTrainer(
                plastic_training_config(
                    plastic__layers_to_sample=3,
                    n_layer=3,
                    depth_order=3,
                    gradient_accumulation_steps=2,
                ),
                train_tokens,
                validation_tokens,
            )
        try:
            with patch(
                "sheet.depth_trajectory.differentiable_chebyshev_first_kind_basis",
                wraps=differentiable_chebyshev_first_kind_basis,
            ) as basis_builder:
                trainer.train_one_update()
                self.assertEqual(basis_builder.call_count, 1)
                self.assertIsNone(trainer.raw_model.trajectory._plastic_depth_basis_cache)
                self.assertFalse(trainer.raw_model.trajectory._plastic_depth_runtime_basis_cache)
        finally:
            trainer.close()


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
