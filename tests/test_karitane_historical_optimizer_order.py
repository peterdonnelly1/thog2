from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch

from run_thog2_owt import OwtTrainer
from sheet.checkpoints import optimizer_group_names
from sheet.compact_identity import GEOMETRY_PRESET_LEGACY_SHEET_COL
from tests.test_karitane_legacy_sheet_checkpoint_resume import (
    KaritaneLegacySheetCheckpointResumeTests,
)


def _historical_schema_one_optimizer(trainer: OwtTrainer) -> torch.optim.Optimizer:
    decay = {}
    no_decay = {}

    for family_name, parameter, metadata in trainer.raw_model.trajectory.named_semantic_parameters():
        target = decay if metadata.weight_decay else no_decay
        target[f"trajectory.coefficients.{family_name}"] = parameter

    sheet_parameter_ids = {
        id(parameter)
        for parameter in trainer.raw_model.trajectory.coefficients.values()
    }
    for name, parameter in trainer.raw_model.named_parameters():
        if id(parameter) in sheet_parameter_ids:
            continue
        if name in {
            "transformer.wte.weight",
            "transformer.wpe.weight",
            "lm_head.weight",
        }:
            decay[name] = parameter
        else:
            no_decay[name] = parameter

    decay_names = tuple(sorted(decay))
    no_decay_names = tuple(sorted(no_decay))
    groups = (
        {
            "params": [decay[name] for name in decay_names],
            "weight_decay": trainer.config.weight_decay,
            "group_name": "decay",
            "parameter_names": decay_names,
        },
        {
            "params": [no_decay[name] for name in no_decay_names],
            "weight_decay": 0.0,
            "group_name": "no_decay",
            "parameter_names": no_decay_names,
        },
    )
    return torch.optim.AdamW(
        groups,
        lr=trainer.config.learning_rate,
        betas=(trainer.config.beta1, trainer.config.beta2),
    )


def _assert_nested_equal(test_case: unittest.TestCase, actual: Any, expected: Any) -> None:
    if isinstance(expected, torch.Tensor):
        test_case.assertIsInstance(actual, torch.Tensor)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        return
    if isinstance(expected, dict):
        test_case.assertEqual(set(actual), set(expected))
        for key in expected:
            _assert_nested_equal(test_case, actual[key], expected[key])
        return
    if isinstance(expected, (list, tuple)):
        test_case.assertEqual(len(actual), len(expected))
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_equal(test_case, actual_item, expected_item)
        return
    test_case.assertEqual(actual, expected)


class KaritaneHistoricalOptimizerOrderTests(unittest.TestCase):
    def test_real_schema_one_optimizer_order_is_reconstructed_before_state_load(self):
        fixture = KaritaneLegacySheetCheckpointResumeTests()
        train_tokens = torch.arange(512, dtype=torch.long) % 32
        validation_tokens = torch.arange(256, dtype=torch.long) % 32

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture.config(
                root / "checkpoint",
                geometry_preset=GEOMETRY_PRESET_LEGACY_SHEET_COL,
            )
            source = OwtTrainer(config, train_tokens, validation_tokens)
            source.optimizer = _historical_schema_one_optimizer(source)
            source.train_one_update()
            payload = fixture.schema_one_payload(source)
            historical_groups = tuple(
                tuple(group)
                for group in payload["optimizer_group_parameter_names"]
            )
            expected_optimizer_state = payload["optimizer"]
            checkpoint = root / "schema_one_historical_optimizer.pt"
            torch.save(payload, checkpoint)
            source.close()

            probe = OwtTrainer(config, train_tokens, validation_tokens)
            self.assertNotEqual(
                optimizer_group_names(probe.optimizer),
                historical_groups,
                msg=(
                    "the regression fixture must retain the historical grouping "
                    "difference that caused the real KARITANE_LONG failure"
                ),
            )
            probe.close()

            resumed = OwtTrainer.from_checkpoint(
                checkpoint,
                train_tokens,
                validation_tokens,
                expected_config=config,
                overrides={
                    "max_updates": 3,
                    "eval_interval": 1,
                    "eval_batches": 1,
                    "checkpoint_interval": 1,
                    "log_interval": 1,
                    "out_dir": str(root / "checkpoint"),
                    "device": "cpu",
                    "dtype": "float32",
                    "nonfinite_update_policy": "skip",
                    "max_nonfinite_update_skips": 10,
                },
            )
            self.assertEqual(
                optimizer_group_names(resumed.optimizer),
                historical_groups,
            )
            _assert_nested_equal(
                self,
                resumed.optimizer.state_dict(),
                expected_optimizer_state,
            )
            resumed.train_one_update()
            self.assertEqual(resumed.state.completed_updates, 2)
            resumed.close()


if __name__ == "__main__":
    unittest.main()
