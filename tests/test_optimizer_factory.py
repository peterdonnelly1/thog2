# vvv THOG
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import torch
from torch import nn

from sheet.optimizer_factory import (
    build_optimizer,
    normalize_optimizer_name,
    optimizer_momentum_from_environment,
)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matrix = nn.Parameter(torch.ones(3, 2))
        self.vector = nn.Parameter(torch.ones(3))


class OptimizerFactoryTests(unittest.TestCase):
    def build(self, optimizer_name: str):
        model = _TinyModel()
        with patch.dict(
            os.environ,
            {
                "THOG2_OPTIMIZER": optimizer_name,
                "THOG2_OPTIMIZER_MOMENTUM": "0.9",
            },
            clear=False,
        ):
            optimizer = build_optimizer(
                model,
                weight_decay=0.1,
                learning_rate=1.0e-3,
                betas=(0.9, 0.95),
                device_type="cpu",
            )
        return optimizer

    def test_adamw_is_default_family(self) -> None:
        optimizer = self.build("adamw")
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.param_groups[0]["thog2_optimizer_name"], "adamw")

    def test_sgd_uses_momentum(self) -> None:
        optimizer = self.build("sgd")
        self.assertIsInstance(optimizer, torch.optim.SGD)
        self.assertEqual(optimizer.defaults["momentum"], 0.9)
        self.assertFalse(optimizer.defaults["nesterov"])

    def test_sgd_nesterov_uses_nesterov_momentum(self) -> None:
        optimizer = self.build("sgd_nesterov")
        self.assertIsInstance(optimizer, torch.optim.SGD)
        self.assertEqual(optimizer.defaults["momentum"], 0.9)
        self.assertTrue(optimizer.defaults["nesterov"])

    @unittest.skipUnless(hasattr(torch.optim, "Adafactor"), "PyTorch lacks Adafactor")
    def test_adafactor_is_available(self) -> None:
        optimizer = self.build("adafactor")
        self.assertIsInstance(optimizer, torch.optim.Adafactor)

    def test_rmsprop_uses_momentum(self) -> None:
        optimizer = self.build("rmsprop")
        self.assertIsInstance(optimizer, torch.optim.RMSprop)
        self.assertEqual(optimizer.defaults["momentum"], 0.9)

    def test_parameter_groups_preserve_decay_split(self) -> None:
        optimizer = self.build("sgd")
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.1)
        self.assertEqual(optimizer.param_groups[1]["weight_decay"], 0.0)
        self.assertEqual(optimizer.param_groups[0]["parameter_names"], ("matrix",))
        self.assertEqual(optimizer.param_groups[1]["parameter_names"], ("vector",))

    def test_aliases_are_normalized(self) -> None:
        self.assertEqual(normalize_optimizer_name("Adam"), "adamw")
        self.assertEqual(normalize_optimizer_name("nesterov"), "sgd_nesterov")
        self.assertEqual(normalize_optimizer_name("sgd-nesterov"), "sgd_nesterov")

    def test_unknown_optimizer_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported optimizer"):
            normalize_optimizer_name("made_up")

    def test_invalid_momentum_is_rejected(self) -> None:
        with patch.dict(
            os.environ,
            {"THOG2_OPTIMIZER_MOMENTUM": "1.0"},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "must be in"):
                optimizer_momentum_from_environment()


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
