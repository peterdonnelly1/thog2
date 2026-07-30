# vvv THOG
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from torch import nn

from sheet import stage4_trainer


class TorchCompileRuntimeTests(unittest.TestCase):
    def test_default_false_returns_raw_model_without_compiling(self) -> None:
        model = nn.Linear(4, 4)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("THOG2_TORCH_COMPILE", None)
            with patch.object(stage4_trainer.torch, "compile") as compile_function:
                execution_model = stage4_trainer._execution_model(model)
        self.assertIs(execution_model, model)
        compile_function.assert_not_called()

    def test_true_wraps_execution_model_once_and_preserves_raw_model(self) -> None:
        model = nn.Linear(4, 4)
        compiled_model = nn.Sequential(model)
        with patch.dict(os.environ, {"THOG2_TORCH_COMPILE": "true"}, clear=False):
            with patch.object(stage4_trainer.torch, "compile", return_value=compiled_model) as compile_function:
                execution_model = stage4_trainer._execution_model(model)
        self.assertIs(execution_model, compiled_model)
        compile_function.assert_called_once_with(model)
        self.assertIs(compiled_model[0], model)

    def test_false_explicitly_bypasses_compile(self) -> None:
        model = nn.Linear(4, 4)
        with patch.dict(os.environ, {"THOG2_TORCH_COMPILE": "false"}, clear=False):
            with patch.object(stage4_trainer.torch, "compile") as compile_function:
                execution_model = stage4_trainer._execution_model(model)
        self.assertIs(execution_model, model)
        compile_function.assert_not_called()

    def test_invalid_environment_value_is_rejected(self) -> None:
        model = nn.Linear(4, 4)
        with patch.dict(os.environ, {"THOG2_TORCH_COMPILE": "maybe"}, clear=False):
            with self.assertRaisesRegex(ValueError, "THOG2_TORCH_COMPILE must be true or false"):
                stage4_trainer._execution_model(model)


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
