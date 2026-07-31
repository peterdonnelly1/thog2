# vvv THOG
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import torch
from torch import nn

from sheet import stage4_trainer, training_model
from sheet.checkpointing import execute_logical_layers


class _RegionalCapableLinear(nn.Linear):
    def __init__(self) -> None:
        super().__init__(4, 4)
        self.compile_modes = []

    def set_torch_compile_mode(self, mode: str) -> None:
        self.compile_modes.append(mode)


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

    def test_regional_keeps_outer_model_eager_and_enables_segment_compile(self) -> None:
        model = _RegionalCapableLinear()
        with patch.dict(os.environ, {"THOG2_TORCH_COMPILE": "regional"}, clear=False):
            with patch.object(stage4_trainer.torch, "compile") as compile_function:
                execution_model = stage4_trainer._execution_model(model)
        self.assertIs(execution_model, model)
        self.assertEqual(model.compile_modes, ["regional"])
        compile_function.assert_not_called()

    def test_regional_rejects_model_without_segment_compile_support(self) -> None:
        model = nn.Linear(4, 4)
        with patch.dict(os.environ, {"THOG2_TORCH_COMPILE": "regional"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "set_torch_compile_mode"):
                stage4_trainer._execution_model(model)

    def test_invalid_environment_value_is_rejected(self) -> None:
        model = nn.Linear(4, 4)
        with patch.dict(os.environ, {"THOG2_TORCH_COMPILE": "maybe"}, clear=False):
            with self.assertRaisesRegex(ValueError, "false, true, or regional"):
                stage4_trainer._execution_model(model)


class RegionalCompileSegmentTests(unittest.TestCase):
    def test_segment_compiler_disables_pointwise_autotuning(self) -> None:
        def logical_block(hidden: torch.Tensor, layer_index: int) -> torch.Tensor:
            return hidden + layer_index

        compiled_runner = lambda hidden: hidden
        with patch.object(training_model.torch, "compile", return_value=compiled_runner) as compile_function:
            runner = training_model._compiled_segment_runner(logical_block, (0, 1, 2, 3))
        self.assertIs(runner, compiled_runner)
        self.assertEqual(compile_function.call_count, 1)
        self.assertEqual(
            compile_function.call_args.kwargs,
            {"options": {"triton.autotune_pointwise": False}},
        )

    def test_checkpoint_executor_uses_requested_regional_segments(self) -> None:
        requested_segments = []

        def logical_block(hidden: torch.Tensor, layer_index: int) -> torch.Tensor:
            return hidden + float(layer_index + 1)

        def regional_segment_runner_factory(layer_indices):
            requested_segments.append(layer_indices)

            def run_segment(hidden: torch.Tensor) -> torch.Tensor:
                output = hidden
                for layer_index in layer_indices:
                    output = logical_block(output, layer_index)
                return output

            return run_segment

        hidden = torch.zeros(1, requires_grad=True)
        output, report = execute_logical_layers(
            hidden,
            n_layer=4,
            segment_size=2,
            logical_block=logical_block,
            training=True,
            regional_segment_runner_factory=regional_segment_runner_factory,
        )
        self.assertEqual(requested_segments, [(0, 1), (2, 3)])
        self.assertEqual(float(output.detach()), 10.0)
        self.assertTrue(report.checkpointing_used)
        self.assertEqual(report.checkpoint_segments, 2)


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
