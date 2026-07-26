# vvv THOG
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sheet.checkpointing import execute_logical_layers
from sheet.layer_dropout import LayerDropoutConfig
from sheet.run_config import OwtRunConfig
from sheet.trainer_step import TrainerStepMixin


class LayerDropoutSamplerTests(unittest.TestCase):
    def test_all_active_bypasses_random_sampling(self) -> None:
        plan = LayerDropoutConfig(
            n_layer=144,
            stratum_size=144,
            active_per_stratum=144,
            resample_steps=1,
            seed=17,
        )
        with patch("sheet.layer_dropout.torch.randperm", side_effect=AssertionError("RNG must not run")):
            self.assertIsNone(plan.active_layer_indices(0))

    def test_exact_stratified_cardinality(self) -> None:
        plan = LayerDropoutConfig(
            n_layer=144,
            stratum_size=4,
            active_per_stratum=2,
            resample_steps=1,
            seed=17,
        )
        selected = plan.active_layer_indices(0)
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(len(selected), 72)
        self.assertEqual(tuple(sorted(selected)), selected)
        self.assertEqual(len(set(selected)), 72)
        for start in range(0, 144, 4):
            self.assertEqual(sum(start <= value < start + 4 for value in selected), 2)

    def test_single_stratum_is_uniform_exact_cardinality(self) -> None:
        plan = LayerDropoutConfig(
            n_layer=144,
            stratum_size=144,
            active_per_stratum=72,
            resample_steps=1,
            seed=17,
        )
        selected = plan.active_layer_indices(0)
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(len(selected), 72)
        self.assertEqual(len(set(selected)), 72)
        self.assertTrue(all(0 <= value < 144 for value in selected))

    def test_resample_bucket_reuses_selection(self) -> None:
        plan = LayerDropoutConfig(
            n_layer=144,
            stratum_size=4,
            active_per_stratum=2,
            resample_steps=10,
            seed=17,
        )
        self.assertEqual(plan.active_layer_indices(0), plan.active_layer_indices(9))
        self.assertNotEqual(plan.active_layer_indices(9), plan.active_layer_indices(10))

    def test_invalid_configuration_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            LayerDropoutConfig(n_layer=144, stratum_size=5, active_per_stratum=2)
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            LayerDropoutConfig(n_layer=144, stratum_size=4, active_per_stratum=5)


class LogicalLayerExecutionTests(unittest.TestCase):
    def test_sparse_execution_preserves_nominal_indices(self) -> None:
        seen = []

        def logical_block(hidden: torch.Tensor, layer_index: int) -> torch.Tensor:
            seen.append(layer_index)
            return hidden + float(layer_index + 1)

        output, report = execute_logical_layers(
            torch.tensor(0.0),
            n_layer=8,
            segment_size=0,
            logical_block=logical_block,
            training=True,
            layer_indices=(1, 3, 7),
        )
        self.assertEqual(seen, [1, 3, 7])
        self.assertEqual(float(output.item()), 14.0)
        self.assertEqual(report.logical_layers, 3)
        self.assertFalse(report.checkpointing_used)

    def test_sparse_checkpointing_chunks_active_sequence(self) -> None:
        seen = []

        def logical_block(hidden: torch.Tensor, layer_index: int) -> torch.Tensor:
            seen.append(layer_index)
            return hidden + float(layer_index + 1)

        hidden = torch.tensor(0.0, requires_grad=True)
        output, report = execute_logical_layers(
            hidden,
            n_layer=8,
            segment_size=2,
            logical_block=logical_block,
            training=True,
            layer_indices=(1, 3, 7),
        )
        output.backward()
        self.assertEqual(seen[:3], [1, 3, 7])
        self.assertEqual(float(output.detach().item()), 14.0)
        self.assertEqual(report.logical_layers, 3)
        self.assertEqual(report.checkpoint_segments, 2)
        self.assertTrue(report.checkpointing_used)
        self.assertEqual(float(hidden.grad.item()), 1.0)

    def test_all_layer_executor_path_is_unchanged_semantically(self) -> None:
        seen = []

        def logical_block(hidden: torch.Tensor, layer_index: int) -> torch.Tensor:
            seen.append(layer_index)
            return hidden + float(layer_index + 1)

        output, report = execute_logical_layers(
            torch.tensor(0.0),
            n_layer=4,
            segment_size=0,
            logical_block=logical_block,
            training=True,
        )
        self.assertEqual(seen, [0, 1, 2, 3])
        self.assertEqual(float(output.item()), 10.0)
        self.assertEqual(report.logical_layers, 4)


class LayerDropoutTrainerLifecycleTests(unittest.TestCase):
    class _DummyTrainer(TrainerStepMixin):
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                layer_dropout_enabled=True,
                layer_dropout_resample_steps=3,
                n_layer=12,
                layer_dropout_stratum_size=4,
                layer_dropout_active_per_stratum=2,
                model_seed=31,
            )
            self.state = SimpleNamespace(completed_updates=0)
            self.raw_model = SimpleNamespace(set_active_layer_indices=self._set_active)
            self.selections = []
            self.events = []

        def _set_active(self, indices) -> None:
            self.selections.append(tuple(indices))

        def _record(self, name: str, **payload) -> None:
            self.events.append((name, payload))

    def test_selection_changes_only_at_resample_boundary(self) -> None:
        trainer = self._DummyTrainer()
        trainer._prepare_layer_dropout_for_update()
        first = trainer.selections[-1]
        trainer.state.completed_updates = 1
        trainer._prepare_layer_dropout_for_update()
        trainer.state.completed_updates = 2
        trainer._prepare_layer_dropout_for_update()
        self.assertEqual(len(trainer.selections), 1)
        trainer.state.completed_updates = 3
        trainer._prepare_layer_dropout_for_update()
        self.assertEqual(len(trainer.selections), 2)
        self.assertNotEqual(first, trainer.selections[-1])

    def test_all_active_trainer_path_does_not_construct_sampler(self) -> None:
        trainer = self._DummyTrainer()
        trainer.config.layer_dropout_enabled = False
        with patch("sheet.trainer_step.LayerDropoutConfig", side_effect=AssertionError("sampler must not be constructed")):
            trainer._prepare_layer_dropout_for_update()
        self.assertEqual(trainer.selections, [])


class LayerDropoutRunConfigTests(unittest.TestCase):
    def _config(self, **overrides) -> OwtRunConfig:
        values = {
            "model_type": "sheet",
            "n_layer": 144,
            "n_head": 12,
            "n_embd": 768,
            "max_iters": 100,
            "warmup_iters": 10,
        }
        values.update(overrides)
        return OwtRunConfig(**values)

    def test_omitted_controls_resolve_to_all_active(self) -> None:
        config = self._config()
        self.assertEqual(config.layer_dropout_stratum_size, 144)
        self.assertEqual(config.layer_dropout_active_per_stratum, 144)
        self.assertEqual(config.layer_dropout_n_strata, 1)
        self.assertEqual(config.n_active_layers, 144)
        self.assertFalse(config.layer_dropout_enabled)
        self.assertNotIn("LDs_", config.parameter_artifact_fragment())

    def test_active_dropout_is_named_and_propagated(self) -> None:
        config = self._config(
            layer_dropout_stratum_size=4,
            layer_dropout_active_per_stratum=2,
            layer_dropout_resample_steps=5,
        )
        self.assertEqual(config.layer_dropout_n_strata, 36)
        self.assertEqual(config.n_active_layers, 72)
        self.assertTrue(config.layer_dropout_enabled)
        fragment = config.parameter_artifact_fragment()
        self.assertIn("LDs_4", fragment)
        self.assertIn("LDa_2", fragment)
        self.assertIn("LDr_5", fragment)
        training = config.to_training_config(vocab_size=50304, world_size=1, out_dir=torch.serialization.os.PathLike if False else __import__("pathlib").Path("out"))
        self.assertEqual(training.layer_dropout_stratum_size, 4)
        self.assertEqual(training.layer_dropout_active_per_stratum, 2)
        self.assertEqual(training.layer_dropout_resample_steps, 5)
        self.assertEqual(training.n_active_layers, 72)


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
