# vvv THOG
from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch

from sheet.basis import build_stabilized_basis, normalized_coordinates, stabilized_chebyshev_basis_at_coordinates
from sheet.checkpoints import load_payload, optimizer_group_names, save_payload
from sheet.depth_trajectory import DepthTrajectory
from sheet.geometry import SheetGeometryConfig
from sheet.model import SheetGPT, SheetGPTConfig
from sheet.plastic_depth import (
    PlasticDepthCandidateMeasurement,
    PlasticDepthSamplingLattice,
    choose_plastic_depth_candidate,
    evenly_distributed_active_ranks,
    public_to_internal_depth,
    resolve_plastic_depth_counts,
)
from sheet.trainer import SharedTrainer
from sheet.training_model import TrainingSheetGPT
from tests.stage3_test_support import assert_nested_equal, stage3_config, token_splits


def plastic_sheet_config(**overrides):
    values = dict(
        block_size=8,
        vocab_size=32,
        n_layer=4,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=True,
        depth_order=3,
        geometry_preset="depth",
        basis_family="chebyshev",
        plastic__enabled=True,
        plastic__layers_to_sample=4,
        plastic__freeze_geometry_during_warmup=False,
    )
    values.update(overrides)
    return SheetGPTConfig(**values)


def plastic_training_config(**overrides):
    values = dict(
        geometry_preset="depth",
        basis_family="chebyshev",
        plastic__enabled=True,
        plastic__layers_to_sample=4,
        plastic__freeze_geometry_during_warmup=False,
        depth_order=3,
        n_layer=4,
        max_updates=6,
    )
    values.update(overrides)
    return stage3_config("thog2_sheet", **values)


class PlasticDepthPrimitiveTests(unittest.TestCase):
    def test_count_resolution_fixed_and_learned(self) -> None:
        fixed = resolve_plastic_depth_counts(
            n_layer=12,
            enabled=True,
            layers_to_sample=7,
            do_learn_layer_count=False,
            initial_layer_count=None,
            max_permitted_layers=None,
        )
        self.assertEqual((fixed.maximum_layers, fixed.initial_active_layers, fixed.fixed_active_layers), (7, 7, 7))
        learned = resolve_plastic_depth_counts(
            n_layer=12,
            enabled=True,
            layers_to_sample=None,
            do_learn_layer_count=True,
            initial_layer_count=5,
            max_permitted_layers=9,
        )
        self.assertEqual((learned.maximum_layers, learned.initial_active_layers, learned.fixed_active_layers), (9, 5, None))
        with self.assertRaisesRegex(ValueError, "may not be supplied"):
            resolve_plastic_depth_counts(
                n_layer=12,
                enabled=True,
                layers_to_sample=5,
                do_learn_layer_count=True,
                initial_layer_count=5,
                max_permitted_layers=9,
            )

    def test_public_ruler_and_equidistant_initialisation(self) -> None:
        lattice = PlasticDepthSamplingLattice(
            4,
            initial_active_layers=4,
            initialisation="equidistant",
            seed=11,
        )
        torch.testing.assert_close(
            lattice.public_coordinates(),
            torch.tensor([1.0, 34.0, 67.0, 100.0]),
            rtol=0.0,
            atol=2.0e-5,
        )
        torch.testing.assert_close(
            public_to_internal_depth(lattice.public_coordinates()),
            torch.linspace(-1.0, 1.0, 4),
            rtol=0.0,
            atol=2.0e-6,
        )
        single = PlasticDepthSamplingLattice(
            1,
            initial_active_layers=1,
            initialisation="equidistant",
            seed=11,
        )
        self.assertEqual(single.public_coordinates().tolist(), [50.5])

    def test_fixed_count_lattice_has_no_controller_state(self) -> None:
        lattice = PlasticDepthSamplingLattice(
            4,
            initial_active_layers=4,
            initialisation="equidistant",
            seed=11,
            learn_layer_count=False,
        )
        self.assertEqual(lattice.current_active_layers, 4)
        self.assertEqual(
            tuple(lattice.state_dict()),
            ("raw_intervals", "initial_public_coordinates"),
        )
        for name in (
            "active_layer_count",
            "last_count_decision_update",
            "count_decision_number",
            "reference_training_time",
            "training_time_ema",
            "training_time_observations",
            "optimizer_step_time_ema",
            "optimizer_step_time_observations",
            "peak_allocated_gib",
            "peak_reserved_gib",
        ):
            self.assertFalse(hasattr(lattice, name), name)
        with self.assertRaisesRegex(RuntimeError, "no layer-count controller"):
            lattice.set_active_layer_count(3)

    def test_random_initialisation_is_ordered_reproducible_and_seeded(self) -> None:
        first = PlasticDepthSamplingLattice(8, initial_active_layers=5, initialisation="random", seed=7)
        repeated = PlasticDepthSamplingLattice(8, initial_active_layers=5, initialisation="random", seed=7)
        different = PlasticDepthSamplingLattice(8, initial_active_layers=5, initialisation="random", seed=8)
        coordinates = first.public_coordinates().detach()
        self.assertAlmostEqual(float(coordinates[0]), 1.0, places=5)
        self.assertAlmostEqual(float(coordinates[-1]), 100.0, places=4)
        self.assertTrue(bool(torch.all(coordinates[1:] > coordinates[:-1]).item()))
        torch.testing.assert_close(coordinates, repeated.public_coordinates(), rtol=0.0, atol=0.0)
        self.assertFalse(torch.equal(coordinates, different.public_coordinates()))

    def test_active_ranks_are_even_and_include_endpoints(self) -> None:
        self.assertEqual(evenly_distributed_active_ranks(8, 1), (0,))
        self.assertEqual(evenly_distributed_active_ranks(8, 2), (0, 7))
        self.assertEqual(evenly_distributed_active_ranks(8, 4), (0, 2, 5, 7))
        self.assertEqual(evenly_distributed_active_ranks(8, 8), tuple(range(8)))

    def test_learned_count_owns_fixed_capacity_absolute_ruler(self) -> None:
        lattice = PlasticDepthSamplingLattice(
            8,
            initial_active_layers=4,
            initialisation="equidistant",
            seed=11,
            learn_layer_count=True,
        )

        torch.testing.assert_close(
            lattice.public_coordinates(),
            torch.linspace(1.0, 100.0, 8),
            rtol=0.0,
            atol=2.0e-5,
        )
        self.assertEqual(lattice.active_ranks(), (0, 1, 2, 3))
        torch.testing.assert_close(
            lattice.active_public_coordinates(),
            lattice.public_coordinates()[:4],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            lattice.probe_public_coordinate(),
            lattice.public_coordinates()[4:5],
            rtol=0.0,
            atol=0.0,
        )
        report = lattice.interval_report()
        self.assertEqual(report["active_sample_layer_coordinates"], (1.0, 2.0, 3.0, 4.0))
        self.assertEqual(report["sample_layer_coordinates"], (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0))
        self.assertEqual(report["probe_sample_layer_coordinate"], 5.0)

    def test_future_capacity_is_part_of_absolute_ruler_and_receives_gradient(self) -> None:
        lattice = PlasticDepthSamplingLattice(
            8,
            initial_active_layers=4,
            initialisation="random",
            seed=17,
            learn_layer_count=True,
        )
        before = lattice.public_coordinates().detach().clone()
        with torch.no_grad():
            lattice.raw_intervals[4:].copy_(torch.linspace(-100.0, 100.0, 3))
        after = lattice.public_coordinates()

        self.assertGreater(float((after - before).abs().max().item()), 0.0)
        after.square().sum().backward()
        self.assertIsNotNone(lattice.raw_intervals.grad)
        self.assertGreater(float(lattice.raw_intervals.grad[4:].abs().sum().item()), 0.0)

    def test_arbitrary_chebyshev_sampling_preserves_reference_basis(self) -> None:
        coordinates = normalized_coordinates(6, dtype=torch.float64)
        expected = build_stabilized_basis(6, 4, runtime_dtype=torch.float64)
        actual = stabilized_chebyshev_basis_at_coordinates(
            coordinates,
            reference_sample_count=6,
            order=4,
            runtime_dtype=torch.float64,
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=2.0e-15)


class PlasticDepthModelTests(unittest.TestCase):
    def test_equidistant_plastic_model_matches_existing_depth_model(self) -> None:
        torch.manual_seed(1234)
        baseline = SheetGPT(
            SheetGPTConfig(
                block_size=8,
                vocab_size=32,
                n_layer=4,
                n_head=2,
                n_embd=16,
                dropout=0.0,
                bias=True,
                depth_order=3,
                geometry_preset="depth",
                basis_family="chebyshev",
            )
        )
        torch.manual_seed(1234)
        plastic = SheetGPT(plastic_sheet_config())
        baseline_state = baseline.state_dict()
        plastic_state = plastic.state_dict()
        for name, value in baseline_state.items():
            plastic_state[name].copy_(value)
        indices = torch.arange(8, dtype=torch.long).view(1, 8) % 32
        baseline.eval()
        plastic.eval()
        with torch.no_grad():
            baseline_logits, _ = baseline(indices)
            plastic_logits, _ = plastic(indices)
        torch.testing.assert_close(plastic_logits, baseline_logits, rtol=2.0e-5, atol=2.0e-6)

    def test_fractional_sampling_receives_gradient_after_higher_modes_exist(self) -> None:
        geometry = SheetGeometryConfig(
            n_layer=4,
            n_embd=8,
            n_head=2,
            depth_order=3,
            base_row_order=1,
            mlp_channel_order=1,
            o_attn_d_model=1,
            o_attn_qkv_per_channel=1,
            o_attn_out_per_channel=1,
            o_mlp_d_model=1,
            o_mlp_hidden=1,
            bias=True,
        )
        trajectory = DepthTrajectory(
            geometry,
            runtime_dtype=torch.float32,
            basis_family="chebyshev",
            depth_compress_layer_norm_and_bias=False,
            plastic_enabled=True,
            plastic_initial_active_layers=4,
            plastic_sampling_initialisation="equidistant",
            plastic_seed=3,
        )
        with torch.no_grad():
            trajectory.coefficients["attention_query_weight"][:, :, 1].fill_(0.25)
            trajectory.coefficients["attention_query_weight"][:, :, 2].fill_(-0.10)
        value = trajectory.materialize("attention_query_weight", 1).square().mean()
        value.backward()
        gradient = trajectory.plastic_sampling.raw_intervals.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum().item()), 0.0)

    def test_optimizer_has_separate_no_decay_geometry_group(self) -> None:
        model = SheetGPT(
            plastic_sheet_config(
                plastic__geometry_learning_rate_multiplier=0.125,
                plastic__freeze_geometry_during_warmup=True,
            )
        )
        groups = model.optimizer_parameter_groups(weight_decay=0.1)
        self.assertEqual(len(groups), 3)
        geometry = groups[2]
        self.assertEqual(geometry["weight_decay"], 0.0)
        self.assertEqual(geometry["thog2_lr_multiplier"], 0.125)
        self.assertTrue(geometry["thog2_freeze_during_warmup"])
        self.assertEqual(geometry["parameter_names"], ("trajectory.plastic_sampling.raw_intervals",))

    def test_future_capacity_changes_absolute_coordinates_even_if_output_cache_is_identical(self) -> None:
        torch.manual_seed(1357)
        model = SheetGPT(
            plastic_sheet_config(
                n_layer=8,
                depth_order=4,
                plastic__layers_to_sample=None,
                plastic__do_learn_layer_count=True,
                plastic__initial_layer_count=4,
                plastic__max_permitted_layers=8,
            )
        )
        model.eval()
        indices = torch.arange(8, dtype=torch.long).view(1, 8) % 32
        with torch.no_grad():
            before_coordinates = model.trajectory.plastic_sampling.active_public_coordinates().detach().clone()
            before, _ = model(indices)
            model.trajectory.plastic_sampling.raw_intervals[4:].copy_(
                torch.linspace(-100.0, 100.0, 3)
            )
            after_coordinates = model.trajectory.plastic_sampling.active_public_coordinates().detach().clone()
            after, _ = model(indices)
        self.assertGreater(float((after_coordinates - before_coordinates).abs().max().item()), 0.0)
        torch.testing.assert_close(after, before, rtol=0.0, atol=0.0)

    def test_atomic_add_and_subtract_regauge_preserve_all_generated_families(self) -> None:
        torch.manual_seed(4102)
        model = SheetGPT(
            plastic_sheet_config(
                n_layer=5,
                depth_order=4,
                plastic__layers_to_sample=None,
                plastic__do_learn_layer_count=True,
                plastic__initial_layer_count=3,
                plastic__max_permitted_layers=5,
                plastic__layer_sampling_initialisation="random",
            )
        )
        trajectory = model.trajectory
        with torch.no_grad():
            for item in trajectory.metadata:
                if trajectory._plastic_generated_depth_axis(item) is not None:
                    trajectory.coefficients[item.name].normal_(mean=0.0, std=0.02)

        def generated_snapshot(indices):
            return {
                (item.name, index): trajectory.materialize(item.name, index).detach().clone()
                for item in trajectory.metadata
                if trajectory._plastic_generated_depth_axis(item) is not None
                for index in indices
            }

        before_add = generated_snapshot(range(4))
        coefficient_before = {
            name: parameter.detach().clone()
            for name, parameter in trajectory.coefficients.items()
        }
        add = model.prepare_plastic_depth_count_transition(4)
        self.assertEqual(trajectory.plastic_sampling.current_active_layers, 3)
        for name, expected in coefficient_before.items():
            torch.testing.assert_close(trajectory.coefficients[name], expected, rtol=0.0, atol=0.0)
        add_report = model.commit_plastic_depth_count_transition(add)
        self.assertEqual(trajectory.plastic_sampling.current_active_layers, 4)
        self.assertGreater(add_report["transformed_family_count"], 0)
        self.assertLessEqual(
            add_report["maximum_absolute_error"],
            add_report["absolute_tolerance"] + 1.0e-12,
        )
        after_add = generated_snapshot(range(4))
        for key, expected in before_add.items():
            torch.testing.assert_close(after_add[key], expected, rtol=3.0e-5, atol=3.0e-6)

        before_subtract = generated_snapshot(range(4))
        subtract = model.prepare_plastic_depth_count_transition(3)
        subtract_report = model.commit_plastic_depth_count_transition(subtract)
        self.assertEqual(trajectory.plastic_sampling.current_active_layers, 3)
        self.assertLessEqual(
            subtract_report["maximum_absolute_error"],
            subtract_report["absolute_tolerance"] + 1.0e-12,
        )
        after_subtract = generated_snapshot(range(4))
        for key, expected in before_subtract.items():
            torch.testing.assert_close(after_subtract[key], expected, rtol=3.0e-5, atol=3.0e-6)

    def test_atomic_regauge_rejects_stale_coefficients_without_changing_count(self) -> None:
        model = SheetGPT(
            plastic_sheet_config(
                n_layer=5,
                depth_order=4,
                plastic__layers_to_sample=None,
                plastic__do_learn_layer_count=True,
                plastic__initial_layer_count=3,
                plastic__max_permitted_layers=5,
            )
        )
        transition = model.prepare_plastic_depth_count_transition(4)
        first_name = transition.replacements[0].name
        with torch.no_grad():
            model.trajectory.coefficients[first_name].add_(1.0e-4)
        with self.assertRaisesRegex(RuntimeError, "coefficient changed"):
            model.commit_plastic_depth_count_transition(transition)
        self.assertEqual(model.trajectory.plastic_sampling.current_active_layers, 3)

    def test_full_active_count_uses_full_execution_and_lower_count_uses_subset(self) -> None:
        model = SheetGPT(
            plastic_sheet_config(
                plastic__layers_to_sample=None,
                plastic__do_learn_layer_count=True,
                plastic__initial_layer_count=4,
                plastic__max_permitted_layers=4,
            )
        )
        self.assertEqual(model.plastic_depth_active_layer_indices(), (0, 1, 2, 3))
        model.set_plastic_depth_active_layer_count(2)
        self.assertEqual(model.plastic_depth_active_layer_indices(), (0, 1))

    def test_geometry_learning_rate_multiplier_and_warmup_freeze(self) -> None:
        train_tokens, validation_tokens = token_splits()
        trainer = SharedTrainer(
            plastic_training_config(
                warmup_updates=2,
                plastic__geometry_learning_rate_multiplier=0.25,
                plastic__freeze_geometry_during_warmup=True,
            ),
            train_tokens,
            validation_tokens,
        )
        try:
            base_learning_rate = trainer._set_learning_rate()
            self.assertEqual(trainer.optimizer.param_groups[2]["lr"], 0.0)
            trainer.state.completed_updates = 2
            base_learning_rate = trainer._set_learning_rate()
            self.assertEqual(
                trainer.optimizer.param_groups[2]["lr"],
                base_learning_rate * 0.25,
            )
        finally:
            trainer.close()

    def test_fast_discard_false_cpu_update_uses_plastic_active_ranks(self) -> None:
        train_tokens, validation_tokens = token_splits()
        with patch.dict(os.environ, {"THOG2_FAST_DISCARD": "false"}):
            trainer = SharedTrainer(
                plastic_training_config(plastic__layers_to_sample=3, n_layer=3, depth_order=3),
                train_tokens,
                validation_tokens,
            )
        try:
            metrics = trainer.train_one_update()
            self.assertEqual(metrics["plastic_active_layers"], 3.0)
            self.assertIn("plastic_coefficient_gradient_norm", metrics)
            self.assertIn("plastic_geometry_gradient_norm", metrics)
            self.assertEqual(len(metrics["plastic_full_public_coordinates"]), 3)
            report = trainer.raw_model.update_retained_materialization_report()
            self.assertTrue(report["enabled"])
            self.assertFalse(report["active"])
        finally:
            trainer.close()

    def test_single_sample_has_no_spurious_geometry_optimizer_group(self) -> None:
        model = SheetGPT(
            plastic_sheet_config(
                n_layer=1,
                depth_order=1,
                plastic__layers_to_sample=1,
            )
        )
        self.assertEqual(model.plastic_depth_active_layer_indices(), (0,))
        self.assertEqual(len(model.optimizer_parameter_groups(weight_decay=0.1)), 2)
        indices = torch.arange(8, dtype=torch.long).view(1, 8) % 32
        logits, loss = model(indices, indices)
        self.assertEqual(tuple(logits.shape), (1, 8, 32))
        self.assertIsNotNone(loss)

    def test_training_moves_geometry_and_emits_required_diagnostics(self) -> None:
        train_tokens, validation_tokens = token_splits()
        trainer = SharedTrainer(
            plastic_training_config(max_updates=3),
            train_tokens,
            validation_tokens,
        )
        try:
            first = trainer.train_one_update()
            second = trainer.train_one_update()
            self.assertEqual(first["plastic_mean_absolute_movement"], 0.0)
            self.assertGreater(second["plastic_geometry_gradient_norm"], 0.0)
            self.assertGreater(second["plastic_mean_absolute_movement"], 0.0)
            self.assertGreater(second["plastic_maximum_interval"], second["plastic_minimum_interval"])
            self.assertGreater(second["plastic_training_only_seconds"], 0.0)
            self.assertGreater(second["plastic_optimizer_step_seconds"], 0.0)
        finally:
            trainer.close()

    def test_regional_compile_rejects_sparse_plastic_count(self) -> None:
        model = TrainingSheetGPT(
            plastic_sheet_config(
                plastic__layers_to_sample=None,
                plastic__do_learn_layer_count=True,
                plastic__initial_layer_count=2,
                plastic__max_permitted_layers=4,
            )
        )
        model.set_checkpoint_segment_size(2)
        model.set_torch_compile_mode("regional")
        indices = torch.arange(8, dtype=torch.long).view(1, 8) % 32
        with self.assertRaisesRegex(RuntimeError, "PLASTIC DEPTH active count"):
            model(indices, indices)


class PlasticDepthControllerTests(unittest.TestCase):
    def test_fixed_count_checkpoint_contains_geometry_without_controller_buffers(self) -> None:
        train_tokens, validation_tokens = token_splits()
        trainer = SharedTrainer(plastic_training_config(), train_tokens, validation_tokens)
        try:
            payload = trainer.checkpoint_payload()
            keys = tuple(
                name
                for name in payload["model"]
                if name.startswith("trajectory.plastic_sampling.")
            )
            self.assertEqual(
                keys,
                (
                    "trajectory.plastic_sampling.raw_intervals",
                    "trajectory.plastic_sampling.initial_public_coordinates",
                ),
            )
        finally:
            trainer.close()

    def test_obsolete_external_controller_does_not_probe_or_consume_rng(self) -> None:
        train_tokens, validation_tokens = token_splits()
        trainer = SharedTrainer(
            plastic_training_config(
                plastic__layers_to_sample=None,
                plastic__do_learn_layer_count=True,
                plastic__initial_layer_count=2,
                plastic__max_permitted_layers=4,
                plastic__layer_count_update_brake=0,
                plastic__layer_count_objective="relative_training_wall_time",
                plastic__layer_count_cost_weight=0.1,
                eval_batches=1,
            ),
            train_tokens,
            validation_tokens,
        )
        try:
            trainer.state.completed_updates = 1
            lattice = trainer.raw_model.trajectory.plastic_sampling
            lattice.reference_training_time.fill_(1.0)
            random_state = torch.get_rng_state().clone()
            draws = []

            def timing_probe(active_layers, batches):
                draws.append(float(torch.rand(()).item()))
                return float(active_layers)

            with patch.object(trainer, "_plastic_depth_candidate_loss", return_value=1.0), patch.object(
                trainer,
                "_plastic_depth_candidate_training_time",
                side_effect=timing_probe,
            ):
                trainer._prepare_plastic_depth_for_update()
            self.assertEqual(draws, [])
            self.assertEqual(int(lattice.active_layer_count.item()), 2)
            self.assertEqual(int(lattice.count_decision_number.item()), 0)
            torch.testing.assert_close(torch.get_rng_state(), random_state, rtol=0.0, atol=0.0)
        finally:
            trainer.close()

    def test_objectives_and_lower_count_tie_break(self) -> None:
        measurements = (
            PlasticDepthCandidateMeasurement(3, 2.0, training_time=3.0, peak_allocated_gib=3.0),
            PlasticDepthCandidateMeasurement(4, 1.9, training_time=5.0, peak_allocated_gib=5.0),
        )
        selected, _ = choose_plastic_depth_candidate(
            measurements,
            objective="lowest_loss",
            maximum_layers=4,
            cost_weight=0.0,
            reference_training_time=3.0,
            memory_budget_gib=None,
        )
        self.assertEqual(selected.active_layers, 4)
        selected, _ = choose_plastic_depth_candidate(
            measurements,
            objective="layer_efficiency",
            maximum_layers=4,
            cost_weight=1.0,
            reference_training_time=3.0,
            memory_budget_gib=None,
        )
        self.assertEqual(selected.active_layers, 3)
        selected, _ = choose_plastic_depth_candidate(
            measurements,
            objective="relative_training_wall_time",
            maximum_layers=4,
            cost_weight=1.0,
            reference_training_time=3.0,
            memory_budget_gib=None,
        )
        self.assertEqual(selected.active_layers, 3)
        selected, _ = choose_plastic_depth_candidate(
            measurements,
            objective="memory_budget",
            maximum_layers=4,
            cost_weight=0.0,
            reference_training_time=None,
            memory_budget_gib=4.0,
        )
        self.assertEqual(selected.active_layers, 3)
        tied, _ = choose_plastic_depth_candidate(
            (
                PlasticDepthCandidateMeasurement(2, 1.0),
                PlasticDepthCandidateMeasurement(3, 1.0),
            ),
            objective="lowest_loss",
            maximum_layers=3,
            cost_weight=0.0,
            reference_training_time=None,
            memory_budget_gib=None,
        )
        self.assertEqual(tied.active_layers, 2)

    def test_learned_count_changes_without_rebuilding_optimizer(self) -> None:
        train_tokens, validation_tokens = token_splits()
        trainer = SharedTrainer(
            plastic_training_config(
                plastic__layers_to_sample=None,
                plastic__do_learn_layer_count=True,
                plastic__initial_layer_count=2,
                plastic__max_permitted_layers=4,
                plastic__layer_count_update_brake=0,
                eval_batches=1,
            ),
            train_tokens,
            validation_tokens,
        )
        try:
            optimizer_id = id(trainer.optimizer)
            group_names = optimizer_group_names(trainer.optimizer)
            trainer.train_one_update()
            trainer.train_one_update()
            self.assertEqual(id(trainer.optimizer), optimizer_id)
            self.assertEqual(optimizer_group_names(trainer.optimizer), group_names)
            lattice = trainer.raw_model.trajectory.plastic_sampling
            self.assertGreaterEqual(int(lattice.count_decision_number.item()), 1)
            self.assertGreaterEqual(int(lattice.active_layer_count.item()), 1)
            self.assertLessEqual(int(lattice.active_layer_count.item()), 4)
        finally:
            trainer.close()

    def test_pre_plastic_schema2_checkpoint_remains_resumable(self) -> None:
        train_tokens, validation_tokens = token_splits()
        plastic_fields = (
            "plastic__enabled",
            "plastic__layers_to_sample",
            "plastic__do_learn_layer_count",
            "plastic__initial_layer_count",
            "plastic__max_permitted_layers",
            "plastic__layer_sampling_initialisation",
            "plastic__layer_count_objective",
            "plastic__layer_count_update_brake",
            "plastic__layer_count_probe__window_size_as_number_of_probes",
            "plastic__layer_count_min_probes",
            "plastic__layer_count_probe_noise_lambda",
            "plastic__layer_count_cost_weight",
            "plastic__layer_memory_budget_gib",
            "plastic__geometry_learning_rate_multiplier",
            "plastic__freeze_geometry_during_warmup",
            "plastic__initial_active_layers",
        )
        with tempfile.TemporaryDirectory() as directory:
            trainer = SharedTrainer(stage3_config("thog2_sheet"), train_tokens, validation_tokens)
            try:
                trainer.run(target_updates=1)
                source_path = trainer.save_checkpoint(Path(directory) / "source.pt")
            finally:
                trainer.close()
            payload = load_payload(source_path)
            self.assertFalse(any(name.startswith("plastic__") for name in payload["trainer_config"]))
            self.assertFalse(any(name.startswith("plastic__") for name in payload["model_args"]))
            for name in plastic_fields:
                payload["trainer_config"].pop(name, None)
                payload["compatibility_signature"].pop(name, None)
                payload["model_args"].pop(name, None)
            legacy_path = save_payload(payload, Path(directory) / "pre_plastic_schema2.pt")
            resumed = SharedTrainer.from_checkpoint(legacy_path, train_tokens, validation_tokens)
            try:
                self.assertFalse(resumed.config.plastic__enabled)
                self.assertEqual(resumed.state.completed_updates, 1)
                resumed.train_one_update()
                self.assertEqual(resumed.state.completed_updates, 2)
            finally:
                resumed.close()

    def test_inline_count_controller_can_move_down_and_up(self) -> None:
        train_tokens, validation_tokens = token_splits()
        for expected_count in (1, 3):
            trainer = SharedTrainer(
                plastic_training_config(
                    plastic__layers_to_sample=None,
                    plastic__do_learn_layer_count=True,
                    plastic__initial_layer_count=2,
                    plastic__max_permitted_layers=4,
                    plastic__layer_count_update_brake=0,
                    plastic__layer_count_probe__window_size_as_number_of_probes=8,
                    plastic__layer_count_min_probes=1,
                    plastic__layer_count_probe_noise_lambda=0.0,
                    gradient_accumulation_steps=1,
                    eval_batches=1,
                ),
                train_tokens,
                validation_tokens,
            )

            def choose(measurements, **_kwargs):
                measurements = tuple(measurements)
                selected = next(
                    measurement
                    for measurement in measurements
                    if measurement.active_layers == expected_count
                )
                report = tuple(
                    {
                        "active_layers": measurement.active_layers,
                        "validation_loss": measurement.validation_loss,
                        "feasible": True,
                        "score": 0.0 if measurement.active_layers == expected_count else 1.0,
                    }
                    for measurement in measurements
                )
                return selected, report

            try:
                with patch(
                    "sheet.trainer_step.choose_plastic_depth_candidate",
                    side_effect=choose,
                ):
                    trainer.train_one_update()
                lattice = trainer.raw_model.trajectory.plastic_sampling
                self.assertEqual(int(lattice.active_layer_count.item()), expected_count)
                self.assertEqual(int(lattice.count_decision_number.item()), 1)
            finally:
                trainer.close()

    def test_checkpoint_resume_preserves_geometry_controller_and_optimizer(self) -> None:
        train_tokens, validation_tokens = token_splits()
        with tempfile.TemporaryDirectory() as directory:
            trainer = SharedTrainer(
                plastic_training_config(
                    plastic__layers_to_sample=None,
                    plastic__do_learn_layer_count=True,
                    plastic__initial_layer_count=2,
                    plastic__max_permitted_layers=4,
                    plastic__layer_count_update_brake=0,
                    eval_batches=1,
                ),
                train_tokens,
                validation_tokens,
            )
            trainer.run(target_updates=2)
            path = trainer.save_checkpoint(Path(directory) / "ckpt.pt")
            resumed = SharedTrainer.from_checkpoint(path, train_tokens, validation_tokens)
            try:
                self.assertEqual(
                    resumed.state.plastic_depth_probe_histories,
                    trainer.state.plastic_depth_probe_histories,
                )
                self.assertEqual(
                    resumed.state.plastic_depth_last_count_change_update,
                    trainer.state.plastic_depth_last_count_change_update,
                )
                for name, value in trainer.raw_model.state_dict().items():
                    torch.testing.assert_close(
                        value,
                        resumed.raw_model.state_dict()[name],
                        rtol=0.0,
                        atol=0.0,
                        equal_nan=True,
                    )
                assert_nested_equal(self, trainer.optimizer.state_dict(), resumed.optimizer.state_dict())
                trainer.train_one_update()
                resumed.train_one_update()
                for name, value in trainer.raw_model.state_dict().items():
                    if "training_time" in name or "optimizer_step_time" in name:
                        continue
                    torch.testing.assert_close(
                        value,
                        resumed.raw_model.state_dict()[name],
                        rtol=0.0,
                        atol=0.0,
                        equal_nan=True,
                    )
            finally:
                trainer.close()
                resumed.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
