# vvv THOG
from __future__ import annotations

import math
import os
import unittest
from pathlib import Path

import torch

from model import GPT, GPTConfig
from sheet.approximation import (
    fit_sampled_sheets,
    is_within_epsilon,
    project_sampled_sheets,
    projection_error,
    reconstruct_sampled_sheets,
)
from sheet.basis import build_stabilized_basis
from sheet.geometry import SheetGeometryConfig
from sheet.model import SheetGPT, SheetGPTConfig
from sheet.trajectory import SheetTrajectory


class Stage2ReferenceModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    @staticmethod
    def tiny_config(*, bias: bool = True, dropout: float = 0.0) -> SheetGPTConfig:
        return SheetGPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=3,
            n_head=4,
            n_embd=16,
            dropout=dropout,
            bias=bias,
            depth_order=3,
            base_row_order=8,
        )

    def test_s2_01_coefficient_tensor_shapes(self) -> None:
        config = self.tiny_config()
        trajectory = SheetTrajectory(config.sheet_geometry())
        for item in trajectory.metadata:
            with self.subTest(family=item.name):
                self.assertEqual(
                    tuple(trajectory.coefficients[item.name].shape),
                    item.coefficient_shape(config.depth_order),
                )
        explicit = sum(parameter.numel() for parameter in trajectory.coefficients.values())
        self.assertEqual(explicit, trajectory.sheet_parameter_count())

    def test_s2_02_one_layer_materialisation(self) -> None:
        torch.manual_seed(11)
        config = self.tiny_config()
        trajectory = SheetTrajectory(config.sheet_geometry(), runtime_dtype=torch.float64)
        with torch.no_grad():
            for parameter in trajectory.coefficients.values():
                parameter.normal_(mean=0.0, std=0.1)

        for item in trajectory.metadata:
            for layer_index in (0, config.n_layer - 1):
                with self.subTest(family=item.name, layer=layer_index):
                    generated = trajectory.materialize(item.name, layer_index)
                    coefficient = trajectory.coefficients[item.name]
                    depth_row = trajectory.depth_basis[layer_index]
                    row_basis = trajectory.row_basis(item.name)
                    direct_rows = torch.stack(
                        [
                            depth_row @ coefficient[row_index] @ row_basis.transpose(0, 1)
                            for row_index in range(item.geometry.output_rows)
                        ]
                    )
                    torch.testing.assert_close(generated, direct_rows, rtol=0.0, atol=2.0e-12)

    def test_s2_03_direct_point_evaluation(self) -> None:
        torch.manual_seed(12)
        trajectory = SheetTrajectory(
            self.tiny_config().sheet_geometry(),
            runtime_dtype=torch.float64,
        )
        with torch.no_grad():
            trajectory.coefficients["attention_output_weight"].normal_(0.0, 0.2)
        generated = trajectory.materialize("attention_output_weight", 1)
        for output_row, row_index in ((0, 0), (3, 7), (15, 15)):
            direct = trajectory.direct_value(
                "attention_output_weight",
                1,
                output_row,
                row_index,
            )
            torch.testing.assert_close(
                direct,
                generated[output_row, row_index],
                rtol=0.0,
                atol=2.0e-12,
            )

    def test_s2_04_saturated_sampled_completeness(self) -> None:
        torch.manual_seed(13)
        layers, output_rows, row_width = 4, 3, 5
        sampled = torch.randn(layers, output_rows, row_width, dtype=torch.float64)
        depth_basis = build_stabilized_basis(layers, layers, runtime_dtype=torch.float64)
        row_basis = build_stabilized_basis(row_width, row_width, runtime_dtype=torch.float64)
        coefficients = fit_sampled_sheets(sampled, depth_basis, row_basis)
        reconstructed = reconstruct_sampled_sheets(coefficients, depth_basis, row_basis)
        error = projection_error(sampled, reconstructed)
        self.assertLess(error.max_abs, 2.0e-12)
        self.assertTrue(is_within_epsilon(sampled, reconstructed, 2.0e-12))

    def test_s2_05_conventional_generated_shapes(self) -> None:
        config = self.tiny_config()
        trajectory = SheetTrajectory(config.sheet_geometry())
        expected = {
            "attention_input_weight": (48, 16),
            "attention_output_weight": (16, 16),
            "mlp_expansion_weight": (64, 16),
            "mlp_contraction_weight": (16, 64),
            "ln_1_weight": (16,),
            "ln_2_weight": (16,),
            "ln_1_bias": (16,),
            "ln_2_bias": (16,),
            "attention_input_bias": (48,),
            "attention_output_bias": (16,),
            "mlp_expansion_bias": (64,),
            "mlp_contraction_bias": (16,),
        }
        for name, shape in expected.items():
            with self.subTest(family=name):
                if len(shape) == 1:
                    generated = trajectory.materialize_vector(name, 0)
                else:
                    generated = trajectory.materialize(name, 0)
                self.assertEqual(tuple(generated.shape), shape)

    def test_s2_06_family_isolation(self) -> None:
        trajectory = SheetTrajectory(self.tiny_config().sheet_geometry())
        layer_index = 1
        before = {
            item.name: trajectory.materialize(item.name, layer_index).detach().clone()
            for item in trajectory.metadata
        }
        with torch.no_grad():
            trajectory.coefficients["attention_input_weight"][0, 0, 0].add_(1.0)
        after = {
            item.name: trajectory.materialize(item.name, layer_index).detach().clone()
            for item in trajectory.metadata
        }
        self.assertFalse(
            torch.equal(before["attention_input_weight"], after["attention_input_weight"])
        )
        for name in before:
            if name != "attention_input_weight":
                self.assertTrue(torch.equal(before[name], after[name]), name)

    def test_s2_07_gradient_reachability(self) -> None:
        torch.manual_seed(14)
        model = SheetGPT(self.tiny_config())
        idx = torch.randint(0, model.config.vocab_size, (2, 8))
        targets = torch.randint(0, model.config.vocab_size, (2, 8))
        _, loss = model(idx, targets)
        self.assertIsNotNone(loss)
        loss.backward()
        for name, parameter in model.trajectory.coefficients.items():
            with self.subTest(family=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(int(torch.count_nonzero(parameter.grad)), 0)

    def test_s2_08_coefficient_gradient_reference(self) -> None:
        torch.manual_seed(15)
        trajectory = SheetTrajectory(
            self.tiny_config().sheet_geometry(),
            runtime_dtype=torch.float64,
        )
        family = "mlp_contraction_weight"
        coefficient = trajectory.coefficients[family]
        with torch.no_grad():
            coefficient.normal_(0.0, 0.1)
        probe = torch.randn(16, 64, dtype=torch.float64)

        generated = trajectory.materialize(family, 2)
        loss = torch.sum(generated * probe)
        loss.backward()
        implementation_gradient = coefficient.grad.detach().clone()

        reference_coefficient = coefficient.detach().clone().requires_grad_(True)
        depth_row = trajectory.depth_basis[2]
        row_basis = trajectory.row_basis(family)
        direct = torch.einsum(
            "p,rpq,cq->rc",
            depth_row,
            reference_coefficient,
            row_basis,
        )
        reference_loss = torch.sum(direct * probe)
        reference_loss.backward()
        torch.testing.assert_close(
            implementation_gradient,
            reference_coefficient.grad,
            rtol=0.0,
            atol=2.0e-12,
        )

    def test_s2_09_initialization_exact_structure(self) -> None:
        torch.manual_seed(16)
        trajectory = SheetTrajectory(self.tiny_config().sheet_geometry())
        for item in trajectory.metadata:
            coefficient = trajectory.coefficients[item.name]
            self.assertEqual(int(torch.count_nonzero(coefficient[:, 1:, :])), 0)
            if item.semantic_type == "layernorm":
                for layer_index in range(trajectory.config.n_layer):
                    generated = trajectory.materialize_vector(item.name, layer_index)
                    torch.testing.assert_close(
                        generated,
                        torch.ones_like(generated),
                        rtol=0.0,
                        atol=3.0e-6,
                    )
            elif item.semantic_type == "bias":
                self.assertEqual(int(torch.count_nonzero(coefficient)), 0)
                generated = trajectory.materialize_vector(item.name, 0)
                self.assertEqual(int(torch.count_nonzero(generated)), 0)

    def test_s2_10_initialization_statistics(self) -> None:
        torch.manual_seed(17)
        geometry = SheetGeometryConfig(
            n_layer=8,
            n_embd=64,
            n_head=4,
            depth_order=4,
            base_row_order=32,
            bias=True,
        )
        trajectory = SheetTrajectory(geometry)
        for item in trajectory.metadata:
            if item.semantic_type != "matrix":
                continue
            generated = trajectory.materialize(item.name, 0).detach()
            observed_mean = float(generated.mean())
            observed_std = float(generated.std(unbiased=False))
            target = item.target_weight_std
            with self.subTest(family=item.name):
                self.assertLess(abs(observed_mean), max(0.0025, 0.35 * target))
                self.assertLess(abs(observed_std - target) / target, 0.30)
                self.assertLess(float(torch.max(torch.abs(generated))), 8.0 * target)
                endpoint = trajectory.materialize(item.name, geometry.n_layer - 1)
                torch.testing.assert_close(generated, endpoint, rtol=0.0, atol=2.0e-7)

    def test_s2_11_shared_depth_initialization(self) -> None:
        trajectory = SheetTrajectory(self.tiny_config().sheet_geometry())
        for item in trajectory.metadata:
            first = trajectory.materialize(item.name, 0)
            for layer_index in range(1, trajectory.config.n_layer):
                with self.subTest(family=item.name, layer=layer_index):
                    torch.testing.assert_close(
                        first,
                        trajectory.materialize(item.name, layer_index),
                        rtol=0.0,
                        atol=2.0e-7,
                    )

    def test_s2_12_optimizer_coverage(self) -> None:
        model = SheetGPT(self.tiny_config())
        groups = model.optimizer_parameter_groups(weight_decay=0.1)
        grouped = [parameter for group in groups for parameter in group["params"]]
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        self.assertEqual(len(grouped), len({id(parameter) for parameter in grouped}))
        self.assertEqual({id(parameter) for parameter in grouped}, {id(parameter) for parameter in trainable})

    def test_s2_13_weight_decay_classification(self) -> None:
        model = SheetGPT(self.tiny_config())
        decay_group, no_decay_group = model.optimizer_parameter_groups(weight_decay=0.1)
        decay_names = set(decay_group["parameter_names"])
        no_decay_names = set(no_decay_group["parameter_names"])
        for family_name in (
            "attention_input_weight",
            "attention_output_weight",
            "mlp_expansion_weight",
            "mlp_contraction_weight",
        ):
            self.assertIn(f"trajectory.coefficients.{family_name}", decay_names)
        for family_name in (
            "ln_1_weight",
            "ln_2_weight",
            "ln_1_bias",
            "attention_input_bias",
        ):
            self.assertIn(f"trajectory.coefficients.{family_name}", no_decay_names)
        self.assertIn("transformer.wte.weight", no_decay_names)
        self.assertIn("transformer.wpe.weight", no_decay_names)
        self.assertIn("transformer.ln_f.weight", no_decay_names)

    def test_s2_14_compact_state_guard(self) -> None:
        model = SheetGPT(self.tiny_config())
        self.assertEqual(model.compact_state_violations(), ())
        self.assertEqual(model.trajectory.persistent_basis_keys(), ())
        state_keys = tuple(model.state_dict().keys())
        self.assertTrue(any(key.startswith("trajectory.coefficients.") for key in state_keys))
        self.assertFalse(any(key.startswith("trajectory.bases.") for key in state_keys))
        self.assertFalse(any("transformer.h." in key for key in state_keys))

    def test_s2_15_tiny_model_forward(self) -> None:
        torch.manual_seed(18)
        model = SheetGPT(self.tiny_config())
        idx = torch.randint(0, model.config.vocab_size, (2, 8))
        targets = torch.randint(0, model.config.vocab_size, (2, 8))
        logits, loss = model(idx, targets)
        self.assertEqual(tuple(logits.shape), (2, 8, model.config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(loss))
        inference_logits, inference_loss = model(idx)
        self.assertEqual(tuple(inference_logits.shape), (2, 1, model.config.vocab_size))
        self.assertIsNone(inference_loss)

    def test_s2_16_tiny_model_backward_update(self) -> None:
        torch.manual_seed(19)
        model = SheetGPT(self.tiny_config())
        optimizer = model.configure_optimizers(0.1, 3.0e-3, (0.9, 0.95), "cpu")
        idx = torch.randint(0, model.config.vocab_size, (2, 8))
        targets = torch.randint(0, model.config.vocab_size, (2, 8))
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.trajectory.coefficients.items()
        }
        _, loss = model(idx, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        changed = [
            name
            for name, parameter in model.trajectory.coefficients.items()
            if not torch.equal(before[name], parameter.detach())
        ]
        self.assertEqual(set(changed), set(model.trajectory.coefficients.keys()))

    def test_s2_17_short_cpu_learning_smoke(self) -> None:
        torch.manual_seed(20)
        config = SheetGPTConfig(
            block_size=6,
            vocab_size=16,
            n_layer=2,
            n_head=2,
            n_embd=16,
            dropout=0.0,
            bias=True,
            depth_order=2,
            base_row_order=8,
        )
        model = SheetGPT(config)
        optimizer = model.configure_optimizers(0.0, 1.0e-2, (0.9, 0.95), "cpu")
        idx = torch.randint(0, config.vocab_size, (4, config.block_size))
        targets = torch.roll(idx, shifts=-1, dims=1)
        losses = []
        for _ in range(30):
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(idx, targets)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        self.assertLess(min(losses[-5:]), 0.90 * losses[0])

    def test_s2_18_dense_path_regression(self) -> None:
        torch.manual_seed(21)
        config = GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=2,
            n_head=2,
            n_embd=16,
            dropout=0.0,
            bias=True,
        )
        model = GPT(config)
        idx = torch.randint(0, config.vocab_size, (2, config.block_size))
        targets = torch.randint(0, config.vocab_size, (2, config.block_size))
        logits, loss = model(idx, targets)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None))
        self.assertFalse(any(name.startswith("trajectory.") for name, _ in model.named_parameters()))

    def test_s2_19_epsilon_approximation_contract(self) -> None:
        torch.manual_seed(22)
        layers, rows, width = 8, 2, 12
        smooth_depth_basis = build_stabilized_basis(layers, 3, runtime_dtype=torch.float64)
        smooth_row_basis = build_stabilized_basis(width, 4, runtime_dtype=torch.float64)
        source_coefficients = torch.randn(rows, 3, 4, dtype=torch.float64)
        smooth_sheet = reconstruct_sampled_sheets(
            source_coefficients,
            smooth_depth_basis,
            smooth_row_basis,
        )

        larger_depth_basis = build_stabilized_basis(layers, 5, runtime_dtype=torch.float64)
        larger_row_basis = build_stabilized_basis(width, 7, runtime_dtype=torch.float64)
        smooth_projection = project_sampled_sheets(
            smooth_sheet,
            larger_depth_basis,
            larger_row_basis,
        )
        self.assertTrue(
            is_within_epsilon(
                smooth_sheet,
                smooth_projection,
                2.0e-11,
                metric="max_abs",
            )
        )

        arbitrary = torch.randn(layers, rows, width, dtype=torch.float64)
        saturated_depth = build_stabilized_basis(layers, layers, runtime_dtype=torch.float64)
        saturated_row = build_stabilized_basis(width, width, runtime_dtype=torch.float64)
        saturated_projection = project_sampled_sheets(
            arbitrary,
            saturated_depth,
            saturated_row,
        )
        self.assertTrue(is_within_epsilon(arbitrary, saturated_projection, 3.0e-12))

        low_projection = project_sampled_sheets(
            arbitrary,
            build_stabilized_basis(layers, 2, runtime_dtype=torch.float64),
            build_stabilized_basis(width, 3, runtime_dtype=torch.float64),
        )
        medium_projection = project_sampled_sheets(
            arbitrary,
            build_stabilized_basis(layers, 4, runtime_dtype=torch.float64),
            build_stabilized_basis(width, 6, runtime_dtype=torch.float64),
        )
        low_error = projection_error(arbitrary, low_projection).frobenius
        medium_error = projection_error(arbitrary, medium_projection).frobenius
        self.assertLessEqual(medium_error, low_error + 1.0e-11)
        self.assertFalse(is_within_epsilon(arbitrary, low_projection, 1.0e-3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
