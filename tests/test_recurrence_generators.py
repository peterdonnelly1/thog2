# vvv THOG
from __future__ import annotations

import unittest

import torch

from sheet.bases import BASIS_FAMILIES, BASIS_REGISTRY
from sheet.depth_trajectory import DepthTrajectory
from sheet.geometry import SheetGeometryConfig
from sheet.geometry_registry import COMPRESSOR_REGISTRY, format_geometry_registry, resolve_geometry_plan
from sheet.recurrence_generators import BQRG_FAMILY, BQRG_PERSISTENT_WIDTH, BQRG_VERSION, RECURRENCE_GENERATOR_REGISTRY, get_recurrence_generator_definition, materialize_bqrg_at, materialize_bqrg_sequence
from sheet.semantic_materializer import ATTENTION_QUERY_WEIGHT
from sheet.training_config import TrainingConfig
from sheet.training_model_factory import build_training_model


LEGACY_ORDERS = {
    "o_depth": 16,
    "o_attn_d_model": 1,
    "o_attn_qkv_per_channel": 1,
    "o_attn_out_per_channel": 1,
    "o_mlp_d_model": 1,
    "o_mlp_hidden": 1,
}


def tiny_geometry(*, depth_order: int = 16) -> SheetGeometryConfig:
    return SheetGeometryConfig(
        n_layer=16,
        n_embd=8,
        n_head=2,
        depth_order=depth_order,
        base_row_order=1,
        mlp_channel_order=1,
        o_attn_d_model=1,
        o_attn_qkv_per_channel=1,
        o_attn_out_per_channel=1,
        o_mlp_d_model=1,
        o_mlp_hidden=1,
        bias=True,
    )


def tiny_training_config() -> TrainingConfig:
    return TrainingConfig(
        model_type="thog2_sheet",
        block_size=4,
        vocab_size=64,
        n_layer=16,
        n_head=2,
        n_embd=8,
        depth_order=16,
        base_row_order=1,
        mlp_channel_order=1,
        o_attn_d_model=1,
        o_attn_qkv_per_channel=1,
        o_attn_out_per_channel=1,
        o_mlp_d_model=1,
        o_mlp_hidden=1,
        geometry_preset="depth",
        basis_family=BQRG_FAMILY,
        basis_version=BQRG_VERSION,
        checkpoint_segment_size=0,
        batch_size=1,
        gradient_accumulation_steps=1,
        max_updates=2,
        decay_updates=2,
        eval_batches=1,
        log_interval=1,
        device="cpu",
        dtype="float32",
    )


class RecurrenceGeneratorRegistryTests(unittest.TestCase):
    def test_bqrg_is_registered_as_generator_not_fixed_basis(self) -> None:
        self.assertIn(BQRG_FAMILY, RECURRENCE_GENERATOR_REGISTRY)
        self.assertIn(BQRG_FAMILY, COMPRESSOR_REGISTRY)
        self.assertNotIn(BQRG_FAMILY, BASIS_REGISTRY)
        self.assertNotIn(BQRG_FAMILY, BASIS_FAMILIES)
        definition = get_recurrence_generator_definition(BQRG_FAMILY)
        self.assertEqual(definition.version, BQRG_VERSION)
        self.assertEqual(definition.persistent_widths, (BQRG_PERSISTENT_WIDTH,))
        self.assertEqual(definition.supported_targets, ("DEPTH",))

    def test_geometry_help_lists_recurrence_generator_subsection(self) -> None:
        help_text = format_geometry_registry()
        self.assertIn("recurrence generator registry", help_text)
        self.assertIn("bqrg", help_text)
        self.assertIn("bqrg_v1", help_text)
        self.assertIn("DEPTH", help_text)

    def test_bqrg_resolves_for_depth_p16(self) -> None:
        plan = resolve_geometry_plan(
            select_depth=True,
            selected_elements=(),
            option_assignments=("DEPTH.compressor=bqrg",),
            legacy_orders=LEGACY_ORDERS,
        )
        self.assertTrue(plan.materializer.implemented)
        self.assertEqual(plan.depth_compressor, BQRG_FAMILY)
        self.assertEqual(plan.depth_compressor_version, BQRG_VERSION)
        self.assertEqual(plan.depth_order, BQRG_PERSISTENT_WIDTH)
        self.assertEqual(plan.materializer.materialization_version, BQRG_VERSION)

    def test_bqrg_rejects_non_p16_depth(self) -> None:
        for width in (15, 17):
            orders = dict(LEGACY_ORDERS)
            orders["o_depth"] = width
            with self.assertRaisesRegex(ValueError, "persistent width"):
                resolve_geometry_plan(
                    select_depth=True,
                    selected_elements=(),
                    option_assignments=("DEPTH.compressor=bqrg",),
                    legacy_orders=orders,
                )

    def test_bqrg_rejects_generator_options_not_in_schema(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not accept options"):
            resolve_geometry_plan(
                select_depth=True,
                selected_elements=(),
                option_assignments=("DEPTH.compressor=bqrg", "DEPTH.generator_example=1"),
                legacy_orders=LEGACY_ORDERS,
            )

    def test_bqrg_rejects_non_depth_selector(self) -> None:
        plan = resolve_geometry_plan(
            select_depth=False,
            selected_elements=("MLP_UP.MLP_HIDDEN",),
            option_assignments=("MLP_UP.compressor=bqrg",),
            legacy_orders=LEGACY_ORDERS,
        )
        self.assertFalse(plan.materializer.implemented)
        self.assertIn("not valid", plan.materializer.message)
        self.assertIn("DEPTH", plan.materializer.message)


class BqrgMaterialisationTests(unittest.TestCase):
    def test_sequence_matches_indexed_materialisation(self) -> None:
        torch.manual_seed(7)
        parameters = torch.randn(3, BQRG_PERSISTENT_WIDTH, dtype=torch.float64) * 0.1
        sequence = materialize_bqrg_sequence(parameters, 16)
        self.assertEqual(sequence.shape, (3, 16))
        for index in (0, 1, 7, 15):
            self.assertTrue(torch.allclose(sequence[..., index], materialize_bqrg_at(parameters, index), atol=1.0e-12, rtol=1.0e-12))

    # vvv THOG indexed training materialisation is activation-checkpointed; prove that execution optimisation leaves exact BQRG gradients unchanged.
    def test_checkpointed_indexed_materialisation_matches_sequence_gradient(self) -> None:
        torch.manual_seed(9)
        checkpointed_parameters = (torch.randn(4, BQRG_PERSISTENT_WIDTH, dtype=torch.float64) * 0.05).requires_grad_()
        reference_parameters = checkpointed_parameters.detach().clone().requires_grad_()
        checkpointed = materialize_bqrg_at(checkpointed_parameters, 15)
        reference = materialize_bqrg_sequence(reference_parameters, 16)[..., 15]
        self.assertTrue(torch.allclose(checkpointed, reference, atol=1.0e-12, rtol=1.0e-12))
        checkpointed.square().sum().backward()
        reference.square().sum().backward()
        self.assertIsNotNone(checkpointed_parameters.grad)
        self.assertIsNotNone(reference_parameters.grad)
        self.assertTrue(torch.allclose(checkpointed_parameters.grad, reference_parameters.grad, atol=1.0e-10, rtol=1.0e-10))
    # ^^^ THOG

    def test_materialisation_is_differentiable(self) -> None:
        torch.manual_seed(11)
        parameters = (torch.randn(4, BQRG_PERSISTENT_WIDTH, dtype=torch.float64) * 0.05).requires_grad_()
        generated = materialize_bqrg_sequence(parameters, 16)
        loss = generated.square().mean() + generated[..., -1].mean()
        loss.backward()
        self.assertIsNotNone(parameters.grad)
        self.assertTrue(torch.isfinite(parameters.grad).all())
        self.assertGreater(float(parameters.grad.abs().sum()), 0.0)

    def test_output_rescale_preserves_dynamics_and_scales_sequence(self) -> None:
        torch.manual_seed(13)
        parameters = torch.randn(5, BQRG_PERSISTENT_WIDTH, dtype=torch.float64) * 0.1
        before_parameters = parameters.clone()
        before_sequence = materialize_bqrg_sequence(parameters, 16)
        get_recurrence_generator_definition(BQRG_FAMILY).rescale_output(parameters, 0.25)
        after_sequence = materialize_bqrg_sequence(parameters, 16)
        self.assertTrue(torch.equal(parameters[..., :14], before_parameters[..., :14]))
        self.assertTrue(torch.allclose(after_sequence, before_sequence * 0.25, atol=1.0e-12, rtol=1.0e-12))

    def test_wrong_parameter_width_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "persistent width 16"):
            materialize_bqrg_sequence(torch.zeros(2, 15), 16)

    def test_depth_trajectory_materialises_bqrg_and_backpropagates(self) -> None:
        trajectory = DepthTrajectory(
            tiny_geometry(),
            runtime_dtype=torch.float32,
            basis_family=BQRG_FAMILY,
            basis_version=BQRG_VERSION,
            depth_compress_layer_norm_and_bias=False,
        )
        coefficient = trajectory.coefficients[ATTENTION_QUERY_WEIGHT]
        self.assertEqual(coefficient.shape[-1], BQRG_PERSISTENT_WIDTH)
        self.assertEqual(trajectory.persistent_basis_keys(), ())
        generated = trajectory.materialize(ATTENTION_QUERY_WEIGHT, 15)
        self.assertEqual(generated.shape, (8, 8))
        self.assertTrue(torch.isfinite(generated).all())
        generated.square().mean().backward()
        self.assertIsNotNone(coefficient.grad)
        self.assertTrue(torch.isfinite(coefficient.grad).all())
        self.assertGreater(float(coefficient.grad.abs().sum()), 0.0)

    def test_complete_cpu_model_forward_backward(self) -> None:
        model = build_training_model(tiny_training_config(), device=torch.device("cpu"))
        model.train()
        inputs = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        targets = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
        logits, loss = model(inputs, targets)
        self.assertEqual(logits.shape, (1, 4, 64))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        coefficient = model.trajectory.coefficients[ATTENTION_QUERY_WEIGHT]
        self.assertIsNotNone(coefficient.grad)
        self.assertTrue(torch.isfinite(coefficient.grad).all())

    def test_bqrg_rejected_from_private_depth_fallback(self) -> None:
        with self.assertRaisesRegex(ValueError, "only by public DEPTH"):
            DepthTrajectory(
                tiny_geometry(),
                runtime_dtype=torch.float32,
                basis_family=BQRG_FAMILY,
                basis_version=BQRG_VERSION,
                depth_compress_layer_norm_and_bias=None,
            )


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
