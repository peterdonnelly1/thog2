# vvv THOG
from __future__ import annotations

import unittest

import torch

import run_thog2_owt  # noqa: F401  # <<< THOG install the public DEPTH materialisation runtime policy under test
from sheet.depth_trajectory import DepthTrajectory
from sheet.geometry import SheetGeometryConfig
from sheet.semantic_materializer import (
    ATTENTION_OUTPUT_WEIGHT,
    LEGACY_ATTENTION_INPUT_WEIGHT,
    MLP_CONTRACTION_WEIGHT,
    MLP_EXPANSION_WEIGHT,
)
from sheet.stage6_trainer import format_progress_line


class DepthMaterialisationRuntimeTests(unittest.TestCase):
    @staticmethod
    def _trajectory() -> DepthTrajectory:
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
            depth_compress_layer_norm_and_bias=False,
        )
        generator = torch.Generator().manual_seed(12345)
        with torch.no_grad():
            for name in (
                "attention_query_weight",
                "attention_key_weight",
                "attention_value_weight",
                ATTENTION_OUTPUT_WEIGHT,
                MLP_EXPANSION_WEIGHT,
                MLP_CONTRACTION_WEIGHT,
            ):
                trajectory.coefficients[name].copy_(
                    torch.randn(
                        trajectory.coefficients[name].shape,
                        generator=generator,
                        dtype=trajectory.coefficients[name].dtype,
                    )
                )
        return trajectory

    def test_matmul_matches_established_einsum_path(self) -> None:
        trajectory = self._trajectory()
        for name in (
            LEGACY_ATTENTION_INPUT_WEIGHT,
            ATTENTION_OUTPUT_WEIGHT,
            MLP_EXPANSION_WEIGHT,
            MLP_CONTRACTION_WEIGHT,
        ):
            trajectory.depth_materialisation_matmul = False
            expected = trajectory.materialize(name, 2)
            trajectory.depth_materialisation_matmul = True
            actual = trajectory.materialize(name, 2)
            torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-6)

    def test_cpu_profiler_reports_one_end_to_end_layer_sample(self) -> None:
        trajectory = self._trajectory()
        trajectory.begin_materialisation_profiling()
        for name in (
            LEGACY_ATTENTION_INPUT_WEIGHT,
            ATTENTION_OUTPUT_WEIGHT,
            MLP_EXPANSION_WEIGHT,
            MLP_CONTRACTION_WEIGHT,
        ):
            trajectory.materialize(name, 1)
        trajectory.end_materialisation_profiling()
        samples = trajectory.consume_materialisation_penalty_samples()
        self.assertEqual(len(samples), 1)
        self.assertGreaterEqual(samples[0], 0.0)

    def test_console_progress_places_materialisation_penalty_after_step_time(self) -> None:
        line = format_progress_line(
            "run-id",
            "optimizer_progress",
            {
                "completed_updates": "    10",
                "timestamp": "260730:1800",
                "cumulative_training_seconds": "   120",
                "mean_step_seconds": " 12.0000",
                "tok/s": " 10240",
                "consumed_tokens": " 12,345,678",
                "training_loss": "   3.2000",
                "learning_rate": " 9.000e-04",
                "gradient_norm": "   0.250",
                "materialisation_penalty": "0.0123±0.0012s/layer",
            },
        )
        self.assertIn(
            "Δstep= 12.0000s  materialisation penalty=0.0123±0.0012s/layer  tok/s= 10240",
            line,
        )


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
