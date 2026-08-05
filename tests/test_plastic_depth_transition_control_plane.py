# vvv THOG
from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from sheet.model import SheetGPT, SheetGPTConfig
import sheet.plastic_depth_active_gauge_patch as active_gauge_patch


class PlasticDepthTransitionControlPlaneTests(unittest.TestCase):
    @staticmethod
    def _model(*, device: str = "cpu") -> SheetGPT:
        model = SheetGPT(
            SheetGPTConfig(
                block_size=8,
                vocab_size=32,
                n_layer=5,
                n_head=2,
                n_embd=16,
                dropout=0.0,
                bias=True,
                depth_order=4,
                geometry_preset="depth",
                basis_family="chebyshev",
                plastic__enabled=True,
                plastic__layers_to_sample=None,
                plastic__do_learn_layer_count=True,
                plastic__initial_layer_count=3,
                plastic__max_permitted_layers=5,
                plastic__layer_sampling_initialisation="equidistant",
                plastic__freeze_geometry_during_warmup=False,
            )
        )
        return model.to(device)

    def test_chart_solve_receives_cpu_float64_matrix(self) -> None:
        model = self._model()
        original = active_gauge_patch.stabilized_chebyshev_affine_change_of_chart
        observed = []

        def checked(inverse_r, **kwargs):
            observed.append((inverse_r.device.type, inverse_r.dtype))
            return original(inverse_r, **kwargs)

        with patch.object(
            active_gauge_patch,
            "stabilized_chebyshev_affine_change_of_chart",
            side_effect=checked,
        ):
            transition = model.prepare_plastic_depth_count_transition(4)

        self.assertEqual(observed, [("cpu", torch.float64)])
        self.assertEqual(transition.geometry.new_active_layers, 4)

    def test_control_plane_linalg_failure_falls_back_to_geometry_only(self) -> None:
        model = self._model()
        trajectory = model.trajectory
        coefficient_versions = {
            name: int(parameter._version)
            for name, parameter in trajectory.coefficients.items()
        }

        with patch.object(
            active_gauge_patch,
            "stabilized_chebyshev_affine_change_of_chart",
            side_effect=RuntimeError("forced control-plane linalg failure"),
        ):
            transition = model.prepare_plastic_depth_count_transition(4)

        self.assertEqual(transition.geometry.new_active_layers, 4)
        self.assertEqual(transition.replacements, ())
        self.assertEqual(transition.condition_number, 1.0)
        report = model.commit_plastic_depth_count_transition(transition)
        self.assertEqual(trajectory.plastic_sampling.current_active_layers, 4)
        self.assertEqual(report["transformed_family_count"], 0)
        self.assertEqual(
            coefficient_versions,
            {
                name: int(parameter._version)
                for name, parameter in trajectory.coefficients.items()
            },
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_model_still_solves_chart_on_cpu(self) -> None:
        model = self._model(device="cuda")
        original = active_gauge_patch.stabilized_chebyshev_affine_change_of_chart
        observed = []

        def checked(inverse_r, **kwargs):
            observed.append((inverse_r.device.type, inverse_r.dtype))
            return original(inverse_r, **kwargs)

        with patch.object(
            active_gauge_patch,
            "stabilized_chebyshev_affine_change_of_chart",
            side_effect=checked,
        ):
            transition = model.prepare_plastic_depth_count_transition(4)

        self.assertEqual(observed, [("cpu", torch.float64)])
        self.assertEqual(transition.transform.device.type, "cuda")


if __name__ == "__main__":
    unittest.main(verbosity=2)
# ^^^ THOG
