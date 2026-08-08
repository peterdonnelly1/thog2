# vvv THOG
from __future__ import annotations

import io
import os
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import run_thog2_owt as runner


class PlasticDepthStartupLookaheadUiTests(unittest.TestCase):
    @staticmethod
    def _config(**overrides):
        values = {
            "plastic__enabled": True,
            "plastic__do_learn_layer_count": True,
            "plastic__initial_active_layers": 12,
            "n_layer": 64,
            "plastic__layers_to_sample": None,
            "plastic__initial_layer_count": 12,
            "plastic__max_permitted_layers": 64,
            "plastic__layer_sampling_initialisation": "equidistant",
            "plastic__layer_count_objective": "relative_training_wall_time",
            "plastic__layer_count_update_brake": 30,
            "plastic__layer_count_probe__probe_every_n_steps": 10,
            "plastic__layer_count_probe__window_size_as_number_of_probes": 16,
            "plastic__layer_count__adding_layers__discount_factor_for_extrapolation_evidence": 0.8,
            "plastic__layer_count_probe_noise_lambda": 0.5,
            "plastic__layer_count_cost_weight": 0.03,
            "plastic__layer_memory_budget_gib": None,
            "plastic__cuda_allocator_reserve_gib": 1.5,
            "plastic__geometry_learning_rate_multiplier": 0.25,
            "plastic__freeze_geometry_during_warmup": True,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    @staticmethod
    def _trainer():
        return SimpleNamespace(
            parameter_report={"plastic_depth": {"active_layers": 12}},
        )

    @staticmethod
    def _render(config) -> str:
        output = io.StringIO()
        with redirect_stdout(output):
            runner._print_plastic_depth_section(config, PlasticDepthStartupLookaheadUiTests._trainer())
        return output.getvalue()

    def test_configured_probe_radius_and_max_step_are_shown(self) -> None:
        config = self._config(
            plastic__layer_count_probe_radius=3,
            plastic__layer_count__max_allowable_layer_change=1,
        )
        with patch.dict(
            os.environ,
            {
                "THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS": "9",
                "THOG2_PLASTIC_LAYER_COUNT_MAX_STEP": "8",
            },
        ):
            rendered = self._render(config)
        self.assertRegex(rendered, r"plastic__layer_count_probe_radius:\s+3\n")
        self.assertRegex(rendered, r"plastic__layer_count__max_allowable_layer_change:\s+1\n")

    def test_exported_wrapper_values_are_used_as_fallback(self) -> None:
        config = self._config()
        with patch.dict(
            os.environ,
            {
                "THOG2_PLASTIC_LAYER_COUNT_PROBE_RADIUS": "4",
                "THOG2_PLASTIC_LAYER_COUNT_MAX_STEP": "2",
            },
        ):
            rendered = self._render(config)
        self.assertRegex(rendered, r"plastic__layer_count_probe_radius:\s+4\n")
        self.assertRegex(rendered, r"plastic__layer_count__max_allowable_layer_change:\s+2\n")


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
