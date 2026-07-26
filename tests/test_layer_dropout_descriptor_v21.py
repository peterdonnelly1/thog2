# vvv THOG
from __future__ import annotations

import unittest

from sheet.run_config import OwtRunConfig


class LayerDropoutDescriptorV21Tests(unittest.TestCase):
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

    def test_active_dropout_fields_follow_layer_count(self) -> None:
        config = self._config(
            layer_dropout_stratum_size=4,
            layer_dropout_active_per_stratum=2,
            layer_dropout_resample_steps=1,
        )
        self.assertIn("_L_144_LS_4_LA_2__", config.artifact_name)
        self.assertNotIn("_LI_1", config.artifact_name)
        self.assertNotIn("LDs_", config.artifact_name)
        self.assertNotIn("LDa_", config.artifact_name)
        self.assertNotIn("LDr_", config.artifact_name)

    def test_nondefault_interval_follows_active_count(self) -> None:
        config = self._config(
            layer_dropout_stratum_size=4,
            layer_dropout_active_per_stratum=2,
            layer_dropout_resample_steps=10,
        )
        self.assertIn("_L_144_LS_4_LA_2_LI_10__", config.artifact_name)

    def test_explicit_all_active_configuration_preserves_existing_name(self) -> None:
        baseline = self._config()
        explicit = self._config(
            layer_dropout_stratum_size=144,
            layer_dropout_active_per_stratum=144,
            layer_dropout_resample_steps=10,
        )
        self.assertEqual(explicit.artifact_name, baseline.artifact_name)
        self.assertNotIn("_LS_", explicit.artifact_name)
        self.assertNotIn("_LA_", explicit.artifact_name)
        self.assertNotIn("_LI_", explicit.artifact_name)


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
