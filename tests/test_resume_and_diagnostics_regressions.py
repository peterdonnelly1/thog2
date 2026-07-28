# vvv THOG
from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

import torch

from sheet.checkpoints import validate_compatibility
from sheet.stage6_diagnostics import coefficient_utilization_report
from sheet.training_config import CHECKPOINT_SCHEMA_VERSION, MODEL_COMPATIBILITY_FIELDS


class _ExpectedConfig:
    def __init__(self, geometry_plan):
        self.geometry_plan = geometry_plan

    def compatibility_signature(self):
        signature = {name: None for name in MODEL_COMPATIBILITY_FIELDS}
        signature["resolved_geometry_plan"] = self.geometry_plan
        return signature

    def compact_identity_metadata(self):
        return {
            "kind": "test_identity",
            "resolved_geometry_plan": self.geometry_plan,
        }


def _payload_for_geometry_plan(geometry_plan):
    signature = {name: None for name in MODEL_COMPATIBILITY_FIELDS}
    signature["resolved_geometry_plan"] = geometry_plan
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "compatibility_signature": signature,
        "compact_identity": {
            "kind": "test_identity",
            "resolved_geometry_plan": geometry_plan,
        },
    }


def _depth_plan(parsed_options):
    return {
        "schema_version": 1,
        "registry_version": "geometry_registry_v5",
        "depth_enabled": True,
        "depth_compressor": "chebyshev",
        "depth_compressor_version": "chebyshev_first_kind_qr_v1",
        "depth_order": 16,
        "selections": [],
        "parsed_options": parsed_options,
        "shared_non_depth_compressor": None,
        "shared_non_depth_compressor_version": None,
        "materializer": {
            "implemented": True,
            "legacy_geometry_preset": "depth",
            "legacy_basis_family": "chebyshev",
            "legacy_basis_version": "chebyshev_first_kind_qr_v1",
            "legacy_mlp_hidden_compressor": None,
            "legacy_mlp_hidden_group_size": None,
            "materialization_version": "depth_v1",
            "message": "Implemented by the existing DEPTH trajectory.",
            "legacy": False,
        },
    }


class ResumeGeometryProvenanceRegressionTest(unittest.TestCase):
    def test_redundant_depth_order_option_does_not_break_resume(self):
        checkpoint_plan = _depth_plan([
            {
                "target": "DEPTH",
                "property": "compressor",
                "value": "chebyshev",
                "source": "DEPTH.compressor=chebyshev",
            }
        ])
        expected_plan = _depth_plan([
            {
                "target": "DEPTH",
                "property": "compressor",
                "value": "chebyshev",
                "source": "DEPTH.compressor=chebyshev",
            },
            {
                "target": "DEPTH",
                "property": "order",
                "value": "16",
                "source": "DEPTH.order=16",
            },
        ])

        validate_compatibility(
            _payload_for_geometry_plan(checkpoint_plan),
            _ExpectedConfig(expected_plan),
        )

    def test_semantic_geometry_change_still_breaks_resume(self):
        checkpoint_plan = _depth_plan([])
        expected_plan = copy.deepcopy(checkpoint_plan)
        expected_plan["depth_order"] = 17

        with self.assertRaisesRegex(ValueError, "resolved_geometry_plan"):
            validate_compatibility(
                _payload_for_geometry_plan(checkpoint_plan),
                _ExpectedConfig(expected_plan),
            )


class UnsupportedCoefficientDiagnosticRegressionTest(unittest.TestCase):
    def test_uncompressed_depth_vector_shape_is_reported_not_raised(self):
        family_name = "depth_vector_example"
        model = SimpleNamespace(
            config=SimpleNamespace(depth_order=16),
            trajectory=SimpleNamespace(
                metadata=(
                    SimpleNamespace(
                        name=family_name,
                        semantic_type="depth_vector",
                    ),
                ),
                coefficients={family_name: torch.ones(32, 1, 1024)},
            ),
        )

        row = coefficient_utilization_report(model)[family_name]

        self.assertFalse(row["order_axis_diagnostics_supported"])
        self.assertEqual(row["shape"], [32, 1, 1024])
        self.assertEqual(row["coefficient_rms"], 1.0)
        self.assertIsNone(row["depth_order_energy_fraction"])
        self.assertIsNone(row["row_order_energy_fraction"])
        self.assertIsNone(row["high_depth_order_energy_fraction"])
        self.assertIsNone(row["high_row_order_energy_fraction"])
        self.assertIn("unsupported coefficient diagnostic shape", row["order_axis_diagnostics_error"])

    def test_supported_shape_keeps_order_axis_diagnostics(self):
        family_name = "spectral_example"
        model = SimpleNamespace(
            config=SimpleNamespace(depth_order=16),
            trajectory=SimpleNamespace(
                metadata=(
                    SimpleNamespace(
                        name=family_name,
                        semantic_type="spectral",
                    ),
                ),
                coefficients={family_name: torch.ones(3, 16, 4)},
            ),
        )

        row = coefficient_utilization_report(model)[family_name]

        self.assertTrue(row["order_axis_diagnostics_supported"])
        self.assertEqual(len(row["depth_order_energy_fraction"]), 16)
        self.assertEqual(len(row["row_order_energy_fraction"]), 4)
        self.assertNotIn("order_axis_diagnostics_error", row)


if __name__ == "__main__":
    unittest.main()
# ^^^ THOG
